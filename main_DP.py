import os
import time
import random
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

if torch.__version__ == 'parrots':
    from pavi import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter

from data import PlanningDataset, SequencePlanningDataset, Comma2k19SequenceDataset
from model import PlaningNetwork, MultipleTrajectoryPredictionLoss, SequencePlanningNetwork
from utils import draw_trajectory_on_ax, get_val_metric, get_val_metric_keys


# -------------------------------
# Hyperparameters / CLI
# -------------------------------
def get_hyperparameters(parser: ArgumentParser):
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--log_per_n_step', type=int, default=20)
    parser.add_argument('--val_per_n_epoch', type=int, default=1)

    parser.add_argument('--resume', type=str, default='')

    parser.add_argument('--M', type=int, default=5)
    parser.add_argument('--num_pts', type=int, default=33)
    parser.add_argument('--mtp_alpha', type=float, default=1.0)
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--sync_bn', type=bool, default=True,
                        help='Ignored in DataParallel mode (SyncBN requires DDP).')
    parser.add_argument('--tqdm', type=bool, default=False)
    parser.add_argument('--optimize_per_n_step', type=int, default=40)

    exp_name = str(time.time())
    parser.add_argument('--exp_name', type=str, default=exp_name)

    # dataset paths (adjust if needed)
    parser.add_argument('--train_index', type=str, default='data/comma2k19_train_non_overlap.txt')
    parser.add_argument('--val_index',   type=str, default='data/comma2k19_val_non_overlap.txt')
    parser.add_argument('--data_root',   type=str, default='data/comma2k19/')

    return parser


# -------------------------------
# Data
# -------------------------------
def get_dataloader(batch_size, pin_memory=False, num_workers=0, data_root='data/comma2k19/',
                   train_index='data/comma2k19_train_non_overlap.txt',
                   val_index='data/comma2k19_val_non_overlap.txt'):
    train = Comma2k19SequenceDataset(train_index, data_root, 'train', use_memcache=False)
    val   = Comma2k19SequenceDataset(val_index,   data_root, 'demo',  use_memcache=False)

    loader_args = dict(
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2,
        pin_memory=pin_memory,
    )
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, **loader_args)
    # keep val deterministic and small batch
    val_loader   = DataLoader(val, batch_size=1, shuffle=False, **loader_args)
    return train_loader, val_loader


# -------------------------------
# Model wrapper
# -------------------------------
class SequenceBaselineV1(nn.Module):
    def __init__(self, M, num_pts, mtp_alpha, lr, optimizer, optimize_per_n_step=40) -> None:
        super().__init__()
        self.M = M
        self.num_pts = num_pts
        self.mtp_alpha = mtp_alpha
        self.lr = lr
        self.optimizer = optimizer
        self.net = SequencePlanningNetwork(M, num_pts)
        self.optimize_per_n_step = optimize_per_n_step  # for the GRU module

    @staticmethod
    def configure_optimizers(args, model):
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
        elif args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        else:
            raise NotImplementedError
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.9)
        return optimizer, lr_scheduler

    def forward(self, x, hidden=None):
        # Make sure hidden is on the same device as inputs
        if hidden is None:
            hidden = torch.zeros((2, x.size(0), 512), device=x.device)
        return self.net(x, hidden)


# -------------------------------
# Train / Val
# -------------------------------
def main(args):
    # Device + DP enablement
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    writer = SummaryWriter()

    train_loader, val_loader = get_dataloader(
        batch_size=args.batch_size,
        pin_memory=(device.type == 'cuda'),
        num_workers=args.n_workers,
        data_root=args.data_root,
        train_index=args.train_index,
        val_index=args.val_index,
    )

    model = SequenceBaselineV1(args.M, args.num_pts, args.mtp_alpha, args.lr,
                               args.optimizer, args.optimize_per_n_step)

    # SyncBN is DDP-only; ignore in DP
    if args.sync_bn:
        print("[Info] --sync_bn requested, but ignored in DataParallel / single-process mode.")

    # Optional DataParallel
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[Info] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    optimizer, lr_scheduler = SequenceBaselineV1.configure_optimizers(args, model)
    loss_fn = MultipleTrajectoryPredictionLoss(args.mtp_alpha, args.M, args.num_pts, distance_type='angle')

    # Resume
    if args.resume:
        print('Loading weights from', args.resume)
        sd = torch.load(args.resume, map_location='cpu')
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError:
            # handle "module." prefix differences
            from collections import OrderedDict
            new_sd = OrderedDict((k.replace('module.', ''), v) for k, v in sd.items())
            model.load_state_dict(new_sd, strict=False)

    num_steps = 0
    disable_tqdm = (not args.tqdm)

    for epoch in tqdm(range(args.epochs), disable=disable_tqdm, position=0):
        model.train()

        for batch_idx, data in enumerate(tqdm(train_loader, leave=False, disable=disable_tqdm, position=1)):
            seq_inputs = data['seq_input_img'].to(device, non_blocking=True)
            seq_labels = data['seq_future_poses'].to(device, non_blocking=True)

            bs = seq_labels.size(0)
            seq_length = seq_labels.size(1)

            hidden = torch.zeros((2, bs, 512), device=device)
            total_loss = 0

            # Unroll sequence
            for t in tqdm(range(seq_length), leave=False, disable=disable_tqdm, position=2):
                num_steps += 1
                inputs = seq_inputs[:, t, :, :, :]
                labels = seq_labels[:, t, :, :]

                pred_cls, pred_trajectory, hidden = model(inputs, hidden)

                cls_loss, reg_loss = loss_fn(pred_cls, pred_trajectory, labels)
                # Accumulate / normalize by optimize_per_n_step
                opt_per_n = getattr(getattr(model, 'module', model), 'optimize_per_n_step')
                total_loss = total_loss + (cls_loss + args.mtp_alpha * reg_loss.mean()) / opt_per_n

                if (num_steps + 1) % args.log_per_n_step == 0:
                    writer.add_scalar('train/epoch', epoch, num_steps)
                    writer.add_scalar('loss/cls', cls_loss, num_steps)
                    writer.add_scalar('loss/reg', reg_loss.mean(), num_steps)
                    writer.add_scalar('loss/reg_x', reg_loss[0], num_steps)
                    writer.add_scalar('loss/reg_y', reg_loss[1], num_steps)
                    writer.add_scalar('loss/reg_z', reg_loss[2], num_steps)
                    writer.add_scalar('param/lr', optimizer.param_groups[0]['lr'], num_steps)

                if (t + 1) % opt_per_n == 0:
                    hidden = hidden.clone().detach()
                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    writer.add_scalar('loss/total', total_loss, num_steps)
                    total_loss = 0

            # flush any remainder
            if not isinstance(total_loss, int):
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                writer.add_scalar('loss/total', total_loss, num_steps)

        lr_scheduler.step()

        # ---------- Validation ----------
        if (epoch + 1) % args.val_per_n_epoch == 0:
            # Save checkpoint
            ckpt_path = os.path.join(writer.log_dir, f'epoch_{epoch}.pth')
            torch.save(getattr(model, 'module', model).state_dict(), ckpt_path)
            print(f'[Epoch {epoch}] checkpoint saved at {ckpt_path}')

            model.eval()
            with torch.no_grad():
                saved_metric_epoch = get_val_metric_keys()
                for batch_idx, data in enumerate(tqdm(val_loader, leave=False, disable=True, position=1)):
                    seq_inputs = data['seq_input_img'].to(device, non_blocking=True)
                    seq_labels = data['seq_future_poses'].to(device, non_blocking=True)

                    bs = seq_labels.size(0)
                    seq_length = seq_labels.size(1)

                    hidden = torch.zeros((2, bs, 512), device=device)
                    for t in range(seq_length):
                        inputs = seq_inputs[:, t, :, :, :]
                        labels = seq_labels[:, t, :, :]

                        pred_cls, pred_trajectory, hidden = model(inputs, hidden)

                        metrics = get_val_metric(pred_cls, pred_trajectory.view(-1, args.M, args.num_pts, 3), labels)
                        for k, v in metrics.items():
                            saved_metric_epoch[k].append(v.float().mean().item())

                # write mean metrics
                for k in sorted(saved_metric_epoch.keys()):
                    writer.add_scalar(k, np.mean(saved_metric_epoch[k]), num_steps)

            model.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = get_hyperparameters(parser)
    args = parser.parse_args()

    # Seed for reproducibility (optional)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f'[{time.time():.2f}] starting job... single process, DataParallel optional.')
    main(args)
