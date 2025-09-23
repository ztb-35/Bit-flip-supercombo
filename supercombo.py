# train_supercombo.py
import os, time, random
from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

# ---- repo imports ----
from data import Comma2k19SequenceDataset
# bring in your supercombo torch port
from openpilot_torch import OpenPilotModel

# =========================================================
# Indices: slice supercombo outputs to get trajectory + class
# These match the checks printed in your openpilot_torch.py
# x_rel = out[0, 5755:5779:4], y_rel = out[0, 5756:5780:4]
# class logits example: out[:, 6010:6013]
# =========================================================
TRAJ_X_IDXS = list(range(5755, 5779, 4))   # 6 waypoints in this demo
TRAJ_Y_IDXS = list(range(5756, 5780, 4))
IDX_CLS_SLICE = slice(6010, 6013)          # 3-class logits
K_WAYPOINTS = len(TRAJ_X_IDXS)

# -------------------------------
# Hyperparameters / CLI
# -------------------------------
def get_hyperparameters(parser: ArgumentParser):
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--log_per_n_step', type=int, default=50)
    parser.add_argument('--val_per_n_epoch', type=int, default=1)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--sync_bn', type=bool, default=False, help='(DDP only)')

    # Loss weights
    parser.add_argument('--w_traj', type=float, default=1.0)
    parser.add_argument('--w_cls',  type=float, default=0.5)  # set 0 to ignore class loss

    # Sequence unroll & optimizer
    parser.add_argument('--optimize_per_n_step', type=int, default=40)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'])

    # Data
    parser.add_argument('--num_pts', type=int, default=33)  # GT horizon in dataset
    parser.add_argument('--train_index', type=str, default='data/comma2k19_train_non_overlap.txt')
    parser.add_argument('--val_index',   type=str, default='data/comma2k19_val_non_overlap.txt')
    parser.add_argument('--data_root',   type=str, default='data/comma2k19/')

    exp_name = time.strftime('supercombo_%Y%m%d_%H%M%S')
    parser.add_argument('--exp_name', type=str, default=exp_name)
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
        #prefetch_factor=2,
        pin_memory=pin_memory,
    )
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, **loader_args)
    val_loader   = DataLoader(val, batch_size=1, shuffle=False, **loader_args)
    return train_loader, val_loader

# -------------------------------
# Supercombo wrapper (trajectory-only view)
# -------------------------------
class SupercomboTrajWrapper(nn.Module):
    """
    Wrap OpenPilotModel and expose only the trajectory slice (and optional class logits).
    """
    def __init__(self):
        super().__init__()
        self.net = OpenPilotModel()

    def forward(self, img12, desire, traffic, h0):
        # net returns (B, 6609)
        out = self.net(img12, desire, traffic, h0)
        # --- trajectory slice ---
        x = out[:, TRAJ_X_IDXS]                  # (B,K)
        y = out[:, TRAJ_Y_IDXS]                  # (B,K)
        traj = torch.stack([x, y], dim=-1)       # (B,K,2)
        # --- optional class logits ---
        idx_logits = out[:, IDX_CLS_SLICE]       # (B,3)
        return traj, idx_logits

# -------------------------------
# Optimizer helper
# -------------------------------
def configure_optim(args, model):
    if args.optimizer == 'sgd':
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.9)
    return opt, sch

# -------------------------------
# Train / Val
# -------------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(comment=args.exp_name)

    train_loader, val_loader = get_dataloader(
        batch_size=args.batch_size,
        pin_memory=False,  # was True → can increase host memory pressure
        num_workers=0,  # was 4 → workers + prefetch can balloon RAM
        data_root=args.data_root,
        train_index=args.train_index,
        val_index=args.val_index,
    )

    model = SupercomboTrajWrapper()

    # Optional DataParallel
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[Info] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    opt, sch = configure_optim(args, model)

    # Losses
    traj_loss_fn = nn.SmoothL1Loss(reduction='mean')  # Huber is robust for meters
    cls_loss_fn  = nn.CrossEntropyLoss()

    # Resume
    if args.resume:
        print('Loading weights from', args.resume)
        sd = torch.load(args.resume, map_location='cpu')
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError:
            from collections import OrderedDict
            new_sd = OrderedDict((k.replace('module.', ''), v) for k, v in sd.items())
            model.load_state_dict(new_sd, strict=False)

    num_steps = 0
    opt_per_n = args.optimize_per_n_step
    running_loss = 0.0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            # ---- fetch sequence batch ----
            # seq_input_img: (B, T, C, H, W)  (C is 6 or 12 depending on your preprocessing)
            # seq_future_poses: (B, T, num_pts, 3) with (x, y, psi)
            seq_imgs   = batch['seq_input_img'].to(device, non_blocking=True)
            seq_labels = batch['seq_future_poses'].to(device, non_blocking=True)

            B, T, C, H, W = seq_imgs.shape
            # supercombo expects 12-channel inputs; if your dataset currently outputs 6-ch,
            # you can temporarily tile to 12-ch (better: modify dataset to emit true 12-ch)
            if C == 6:
                seq_imgs = torch.cat([seq_imgs, seq_imgs], dim=2)  # (B,T,12,H,W)
                C = 12

            # build extra inputs (zeros are fine to start)
            desire  = torch.zeros((B, 8),    device=device)
            traffic = torch.tensor([[1., 0.]], device=device).repeat(B, 1)  # right-hand traffic
            h0      = torch.zeros((B, 512),  device=device)

            # ---- unroll time ----
            hidden_carry = h0  # placeholder; your torch port handles RNN inside
            total_loss = 0.0

            for t in range(T):
                num_steps += 1
                imgs12 = seq_imgs[:, t]  # (B,12,128,256)
                gt_xy  = seq_labels[:, t, :K_WAYPOINTS, :2]  # (B,K,2) — use first K points

                traj_pred, idx_logits = model(imgs12, desire, traffic, hidden_carry)
                # losses
                loss_traj = traj_loss_fn(traj_pred, gt_xy)
                if args.w_cls > 0:
                    # If you have a real label, replace this placeholder rule.
                    # Example dummy rule: 3-way based on first-step heading sign
                    v = gt_xy[:, 1] - gt_xy[:, 0]          # (B,2)
                    angle = torch.atan2(v[:, 1], v[:, 0])  # (B,)
                    idx_gt = torch.zeros(B, dtype=torch.long, device=device)
                    idx_gt[angle >  0.1] = 1
                    idx_gt[angle < -0.1] = 2
                    loss_cls = cls_loss_fn(idx_logits, idx_gt)
                else:
                    loss_cls = torch.zeros((), device=device)

                loss = args.w_traj * loss_traj + args.w_cls * loss_cls
                total_loss = total_loss + loss / opt_per_n
                running_loss += loss.item()

                # Log
                if (num_steps % args.log_per_n_step) == 0:
                    writer.add_scalar('train/loss_traj', loss_traj.item(), num_steps)
                    if args.w_cls > 0:
                        with torch.no_grad():
                            acc = (idx_logits.argmax(1) == idx_gt).float().mean().item()
                        writer.add_scalar('train/loss_cls', loss_cls.item(), num_steps)
                        writer.add_scalar('train/acc_cls',  acc, num_steps)
                    writer.add_scalar('train/lr', opt.param_groups[0]['lr'], num_steps)
                    writer.add_scalar('train/loss_running', running_loss/args.log_per_n_step, num_steps)
                    running_loss = 0.0

                # step every N unrolled steps
                if (t + 1) % opt_per_n == 0:
                    opt.zero_grad(set_to_none=True)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    total_loss = 0.0

            # flush any remainder
            if isinstance(total_loss, torch.Tensor):
                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        sch.step()

        # ------------- Validation + checkpoint -------------
        if (epoch + 1) % args.val_per_n_epoch == 0:
            ckpt_dir = writer.log_dir
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch}.pth')
            torch.save(getattr(model, 'module', model).state_dict(), ckpt_path)
            print(f'[Epoch {epoch}] checkpoint saved at {ckpt_path}')

            model.eval()
            with torch.no_grad():
                traj_l, cls_l, cls_acc = [], [], []
                for batch in tqdm(val_loader, leave=False, desc='Val'):
                    seq_imgs   = batch['seq_input_img'].to(device, non_blocking=True)
                    seq_labels = batch['seq_future_poses'].to(device, non_blocking=True)
                    B, T, C, H, W = seq_imgs.shape
                    if C == 6: seq_imgs = torch.cat([seq_imgs, seq_imgs], dim=2)

                    desire  = torch.zeros((B, 8),   device=device)
                    traffic = torch.tensor([[1.,0.]], device=device).repeat(B,1)
                    h0      = torch.zeros((B, 512), device=device)

                    for t in range(T):
                        imgs12 = seq_imgs[:, t]
                        gt_xy  = seq_labels[:, t, :K_WAYPOINTS, :2]
                        traj_pred, idx_logits = model(imgs12, desire, traffic, h0)
                        lt = nn.functional.smooth_l1_loss(traj_pred, gt_xy, reduction='mean')
                        traj_l.append(lt.item())
                        if args.w_cls > 0:
                            v = gt_xy[:, 1] - gt_xy[:, 0]
                            angle = torch.atan2(v[:, 1], v[:, 0])
                            idx_gt = torch.zeros(B, dtype=torch.long, device=device)
                            idx_gt[angle >  0.1] = 1
                            idx_gt[angle < -0.1] = 2
                            lc = nn.functional.cross_entropy(idx_logits, idx_gt)
                            cls_l.append(lc.item())
                            cls_acc.append((idx_logits.argmax(1) == idx_gt).float().mean().item())

                writer.add_scalar('val/traj_loss', np.mean(traj_l) if traj_l else 0.0, epoch)
                if cls_l:
                    writer.add_scalar('val/cls_loss',  np.mean(cls_l), epoch)
                    writer.add_scalar('val/cls_acc',   np.mean(cls_acc), epoch)

            model.train()

# -------------------------------
# Entrypoint
# -------------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser = get_hyperparameters(parser)
    args = parser.parse_args()

    # Seeds
    seed = 42
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    print(f'[{time.time():.2f}] start training supercombo (traj-only)…')
    main(args)
