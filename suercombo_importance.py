import os
import math
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from numpy.array_api import squeeze
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- your project imports ----
from openpilot_torch import OpenPilotModel
from data import Comma2k19SequenceDataset

distance_func = nn.CosineSimilarity(dim=2)
cls_loss = nn.CrossEntropyLoss()
reg_loss = nn.SmoothL1Loss(reduction='none')
def make_supercombo_inputs(batch, device, K: int = 33, target_coord: str = "y"):
    """
    batch:
      - 'seq_input_img': (B,T,C,H,W)  (C = 6 or 12)
      - 'seq_future_poses': (B,T,K,3) with (x, y, psi) or (x, y, z)
    returns:
      imgs12:   (B,12,H,W)
      desire:   (B,8)
      traffic:  (B,2)
      h0:       (B,512)
      traj_gt:  (B,K,15)  scalar target grid for plan head (one scalar per lattice cell)
    """
    seq_imgs   = batch['seq_input_img'].to(device, non_blocking=True)      # (B,T,C,H,W)
    seq_labels = batch['seq_future_poses'].to(device, non_blocking=True)   # (B,T,K,3)

    B, T, C, H, W = seq_imgs.shape
    # supercombo expects 12 channels; if we have 6 (paired RGB), tile as a stopgap
    if C == 6:
        seq_imgs = torch.cat([seq_imgs, seq_imgs], dim=2)  # (B,T,12,H,W)

    imgs12  = seq_imgs[:, -1]                               # (B,12,H,W)
    desire  = torch.zeros((B, 8),   device=device)
    traffic = torch.tensor([[1., 0.]], device=device).repeat(B, 1)
    h0      = torch.zeros((B, 512), device=device)

    traj_gt = seq_labels[:,-1,:,:].squeeze(1)

    return imgs12, desire, traffic, h0, traj_gt

def score_tensor(p: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Returns a tensor of scores same shape as p (no grad).
    mode:
      - 'w'         : |w|
      - 'grad'      : |grad|
      - 'gradxw'    : |grad * w|
      - 'fisher'    : E[(grad)^2]  (approx Fisher diag)  -> caller accumulates mean
      - 'taylor1'   : |grad * w| (same as gradxw; name alias)
    """
    with torch.no_grad():
        if mode == 'w':
            return p.detach().abs()
        if p.grad is None:
            return torch.zeros_like(p, dtype=torch.float32)
        if mode == 'grad':
            return p.grad.detach().abs()
        if mode in ('gradxw', 'taylor1'):
            return (p.grad.detach() * p.detach()).abs()
        if mode == 'fisher':
            return (p.grad.detach() ** 2)  # caller will average across batches
        raise ValueError(f'unknown mode {mode}')

def accumulate_importance(model: nn.Module, data_loader: DataLoader,
                          device: torch.device,
                          num_batches: int,
                          mode: str = 'gradxw',
                          use_amp: bool = True) -> Dict[str, torch.Tensor]:
    """
    Accumulates per-parameter importance over a few mini-batches.
    Returns a dict param_name -> importance_tensor (same shape).
    """
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    traj_loss_fn = nn.SmoothL1Loss()

    # buffers for 'fisher' average; else we sum and later normalize by num_batches
    running: Dict[str, torch.Tensor] = {}

    model.train()
    it = iter(data_loader)
    for b in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        imgs12, desire, traffic, h0, gt = make_supercombo_inputs(batch, device)
        # clear grads
        for p in model.parameters():
            p.grad = None

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(imgs12, desire, traffic, h0)  # (B, 6690) in your setup
            B = out.shape[0]
            plan = out[:, :5 * 991].view(B, 5, 991)  # (B,5,991)

            pred_cls = plan[:, :, -1]  # (B,5)
            params_flat = plan[:, :, :-1]  # (B,5,990)
            pred_trajectory = params_flat.view(B, 5, 2, 33, 15)  # 2=(mean,std_param)
            pred_trajectory = pred_trajectory[:, :, 0, :, :3]
            assert len(pred_cls) == len(pred_trajectory) == len(gt)
            with torch.no_grad():
                # step 1: calculate distance between gt and each prediction
                pred_end_positions = pred_trajectory[:, :, 32, :]  # B, M, 3
                gt_end_positions = gt[:, 32:, :].expand(-1, 5, -1)  # B, 1, 3 -> B, M, 3

                distances = 1 - distance_func(pred_end_positions, gt_end_positions)  # B, M
                index = distances.argmin(dim=1)  # B

            gt_cls = index
            pred_trajectory = pred_trajectory[
                torch.tensor(range(len(gt_cls)), device=gt_cls.device), index, ...]  # B, num_pts, 3

            cls_loss_ = cls_loss(pred_cls, gt_cls)

            reg_loss_ = reg_loss(pred_trajectory, gt).mean(dim=(0, 1))
            loss = cls_loss_ + reg_loss_.mean()

        scaler.scale(loss).backward()
        scaler.step(torch.optim.SGD(model.parameters(), lr=0.0))  # dummy step to satisfy scaler
        scaler.update()

        # accumulate per-parameter scores (no inplace on grads)
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            s = score_tensor(p, mode).detach()
            if name not in running:
                running[name] = s.clone()
            else:
                running[name] += s

    # normalize
    if b >= 0:  # at least one batch processed
        for k in list(running.keys()):
            running[k] = running[k] / float(b + 1)

    return running

def flatten_scores(score_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[str,int,int]]]:
    """
    Flattens all scores into a single 1D tensor and returns mapping information.
    mapping entries: (param_name, numel, cumulative_offset)
    """
    flats = []
    mapping = []
    offset = 0
    for name, t in score_dict.items():
        v = t.reshape(-1).cpu()
        flats.append(v)
        mapping.append((name, t.numel(), offset))
        offset += t.numel()
    if not flats:
        return torch.empty(0), []
    all_scores = torch.cat(flats, dim=0)
    return all_scores, mapping

# ----------------- bit flip -----------------
@torch.no_grad()
def bitflip_inplace(param: torch.Tensor, flat_indices: torch.Tensor, bit: int):
    """
    Flip `bit` (0..31) in float32 representation at positions in `flat_indices`.
    Performs flip on CPU view for correctness, writes back to original device.
    """
    if param.dtype != torch.float32:
        return
    was_cuda = param.is_cuda
    p_cpu = param.detach().cpu().contiguous()
    iview = p_cpu.view(torch.int32).view(-1)
    mask = torch.tensor(1 << bit, dtype=torch.int32)
    iview[flat_indices.cpu()] ^= mask
    if was_cuda:
        param.copy_(p_cpu.to(param.device))
    else:
        param.copy_(p_cpu)

def select_and_flip(model: nn.Module,
                    score_dict: Dict[str, torch.Tensor],
                    topk: int,
                    bit: int,
                    restrict_to: List[str] = None) -> int:
    """
    Selects global top-k elements by score and flips the given bit.
    If `restrict_to` is provided, only parameters whose name contains any
    of the substrings in `restrict_to` are eligible.
    Returns number of elements flipped.
    """
    # optionally filter
    filtered = {}
    for name, s in score_dict.items():
        if restrict_to is None or any(tag in name for tag in restrict_to):
            filtered[name] = s
    all_scores, mapping = flatten_scores(filtered)
    if all_scores.numel() == 0:
        return 0
    k = min(topk, all_scores.numel())
    top_vals, top_idx = torch.topk(all_scores, k, largest=True, sorted=False)

    # build reverse lookup name -> param
    name_to_param = {n: p for n, p in model.named_parameters()}

    flipped = 0
    for i, (name, numel, offset) in enumerate(mapping):
        # local picks for this tensor
        mask = (top_idx >= offset) & (top_idx < offset + numel)
        if not mask.any():
            continue
        local = (top_idx[mask] - offset).to(torch.long)
        p = name_to_param[name]
        bitflip_inplace(p, local, bit)
        flipped += local.numel()
    return flipped

# ----------------- main -----------------
def build_loaders(args, device):
    train = Comma2k19SequenceDataset(args.train_index, args.data_root, 'train', use_memcache=False)
    val   = Comma2k19SequenceDataset(args.val_index,   args.data_root, 'val',   use_memcache=False)

    loader_args = dict(num_workers=0, pin_memory=(device.type=='cuda'))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, **loader_args)
    val_loader   = DataLoader(val,   batch_size=1,               shuffle=False, **loader_args)
    return train_loader, val_loader

def load_weights(model, ckpt_path: str):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"[Load] WARNING: checkpoint not found: {ckpt_path} (using random init).")
        return
    print(f"[Load] loading {ckpt_path}")
    sd = torch.load(ckpt_path, map_location='cpu')
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        from collections import OrderedDict
        new_sd = OrderedDict((k.replace('module.', ''), v) for k, v in sd.items())
        model.load_state_dict(new_sd, strict=False)

def evaluate_loss_mdn(model, data_loader, device, num_batches: int = 5, use_amp: bool = True):
    """
    Expects make_supercombo_inputs(...)-> traj_gt of shape (B,33,15)
    Model output is the full supercombo vector; we slice the first 5*991 for plan head.
    """
    model.eval()
    losses = []
    it = iter(data_loader)
    with torch.no_grad():
        for _ in range(num_batches):
            try:
                batch = next(it)
            except StopIteration:
                break

            # gt: (B,33,15)  scalar grid
            imgs12, desire, traffic, h0, gt = make_supercombo_inputs(batch, device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out  = model(imgs12, desire, traffic, h0)        # (B, 6690) in your setup
                B    = out.shape[0]
                plan = out[:, :5 * 991].view(B, 5, 991)          # (B,5,991)

                pred_cls = plan[:, :, -1]                      # (B,5)
                params_flat = plan[:, :, :-1]                     # (B,5,990)
                pred_trajectory = params_flat.view(B, 5, 2, 33, 15)   # 2=(mean,std_param)
                pred_trajectory = pred_trajectory[:,:,0,:,:3]
                assert len(pred_cls) == len(pred_trajectory) == len(gt)
                with torch.no_grad():
                    # step 1: calculate distance between gt and each prediction
                    pred_end_positions = pred_trajectory[:, :, 32, :]  # B, M, 3
                    gt_end_positions = gt[:, 32:, :].expand(-1, 5, -1)  # B, 1, 3 -> B, M, 3

                    distances = 1 - distance_func(pred_end_positions, gt_end_positions)  # B, M
                    index = distances.argmin(dim=1)  # B

                gt_cls = index
                pred_trajectory = pred_trajectory[
                    torch.tensor(range(len(gt_cls)), device=gt_cls.device), index, ...]  # B, num_pts, 3

                cls_loss_ = cls_loss(pred_cls, gt_cls)

                reg_loss_ = reg_loss(pred_trajectory, gt).mean(dim=(0, 1))
                loss = cls_loss_ + reg_loss_.mean()

            losses.append(loss.item())

    return float(np.mean(losses)) if losses else float('nan')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_index', type=str, default='data/comma2k19_train_non_overlap.txt')
    p.add_argument('--val_index',   type=str, default='data/comma2k19_val_non_overlap.txt')
    p.add_argument('--data_root',   type=str, default='data/comma2k19/')
    p.add_argument('--batch_size',  type=int, default=2)

    p.add_argument('--ckpt',        type=str, default='openpilot_model/supercombo_torch_weights.pth')  # set your checkpoint
    p.add_argument('--mode',        type=str, default='gradxw',
                   choices=['w','grad','gradxw','taylor1','fisher'])
    p.add_argument('--calib_batches', type=int, default=8,
                   help='how many mini-batches to accumulate for importance')
    p.add_argument('--K', type=int, default=100, help='how many scalar weights to flip')
    p.add_argument('--topk',        type=int, default=100, help='how many scalar weights to flip')
    p.add_argument('--bit',         type=int, default=10,  help='which bit to flip (0..31 for float32)')
    p.add_argument('--restrict',    type=str, default='',
                   help='comma-separated substrings to restrict flipping to, e.g. "temporal_policy.plan,vision_net"')
    p.add_argument('--amp',         action='store_true', help='use mixed precision for calibration/eval')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False

    restrict_to = [s.strip() for s in args.restrict.split(',') if s.strip()] or None

    # loaders
    train_loader, val_loader = build_loaders(args, device)
    print("train samples:", len(train_loader.dataset), "val samples:", len(val_loader.dataset))

    # model
    model = OpenPilotModel().to(device)
    load_weights(model, args.ckpt)

    # baseline loss
    #base_loss = evaluate_loss(model, val_loader, device, num_batches=5, use_amp=args.amp)
    base_loss = evaluate_loss_mdn(model, val_loader, device, num_batches=5, use_amp=args.amp)
    print(f"[Eval] baseline traj loss: {base_loss:.6f}")

    # accumulate importance
    print(f"[Importance] mode={args.mode}, batches={args.calib_batches}")
    imp = accumulate_importance(model, train_loader, device,
                                num_batches=args.calib_batches,
                                mode=args.mode, use_amp=args.amp)

    # select & flip
    for args.bit in [31, 30, 29, 23, 22, 21, 0]:
        flipped = select_and_flip(model, imp, topk=args.topk, bit=args.bit, restrict_to=restrict_to)
        print(f"[Flip] flipped bit {args.bit} in {flipped} elements.")

        # post-flip loss
        #post_loss = evaluate_loss(model, val_loader, device, num_batches=5, use_amp=args.amp)
        post_loss = evaluate_loss_mdn(model, val_loader, device, num_batches=5, use_amp=args.amp)
        print(f"[Eval] post-flip traj loss: {post_loss:.6f} (Δ={post_loss - base_loss:+.6f})")

if __name__ == "__main__":
    main()
