import os
import re
import math
import json
import time
import argparse
import sys
import io
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

# ---- your project imports ----
from openpilot_torch import OpenPilotModel
from data import Comma2k19SequenceDataset
from transfer_onnx_compare import export_to_onnx

# ----------------- tiny tee logger (capture stdout+stderr but still print) -----------------
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

def start_log_capture():
    """Route stdout+stderr to both console and an in-memory buffer; return (restore_fn, buffer)."""
    buf = io.StringIO()
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, buf)
    sys.stderr = _Tee(orig_err, buf)
    def restore():
        sys.stdout = orig_out
        sys.stderr = orig_err
    return restore, buf

# ----------------- losses & helpers -----------------
distance_func = nn.CosineSimilarity(dim=2)
cls_loss = nn.CrossEntropyLoss()
reg_loss = nn.SmoothL1Loss(reduction='none')

def timestamp_id() -> str:
    # e.g., "20250927-130542"
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def make_supercombo_inputs(batch, device, K: int = 33):
    """
    batch:
      - 'seq_input_img': (B,T,C,H,W)  (C = 6 or 12; if 6 we tile to 12)
      - 'seq_future_poses': (B,T,K,3) -> (x,y,z) or (x,y,psi); we use (x,y,z)
    returns:
      imgs12:  (B,12,H,W)
      desire:  (B,8)
      traffic: (B,2)
      h0:      (B,512)
      traj_gt: (B,K,3)  future positions in ego-at-now
    """
    seq_imgs   = batch['seq_input_img'].to(device, non_blocking=True)      # (B,T,C,H,W)
    seq_labels = batch['seq_future_poses'].to(device, non_blocking=True)   # (B,T,K,3)

    B, T, C, H, W = seq_imgs.shape
    if C == 6:
        seq_imgs = torch.cat([seq_imgs, seq_imgs], dim=2)  # (B,T,12,H,W)

    imgs12  = seq_imgs[:, -1]                      # (B,12,H,W)
    desire  = torch.zeros((B, 8),   device=device)
    traffic = torch.tensor([[1., 0.]], device=device).repeat(B, 1)
    h0      = torch.zeros((B, 512), device=device)
    traj_gt = seq_labels[:, -1, :, :]              # (B,K,3)

    return imgs12, desire, traffic, h0, traj_gt

def score_tensor(p: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Returns a tensor of scores same shape as p (no grad).
    mode: 'w' | 'grad' | 'gradxw' | 'taylor1' | 'fisher'
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
            return (p.grad.detach() ** 2)
        raise ValueError(f'unknown mode {mode}')

# ----------------- importance accumulation -----------------
def accumulate_importance(model: nn.Module, data_loader: DataLoader,
                          device: torch.device,
                          num_batches: int,
                          mode: str = 'gradxw',
                          use_amp: bool = True) -> Dict[str, torch.Tensor]:
    """
    Accumulates per-parameter importance over a few mini-batches.
    Returns: dict param_name -> importance_tensor (same shape).
    """
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    running: Dict[str, torch.Tensor] = {}

    # dummy optimizer so we can unscale grads properly with AMP
    opt = torch.optim.SGD(model.parameters(), lr=0.0)

    model.train()
    it = iter(data_loader)
    processed = 0
    for _ in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        imgs12, desire, traffic, h0, gt = make_supercombo_inputs(batch, device)

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(imgs12, desire, traffic, h0)      # (B, 6690)
            B = out.shape[0]
            plan = out[:, :5 * 991].view(B, 5, 991)       # (B,5,991)

            pred_cls = plan[:, :, -1]                     # (B,5)
            params_flat = plan[:, :, :-1]                 # (B,5,990)
            pred_trajectory = params_flat.view(B, 5, 2, 33, 15)[:, :, 0, :, :3]  # mean xyz

            assert len(pred_cls) == len(pred_trajectory) == len(gt)

            with torch.no_grad():
                pred_end = pred_trajectory[:, :, 32, :]            # (B,5,3)
                gt_end = gt[:, 32:33, :].expand(-1, 5, -1)         # (B,5,3)
                distances = 1 - distance_func(pred_end, gt_end)    # (B,5)
                index = distances.argmin(dim=1)                    # (B,)

            gt_cls = index
            row_idx = torch.arange(len(gt_cls), device=gt_cls.device)
            best_traj = pred_trajectory[row_idx, gt_cls, :, :]     # (B,33,3)

            cls_loss_ = cls_loss(pred_cls, gt_cls)                 # CE over 5 modes
            reg_loss_ = reg_loss(best_traj, gt).mean(dim=(0, 1))   # SmoothL1 over (B,33,3)
            loss = cls_loss_ + reg_loss_.mean()

        scaler.scale(loss).backward()
        scaler.unscale_(opt)     # grads now true-valued for scoring
        scaler.update()

        # accumulate per-parameter scores
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                s = score_tensor(p, mode).detach()
                running[name] = s.clone() if name not in running else (running[name] + s)

        processed += 1

    if processed > 0:
        for k in running.keys():
            running[k] = running[k] / float(processed)

    return running

def flatten_scores(score_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[str,int,int]]]:
    """
    Flattens all scores into one 1D tensor and returns mapping info:
    [(param_name, numel, cumulative_offset), ...]
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
    return torch.cat(flats, dim=0), mapping

# ----------------- bit flip -----------------
def _sanitize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]+', '_', name)

def _unravel(flat_idx: int, shape: torch.Size):
    return np.unravel_index(int(flat_idx), tuple(shape), order='C')

@torch.no_grad()
def bitflip_inplace_and_log(param: torch.Tensor,
                            flat_indices: torch.Tensor,
                            bit: int,
                            param_name: str) -> List[Dict[str, Any]]:
    """
    Flip `bit` (0..31) in float32 representation at positions in `flat_indices`.
    Returns a list of flip records with old/new values and nd indices.
    """
    flips: List[Dict[str, Any]] = []
    if param.dtype != torch.float32 or flat_indices.numel() == 0:
        return flips

    was_cuda = param.is_cuda
    p_cpu = param.detach().cpu().contiguous()
    fview = p_cpu.view(torch.float32).view(-1)
    iview = p_cpu.view(torch.int32).view(-1)
    mask = torch.tensor(1 << bit, dtype=torch.int32)

    idx_cpu = flat_indices.detach().cpu().to(torch.long)
    for fi in idx_cpu:
        old_f = float(fview[fi])
        old_i = int(iview[fi])
        iview[fi] = old_i ^ int(mask)
        new_f = float(fview[fi])
        flips.append({
            "name": param_name,
            "bit": int(bit),
            "index_flat": int(fi),
            "index": tuple(map(int, _unravel(int(fi), param.shape))),
            "old": old_f,
            "new": new_f,
        })

    if was_cuda:
        param.copy_(p_cpu.to(param.device))
    else:
        param.copy_(p_cpu)
    return flips

def filter_params_for_flipping(model: nn.Module,
                               allow_bias: bool,
                               restrict_to: List[str] = None,
                               kinds: List[str] = None) -> Dict[str, torch.Tensor]:
    """
    Return {name: tensor} eligible for flipping.
    kinds: optional list of substrings to include by tensor kind (e.g., ['weight','bias'])
    """
    out = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dtype != torch.float32:
            continue
        if not allow_bias and name.endswith(".bias"):
            continue
        if restrict_to is not None and not any(tag in name for tag in restrict_to):
            continue
        if kinds is not None and not any(k in name for k in kinds):
            continue
        out[name] = p
    return out

def select_topk_elements(score_dict: Dict[str, torch.Tensor],
                         model: nn.Module,
                         allow_bias: bool,
                         restrict_to: List[str] = None,
                         kinds: List[str] = None,
                         topk: int = 100) -> Tuple[List[Tuple[str,int]], Dict[str, torch.Tensor]]:
    """Return [(param_name, local_flat_idx), ...] for the top-K most important scalars."""
    eligible = filter_params_for_flipping(model, allow_bias, restrict_to, kinds)
    filt_scores = {n: s for n, s in score_dict.items() if n in eligible}
    all_scores, mapping = flatten_scores(filt_scores)
    if all_scores.numel() == 0:
        return [], {}

    k = min(topk, all_scores.numel())
    vals, idx = torch.topk(all_scores, k, largest=True, sorted=True)
    items: List[Tuple[str,int]] = []
    for gpos in idx.tolist():
        for name, numel, off in mapping:
            if off <= gpos < off + numel:
                items.append((name, gpos - off))
                break
    name_to_param = {n: p for n, p in model.named_parameters()}
    return items, name_to_param

# --- bit sets for FP32 ---
_MANTISSA      = list(range(0, 23))   # 23 bits (0..22)
_EXPONENT      = list(range(23, 31))  # 8 bits (23..30)
_SIGN          = [31]
_MANTISSA_LOW  = list(range(0, 7))    # gentler

def parse_bits_spec(bits_spec: str) -> List[int]:
    s = bits_spec.strip().lower()
    if s == 'mantissa':     return _MANTISSA
    if s == 'mantissa_low': return _MANTISSA_LOW
    if s == 'exponent':     return _EXPONENT
    if s == 'sign':         return _SIGN
    bits = [int(b.strip()) for b in s.split(',') if b.strip()!='']
    if not bits:
        raise ValueError(f'invalid --bits: {bits_spec}')
    for b in bits:
        if b < 0 or b > 31:
            raise ValueError(f'bit {b} out of range [0,31]')
    return bits

@torch.no_grad()
def mbu_flip_multi_bits(model: nn.Module,
                        name_to_param: Dict[str, torch.Tensor],
                        items: List[Tuple[str,int]],
                        bits: List[int],
                        bits_per_weight: int) -> List[Dict[str, Any]]:
    """
    For each selected scalar, flip `bits_per_weight` bit positions within the same 32-bit word.
    Uses the first `bits_per_weight` entries of `bits` (re-uses last if fewer provided).
    """
    logs: List[Dict[str, Any]] = []
    if not items:
        return logs
    for (pname, lidx) in items:
        p = name_to_param[pname]
        for j in range(bits_per_weight):
            bit = bits[min(j, len(bits)-1)]
            logs += bitflip_inplace_and_log(p, torch.tensor([lidx], dtype=torch.long), bit, pname)
    return logs

# ----------------- eval -----------------
@torch.no_grad()
def evaluate_loss_mdn(model, data_loader, device, num_batches: int = 5, use_amp: bool = True):
    """
    Evaluates a simple plan-head loss using closest mode (end-point matching) + SmoothL1 on xyz.
    """
    model.eval()
    losses = []
    it = iter(data_loader)
    for _ in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        imgs12, desire, traffic, h0, gt = make_supercombo_inputs(batch, device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out  = model(imgs12, desire, traffic, h0)         # (B, 6690)
            B    = out.shape[0]
            plan = out[:, :5 * 991].view(B, 5, 991)           # (B,5,991)

            pred_cls = plan[:, :, -1]                         # (B,5)
            params_flat = plan[:, :, :-1]                     # (B,5,990)
            pred_traj = params_flat.view(B, 5, 2, 33, 15)[:, :, 0, :, :3]  # mean xyz

            with torch.no_grad():
                pred_end = pred_traj[:, :, 32, :]               # (B,5,3)
                gt_end   = gt[:, 32:33, :].expand(-1, 5, -1)    # (B,5,3)
                distances = 1 - distance_func(pred_end, gt_end) # (B,5)
                index = distances.argmin(dim=1)                 # (B,)

            gt_cls = index
            row_idx = torch.arange(len(gt_cls), device=gt_cls.device)
            best_traj = pred_traj[row_idx, gt_cls, :, :]        # (B,33,3)

            cls_loss_ = cls_loss(pred_cls, gt_cls)
            reg_loss_ = reg_loss(best_traj, gt).mean(dim=(0, 1))
            loss = cls_loss_ + reg_loss_.mean()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float('nan')

# ----------------- ONNX export -----------------
@torch.no_grad()
def export_flipped_model_onnx(model: nn.Module,
                              val_loader: DataLoader,
                              save_dir: str,
                              prefix: str,
                              flips: List[Dict[str, Any]],
                              opset: int = 13,
                              logs_text: str = None) -> str:
    """
    Exports the (already-flipped) model to ONNX with a short time-based name.
    Uses one batch from val_loader as example inputs (shapes correct).
    Writes a JSON sidecar with full flip metadata + captured logs.
    """
    os.makedirs(save_dir, exist_ok=True)

    # build base name (seconds only) & collision guard
    ts = timestamp_id()  # e.g., "20250927-130542"
    base = f"{prefix}_{ts}"
    onnx_path = os.path.join(save_dir, base + ".onnx")
    json_path = os.path.join(save_dir, base + ".json")
    suffix = 0
    while os.path.exists(onnx_path) or os.path.exists(json_path):
        suffix += 1
        onnx_path = os.path.join(save_dir, f"{base}-{suffix:03d}.onnx")
        json_path = os.path.join(save_dir, f"{base}-{suffix:03d}.json")

    # example inputs (CPU) from data
    device_cpu = torch.device("cpu")
    try:
        batch = next(iter(val_loader))
    except StopIteration:
        raise RuntimeError("val_loader is empty; need one batch to infer input shapes for ONNX export.")

    imgs = torch.randn(1, 12, 128, 256, dtype=torch.float32)
    desire = torch.zeros(1, 8, dtype=torch.float32)
    traffic = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    h0 = torch.zeros(1, 512, dtype=torch.float32)

    # ensure eval + CPU for export
    model.eval()
    model_cpu = model.to(device_cpu)

    torch.onnx.export(
        model,
        (imgs, desire, traffic, h0),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=False,
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input_imgs", "desire", "traffic_convention", "initial_state"],
        output_names=["outputs"],
    )
    # write JSON sidecar with flip metadata + logs
    meta = {
        "arch": model.__class__.__name__,
        "num_flips": len(flips),
        "timestamp": os.path.splitext(os.path.basename(onnx_path))[0].split(prefix + "_", 1)[-1],
        "opset": opset,
    }
    payload = {"flips": flips, "meta": meta}
    if logs_text is not None:
        payload["logs"] = logs_text

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    return onnx_path

# ----------------- data & weights -----------------
def build_loaders(args, device):
    train = Comma2k19SequenceDataset(args.train_index, args.data_root, 'train', use_memcache=False)
    val   = Comma2k19SequenceDataset(args.val_index,   args.data_root, 'val',   use_memcache=False)

    loader_args = dict(num_workers=0, pin_memory=(device.type=='cuda'))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,  **loader_args)
    val_loader   = DataLoader(val,   batch_size=1,               shuffle=False, **loader_args)
    return train_loader, val_loader

def load_weights(model, ckpt_path: str):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"[Load] WARNING: checkpoint not found: {ckpt_path} (using random init).")
        return
    print(f"[Load] loading {ckpt_path}")
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        from collections import OrderedDict
        new_sd = OrderedDict((k.replace('module.', ''), v) for k, v in sd.items())
        model.load_state_dict(new_sd, strict=False)

# ----------------- args & main -----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_index', type=str, default='data/comma2k19_train_non_overlap.txt')
    p.add_argument('--val_index',   type=str, default='data/comma2k19_val_non_overlap.txt')
    p.add_argument('--data_root',   type=str, default='data/comma2k19/')
    p.add_argument('--batch_size',  type=int, default=2)

    p.add_argument('--ckpt',        type=str, default='openpilot_model/supercombo_torch_weights.pth')
    p.add_argument('--mode',        type=str, default='gradxw',
                   choices=['w','grad','gradxw','taylor1','fisher'])
    p.add_argument('--calib_batches', type=int, default=8,
                   help='mini-batches to accumulate importance')

    # selection & flipping
    p.add_argument('--topk',        type=int, default=100, help='how many scalars (weights) to target')
    p.add_argument('--attack',      type=str, default='single',
                   choices=['single','double','triple'],
                   help='number of bits to flip PER selected scalar (MBU)')
    p.add_argument('--bits',        type=str, default='7',
                   help='bit list or preset: "5", "5,10", "5,10,23", "mantissa_low", "mantissa", "exponent", "sign"')
    p.add_argument('--restrict',    type=str, default='',
                   help='comma-separated substrings to restrict params (e.g. "temporal_policy.plan,vision_net")')
    p.add_argument('--kinds',       type=str, default='',
                   help='optional substrings to include by tensor kind (e.g. "weight,bias")')
    p.add_argument('--allow_bias',  type=int, default=1, help='1=allow flipping bias tensors, 0=disallow')

    # runtime & export
    p.add_argument('--amp',         action='store_true', help='use mixed precision')
    p.add_argument('--save_dir',    type=str, default='flipped_models')
    p.add_argument('--save_prefix', type=str, default='model')
    p.add_argument('--onnx_opset',  type=int, default=13, help='ONNX opset version (e.g., 13 or 17)')
    return p.parse_args()

def main():
    # start capturing logs (stdout+stderr) while still printing to console
    restore_streams, log_buffer = start_log_capture()
    try:
        #args = parse_args()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = False

        restrict_to = [s.strip() for s in args.restrict.split(',') if s.strip()] or None
        kinds = [s.strip() for s in args.kinds.split(',') if s.strip()] or None
        allow_bias = bool(args.allow_bias)

        # loaders
        train_loader, val_loader = build_loaders(args, device)
        print("train samples:", len(train_loader.dataset), "val samples:", len(val_loader.dataset))

        # model
        model = OpenPilotModel().to(device)
        load_weights(model, args.ckpt)

        # baseline loss
        base_loss = evaluate_loss_mdn(model, val_loader, device, num_batches=5, use_amp=args.amp)
        print(f"[Eval] baseline traj loss: {base_loss:.6f}")

        # accumulate importance
        print(f"[Importance] mode={args.mode}, batches={args.calib_batches}")
        imp = accumulate_importance(model, train_loader, device,
                                    num_batches=args.calib_batches,
                                    mode=args.mode, use_amp=args.amp)

        # select targets
        items, name_to_param = select_topk_elements(
            imp, model, allow_bias, restrict_to, kinds, topk=args.topk
        )
        if not items:
            print("[Flip] No eligible elements to flip.")
            return

        # decide bits per weight
        bits_per_weight = {'single': 1, 'double': 2, 'triple': 3}[args.attack]
        bits_list = parse_bits_spec(args.bits)

        # flip (MBU: multiple bits within each selected scalar)
        flips = mbu_flip_multi_bits(model, name_to_param, items, bits_list, bits_per_weight)
        print(f"[Flip][MBU] topK={args.topk}, per-weight bits={bits_per_weight}, "
              f"bits_spec={args.bits} -> total flips={len(flips)}")

        # post-flip loss
        post_loss = evaluate_loss_mdn(model, val_loader, device, num_batches=5, use_amp=args.amp)
        print(f"[Eval] post-MBU traj loss: {post_loss:.6f} (Δ={post_loss - base_loss:+.6f})")

        # export ONNX + JSON metadata (now includes logs)
        if flips:
            logs_text = log_buffer.getvalue()
            onnx_path = export_flipped_model_onnx(
                model, val_loader, args.save_dir, args.save_prefix, flips,
                opset=args.onnx_opset, logs_text=logs_text
            )
            print(f"[Save] wrote {onnx_path} and {onnx_path.replace('.onnx', '.json')}")
    finally:
        # restore stdout/stderr no matter what
        restore_streams()

if __name__ == "__main__":
    args = parse_args()
    for i in range(1):
        args.bits = str(i)
        main()
