# bit_flip_multi.py  — revised to mirror a working openpilot method
import os, re, math, json, time, argparse, sys, io
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

import onnx
import onnxruntime as ort

# ---- your project imports ----
from openpilot_torch import OpenPilotModel
from data import Comma2k19SequenceDataset

# ----------------- tiny tee logger (capture stdout+stderr but still print) -----------------
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):         [s.write(data) for s in self.streams]
    def flush(self):               [s.flush() for s in self.streams]

def start_log_capture():
    buf = io.StringIO()
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, buf); sys.stderr = _Tee(orig_err, buf)
    def restore():
        sys.stdout = orig_out; sys.stderr = orig_err
    return restore, buf

# ----------------- losses & helpers -----------------
distance_func = nn.CosineSimilarity(dim=2)
cls_loss = nn.CrossEntropyLoss()
reg_loss = nn.SmoothL1Loss(reduction='none')

def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def make_supercombo_inputs(batch, device, K: int = 33):
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
    with torch.no_grad():
        if mode == 'w':      return p.detach().abs()
        if p.grad is None:   return torch.zeros_like(p, dtype=torch.float32)
        if mode == 'grad':   return p.grad.detach().abs()
        if mode in ('gradxw','taylor1'): return (p.grad.detach() * p.detach()).abs()
        if mode == 'fisher': return (p.grad.detach() ** 2)
        raise ValueError(f'unknown mode {mode}')

# ----------------- importance accumulation -----------------
def accumulate_importance(model: nn.Module, data_loader: DataLoader,
                          device: torch.device, num_batches: int,
                          mode: str = 'gradxw', use_amp: bool = True) -> Dict[str, torch.Tensor]:
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    running: Dict[str, torch.Tensor] = {}
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
            out = model(imgs12, desire, traffic, h0)      # (B, ~6609/6690 depending on variant)
            B = out.shape[0]
            plan = out[:, :5 * 991].view(B, 5, 991)

            pred_cls = plan[:, :, -1]
            params_flat = plan[:, :, :-1]
            pred_trajectory = params_flat.view(B, 5, 2, 33, 15)[:, :, 0, :, :3]

            with torch.no_grad():
                pred_end = pred_trajectory[:, :, 32, :]
                gt_end   = gt[:, 32:33, :].expand(-1, 5, -1)
                distances = 1 - distance_func(pred_end, gt_end)
                index = distances.argmin(dim=1)

            gt_cls = index
            row_idx = torch.arange(len(gt_cls), device=gt_cls.device)
            best_traj = pred_trajectory[row_idx, gt_cls, :, :]

            cls_loss_ = cls_loss(pred_cls, gt_cls)
            reg_loss_ = reg_loss(best_traj, gt).mean(dim=(0, 1))
            loss = cls_loss_ + reg_loss_.mean()

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        scaler.update()

        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad: continue
                s = score_tensor(p, mode).detach()
                running[name] = s.clone() if name not in running else (running[name] + s)
        processed += 1

    if processed > 0:
        for k in running.keys(): running[k] = running[k] / float(processed)
    return running

def flatten_scores(score_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[str,int,int]]]:
    flats, mapping, offset = [], [], 0
    for name, t in score_dict.items():
        v = t.reshape(-1).cpu()
        flats.append(v)
        mapping.append((name, t.numel(), offset))
        offset += t.numel()
    if not flats:
        return torch.empty(0), []
    return torch.cat(flats, dim=0), mapping

# ----------------- bit flip (tuple-assign, safe) -----------------
def _unravel(flat_idx: int, shape: torch.Size):
    return np.unravel_index(int(flat_idx), tuple(shape), order='C')

@torch.no_grad()
def flip_bit_assign_(t: torch.Tensor, index_tuple, bit_idx: int) -> float:
    """
    Flip one bit of a single scalar and assign it back to tensor t at index_tuple.
    Returns the new float value (or leaves unchanged if it would become non-finite).
    """
    # pull scalar value
    val = float(t[index_tuple].item())
    # make a 1-element float32 tensor, view as int32, flip, view back
    buf = torch.tensor([val], dtype=torch.float32)
    int_view = buf.view(torch.int32)
    int_view[0] = int_view[0] ^ (1 << int(bit_idx))
    flipped = float(buf.view(torch.float32)[0].item())
    # safety: avoid NaN/Inf (common if exponent/sign combos go wild)
    if not (math.isfinite(flipped)):
        return val  # skip
    t[index_tuple] = flipped
    return flipped

@torch.no_grad()
def bitflip_inplace_and_log(param: torch.Tensor,
                            flat_indices: torch.Tensor,
                            bit: int,
                            param_name: str) -> List[Dict[str, Any]]:
    """
    Tuple-assign version (matches the method that works in openpilot).
    """
    flips: List[Dict[str, Any]] = []
    if param.dtype != torch.float32 or flat_indices.numel() == 0:
        return flips
    for fi in flat_indices.detach().cpu().to(torch.long).tolist():
        idx_tup = tuple(map(int, _unravel(int(fi), param.shape)))
        old = float(param[idx_tup].item())
        new = flip_bit_assign_(param, idx_tup, bit)
        if new != old:
            flips.append({
                "name": param_name,
                "bit": int(bit),
                "index_flat": int(fi),
                "index": idx_tup,
                "old": old,
                "new": new,
            })
    return flips

def filter_params_for_flipping(model: nn.Module,
                               allow_bias: bool,
                               restrict_to: List[str] = None,
                               kinds: List[str] = None) -> Dict[str, torch.Tensor]:
    out = {}
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.dtype != torch.float32: continue
        if not allow_bias and name.endswith(".bias"): continue
        if restrict_to is not None and not any(tag in name for tag in restrict_to): continue
        if kinds is not None and not any(k in name for k in kinds): continue
        out[name] = p
    return out

def select_topk_elements(score_dict: Dict[str, torch.Tensor],
                         model: nn.Module,
                         allow_bias: bool,
                         restrict_to: List[str] = None,
                         kinds: List[str] = None,
                         topk: int = 100) -> Tuple[List[Tuple[str,int]], Dict[str, torch.Tensor]]:
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
_MANTISSA      = list(range(0, 23))
_EXPONENT      = list(range(23, 31))
_SIGN          = [31]
_MANTISSA_LOW  = list(range(0, 7))

def parse_bits_spec(bits_spec: str) -> List[int]:
    s = bits_spec.strip().lower()
    if s == 'mantissa':     return _MANTISSA
    if s == 'mantissa_low': return _MANTISSA_LOW
    if s == 'exponent':     return _EXPONENT
    if s == 'sign':         return _SIGN
    bits = [int(b.strip()) for b in s.split(',') if b.strip()!='']
    if not bits: raise ValueError(f'invalid --bits: {bits_spec}')
    for b in bits:
        if b < 0 or b > 31: raise ValueError(f'bit {b} out of range [0,31]')
    return bits

@torch.no_grad()
def mbu_flip_multi_bits(model: nn.Module,
                        name_to_param: Dict[str, torch.Tensor],
                        items: List[Tuple[str,int]],
                        bits: List[int],
                        bits_per_weight: int) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    if not items: return logs
    for (pname, lidx) in items:
        p = name_to_param[pname]
        for j in range(bits_per_weight):
            bit = bits[min(j, len(bits)-1)]
            logs += bitflip_inplace_and_log(p, torch.tensor([lidx], dtype=torch.long), bit, pname)
    return logs

# ----------------- eval -----------------
@torch.no_grad()
def evaluate_loss_mdn(model, data_loader, device, num_batches: int = 5, use_amp: bool = True):
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
            out  = model(imgs12, desire, traffic, h0)
            B    = out.shape[0]
            plan = out[:, :5 * 991].view(B, 5, 991)

            pred_cls = plan[:, :, -1]
            params_flat = plan[:, :, :-1]
            pred_traj = params_flat.view(B, 5, 2, 33, 15)[:, :, 0, :, :3]

            with torch.no_grad():
                pred_end = pred_traj[:, :, 32, :]
                gt_end   = gt[:, 32:33, :].expand(-1, 5, -1)
                distances = 1 - distance_func(pred_end, gt_end)
                index = distances.argmin(dim=1)

            gt_cls = index
            row_idx = torch.arange(len(gt_cls), device=gt_cls.device)
            best_traj = pred_traj[row_idx, gt_cls, :, :]

            cls_loss_ = cls_loss(pred_cls, gt_cls)
            reg_loss_ = reg_loss(best_traj, gt).mean(dim=(0, 1))
            loss = cls_loss_ + reg_loss_.mean()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float('nan')

# ----------------- ONNX helpers -----------------
def read_original_onnx_signature(path: str):
    if not (path and os.path.isfile(path)):
        return None
    m = onnx.load(path)
    opset = max((op.version for op in m.opset_import), default=None)
    try:
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        outs = sess.get_outputs()
        out_name = 'outputs' if any(o.name=='outputs' for o in outs) else outs[0].name
        # build a zero feed using model-declared shapes (fallback to known shapes)
        feed = {}
        for i in sess.get_inputs():
            shp = [d if isinstance(d, int) and d>0 else {
                'input_imgs': [1,12,128,256],
                'desire': [1,8],
                'traffic_convention': [1,2],
                'initial_state': [1,512],
            }.get(i.name, None)[k] for k, d in enumerate(i.shape)]
            if None in shp:  # fallback if any dim unresolved
                shp = {
                    'input_imgs': [1,12,128,256],
                    'desire': [1,8],
                    'traffic_convention': [1,2],
                    'initial_state': [1,512],
                }[i.name]
            feed[i.name] = np.zeros(tuple(shp), np.float32)
        y = sess.run([out_name], feed)[0]
        out_len = int(y.shape[1])
        return {'opset': int(opset) if opset else None, 'out_name': out_name, 'out_len': out_len}
    except Exception:
        return {'opset': int(opset) if opset else None, 'out_name': 'outputs', 'out_len': None}

@torch.no_grad()
def export_flipped_model_onnx(model: nn.Module,
                              save_dir: str,
                              prefix: str,
                              flips: List[Dict[str, Any]],
                              mirror_sig: Dict[str, Any] = None,
                              fallback_opset: int = 17,
                              fold_constants: bool = True,
                              logs_text: str = None) -> str:
    os.makedirs(save_dir, exist_ok=True)
    ts = timestamp_id()
    base = f"{prefix}_{ts}"
    onnx_path = os.path.join(save_dir, base + ".onnx")
    json_path = os.path.join(save_dir, base + ".json")
    i = 0
    while os.path.exists(onnx_path) or os.path.exists(json_path):
        i += 1
        onnx_path = os.path.join(save_dir, f"{base}-{i:03d}.onnx")
        json_path = os.path.join(save_dir, f"{base}-{i:03d}.json")

    # dummy inputs (torch tensors, static shapes)
    imgs   = torch.zeros(1, 12, 128, 256, dtype=torch.float32)
    desire = torch.zeros(1, 8, dtype=torch.float32)
    traffic= torch.tensor([[1., 0.]], dtype=torch.float32)
    h0     = torch.zeros(1, 512, dtype=torch.float32)

    model.eval(); model_cpu = model.cpu()

    opset = (mirror_sig or {}).get('opset') or fallback_opset
    out_name = (mirror_sig or {}).get('out_name') or 'outputs'
    torch.onnx.export(
        model_cpu,
        (imgs, desire, traffic, h0),
        onnx_path,
        input_names=["input_imgs", "desire", "traffic_convention", "initial_state"],
        output_names=[out_name],
        export_params=True,
        opset_version=opset,
        do_constant_folding=bool(fold_constants),
        training=torch.onnx.TrainingMode.EVAL,
        dynamic_axes=None
    )

    # (optional) verify with ORT that output length matches original (if known)
    try:
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        y = sess.run([out_name], {
            "input_imgs": np.zeros((1,12,128,256), np.float32),
            "desire": np.zeros((1,8), np.float32),
            "traffic_convention": np.array([[1.,0.]], np.float32),
            "initial_state": np.zeros((1,512), np.float32),
        })[0]
        out_len = int(y.shape[1])
    except Exception:
        out_len = None

    meta = {
        "arch": model.__class__.__name__,
        "num_flips": len(flips),
        "timestamp": os.path.splitext(os.path.basename(onnx_path))[0].split(prefix + "_", 1)[-1],
        "opset": opset,
        "fold_constants": bool(fold_constants),
        "out_len": out_len,
        "mirrored": (mirror_sig is not None)
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
    p.add_argument('--calib_batches', type=int, default=8)

    # selection & flipping
    p.add_argument('--topk',        type=int, default=100)
    p.add_argument('--attack',      type=str, default='single',
                   choices=['single','double','triple'])
    p.add_argument('--bits',        type=str, default='sign',   # default to safe & working like your demo
                   help='bit list or preset: "5", "5,10", "mantissa_low", "mantissa", "exponent", "sign"')
    p.add_argument('--restrict',    type=str, default='')
    p.add_argument('--kinds',       type=str, default='')
    p.add_argument('--allow_bias',  type=int, default=1)

    # export & mirroring
    p.add_argument('--orig_onnx',   type=str, default='openpilot_model/supercombo.onnx',
                   help='mirror opset/output name/length from this ONNX if available')
    p.add_argument('--onnx_opset',  type=int, default=17, help='fallback opset if mirroring not possible')
    p.add_argument('--fold_const',  type=int, default=1,  help='1=enable constant folding (matches your working method)')
    p.add_argument('--save_dir',    type=str, default='flipped_models')
    p.add_argument('--save_prefix', type=str, default='model')
    p.add_argument('--amp',         action='store_true')
    return p.parse_args()

def main():
    restore_streams, log_buffer = start_log_capture()
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = False

        # args from outer scope
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

        # flip (tuple-assign, safe)
        flips = mbu_flip_multi_bits(model, name_to_param, items, bits_list, bits_per_weight)
        print(f"[Flip] topK={args.topk}, per-weight bits={bits_per_weight}, "
              f"bits_spec={args.bits} -> total flips={len(flips)}")

        # post-flip loss
        post_loss = evaluate_loss_mdn(model, val_loader, device, num_batches=5, use_amp=args.amp)
        print(f"[Eval] post-flip traj loss: {post_loss:.6f} (Δ={post_loss - base_loss:+.6f})")

        # mirror original ONNX opset/output if available
        mirror_sig = read_original_onnx_signature(args.orig_onnx)
        if mirror_sig:
            print(f"[Mirror] using opset={mirror_sig['opset']} out_name={mirror_sig['out_name']} "
                  f"orig_out_len={mirror_sig['out_len']}")
        else:
            print(f"[Mirror] original ONNX not available; fallback opset={args.onnx_opset}")

        # export ONNX + JSON metadata
        if flips:
            logs_text = log_buffer.getvalue()
            onnx_path = export_flipped_model_onnx(
                model, args.save_dir, args.save_prefix, flips,
                mirror_sig=mirror_sig, fallback_opset=args.onnx_opset,
                fold_constants=bool(args.fold_const), logs_text=logs_text
            )
            print(f"[Save] wrote {onnx_path} and {onnx_path.replace('.onnx', '.json')}")
    finally:
        restore_streams()

if __name__ == "__main__":
    args = parse_args()
    # if you want to sweep bits (like before), do it outside or parametrize; default keeps your working style
    main()
