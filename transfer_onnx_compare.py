# transfer_onnx.py
import os
import argparse
import numpy as np
import torch
import onnxruntime as ort

from openpilot_torch import OpenPilotModel, load_weights_from_onnx


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def export_to_onnx(model, onnx_path, imgs, desire, traffic, h0, mode: str):
    """
    mode: 'train' -> keep BatchNormalization nodes (no folding)
          'eval'  -> typical inference export (may fuse BN)
    """
    if mode == "train":
        model.train()
        do_fold = False
        training_flag = torch.onnx.TrainingMode.TRAINING
    else:
        model.eval()
        do_fold = True
        training_flag = torch.onnx.TrainingMode.EVAL

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (imgs, desire, traffic, h0),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.TRAINING,
            input_names=["input_imgs", "desire", "traffic_convention", "initial_state"],
            output_names=["outputs"],
            verbose=False,
        )
    print(f"✅ Exported ONNX → {onnx_path} (mode={mode})")


def run_onnx(onnx_path, imgs, desire, traffic, h0):
    sess = ort.InferenceSession(onnx_path, None)
    out = sess.run(
        ["outputs"],
        {
            "input_imgs": to_numpy(imgs),
            "desire": to_numpy(desire),
            "traffic_convention": to_numpy(traffic),
            "initial_state": to_numpy(h0),
        },
    )[0]
    return out


def run_torch(model, imgs, desire, traffic, h0):
    with torch.no_grad():
        out = model(imgs, desire, traffic, h0)
    return to_numpy(out)


def compare_outputs(y_onnx: np.ndarray, y_torch: np.ndarray, tol: float = 1e-5):
    diff = np.abs(y_onnx - y_torch)
    max_abs = diff.max()
    mean_abs = diff.mean()
    # avoid divide-by-zero
    denom = np.maximum(np.abs(y_torch), 1e-12)
    max_rel = (diff / denom).max()
    passed = bool(max_abs <= tol or max_rel <= 1e-4)  # allow small relative tolerance
    print("\n=== Equivalence check ===")
    print(f"shape: {y_onnx.shape}")
    print(f"max |Δ| = {max_abs:.6g}")
    print(f"mean|Δ| = {mean_abs:.6g}")
    print(f"max rel Δ = {max_rel:.6g}")
    print(f"RESULT: {'PASS ✅' if passed else 'FAIL ❌'} (tol={tol})")
    return passed


def print_slices(tag, y):
    print(f"\n########## {tag} ##########")
    print("shape:", y.shape)
    # lead prob 3-dim slice
    print("Lead prob:", sigmoid(y[0, 6010:6013]))
    # relative x/y positions (your indices)
    x_rel = y[0, 5755:5779:4]
    y_rel = y[0, 5756:5780:4]
    print("x_rel:", x_rel)
    print("y_rel:", y_rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", default="openpilot_model/supercombo_torch_weights.pth")
    ap.add_argument("--onnx", default="models/model.onnx")
    ap.add_argument("--origi_onnx", default="openpilot_model/supercombo_server3.onnx")
    ap.add_argument("--mode", choices=["train", "eval"], default="eval",
                    help="Export style: 'train' keeps BN nodes; 'eval' may fuse BN.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0, help="Seed for reproducible dummy inputs")
    ap.add_argument("--tol", type=float, default=1e-5, help="Absolute tolerance for equality check")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # 1) Build and load weights
    model = OpenPilotModel().to(device)
    load_weights_from_onnx(model)
    torch.save(model.state_dict(), 'openpilot_model/supercombo_torch_weights.pth')
    # sd = torch.load(args.pth, map_location="cpu")
    # if isinstance(sd, dict) and "state_dict" in sd:
    #     sd = sd["state_dict"]
    # missing, unexpected = model.load_state_dict(sd, strict=False)
    # if missing:   print(f"[load][missing] {len(missing)} keys (showing first 10): {missing[:10]}")
    # if unexpected:print(f"[load][unexpected] {len(unexpected)} keys (showing first 10): {unexpected[:10]}")

    # 2) Create identical dummy inputs (float32)
    imgs = torch.randn(1, 12, 128, 256, dtype=torch.float32, device=device)
    desire = torch.zeros(1, 8, dtype=torch.float32, device=device)
    traffic = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device)
    h0 = torch.zeros(1, 512, dtype=torch.float32, device=device)

    # 3) Export to ONNX
    export_to_onnx(model, args.onnx, imgs, desire, traffic, h0, mode=args.mode)

    # 4) Run both models on the SAME input
    #    IMPORTANT: keep model in the same mode as export for fair comparison
    if args.mode == "train":
        model.train()
    else:
        model.eval()

    y_torch = run_torch(model, imgs, desire, traffic, h0)
    y_onnx = run_onnx(args.onnx, imgs, desire, traffic, h0)
    y_origi_onnx = run_onnx(args.origi_onnx, imgs, desire, traffic, h0)
    # 5) Print slices and compare numerically
    print_slices("PyTorch Model Output", y_torch)
    #print_slices("ONNX Model Output", y_onnx)
    print_slices("Original ONNX Model Output", y_origi_onnx)
    print("The output diff between the saved model.onnx and the converted model.pth")
    compare_outputs(y_onnx, y_torch, tol=args.tol)
    print("The output diff between the original model.onnx and the converted model.pth")
    compare_outputs(y_origi_onnx, y_torch, tol=args.tol)
    print("The output diff between the original model.onnx and the saved model.onnx")
    compare_outputs(y_onnx, y_origi_onnx, tol=args.tol)
    # 6) (Optional) save PyTorch weights again if you want a clean copy
    # torch.save(model.state_dict(), "openpilot_model/supercombo_torch_weights_checked.pth")


if __name__ == "__main__":
    main()
