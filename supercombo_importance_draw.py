# file: plot_conv_topk_step.py
# usage:
#   python plot_conv_topk_step.py \
#       --scores score_dict.pt \
#       --topk 50 500 0.01 \
#       --match conv \
#       --out importance_location.png
#
# Input:
#   --scores: torch.save()'d dict {param_name: tensor_of_importance_scores}
# Behavior:
#   * Only parameters whose name contains --match (case-insensitive) AND endswith ".weight"
#   * Each matched tensor counts as ONE conv layer; layer index increments by encounter order.
#   * Select global Top-K elements (K can be an integer or a fraction in (0,1]).
#   * For each K, draw a step plot: vertical up to the count at that layer, then horizontal to next layer.

import argparse, math, torch
import matplotlib.pyplot as plt

def parse_topk(values, total):
    out=[]
    for v in values:
        if "." in v or "e" in v.lower():
            f=float(v)
            if not (0.0 < f <= 1.0):
                raise ValueError("--topk fraction must be in (0,1]")
            out.append(max(1, int(math.ceil(total*f))))
        else:
            out.append(int(v))
    # unique while preserving order
    seen=set(); res=[]
    for k in out:
        if k not in seen:
            res.append(k); seen.add(k)
    return res

def collect_conv_elements(score_dict, match="conv"):
    """Return (layer_names, per_elem) where per_elem is list[(layer_idx, score)]."""
    layer_names=[]
    per_elem=[]
    m = match.lower()
    for name, sc in score_dict.items():
        n = name.lower()
        if m not in n:          # e.g., 'conv' or 'conv'
            continue
        if not n.endswith(".weight"):
            continue
        t = sc if isinstance(sc, torch.Tensor) else torch.tensor(sc)
        t = t.detach().float().reshape(-1)
        lidx = len(layer_names)  # new conv layer
        layer_names.append(name)
        per_elem.extend((lidx, float(v)) for v in t)
    return layer_names, per_elem

def counts_for_topk(per_elem, K, n_layers):
    # pick global top-K by score
    scores = torch.tensor([s for _, s in per_elem])
    K = min(K, scores.numel())
    _, idx = torch.topk(scores, K, largest=True, sorted=False)
    idx = set(idx.tolist())
    counts = [0]*n_layers
    for i,(lidx,_) in enumerate(per_elem):
        if i in idx:
            counts[lidx]+=1
    return counts

def step_plot(ax, counts, *, color, label, baseline=1, lw=2, alpha=0.95):
    """
    Draw 'up-then-right' steps at integer x positions:
      - vertical from baseline to count at x=i,
      - horizontal to x=i+1 at same height.
    """
    n = len(counts)
    x, y = [], []
    cur_y = baseline
    for i, c in enumerate(counts):
        h = max(baseline, c)        # ensure >= baseline for log-scale
        # vertical up at x=i
        x.extend([i, i])
        y.extend([cur_y, h])
        # horizontal right to x=i+1
        x.extend([i, i+1])
        y.extend([h, h])
        cur_y = h
    ax.plot(x, y, color=color, linewidth=lw, alpha=alpha, label=label)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default='flipped_models/score_dict.pt', help="Path to torch-saved dict {name: tensor(scores)}")
    ap.add_argument("--topk", nargs="+", default=["50","500","0.01"],
                    help="Integers and/or fractions (0,1]; e.g., 50 500 0.01")
    ap.add_argument("--match", default="conv", help="Substring to match conv layers (default: 'conv'). Use 'conv' if needed.")
    ap.add_argument("--out", default="importance_location.png")
    ap.add_argument("--title", default="Locations of Top-k Important Weights by Layer")
    args = ap.parse_args()

    score_dict = torch.load(args.scores, map_location="cpu")
    if not isinstance(score_dict, dict):
        raise TypeError("scores must load to a dict[name -> tensor]")

    layer_names, per_elem = collect_conv_elements(score_dict, match=args.match)
    if not per_elem:
        raise RuntimeError(f"No layers found with substring '{args.match}' and '.weight'.")

    n_layers = len(layer_names)
    total = len(per_elem)
    topks = parse_topk(args.topk, total)

    # compute counts for each K
    k2counts = []
    for K in topks:
        counts = counts_for_topk(per_elem, K, n_layers)
        k2counts.append((K, counts))

    # plot
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=140)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (K, counts) in enumerate(k2counts):
        step_plot(ax, counts, color=colors[i % len(colors)],
                  label=f"Top {K} (Most Critical)", lw=2, alpha=0.92)

    ax.set_yscale("log")
    ax.set_xlabel("Layer (index)")     # increments by 1 per matched conv weight tensor
    ax.set_ylabel("Number of Weights (Log Scale)")
    ax.set_title(args.title)
    ax.legend()
    ax.set_xlim(0, n_layers)           # include last horizontal segment
    ax.set_ylim(1, None)               # baseline at 1 for log scale
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved: {args.out}")
    print(f"layers counted: {n_layers}")
    print("e.g., first 5 layers:")
    for i, n in enumerate(layer_names[:5]):
        print(f"  [{i}] {n}")

if __name__ == "__main__":
    main()
