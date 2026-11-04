"""
plot_fig1_vs_ideal.py — Fig 1 (pooled sensitivity curve) with an ideal-verifier benchmark overlaid.

    python plot_fig1_vs_ideal.py --runs results/sweep_1p5b.csv --out results/sweep_1p5b/fig1b_vs_ideal.png

The benchmark is a STATIC, SYNTHETIC curve (no model is called): an exact equality checker says
yes iff k = a+b, i.e. P(yes) = 1 at offset 0 and 0 everywhere else. Label it honestly with
--ideal_label; the default names the rule, not a model, because no frontier model was measured.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_sweep import bootstrap_ci, SURFACE, INK, INK2, BLUE, BLUE_LIGHT

ORANGE_LIGHT = "#f5a86f"  # lighter companion to plot_sweep's ORANGE #eb6834

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 11,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_label", default="Qwen2.5-1.5B")
    ap.add_argument("--ideal_label", default="ideal verifier (equality rule, synthetic)")
    ap.add_argument("--legend_loc", default="center right")
    ap.add_argument("--footnote", default="", help="small gray note at the bottom right of the axes")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    df = pd.read_csv(args.runs)
    df["item"] = df["a"].astype(str) + "+" + df["b"].astype(str)
    piv = df.pivot_table(index="item", columns="offset", values="p_yes")
    offsets = piv.columns.values
    mean, lo, hi = bootstrap_ci(piv.values)

    ideal = (offsets == 0).astype(float)  # yes iff k = a+b

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(offsets, lo, hi, color=BLUE_LIGHT, alpha=0.6, linewidth=0,
                    label="95% bootstrap CI (items)")
    ax.plot(offsets, mean, color=BLUE, linewidth=2, marker="o", markersize=5,
            label=f"{args.model_label}  mean P(yes)")
    ax.plot(offsets, ideal, color=ORANGE_LIGHT, linewidth=2, marker="s", markersize=4,
            linestyle="--", label=args.ideal_label, zorder=1)
    ax.axvline(0, color=INK2, linewidth=1, linestyle="--")
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
    ax.set_xlabel("stated result minus true sum  (k − (a+b))")
    ax.set_ylabel("P(yes)  from first-token logits")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(np.arange(offsets.min(), offsets.max() + 1, 2))
    ax.set_title("Verdict vs stated result: measured model against an ideal checker")
    ax.legend(frameon=False, loc=args.legend_loc, fontsize=9)
    ax.text(offsets.min(), 0.06, f"n = {piv.shape[0]} (a,b) items", color=INK2, fontsize=9)
    if args.footnote:
        ax.text(0.99, 0.07, args.footnote, transform=ax.transAxes, ha="right",
                color=INK2, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
