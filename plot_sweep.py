"""
plot_sweep.py — figures and summary for the RHS sensitivity sweep.

    python plot_sweep.py --runs results/sweep_1p5b.csv --out_dir results/sweep_1p5b

Produces
  fig1_sensitivity_curve.png   mean P(yes) vs offset (k - true) with 95% bootstrap CI over items
  fig2_item_heatmap.png        per-item P(yes) over offset, items sorted by true sum
  fig3_sensitivity_index.png   per-item sensitivity S = P(yes|k=true) - mean P(yes|k wrong), split into
                               undershoot (k < a+b) and overshoot (k > a+b) panels
  fig4_operand_aligned.png     too-small answers only, P(yes) vs k - a and vs k - b: shows whether an
                               operand sets the lower edge of what the model accepts
  summary.txt                  balanced accuracy, yes-rates (under/over), then_corrects (if generated),
                               per-item AUC, S / S_under / S_over quantiles
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, BLUE_LIGHT, ORANGE = "#2a78d6", "#9ec5f4", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 11,
})


def bootstrap_ci(mat, n_boot=2000, seed=0):
    """mat: items x offsets with NaN allowed. Returns mean, lo, hi per column (resampling items)."""
    rng = np.random.default_rng(seed)
    n = mat.shape[0]
    means = np.nanmean(mat, axis=0)
    boots = np.empty((n_boot, mat.shape[1]))
    with warnings.catch_warnings():  # sparse columns can be all-NaN in a resample; nanpercentile skips them
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            boots[i] = np.nanmean(mat[idx], axis=0)
    return means, np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_label", default="")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.runs)
    df["item"] = df["a"].astype(str) + "+" + df["b"].astype(str)
    piv = df.pivot_table(index="item", columns="offset", values="p_yes")
    order = df.groupby("item")["true"].first().sort_values().index
    piv = piv.loc[order]
    offsets = piv.columns.values
    mat = piv.values

    # ---- fig 1: pooled sensitivity curve
    mean, lo, hi = bootstrap_ci(mat)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(offsets, lo, hi, color=BLUE_LIGHT, alpha=0.6, linewidth=0, label="95% bootstrap CI (items)")
    ax.plot(offsets, mean, color=BLUE, linewidth=2, marker="o", markersize=5, label="mean P(yes)")
    ax.axvline(0, color=INK2, linewidth=1, linestyle="--")
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
    ax.set_xlabel("stated result minus true sum  (k − (a+b))")
    ax.set_ylabel("P(yes)  from first-token logits")
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(offsets.min(), offsets.max() + 1, 2))
    ax.set_title(f"Does the verdict depend on the stated result? {args.model_label}".strip())
    ax.legend(frameon=False, loc="upper right")
    ax.text(offsets.min(), 0.03, f"n = {mat.shape[0]} (a,b) items", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig1_sensitivity_curve.png"), dpi=200)
    plt.close(fig)

    # ---- fig 2: per-item heatmap (sequential, one hue)
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.18 * mat.shape[0] + 1.2)))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                   extent=[offsets.min() - 0.5, offsets.max() + 0.5, mat.shape[0] - 0.5, -0.5])
    ax.axvline(0, color=ORANGE, linewidth=1.2)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_xlabel("k − (a+b)")
    ax.set_ylabel("item (sorted by true sum)")
    ax.grid(False)
    ax.set_title("P(yes) per item and stated result")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("P(yes)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig2_item_heatmap.png"), dpi=200)
    plt.close(fig)

    # ---- fig 3: sensitivity index per item, split by direction of the error
    # The smoke run showed the model treats undershoots (k < a+b) and overshoots (k > a+b) very
    # differently, so a single S averaging both sides hides the effect. S_under / S_over each
    # compare P(yes | true) with the mean P(yes) on one side only.
    p_true = piv[0] if 0 in piv.columns else pd.Series(np.nan, index=piv.index)
    under_cols = [c for c in piv.columns if c < 0]
    over_cols = [c for c in piv.columns if c > 0]
    S = (p_true - piv[under_cols + over_cols].mean(axis=1)).dropna()
    S_under = (p_true - piv[under_cols].mean(axis=1)).dropna()
    S_over = (p_true - piv[over_cols].mean(axis=1)).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, s, lab in [(axes[0], S_under, "k < a+b (undershoot)"), (axes[1], S_over, "k > a+b (overshoot)")]:
        ax.hist(s, bins=np.linspace(-1, 1, 21), color=BLUE, edgecolor=SURFACE, linewidth=1)
        ax.axvline(0, color=INK2, linewidth=1, linestyle="--")
        ax.axvline(s.median(), color=ORANGE, linewidth=1.5, label=f"median {s.median():.2f}")
        ax.set_xlabel("S = P(yes | true) − mean P(yes | wrong k)")
        ax.set_title(lab, fontsize=11)
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    axes[0].set_ylabel("items")
    fig.suptitle("Sensitivity per item, by direction of the wrong result")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig3_sensitivity_index.png"), dpi=200)
    plt.close(fig)

    # ---- fig 4: undershoots re-aligned on each operand (where is the lower edge?)
    # The 1.5B sweep suggested a range rule "a <= k <= a+b". Restricting to k < a+b removes the
    # overshoot cliff, so any step left in these curves comes from comparing k with an operand.
    under = df[df.offset < 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, op, name in [(axes[0], "a", "first operand a"), (axes[1], "b", "second operand b")]:
        d = under.assign(rel=under.k - under[op])
        m = d.pivot_table(index="item", columns="rel", values="p_yes")
        keep = m.notna().sum(axis=0) >= 5  # only show positions covered by >= 5 items
        m = m.loc[:, keep]
        mu, lo_, hi_ = bootstrap_ci(m.values)
        x = m.columns.values
        ax.fill_between(x, lo_, hi_, color=BLUE_LIGHT, alpha=0.6, linewidth=0, label="95% bootstrap CI (items)")
        ax.plot(x, mu, color=BLUE, linewidth=2, marker="o", markersize=4, label="mean P(yes)")
        ax.axvline(-0.5, color=ORANGE, linewidth=1.2, label=f"k = {op} boundary")
        ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_xlabel(f"k − {op}   (only wrong answers with k < a+b)")
        ax.set_title(f"aligned on the {name}", fontsize=11)
        ax.set_ylim(0, 1)
        lo_all, hi_all = ((under.k < under[op]).sum(), (under.k >= under[op]).sum())
        pb = under.loc[under.k < under[op], "p_yes"].mean()
        pa = under.loc[under.k >= under[op], "p_yes"].mean()
        ax.text(0.02, 0.04, f"P(yes): k<{op} {pb:.2f} (n={lo_all})   k≥{op} {pa:.2f} (n={hi_all})",
                transform=ax.transAxes, color=INK2, fontsize=8.5)
    axes[0].set_ylabel("P(yes)  from first-token logits")
    axes[1].legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.suptitle(f"Too-small answers: is the lower edge set by an operand? {args.model_label}".strip())
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig4_operand_aligned.png"), dpi=200)
    plt.close(fig)

    # per-item AUC: probability that the true equation gets higher log-odds than a wrong one
    aucs = []
    for _, g in df.groupby("item"):
        t = g.loc[g.is_true == 1, "logodds"]
        if len(t) == 0:
            continue
        fl = g.loc[g.is_true == 0, "logodds"]
        aucs.append((fl < t.iloc[0]).mean() + 0.5 * (fl == t.iloc[0]).mean())
    aucs = pd.Series(aucs)

    # ---- summary
    true_rows = df[df.is_true == 1]
    false_rows = df[df.is_true == 0]
    tpr = (true_rows.verdict == "yes").mean()
    tnr = (false_rows.verdict == "no").mean()
    under_rows, over_rows = df[df.offset < 0], df[df.offset > 0]
    tc = pd.to_numeric(df.get("then_corrects"), errors="coerce") if "then_corrects" in df else pd.Series(dtype=float)
    if tc.notna().any():
        tc_false = tc[df.is_true == 0]
        tc_line = (f"then_corrects (said yes, restated a different result): {int(tc.sum())} rows "
                   f"({tc.mean():.3f} of all, {tc_false.mean():.3f} of false)")
    else:
        tc_line = "then_corrects: not measured (run without --generate)"
    lines = [
        f"rows: {len(df)}  items: {df.item.nunique()}  offsets: {offsets.min()}..{offsets.max()}",
        f"P(yes) on true equations:  mean {true_rows.p_yes.mean():.3f}   greedy-yes rate {tpr:.3f}",
        f"P(yes) on false equations: mean {false_rows.p_yes.mean():.3f}   greedy-yes rate {1 - tnr:.3f}",
        f"  undershoot (k < a+b):    mean {under_rows.p_yes.mean():.3f}   greedy-yes rate {(under_rows.verdict=='yes').mean():.3f}",
        f"  overshoot  (k > a+b):    mean {over_rows.p_yes.mean():.3f}   greedy-yes rate {(over_rows.verdict=='yes').mean():.3f}",
        f"balanced accuracy (mean of true-yes and false-no rates): {(tpr + tnr) / 2:.3f}   "
        f"[true-yes {tpr:.3f}, false-no {tnr:.3f}]",
        f"raw accuracy (greedy verdict == truth): {((df.verdict=='yes') == (df.is_true==1)).mean():.3f}   "
        f"(always-'no' baseline: {(df.is_true==0).mean():.3f})",
        tc_line,
        f"per-item AUC (true eq. ranked above a wrong one; 0.5 = chance): mean {aucs.mean():.3f}  "
        f"median {aucs.median():.3f}",
        f"S (both sides):  median {S.median():.3f}  IQR [{S.quantile(.25):.3f}, {S.quantile(.75):.3f}]  "
        f"S < 0.1: {(S<0.1).sum()}/{len(S)}   S > 0.5: {(S>0.5).sum()}/{len(S)}",
        f"S_under:         median {S_under.median():.3f}  IQR [{S_under.quantile(.25):.3f}, {S_under.quantile(.75):.3f}]  "
        f"S < 0.1: {(S_under<0.1).sum()}/{len(S_under)}   S > 0.5: {(S_under>0.5).sum()}/{len(S_under)}",
        f"S_over:          median {S_over.median():.3f}  IQR [{S_over.quantile(.25):.3f}, {S_over.quantile(.75):.3f}]  "
        f"S < 0.1: {(S_over<0.1).sum()}/{len(S_over)}   S > 0.5: {(S_over>0.5).sum()}/{len(S_over)}",
        "",
        "P(yes) by offset (mean over items):",
    ]
    for o, m in zip(offsets, mean):
        lines.append(f"  {int(o):+3d}: {m:.3f}")
    lines += ["", "P(yes) on false equations by true-sum bucket:"]
    df["bucket"] = pd.cut(df["true"], [0, 10, 20, 30, 45], labels=["2-10", "11-20", "21-30", "31-40"])
    for bkt, g in false_rows.assign(bucket=df.loc[false_rows.index, "bucket"]).groupby("bucket", observed=True):
        lines.append(f"  true sum {bkt}: {g.p_yes.mean():.3f}  (n={len(g)})")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
