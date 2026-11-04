"""
exp3_probes.py — "When" experiment 3: contrast probes (difference-of-means / mass-mean probing).

DEPRECATED from the main pipeline (see README.md): the probes agreed with patching everywhere but
are correlational and add a ~400 MB activation cache. Kept for the record and because exp6_ov.py
currently reads the cache this script writes; results preserved in results/when_exp3/.

For each (layer, position), compute the mean residual-stream difference between two contrast
classes on TRAINING items and measure held-out AUC of the projection on TEST items (items never
shared between train and test, 5 random splits). Basis-free: detects the verdict/comparison even
when it is not yet aligned with the unembedding, but correlational — a decodable feature need not
be causally used.

Contrasts (each probed independently):
  A. verdict   — model says yes vs no (logodds > 0), all rows: when is the DECISION decodable?
  B. over      — k > a+b vs k <= a+b, all rows: the upper-edge comparison.
  C. below_a   — k < a vs a <= k < a+b, undershoot rows only: the lower-edge comparison.

Positions cached: '=', k first, k last, k_last+1 (' correct'), k_last+2 ('?'), final prompt token.

Usage:
    python exp3_probes.py --out_dir results/when_exp3 --cache <path>/acts.npz
Re-running with an existing cache skips the forwards.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rhs_sweep import (SYSTEM, USER_TEMPLATE, YES_VARIANTS, NO_VARIANTS,  # noqa: F401
                       build_inputs, locate_positions, single_token_ids)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#2e8b57", "#c0392b"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 11,
})

POS_NAMES = ["eq", "k_first", "k_last", "after1", "qmark", "final"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--out_dir", default="results/when_exp3")
    p.add_argument("--cache", default="", help="npz file for cached activations (reused if present)")
    p.add_argument("--n_pairs", type=int, default=40)
    p.add_argument("--lo", type=int, default=1)
    p.add_argument("--hi", type=int, default=20)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n_splits", type=int, default=5)
    return p.parse_args()


def sweep_pairs(args):
    import random
    random.seed(args.seed)
    all_pairs = [(a, b) for a in range(args.lo, args.hi + 1) for b in range(args.lo, args.hi + 1)]
    return random.sample(all_pairs, args.n_pairs)


def collect(args):
    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model} ({args.dtype}, {args.device})")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    yes_ids = single_token_ids(tok, YES_VARIANTS)
    no_ids = single_token_ids(tok, NO_VARIANTS)
    pairs = sweep_pairs(args)

    meta, acts = [], {p: [] for p in POS_NAMES}
    t0, n = time.time(), 0
    total = len(pairs) * (2 * args.window + 1)
    for a, b in pairs:
        true = a + b
        for k in range(true - args.window, true + args.window + 1):
            n += 1
            if k < 0:
                continue
            enc = build_inputs(tok, a, b, k)
            ids = enc["input_ids"][0].tolist()
            eq, k1, k2, _ = locate_positions(tok, ids, k)
            enc = {kk: v.to(args.device) for kk, v in enc.items()}
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            lo = (torch.logsumexp(out.logits[0, -1, yes_ids].float(), 0)
                  - torch.logsumexp(out.logits[0, -1, no_ids].float(), 0)).item()
            pos = dict(eq=eq, k_first=k1, k_last=k2, after1=k2 + 1, qmark=k2 + 2, final=len(ids) - 1)
            hs = out.hidden_states  # n_layers+1 entries; [-1] already final-normed (fine for probing)
            for pname, p in pos.items():
                acts[pname].append(np.stack([hs[L][0, p].numpy() for L in range(n_layers + 1)])
                                   .astype(np.float16))
            meta.append((a, b, k, true, lo))
            if n % 50 == 0 or n == total:
                print(f"[{n}/{total}] {(time.time()-t0)/60:.1f} min", flush=True)
    meta = np.array(meta, dtype=np.float64)
    data = {f"acts_{p}": np.stack(acts[p]) for p in POS_NAMES}
    np.savez_compressed(args.cache, meta=meta, **data)
    print(f"cached activations -> {args.cache}")


def auc_score(scores, labels):
    """Rank-based AUC (probability a positive outranks a negative)."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if not args.cache:
        args.cache = os.path.join(args.out_dir, "acts.npz")
    if not os.path.exists(args.cache):
        collect(args)
    z = np.load(args.cache)
    meta = z["meta"]  # a, b, k, true, final_logodds
    a, b, k, true, lo = meta.T
    item_id = a * 1000 + b
    labels = {
        "A_verdict": (lo > 0).astype(int),
        "B_over": (k > true).astype(int),
        "C_below_a": (k < a).astype(int),
    }
    masks = {
        "A_verdict": np.ones(len(meta), bool),
        "B_over": np.ones(len(meta), bool),
        "C_below_a": k < true,  # undershoots only: below-a vs in-range
    }
    acts = {p: z[f"acts_{p}"] for p in POS_NAMES}  # (N, L+1, d) float16
    n_layerpts = acts["final"].shape[1]
    items = np.unique(item_id)
    rng = np.random.default_rng(args.seed)

    # held-out AUC per (contrast, position, layer), averaged over splits
    results = {}
    for cname, y in labels.items():
        m = masks[cname]
        A = np.zeros((args.n_splits, len(POS_NAMES), n_layerpts))
        for s in range(args.n_splits):
            test_items = rng.choice(items, size=len(items) // 4, replace=False)
            te = m & np.isin(item_id, test_items)
            tr = m & ~np.isin(item_id, test_items)
            if y[tr].sum() in (0, tr.sum()) or y[te].sum() in (0, te.sum()):
                A[s] = np.nan
                continue
            for pi, pname in enumerate(POS_NAMES):
                X = acts[pname].astype(np.float32)
                for L in range(n_layerpts):
                    Xtr, Xte = X[tr, L], X[te, L]
                    d = Xtr[y[tr] == 1].mean(0) - Xtr[y[tr] == 0].mean(0)
                    A[s, pi, L] = auc_score(Xte @ d, y[te])
        results[cname] = np.nanmean(A, axis=0)  # (positions, layers)
        print(f"{cname}: done", flush=True)

    np.savez(os.path.join(args.out_dir, "probe_auc.npz"),
             **{c: v for c, v in results.items()}, positions=np.array(POS_NAMES))

    # ---------- figures ----------
    layers = np.arange(n_layerpts)
    # AUC vs layer at k_last and final, per contrast
    fig, ax = plt.subplots(figsize=(8, 4.6))
    styles = {"A_verdict": (INK, "-"), "B_over": (RED, "-"), "C_below_a": (ORANGE, "-")}
    for cname, (color, ls) in styles.items():
        ax.plot(layers, results[cname][POS_NAMES.index("k_last"), :], color=color, linestyle="--",
                linewidth=1.6, label=f"{cname} @ k last token")
        ax.plot(layers, results[cname][POS_NAMES.index("final"), :], color=color, linestyle="-",
                linewidth=2, label=f"{cname} @ final position")
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
    ax.axhline(0.9, color=INK2, linewidth=0.8, linestyle="--")
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out AUC (items never shared)")
    ax.set_ylim(0.35, 1.02)
    ax.set_title("Contrast probes: when is each signal linearly decodable?")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_e3_auc_curves.png"), dpi=200)
    plt.close(fig)

    # layer x position heatmaps
    for cname, R in results.items():
        fig, ax = plt.subplots(figsize=(7, 4.4))
        im = ax.imshow(R.T, aspect="auto", cmap="Blues", vmin=0.5, vmax=1.0, origin="lower")
        ax.set_xticks(range(len(POS_NAMES)))
        ax.set_xticklabels(POS_NAMES)
        ax.set_ylabel("layer")
        ax.set_xlabel("position")
        ax.set_title(f"{cname}: held-out AUC by layer and position")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("AUC")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"fig_e3_heatmap_{cname}.png"), dpi=200)
        plt.close(fig)

    # ---------- summary ----------
    lines = []
    for cname, R in results.items():
        for pname in ["k_last", "final"]:
            curve = R[POS_NAMES.index(pname)]
            first = next((int(L) for L in layers if np.all(curve[L:] >= 0.9)), None)
            peak = float(np.nanmax(curve))
            lines.append(f"{cname:12s} @ {pname:7s}: earliest layer with AUC>=0.9 sustained: "
                         f"{first}   peak AUC {peak:.3f}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.out_dir, "summary_e3.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
