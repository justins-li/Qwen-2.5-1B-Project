"""
exp1_logit_lens.py — "When" experiment 1: logit lens.

At every layer ℓ, take the residual stream, apply the model's own final RMSNorm + unembedding,
and read logit(yes) − logit(no) as if the model stopped at layer ℓ. Observational: shows at which
layer the verdict appears *in the unembedding basis* (a readout, not a causal claim).

Main run: the same 820-row grid as rhs_sweep.py (seed 2026), lens at the FINAL prompt position.
Secondary: 8 pairs x {true, overshoot +2, undershoot below a}, lens at every position from '='
onward -> layer x position heatmap data.

Usage:
    python exp1_logit_lens.py --out_dir results/when_exp1
Writes lens_final_position.csv, lens_layer_position.npz, and figures.
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--out_dir", default="results/when_exp1")
    p.add_argument("--n_pairs", type=int, default=40)
    p.add_argument("--lo", type=int, default=1)
    p.add_argument("--hi", type=int, default=20)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def sweep_pairs(args):
    """Exactly the same item sample as rhs_sweep.py."""
    import random
    random.seed(args.seed)
    all_pairs = [(a, b) for a in range(args.lo, args.hi + 1) for b in range(args.lo, args.hi + 1)]
    return random.sample(all_pairs, args.n_pairs)


def lens_logodds(hidden, norm, lm_head, yes_ids, no_ids, already_normed=False):
    """hidden: (d,) residual-stream vector -> yes/no logodds through final norm + unembedding.

    In this transformers version hidden_states[-1] already has the final RMSNorm applied
    (verified: lm_head(hidden_states[-1]) == logits exactly), so the top entry skips the norm.
    """
    with torch.no_grad():
        h = hidden if already_normed else norm(hidden.unsqueeze(0)).squeeze(0)
        logits = lm_head(h).float()
        ly = torch.logsumexp(logits[yes_ids], dim=0).item()
        ln = torch.logsumexp(logits[no_ids], dim=0).item()
    return ly - ln


def condition(a, b, k):
    true = a + b
    if k == true:
        return "true"
    if k > true:
        return "over"
    return "under_below_a" if k < a else "under_in_range"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model} ({args.dtype}, {args.device})")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device).eval()
    norm, lm_head = model.model.norm, model.lm_head
    n_layers = model.config.num_hidden_layers
    yes_ids = single_token_ids(tok, YES_VARIANTS)
    no_ids = single_token_ids(tok, NO_VARIANTS)

    pairs = sweep_pairs(args)

    # ---------- main run: lens at final position, full grid ----------
    csv_path = os.path.join(args.out_dir, "lens_final_position.csv")
    rows, t0 = [], time.time()
    n = 0
    total = len(pairs) * (2 * args.window + 1)
    for a, b in pairs:
        true = a + b
        for k in range(true - args.window, true + args.window + 1):
            n += 1
            if k < 0:
                continue
            enc = build_inputs(tok, a, b, k)
            enc = {kk: v.to(args.device) for kk, v in enc.items()}
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states  # tuple len n_layers+1; hs[0]=embeddings, hs[L]=resid after layer L
            per_layer = [lens_logodds(hs[L][0, -1], norm, lm_head, yes_ids, no_ids,
                                      already_normed=(L == n_layers))
                         for L in range(n_layers + 1)]
            # validation: lens at the top layer must equal the real logits
            real = out.logits[0, -1].float()
            real_lo = (torch.logsumexp(real[yes_ids], 0) - torch.logsumexp(real[no_ids], 0)).item()
            assert abs(per_layer[-1] - real_lo) < 1e-3, (per_layer[-1], real_lo)
            rows.append(dict(a=a, b=b, k=k, offset=k - true, cond=condition(a, b, k),
                             final_logodds=round(real_lo, 4),
                             **{f"L{L}": round(v, 4) for L, v in enumerate(per_layer)}))
            if n % 50 == 0 or n == total:
                print(f"[{n}/{total}] {(time.time()-t0)/60:.1f} min", flush=True)
    import csv as _csv
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"final-position lens -> {csv_path}")

    # ---------- secondary run: layer x position heatmap on a subset ----------
    # 8 pairs with two-digit a so that true, +2 and a-2 all keep token length (comparability only;
    # the lens itself doesn't need it, but it keeps position indices aligned across conditions)
    big_a = [(a, b) for (a, b) in pairs if a >= 13][:8]
    if len(big_a) < 8:  # tiny test runs: top up deterministically from the full population
        import random as _r
        _r.seed(args.seed + 1)
        pool = [(a, b) for a in range(13, args.hi + 1) for b in range(args.lo, args.hi + 1)
                if (a, b) not in big_a]
        big_a += _r.sample(pool, 8 - len(big_a))
    conds = {"true": lambda a, b: a + b, "over": lambda a, b: a + b + 2, "below_a": lambda a, b: a - 2}
    maps = {c: [] for c in conds}
    rel_positions = None
    for a, b in big_a:
        for cname, kf in conds.items():
            k = kf(a, b)
            enc = build_inputs(tok, a, b, k)
            ids = enc["input_ids"][0].tolist()
            eq_pos, k_pos, k_last, _ = locate_positions(tok, ids, k)
            enc = {kk: v.to(args.device) for kk, v in enc.items()}
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states
            T = len(ids)
            pos_list = list(range(eq_pos, T))
            if rel_positions is None:
                rel_positions = [p - (k_last + 1) for p in pos_list]  # 0 = first token after k
            m = np.zeros((n_layers + 1, len(pos_list)))
            for L in range(n_layers + 1):
                for j, p in enumerate(pos_list):
                    m[L, j] = lens_logodds(hs[L][0, p], norm, lm_head, yes_ids, no_ids,
                                           already_normed=(L == n_layers))
            maps[cname].append(m)
        print(f"heatmap pair {a}+{b} done", flush=True)
    np.savez(os.path.join(args.out_dir, "lens_layer_position.npz"),
             rel_positions=np.array(rel_positions),
             **{c: np.stack(v) for c, v in maps.items()})

    # ---------- figures ----------
    import pandas as pd
    df = pd.DataFrame(rows)
    Lcols = [f"L{L}" for L in range(n_layers + 1)]
    layers = np.arange(n_layers + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for cname, color in [("true", GREEN), ("under_in_range", BLUE),
                         ("under_below_a", ORANGE), ("over", RED)]:
        g = df[df.cond == cname]
        tr = g[Lcols].values
        mu = tr.mean(0)
        se = tr.std(0) / np.sqrt(len(tr))
        ax.plot(layers, mu, color=color, linewidth=2, label=f"{cname} (n={len(g)})")
        ax.fill_between(layers, mu - 2 * se, mu + 2 * se, color=color, alpha=0.18, linewidth=0)
    ax.axhline(0, color=INK2, linewidth=0.8, linestyle=":")
    ax.set_xlabel("layer (lens readout of residual stream)")
    ax.set_ylabel("logit(yes) − logit(no)")
    ax.set_title("Logit lens at the final position: when does the verdict appear?")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_e1_layer_trajectories.png"), dpi=200)
    plt.close(fig)

    # decision layer: sign match with final verdict + rank correlation with final logodds
    final = df["final_logodds"].values
    match, rho = [], []
    for c in Lcols:
        v = df[c].values
        match.append(((v > 0) == (final > 0)).mean())
        r1 = pd.Series(v).rank().values
        r2 = pd.Series(final).rank().values
        rho.append(np.corrcoef(r1, r2)[0, 1])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(layers, match, color=BLUE, linewidth=2, marker="o", markersize=4, label="sign agreement with final verdict")
    ax.plot(layers, rho, color=ORANGE, linewidth=2, marker="s", markersize=4, label="Spearman ρ with final logodds")
    ax.axhline(0.9, color=INK2, linewidth=0.8, linestyle="--")
    dec = next((int(L) for L in layers if all(m >= 0.9 for m in match[L:])), None)
    if dec is not None:
        ax.axvline(dec, color=GREEN, linewidth=1.2, label=f"decision layer ≈ {dec} (≥90% from here on)")
    ax.set_xlabel("layer")
    ax.set_ylabel("agreement")
    ax.set_ylim(-0.1, 1.02)
    ax.set_title("Layer at which the lens readout matches the final verdict")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_e1_decision_layer.png"), dpi=200)
    plt.close(fig)

    # layer x position heatmap: difference over(+2) − true, and below_a − true
    rel = np.array(rel_positions)
    for cname, ref in [("over", "true"), ("below_a", "true")]:
        d = np.stack(maps[cname]).mean(0) - np.stack(maps[ref]).mean(0)
        fig, ax = plt.subplots(figsize=(8, 4.6))
        vmax = np.abs(d).max()
        im = ax.imshow(d, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower",
                       extent=[rel.min() - 0.5, rel.max() + 0.5, -0.5, n_layers + 0.5])
        ax.axvline(-0.5, color=INK, linewidth=1)  # boundary: left of 0 = k tokens and before
        ax.set_xlabel("position relative to end of k  (0 = first token after k)")
        ax.set_ylabel("layer")
        ax.set_title(f"Lens logodds difference: {cname} − {ref}  (mean over {len(big_a)} items)")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("Δ logit(yes)−logit(no)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"fig_e1_heatmap_{cname}.png"), dpi=200)
        plt.close(fig)

    # summary
    out = [f"decision layer (sign agreement ≥0.9 sustained): {dec}",
           "sign agreement by layer: " + " ".join(f"{L}:{m:.2f}" for L, m in zip(layers, match)),
           "spearman by layer:       " + " ".join(f"{L}:{r:.2f}" for L, r in zip(layers, rho))]
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(args.out_dir, "summary_e1.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
