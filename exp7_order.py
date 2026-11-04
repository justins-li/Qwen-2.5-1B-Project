"""
exp7_order.py — operand-order control: "Is {b} + {a} = {k} correct?"

Step 1 found the 1.5B accepts k iff  first_operand <= k <= a+b  (sharp lower edge at the FIRST
printed operand — but in that data the first operand was always a, so "edge at a" and "edge at
whatever is printed first" were confounded). Swapping the printed order disambiguates:

  Phase A (behaviour): rerun the sweep with the operands swapped in the prompt. For undershoot
  rows, align P(yes) on k - first_printed and k - second_printed. If the step follows the first
  printed operand, the rule is positional ("the number right after 'Is'"); if it stays at the
  same operand identity it would be magnitude-driven (check vs min/max operand too).

  Phase B (heads): exp4-style per-head patching at the final position (layers 16-19) on the
  swapped prompts, corruptions over (sum+2) and below_first (first printed - 2). Does L17H0
  still carry the verdict with the operands swapped?

Usage:
    python exp7_order.py --out_dir results/when_exp7
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rhs_sweep import (SYSTEM, USER_TEMPLATE, YES_VARIANTS, NO_VARIANTS,  # noqa: F401
                       build_inputs, locate_positions, single_token_ids)
from exp2_patching import candidate_items, class_logodds_batch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#2e8b57", "#c0392b"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "font.size": 11,
})

HEAD_LAYERS = [16, 17, 18, 19]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--out_dir", default="results/when_exp7")
    p.add_argument("--orig_csv", default="results/sweep_1p5b.csv")
    p.add_argument("--n_pairs", type=int, default=40)
    p.add_argument("--n_items", type=int, default=8)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--hi", type=int, default=20)
    p.add_argument("--lo", type=int, default=1)
    return p.parse_args()


def sweep_pairs(args):
    import random
    random.seed(args.seed)
    all_pairs = [(a, b) for a in range(args.lo, args.hi + 1) for b in range(args.lo, args.hi + 1)]
    return random.sample(all_pairs, args.n_pairs)


def aligned_step(df, op_col, lo=-6, hi=6, min_n=5):
    """Mean p_yes among undershoots by k - df[op_col]; columns with >= min_n rows."""
    um = df[df.k < df.true]
    out = {}
    for d in range(lo, hi + 1):
        rows = um[um.k - um[op_col] == d]
        if len(rows) >= min_n:
            out[d] = (rows.p_yes.mean(), len(rows))
    return out


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
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    layers = model.model.layers
    yes_ids = single_token_ids(tok, YES_VARIANTS)
    no_ids = single_token_ids(tok, NO_VARIANTS)

    # ================= Phase A: swapped-order behavioural sweep =================
    rows = []
    t0 = time.time()
    pairs = sweep_pairs(args)
    total = len(pairs) * (2 * args.window + 1)
    n = 0
    for a, b in pairs:
        true = a + b
        for k in range(true - args.window, true + args.window + 1):
            n += 1
            if k < 0:
                continue
            enc = build_inputs(tok, b, a, k)  # SWAPPED: b printed first
            with torch.no_grad():
                out = model(**enc)
            lo = class_logodds_batch(out.logits[:, -1], yes_ids, no_ids).item()
            rows.append(dict(a=a, b=b, k=k, true=true, first=b, second=a,
                             logodds=lo, p_yes=float(1 / (1 + np.exp(-lo)))))
            if n % 100 == 0 or n == total:
                print(f"[sweep {n}/{total}] {(time.time()-t0)/60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "sweep_swapped.csv"), index=False)

    # analysis: edge alignment, pairs with |a-b| >= 3 only (disambiguating pairs)
    dd = df[(df.a - df.b).abs() >= 3].copy()
    step_first = aligned_step(dd, "first")
    step_second = aligned_step(dd, "second")
    dd["mn"] = dd[["a", "b"]].min(axis=1)
    dd["mx"] = dd[["a", "b"]].max(axis=1)
    step_min = aligned_step(dd, "mn")
    step_max = aligned_step(dd, "mx")

    def edge_sharpness(step):
        below = [v for d, (v, _) in step.items() if -4 <= d <= -1]
        at = [v for d, (v, _) in step.items() if 0 <= d <= 3]
        if not below or not at:
            return np.nan, np.nan, np.nan
        return float(np.mean(below)), float(np.mean(at)), float(np.mean(at) - np.mean(below))

    lines = ["=== Phase A: swapped-order sweep (b printed first), |a-b|>=3 undershoots ==="]
    for name, st in [("first printed (b)", step_first), ("second printed (a)", step_second),
                     ("min operand", step_min), ("max operand", step_max)]:
        lo_m, at_m, gap = edge_sharpness(st)
        lines.append(f"edge on k - {name:18s}: P(yes) below {lo_m:.2f}  at/above {at_m:.2f}  "
                     f"step {gap:+.2f}")

    # comparison with the original sweep (a printed first)
    try:
        od = pd.read_csv(args.orig_csv)
        od = od[(od.a - od.b).abs() >= 3]
        od["first"] = od.a
        od["second"] = od.b
        o_first = aligned_step(od, "first")
        o_second = aligned_step(od, "second")
        lo_m, at_m, gap = edge_sharpness(o_first)
        lines.append(f"[original a-first] edge on k - first printed (a): below {lo_m:.2f}  "
                     f"at/above {at_m:.2f}  step {gap:+.2f}")
        lo_m, at_m, gap = edge_sharpness(o_second)
        lines.append(f"[original a-first] edge on k - second printed (b): below {lo_m:.2f}  "
                     f"at/above {at_m:.2f}  step {gap:+.2f}")
    except FileNotFoundError:
        o_first = o_second = None
        lines.append(f"(original csv {args.orig_csv} not found; skipping comparison)")

    # figure: aligned curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, steps, title in [
            (axes[0], [("k - first printed", step_first, BLUE), ("k - second printed", step_second, ORANGE)],
             "swapped prompt: b + a = k"),
            (axes[1], ([("k - first printed", o_first, BLUE), ("k - second printed", o_second, ORANGE)]
                       if o_first else []), "original prompt: a + b = k")]:
        for lab, st, col in steps:
            if not st:
                continue
            ds = sorted(st)
            ax.plot(ds, [st[d][0] for d in ds], color=col, marker="o", markersize=4,
                    linewidth=2, label=lab)
        ax.axvline(0, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_xlabel("k - operand")
        ax.set_title(title, fontsize=11)
        ax.grid(True, color="#e6e5e1", linewidth=0.6)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("P(yes), undershoots, |a-b|>=3")
    fig.suptitle("Does the lower edge follow the printed position?")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_e7_edges.png"), dpi=200)
    plt.close(fig)
    print("\n".join(lines), flush=True)

    # ================= Phase B: head patching on swapped prompts =================
    # first printed operand is b -> need b >= 13 two-digit; reuse candidate_items and swap roles.
    cands = [(b, a) for (a, b) in candidate_items(args)]  # (first, second) with first >= 13
    corruptions = {"over": lambda f, s: f + s + 2, "below_first": lambda f, s: f - 2}
    head_results = {}
    for ctype, kf in corruptions.items():
        rec_heads, used = [], []
        for first, second in cands:
            if len(used) >= args.n_items:
                break
            k_clean, k_corr = first + second, kf(first, second)
            enc_c = build_inputs(tok, first, second, k_clean)
            enc_x = build_inputs(tok, first, second, k_corr)
            if enc_c["input_ids"].shape[1] != enc_x["input_ids"].shape[1]:
                continue
            fin = enc_x["input_ids"].shape[1] - 1
            cache = {}
            handles = []
            for L in HEAD_LAYERS:
                def mk(L):
                    def h(m, hargs):
                        cache[L] = hargs[0][0, fin].detach().clone()
                    return h
                handles.append(layers[L].self_attn.o_proj.register_forward_pre_hook(mk(L)))
            with torch.no_grad():
                out_c = model(**enc_c)
            for h in handles:
                h.remove()
            ld_clean = class_logodds_batch(out_c.logits[:, -1], yes_ids, no_ids).item()
            with torch.no_grad():
                out_x = model(**enc_x)
            ld_corr = class_logodds_batch(out_x.logits[:, -1], yes_ids, no_ids).item()
            if not (ld_clean > 0.5 and ld_corr < -0.5):
                print(f"[{ctype}] skip {first}+{second}: LD {ld_clean:+.2f} / {ld_corr:+.2f}",
                      flush=True)
                continue
            denom = ld_clean - ld_corr
            batch_ids = enc_x["input_ids"].repeat(n_heads, 1)
            rh = np.zeros((len(HEAD_LAYERS), n_heads), np.float32)
            for li, L in enumerate(HEAD_LAYERS):
                def h_head(m, hargs):
                    x = hargs[0]
                    for i in range(n_heads):
                        x[i, fin, i * head_dim:(i + 1) * head_dim] = \
                            cache[L][i * head_dim:(i + 1) * head_dim]
                    return None
                hd = layers[L].self_attn.o_proj.register_forward_pre_hook(h_head)
                with torch.no_grad():
                    out_p = model(input_ids=batch_ids)
                hd.remove()
                ld = class_logodds_batch(out_p.logits[:, -1], yes_ids, no_ids).numpy()
                rh[li] = (ld - ld_corr) / denom
            rec_heads.append(rh)
            used.append((first, second))
            print(f"[{ctype}] {first}+{second}: LD {ld_clean:+.2f}/{ld_corr:+.2f}  "
                  f"best rec {rh.max():.2f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        H = np.stack(rec_heads).mean(0)
        head_results[ctype] = (H, used)
        np.savez(os.path.join(args.out_dir, f"heads_swapped_{ctype}.npz"),
                 rec_heads=np.stack(rec_heads), used=np.array(used),
                 head_layers=np.array(HEAD_LAYERS))

    lines.append("=== Phase B: head patching, swapped prompts ===")
    for ctype, (H, used) in head_results.items():
        order = np.dstack(np.unravel_index(np.argsort(H, axis=None)[::-1], H.shape))[0][:5]
        tops = ", ".join(f"L{HEAD_LAYERS[li]}H{h}={H[li, h]:.2f}" for li, h in order)
        lines += [f"[{ctype}] items: {used}",
                  f"[{ctype}] top heads: {tops}",
                  f"[{ctype}] L17H0 recovery: {H[HEAD_LAYERS.index(17), 0]:.2f}"]

    # heads figure
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    vmax = max(np.abs(H).max() for H, _ in head_results.values())
    for ax, (ctype, (H, _)) in zip(axes, head_results.items()):
        im = ax.imshow(H, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_yticks(range(len(HEAD_LAYERS)))
        ax.set_yticklabels(HEAD_LAYERS)
        ax.set_xticks(range(n_heads))
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title(f"{ctype} (swapped prompt)")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle("Per-head recovery @ final position, b + a = k")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_e7_heads.png"), dpi=200)
    plt.close(fig)

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.out_dir, "summary_e7.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("done.", flush=True)


if __name__ == "__main__":
    main()
