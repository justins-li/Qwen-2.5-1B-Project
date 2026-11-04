"""
exp5_ablate.py — "When" experiment 5: is L17H0 NECESSARY, not just sufficient?

Exp4 (patching) showed L17H0's clean output is sufficient to restore ~0.4 of the verdict in a
corrupted run. The converse test: in CLEAN runs, knock out one head's contribution at the final
position and see how much the verdict degrades.

Two ablation types per head:
  zero  — zero the head's slice of the o_proj input at the final position
  mean  — replace it with the mean slice over all prompts (true + both corruptions),
          which removes item-specific information but keeps the average write

Conditions: all 12 heads of L17 (within-layer specificity), the exp4 support heads
(L18H11, L16H7, L18H7, L19H11), the negative head L18H10, and two combos
(L17H0+L18H11; L17H0+L18H10 to see whether the backup compensates).

Prompts per item: true (k=a+b), over (k=a+b+2), below_a (k=a-2). Screened like exp2/exp4.
Ablating a verdict-carrying head should push true-equation LD down toward the false level
(verdict flips); false-equation LDs are the control (should move much less).

Usage:
    python exp5_ablate.py --out_dir results/when_exp5
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

PROMPT_TYPES = ["true", "over", "below_a"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--out_dir", default="results/when_exp5")
    p.add_argument("--n_items", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--hi", type=int, default=20)
    p.add_argument("--lo", type=int, default=1)
    return p.parse_args()


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

    # ---- conditions: list of (name, [(L, h), ...])
    conditions = [(f"L17H{h}", [(17, h)]) for h in range(n_heads)]
    conditions += [("L18H11", [(18, 11)]), ("L16H7", [(16, 7)]), ("L18H7", [(18, 7)]),
                   ("L19H11", [(19, 11)]), ("L18H10", [(18, 10)]),
                   ("L17H0+L18H11", [(17, 0), (18, 11)]),
                   ("L17H0+L18H10", [(17, 0), (18, 10)])]
    abl_layers = sorted({L for _, hs in conditions for (L, _) in hs})

    # ---- collect screened items and their three prompts, with baseline LDs
    cands = candidate_items(args)
    kfs = {"true": lambda a, b: a + b, "over": lambda a, b: a + b + 2,
           "below_a": lambda a, b: a - 2}
    items, prompts = [], []  # prompts: dict(item_idx, ptype, enc, fin, ld_base, oproj{L: vec})
    t0 = time.time()
    for a, b in cands:
        if len(items) >= args.n_items:
            break
        trio = []
        ok = True
        for pt in PROMPT_TYPES:
            k = kfs[pt](a, b)
            enc = build_inputs(tok, a, b, k)
            fin = enc["input_ids"].shape[1] - 1
            cache = {}
            handles = []
            for L in abl_layers:
                def mk(L):
                    def h(m, hargs):
                        cache[L] = hargs[0][0, fin].detach().clone()
                    return h
                handles.append(layers[L].self_attn.o_proj.register_forward_pre_hook(mk(L)))
            with torch.no_grad():
                out = model(**enc)
            for h in handles:
                h.remove()
            ld = class_logodds_batch(out.logits[:, -1], yes_ids, no_ids).item()
            if (pt == "true" and ld <= 0.5) or (pt != "true" and ld >= -0.5):
                ok = False
                break
            trio.append(dict(item=(a, b), ptype=pt, enc=enc, fin=fin, ld_base=ld, oproj=cache))
        if not ok:
            print(f"skip {a}+{b}", flush=True)
            continue
        items.append((a, b))
        prompts += trio
        print(f"{a}+{b}: LD true {trio[0]['ld_base']:+.2f}  over {trio[1]['ld_base']:+.2f}  "
              f"below_a {trio[2]['ld_base']:+.2f}", flush=True)
    print(f"{len(items)} items screened in ({(time.time()-t0)/60:.1f} min)", flush=True)

    # mean head slice over ALL prompts, per layer (for mean-ablation)
    mean_oproj = {L: torch.stack([p["oproj"][L] for p in prompts]).mean(0) for L in abl_layers}

    # ---- run ablations
    n_cond = len(conditions)
    # ld[abl_type, cond, prompt]
    LD = np.zeros((2, n_cond, len(prompts)), np.float32)
    for pi, p in enumerate(prompts):
        fin = p["fin"]
        for ai, abl in enumerate(["zero", "mean"]):
            for ci, (name, hs) in enumerate(conditions):
                handles = []
                for (L, h) in hs:
                    def mk(L, h):
                        sl = slice(h * head_dim, (h + 1) * head_dim)
                        repl = (torch.zeros(head_dim, dtype=dtype) if abl == "zero"
                                else mean_oproj[L][sl])
                        def hook(m, hargs):
                            hargs[0][:, fin, sl] = repl
                            return None
                        return hook
                    handles.append(
                        layers[L].self_attn.o_proj.register_forward_pre_hook(mk(L, h)))
                with torch.no_grad():
                    out = model(**p["enc"])
                for hd in handles:
                    hd.remove()
                LD[ai, ci, pi] = class_logodds_batch(out.logits[:, -1], yes_ids, no_ids).item()
        print(f"prompt {pi+1}/{len(prompts)} ({p['item']}, {p['ptype']}) done  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    ld_base = np.array([p["ld_base"] for p in prompts], np.float32)
    ptype = np.array([p["ptype"] for p in prompts])
    np.savez(os.path.join(args.out_dir, "ablate.npz"),
             LD=LD, ld_base=ld_base, ptype=ptype,
             cond_names=np.array([c[0] for c in conditions]),
             items=np.array(items))

    # ---- summary
    lines = [f"items: {items}",
             "baseline LD   true {:+.2f}   over {:+.2f}   below_a {:+.2f}".format(
                 *[ld_base[ptype == pt].mean() for pt in PROMPT_TYPES])]
    for ai, abl in enumerate(["zero", "mean"]):
        lines.append(f"--- {abl} ablation (delta LD vs baseline; true col also: verdict flips) ---")
        for ci, (name, _) in enumerate(conditions):
            d = LD[ai, ci] - ld_base
            dt = d[ptype == "true"]
            flips = int((LD[ai, ci][ptype == "true"] < 0).sum())
            lines.append(f"{name:14s} true {dt.mean():+.2f} ({flips}/{len(dt)} flip)   "
                         f"over {d[ptype == 'over'].mean():+.2f}   "
                         f"below_a {d[ptype == 'below_a'].mean():+.2f}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.out_dir, "summary_e5.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    # ---- figure: delta LD on true prompts, per condition
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    names = [c[0] for c in conditions]
    xs = np.arange(n_cond)
    for ax, (ai, abl) in zip(axes, enumerate(["zero", "mean"])):
        for pt, col in zip(PROMPT_TYPES, [BLUE, ORANGE, GREEN]):
            m = ptype == pt
            d = (LD[ai][:, m] - ld_base[m])
            ax.bar(xs + {"true": -0.27, "over": 0.0, "below_a": 0.27}[pt], d.mean(1),
                   width=0.26, color=col, label=pt)
            ax.errorbar(xs + {"true": -0.27, "over": 0.0, "below_a": 0.27}[pt], d.mean(1),
                        yerr=d.std(1) / np.sqrt(m.sum()), fmt="none", ecolor=INK2, linewidth=0.8)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax.set_title(f"{abl} ablation @ final position", fontsize=11)
        ax.grid(True, axis="y", color="#e6e5e1", linewidth=0.6)
    axes[0].set_ylabel("delta log-odds vs clean")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Ablating single heads in CLEAN runs (necessity test)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_e5_ablation.png"), dpi=200)
    plt.close(fig)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
