"""
exp6_ov.py — "When" experiment 6: WHAT does L17H0 write into the residual stream?

Exp4/5 showed L17H0 moves the verdict (sufficient by patching, necessary by ablation). Two
hypotheses about its OV output at the final position:
  (i)  it transports a ready-made yes/no signal (write aligned with the unembedding yes-no
       direction; large direct logit attribution), or
  (ii) it transports a raw "k is in range" feature that downstream MLPs (L18-22) convert
       (write aligned with the edge-probe directions; small direct attribution vs the total
       causal effect measured in exp5).

Per screened prompt (same items as exp5; true / over / below_a), we capture:
  - each head-of-interest's write vector  w = W_O[:, head slice] @ (o_proj input slice at final pos)
  - the raw final residual at the final position (pre-final-norm, via hook)

and compute:
  1. direct logit attribution (frozen-denominator RMSNorm):  (w / rms(resid) * gamma) . (u_yes - u_no)
  2. cosine of w with difference-of-means probe directions built from the exp3 cache at the
     final position, residual entering layer 18 (right after L17 writes): A_verdict, B_over,
     C_below_a — plus the gamma-scaled unembedding yes-no direction
  3. one-axis test: PC1 of the writes across all prompts (variance explained; projection vs verdict)
  4. top vocab tokens promoted / suppressed by the mean (true - false) write difference

Heads: L17H0 (the mover), L18H11 (support), L18H10 (suppressor), L17H3 (null control).

Usage:
    python exp6_ov.py --out_dir results/when_exp6 --acts <path>/acts.npz
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
HEADS = [(17, 0), (18, 11), (18, 10), (17, 3)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--out_dir", default="results/when_exp6")
    p.add_argument("--acts", required=True, help="exp3 activation cache (acts.npz)")
    p.add_argument("--n_items", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--hi", type=int, default=20)
    p.add_argument("--lo", type=int, default=1)
    return p.parse_args()


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


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
    gamma = model.model.norm.weight.detach().float().numpy()
    W_U = model.lm_head.weight.detach().float().numpy()  # (V, d)
    u_yes = W_U[yes_ids].mean(0)
    u_no = W_U[no_ids].mean(0)
    u_diff = u_yes - u_no
    hook_layers = sorted({L for (L, _) in HEADS})

    # ---------- probe directions from the exp3 cache (final position) ----------
    z = np.load(args.acts)
    meta = z["meta"]  # a, b, k, true, final_logodds
    ma, mb, mk, mtrue, mlo = meta.T
    X18 = z["acts_final"][:, 18].astype(np.float32)   # residual ENTERING layer 18 (after L17)
    dirs = {}
    dirs["A_verdict"] = X18[mlo > 0].mean(0) - X18[mlo <= 0].mean(0)
    dirs["B_over"] = X18[mk > mtrue].mean(0) - X18[mk <= mtrue].mean(0)
    um = mk < mtrue
    dirs["C_below_a"] = X18[um & (mk < ma)].mean(0) - X18[um & (mk >= ma)].mean(0)
    # unembedding yes-no direction mapped back to pre-norm basis (per-dim gamma scaling)
    dirs["U_yesno"] = u_diff * gamma
    print("probe directions built from", X18.shape[0], "cached rows")

    # ---------- collect writes on the screened prompt set ----------
    cands = candidate_items(args)
    kfs = {"true": lambda a, b: a + b, "over": lambda a, b: a + b + 2,
           "below_a": lambda a, b: a - 2}
    items = []
    writes = {hd: [] for hd in HEADS}   # per prompt: (d,) write vector
    dla = {hd: [] for hd in HEADS}      # direct logit attribution on yes-no
    resid_dla, ld_list, ptypes = [], [], []
    t0 = time.time()
    for a, b in cands:
        if len(items) >= args.n_items:
            break
        trio, ok = [], True
        for pt in PROMPT_TYPES:
            k = kfs[pt](a, b)
            enc = build_inputs(tok, a, b, k)
            fin = enc["input_ids"].shape[1] - 1
            cache = {}
            handles = []
            for L in hook_layers:
                def mk(L):
                    def h(m, hargs):
                        cache[L] = hargs[0][0, fin].detach().clone()
                    return h
                handles.append(layers[L].self_attn.o_proj.register_forward_pre_hook(mk(L)))

            def h_norm(m, hargs):
                cache["resid"] = hargs[0][0, fin].detach().clone()
            handles.append(model.model.norm.register_forward_pre_hook(h_norm))
            with torch.no_grad():
                out = model(**enc)
            for h in handles:
                h.remove()
            ld = class_logodds_batch(out.logits[:, -1], yes_ids, no_ids).item()
            if (pt == "true" and ld <= 0.5) or (pt != "true" and ld >= -0.5):
                ok = False
                break
            trio.append((pt, ld, dict(cache)))
        if not ok:
            print(f"skip {a}+{b}", flush=True)
            continue
        items.append((a, b))
        for pt, ld, cache in trio:
            resid = cache["resid"].float().numpy()
            rms = float(np.sqrt((resid ** 2).mean()) + cfg.rms_norm_eps)
            resid_dla.append(float((resid / rms * gamma) @ u_diff))
            ld_list.append(ld)
            ptypes.append(pt)
            for (L, h) in HEADS:
                sl = slice(h * head_dim, (h + 1) * head_dim)
                W_O = layers[L].self_attn.o_proj.weight.detach().float().numpy()
                w = W_O[:, sl] @ cache[L][sl].float().numpy()
                writes[(L, h)].append(w)
                dla[(L, h)].append(float((w / rms * gamma) @ u_diff))
        print(f"{a}+{b} done ({(time.time()-t0)/60:.1f} min)", flush=True)

    ptypes = np.array(ptypes)
    ld_arr = np.array(ld_list)
    is_true = ptypes == "true"

    # ---------- analysis ----------
    lines = [f"items: {items}",
             f"baseline LD   true {ld_arr[is_true].mean():+.2f}   false {ld_arr[~is_true].mean():+.2f}",
             f"full-residual DLA (yes-no):  true {np.mean([d for d, t in zip(resid_dla, is_true) if t]):+.2f}"
             f"   false {np.mean([d for d, t in zip(resid_dla, is_true) if not t]):+.2f}"]
    pc_data = {}
    for (L, h) in HEADS:
        W = np.stack(writes[(L, h)])  # (P, d)
        d_arr = np.array(dla[(L, h)])
        # 1. direct logit attribution by prompt type
        lines.append(f"L{L}H{h}  DLA yes-no:  true {d_arr[is_true].mean():+.2f}  "
                     f"over {d_arr[ptypes == 'over'].mean():+.2f}  "
                     f"below_a {d_arr[ptypes == 'below_a'].mean():+.2f}   "
                     f"(true-false gap {d_arr[is_true].mean() - d_arr[~is_true].mean():+.2f})")
        # 2. cosine of the true-false write DIFFERENCE with each direction
        wdiff = W[is_true].mean(0) - W[~is_true].mean(0)
        cs = "  ".join(f"{name}={cos(wdiff, d):+.2f}" for name, d in dirs.items())
        lines.append(f"L{L}H{h}  cos(write_true - write_false, dir):  {cs}")
        # mean cosine of individual writes with the verdict direction, signed by class
        cv = np.array([cos(w, dirs["A_verdict"]) for w in W])
        lines.append(f"L{L}H{h}  cos(write, A_verdict) per class:  true {cv[is_true].mean():+.2f}  "
                     f"false {cv[~is_true].mean():+.2f}")
        # 3. one-axis test: PCA on centered writes
        Wc = W - W.mean(0)
        U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
        var = S ** 2 / (S ** 2).sum()
        proj = Wc @ Vt[0]
        sgn = np.sign(np.corrcoef(proj, is_true.astype(float))[0, 1] + 1e-12)
        r = np.corrcoef(sgn * proj, ld_arr)[0, 1]
        lines.append(f"L{L}H{h}  PC1 var explained {var[0]:.2f} (PC2 {var[1]:.2f}); "
                     f"corr(PC1 proj, logodds) r={r:+.2f}; cos(PC1, A_verdict)={cos(Vt[0]*sgn, dirs['A_verdict']):+.2f}")
        pc_data[(L, h)] = (sgn * proj, var)

    # 4. vocab readout of the mover's true-false write difference
    W = np.stack(writes[(17, 0)])
    wdiff = W[is_true].mean(0) - W[~is_true].mean(0)
    rms_t = np.linalg.norm(wdiff) / np.sqrt(len(wdiff))
    logits_w = W_U @ (wdiff / rms_t * gamma)
    top = np.argsort(logits_w)
    promo = [tok.convert_ids_to_tokens(int(i)) for i in top[::-1][:12]]
    supp = [tok.convert_ids_to_tokens(int(i)) for i in top[:12]]
    lines.append("L17H0 write (true-false) promotes: " + " ".join(promo))
    lines.append("L17H0 write (true-false) suppresses: " + " ".join(supp))

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.out_dir, "summary_e6.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    np.savez(os.path.join(args.out_dir, "ov.npz"),
             ptypes=ptypes, ld=ld_arr, items=np.array(items),
             **{f"writes_L{L}H{h}": np.stack(writes[(L, h)]) for (L, h) in HEADS},
             **{f"dla_L{L}H{h}": np.array(dla[(L, h)]) for (L, h) in HEADS},
             **{f"dir_{n}": d for n, d in dirs.items()})

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    # left: DLA per head by class
    xs = np.arange(len(HEADS))
    for pt, col, off in zip(PROMPT_TYPES, [BLUE, ORANGE, GREEN], [-0.27, 0.0, 0.27]):
        m = ptypes == pt
        vals = [np.array(dla[hd])[m].mean() for hd in HEADS]
        errs = [np.array(dla[hd])[m].std() / np.sqrt(m.sum()) for hd in HEADS]
        axes[0].bar(xs + off, vals, width=0.26, color=col, label=pt)
        axes[0].errorbar(xs + off, vals, yerr=errs, fmt="none", ecolor=INK2, linewidth=0.8)
    axes[0].axhline(0, color=INK2, linewidth=0.8)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([f"L{L}H{h}" for (L, h) in HEADS])
    axes[0].set_ylabel("direct logit attribution (yes - no)")
    axes[0].set_title("Direct path: write -> unembedding")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(True, axis="y", color="#e6e5e1", linewidth=0.6)
    # right: cosine of write difference with each direction
    dnames = list(dirs.keys())
    xs2 = np.arange(len(dnames))
    cols = [BLUE, ORANGE, RED, GREEN]
    for i, (L, h) in enumerate(HEADS):
        W = np.stack(writes[(L, h)])
        wdiff = W[is_true].mean(0) - W[~is_true].mean(0)
        axes[1].bar(xs2 + (i - 1.5) * 0.2, [cos(wdiff, dirs[n]) for n in dnames],
                    width=0.19, color=cols[i], label=f"L{L}H{h}")
    axes[1].axhline(0, color=INK2, linewidth=0.8)
    axes[1].set_xticks(xs2)
    axes[1].set_xticklabels(dnames, fontsize=9)
    axes[1].set_ylabel("cosine")
    axes[1].set_title("cos(write_true - write_false, feature direction)")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(True, axis="y", color="#e6e5e1", linewidth=0.6)
    fig.suptitle("What does each head write at the final position?")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_e6_ov.png"), dpi=200)
    plt.close(fig)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
