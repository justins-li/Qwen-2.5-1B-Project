"""
exp4_heads.py — "When" experiment 4: which attention heads move the verdict?

Follow-up to exp2 (same causal method, finer granularity). Exp2 showed the verdict information
hops from the k tokens to the final position at layers 16-18. Here we patch, from the clean run
into the corrupted run, AT THE FINAL POSITION ONLY:
  (a) each layer's attention output and MLP output separately  (component decomposition, layers 0..27)
  (b) each attention head's contribution (slice of the o_proj input), layers 13..21

and we record, from the clean run, where each head's final-position query attends (eager attention),
bucketed into: k tokens, a tokens, b tokens, '=', other.

Same items, corruptions (over: k=a+b+2; below_a: k=a-2), screening and recovery metric as exp2.

Usage:
    python exp4_heads.py --out_dir results/when_exp4
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

HEAD_LAYERS = list(range(13, 22))  # window from exp2 (16-18) plus margin on both sides


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--out_dir", default="results/when_exp4")
    p.add_argument("--n_items", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--hi", type=int, default=20)
    p.add_argument("--lo", type=int, default=1)
    return p.parse_args()


def operand_spans(toks, eq_pos):
    """Token-position runs of the two operands before '=' (first run = a, second = b)."""
    runs, cur = [], []
    for i in range(eq_pos):
        if any(ch.isdigit() for ch in toks[i]):
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    assert len(runs) >= 2, f"could not find two operand runs in {toks[:eq_pos]}"
    return runs[-2], runs[-1]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model} ({args.dtype}, {args.device}, eager attention)")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                                 attn_implementation="eager")
    model.to(args.device).eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    layers = model.model.layers
    yes_ids = single_token_ids(tok, YES_VARIANTS)
    no_ids = single_token_ids(tok, NO_VARIANTS)
    print(f"{n_layers} layers, {n_heads} heads of dim {head_dim}")

    cands = candidate_items(args)
    corruptions = {"over": lambda a, b: a + b + 2, "below_a": lambda a, b: a - 2}
    BUCKETS = ["k", "a", "b", "eq", "final", "other"]

    results = {}
    t0 = time.time()
    for ctype, kf in corruptions.items():
        rec_attn, rec_mlp, rec_heads, attn_mass, used = [], [], [], [], []
        for a, b in cands:
            if len(used) >= args.n_items:
                break
            k_clean, k_corr = a + b, kf(a, b)
            enc_c = build_inputs(tok, a, b, k_clean)
            enc_x = build_inputs(tok, a, b, k_corr)
            ids_c = enc_c["input_ids"][0].tolist()
            ids_x = enc_x["input_ids"][0].tolist()
            if len(ids_c) != len(ids_x):
                continue
            eq, k1, k2, toks_x = locate_positions(tok, ids_x, k_corr)
            T = len(ids_x)
            fin = T - 1

            # ---- clean run: cache per-layer o_proj input, attn out, mlp out at final position
            cache = {"oproj": [None] * n_layers, "attn": [None] * n_layers, "mlp": [None] * n_layers}
            handles = []
            for L in range(n_layers):
                def mk_oproj(L):
                    def h(m, hargs):
                        cache["oproj"][L] = hargs[0][0, fin].detach().clone()
                    return h

                def mk_attn(L):
                    def h(m, hargs, out):
                        o = out[0] if isinstance(out, tuple) else out
                        cache["attn"][L] = o[0, fin].detach().clone()
                    return h

                def mk_mlp(L):
                    def h(m, hargs, out):
                        cache["mlp"][L] = out[0, fin].detach().clone()
                    return h

                handles += [layers[L].self_attn.o_proj.register_forward_pre_hook(mk_oproj(L)),
                            layers[L].self_attn.register_forward_hook(mk_attn(L)),
                            layers[L].mlp.register_forward_hook(mk_mlp(L))]
            with torch.no_grad():
                out_c = model(**enc_c, output_attentions=True)
            for h in handles:
                h.remove()
            ld_clean = class_logodds_batch(out_c.logits[:, -1], yes_ids, no_ids).item()
            with torch.no_grad():
                out_x = model(**enc_x)
            ld_corr = class_logodds_batch(out_x.logits[:, -1], yes_ids, no_ids).item()
            if not (ld_clean > 0.5 and ld_corr < -0.5):
                print(f"[{ctype}] skip {a}+{b}: LD {ld_clean:+.2f} / {ld_corr:+.2f}", flush=True)
                continue
            denom = ld_clean - ld_corr

            # clean attention from the final position, bucketed, for the head-window layers
            a_span, b_span = operand_spans(tok.convert_ids_to_tokens(ids_c), eq)
            am_item = np.zeros((len(HEAD_LAYERS), n_heads, len(BUCKETS)), dtype=np.float32)
            for li, L in enumerate(HEAD_LAYERS):
                A = out_c.attentions[L][0, :, fin, :].float().numpy()  # (heads, T)
                buck = {"k": list(range(k1, k2 + 1)), "a": a_span, "b": b_span,
                        "eq": [eq], "final": [fin]}
                taken = set(sum(buck.values(), []))
                for bi, bn in enumerate(BUCKETS[:-1]):
                    am_item[li, :, bi] = A[:, buck[bn]].sum(axis=1)
                am_item[li, :, -1] = A[:, [p for p in range(T) if p not in taken]].sum(axis=1)

            batch_ids_x2 = enc_x["input_ids"].repeat(2, 1)
            batch_ids_xH = enc_x["input_ids"].repeat(n_heads, 1)

            # ---- (a) component decomposition: attn out (elem 0) and mlp out (elem 1), per layer
            ra, rm = np.zeros(n_layers, np.float32), np.zeros(n_layers, np.float32)
            for L in range(n_layers):
                def h_attn(m, hargs, out):
                    o = out[0] if isinstance(out, tuple) else out
                    o[0, fin] = cache["attn"][L]
                    return out

                def h_mlp(m, hargs, out):
                    out[1, fin] = cache["mlp"][L]
                    return out

                h1 = layers[L].self_attn.register_forward_hook(h_attn)
                h2 = layers[L].mlp.register_forward_hook(h_mlp)
                with torch.no_grad():
                    out_p = model(input_ids=batch_ids_x2)
                h1.remove(); h2.remove()
                ld = class_logodds_batch(out_p.logits[:, -1], yes_ids, no_ids).numpy()
                ra[L] = (ld[0] - ld_corr) / denom
                rm[L] = (ld[1] - ld_corr) / denom

            # ---- (b) per-head patch at the final position (slice of o_proj input)
            rh = np.zeros((len(HEAD_LAYERS), n_heads), np.float32)
            for li, L in enumerate(HEAD_LAYERS):
                def h_head(m, hargs):
                    x = hargs[0]
                    for i in range(n_heads):
                        x[i, fin, i * head_dim:(i + 1) * head_dim] = \
                            cache["oproj"][L][i * head_dim:(i + 1) * head_dim]
                    return None

                hd = layers[L].self_attn.o_proj.register_forward_pre_hook(h_head)
                with torch.no_grad():
                    out_p = model(input_ids=batch_ids_xH)
                hd.remove()
                ld = class_logodds_batch(out_p.logits[:, -1], yes_ids, no_ids).numpy()
                rh[li] = (ld - ld_corr) / denom

            rec_attn.append(ra); rec_mlp.append(rm); rec_heads.append(rh)
            attn_mass.append(am_item); used.append((a, b))
            print(f"[{ctype}] {a}+{b}: LD {ld_clean:+.2f}/{ld_corr:+.2f}  "
                  f"best head rec {rh.max():.2f}  ({(time.time()-t0)/60:.1f} min)", flush=True)

        results[ctype] = dict(rec_attn=np.stack(rec_attn), rec_mlp=np.stack(rec_mlp),
                              rec_heads=np.stack(rec_heads), attn_mass=np.stack(attn_mass),
                              used=used)
        np.savez(os.path.join(args.out_dir, f"heads_{ctype}.npz"),
                 rec_attn=np.stack(rec_attn), rec_mlp=np.stack(rec_mlp),
                 rec_heads=np.stack(rec_heads), attn_mass=np.stack(attn_mass),
                 used=np.array(used), head_layers=np.array(HEAD_LAYERS),
                 buckets=np.array(BUCKETS))

    # ================= figures =================
    # components
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (ctype, r) in zip(axes, results.items()):
        ax.plot(r["rec_attn"].mean(0), color=BLUE, linewidth=2, marker="o", markersize=3.5,
                label="attention out @ final pos")
        ax.plot(r["rec_mlp"].mean(0), color=ORANGE, linewidth=2, marker="s", markersize=3.5,
                label="MLP out @ final pos")
        ax.axhline(0, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_xlabel("layer")
        ax.set_title(ctype, fontsize=11)
        ax.grid(True, color="#e6e5e1", linewidth=0.6)
    axes[0].set_ylabel("recovery")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Which component writes the verdict into the final position?")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_e4_components.png"), dpi=200)
    plt.close(fig)

    # heads heatmap
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    vmax = max(np.abs(r["rec_heads"].mean(0)).max() for r in results.values())
    for ax, (ctype, r) in zip(axes, results.items()):
        H = r["rec_heads"].mean(0)
        im = ax.imshow(H, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_yticks(range(len(HEAD_LAYERS)))
        ax.set_yticklabels(HEAD_LAYERS)
        ax.set_xticks(range(H.shape[1]))
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title(f"{ctype}: per-head recovery @ final position")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("recovery")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_e4_heads.png"), dpi=200)
    plt.close(fig)

    # attention of top heads
    mean_rec = np.mean([r["rec_heads"].mean(0) for r in results.values()], axis=0)
    flat = [(mean_rec[li, h], HEAD_LAYERS[li], h) for li in range(len(HEAD_LAYERS))
            for h in range(n_heads)]
    top = sorted(flat, reverse=True)[:5]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.15
    xs = np.arange(len(BUCKETS))
    colors = [BLUE, ORANGE, GREEN, RED, INK2]
    for i, (rec, L, h) in enumerate(top):
        li = HEAD_LAYERS.index(L)
        mass = np.mean([r["attn_mass"][:, li, h, :].mean(0) for r in results.values()], axis=0)
        ax.bar(xs + (i - 2) * width, mass, width, color=colors[i],
               label=f"L{L}H{h} (rec {rec:.2f})")
    ax.set_xticks(xs)
    ax.set_xticklabels(BUCKETS)
    ax.set_ylabel("attention mass from final position")
    ax.set_title("Where do the top verdict-moving heads look? (clean runs, mean over items)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", color="#e6e5e1", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_e4_attention.png"), dpi=200)
    plt.close(fig)

    # ================= summary =================
    lines = []
    for ctype, r in results.items():
        H = r["rec_heads"].mean(0)
        order = np.dstack(np.unravel_index(np.argsort(H, axis=None)[::-1], H.shape))[0][:6]
        tops = ", ".join(f"L{HEAD_LAYERS[li]}H{h}={H[li, h]:.2f}" for li, h in order)
        pa = r["rec_attn"].mean(0)
        pm = r["rec_mlp"].mean(0)
        lines += [f"[{ctype}] items: {r['used']}",
                  f"[{ctype}] top heads (recovery): {tops}",
                  f"[{ctype}] attn-out recovery by layer 14-21: "
                  + " ".join(f"{L}:{pa[L]:.2f}" for L in range(14, 22)),
                  f"[{ctype}] mlp-out recovery by layer 14-21:  "
                  + " ".join(f"{L}:{pm[L]:.2f}" for L in range(14, 22)),
                  f"[{ctype}] max single-head recovery {H.max():.2f} at "
                  f"L{HEAD_LAYERS[np.unravel_index(H.argmax(), H.shape)[0]]}"
                  f"H{np.unravel_index(H.argmax(), H.shape)[1]}"]
    for rec, L, h in top:
        li = HEAD_LAYERS.index(L)
        mass = np.mean([r["attn_mass"][:, li, h, :].mean(0) for r in results.values()], axis=0)
        lines.append(f"attention of L{L}H{h} (rec {rec:.2f}): "
                     + "  ".join(f"{bn}={m:.2f}" for bn, m in zip(BUCKETS, mass)))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.out_dir, "summary_e4.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
