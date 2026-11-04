"""
exp2_patching.py — "When" experiment 2: activation patching (causal tracing).

For matched prompt pairs that differ ONLY in the k tokens (same token count), run the corrupted
prompt (wrong k) and restore the residual stream at one (layer, position) cell from the clean run
(true k). Recovery of the clean verdict localises where, and at what depth, the k-information that
drives the verdict sits.

Two corruption types, patched separately:
  over:    k = a+b+2   (crosses the upper edge of the accepted range)
  below_a: k = a-2     (crosses the lower edge)
Items have a >= 13 so clean and corrupted k are both two digits (identical tokenisation).

Metric: normalised logit-difference recovery
  R = (LD_patched - LD_corrupted) / (LD_clean - LD_corrupted),  LD = logit(yes) - logit(no)
0 = patch does nothing, 1 = patch fully restores the clean verdict.

Patched cells: layer L in 0..27 (residual stream ENTERING layer L) x position p from the first k
token to the last prompt token, plus a combined "both k tokens" cell. Positions before k are
identical between clean and corrupted runs by causality, so they are skipped.

Usage:
    python exp2_patching.py --out_dir results/when_exp2
Writes recovery_<type>.npz and figures.
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
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#2e8b57"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "font.size": 11,
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--out_dir", default="results/when_exp2")
    p.add_argument("--n_items", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--hi", type=int, default=20)
    p.add_argument("--lo", type=int, default=1)
    return p.parse_args()


def candidate_items(args):
    """Sweep items with a >= 13 first (two-digit a), then more drawn deterministically.

    Candidates are screened in main(): an item is usable for a corruption type only when the model
    actually behaves differently on the pair (LD_clean > 0 > LD_corrupted) — patching needs a
    behavioural contrast to trace, and the 1.5B rejects some true equations outright.
    """
    import random
    random.seed(args.seed)
    all_pairs = [(a, b) for a in range(args.lo, args.hi + 1) for b in range(args.lo, args.hi + 1)]
    pairs = random.sample(all_pairs, 40)
    cands = [(a, b) for (a, b) in pairs if a >= 13]
    random.seed(args.seed + 1)
    pool = [(a, b) for a in range(13, args.hi + 1) for b in range(args.lo, args.hi + 1)
            if (a, b) not in cands]
    cands += random.sample(pool, len(pool))
    return cands


def class_logodds_batch(logits_last, yes_ids, no_ids):
    """logits_last: (B, vocab) -> (B,) yes-no logodds."""
    ly = torch.logsumexp(logits_last[:, yes_ids].float(), dim=1)
    ln = torch.logsumexp(logits_last[:, no_ids].float(), dim=1)
    return (ly - ln)


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
    n_layers = model.config.num_hidden_layers
    layers = model.model.layers
    yes_ids = single_token_ids(tok, YES_VARIANTS)
    no_ids = single_token_ids(tok, NO_VARIANTS)

    cands = candidate_items(args)
    corruptions = {"over": lambda a, b: a + b + 2, "below_a": lambda a, b: a - 2}

    results = {}  # type -> dict with recovery (items, layers, cells), labels, LDs
    t0 = time.time()
    for ctype, kf in corruptions.items():
        rec_all, ld_clean_all, ld_corr_all, used = [], [], [], []
        cell_labels = None
        for a, b in cands:
            if len(rec_all) >= args.n_items:
                break
            k_clean, k_corr = a + b, kf(a, b)
            enc_c = build_inputs(tok, a, b, k_clean)
            enc_x = build_inputs(tok, a, b, k_corr)
            ids_c = enc_c["input_ids"][0].tolist()
            ids_x = enc_x["input_ids"][0].tolist()
            assert len(ids_c) == len(ids_x), f"token length mismatch for {a}+{b}: {k_clean} vs {k_corr}"
            eq, k1, k2, toks_x = locate_positions(tok, ids_x, k_corr)
            eq_c, k1_c, k2_c, _ = locate_positions(tok, ids_c, k_clean)
            assert (k1, k2) == (k1_c, k2_c), "k token span differs between clean and corrupted"
            T = len(ids_x)
            # verify prompts identical outside the k span
            assert all(ids_c[i] == ids_x[i] for i in range(T) if not (k1 <= i <= k2))

            # clean run: cache residual stream entering each layer (hidden_states[L], L=0..27)
            with torch.no_grad():
                out_c = model(**enc_c, output_hidden_states=True)
                out_x = model(**enc_x)
            hs_clean = out_c.hidden_states  # hs[L] = input to layer L for L in 0..n_layers-1
            ld_clean = class_logodds_batch(out_c.logits[:, -1], yes_ids, no_ids).item()
            ld_corr = class_logodds_batch(out_x.logits[:, -1], yes_ids, no_ids).item()
            if not (ld_clean > 0.5 and ld_corr < -0.5):
                print(f"[{ctype}] skip {a}+{b} (k {k_clean} vs {k_corr}): no behavioural contrast "
                      f"(LD {ld_clean:+.2f} / {ld_corr:+.2f})", flush=True)
                continue

            # patch cells: single positions k1..T-1, plus combined [k1,k2]
            cells = [[p] for p in range(k1, T)] + [[k1, k2]]
            if cell_labels is None:
                cell_labels = [tok.decode([ids_x[p]]) for p in range(k1, T)] + ["k(both)"]
            B = len(cells)
            batch_ids = enc_x["input_ids"].repeat(B, 1).to(args.device)
            batch_mask = enc_x["attention_mask"].repeat(B, 1).to(args.device)

            rec_item = np.zeros((n_layers, B), dtype=np.float32)
            for L in range(n_layers):
                clean_L = hs_clean[L][0]  # (T, d)

                def hook(module, hook_args):
                    h = hook_args[0]
                    for i, ps in enumerate(cells):
                        for p in ps:
                            h[i, p] = clean_L[p]
                    return None  # in-place

                hnd = layers[L].register_forward_pre_hook(hook)
                with torch.no_grad():
                    out_p = model(input_ids=batch_ids, attention_mask=batch_mask)
                hnd.remove()
                ld_p = class_logodds_batch(out_p.logits[:, -1], yes_ids, no_ids).numpy()
                rec_item[L] = (ld_p - ld_corr) / (ld_clean - ld_corr)

            # validation: patch ALL positions at a middle layer -> must fully recover
            Lmid = n_layers // 2
            clean_L = hs_clean[Lmid][0]

            def hook_all(module, hook_args):
                hook_args[0][0] = clean_L
                return None

            hnd = layers[Lmid].register_forward_pre_hook(hook_all)
            with torch.no_grad():
                out_v = model(**enc_x)
            hnd.remove()
            ld_v = class_logodds_batch(out_v.logits[:, -1], yes_ids, no_ids).item()
            rv = (ld_v - ld_corr) / (ld_clean - ld_corr)
            assert abs(rv - 1) < 0.02, f"full-patch validation failed: recovery {rv:.3f}"

            rec_all.append(rec_item)
            ld_clean_all.append(ld_clean)
            ld_corr_all.append(ld_corr)
            used.append((a, b))
            print(f"[{ctype}] {a}+{b}: clean k={k_clean} LD={ld_clean:+.2f}  corr k={k_corr} "
                  f"LD={ld_corr:+.2f}  ({(time.time()-t0)/60:.1f} min)", flush=True)

        print(f"[{ctype}] items used: {used}")
        results[ctype] = dict(recovery=np.stack(rec_all), labels=cell_labels,
                              ld_clean=np.array(ld_clean_all), ld_corr=np.array(ld_corr_all),
                              used=used, k1=k1, k2=k2)
        np.savez(os.path.join(args.out_dir, f"recovery_{ctype}.npz"),
                 recovery=np.stack(rec_all), labels=np.array(cell_labels),
                 ld_clean=np.array(ld_clean_all), ld_corr=np.array(ld_corr_all),
                 used=np.array(used))

    # ---------- figures ----------
    lines = []
    for ctype, r in results.items():
        R = r["recovery"].mean(0)  # (layers, cells)
        labels = r["labels"]
        fig, ax = plt.subplots(figsize=(9, 5))
        vmax = max(1.0, np.abs(R).max())
        im = ax.imshow(R, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([repr(l)[1:-1] for l in labels], rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("layer whose INPUT residual stream is patched")
        ax.set_xlabel("patched position (token)")
        ax.set_title(f"Activation patching, {ctype}: recovery of clean verdict "
                     f"(mean over {r['recovery'].shape[0]} items)")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("recovery  (0 = none, 1 = full)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"fig_e2_heatmap_{ctype}.png"), dpi=200)
        plt.close(fig)

        # line plot: recovery at the k columns vs the final-position column, across layers
        kcols = [len(labels) - 1]            # k(both)
        fincol = len(labels) - 2             # last prompt token
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(R[:, kcols].mean(1), color=BLUE, linewidth=2, marker="o", markersize=4,
                label="patch k tokens (both)")
        ax.plot(R[:, fincol], color=ORANGE, linewidth=2, marker="s", markersize=4,
                label="patch final position")
        ax.axhline(0, color=INK2, linewidth=0.8, linestyle=":")
        ax.axhline(1, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_xlabel("layer whose input is patched")
        ax.set_ylabel("recovery")
        ax.set_title(f"{ctype}: where does the k-information leave the k position?")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, color="#e6e5e1", linewidth=0.6)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"fig_e2_lines_{ctype}.png"), dpi=200)
        plt.close(fig)

        kboth = R[:, -1]
        fin = R[:, fincol]
        l_leave = next((L for L in range(n_layers) if kboth[L] < 0.5), None)
        l_settle = next((L for L in range(n_layers) if fin[L] > 0.5), None)
        lines += [f"[{ctype}] LD clean mean {r['ld_clean'].mean():+.2f}, corrupted {r['ld_corr'].mean():+.2f}",
                  f"[{ctype}] k-patch recovery drops below 0.5 at layer: {l_leave}",
                  f"[{ctype}] final-position patch recovery exceeds 0.5 at layer: {l_settle}"]

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.out_dir, "summary_e2.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
