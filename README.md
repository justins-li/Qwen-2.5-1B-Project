# Does Qwen2.5-1.5B actually check the math? A range heuristic and the single head that carries it

When Qwen2.5-1.5B-Instruct is asked **"Is {a} + {b} = {k} correct? Answer yes or no."**, it does
not check equality. It accepts any stated result in the range **a ≤ k ≤ a+b** (the first printed
operand up to the true sum): P(yes) is 0.13 just below a vs 0.75 at or above it, and the range rule
matches the model's verdicts on 83% of items vs 63% for true equality. Causally, the verdict is
carried from the k tokens to the final position by **one attention head, L17H0**, and converted into
yes/no by MLPs 18–22.

**The full write-up, with the methods as they would appear in a paper and the figure plan, is in
[REPORT.md](REPORT.md).** All results referenced there are committed under `results/` — nothing
needs re-running to read the report.

## Repository map

```
rhs_sweep.py        behavioral sweep (step 1) — also the shared library: prompt template,
                    tokenization/position helpers, yes/no log-odds metric
plot_sweep.py       figures + summary for the sweep
exp1_logit_lens.py  logit lens per layer/position (observational sanity check)
exp2_patching.py    activation patching, layer x position (the core causal map)
exp4_heads.py       head-level patching at the final position (finds L17H0)
exp5_ablate.py      head ablation in clean runs (necessity test)
exp6_ov.py          what L17H0 writes (direct logit attribution) — optional
exp7_order.py       operand-order control ("Is b + a = k?")
exp3_probes.py      linear probes — DEPRECATED from the pipeline, kept for the record (below)
results/            all outputs: CSVs, .npz arrays, figures, per-experiment summary_eN.txt
docs/               HANDOFF.md (session handoff), ROADMAP_when.md (method design notes),
                    README_experiments.md (original per-experiment README)
```

Model: `Qwen/Qwen2.5-1.5B-Instruct`, float32, CPU. Metric everywhere: first-token log-odds
`LD = logsumexp(yes-variant logits) − logsumexp(no-variant logits)`; the verdict is the argmax
first token (the model only ever answers "Yes."/"No.", so generation stays off).

## The simplified pipeline (no probes)

Five steps, each one script, each answering one question:

```bash
# 1. IS there an effect? Behavioral sweep: 40 (a,b) pairs x k = true sum ± 10  (~10 min CPU)
python rhs_sweep.py --out results/sweep_1p5b.csv
python plot_sweep.py --runs results/sweep_1p5b.csv --out_dir results/sweep_1p5b --model_label "Qwen2.5-1.5B"

# 2. WHERE/WHEN is the verdict formed? Activation patching, layer x position
python exp2_patching.py          # -> results/when_exp2

# 3. WHO moves it? Head-level patching at the final position, layers 13-21
python exp4_heads.py             # -> results/when_exp4

# 4. Is that head NECESSARY? Ablation in clean runs
python exp5_ablate.py            # -> results/when_exp5

# 5. Is the rule positional or arithmetic? Operand-order control
python exp7_order.py             # -> results/when_exp7
```

Optional extras, not needed for the main claims:

- `exp1_logit_lens.py` — cheap observational cross-check. Caveat: the lens is actively misleading
  in the mid-stack here (38% sign agreement with the final verdict while Spearman ρ is 0.96), so
  it must not be the primary "when" evidence. Patching is.
- `exp6_ov.py` — shows L17H0 transports a range *feature*, not a ready-made yes/no (direct logit
  attribution ≈ 0 vs total causal effect ~1.6 → >98% of its effect is indirect through MLPs 18–22).
  It currently reads an activation cache produced by `exp3_probes.py --cache` (~400 MB, not
  committed). If redone from scratch, exp6 should collect the one residual stream it needs
  (entering L18 at the final position) inline — a ~20-line hook — instead of depending on exp3.

### Why the probes were dropped

`exp3_probes.py` (difference-of-means probes, item-held-out AUC) was one of three "when" methods run
in parallel. Its answer agreed with logit lens and patching everywhere, and it is the weakest form of
evidence of the three: correlational, needs a train/test split and an activation cache, and "a probe
can decode X at layer L" never implies the model *uses* X there. Patching alone gives the same
localization with causal force. The probe results are preserved in `results/when_exp3/` and the
script still runs, but the pipeline and the report no longer rest on it.

## Technical gotchas (read before touching the code)

- transformers 5.18.x: `hidden_states[-1]` is **already final-RMSNorm'd** (`lm_head(hs[-1])`
  equals the logits exactly); `hs[L]` for L<28 are raw residuals entering layer L. Lens code
  needs an `already_normed` flag at the top layer.
- Qwen2.5-1.5B: 28 layers, 12 query heads (GQA, 2 KV heads), head_dim 128, hidden 1536. Head
  patching/ablation = slicing `[h*128:(h+1)*128]` of the **o_proj input** via a forward_pre_hook.
- `attn_implementation="eager"` is required for `output_attentions`.
- Screening is essential: require LD_clean > 0.5 and LD_corr < −0.5 — the model rejects some true
  equations outright (e.g. 16+19=35 at P(yes) 0.02). `candidate_items` uses a ≥ 13 so clean and
  corrupted k tokenize to the same length.
- Qwen splits multi-digit numbers into digit tokens: k spans up to 2 tokens; use the **last** k
  token (`pos_k_last`) for any "when" analysis, since k is only fully known there.
- Hooks registered in loops need factory functions (Python late-binding bug otherwise).
- Windows: scripts reconfigure stdout to `errors="backslashreplace"` and write summaries as UTF-8.
  No scipy/sklearn dependency — numpy-only stats (rank AUC).

## Open next steps

- **7B rerun** (same scripts: `--model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --device cuda`):
  does the larger model check equality instead of the range? Same single-courier-head structure?
- Untraced: circuit behavior for sums > 30 (where acceptance of true equations collapses to 29%);
  why MLP L27 patches negative; what MLPs 18–22 compute on the range feature.
