# RHS sensitivity sweep (step 1 of the early-decision project)

Question: when Qwen judges "Is a + b = k correct?", does its verdict depend on k at all?
If P(yes) is flat in k, the verdict is formed without the evidence; if it peaks sharply at
k = a + b, the model reads the result and the question becomes *when*.

## Files
- `rhs_sweep.py`  runs the sweep (single-turn, greedy, fixed spaced template), resumable CSV
- `plot_sweep.py` three figures + `summary.txt`
- `results/synthetic/` what the figures look like on fake data (ignore the numbers)

## Run on the laptop (1.5B, CPU)
```
pip install torch transformers pandas matplotlib      # already there from the corrupted-copy project
python rhs_sweep.py --smoke --out results/smoke_v2.csv  # 10 pairs x +-5, ~1 min: check it runs
python rhs_sweep.py --out results/sweep_1p5b.csv        # 40 pairs x +-10, ~6-10 min (logits only)
python plot_sweep.py --runs results/sweep_1p5b.csv --out_dir results/sweep_1p5b --model_label "Qwen2.5-1.5B"
```
Windows: run the python lines directly (no .sh). Ctrl-C and rerun resumes from the CSV.
A CSV written by an older version of the script (without `pos_k_last`) can't be resumed; use a new `--out`.

Generation is **off by default**: in the smoke run the 1.5B only ever answered "Yes." / "No.", so
the continuation added nothing and the verdict is the argmax first token. Pass `--generate` to also
record the 16-token continuation and `then_corrects` (e.g. on the 7B, or with a prompt that invites
the model to restate the sum). It makes each row about 3-8x slower.

## Run on Colab (7B, T4)
```
!pip -q install transformers accelerate
!python rhs_sweep.py --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --device cuda --out /content/drive/MyDrive/sweep_7b.csv
```
If the 7B does not fit in 16 GB, add `load_in_8bit` via bitsandbytes (first-token logits are unaffected).

## What to look at first
1. The **tokenisation check** printed at start: the `=` token and k's tokens must sit at fixed positions,
   with the first and last tokens of k marked (Qwen splits digits, so a two-digit k spans 2 tokens).
   If "Ġ=" or the digits are merged with neighbours, stop and tell me.
2. `summary.txt`:
   - **balanced accuracy** (mean of the true-yes and false-no rates). Raw accuracy is misleading because
     ~95% of rows are false equations, so always saying "no" would score high; that baseline is printed alongside.
   - yes-rates split into **undershoot** (k < a+b) and **overshoot** (k > a+b).
   - per-item AUC: how often the true equation gets higher log-odds than a wrong one (0.5 = chance).
   - S, plus **S_under / S_over**. A single S averages both sides and hides the asymmetry seen in the
     smoke run (median S_under 0.00 vs S_over 0.58: undershoots get "yes" about as often as the truth).
   - `then_corrects`, only if run with `--generate`.
3. `fig1_sensitivity_curve.png` is the candidate first figure of the paper; `fig3` now has one panel per side.
4. `fig4_operand_aligned.png`: too-small answers only, re-aligned on k − a and k − b. On the 1.5B there is
   a sharp step at k = a (P(yes) 0.13 below, 0.75 at or above), i.e. the model accepts "a ≤ k ≤ a+b".

## Columns in the CSV
a, b, k, true, offset (= k - true), is_true, pos_eq, pos_k, pos_k_last (token positions of '=' and of
the first / last token of k; **use pos_k_last for "when" analyses**, since k is only fully known there),
prompt_len, logit_yes, logit_no (log-sum-exp over casing variants), logodds, p_yes, first_token,
gen_text (blank unless --generate), verdict (yes/no, from the continuation if generated, else from
first_token), restated_result, then_corrects (blank unless --generate), secs.

Note: k < 0 is skipped (sign errors are a different kind of wrongness; handled in the main study).

## Step 2: the "when" analysis (see ROADMAP_when.md)
Three independent methods, one script + results dir each (1.5B, CPU):
- `exp1_logit_lens.py` -> `results/when_exp1`  (observational: lens readout per layer)
- `exp2_patching.py`   -> `results/when_exp2`  (causal: activation patching, over / below-a corruptions)
- `exp3_probes.py`     -> `results/when_exp3`  (correlational: difference-of-means probes, item-held-out AUC)

Result (consistent across all three): the edge comparisons (k vs a, k vs a+b) are computed AT the
k token positions in the mid-stack (probes: over-edge decodable there from ~layer 13 of 28); the
verdict moves from the k position to the final position in a sharp handoff at layers 16-18
(patching: k-patch recovery ~1 before, ~0 after; final-position patch the reverse); from ~layer 18
the verdict is settled at the final position and later layers only rotate it into the yes/no
output direction (lens: rank order locked by ~layer 20, sign calibrated by ~26).

- `exp4_heads.py` -> `results/when_exp4`  (head-level patching at the final position, layers 13-21)

The handoff is mostly ONE head: **L17H0** (recovery 0.41 over / 0.42 below-a, the same head for
both decision edges), which attends ~41% to the k tokens from the final position. Support from
L18H11 (~0.2) and smaller L16H7 / L17H2; L18H10 is a consistent negative/backup head (~-0.35).
Component split: attention writes the verdict at layer 17 (attn-out recovery ~0.5), MLPs at
18-19 and 22 consolidate it (~0.45-0.55 each). No other layer's attention matters at the final
position.

- `exp5_ablate.py` -> `results/when_exp5`  (necessity test: single-head ablation in CLEAN runs)

Mean-ablating L17H0's final-position output alone drags true-equation log-odds down by 1.1
(2/8 verdicts flip) and pushes false ones up by ~0.5 — i.e. the verdict regresses toward
uninformative — while every other L17 head ablates to ~0.00. L18H11 is the second mover
(true -0.57); together with L17H0 the separation loss is ~55% (true -1.6, 2/8 flips).
Ablating the negative head L18H10 *increases* separation (true +0.75), and removing it together
with L17H0 cancels part of the damage (-0.73 vs -1.12) — a genuine suppressor/backup dynamic.
So L17H0 is causally necessary for full-confidence verdicts, but the write is partially
redundant (L18H11 + downstream MLPs reading other residual paths), matching the exp4
recoveries summing below 1. Zero-ablation (off-distribution) is noisier; read the mean columns.

- `exp6_ov.py` -> `results/when_exp6`  (what does L17H0 write? direct logit attribution + probe projections)

L17H0 does NOT transport a ready-made yes/no signal: its write's direct logit attribution on
yes-no is ~0 (true-false gap +0.02, vs a total causal effect of ~1.6 LD in exp5 — >98% of its
effect is indirect, through the L18-22 MLPs), and the write is orthogonal to the unembedding
yes-no direction (cos +0.02). So the head moves a range FEATURE to the final position and the
MLPs convert it into the verdict, matching the exp4 component split (attention at 17, MLPs at
18-22). The write's vocab readout suppresses null/zero-flavoured tokens on true equations —
suggestive, not load-bearing. Caveat: cosines with difference-of-means probe directions are NOT
discriminating here (the causally-null control head L17H3 shows comparable magnitudes), so the
conclusion rests on the DLA-vs-total-effect gap, not the cosines.

- `exp7_order.py` -> `results/when_exp7`  (operand-order control: "Is {b} + {a} = {k} correct?")

The lower edge is POSITIONAL: with the operands swapped, the sharpest step follows the FIRST
PRINTED operand (step +0.61, P(yes) 0.11 below vs 0.72 at/above), just as it followed a when a
was printed first (+0.73 / -0.08 at the second operand in the original sweep). So the rule is
"the number right after 'Is' <= k <= sum", not tied to operand identity. The swapped data is
noisier than the original (a residual step +0.39 on the second operand / +0.59 on the max), so
printed position dominates but magnitude is not fully ignored. And the circuit is order-invariant:
on swapped prompts L17H0 is still the top verdict-mover (recovery 0.41 over / 0.38 below-first,
L18H11 ~0.2 again).
