# Handoff summary — rhs_sweep (early-decision interpretability project)

Context file for continuing this work in a new chat. As of 2026-11-04. All work is on
Qwen2.5-1.5B-Instruct, float32, CPU, in this folder. Shareable write-up with figures:
https://claude.ai/artifact/RveknpBLHoZkr1HmJzibQz

## The question

When Qwen judges "Is {a} + {b} = {k} correct? Answer yes or no.", does the verdict depend on k,
and if so **when/where** in the forward pass is it formed? Metric everywhere: first-token
log-odds LD = logsumexp(yes-variant logits) − logsumexp(no-variant logits); P(yes) = σ(LD).
Generation is off (the model only ever says "Yes."/"No."); verdict = argmax first token.

## Findings, in order

1. **Behaviour (step 1, `rhs_sweep.py` + `plot_sweep.py` → `results/sweep_1p5b/`)**
   The model checks a RANGE, not equality: accept iff first_operand ≤ k ≤ a+b (83% verdict
   agreement vs 63% for equality). Sharp lower edge: P(yes) 0.13 just below a vs 0.75 at/above.
   True-equation acceptance falls from 100% (sums ≤ 10) to 29% (sums 31–40). 40 pairs
   (seed 2026, a,b ∈ 1..20) × k = true±10 = 820 rows in `results/sweep_1p5b.csv`.

2. **When (exp1 logit lens / exp2 activation patching / exp3 contrast probes →
   `results/when_exp1..3/`)** — three methods run separately, all agree:
   - Edge comparisons computed AT the k tokens, layers ~8–15 (probes: k>a+b decodable at the
     last k token from L13, held-out AUC ≥ 0.9).
   - Sharp k→final-position handoff at layers 16–18 (patching crossover at L17/L18, both
     corruptions: over = a+b+2, below_a = a−2).
   - Verdict settled at final position from L18 (probe AUC 1.0); lens rank order locked ~L20,
     sign calibrated ~L26. Lens is MISLEADING mid-stack (38% sign agreement while Spearman 0.96).
   - **Patching judged most convincing** (causal, sharpest, validated: full patches recover ≈1).

3. **Which head (exp4 head patching → `results/when_exp4/`; exp5 ablation → `results/when_exp5/`)**
   - **L17H0** is the verdict-mover for BOTH edges: patching recovery 0.41 (over) / 0.42
     (below_a); attends 41% of final-position attention to the k tokens.
   - Necessity: mean-ablating L17H0 alone in clean runs costs true equations −1.12 LD (2/8
     flips), pushes false ones +0.47 (regression to uninformative); ALL other L17 heads ablate
     to ~0.00. L17H0+L18H11 ≈ additive, −1.61, ~55% separation loss.
   - Support: L18H11 (~0.2); minor L16H7, L17H2. **L18H10 is a suppressor/backup** (~−0.35
     patching; ablating it INCREASES separation +0.75; ablating it with L17H0 does less damage
     than L17H0 alone — IOI-style negative-head compensation).
   - Component split: attention writes the verdict at L17 only (attn-out recovery ~0.5); MLPs
     at L18, L19, L22 consolidate (~0.45–0.55 each); MLP L27 strongly negative when patched alone.

4. **What L17H0 carries (exp6 OV inspection → `results/when_exp6/`)**
   NOT a ready-made yes/no: direct logit attribution of its write ≈ 0 (true−false gap +0.02 vs
   total causal effect ~1.6 → >98% indirect through MLPs 18–22); write orthogonal to the
   unembedding yes−no direction (cos +0.02). It transports a raw range feature; MLPs convert it.
   Caveat: cosines with difference-of-means probe directions are NOT discriminating (null
   control head L17H3 shows similar magnitudes) — the DLA-vs-total-effect gap is the evidence.
   Fun: the write's vocab readout suppresses null/zero tokens on true equations.

5. **Order control (exp7 → `results/when_exp7/`)**
   The lower edge is POSITIONAL: with "Is b + a = k?", the sharpest step follows the
   first-printed operand (+0.61; original a-first: +0.73 first / −0.08 second). Swapped data
   blurrier (+0.39 second, +0.59 max operand) — position dominates, magnitude not fully ignored.
   Circuit is order-invariant: L17H0 still top (0.41/0.38).

**One-line story:** edge comparisons at the k tokens (L8–15) → L17H0 copies the result to the
final position at L17 (L18H11 assists, L18H10 opposes) → MLPs 18–22 cash it out as yes/no →
late layers calibrate. Verified both directions (sufficient + necessary), order-invariant.

## Files

- `rhs_sweep.py` — sweep; exports SYSTEM, USER_TEMPLATE, YES/NO_VARIANTS, build_inputs,
  locate_positions, single_token_ids (used by every exp script). `plot_sweep.py` — figs 1–4.
- `exp1_logit_lens.py`, `exp2_patching.py` (exports candidate_items, class_logodds_batch),
  `exp3_probes.py`, `exp4_heads.py`, `exp5_ablate.py`, `exp6_ov.py` (needs `--acts` = exp3
  cache), `exp7_order.py`. Each writes figures + summary_eN.txt into its results dir.
- `README.md` — full run instructions and per-experiment result paragraphs. `ROADMAP_when.md` —
  method designs/pitfalls. `HANDOFF.md` — this file.
- exp3 activation cache (~400 MB, acts.npz) was in the session scratchpad — regenerate by
  running exp3 if needed (positions: eq, k_first, k_last, after1, qmark, final × 29 layers).

## Technical gotchas (will bite a fresh session)

- transformers 5.18.0: `hidden_states[-1]` is ALREADY final-RMSNorm'd (lm_head(hs[-1]) == logits
  exactly); hs[L] for L<28 are raw residuals entering layer L. Lens code needs an
  `already_normed` flag at the top layer.
- Qwen2.5-1.5B: 28 layers, 12 query heads (GQA, 2 KV heads), head_dim 128, hidden 1536. Head
  patching/ablation = slice [h*128:(h+1)*128] of the o_proj INPUT via forward_pre_hook.
- `attn_implementation="eager"` required for output_attentions.
- Screening is essential: require LD_clean > 0.5 and LD_corr < −0.5 (the model rejects some true
  equations, e.g. 16+19=35 at P(yes) 0.02). candidate_items uses a ≥ 13 so clean/corrupted k
  tokenise to the same length.
- Windows cp1252: every script does sys.stdout.reconfigure(errors="backslashreplace") and writes
  summaries with encoding="utf-8". scipy/sklearn NOT installed — numpy-only stats (rank AUC).
- Hooks in loops need factory functions (late-binding bug otherwise).

## Open next steps

- **7B rerun on Colab** (the main one): same scripts,
  `--model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --device cuda`. Questions: does 7B check
  equality instead of the range? Same single-courier-head structure, and at what relative depth?
- Untraced: circuit behaviour for sums > 30 (behavioural collapse region); why MLP L27 patches
  negative; what the L18–22 MLPs compute on the range feature.

## Memory

Persistent memory files exist at
`~/.claude/projects/C--Users-justi-OneDrive-Desktop-Letting-Claude-Cook-rhs_sweep/memory/`:
`rhs-sweep-1p5b-finding.md`, `when-analysis-1p5b.md` (both current with everything above).
