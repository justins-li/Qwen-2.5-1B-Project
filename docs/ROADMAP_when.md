# "When" analysis roadmap (step 2 of the early-decision project)

Step 1 established *that* the 1.5B's verdict depends on k, via a range rule (accept if a ≤ k ≤ a+b),
not equality. Step 2 asks **when** the verdict is formed: at which layer, and reading from which
token positions. Three methods, run **separately** (one script, one run, one figure set each), so
that no method's assumptions contaminate another's result.

All experiments: Qwen/Qwen2.5-1.5B-Instruct (28 layers, d=1536, tied unembedding), CPU, float32,
same chat template and spacing as `rhs_sweep.py`, single-turn, no generation. The decision position
is the final prompt token (the assistant-start token); the evidence arrives at `pos_k_last`.

---

## Experiment 1 — Logit lens (`exp1_logit_lens.py`, observational, cheapest)

**Idea** (nostalgebraist 2020): take the residual stream at every layer, apply the model's own final
RMSNorm + unembedding, and read off logit("yes") − logit("no") as if the model stopped there.
Shows at which **layer** the verdict (in the yes/no readout basis) appears.

**Design**
- Reuse the full 820-row sweep grid (same seed → same 40 pairs, offsets −10..+10).
- One forward pass per row with `output_hidden_states=True`; at the final position compute
  per-layer logodds_ℓ for ℓ = 0..28 (hidden_states[ℓ] → norm → lm_head, log-sum-exp over casing
  variants, same as step 1).
- Secondary (small subset, 8 pairs × {true, over+2, under<a}): the same readout at every position
  from `=` onward → layer × position heatmap.

**Outputs**: mean logodds-vs-layer trajectories split by condition (true / undershoot in-range /
undershoot below a / overshoot); "decision layer" = earliest ℓ where sign(logodds_ℓ) matches the
final verdict on ≥ 90% of rows; per-layer Spearman correlation with final logodds.

**Pitfalls** (tuned-lens paper): logit lens is a biased readout and can be misleading on some model
families; it is a *readout, not a causal claim* — a late lens transition is compatible with earlier
computation in a different basis. We mitigate by (a) validating that layer-28 lens equals the real
logits exactly, (b) treating it as hypothesis-generating for Exp 2/3.

**Cost**: ~820 + ~700 forwards ≈ 15 min CPU.

---

## Experiment 2 — Activation patching (`exp2_patching.py`, causal)

**Idea** (causal tracing; Heimersheim & Nanda 2024): run a *corrupted* prompt (wrong k), restore the
residual stream at one (layer, position) from the *clean* run (true k), and measure how much of the
clean verdict is recovered. Localises **where and at what depth** the k-information that drives the
verdict flows.

**Design**
- Pairs differ **only in the k tokens**, same token count: items with two-digit a (13..20) so that
  clean k = a+b, overshoot k = a+b+2, and undershoot-below-a k = a−2 are all two digits.
- Two corruption types, patched separately: **overshoot** (crosses the upper edge) and
  **below-a** (crosses the lower edge). 8 items each.
- Denoising direction: corrupted run + clean activation patched in at one cell.
  Cells: layer ℓ ∈ {0..28} (resid_pre of each layer + final) × position p ∈ {pos_k_first .. last
  prompt token} (~15 positions; positions before k are identical by construction, so skipped).
- Metric: **normalised logit-difference recovery** = (LD_patched − LD_corr) / (LD_clean − LD_corr),
  LD = logodds(yes−no). Continuous, linear in logits, 0 = no effect, 1 = full recovery.

**Outputs**: layer × position recovery heatmap per corruption type (mean over items); the
"information path" — where recovery moves from the k-position column to the final-position column.

**Pitfalls**: single-cell patching can under-report when information is spread across several
positions/layers (back-up heads, distributed routing); results depend on the corruption — hence two
corruption types. Interpretation rule: recovery at (ℓ, final position) means the verdict is already
*settled* in the residual stream there; recovery at (ℓ, k position) means the k-evidence has not yet
moved on.

**Cost**: 29 × ~15 cells × 16 pairs ≈ 7k patched forwards, batched over positions per layer →
~45–70 min CPU, run in background.

---

## Experiment 3 — Contrast probes (`exp3_probes.py`, correlational, basis-free)

**Idea** (difference-of-means / mass-mean probing, Marks & Tegmark 2023): for each (layer, position),
compute the mean activation difference between contrast classes on *training items*, project
held-out items onto that direction, and measure AUC. Unlike the logit lens this needs no readout
basis, so it detects the verdict even if it is not yet rotated into the unembedding directions.

**Design**
- Same 820-row grid; one forward per row, cache residual stream at 6 positions
  (`=`, k first, k last, k+1, `?`, final) × 29 layers, float16.
- Contrasts probed (each its own probe):
  A. model-verdict yes vs no (where is the *decision* decodable),
  B. overshoot vs not (k > a+b) — the upper-edge comparison,
  C. below-a vs not (k < a, undershoots only) — the lower-edge comparison.
- Difference-of-means direction, **split by item** (30 train / 10 test, 5 random splits) so the
  probe cannot memorise items; report held-out AUC per (layer, position).

**Outputs**: AUC vs layer curves at k-last and final positions per contrast; layer × position AUC
heatmap; "earliest decodable layer" (AUC > 0.9 held-out).

**Pitfalls**: probes find *correlates*, not causes — a feature can be decodable yet unused
(Geometry-of-Truth caveats); mitigated by requiring item-held-out generalisation and by comparison
with Exp 2's causal map.

**Cost**: ~820 forwards ≈ 10 min + seconds of numpy.

---

## How the three answer "when" differently
| | basis | evidence type | localises |
|---|---|---|---|
| logit lens | unembedding | observational | layer at which verdict readout appears |
| patching | none (causal) | interventional | (layer, position) cells that carry/settle the verdict |
| probes | learned direction | correlational | earliest layer/position where verdict is decodable |

Agreement pattern to look for: probes decodable at layer ℓ₁ ≤ lens transition ℓ₂, and patching
shows the k→final information movement completing between ℓ₁ and ℓ₂. The *causal* map (Exp 2) is
the arbiter; lens and probes date the representation.
