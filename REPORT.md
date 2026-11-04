# A range heuristic, one courier head: how Qwen2.5-1.5B verifies "a + b = k"

*All numbers and figures below come from the committed results in `results/` (Qwen2.5-1.5B-Instruct,
float32, CPU). Nothing here requires re-running code.*

---

## 1. Results

**R1 — The model checks a range, not equality.** Asked "Is {a} + {b} = {k} correct? Answer yes or
no.", the model accepts any k with **a ≤ k ≤ a+b**: the range rule matches its verdicts on 83% of
820 items vs 63% for true equality. The lower edge is sharp — P(yes) is 0.13 just below a and 0.75
at or above it — while everything between a and the true sum is accepted at roughly the same rate as
the truth. Acceptance of *true* equations also decays with size, from 100% (sums ≤ 10) to 29%
(sums 31–40). *Implication: the model's "verification" is a cheap plausibility heuristic, so its
yes/no cannot be trusted as an arithmetic check.*

**R2 — The verdict is assembled in three stages at identifiable places.** Causal patching shows the
edge comparisons (k vs a, k vs a+b) are computed **at the k token positions in layers ~8–15**, the
result is **handed to the final position in a sharp window at layers 16–18**, and from layer 18 the
verdict is settled at the final position while later layers only rotate it into the yes/no output
direction. *Implication: a behavioral shortcut has a discrete, locatable implementation — it is not
smeared across the network.*

**R3 — One attention head carries the verdict.** Head-level patching finds a single head, **L17H0**,
that moves the verdict for *both* decision edges (recovery 0.41 overshoot / 0.42 undershoot; next
head ~0.2) and attends 41% of its final-position attention to the k tokens. It is also necessary:
mean-ablating L17H0 alone in clean runs costs true equations −1.12 log-odds (2/8 verdicts flip)
while every other layer-17 head ablates to ~0.00. L18H11 assists (~0.2), and L18H10 is a
suppressor/backup head — ablating it *increases* separation (+0.75), and ablating it together with
L17H0 does less damage than ablating L17H0 alone. *Implication: the circuit has the same
courier-plus-backup structure found in IOI-style circuits, in a model this small, on a task this
mundane.*

**R4 — The head ships a feature, not an answer.** L17H0's write has ~zero direct effect on the
yes/no logits (direct logit attribution gap +0.02 vs a total causal effect of ~1.6 → >98% indirect):
it transports a raw range feature that **MLPs 18–22 convert into the verdict** (each consolidating
~0.45–0.55 of the effect when patched). *Implication: attention routes, MLPs decide — the division
of labor is clean enough to state in one sentence.*

**R5 — The rule is positional, and the circuit doesn't care.** With the operands swapped
("Is {b} + {a} = {k}?"), the sharp lower edge follows the **first-printed** operand (step +0.61,
vs +0.73 for a when a was printed first and −0.08 at the second operand), so the rule is "the number
right after 'Is' ≤ k ≤ sum". The circuit itself is order-invariant: L17H0 is still the top
verdict-mover (0.41 / 0.38). *Implication: the heuristic reads surface positions, not operand
identity — and the mechanism is robust to the control.*

**One-line story:** edges compared at the k tokens (L8–15) → L17H0 copies the result to the final
position at L17 (L18H11 assists, L18H10 opposes) → MLPs 18–22 cash it out as yes/no → late layers
calibrate. Verified in both directions (sufficiency by patching, necessity by ablation), and
order-invariant.

---

## 2. Methods — as they would appear in a paper

This is the clean route: five steps, each answering one question, each a prerequisite for the next.
It is *not* the order the work actually happened in (three redundant "when" methods were run in
parallel and the strongest kept — see §4).

### 2.1 Task and metric

Single-turn chat prompts, fixed template: system "You are a careful math checker.", user
"Is {a} + {b} = {k} correct? Answer yes or no." The model (Qwen2.5-1.5B-Instruct, float32, greedy)
answers only "Yes." or "No." in this setting, so generation is disabled and the verdict is defined
from the **first-token logits**:

> LD = logsumexp(logits of yes-variants) − logsumexp(logits of no-variants);  P(yes) = σ(LD)

summing over casing/whitespace variants ("Yes", " yes", …). Using log-odds instead of accuracy
gives a continuous, per-item effect size that patching and ablation can move by fractions.

Two tokenization facts are load-bearing: Qwen splits numbers into digit tokens, so k spans up to two
tokens and is only fully known at its **last** token (all positional analyses index that token); and
items are constructed so clean/corrupted prompts tokenize to identical lengths (a ≥ 13), which makes
position-aligned patching exact.

### 2.2 Behavioral sweep — does the verdict depend on k, and how?

40 (a,b) pairs (a,b ∈ 1..20, fixed seed) × k = true sum + offset, offset ∈ [−10, +10], k ≥ 0 →
820 prompts. Report P(yes) as a function of the offset (Fig. 1), balanced accuracy against the
always-no baseline (~95% of rows are false equations, so raw accuracy is meaningless), and —
the step that found the actual rule — **re-align the undershoot errors on k − a and k − b**
(Fig. 2). Fit candidate decision rules (equality; a ≤ k ≤ a+b; b ≤ k ≤ a+b; …) and report verdict
agreement for each.

### 2.3 Causal localization — where and when is the verdict formed?

Activation patching over the full layer × position grid. Eight screened items (LD_clean > 0.5,
LD_corrupted < −0.5 — screening matters because the model rejects some true equations outright),
two corruption families chosen to target each decision edge: **over** (k → a+b+2) and **below_a**
(k → a−2). Patch the clean residual stream into the corrupted run at one (layer, position); metric
is **recovery** = (LD_patched − LD_corr) / (LD_clean − LD_corr), so 0 = no effect, 1 = full verdict
restored. Validate the scale by confirming that patching all positions at once recovers ≈ 1.

This single experiment produces the paper's central figure (Fig. 3): information lives in the k
column up to ~L15, the final-position column takes over above L18, and the crossover at L16–18 *is*
the handoff. Run it for both corruptions; the story must (and does) replicate across edges.

### 2.4 Head-level attribution — which component does the handoff?

Same patching logic, restricted to the handoff window (layers 13–21), at the final position only,
one attention head at a time (GQA: slice the o_proj input per query head). Report the layer × head
recovery heatmap (Fig. 4a) and, for the top heads, their attention distribution from the final
position over labeled position groups {k, a, b, "=", final, other} (Fig. 4b). Complement with a
coarse component split (attention-out vs MLP-out per layer) to see what kind of module acts when.

### 2.5 Necessity — ablation in clean runs

Patching proves sufficiency of a path; it cannot show the model *relies* on it. Mean-ablate each
candidate head's final-position output in **clean** runs (mean over the item batch; zero-ablation is
reported alongside but is off-distribution and noisier) and report Δ log-odds on true and false
equations plus verdict flips (Fig. 5). Include head pairs (L17H0+L18H11, L17H0+L18H10) to expose
additivity and the suppressor dynamic.

### 2.6 What the head writes — feature or verdict?

Direct logit attribution of L17H0's write (project through final RMSNorm and the unembedding yes−no
direction) compared against its total causal effect. A near-zero DLA with a large total effect means
the write is consumed by downstream modules, not read out directly. Negative control: run every
head-level analysis on a causally-null head of the same layer (L17H3); any statistic that doesn't
separate the courier from the control (as happened with probe-direction cosines) is dropped.

### 2.7 Controls

1. **Operand order** ("Is {b} + {a} = {k}?"): distinguishes "first-printed operand" from "operand a";
   re-run the behavioral edge analysis and the head patching on swapped prompts.
2. **Both corruption directions** throughout: any claimed component must move the verdict for both
   the over- and under-shoot edge.
3. **Full-patch calibration** of the recovery metric (§2.3).

### 2.8 What is deliberately *not* in the methods

Linear probes were run in the original work (difference-of-means, item-held-out AUC) and agreed with
patching everywhere. They are omitted here: correlational, they require a ~400 MB activation cache
and a train/test split, and decodability never implies use. The logit lens is likewise demoted to a
sanity check: in this model it is actively misleading mid-stack (38% sign agreement with the final
verdict at layers 13–22 while rank correlation is 0.96 — the *ordering* is right long before the
*sign* is), which is itself a useful cautionary observation but not localization evidence.

---

## 3. Figure plan

The rule of thumb used: **heatmaps for "where/when" questions** (two spatial axes, one effect color —
and a diverging colormap centered at 0 so suppression is visibly blue), **line/step plots for "how
does the verdict change along one variable"**, bars only for small labeled sets (heads). Every figure
answers exactly one question stated in its title.

| # | File | Question it answers | Why this form |
|---|------|--------------------|---------------|
| 1 | `results/sweep_1p5b/fig1_sensitivity_curve.png` | Does P(yes) depend on the stated result at all? | Line + bootstrap CI over items; the asymmetry (flat left shoulder, cliff on the right) is the finding and is visible in one glance. |
| 2 | `results/sweep_1p5b/fig4_operand_aligned.png` | What sets the lower edge? | Same curve re-aligned on k−a vs k−b side by side; the step function appears **only** in the k−a alignment — the comparison of two panels *is* the argument. |
| 3 | `results/when_exp2/fig_e2_heatmap_over.png` (+ `_below_a`) | Where/when does the verdict live? | Layer × position heatmap of recovery. The L-shaped structure — dark k-column below L15, dark final-column above L18, crossover at 16–18 — makes the three-stage story one picture. The paper's centerpiece. |
| 4a | `results/when_exp4/fig_e4_heads.png` | Which head? | Layer × head heatmap, over and below_a side by side: one dark cell (L17H0) in both panels, plus the blue L18H10 suppressor. Sparsity is the point, and a heatmap shows sparsity honestly (a bar chart of the top-5 would hide how empty the rest is). |
| 4b | `results/when_exp4/fig_e4_attention.png` | Is the courier actually looking at k? | Grouped bars over labeled position groups; L17H0's 41% on k vs every other head's ~8% needs categorical, not spatial, axes. |
| 5 | `results/when_exp5/fig_e5_ablation.png` | Is L17H0 necessary? | Bars of Δ log-odds per ablated head, true vs false colored; the single big negative bar at L17H0 among ~0 siblings, and the sign-flipped L18H10 bar, carry R3. Read the **mean-ablation** panel. |
| 6 | `results/when_exp7/fig_e7_edges.png` | Is the rule positional? | Edge-step comparison on swapped prompts. |

Improvements if the figures were remade (no re-runs needed — all underlying arrays are committed as
`.npz`/`.csv`):

- **Merge Fig. 3's two corruptions** into one two-panel figure with a shared colorbar and annotate
  the L16–18 handoff window with a bracket — currently the reader must flip between files.
- The exp1 lens heatmap (`fig_e1_heatmap_over.png`) is washed out (effects concentrated in two
  columns on a wide position axis) — if the lens is shown at all, show only
  `fig_e1_decision_layer.png` (sign agreement vs rank correlation per layer), which is the actual
  lens finding.
- Fig. 5 should show the mean-ablation panel only, trimmed to layer-17 heads + L18H11 + L18H10 +
  the two pairs; the zero-ablation panel and the long tail of ~0 heads belong in an appendix.

---

## 4. If I had to do it all over again

The actual route ran three "when" methods in parallel (lens, patching, probes), then head patching,
ablation, OV analysis, and an order control — roughly 7 scripts and several dead ends (probe-cosine
geometry that a null control invalidated; lens sign-flips that cost a debugging detour before being
recognized as a real property). The redo is five steps, strictly ordered, each gating the next:

1. **Behavioral sweep first, and mine it fully** (~10 min CPU). The range rule was found by
   re-aligning errors on k−a — a pure pandas operation on the sweep CSV. Everything mechanistic
   was downstream of knowing the *behavioral* rule; finding it early selects the right corruptions
   (a−2 and a+b+2 target the two edges directly).
2. **Layer × position patching, both corruptions** (~30 min). This one experiment answers
   "when/where" with causal force. Skip the lens as evidence (keep it as a 5-minute sanity plot at
   most); skip probes entirely.
3. **Head patching in the window patching found** (~20 min), with the attention readout built in —
   the "41% on k" number turns a statistical finding into a mechanism.
4. **Mean ablation of the top heads + pairs** (~10 min). Sufficiency and necessity together license
   the word "circuit"; pairs are cheap and expose the suppressor.
5. **The order control** (~15 min), because "a ≤ k" and "first-printed ≤ k" are different claims
   and only one survives.

Optional sixth: DLA of the courier's write (collect the single needed residual inline — a 20-line
hook — rather than a 400 MB probe cache), with a null-control head mandatory from the start.

Total: an afternoon on a CPU laptop, one clean narrative, and every claim either causal or
explicitly labeled a control. The general lesson: **run the cheapest causal method early** — it
makes the correlational methods unnecessary rather than merely redundant.

---

*Companion files: [`README.md`](README.md) (pipeline + gotchas), [`docs/HANDOFF.md`](docs/HANDOFF.md)
(raw session handoff), [`docs/ROADMAP_when.md`](docs/ROADMAP_when.md) (original method design notes),
per-experiment prose in [`docs/README_experiments.md`](docs/README_experiments.md), and
`results/*/summary_*.txt` for exact numbers.*
