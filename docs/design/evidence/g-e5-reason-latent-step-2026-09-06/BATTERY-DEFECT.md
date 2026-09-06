# E5 step battery: distractor-pool defect and missing instrument-solvability control

This documents a defect in `build_step_battery`'s 5-way candidate set found during
adversarial verification of this evidence pack, not by the original E5 run. It explains
why the untrained (pre-training) predictor scores below the nominal chance figure in
every seed, and it supplies an instrument-solvability control the diagnosis mandates
for the sibling E1 battery and this experiment never added. **It does not change the
KILL verdict** — see "Why the verdict is unaffected" below.

## The defect

`build_step_battery` (`reason_latent_step.py:394-443`) restricts the true-target slot
to steps that carry a corruptible calculator annotation, dropping any window whose
target has none:

```python
corrupted = corrupt_step(target, item_id=ex.item_id, t=t, corruption_seed=corruption_seed)
if corrupted is None:
    continue                                                    # reason_latent_step.py:425-426
pool = [s for owner, _, s in all_steps if owner != ex.item_id]   # reason_latent_step.py:427
```

The three "other" distractors, however, are drawn from `pool` — *every* step of *every*
other holdout item, completely unfiltered by whether that step carries an annotation.
The true-target slot and the distractor slots are therefore drawn from two different
populations of step lines: one restricted to annotated steps, the other not. gsm8k
derivations end with a `#### <final answer>` line, which never carries a calculator
annotation (`<<a op b=c>>`) and so can never land in the true-target slot, but can land
—unfiltered—in a distractor slot.

## Measured impact

Rebuilding the real 727-item battery on CPU from `/mnt/bulk/csd-corpus` (the same
corpus, split, and order manifests the six graded receipts used —
`battery_fingerprint` reproduces `6b6822f690f2c63771a8f535d8b09f1309fca72ffa5ddc9137485ed5855d42dc`
bit for bit, matching every receipt's `counts.battery_fingerprint`):

| | true target | corrupted (twin) | distractors (3×727=2181) |
|---|---|---|---|
| is a `#### N` line | 0/727 (0.0%) | — | 491/2181 (22.5%) |
| mean length (chars) | 79.6 | 79.5 | 62.4 |

No item has the true target text duplicated among its distractors, and no item's
corrupted candidate is textually identical to its true target — the defect is
population-level (length and surface-form skew), not a labeling bug.

Because roughly a fifth of the distractor pool is short, formulaic `#### N` text (which
never competes for the true-target slot) while the true and corrupted candidates are
both ordinary derivation-step prose of near-identical length, the five candidates are
not exchangeable. An **untrained** `LatentStepModel` (same construction the trainer
uses, `torch.manual_seed(seed)` before construction, scored before either arm trains)
lands below the battery's nominal chance of 0.20 in every seed:

| seed | untrained recall@1 (as-shipped) |
|---|---|
| 0 | 0.1458 |
| 1 | 0.1224 |
| 2 | 0.0495 |

These reproduce the receipts' `untrained_predictor_battery.recall@1` /
`results.json.per_run.*.untrained_recall@1` bit for bit.

## Distribution-matched distractors restore the null

Redrawing the three "other" distractors only from steps that carry a calculator
annotation (`t >= 1`, matching the true-target slot's own restriction) and rescoring the
same untrained models, via the verifier's `probe_real2.py` (CPU, read-only, reused
as-is; run again here to confirm):

```
[untrained model, chance would be 0.20]
  seed=0: as-shipped=0.1458   distribution-matched=0.2050
  seed=1: as-shipped=0.1224   distribution-matched=0.2228
  seed=2: as-shipped=0.0495   distribution-matched=0.1884
```

Matching the distractor distribution to the true-target distribution puts all three
seeds back at chance (0.19–0.22 vs. the nominal 0.20). **The sub-chance floor is a
battery-composition artifact, not evidence about either training objective.**

## Missing instrument-solvability control

The reasoning-region diagnosis requires, for the sibling E1 corrupted-derivation
battery, "(a) wrong-problem control — true derivation vs four other problems'
derivations — on which TF-IDF must score ≥ 0.90, proving the instrument is solvable
where the signal is lexical; (b) TF-IDF and BM25 on the corrupted battery must land
within ±0.05 of 0.20, proving it is lexically unsolvable"
(`docs/design/evidence/reason-region-diagnosis-2026-09-06/README.md:85`). E5's
pre-registration (`PREREG.md`) and the six receipts carry the E1 corrupted-derivation
battery only as a regression guard scored on *E1's own* battery — the new E5 step
battery has no instrument-solvability control of its own.

Running the missing control — plain TF-IDF cosine similarity from each item's context
to its five candidates, on the real 727-item battery — via `probe_real2.py`
(reused read-only):

```
[instrument solvability -- the control E1 has and E5 lacks]
  TF-IDF as-shipped (true, corrupted twin, 3 others): recall@1=0.9367  P(argmax in slot0/1)=0.9780  P(slot1 > slot0)=0.0014
  TF-IDF distribution-matched distractors (no near-duplicate): recall@1=0.9821  P(argmax in slot0/1)=0.9862  P(slot1 > slot0)=0.0083
```

Plain lexical overlap (no model, no training) scores recall@1 = 0.9367 as-shipped, and
0.9821 once the near-duplicate corrupted-step candidate is replaced by a
distribution-matched distractor — clearing E1's own ≥ 0.90 solvability bar several
times over. This is an unreported lexical ceiling: every trained or untrained model arm
in this evidence pack (0.05–0.18 recall@1) sits far below what a bag-of-words scorer
achieves on the identical items. The instrument is trivially solvable lexically; the
"below chance" floor comes from candidate-population skew, not from the instrument
being unsolvable.

## Why the verdict is unaffected

Both defects — the non-exchangeable distractor pool and the missing
instrument-solvability control — apply identically to both arms: `build_step_battery`
is called once per run with the same `corruption_seed=0`, and both arms in a seed are
scored against the same battery object. The within-seed margin (`latent-step recall@1 -
sequence-blind recall@1`), which is what the pre-registered rule actually thresholds,
is a paired difference on identical items and is therefore unaffected by any
battery-composition artifact common to both candidate sets. All three seeds' margins
(-0.0124, -0.0440, -0.0234) clear the +0.05 kill threshold by 0.062–0.094, roughly 3–5
binomial standard errors at n=727 — well outside what candidate-population skew shared
by both arms could produce as a spurious margin.

## Follow-up required

Any successor experiment that reuses `build_step_battery` as-is inherits both defects
and must fix them before its results can be trusted at face value:

1. Draw distractors from the same population as the true-target slot (annotated steps
   only), not from every step of every other item.
2. Add the E1-style instrument-solvability pair — a wrong-problem control that a
   bag-of-words scorer must clear (≥ 0.90), and a lexical-ceiling check
   (TF-IDF/BM25 within ±0.05 of chance) on the corrupted-battery item shape actually
   used.

This fix is out of scope for this evidence pack (E5 is KILLed; nothing here reruns any
training) and belongs in whatever pre-registration next reuses this battery shape.

## Reproduction

CPU only, read-only against the committed corpus/split/order manifests; nothing in this
repo or worktree was modified to produce these numbers.

```
export CUDA_VISIBLE_DEVICES=""
.venv/bin/python /akula-data/session-backup-staging/tmp/e5-verify/probe_real.py
.venv/bin/python /akula-data/session-backup-staging/tmp/e5-verify/probe_real2.py
```

`probe_real.py` rebuilds the battery, reproduces `battery_fingerprint` and the
untrained baselines bit for bit against the six receipts, and reports the
candidate-composition statistics (`#### N` fraction, mean lengths, duplicate checks).
`probe_real2.py` reports the TF-IDF instrument-solvability numbers and the
distribution-matched-distractor comparison. Both scripts are from the adversarial
verification pass on this evidence pack and are reused here unmodified.
