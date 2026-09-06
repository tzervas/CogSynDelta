# E5: reason latent-step prediction — KILL

E5 tested whether predicting a gsm8k derivation's next step, as a latent vector, from
prior steps beats a control that cannot see prior steps at all. It does not — the
control wins in all three seeds — so this objective is killed for the `reason` region
(`GO_KILL.md`).

## What was tested

Two arms trained on the same trunk, split, and order manifest, differing only in the
objective's context window (`PREREG.md` has the full pre-registration):

- **latent-step**: predict step `t`'s target-encoder latent from `question + steps[:t]`.
- **sequence-blind**: predict step `t`'s latent from `question` alone (a shortcut-
  detection control, per W1d's pattern).

Three seeds (0, 1, 2), 4000 steps, batch size 256, matched to the `reason` `b256-s0`
baseline. Metric: 5-way step-battery `recall@1` (chance 0.20).

## Results

| Seed | latent-step recall@1 | sequence-blind recall@1 | margin | Verdict |
|---|---|---|---|---|
| 0 | 0.1582 | 0.1706 | -0.0124 | kill |
| 1 | 0.1403 | 0.1843 | -0.0440 | kill |
| 2 | 0.1389 | 0.1623 | -0.0234 | kill |

The pre-registered rule requires the margin to exceed +0.10 in every seed to go, and
kills on a margin at or below +0.05 in any seed. All three margins are negative — the
control that never reads the steps outscores the objective meant to use them, in every
seed. Neither arm reaches the go floor of 0.40 recall@1 either; the best of the six
runs is 0.1843.

Full per-run numbers (including the E1 regression guard, which passed in every run —
this is not a broken pipeline, the objective simply did not learn the intended
structure) are in `results.json`, produced deterministically by `grade_e5.py` from the
six receipts in `receipts/`.

## Controls held

All six receipts carry the identical `corpus_fingerprint`, split manifest sha256, and
batch-order manifest sha256 (`results.json.control_check`), confirming the arms differ
only in the objective as required (G26) — not in data, split, or presentation order.
`code_revision.git_sha` in every receipt matches the worktree's HEAD at the time of
these runs, `dirty: false`.

## What this licenses or blocks

**Blocked**: the `reason` region's production objective should not move to this
JEPA-style latent-step regression next. The shortcut control was expected to be beaten
and was not — the objective is not extracting derivation structure from prior steps at
K=1 with this predictor.

**Left open**: the token-aware retrain direction that W1d's read-out probe result
licensed for the `reason` region tests a different question — whether the *existing*
contrastive objective benefits from a token-aware retrain — and is untouched by this
kill. That path remains available as the next matrix row for `reason`.

## Directory contents

- `PREREG.md` — the pre-registration, verbatim, with source citations.
- `grade_e5.py` — deterministic grader; reads the six receipts, computes recall@1 per
  arm/seed, seed spread, and the pre-registered decision; run twice, byte-identical
  output both times.
- `results.json` — the grader's output.
- `GO_KILL.md` — the verdict, the numbers, the rule quoted, and named deviations.
- `receipts/` — the six training receipts (JSON only; checkpoints are not copied here,
  they remain at `receipts/reason-e5-checkpoints/` in the worktree).
- `SHA256SUMS.json` — checksums of every file in this directory.
