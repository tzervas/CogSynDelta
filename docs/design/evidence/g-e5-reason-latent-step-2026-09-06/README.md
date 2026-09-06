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

Both descriptions above are the arms' *training* objective only. The shared step
battery (`build_step_battery`) has no blind variant — it builds `question + steps[:t]`
unconditionally for every item — so both arms are *evaluated* on the identical sighted
battery; only the training context differs between them. E5 therefore compares two
training objectives on one sighted evaluation, not a sighted arm against a blind one.
See "Mechanism: what 'blind' actually means" below.

Three seeds (0, 1, 2), 4000 steps, batch size 256, matched to the `reason` `b256-s0`
baseline. Metric: 5-way step-battery `recall@1` (chance 0.20).

## Mechanism: what "blind" actually means

`enumerate_step_windows(..., blind=True)` (`reason_latent_step.py:299`) is what makes
the sequence-blind arm blind, and it only runs during *training* — it drops
`steps[:t]` from the context tokenized into that arm's model. It has no counterpart at
evaluation time: `build_step_battery` takes no `blind` argument and builds `context =
question + steps[:t]` unconditionally for every battery item
(`reason_latent_step.py:432`), and the trainer scores both arms' predictors against
that one battery (`csd-train-reason-e5.py:373`). So the sequence-blind arm is scored on
a context shape it never saw once during training. That is out-of-distribution for it,
which makes its win over the latent-step arm in every seed harder to produce than a
same-distribution comparison would — the kill is conservative, not an artifact of a
sighted arm beating a blind one on an unfair battery.

## Results

| Seed | untrained recall@1 | latent-step recall@1 | sequence-blind recall@1 | margin | Verdict |
|---|---|---|---|---|---|
| 0 | 0.1458 | 0.1582 | 0.1706 | -0.0124 | kill |
| 1 | 0.1224 | 0.1403 | 0.1843 | -0.0440 | kill |
| 2 | 0.0495 | 0.1389 | 0.1623 | -0.0234 | kill |

The pre-registered rule requires the margin to exceed +0.10 in every seed to go, and
kills on a margin at or below +0.05 in any seed. All three margins are negative — the
control that never reads the steps outscores the objective meant to use them, in every
seed. Neither arm reaches the go floor of 0.40 recall@1 either; the best of the six
runs is 0.1843.

`untrained recall@1` (`results.json.per_run.*.untrained_recall@1`) is each seed's
pre-training predictor score on the same battery — identical across arms within a seed
by construction (same `torch.manual_seed(seed)` model, scored before either arm trains)
and, notably, below the nominal chance figure of 0.20 in every seed. See "Battery
defect" below for why, and for what that means for how to read these numbers: both
arms score above their seed's own untrained null, and the sequence-blind control does
so by more in every seed (latent-step +0.0124 / +0.0179 / +0.0894 over the null;
sequence-blind +0.0248 / +0.0619 / +0.1128).

Full per-run numbers (including the E1 regression guard, which passed in every run —
this is not a broken pipeline, the objective simply did not learn the intended
structure) are in `results.json`, produced deterministically by `grade_e5.py` from the
six receipts in `receipts/`.

## Controls held

All six receipts carry the identical `corpus_fingerprint`, split manifest sha256, and
batch-order manifest sha256 (`results.json.control_check`), confirming the arms differ
only in the training objective's context window (G26) — not in data, split,
presentation order, or evaluation: both arms are scored on the identical sighted step
battery ("Mechanism" above). `code_revision.git_sha` in every receipt matches the
worktree's HEAD at the time of these runs, `dirty: false`.

## Battery defect

Both arms' below-chance untrained scores trace to how `build_step_battery` builds its
5-way candidate set, not to either objective. Full derivation, numbers, and the
instrument-solvability control the diagnosis mandates for E1 and this experiment omits
are in `BATTERY-DEFECT.md`. Summary: the true target is restricted to steps carrying a
calculator annotation, but distractors are drawn unfiltered from every step of every
other holdout item, so the two pools are not exchangeable (0% of true targets are a
`#### N` line; 22.5% of distractors are) — this alone drives the untrained null from
chance (0.20) down to 0.05–0.15, and distribution-matched distractors restore it to
0.19–0.22. Plain TF-IDF cosine scores recall@1 = 0.9367 on the as-shipped battery
(0.9821 with the near-duplicate corrupted-step candidate replaced), an unreported
lexical ceiling every model arm (0.14–0.18) sits far below.

This does not change the verdict above. Both arms are scored on the identical battery
with paired seeds, so the within-seed margin is unaffected by any battery-composition
artifact common to both candidate sets; all three margins clear the +0.05 kill
threshold by 0.062–0.094, roughly 3–5 binomial standard errors at n=727. Any successor
experiment that reuses `build_step_battery` inherits both defects (the non-exchangeable
distractor pool and the missing instrument-solvability control) and must fix them
first — that fix belongs in a follow-up, not in this evidence pack.

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
- `BATTERY-DEFECT.md` — the step battery's non-exchangeable distractor pool and missing
  instrument-solvability control: derivation, numbers, and why the verdict is unaffected.
- `receipts/` — the six training receipts (JSON only; checkpoints are not copied here,
  they remain at `receipts/reason-e5-checkpoints/` in the worktree).
- `SHA256SUMS.json` — checksums of every file in this directory.
