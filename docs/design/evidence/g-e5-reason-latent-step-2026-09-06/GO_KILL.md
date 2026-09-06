# E5 go/kill verdict

## Rule (pre-registered, verbatim)

> Go: predictor acc@1 >= 0.40 AND >= sequence-blind + 0.10, in every seed.
> Kill: predictor acc@1 <= sequence-blind + 0.05, in any seed.

(`src/cogsyndelta/regions/reason_latent_step.py:101-105`; full context in `PREREG.md`.)

## Verdict: KILL

Every seed's margin (`latent-step recall@1 - sequence-blind recall@1`) is negative —
the sequence-blind control scores *higher* than the objective it was meant to beat, in
all three seeds. That alone satisfies the kill condition (margin <= 0.05) in every
seed, not merely "any seed" as the rule requires.

| Seed | untrained recall@1 | latent-step recall@1 | sequence-blind recall@1 | margin | Verdict |
|---|---|---|---|---|---|
| 0 | 0.1458 | 0.1582 | 0.1706 | -0.0124 | kill |
| 1 | 0.1224 | 0.1403 | 0.1843 | -0.0440 | kill |
| 2 | 0.0495 | 0.1389 | 0.1623 | -0.0234 | kill |

`untrained recall@1` is the seed's pre-training predictor score on the same battery
(`results.json.per_run.*.untrained_recall@1`); margin is still `latent-step -
sequence-blind`, unaffected by this column. See "What this means" below for why the
untrained column, not the nominal chance figure, is the honest null for this battery.

Neither arm reaches the go floor (`recall@1 >= 0.40`) either: the best score across all
six runs is 0.1843, less than half the floor.

## What this means

The sequence-blind control is blind only during *training*: `enumerate_step_windows(...,
blind=True)` (`reason_latent_step.py:299`) drops `steps[:t]` from its training context,
verified directly (perturbing the prior-step text changes the latent-step arm's training
loss and predicted latent, and leaves the sequence-blind arm's bit-for-bit unchanged).
`build_step_battery` has no `blind` parameter at all — it builds `context = question +
steps[:t]` unconditionally (`reason_latent_step.py:432`) — so at evaluation time both
arms are scored on the identical sighted battery. E5 therefore compares two *training*
objectives on one sighted evaluation, not a sighted arm against a blind one. That makes
the sequence-blind control's evaluation out-of-distribution relative to what it was
trained on (it has never seen a step-bearing context before being scored on one), which
makes its win over the latent-step arm harder to produce, not easier — the kill is
conservative, not an artifact of an unfair comparison.

The latent-step predictor still never earns its keep: a predictor trained to use
`question + steps[:t]` should beat one trained to ignore the steps, when both are then
scored on the same sighted context, if it is extracting anything from the steps.
Instead it loses by 0.01–0.04 recall@1 in every seed.

Chance on this battery is 0.20, but the untrained (pre-training) predictor already
scores below chance in every seed — 0.1458 / 0.1224 / 0.0495 for seeds 0/1/2. That is a
battery-composition artifact, not a property of either objective (`BATTERY-DEFECT.md`
traces it to the battery's distractor pool being drawn from a different distribution
than the true-target slot). Measured against that seed's own untrained null instead of
the nominal chance figure, both arms learn something, and the control learns more, in
every seed: latent-step +0.0124 / +0.0179 / +0.0894 over its seed's null, sequence-blind
+0.0248 / +0.0619 / +0.1128. The objective is not learning derivation structure from the
step context; whatever signal the battery is picking up over the null is coming from the
question alone, or from battery-level artifacts, not from reading prior steps.

## Deviations from the pre-registration, named explicitly

- **None in method.** All six runs used the pre-registered seeds `{0, 1, 2}`, both
  pre-registered arms, 4000 steps, batch size 256, and the same split/order manifest
  (confirmed identical `corpus_fingerprint`, `split.sha256`, `batch_order.sha256`
  across all six receipts — see `results.json.control_check`).
- **Wall time overran the pre-registration's ~11 GPU-min/run estimate.** Actual times
  ranged 599.3s–2120.3s (10.0–35.3 min); total across all 6 runs was ~1.84 GPU-hours
  against a ~1.17 GPU-hour budget (~1.6x). This does not affect the verdict — it is a
  scheduling/cost note, not a methods deviation.
- **The go/kill comparison itself is a deliberate one-time exception to how the
  training script works.** `scripts/csd-train-reason-e5.py` never applies the rule
  itself (each run only has one arm of one seed); its docstring names the cross-run
  comparison as future work. `grade_e5.py` in this directory is that future work,
  reading the six receipts the trainer already wrote and applying the same rule
  (`go_kill_note`, mirrored constant-for-constant) the module itself defines. No new
  metric was invented; no threshold was changed.
- **The untrained baseline ran at the run seed, not the pre-registered
  region-specific seed.** `PREREG_CONTROLS` (`reason_latent_step.py:164-171`) calls for
  "an untrained predictor at the region-specific seed" — `w2c_region_seed("reason")` =
  2773680709 (`reason_latent_step.py:184-198`). `csd-train-reason-e5.py:543` instead
  records `"untrained_baseline_seed": args.seed`, so the untrained model in every
  receipt is seeded 0, 1, or 2 (whatever that run's training seed is); `w2c_region_seed`
  is never called anywhere in the trainer (`grep -c` over the file returns 0). This does
  not touch the graded rule, which compares only the two trained arms, but the untrained
  numbers quoted above are per-run-seed, not region-specific-seed, numbers.
- **The sequence-blind control's evaluation is sighted, not blind.** See "What this
  means" above: training-time blindness is real and verified; evaluation-time
  blindness was never implemented (`build_step_battery` takes no `blind` argument),
  and neither this document nor `README.md` said so before this revision.

## What E5 licenses / blocks

**Blocked**: this objective, at K=1, with this predictor and this step battery, does
not license retraining the production `reason` region on latent-step prediction. The
next matrix row for the reason region's production objective should not build on this
result — the shortcut control was expected to be beaten and was not.

**Not addressed by this run**: whether a token-aware retrain of the `reason` region's
existing objective (the direction W1d's read-out probe result licensed) would fare
differently. E5 tested a different, JEPA-style regression objective, not a
token-aware version of the current contrastive one — this kill does not speak to that
open question one way or the other; it remains open.
