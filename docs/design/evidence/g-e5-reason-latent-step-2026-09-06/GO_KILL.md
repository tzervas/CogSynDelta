# E5 go/kill verdict

## Rule (pre-registered, verbatim)

> Go: predictor acc@1 >= 0.40 AND >= sequence-blind + 0.10, in every seed.
> Kill: predictor acc@1 <= sequence-blind + 0.05, in any seed.

(`src/cogsyndelta/regions/reason_latent_step.py:97-99`; full context in `PREREG.md`.)

## Verdict: KILL

Every seed's margin (`latent-step recall@1 - sequence-blind recall@1`) is negative —
the sequence-blind control scores *higher* than the objective it was meant to beat, in
all three seeds. That alone satisfies the kill condition (margin <= 0.05) in every
seed, not merely "any seed" as the rule requires.

| Seed | latent-step recall@1 | sequence-blind recall@1 | margin | Verdict |
|---|---|---|---|---|
| 0 | 0.1582 | 0.1706 | -0.0124 | kill |
| 1 | 0.1403 | 0.1843 | -0.0440 | kill |
| 2 | 0.1389 | 0.1623 | -0.0234 | kill |

Neither arm reaches the go floor (`recall@1 >= 0.40`) either: the best score across all
six runs is 0.1843, less than half the floor.

## What this means

The latent-step predictor never earns its keep against a control that cannot see the
derivation steps at all. A predictor conditioned on `question + steps[:t]` should beat
one conditioned on `question` alone if it is using the steps for anything — instead it
loses by 0.01–0.04 recall@1 in every seed. The objective is not learning derivation
structure from the step context; whatever signal the battery is picking up (chance is
0.20, and both arms sit below chance) is coming from the question alone or from
battery-level artifacts, not from reading prior steps.

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
