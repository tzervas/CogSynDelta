# E2 go/kill — reason region epoch-matched batch ablation

Template source: `docs/design/evidence/reason-region-diagnosis-2026-09-06/README.md:349-368`.
Filled from `results.json` (this directory), computed by `grade_e2.py`.

## Pre-registration (verbatim)

> **E2 — Epoch-matched batch ablation (~53 GPU-min)**
>
> *Change:* batch {256, 512} × steps {2,000, 4,000}, `eval_every` 200, on the fixed
> split. Within each training seed s ∈ {0, 1, 2} all four arms share s and differ
> only in (batch, steps). Existing (256, 4000, s0) and (512, 4000, s0) are reused;
> new: (256, 4000) and (512, 4000) at s1, s2 (2×5.2 + 2×10), (256, 2000) ×3
> (3×2.6), (512, 2000) ×3 (3×5.0).
>
> *Metric:* diagonal r@1/r@10/MRR at the final step and at the best eval point,
> plus E1's battery.
>
> *Predictions:*
> - H2: (512, 2000) ≈ (256, 4000) within 0.03 r@1 and 0.04 r@10 (same 87 epochs)
>   and both beat (512, 4000), in every seed.
> - H6: (512, 2000) trails (256, 4000) by > 0.05 r@10 with the same sign in all
>   three seeds.
>
> *Go/kill:* H2 confirmed means the matrix row moves to an epoch budget and
> "b256 > b512" is retired as a batch finding. H6 confirmed means a temperature
> arm (0.05 vs 0.10) joins E3. If peak − final > 0.02 r@1 in ≥ 4 of the 6
> 4,000-step runs, best-checkpoint retention is added (`PRE:1018`).

## H2

**INSUFFICIENT_DATA.** H2 requires all four (batch, steps) arms present at
every one of seeds {0, 1, 2}. Only seed 0's two 4,000-step arms exist; seed 0's
2,000-step arms and both seeds 1 and 2's four arms were never trained. 0 of 3
seeds are complete (`results.json: h2_h6.seeds_complete = []`). H2 is neither
confirmed nor refuted.

## H6

**INSUFFICIENT_DATA**, same cause as H2 — H6 is also a within-seed, all-three-arm
comparison and the (512, 2000) arm does not exist at any seed.

## Best-checkpoint retention gate (`PRE:1018`)

**Not evaluable.** 2 of the required 6 four-thousand-step runs exist. Both
available runs show `peak − final = 0.0` r@1 (recall@1 was still rising at the
final step in both, see README §Numbers), so the two data points that do exist
give no signal toward the gate either way.

## Decision

**BLOCKED_INSUFFICIENT_DATA.** Neither a go nor a kill call can be made under
the pre-registered protocol:

- The matrix row does **not** move to an epoch budget.
- The "b256 > b512" batch finding is **not** retired.
- No temperature arm is added to E3 on this evidence.
- Best-checkpoint retention is **not** added on this evidence.

The blocker is procedural, not scientific: `/akula-data/csd/matrix/selection.json`
carries a stale sticky selection (`regions=["visual"]`) from a finished prior
run, and `model-matrix plan/run --regions reason` refuses to proceed against it
without `--all` (which would additionally admit every other not-yet-done cell
across the whole matrix — out of this lane's scope) or an operator edit to the
selection file. This lane is instructed read-only on `/akula-data/csd/matrix/`
and did not attempt either.

**Independent of E2's blocked status**, E1's kill already stands: all three
scored arms (b256-s0, b512-s0, untrained baseline) landed at or below 0.25
`derive.recall@1` on the corrupted-derivation battery
(`docs/design/evidence/g48-reason-e1-2026-09-06/README.md`). Diagonal r@1 on
reason is demoted from a gate to a lexical-ceiling fraction for any future
reading of these two arms' numbers.

## To unblock

One of, operator's choice:

1. Clear or widen `/akula-data/csd/matrix/selection.json`'s sticky selection to
   include `reason` (it is tracking metadata, not a cell — safe to edit).
2. Explicitly authorize `--all` for this lane, accepting that it also admits
   every other pending cell in the matrix.

Either unblocks the 10 missing cells; the pre-registered `program/matrix/csd-matrix.yaml`
edits for the reason axes (batch/steps/seed exclusions for the two reused cells)
are already staged, uncommitted, in the worktree from the prior run attempt and
were not touched by this grading pass.
