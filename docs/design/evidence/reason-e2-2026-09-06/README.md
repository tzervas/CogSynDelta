# E2: reason region epoch-matched batch ablation — grading

**E2 did not execute as pre-registered.** Only the two reused seed-0 cells
exist; the other 10 required cells were never trained. The blocker is
procedural (a stale harness selection lock), not a training failure, and is
independent of the science this experiment is testing. No go/kill call can be
made on H2 or H6 from this evidence. This directory grades the lane's actual,
verified state — it does not fabricate or extrapolate the missing arms.

## What was asked, what exists

E2 (`docs/design/evidence/reason-region-diagnosis-2026-09-06/README.md:349-368`)
calls for 12 arms: batch ∈ {256, 512} × steps ∈ {2000, 4000} × seed ∈ {0, 1, 2},
each seed's four arms sharing one training seed and one held-out split. Two of
the twelve are pre-existing cells the spec names for reuse. The table below is
every arm's status, read from `RUN-MANIFEST.json` in this directory.

| batch | steps | seed | status | why |
|---|---|---|---|---|
| 256 | 4000 | 0 | present (reused) | existing cell, satisfies the spec's reuse instruction |
| 512 | 4000 | 0 | present (reused) | existing cell, satisfies the spec's reuse instruction |
| 256 | 4000 | 1 | blocked | seed-1 holdout retired by E0/G26; not a valid E2 split |
| 512 | 4000 | 1 | blocked | seed-1 holdout retired by E0/G26; not a valid E2 split |
| 256 | 4000 | 2 | blocked | matrix run never executed (selection lock, see below) |
| 512 | 4000 | 2 | blocked | matrix run never executed (selection lock, see below) |
| 256 | 2000 | 0 | blocked | matrix run never executed (selection lock, see below) |
| 256 | 2000 | 1 | blocked | matrix run never executed (selection lock, see below) |
| 256 | 2000 | 2 | blocked | matrix run never executed (selection lock, see below) |
| 512 | 2000 | 0 | blocked | matrix run never executed (selection lock, see below) |
| 512 | 2000 | 1 | blocked | matrix run never executed (selection lock, see below) |
| 512 | 2000 | 2 | blocked | matrix run never executed (selection lock, see below) |

**2 of 12 arms present.** This lane is instructed read-only on
`/akula-data/csd/matrix/` and did not train any new cells — cells there are
written only by the `model-matrix` harness, and that harness refused to run
because `/akula-data/csd/matrix/selection.json` carries a stale sticky
selection (`regions=["visual"]`) from a finished prior run. Overriding it
required either `--all` (which also admits every other pending cell in the
matrix, out of this lane's scope) or an operator edit to shared state, neither
of which this lane is authorized to do unilaterally.

## The two arms that do exist

| arm | batch | steps | final r@1 | final r@10 | final MRR | TF-IDF fraction (r@1 ÷ 0.873) | peak − final r@1 |
|---|---|---|---|---|---|---|---|
| reason-b256-s0 | 256 | 4000 | 0.1875 | 0.4199 | 0.2610 | 0.215 | 0.000 |
| reason-b512-s0 | 512 | 4000 | 0.1270 | 0.2676 | 0.1769 | 0.145 | 0.000 |

Both runs' recall@1 was still climbing at the final step (`peak − final = 0`),
so the best-checkpoint-retention gate (`PRE:1018`, fires at > 0.02 on ≥ 4 of 6
4,000-step runs) gets no signal from these two toward firing or not — 2 of the
required 6 runs is not enough to evaluate it either way.

The TF-IDF fraction column applies E1's demotion (below): read these two
numbers as "b256-s0 recovers 21.5% of the lexical ceiling, b512-s0 recovers
14.5%," not as a capability score on their own.

## H2 and H6: neither confirmed nor refuted

H2 predicts (512, 2000) ≈ (256, 4000) and both beat (512, 4000), *in every
seed*. H6 predicts (512, 2000) trails (256, 4000) by > 0.05 r@10, *with the
same sign in all three seeds*. Both are within-seed, four-arm comparisons.
Zero of the three required seeds have all four arms present — seed 0 is
missing its two 2,000-step arms, and seeds 1 and 2 are missing all four. Both
predictions grade **INSUFFICIENT_DATA**, not "not confirmed": there is no
partial signal to report, because the comparison the hypothesis names cannot
be assembled from what exists.

## What this licenses for the reason region

- **Nothing moves.** The matrix row does not switch to an epoch budget, the
  "b256 > b512" batch finding is not retired, and no temperature arm joins E3
  on this evidence — the go/kill in `GO_KILL.md` is `BLOCKED_INSUFFICIENT_DATA`,
  not a negative result.
- **E1's kill stands regardless.** Independent of E2's status, all three
  arms scored on the corrupted-derivation battery (b256-s0, b512-s0, and the
  untrained baseline) landed at or below 0.25 `derive.recall@1`
  (`docs/design/evidence/g48-reason-e1-2026-09-06/README.md`). Diagonal r@1 on
  reason stays demoted from a gate to a lexical-ceiling fraction — the table
  above already reports it that way — for both scored arms and for any future
  E2 arm once it exists.
- **The token-aware retrain is unaffected either way.** It was licensed by W1d
  (`docs/design/evidence/w1d-readout-probe-2026-09-02/`), not by E2; E2's batch
  ablation is a training-recipe question orthogonal to the token-vs-pooled
  representation question W1d settled. This grading changes neither its
  authorization nor its priority.
- **E5 is unaffected and still ahead of E3/E4** on the reasoning-as-not-lookup
  framing from E1 (iterative latent-step refinement, not a lookup
  improvement) — E2 was never a precondition for that ordering.
- **To make E2 mean something**, an operator needs to clear or widen
  `/akula-data/csd/matrix/selection.json`'s sticky `visual` selection (it is
  tracking metadata, safe to edit) or explicitly accept `--all`'s wider scope;
  either unblocks the 10 missing cells against the already-staged
  `program/matrix/csd-matrix.yaml` axis edits sitting uncommitted in the
  worktree. See `GO_KILL.md` → "To unblock" for both options.

## Files

- `RUN-MANIFEST.json` — the 12 required arms, their status, and receipt paths
  for the 2 that exist.
- `grade_e2.py` — deterministic grader; reads only the manifest and the
  receipts it names, prints the same JSON on every run.
- `results.json` — `grade_e2.py`'s output, run twice and diffed identical.
- `GO_KILL.md` — the pre-registered go/kill template filled with the actual
  (blocked) verdict.
- `SHA256SUMS.json` — checksums of every file in this directory.
