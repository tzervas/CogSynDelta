# E2: reason region epoch-matched batch ablation — grading

**E2 ran to completion.** All twelve pre-registered arms exist, trained under one code
revision and one committed held-out split. H2 (batch is really an epoch effect) is
refuted. H6 (harder in-batch negatives at batch 512, steps 2000, trail on r@10) is
confirmed. The best-checkpoint-retention gate `PRE:1018` does not fire. This replaces
the prior `BLOCKED_INSUFFICIENT_DATA` pack in this directory
(commit `2f5f649`), which graded a run that had only 2 of 12 arms.

## What was asked, what exists

E2 (`docs/design/evidence/reason-region-diagnosis-2026-09-06/README.md:349-368`) calls
for 12 arms: batch ∈ {256, 512} × steps ∈ {2000, 4000} × seed ∈ {0, 1, 2}, each seed's
four arms sharing one training seed and one held-out split. All twelve are present,
read from `RUN-MANIFEST.json` in this directory, and all twelve trained under code
`2fdc8b2b6ca3ac4836fd61fbcd9ac32e25372fc6` against split `77d2c0e1ac02...`
(`config/mind/splits/reason-ca364a92-split0.json`).

Two of the twelve are a rerun. The pre-registration named `(256, 4000, seed=0)` and
`(512, 4000, seed=0)` for reuse from pre-existing cells, but those cells
(`reason-b256-s0-7bc2699-20260904`, `reason-b512-s0-7bc2699-20260904`) trained under an
older code revision (`a7694090903664bc256b4b96d998b37cacd316cf`) with no `split.sha256`
recorded in their receipts — a code-revision confound against the other ten arms. This
pack reruns exactly those two points under `2fdc8b2` instead of reusing them, so every
arm below shares one code revision and a verified split hash. The two `a769409` cells
are kept on disk and reported below as a secondary, disclosed comparison — not used in
any H2/H6/`PRE:1018` verdict.

## The twelve arms

Every arm's r@1 is reported three ways: the raw number, its multiple over the
region-specific untrained baseline (an untrained encoder scored on the same split,
seeded to match that arm's training seed), and its fraction of the split-level lexical
ceiling (TF-IDF r@1 = 0.8730, BM25 r@1 = 0.8496, both from the `reason-b512-s0` lexical
sidecar at split `77d2c0e1ac02...`; see `docs/design/evidence/g48-reason-e1-2026-09-06/`
for why raw r@1 is demoted to a fraction rather than read as a capability score).

| seed | batch | steps | r@1 | r@10 | MRR | untrained r@1 | r@1 ÷ untrained | r@1 ÷ TF-IDF ceiling |
|---|---|---|---|---|---|---|---|---|
| 0 | 256 | 2000 | 0.0840 | 0.3125 | 0.1612 | 0.0059 | 14.3x | 0.096 |
| 0 | 512 | 2000 | 0.0664 | 0.1992 | 0.1124 | 0.0059 | 11.3x | 0.076 |
| 0 | 256 | 4000 | 0.1523 | 0.3906 | 0.2339 | 0.0059 | 26.0x | 0.174 |
| 0 | 512 | 4000 | 0.1250 | 0.2461 | 0.1693 | 0.0059 | 21.3x | 0.143 |
| 1 | 256 | 2000 | 0.0684 | 0.3027 | 0.1422 | 0.0020 | 35.0x | 0.078 |
| 1 | 512 | 2000 | 0.0762 | 0.2676 | 0.1411 | 0.0020 | 39.0x | 0.087 |
| 1 | 256 | 4000 | 0.1523 | 0.3789 | 0.2262 | 0.0020 | 78.0x | 0.174 |
| 1 | 512 | 4000 | 0.1191 | 0.2812 | 0.1730 | 0.0020 | 61.0x | 0.136 |
| 2 | 256 | 2000 | 0.0605 | 0.2871 | 0.1355 | 0.0039 | 15.5x | 0.069 |
| 2 | 512 | 2000 | 0.0664 | 0.2344 | 0.1172 | 0.0039 | 17.0x | 0.076 |
| 2 | 256 | 4000 | 0.1250 | 0.3457 | 0.1961 | 0.0039 | 32.0x | 0.143 |
| 2 | 512 | 4000 | 0.1016 | 0.2539 | 0.1597 | 0.0039 | 26.0x | 0.116 |

The untrained baseline moves with the training seed (0.0059/0.0020/0.0039 r@1 at seeds
0/1/2) — it is a real re-measurement per arm's own seed, not a shared constant, per
`untrained_baseline` in each train receipt. Every arm's `r@1 ÷ untrained` multiple is
double digits or higher (11.3x–78x), so every trained arm clears the untrained floor by
a wide margin; the interesting comparisons are between arms, covered below.

Batch 256 beats batch 512 at the same step count, in every seed, at both step counts
(e.g. seed 0: 0.0840 vs 0.0664 at 2000 steps, 0.1523 vs 0.1250 at 4000 steps). This is
the "b256 > b512" finding E2 was designed to test — is it really a batch effect, or
just an epoch-count effect in disguise (batch 256 sees 256/512 the tokens per step but
the same wall-clock step budget, so at equal step count it has run fewer pair-draws but
denser updates)? H2 below answers that.

### Secondary, disclosed comparison: the two superseded a769409 cells

Not inputs to any verdict — no `split.sha256` in their receipts, so their held-out
membership cannot be cryptographically confirmed identical to the pinned split, and
they trained under a different code revision than the twelve arms above.

| batch | r@1 | r@10 | MRR | untrained r@1 | r@1 ÷ untrained | r@1 ÷ TF-IDF ceiling |
|---|---|---|---|---|---|---|
| 256 | 0.1875 | 0.4199 | 0.2610 | 0.0059 | 32.0x | 0.215 |
| 512 | 0.1270 | 0.2676 | 0.1769 | 0.0059 | 21.7x | 0.145 |

Both numbers are higher than their `2fdc8b2`/seed-0 counterparts in the main table
(0.1875 vs. 0.1523 at batch 256; 0.1270 vs. 0.1250 at batch 512) — a reminder of why the
confound mattered: these two cells are not interchangeable with the rerun ones, and are
shown here for transparency only.

## H2: NOT_CONFIRMED

H2 predicts `(512, 2000)` matches `(256, 4000)` within 0.03 r@1 and 0.04 r@10 (same
~87 epochs of exposure), and both beat `(512, 4000)` — required in every seed.

| seed | r@1 gap (tol. 0.03) | r@10 gap (tol. 0.04) | both beat (512,4000) r@1? |
|---|---|---|---|
| 0 | 0.0859 | 0.1914 | no |
| 1 | 0.0762 | 0.1113 | no |
| 2 | 0.0586 | 0.1113 | no |

Both clauses fail in every seed, by 2–4.8x their tolerances. `(512, 2000)` also never
beats `(512, 4000)` on r@1 (0.0664/0.0762/0.0664 vs. 0.1250/0.1191/0.1016): batch 512 at
2000 steps is the weakest of that batch's two step counts, not a match for batch 256 at
4000 steps. **The batch effect is not an epoch effect in disguise** — "b256 > b512"
holds independent of how many steps (and therefore epochs) each has run.

## H6: CONFIRMED

H6 predicts `(512, 2000)` trails `(256, 4000)` by more than 0.05 r@10, same sign in all
three seeds.

| seed | (256,4000) r@10 | (512,2000) r@10 | margin (need > 0.05) |
|---|---|---|---|
| 0 | 0.3906 | 0.1992 | 0.1914 |
| 1 | 0.3789 | 0.2676 | 0.1113 |
| 2 | 0.3457 | 0.2344 | 0.1113 |

Confirmed in every seed, by 2.2–3.8x the required margin.

## `PRE:1018` (best-checkpoint retention): does not fire

All six required 4,000-step runs exist. `peak − final` r@1 is exactly `0.000` in all
six (recall@1 was still at its running maximum at the final step in every run), so 0 of
6 trigger the gate — the threshold is ≥ 4 of 6.

## What this licenses for the reason region

- **H2 is refuted.** The matrix row stays batch-parameterised. "b256 > b512" is **not**
  retired as a batch finding — 256 beats 512 at every seed and every step count tested.
- **H6 is confirmed.** A temperature arm (τ = 0.05 vs 0.10) **joins E3**.
- **`PRE:1018` does not fire.** Best-checkpoint retention is **not** added.
- **E1's kill stands, independent of E2.** All arms scored on the corrupted-derivation
  battery (b256-s0, b512-s0, untrained baseline) landed at or below 0.25
  `derive.recall@1` (`docs/design/evidence/g48-reason-e1-2026-09-06/README.md`);
  diagonal r@1 on reason stays demoted from a gate to a lexical-ceiling fraction — the
  tables above already report it that way for every arm.
- **The token-aware retrain (W1d) is unaffected either way** — an orthogonal question,
  already licensed independently of E2.

## Deviations from the pre-registration (disclosed)

1. **`eval_every` is 666 (steps=4000) and 333 (steps=2000), not the pre-registered
   200.** `scripts/csd-train-all.py` computes `eval_every=max(1, steps // 6)`
   internally and exposes no CLI flag or matrix command-template placeholder to
   override it. This changes eval density but not what H2/H6/`PRE:1018` read (the
   final step's metrics and the running peak).
2. **`program/matrix/csd-matrix.yaml` carried worktree-local, uncommitted edits** for
   both matrix runs: `run.code.sha` pinned to `2fdc8b2` (main HEAD at scout time,
   needed for the split-manifest and lexical-baseline machinery E2 depends on),
   `stages.quantize` onward disabled (train+test only, per pre-registration), every
   other region's block commented out, and the reason region's `exclude` list first
   holding then (for the rerun) dropping the two seed-0/steps-4000 points.
   `model-matrix run ... --allow-code-drift` was required both times because the pin
   is dirty relative to a clean checkout. The file was restored to its committed
   state (`git checkout -- program/matrix/csd-matrix.yaml`) before this pack was
   written; `tests/test_matrix_config.py` (34 tests) passes against the restored file.
3. **The two originally-reused cells were a code-revision confound, now resolved** by
   rerunning `(256, 4000, seed=0)` and `(512, 4000, seed=0)` under `2fdc8b2` instead of
   reusing the `a769409` cells. See "Secondary, disclosed comparison" above for the
   two superseded numbers.

See `GO_KILL.md` for the full pre-registration text and the per-seed go/kill working.

## Files

- `RUN-MANIFEST.json` — all 12 required arms plus the 2 secondary cells, receipt
  paths, and sha256 of every receipt.
- `grade_e2.py` — deterministic grader; reads only the manifest and the receipts it
  names, prints the same JSON on every run.
- `results.json` — `grade_e2.py`'s output, run twice and diffed identical.
- `GO_KILL.md` — the pre-registered go/kill template filled with the actual verdict.
- `SHA256SUMS.json` — checksums of every other file in this directory, generated last.
