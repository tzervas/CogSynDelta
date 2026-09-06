# E2 go/kill — reason region epoch-matched batch ablation

Template source: `docs/design/evidence/reason-region-diagnosis-2026-09-06/README.md:349-368`.
Filled from `results.json` (this directory), computed by `grade_e2.py`. All twelve
required arms are present and graded; nothing here is `INSUFFICIENT_DATA`.

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

## H2 — NOT_CONFIRMED

Fails in all three seeds, on both clauses:

| seed | (512,2000) r@1 | (256,4000) r@1 | r@1 gap | tolerance | (512,2000) r@10 | (256,4000) r@10 | r@10 gap | tolerance | both beat (512,4000)? |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0664 | 0.1523 | 0.0859 | 0.03 | 0.1992 | 0.3906 | 0.1914 | 0.04 | no |
| 1 | 0.0762 | 0.1523 | 0.0762 | 0.03 | 0.2676 | 0.3789 | 0.1113 | 0.04 | no |
| 2 | 0.0664 | 0.1250 | 0.0586 | 0.03 | 0.2344 | 0.3457 | 0.1113 | 0.04 | no |

The r@1 gap is 2–2.9x the 0.03 tolerance and the r@10 gap is 2.8–4.8x the 0.04
tolerance in every seed — this is not a near miss. `beats(512,4000)` also fails in
every seed: `(512,2000)` never beats `(512,4000)` on r@1 (0.0664/0.0762/0.0664 vs.
0.1250/0.1191/0.1016), because `(512,2000)` is the weakest of the four batch=512 arms
regardless of the steps=4000 comparator. **H2 does not confirm.**

## H6 — CONFIRMED

`(256,4000)` beats `(512,2000)` on r@10 by more than 0.05 in every seed, same sign
throughout:

| seed | (256,4000) r@10 | (512,2000) r@10 | margin | threshold |
|---|---|---|---|---|
| 0 | 0.3906 | 0.1992 | 0.1914 | > 0.05 |
| 1 | 0.3789 | 0.2676 | 0.1113 | > 0.05 |
| 2 | 0.3457 | 0.2344 | 0.1113 | > 0.05 |

**H6 confirms.**

## Best-checkpoint retention gate (`PRE:1018`) — does not fire

All 6 required 4,000-step runs exist (evaluable). `peak − final` r@1 is exactly `0.0`
in all six — recall@1 was still at (or tied with) its running maximum at the final
step in every 4,000-step run, so 0 of 6 trigger the gate (threshold: ≥ 4 of 6).

| batch | seed | peak r@1 | final r@1 | peak − final |
|---|---|---|---|---|
| 256 | 0 | 0.1523 | 0.1523 | 0.000 |
| 512 | 0 | 0.1250 | 0.1250 | 0.000 |
| 256 | 1 | 0.1523 | 0.1523 | 0.000 |
| 512 | 1 | 0.1191 | 0.1191 | 0.000 |
| 256 | 2 | 0.1250 | 0.1250 | 0.000 |
| 512 | 2 | 0.1016 | 0.1016 | 0.000 |

**`PRE:1018` does not fire.**

## Decision

- **H2 is refuted, not confirmed.** The matrix row does **not** move to an epoch
  budget; "b256 > b512" is **not** retired as a batch finding — it holds at every
  seed and every step count tested here (256 beats 512 at both 2,000 and 4,000
  steps, all three seeds).
- **H6 is confirmed.** A temperature arm (τ = 0.05 vs 0.10) **joins E3**.
- **`PRE:1018` does not fire.** Best-checkpoint retention is **not** added on this
  evidence — every 4,000-step run's recall@1 was still rising (or flat) at the
  final step, not past its peak.
- These three findings are now direct, not blocked: all twelve arms share code
  `2fdc8b2` and the committed split manifest (`split.sha256`
  `77d2c0e1ac02...`), so every within-seed comparison H2/H6 needs is a clean
  apples-to-apples read, not a confounded one.

## Deviations from the pre-registration (disclosed)

1. **`eval_every` is 666 (steps=4000) and 333 (steps=2000), not the pre-registered
   200.** `scripts/csd-train-all.py` computes `eval_every=max(1, steps // 6)`
   internally for the reason region's call site and exposes no CLI flag or matrix
   command-template placeholder to override it — confirmed by grep over
   `program/matrix/csd-matrix.yaml` and `src/cogsyndelta/regions/pretrain.py`. At
   steps=4000 this samples 7 points before the run ends (0, 666, 1332, 1998, 2664,
   3330, 3996) plus a final-step check at 3999, versus ~21 pre-registered (every
   200). H2 and H6 both read only the final step's metrics, which this does not
   change. `PRE:1018` is different: it reads `peak − final`, and a coarser sample is
   less likely to land on a run's true interior peak, so this weakens that gate's
   instrument, not merely its cadence. The available evidence argues the coarser
   sampling likely did not flip the call here: two of the twelve 4,000-step arms are
   already non-monotone in r@1 at this same coarse sampling —
   `reason-b512-st4000-s1-2fdc8b2-20260906` dips 0.0059 between steps 2664→3330,
   `reason-b512-st4000-s2-2fdc8b2-20260906` dips 0.0059 between steps 1998→2664 —
   real swings an order of magnitude under the 0.02 gate threshold, even caught by
   only 7–8 samples. That is evidence, not proof: a densely-sampled rerun could still
   surface a larger interior peak these samples missed.
2. **`run.code.sha` and the reason `axes`/`exclude` block were edited worktree-local,
   uncommitted, in `program/matrix/csd-matrix.yaml`** for the duration of both matrix
   runs (the original 10-cell run and this pack's 2-cell rerun): `sha` pinned to
   `2fdc8b2b6ca3ac4836fd61fbcd9ac32e25372fc6` (main HEAD at scout time, needed for the
   split-manifest and lexical-baseline machinery E2 depends on), `stages.quantize`
   forward disabled (train+test only, per pre-registration), and every other region's
   block commented out so only `reason` ran. `model-matrix run ... --allow-code-drift`
   was required because that pin is dirty relative to a clean checkout. The file was
   restored to its committed state (`git checkout -- program/matrix/csd-matrix.yaml`)
   before this pack was written; `tests/test_matrix_config.py` (34 tests) passes
   against the restored file.
3. **The two originally-reused seed-0/steps=4000 cells were a code-revision
   confound, now resolved.** `reason-b256-s0-7bc2699-20260904` and
   `reason-b512-s0-7bc2699-20260904` ran under code
   `a7694090903664bc256b4b96d998b37cacd316cf` with no `split.sha256` in their
   receipts (pre-dating the split-manifest machinery), while the other ten arms ran
   under `2fdc8b2`. This pack reruns exactly those two `(batch, steps=4000, seed=0)`
   points under `2fdc8b2` (`reason-b256-st4000-s0-2fdc8b2-20260906`,
   `reason-b512-st4000-s0-2fdc8b2-20260906`), so all twelve arms used for
   H2/H6/`PRE:1018` now share one code revision and the committed split manifest.
   The two `a769409` cells are kept on disk and reported in `RUN-MANIFEST.json` and
   `README.md` as a secondary, disclosed comparison only — they are not inputs to
   any verdict above.
4. **Learning rate covaries with batch; E2 does not hold it fixed.**
   `scripts/csd-train-all.py`'s `lr_for_batch` (`:284-300`, using `BASE_BATCH = 256`
   and `BASE_LR = 3e-4` at `:246-247`) sets `lr = 3e-4 * sqrt(batch / 256)`: every
   batch=256 arm trains at lr=3.0000e-4 and every batch=512 arm at lr=4.2426e-4 (both
   exact, in every arm's `config.lr`). `warmup_steps = max(50, steps // 15)`
   (`:1004`, `:1475`) is 133 at steps=2000 and 266 at steps=4000, so warmup length
   moves with steps, independent of batch. No arm in this matrix varies batch at a
   fixed learning rate, so H6's r@10 margin and H2's refutation ("b256 > b512") are a
   joint batch-and-lr effect, not an isolated batch effect — "harder in-batch
   negatives at larger batch" (H6's stated mechanism,
   `docs/design/evidence/reason-region-diagnosis-2026-09-06/README.md:250`) is one of
   two live explanations, the other being a learning rate too high for batch 512 at
   this corpus size. A follow-up arm at batch=512, lr=3.0e-4 (held fixed at the
   batch=256 value), same three seeds and the same split, would disambiguate the two.

## What this licenses for the reason region

- **E3 still gains a temperature arm** (τ = 0.05 vs 0.10), per H6 — but see deviation
  4: batch and lr covary in every arm here, so "harder in-batch negatives at larger
  batch" is one of two live explanations for H6's margin, not a confirmed mechanism.
  The disambiguating follow-up (batch=512 at lr=3.0e-4, same seeds and split) should
  run before or alongside the temperature arm.
- **The matrix row stays batch-parameterised**, not epoch-parameterised — H2's
  refutation means "epochs, not batch" is not a safe simplification for this
  region's matrix axis.
- **E1's kill (recall demoted to a lexical-ceiling fraction) is unaffected and
  stands independently** (`docs/design/evidence/g48-reason-e1-2026-09-06/README.md`);
  every arm's r@1 in `README.md`'s table is reported as both a raw number and a
  fraction of the 0.873 TF-IDF ceiling for exactly that reason.
- **The token-aware retrain (W1d) is unaffected either way** — an orthogonal
  question, already licensed independently of E2.
