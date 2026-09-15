# g22 visual pre-registration, run 1 (2026-09-05): both seeds pass H1

**Result.** Both seeds pass H1 of the g22 pre-registration: the trained EuroSAT probe
clears the untrained-plus-margin threshold without collapse. The harness test stage failed
structurally (no eval receipt), so these cells carry no checkpoint binding and are not
shimmed. The fix is Grok job g43 and the visual row is re-run after it merges.

## Provenance of the grade

The pre-registration is `/akula-data/session-backup-staging/tools/grok-jobs/g22-visual-prereg/PREREG.md`
(Grok g22, locked before the run). H1 and the falsifiers F1–F4 are quoted below verbatim in
substance. The grade is taken from the receipts in this directory, never from the run log.

## Run

| item | value |
|---|---|
| command | `model-matrix run program/matrix/csd-matrix.yaml --regions visual --allow-code-drift` (model-matrix main `8192fb5`) |
| cells | `visual-b128-s0-198074a-20260905`, `visual-b128-s1-198074a-20260905` under `/akula-data/csd/matrix/` |
| code | CogSynDelta main `d90d8b6`, pin `run.code.sha` = `198074a` (note below) |
| corpus | `visual-clean-v1` (OD-4 Mix B), fingerprint `ab5761b65714e4ba4d7c36df095f3599` |
| config | W7v-cfg (table below) |
| masking | `random-permutation` (receipt field) |
| host | akula-prime, GPU 0, peak 4,590 MiB |

The table identifies the run. The pin `198074a` is the visual-wiring merge. The drift
allowed between the pin and `d90d8b6` is the probe/guard refactor, the matrix yaml and the
CI workflow; the trainer is unchanged. The corpus has 581,280 train images, largest share
pxhere 0.2575.

| parameter | value |
|---|---|
| image size | 128 px |
| patching | patch 8 / 256 patches |
| encoder | dim 384, depth 6, 6 heads |
| predictor | 192×3 |
| EMA | 0.996→1.0 |
| context_keep | 0.4 |
| targets | 4 target blocks, sincos2d |
| schedule | steps 4000, batch 128, lr 1.5e-4 |

The table lists the W7v-cfg training configuration shared by both seeds.

`code_revision.dirty` is `true` in both receipts. The only working-tree difference on main at
run time was the untracked directory `docs/design/evidence/g6-ternary-memory-gate-2026-09-04/`
(operator-held, never touched). No tracked file differed from `d90d8b6`.

## H1 (PREREG §1)

> On Mix B at W7v-cfg 128 px, two seeds, steps 4000, batch 128, the EMA-target pooled linear
> probe stamped as `held_out.top1` on EuroSAT test (n_eval 5400, 10-way, chance 0.1) exceeds
> `max(untrained_baseline.top1, 0.1) + 0.01` on **both** seeds, with `collapsed = false`.

| seed | untrained top1 | threshold | trained top1 | trained top5 | collapse ratio | collapsed | verdict |
|---|---|---|---|---|---|---|---|
| 0 | 0.6246 | 0.6346 | **0.7191** | 0.9809 | 0.8435 | false | PASS |
| 1 | 0.6391 | 0.6491 | **0.7298** | 0.9793 | 0.8652 | false | PASS |

The table shows the H1 inputs and verdict per seed, from the training receipts.

| seed | margin over the untrained probe |
|---|---|
| 0 | +0.094 |
| 1 | +0.091 |

The table shows the trained probe's margin over the untrained probe, per seed. The margin
over chance is +0.62 on both seeds.

**H1: PASS on both seeds.**

### Falsifiers (PREREG §4)

- F1 (`held_out.top1 ≤ threshold`, either seed): not triggered.
- F2: not applicable to this run (no resume).
- F3 (`not_collapsed` false): not triggered.
- F4 (untrained block missing, or measured after training): not triggered. Both receipts
  carry `untrained_baseline` written before step 1, `resumed = false`.

## Transfer (recorded, not part of H1)

The transfer set is Fashion-MNIST official t10k, role `transfer` in the manifest
(`fashion-t10k`), `n_eval` 2000 (a 2,000-image subsample of the 10,000). The run log prints
the label `transfer (cifar100)`. That string is stale: CIFAR-100 was killed in PREREG §3 and
is not on disk (`scripts/csd-train-all.py:1386`).

The receipts do not name the transfer source in the `transfer` block. The identity above
comes from the manifest's `probe_sets` and the shard the run resolved. Grok g43 makes the
receipt name it.

| seed | untrained transfer top1 | trained transfer top1 |
|---|---|---|
| 0 | 0.7100 | 0.8000 |
| 1 | 0.7060 | 0.8045 |

The table shows the Fashion-MNIST transfer probe before and after training, per seed.

## What this does and does not show

- A 4,000-step I-JEPA run on the admitted clean mix produces an EMA-target representation
  whose pooled linear probe beats a random-init encoder's probe by about nine points on
  EuroSAT test, on two seeds, without collapse. That is all H1 claims.
- The untrained baseline is high (0.62–0.64). A linear probe over random 128-px patch
  features already separates EuroSAT's ten land-use classes well, so the number that matters
  is the margin, measured against that baseline and not against chance.
- `parameters` = 22,905,216 counts every parameter of the training module (context encoder,
  EMA target encoder and predictor). The deployable half is the EMA target encoder. Its own
  count belongs on the eval receipt and the card, not here.
- No quantized numbers, no latency, no cross-faculty measurement (DEC-78: `visual_pairs: NOT
  MEASURED`).

## Harness outcome and follow-up

Stage states for both cells: `train: done`, `test: failed`, `quantize: skipped`,
`test-quant: skipped`. The test stage failed structurally, not on a gate:

```
KeyError: "no REGIONS entry for 'visual'; known regions: ['code', 'compress', 'language', 'memory', 'reason', 'retrieve']"
```

`scripts/csd-benchmark.py` and `scripts/csd-quantize.py` resolve regions through `REGIONS`,
while the visual region lives in `VL_REGIONS`. model-matrix refused to read the exit code as
a gate decision because no eval receipt was written, which is correct behaviour.

The fix is in flight as Grok job g43 (`tools/grok-jobs/g43-visual-eval/BRIEF.md`). It makes
the receipt name its transfer set and bind `checkpoint` + `checkpoint_sha256`, and adds visual
branches for the eval and quantize stages. These two cells are not shimmed. After g43 merges,
the pin moves to the merge and the visual row is re-run fresh: a replication under the
corrected receipt, with a clean code-drift check.

## Files

`seed0/`, `seed1/`: the training receipt (`visual-<ts>.json`), the program-run summary and
`cell.json` copied verbatim from the cell directories. `grade.json` is the machine grade.
`SHA256SUMS.json` covers every file here.
