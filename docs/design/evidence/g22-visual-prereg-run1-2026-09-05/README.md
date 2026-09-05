# g22 visual pre-registration, run 1 (2026-09-05): both seeds pass H1

Pre-registration: `/akula-data/session-backup-staging/tools/grok-jobs/g22-visual-prereg/PREREG.md`
(Grok g22, locked before the run; H1 and the falsifiers F1–F4 are quoted below verbatim in
substance). Graded from the receipts in this directory, never from the run log.

## Run

| item | value |
|---|---|
| command | `model-matrix run program/matrix/csd-matrix.yaml --regions visual --allow-code-drift` (model-matrix main `8192fb5`) |
| cells | `visual-b128-s0-198074a-20260905`, `visual-b128-s1-198074a-20260905` under `/akula-data/csd/matrix/` |
| code | CogSynDelta main `d90d8b6` (pin `run.code.sha` = `198074a`, the visual-wiring merge; drift allowed = probe/guard refactor, matrix yaml, CI workflow; trainer unchanged) |
| corpus | `visual-clean-v1` (OD-4 Mix B), fingerprint `ab5761b65714e4ba4d7c36df095f3599`, 581,280 train images, largest share pxhere 0.2575 |
| config | W7v-cfg: 128 px / patch 8 / 256 patches, dim 384, depth 6, 6 heads, predictor 192×3, EMA 0.996→1.0, context_keep 0.4, 4 target blocks, sincos2d, steps 4000, batch 128, lr 1.5e-4 |
| masking | `random-permutation` (receipt field) |
| host | akula-prime, GPU 0, peak 4,590 MiB |

`code_revision.dirty` is `true` in both receipts. The only working-tree difference on main at
run time was the untracked directory `docs/design/evidence/g6-ternary-memory-gate-2026-09-04/`
(operator-held, never touched); no tracked file differed from `d90d8b6`.

## H1 (PREREG §1)

> On Mix B at W7v-cfg 128 px, two seeds, steps 4000, batch 128, the EMA-target pooled linear
> probe stamped as `held_out.top1` on EuroSAT test (n_eval 5400, 10-way, chance 0.1) exceeds
> `max(untrained_baseline.top1, 0.1) + 0.01` on **both** seeds, with `collapsed = false`.

| seed | untrained top1 | threshold | trained top1 | trained top5 | collapse ratio | collapsed | verdict |
|---|---|---|---|---|---|---|---|
| 0 | 0.6246 | 0.6346 | **0.7191** | 0.9809 | 0.8435 | false | PASS |
| 1 | 0.6391 | 0.6491 | **0.7298** | 0.9793 | 0.8652 | false | PASS |

Margin over the untrained probe: +0.094 (seed 0), +0.091 (seed 1). Margin over chance: +0.62.

Falsifiers (PREREG §4): F1 (`held_out.top1 ≤ threshold`, either seed) not triggered; F3
(`not_collapsed` false) not triggered; F4 (untrained block missing, or measured after training)
not triggered: both receipts carry `untrained_baseline` written before step 1, `resumed = false`.
F2 is not applicable to this run (no resume).

**H1: PASS on both seeds.**

## Transfer (recorded, not part of H1)

Transfer set: Fashion-MNIST official t10k, role `transfer` in the manifest (`fashion-t10k`),
`n_eval` 2000 (a 2,000-image subsample of the 10,000). The run log prints the label
`transfer (cifar100)`; that string is stale (`scripts/csd-train-all.py:1386`), CIFAR-100 was
killed in PREREG §3 and is not on disk. The receipts do not name the transfer source in the
`transfer` block; the identity above comes from the manifest's `probe_sets` and the shard the
run resolved (Grok g43 makes the receipt name it).

| seed | untrained transfer top1 | trained transfer top1 |
|---|---|---|
| 0 | 0.7100 | 0.8000 |
| 1 | 0.7060 | 0.8045 |

## What this does and does not show

- A 4,000-step I-JEPA run on the admitted clean mix produces an EMA-target representation
  whose pooled linear probe beats a random-init encoder's probe by about nine points on
  EuroSAT test, on two seeds, without collapse. That is all H1 claims.
- The untrained baseline is high (0.62–0.64): a linear probe over random 128-px patch
  features already separates EuroSAT's ten land-use classes well. The number that matters is
  the margin, and it is measured against that baseline, not against chance.
- `parameters` = 22,905,216 counts every parameter of the training module (context encoder,
  EMA target encoder and predictor). The deployable half is the EMA target encoder; its own
  count belongs on the eval receipt and the card, not here.
- No quantized numbers, no latency, no cross-faculty measurement (DEC-78: `visual_pairs: NOT
  MEASURED`).

## Harness outcome and follow-up

Stage states for both cells: `train: done`, `test: failed`, `quantize: skipped`,
`test-quant: skipped`. The test stage failed structurally, not on a gate:

```
KeyError: "no REGIONS entry for 'visual'; known regions: ['code', 'compress', 'language', 'memory', 'reason', 'retrieve']"
```

`scripts/csd-benchmark.py` and `scripts/csd-quantize.py` resolve regions through `REGIONS`;
the visual region lives in `VL_REGIONS`. model-matrix refused to read the exit code as a gate
decision because no eval receipt was written (correct behaviour). Fix in flight as Grok job
g43 (`tools/grok-jobs/g43-visual-eval/BRIEF.md`): receipt names its transfer set and binds
`checkpoint` + `checkpoint_sha256`; visual branches for the eval and quantize stages. These
two cells are not shimmed; after g43 merges the pin moves to the merge and the visual row is
re-run fresh (replication under the corrected receipt, clean code-drift check).

## Files

`seed0/`, `seed1/`: the training receipt (`visual-<ts>.json`), the program-run summary and
`cell.json` copied verbatim from the cell directories. `grade.json` is the machine grade;
`SHA256SUMS.json` covers every file here.
