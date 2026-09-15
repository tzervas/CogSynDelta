# g22 visual pre-registration, run 2 (2026-09-05): replication under the fixed pipeline, both seeds pass H1, all four stages real

**Result.** Both seeds pass H1 again under the merged pipeline fix, and the eval, quantize
and quantized-eval stages all ran and wrote receipts bound to the checkpoint. The eval stage
reproduced the training probe bitwise. The PTQ plan chose the 3-bit floor on every
target-encoder tensor with a probe drop inside budget; whether that is robustness or an
insensitive probe is filed as a follow-up, not claimed on the card.

## Why this run exists

Run 1 (`docs/design/evidence/g22-visual-prereg-run1-2026-09-05/`) trained under pin `198074a`
and passed H1 on both seeds, but its test stage failed structurally and its receipts carried no
checkpoint binding. This run repeats the experiment under the merged fix (#51, visual eval and
quantize stages; #52 pin bump) with the same corpus, config and seeds. It is graded from the
receipts in this directory, never from the run log.

## Run

| item | value |
|---|---|
| command | `model-matrix run program/matrix/csd-matrix.yaml --regions visual --allow-code-drift` (model-matrix main `8192fb5`) |
| cells | `visual-b128-s0-3ce18db-20260905`, `visual-b128-s1-3ce18db-20260905` |
| code | pin `run.code.sha` = `3ce18db` (the #51 merge); checkout main `6964c0a` (the #52 merge) |
| corpus | `visual-clean-v1` (Mix B), fingerprint `ab5761b65714e4ba4d7c36df095f3599`, 581,280 train images |
| config | W7v-cfg (note below) |
| masking | `random-permutation` |
| host | akula-prime, GPU 0 |

The table identifies the run. The only drift between the pin and the checkout is
`program/matrix/csd-matrix.yaml` (the pin line itself). W7v-cfg is 128 px / patch 8 /
dim 384 / depth 6 / 6 heads / predictor 192×3, at steps 4000, batch 128.

`code_revision.dirty` is `true` in both receipts for the same reason as run 1: the untracked,
operator-held directory `docs/design/evidence/g6-ternary-memory-gate-2026-09-04/` on main. No
tracked file differed from `6964c0a`.

## H1 (PREREG §1)

The criterion is the EuroSAT-test pooled linear probe against the threshold
`max(untrained, 0.1) + 0.01`, with `collapsed = false`, on both seeds.

| seed | untrained top1 | threshold | trained top1 | trained top5 | collapse ratio | collapsed | verdict |
|---|---|---|---|---|---|---|---|
| 0 | 0.6252 | 0.6352 | **0.7224** | 0.9809 | 0.8385 | false | PASS |
| 1 | 0.6337 | 0.6437 | **0.7237** | 0.9796 | 0.8705 | false | PASS |

The table shows the H1 inputs and verdict per seed, from the training receipts.

F1, F3, F4 not triggered on either seed (untrained baseline written before step 1,
`resumed = false`).

**H1: PASS on both seeds.**

| seed | trained top1, first run | trained top1, this run |
|---|---|---|
| 0 | 0.7191 | 0.7224 |
| 1 | 0.7298 | 0.7237 |

The table is the replication against run 1, per seed (same seeds, same corpus, same
config). CUDA training is not bitwise, so a few tenths of a point of spread is expected.

## Transfer

The transfer set is Fashion-MNIST t10k, n_eval 2000, named in the receipt as `fashion-t10k` /
`zalando/fashion-mnist`.

| seed | untrained | trained |
|---|---|---|
| 0 | 0.7100 | 0.8005 |
| 1 | 0.7080 | 0.8030 |

The table shows the transfer probe top1 before and after training, per seed.

## The three downstream stages, now real

| seed | eval `probe.top1` (fp32) | bitwise equal to train `held_out.top1` | quantize `within_budget` | fp32 → plan | eval-quantized `probe.top1` | storage ratio |
|---|---|---|---|---|---|---|
| 0 | 0.7224 | yes | yes | 0.7224 → 0.7198 (drop 0.0026) | 0.7198 | 10.05× |
| 1 | 0.7237 | yes | yes | 0.7237 → 0.7243 (drop −0.0006) | 0.7243 | 10.05× |

The table shows, per seed, the eval and quantize stage outcomes, the PTQ plan's probe against
fp32, the quantized-artifact re-measurement and the storage ratio.

- The eval stage re-measured the training receipt's numbers **bitwise** on both seeds (same probe
  protocol, seeded linear head, same collapse reduction), and bound the same checkpoint sha.
- Quantize packs the deployed EMA target encoder only (`target_encoder.*`, 10,712,448 params,
  42,849,792 fp32 bytes → 4,265,472 stored). The training-only context encoder and predictor
  are never packed. 27 probe evaluations per plan, about 6 minutes per cell.
- The `--quantized` eval loaded the packed artifact and reproduced the plan metric.

## Observation to carry forward (not a claim on the card)

The PTQ plan chose the 3-bit floor for all 25 target-encoder tensors on both seeds and the
EuroSAT probe moved by at most 0.0026, well inside the 0.01 budget. Either the encoder is
genuinely robust to 3-bit weights on this probe, or the pooled linear probe is an insensitive
quantization metric.

Before 3-bit is trusted as a shipped width, check a more sensitive read-out (token-surface
probe, or the transfer set's drop) against the same plan. This is filed as a follow-up. The
card reports the measured numbers with the storage-not-speed label.

## Files

`seed0/`, `seed1/`: training receipt, eval receipt, quant receipt, eval-quantized receipt,
program-run summary and `cell.json`, copied verbatim. `SHA256SUMS.json` covers every file.
