# Visual PTQ sensitivity (2026-09-06): the pooled probe is insensitive, not the encoder

**Verdict: the probe is insensitive.** Every linear-probe read-out tested against the visual
region's 3-to-8-bit PTQ ladder — the pooled EuroSAT probe, the Fashion-t10k transfer probe,
and a new pre-pool token-surface probe — stays flat within measurement noise across the whole
ladder. The encoder's actual output geometry does not: per-image cosine similarity to the fp32
latent falls from 0.999972 at 8-bit to 0.952436 at 3-bit, and top-10 nearest-neighbour identity
degrades from 99.6% agreement to 90.8%. The 3-bit floor the production plan chose is real
compression bought against a probe that cannot see the thing quantization is actually doing to
the representation — not evidence that the representation is undisturbed.

## Context

`docs/design/evidence/g22-visual-prereg-run2-2026-09-05/README.md` measured the seed-1 cell's
quantize stage choosing the 3-bit floor for all 25 target-encoder weight tensors, with a EuroSAT
pooled-probe drop of at most 0.0026 (well inside the 0.01 tolerance), and named the open
question directly: "Either the encoder is genuinely robust to 3-bit weights on this probe, or
the pooled linear probe is an insensitive quantization metric... check a more sensitive
read-out (token-surface probe, or the transfer set's drop) against the same plan." This
directory answers that by measurement.

## Method

Everything below reuses the production probe machinery — `_linear_probe`,
`_measure_visual_probes`, `_visual_latents`, `wrap_deployed_visual_encoder` (from
`scripts/csd-benchmark.py` / `src/cogsyndelta/regions/vl_pretrain.py`) — rather than
reimplementing "linear probe" a second time.

`src/` and `scripts/` are unmodified; the bits
ladder repacks the target encoder with `cogsyndelta.quant.ptq.quantize_tensor` /
`dequantize_tensor` via `apply_plan`, in memory, at a FIXED width per rung — no sensitivity
search, no new file written next to the checkpoint. Script:
[`measure_visual_ptq_sensitivity.py`](./measure_visual_ptq_sensitivity.py); full numeric output:
[`results.json`](./results.json).

| item | value |
|---|---|
| cell | `visual-b128-s1-3ce18db-20260905` (same seed-1 cell g22 run 2 graded) |
| checkpoint | `step-4000.pt`, sha256 `219328ea...` (matches the training receipt) |
| device | CUDA (RTX 3090 Ti, GPU 0), confirmed idle (`nvidia-smi --query-compute-apps` empty) before and during the run; wall time 270s, of which ~228s was GPU-bound battery time across 7 model variants — inside the 10-minute GPU budget |
| probe protocol | `probe_steps=600`, `probe_lr=0.001`, `seed=1` — identical to production; EuroSAT probe-train 20,000 images (`probe_limit` default, unmodified), probe-eval 5,400 (full, always), Fashion transfer 8,000/2,000 train/eval split |
| token-surface probe | new for this evidence: 16 patches sampled per image (of 256, `image_size=128 / patch_size=8`) from `ViTEncoder.tokens()` — the pre-pool, per-patch representation — labelled with their image's class, probed with the same `_linear_probe` fitting procedure as the pooled probe |
| geometry | per-image cosine similarity and top-10 nearest-neighbour agreement (`_topk_neighbor_mask`) between fp32 and each bit-width's pooled eval latents, over the same 5,400 EuroSAT probe-eval images the primary probe scores |
| measurement seed | `20260906` (patch sampling only; probe fitting reuses the production `seed=1`) |

## (1) Reproduction: fp32 vs the real packed 3-bit artifact

Confirms the harness before trusting anything new it measures.

| metric | fp32 (recomputed) | fp32 (receipt) | packed 3-bit (recomputed) | packed 3-bit (receipt) |
|---|---|---|---|---|
| `probe.top1` | 0.7237037 | 0.7237037 | 0.7242593 | 0.7242593 |

Exact bitwise match on both sides, and the packed artifact's sha256 (`51845d8b...`) matches the
`eval-quantized` receipt's recorded `quantized_sha256`. This harness reproduces production
exactly; the read-outs below extend it, they do not replace it.

A second internal check: repacking the same 25 tensors at 3-bit **in memory** via `apply_plan`
(the method the bits ladder in part (3) uses) reproduces the packed-artifact numbers bit for
bit — `probe.top1` 0.7242593 both ways, `transfer.top1` 0.8100001 both ways, token-surface
`top1` 0.6128820 both ways, identical geometry. The only difference is `stored_bytes`: 4,301,963
(the packed `.pt` file's actual size, including `torch.save`'s pickle/zip container framing) vs
4,265,472 (`apply_plan`'s pure tensor-payload count, matching the quant-plan receipt's own
`stored_bytes`) — a real and expected difference in what is being counted, not a discrepancy in
the weights.

## (2) Read-outs the quant plan never optimized for: fp32 vs packed 3-bit

| read-out | fp32 | packed 3-bit | Δ (3-bit − fp32) |
|---|---|---|---|
| EuroSAT `probe.top1` (primary, n=5,400) | 0.723704 | 0.724259 | +0.000556 |
| EuroSAT `probe.top5` | 0.979630 | 0.979259 | −0.000370 |
| Fashion-t10k `transfer.top1` (n=2,000) | 0.803000 | 0.810000 | +0.007000 |
| Fashion-t10k `transfer.top5` **(new — not in the eval-quantized receipt)** | 0.995500 | 0.998000 | +0.002500 |
| token-surface probe `top1` (n=86,400 patches) **(new)** | 0.610764 | 0.612882 | +0.002118 |
| token-surface probe `top5` **(new)** | 0.955359 | 0.955289 | −0.000070 |
| `repr.rep_std` (collapse signal) | 0.390928 | 0.401697 | +0.010769 |
| mean per-image cosine, fp32 vs 3-bit latent **(new)** | — | 0.952436 | (min 0.914011) |
| eval-latents std ratio (3-bit / fp32) **(new)** | — | 1.060542 | |
| top-10 NN rank agreement **(new)** | — | 0.908500 | (9.15% of neighbour identities changed) |

Every probe-style read-out (top1/top5, pooled or transfer or per-token) moves by a few tenths of
a point — the same order as the CUDA-nondeterminism noise floor `cogsyndelta-visual-eval-*`
already documents ("Smoke 24-step... train `held_out.top1` 0.6269 vs eval `probe.top1` 0.6215").
The geometry read-outs do not move by a "few tenths of a point" — 9.15% of the top-10 neighbour
set changed identity, and the *minimum* per-image cosine similarity (0.914) is an order of
magnitude further from 1 than any probe-metric delta above.

## (3) Bits ladder: same 25 tensors, fixed width, no sensitivity search

| bits | storage ratio | probe `top1` | transfer `top1` | token `top1` | `rep_std` | mean cosine vs fp32 | NN@10 agreement vs fp32 |
|---|---|---|---|---|---|---|---|
| fp32 | 1.00× | 0.723704 | 0.803000 | 0.610764 | 0.390928 | 1.000000 | 1.000000 |
| 3 | 10.05× | 0.724259 | 0.810000 | 0.612882 | 0.401697 | 0.952436 | 0.908500 |
| 4 | 7.65× | 0.725370 | 0.804500 | 0.611389 | 0.391357 | 0.990054 | 0.952074 |
| 5 | 6.18× | 0.723889 | 0.805500 | 0.610266 | 0.391701 | 0.997964 | 0.971333 |
| 6 | 5.18× | 0.722593 | 0.803000 | 0.610556 | 0.391096 | 0.999494 | 0.988722 |
| 8 | 3.91× | 0.724074 | 0.803500 | 0.610602 | 0.390878 | 0.999972 | 0.996259 |

Reading the columns left to right at fixed bit-width: the four task-probe columns (`probe.top1`,
`transfer.top1`, `token.top1`, `rep_std`) are flat across all six rows — max spread 0.0028 on
`probe.top1`, 0.0070 on `transfer.top1`, 0.0026 on `token.top1`, 0.0108 on `rep_std`, none of it
trending with bit-width (6-bit's `probe.top1` is lower than 3-bit's).

The two geometry columns
move monotonically and by an order of magnitude more: mean cosine climbs from 0.9524 (3-bit) to
0.999972 (8-bit) essentially on a saturating curve, and NN@10 agreement climbs from 0.9085 to
0.9963 over the same range. **The curve shows where each read-out starts to move: the geometry
read-outs start moving immediately below 8-bit and are still moving at 3-bit; none of the
task-probe read-outs move measurably anywhere on this ladder.**

## Verdict

**The probe is insensitive**, not the encoder. Restated against the task's own framing: the
production pooled EuroSAT probe shows no meaningful drop anywhere from 8-bit down to 3-bit —
but neither does the Fashion transfer probe, nor a token-surface probe built on the pre-pool
patch representations the plan never scored. All three are linear-probe read-outs, and all
three are blind to a real, monotonic, order-of-magnitude-larger disturbance in the encoder's
actual output geometry (cosine drift, neighbour-rank churn) that shows up the moment precision
drops below 8 bits.

The 3-bit floor's measured 0.0026 drop was never proof the weights survived
3-bit quantization intact — the geometry numbers above show plainly that they did not — it was
proof that a linear probe evaluated through 384-dimension mean pooling cannot see the direction
quantization noise moves the representation in.

A downstream consumer of these latents that is
NOT a mean-pooled linear classifier (nearest-neighbour retrieval, anything using patch tokens
directly, any component sensitive to representation geometry rather than a learned linear
decision boundary) should not assume 3-bit is safe on the strength of this probe.

## Files

- [`measure_visual_ptq_sensitivity.py`](./measure_visual_ptq_sensitivity.py) — the measurement
  script, run with `--device cuda` (production settings, no `--probe-limit` override).
- [`results.json`](./results.json) — full structured output: every read-out for every
  condition (`fp32`, `packed_artifact_3bit`, `ladder_3bit`..`ladder_8bit`), input receipt
  paths and sha256s, and the reproduction check.
- `SHA256SUMS.json` — sha256 of `README.md`, `measure_visual_ptq_sensitivity.py` and
  `results.json` (every other file in this directory).

Reproduce (cwd = repo root; GPU 0 must be idle per `nvidia-smi --query-compute-apps`, or pass
`--device cpu` / `CUDA_VISIBLE_DEVICES=""`):

```bash
uv run python docs/design/evidence/visual-ptq-sensitivity-2026-09-06/measure_visual_ptq_sensitivity.py \
    --device cuda --out docs/design/evidence/visual-ptq-sensitivity-2026-09-06/results.json
```
