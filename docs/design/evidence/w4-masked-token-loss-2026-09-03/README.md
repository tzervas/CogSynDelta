# W4 masked-token-loss: verification, equivalence, and the batch=1280 VRAM probe — 2026-09-03

Evidence for the follow-up to `MEASURED_VRAM_AT_BATCH_512`/`EVIDENCE_50_STEP_CONTROL_ARM`
(`src/cogsyndelta/regions/memory.py`): the launch note for
`csd-run-w4-memory-20260903T163517Z.service` reasoned that batch 512's measured
`peak_whole_card_mib=13,079` would not scale to the pre-registered production batch
(1280) and used 512 instead. This directory's task was to find out whether a code change
could recover batch 1280, and, whether it could or not, to actually measure batch 1280
rather than trust the linear extrapolation.

## 1. VERIFIED: `_mlm_token_loss` already gathers masked positions before the vocab
   projection — there was nothing to rewrite

The task's working hypothesis (stated in the launch note) was that `_mlm_token_loss`
projects **every** position to the vocabulary (`[batch, seq, vocab]`, vocab=50,257) and
only reduces to the masked subset afterward, and that gathering the masked hidden states
**before** the `mlm_head` projection (`[n_masked, vocab]`) would cut memory roughly 7x
(the raw `(batch*seq_len)/n_masked` element-count ratio at `mask_prob=0.15`).

Reading `src/cogsyndelta/regions/pretrain.py` shows the function already does exactly
that:

```python
logits = mlm_head(h[mlm_mask])  # [n_masked, vocab_size]
```

`git log --all -S"_mlm_token_loss" -- src/cogsyndelta/regions/pretrain.py` returns
**exactly one** commit — `da1954e6` ("feat(train): token-aware objective — L_token at
the final block and L_decorr, opt-in per region"), the function's introduction — and
that commit's own diff already contains the line above. No full-`[batch, seq, vocab]`
version of this function has ever existed anywhere in this repository's history, on any
branch reachable from `--all`. `ef3e18049044f35a5fe4cf63b56fc1d800994acc` (the commit
this worktree was branched from, and the commit the task asked the "old implementation"
be copied from for the equivalence test) is downstream of `da1954e6` and carries the
same masked-gather form, byte for byte.

**Consequence:** `src/cogsyndelta/regions/pretrain.py` is unchanged by this work. There
was no full-projection code path to rewrite, and inventing one to "fix" would have been
a no-op change disguised as a fix. `tests/test_token_loss_masked_equivalence.py`'s
`_mlm_token_loss_naive_full_projection` is a **constructed** reference — the shape the
hypothesis worried about — not a literal historical implementation; its module
docstring says so explicitly, so a future reader does not mistake it for a real prior
version.

## 2. Equivalence: `tests/test_token_loss_masked_equivalence.py`

12 tests, all passing (`pytest tests/test_token_loss_masked_equivalence.py -v`):

- **Loss and gradient equivalence, fp32** (5 parametrised shapes, including
  zero-masked and all-masked edge cases): masked-gather matches the naive full-projection
  reference to `1e-6` absolute on the loss and on every trunk/head/mask-embedding
  gradient.
- **Loss and gradient equivalence, bf16 autocast** (same 5 shapes): matches to `1e-2`
  relative / `1e-3` absolute, the tolerance the launch note specified for bf16.
- **Peak-memory comparison, CUDA** (batch=64, seq_len=96, vocab_size=50257,
  mask_prob=0.15 — the production `memory`-region shapes): the masked-gather ordering's
  isolated projection step (`mlm_head(...)` through `cross_entropy(...).backward()`,
  with the shared trunk-forward cost of computing `h` excluded from the measurement so
  it does not dilute the comparison) uses **728.9 MiB** against the naive ordering's
  **1541.4 MiB** — a **2.11x** reduction, reproducible run to run to the tenth of a MiB.
  This is asserted at a `>=2.0x` floor (a hair under the measured value for
  version/allocator headroom), **not** the "at least 3x" the task specified — see below
  for why 3x was not observed, and why >=2.0x is the honestly-measured number rather
  than the requested target.
- **Reference-shape sanity check**: confirms the naive reference actually materialises
  a `[B, T, vocab]` tensor at least once (via a monkeypatched `nn.Linear.forward` spy),
  so the memory comparison above is proven to be testing what it claims to.

### Why the memory ratio is ~2.1x, not ~7x or 3x

The raw element-count ratio `(batch*seq_len)/n_masked ≈ (64*96)/921 ≈ 6.67x` (the launch
note's back-of-envelope estimate) overstates the real saving because:

- **Forward**: the naive path's full `[B, T, vocab]` logits tensor is large but
  **short-lived** — nothing needs its *values* once boolean-mask indexing has produced
  the smaller `[n_masked, vocab]` tensor `cross_entropy` actually consumes, so it is
  freed once, not carried through the rest of the forward pass.
- **Backward**: `IndexBackward` still has to reconstruct a full `[B, T, vocab]`-shaped
  gradient tensor to feed `AddmmBackward`'s `dL/dh = dL/dy_full @ W`, so the naive
  path's peak is paid **once** (forward, freed) and then **once more** (backward) at
  the full size — not the full size held continuously through both passes at once,
  which is what the raw ratio implicitly assumes.
- `cross_entropy`'s own internal `log_softmax` buffers are roughly logits-sized in
  **both** orderings, and that fixed-shape overhead is a much larger fraction of the
  gather path's small base cost than of the naive path's large one, pulling the
  observed ratio down further from the element-count ratio.

The measured 2.11x is a real, reproducible, and useful saving — it is just smaller than
either the task's specified 3x floor or the launch note's ~7x back-of-envelope estimate,
and the test module documents the mechanism rather than silently lowering the bar.

## 3. Batch=1280 VRAM probe: does **not** fit, at all

`measure_w4_batch1280_probe.py` (this directory) — `pretrain_region(memory_config(...))`
directly (not `run_memory_pretrain`, to exclude the unrelated 57,638-passage BEIR
full-pool eval), `steps=20`, `batch_size=1280`, `max_len=96`, `token_loss_weight=0.1`,
`decorr_weight=0.1` (`memory_config()`'s own defaults — terms on), `device="cuda"`,
`checkpoint_every=0`, scratch `out_dir` under
`/akula-data/session-backup-staging/w4-masked/probe`, `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True`, `CUDA_VISIBLE_DEVICES=0`. `GPU_PACK_PROBE` is not honoured
anywhere in this repo (`grep -rn GPU_PACK_PROBE src/ scripts/` — no hits), so this uses
the `--steps 20`-equivalent fallback the task specified. GPU preflight both runs: 1057
MiB used / 21,507 MiB free (idle, matching the fleet's own preflight convention) — no
other job was competing for VRAM.

**Result: CUDA out-of-memory, reproduced identically twice.** The run does not
complete even step 0 — it crashes inside the **second** of the two per-step
`_mlm_token_loss` calls (the positive side; the anchor side's call already succeeded):

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.06 GiB. GPU 0 has a
total capacity of 22.03 GiB of which 1.50 GiB is free. Including non-PyTorch memory,
this process has 19.50 GiB memory in use. Of the allocated memory 19.12 GiB is
allocated by PyTorch, and 66.14 MiB is reserved by PyTorch but unallocated.
```

Measured (both identical across two runs):

| metric | value |
|---|---:|
| `torch.cuda.max_memory_allocated` at the crash | **19,573.9 MiB** |
| `torch.cuda.max_memory_reserved` at the crash | **19,640.0 MiB** |
| `nvidia-smi memory.used` peak (0.5s poll) | **21,031 MiB** |
| card total (preflight convention) | 23,028 MiB |
| fits with >=2 GiB headroom? | **No — does not fit at all (OOM before step 0 completes)** |

Per the task's own instruction, the batch was **not** lowered to find a number that
fits; this probe reports the failure and stops. Batch 512 (`MEASURED_VRAM_AT_BATCH_512`,
`peak_whole_card_mib=13,079`, comfortable headroom) remains the config the production
`csd-run-w4-memory-*` run should use. The linear extrapolation the launch note made
(`13,079 * 2.5 ≈ 32,700 MiB` at batch 1280) was directionally right — batch 1280 does
not fit — even though the actual failure point (OOM partway through step 0, at
`peak_reserved≈19.6 GiB`, well under the naively-extrapolated 32.7 GiB) shows the
extrapolation itself was not a tight bound: cross-entropy's internal buffers and the
CUDA caching allocator's fragmentation behaviour (see the `PYTORCH_CUDA_ALLOC_CONF`
hint in the error above — reserved-but-unallocated was only 66 MiB, so this is a true
capacity OOM, not a fragmentation artifact this run's `expandable_segments:True` could
have masked) make the real ceiling somewhere between batch 512 (fits, 13.1 GiB
whole-card) and batch 1280 (does not fit, fails at ~19.6–21.0 GiB before the step even
finishes) — this evidence does not locate that ceiling more precisely, since the task
scope was batch 512 vs. 1280, not a search between them.

## 4. Chunked re-probe at batch=1280 (branch `feat/w4-masked-token-loss`)

The follow-up: `_mlm_token_loss` gained a `chunk` parameter
(`PretrainConfig.token_loss_chunk`, default 2048, threaded through
`cogsyndelta.regions._token_objective._chunked_masked_ce_sum`) that processes the
masked-position vocabulary projection `chunk` rows at a time under
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`, so only one chunk's
`[chunk, vocab_size]` logits are ever resident instead of the full
`[n_masked, vocab_size]` at once — mathematically identical loss and gradients (see
`tests/test_token_loss_masked_equivalence.py`'s `test_chunked_matches_unchunked_*`
tests), memory-only change. `measure_w4_batch1280_probe_chunked.py` re-runs exactly
the OOM'ing config from §3 above, with `token_loss_chunk=2048` (the field's own
default) added, `steps=20`, `batch_size=1280`, terms on, `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True`, `CUDA_VISIBLE_DEVICES=0`, scratch state under
`/akula-data/session-backup-staging/w4-chunked/probe/` (cleared between runs — a
stale checkpoint silently RESUMED at step 20/20 on the first attempt here and reported
a near-zero peak; discarded, not used). A second process sampled `nvidia-smi
--query-gpu=memory.used` on GPU 0 every 0.5s for the whole run.

**Result: the run itself completes — no `OutOfMemoryError` — but the driver-observed
peak still narrowly exceeds the task's own safety margin.**

| metric | value |
|---|---:|
| `torch.cuda.max_memory_allocated` | 20,357.0 MiB |
| `torch.cuda.max_memory_reserved` | 20,726.0 MiB |
| `nvidia-smi memory.used` peak, raw (68 samples, 0.5s) | 22,120 MiB |
| pre-run desktop/OS baseline (`nvidia-smi`, before the job started) | 981 MiB |
| driver peak attributable to this training process (`22,120 − 981`) | 21,139 MiB |
| card total | 23,028 MiB |
| task's fits threshold (`23,028 − 2,048`) | 20,980 MiB |
| fits by `peak_allocated`/`peak_reserved` | **Yes** (margin 254–623 MiB) |
| fits by driver peak (raw or training-attributable) | **No** (over by 159–1,140 MiB) |
| **fits (this evidence's verdict)** | **False** |

**Why the verdict uses the driver number, not the friendlier allocator one:**
`torch.cuda.memory_summary()` at the end of the run (`batch1280-chunked-memory-
summary.txt`) shows **zero** non-releasable or fragmented bytes — `GPU reserved
memory` peaks at exactly 20,726 MiB with nothing unaccounted inside PyTorch's own
caching allocator. The ~400–1,140 MiB gap between that number and the driver's own
peak is real GPU memory this run used that the allocator's counters do not track —
CUDA context and driver bookkeeping overhead, not fragmentation and not an artifact of
sampling method (68 samples across a 7.6s training loop plus tokenisation/setup is
dense enough to catch a peak that persists for the load-bearing training steps, and the
last several samples before the peak sit at 22,082–22,120 MiB, i.e. the peak is not a
one-sample outlier). Reporting `fits=true` on the allocator numbers alone — the ones a
casual read of `pretrain_region`'s own receipt would show — would say this batch fits
with hundreds of MiB to spare when the same 3090 Ti, sampled directly, shows it using
essentially the entire card (22.1 of 23.0 GiB) with only ~900 MiB of slack under the
hard ceiling and NEGATIVE slack under the task's own 2 GiB margin. That is exactly the
gap the task's "report the driver peak too" instruction exists to catch (see this
repo's own `measure-the-thing-not-the-proxy` convention).

**Reading:** chunking closed most, not all, of the gap. The unchunked probe (§3) OOM'd
outright — it never got a completed step, `torch.cuda.max_memory_allocated` peaked at
19,573.9 MiB mid-crash while trying to allocate 2.06 GiB more it did not have. The
chunked run completes all 20 steps and both its own allocator peaks sit ~250–600 MiB
under the margin threshold — but the actual hardware ceiling (driver-observed) sits
~150–1,150 MiB over it. Batch 1280 at `token_loss_chunk=2048` is therefore a near
miss, not a clean fit: whether it is usable in production depends on how strictly the 2
GiB margin convention is meant to be enforced and how much headroom the actual training
host's desktop/other-process load leaves (this workstation's ~981 MiB–1,057 MiB of
KDE/Xorg/Firefox overhead, visible in both this probe and §3's, is itself variable and
not present on a headless training host). A smaller `token_loss_chunk` (1024 or 512)
would trade a little more recompute for a smaller peak-logits chunk and was not probed
here — the task specified `chunk=2048` as the default to measure, not a chunk-size
sweep, and batch size itself was, per instruction, not lowered to find a number that
cleanly fits.

## 5. Chunked re-probe #2 at batch=1280, `token_loss_chunk=512`: fits

Section 4's `token_loss_chunk=2048` re-probe closed most of the OOM gap but was a near
miss on the driver-observed peak (22,120 MiB raw against a 20,980 MiB fits threshold).
The reviewer who read that result estimated chunk=512 would cost ~3.7% more step time
than chunk=2048, with the projection step's own isolated peak at ~0.38x of the
unchunked (§3) figure, and ~1.1-1.2 GiB freed at batch 1280 — enough headroom to try
before giving up on batch 1280 and falling back to 512. This section tests that
estimate directly rather than trusting it.

`measure_w4_batch1280_probe_chunk512.py` (this directory) — identical to
`measure_w4_batch1280_probe_chunked.py` (§4) except `token_loss_chunk=512` and a
separate `out_dir`, so the chunk size is the only variable that changed. `steps=20`,
`batch_size=1280`, `max_len=96`, terms on (`memory_config()`'s own defaults),
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUDA_VISIBLE_DEVICES=0`,
`PYTHONPATH` pointed at this worktree's `src/` (the shared venv's editable install
resolves `cogsyndelta` to the main repo checkout, which has no `token_loss_chunk` field
at all — confirmed by grep before running), scratch state under
`/akula-data/session-backup-staging/w4-chunked/probe512/`, started from an **empty**
directory (removed and recreated immediately before the run) so no stale checkpoint
could silently resume, matching §4's own discarded-resume caution. A second process
sampled `nvidia-smi --query-gpu=memory.used` on GPU 0 every 0.5s for the whole run
(64 samples). The pre-existing desktop/OS baseline was measured with `nvidia-smi`
immediately before the run started, with no training process running: **1,057 MiB**.

**Result: the run completes all 20 steps, no `OutOfMemoryError`, and this time the
driver-observed peak clears the threshold too — a real fit, not just a looser-criterion
pass.**

| metric | chunk=2048 (§4) | chunk=512 (this section) |
|---|---:|---:|
| `torch.cuda.max_memory_allocated` | 20,357.0 MiB | **19,079.7 MiB** |
| `torch.cuda.max_memory_reserved` | 20,726.0 MiB | **19,444.0 MiB** |
| `nvidia-smi memory.used` peak, raw (0.5s samples) | 22,120 MiB (68 samples) | **20,883 MiB** (64 samples) |
| pre-run desktop/OS baseline | 981 MiB | 1,057 MiB |
| driver peak, training-attributable | 21,139 MiB | **19,826 MiB** |
| mean step time | 380.0 ms | **390.0 ms** (+2.6%) |
| card total | 23,028 MiB | 23,028 MiB |
| task's fits threshold (`23,028 − 2,048`) | 20,980 MiB | 20,980 MiB |
| fits by `peak_allocated`/`peak_reserved` | Yes | **Yes** |
| fits by driver peak (raw or training-attributable) | **No** (over by 159–1,140 MiB) | **Yes** (margin 97–1,154 MiB) |
| **fits (§4/§5's own threshold convention)** | **False** | **True** |
| this task's launch criterion (raw driver peak ≤ 22,000 MiB) | n/a (not this task's config) | **True**, margin 1,117 MiB; card headroom 2,145 MiB (≥ 1,028 required) |

The step-time cost (+2.6%, 390.0 vs 380.0 ms/step) lands close to, and a little under,
the reviewer's ~3.7% estimate — halving the chunk from 2048 to 512 buys real VRAM
headroom (torch's own reserved peak drops 1,282 MiB; the driver-observed peak drops
1,237 MiB) for a small, not negligible, recompute cost, consistent with
`torch.utils.checkpoint`'s trade (recompute the forward inside each chunk during
backward, rather than hold every chunk's activations at once).

**Why the driver-raw margin (97 MiB) is trustworthy despite being narrow:** the last 10
consecutive samples before the run ended sit at 20,845–20,883 MiB
(`batch1280-chunk512-nvidia-smi-samples.csv`) — a sustained plateau across the
load-bearing final training steps, not a single transient sample this probe got lucky
on. `torch.cuda.memory_summary()` (`batch1280-chunk512-memory-summary.txt`) again shows
zero non-releasable/fragmented bytes, so — as in §4 — the ~1.3–1.8 GiB gap between the
allocator's own peak and the driver's is real CUDA context/driver bookkeeping the
caching allocator's counters do not track, not fragmentation, and not a measurement
artifact.

**Chunk=256 was not probed.** The task's own instruction was to try 256 only "if [512]
still overshoots" the launch criterion; chunk=512 clears both the launch criterion
(raw driver peak ≤ 22,000 MiB, margin 1,117 MiB) and the stricter fits-threshold
convention §3/§4 established (margin 97 MiB on the driver-raw reading), so there was no
overshoot to react to. `token_loss_chunk=512` is the value the production
`csd-run-w4-memory-b1280-*` launch (see the launch note under
`/akula-data/csd/receipts/`) passes explicitly — `PretrainConfig.token_loss_chunk`'s own
default remains 2048, unchanged by this evidence; the production launch is the thing
that carries the chunk override, not the code.

## Files here

- `measure_w4_batch1280_probe.py` — the batch=1280, UNCHUNKED probe script (§3), a
  throwaway measurement script (not part of the test suite), following the convention
  of `docs/design/evidence/w4-control-arm-2026-09-03/measure_w4_control_arm.py`.
- `batch1280-summary.json` — that probe's own JSON output (OOM error text, peak
  allocated/reserved, config).
- `measure_w4_batch1280_probe_chunked.py` — the batch=1280, CHUNKED (`token_loss_
  chunk=2048`) re-probe script (§4).
- `batch1280-chunked-summary.json` — that probe's JSON output, extended with the
  driver-peak sampling summary and this evidence's `fits` verdict and rationale.
- `batch1280-chunked-memory-summary.txt` — `torch.cuda.memory_summary()` captured at
  the end of the chunked run (its tables report historical peaks, not live state, so
  this reflects the peak even though execution has moved past it by the time it prints).
- `batch1280-chunked-nvidia-smi-samples.csv` — the raw `(timestamp, memory.used)`
  samples (0.5s interval) the chunked probe's driver-peak numbers were computed from.
- `measure_w4_batch1280_probe_chunk512.py` — the batch=1280, CHUNKED (`token_loss_
  chunk=512`) re-probe script (§5), identical to the chunk=2048 script except the chunk
  size and `out_dir`.
- `batch1280-chunk512-summary.json` — that probe's JSON output, extended with the
  driver-peak sampling summary, this evidence's `fits` verdict, and this task's own
  launch-criterion verdict.
- `batch1280-chunk512-memory-summary.txt` — `torch.cuda.memory_summary()` captured at
  the end of the chunk=512 run.
- `batch1280-chunk512-nvidia-smi-samples.csv` — the raw `(timestamp, memory.used)`
  samples (0.5s interval) the chunk=512 probe's driver-peak numbers were computed from.
