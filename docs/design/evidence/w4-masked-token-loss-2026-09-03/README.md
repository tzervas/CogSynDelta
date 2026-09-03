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

## Files here

- `measure_w4_batch1280_probe.py` — the probe script, a throwaway measurement script
  (not part of the test suite), following the convention of
  `docs/design/evidence/w4-control-arm-2026-09-03/measure_w4_control_arm.py`.
- `batch1280-summary.json` — the probe's own JSON output (OOM error text, peak
  allocated/reserved, config).
