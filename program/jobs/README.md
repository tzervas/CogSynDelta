# CSD job specs for gpu-pack

`gpu-pack` (`tzervas/gpu-pack`, private) probes, budgets, and admits GPU jobs by VRAM.
CogSynDelta is a *client* of it, not a dependency — see
`tzervas/gpu-pack`'s `src/gpu_pack/spec.py:JobSpec` for the shape these files follow.
This directory holds example specs, not a runtime path CSD code reads itself.

## The CSD side of the protocol

`src/cogsyndelta/util/gpu_budget.py` is the thin adapter every entry point below calls
at CUDA init and again at the end of the run:

- `GPU_PACK_BUDGET_MIB` (int, MiB): if set and CUDA is available, caps this process's
  CUDA allocator via `torch.cuda.set_per_process_memory_fraction`. gpu-pack's launcher
  sets this from the ledger (or a spec's explicit `budget_mib`) before it starts the
  job; it is never set by these spec files themselves — a spec's `budget_mib: null`
  means "look it up in the ledger", not "no cap".
- `GPU_PACK_PROBE=1`: `regions/pretrain.py`'s `pretrain_region` caps `cfg.steps` to 20
  (`gpu_budget.PROBE_STEPS`) so a probe run measures peak VRAM in seconds instead of the
  full run. `scripts/csd-quantize.py` does not honour it — a quant job's `argv` already
  runs a single small ladder, so `JobSpec.probe` should stay `null` for quant specs (the
  launcher probes with the full argv).
- On exit, both `pretrain_region` and `csd-quantize.py`'s `main()` print
  `GPU_PACK_PEAK_MIB=<n>` to stderr once (skipped entirely on a CPU-only run) — this is
  what a probe launch reads back instead of trusting the driver's sampled figure alone.

## The specs here

| file | kind | region(s) | steps | batch | purpose |
|---|---|---|---|---|---|
| `train-code-tiny.json` | train | `code` | 30 | 128 | demo/probe-sized |
| `train-compress-tiny.json` | train | `compress` | 30 | 128 | demo/probe-sized |
| `quant-compress.json` | quantize | `compress` | — | — | quantizes the checkpoint the tiny compress run above just wrote (points its `--state` at `train-compress-tiny`'s state dir — run that spec first) |
| `train-memory-batch512.json` | train | `memory` | 8000 | 512 | **production-shaped**, not runnable on this branch yet |

Every spec's `max_len` is 64 for the tiny pair (fits the "suitable for the demo" 20-50
step / small-batch etiquette in the operator's GPU rules); `train-memory-batch512.json`
uses `max_len=96`, matching the measured config in
`git show feat/w4-memory-merge:src/cogsyndelta/regions/memory.py` (`MEASURED_VRAM_AT_BATCH_512`:
peak_reserved_mib=11678.0 at batch=512/max_len=96/dim=256/depth=4 on the 3090 Ti — a
50-step measurement, not this spec's full 8000-step run).

`state_dir` (gpu-pack's lock-file identity, one per job) is distinct from each job's own
`--state` argv flag (CSD's checkpoint/receipt root) — they happen to nest under the same
`/akula-data/csd/state-packs/<name>/` prefix here for readability, not because gpu-pack
requires it.

### TODO: `train-memory-batch512.json` has no budget

`budget_mib` is `null` (ledger lookup) and there is no ledger entry yet: the `memory`
region does not exist on this branch's `REGIONS` table in `scripts/csd-train-all.py` —
it lands with `feat/w4-memory-merge`. Until that merges *and* a probe run
(`GPU_PACK_PROBE=1`) records `train-memory-batch512`'s peak in gpu-pack's ledger, admission
must refuse this spec rather than guess; do not hand it an explicit `budget_mib` from the
50-step `MEASURED_VRAM_AT_BATCH_512` note above — that number is a different step count
than this spec's 8000 and has not been re-measured with `gpu_budget`'s cap wired in.

## Running one by hand (bypassing gpu-pack, for smoke-testing the adapter itself)

```bash
GPU_PACK_PROBE=1 GPU_PACK_BUDGET_MIB=4096 \
  /home/kang/code/personal/tzervas/CogSynDelta/.venv/bin/python \
  scripts/csd-train-all.py --regions code --steps 30 --batch 128 --max-len 64 \
  --state /tmp/csd-gpu-pack-smoke/state
```

Expect a `GPU_PACK_PEAK_MIB=<n>` line on stderr near the end (or none, on a CPU-only
box) and the run capped to `min(30, 20)=20` steps by the probe override.
