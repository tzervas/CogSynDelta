# ADR-0015: LAN GPU inference pool (3090 Ti + 5080)

**Status**: Accepted (Stage A placement + migrate). Stage B RPC split is
specified, not enabled.
**Date**: 2026-08-31
**Decision Makers**: tzervas
**Technical Story**: Lab autodev must schedule helpers without OOM, and later
run GGUFs that need ~24 GiB + ~16 GiB ≈ 40 GiB combined.

## Context

The lab has two mismatched NVIDIA cards on two hosts:

| Host | GPU | Arch / SM | VRAM | Strength | Weakness |
|---|---|---|---|---|---|
| akula-prime | RTX 3090 Ti | Ampere sm_86 | ~23028 MiB | High VRAM, LocalAI GGUF, tight decode | No native FP8/FP4, not sm_120 |
| gpu5080 | RTX 5080 | Blackwell sm_120 | ~16303 MiB | CUDA/Triton/eval, FP8/FP4, GPU CI | Less VRAM, no LocalAI GGUF today |

Treating them as one interchangeable pool OOMs or mis-schedules: a 14B 32k
GGUF does not fit the 5080; an sm_120 kernel does not run on the 3090 Ti.
`G-SPLIT` (one CogSynDelta mind sharded across cards) stays **blocked** until
Phase 3 one-GPU Python proof. This ADR is **inference/helper placement**, not
CSD training split.

## Decision

The system will:

1. Tag **hosts** with capability fields (`caps`, `lacks`, `arch`, `sm`,
   `safeguard_mib`) and **aliases** with `needs_caps`, `prefer_caps`,
   `latency` (`tight` | `ok-lan`), `fits_5080` / `fits_prime`, `pool_ok`.
2. **Stage A (now):** whole-model placement. Autodev `local/code` is sticky
   on the 3090. Small `fits_5080` + `ok-lan` models migrate to the 5080 when
   a tight 14B/heavy must load. Restore only after `min_dwell_s` (600) and
   only when it makes sense (5080 needs CUDA, or 3090 no longer holds an
   exclusive class). No ping-pong.
3. **Stage B (near, not enabled):** one GGUF via llama.cpp RPC across both
   cards for models with `pool_ok` and `latency=ok-lan` that need more VRAM
   than either card. Layer split by 24:16 VRAM ratio (~37 GiB usable after
   1536 MiB headroom per side). Tight-latency decode never splits.
4. Never dual 14B. Never preempt autodev for a pooled load unless steer
   says so. Never claim a 5080 GGUF load while `gguf_count` is 0 (logical
   park only). CUDA jobs keep `needs_caps: ["sm_120"]` so they cannot land
   on Ampere.

`G-SPLIT` remains a later Phase 5 experiment for **one CSD mind**. `G-POOL`
is this lab serving pool.

## Rationale

### Why This Approach

Capability tags encode the real disparity instead of a single free-MiB
number. Latency class keeps interactive autodev decode on one PCIe domain.
RPC split is the GGUF-native way to use ~40 GiB without NCCL training code.

### Alternatives Considered

#### Option 1: Always pin models to one card

- **Pros**: Simple, no LAN RTT.
- **Cons**: Cannot run ~32–40 GiB GGUFs; leftover VRAM on the other card sits idle.
- **Why Rejected**: Operator wants combined VRAM when it is worth the hop.

#### Option 2: NCCL / pipeline-parallel PyTorch now

- **Pros**: Familiar for training.
- **Cons**: Wrong stack for LocalAI GGUF autodev; sm_120 vs sm_86 binaries.
- **Why Rejected**: That is `G-SPLIT` after one-GPU proof, not helper scheduling.

#### Option 3: Treat cards as homogeneous

- **Pros**: Shorter scheduler.
- **Cons**: 14B on 5080 OOM; CUDA on 3090 steals LocalAI; FP8 kernels fail.
- **Why Rejected**: Feature disparity is the scheduling signal.

## Consequences

### Positive

- Autodev can pack embed/8B into leftover VRAM without a second 14B.
- Hosted Grok stays planner/safety; the router is local.
- A measured path exists to ~40 GiB pooled GGUF later.

### Negative

- llama.cpp RPC adds LAN RTT on offloaded layers (decode-sensitive).
- **Mitigation**: `latency=tight` never splits; only `ok-lan` pool_ok models.
- 5080 has zero GGUFs today, so migrate is logical until weights exist.

### Neutral

- Comfy stays masked during autodev; GPU CI still takes `gpu5080.lock`.

## Implementation

Catalog: `config/model-router.json`. CLI: `scripts/csd-model-router`.
Loop: `scripts/csd-autodev-loop` calls `request --noping` then `pack`.
Context: `docs/program/GPU-POOL.md`. Goal: `G-POOL` in `GOALS.md`.

Stage B stays `pool.enabled=false` until 5080 GGUFs + llama-rpc-server
are measured.

## References

- `docs/program/GPU-POOL.md`
- `docs/program/GOALS.md` (`G-POOL` vs `G-SPLIT`)
- `docs/CODEX-OPS.md` (safeguard 14 GiB / 20 GiB single-card)
- llama.cpp RPC (`rpc-server`, `--rpc host:port`)
