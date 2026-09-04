# GPU pool — autodev context pack

Cold-start for `local/code` and the lab console. Not `STATUS.md`. Not
`G-SPLIT` (one CSD mind). This is **inference/helper placement** across
akula-prime (3090 Ti) and gpu5080 (5080).

## Drive split

Hosted Grok = planner / safety / unblock / this pack. Implement =
`scripts/csd-autodev-loop` → `csd-model-router` → `csd-localai-queue` →
3090 `local/code`. Do not start `/csd-python-first-drive` for implement.

## Cards (disparity is a routing tag, not a bug)

| Host | GPU | SM | VRAM | Caps (prefer) | Lacks |
|---|---|---|---|---|---|
| akula-prime | 3090 Ti | 8.6 Ampere | ~23028 MiB | `gguf-infer` `high-vram` `localai` `decode-tight` | native FP8/FP4, sm_120 |
| gpu5080 | 5080 | 12.0 Blackwell | ~16303 MiB | `cuda-eval` `sm_120` `fp8` `fp4` `embed` `gpu-ci` | high-vram, LocalAI GGUF today |

Never schedule sm_120 CUDA on the 3090. Never put a 14B 32k GGUF on the
5080. Combined usable for a **pooled** GGUF is ~37 GiB (39331 − headroom).
Single-card autodev still uses 20 GiB / 14 GiB safeguards.

## Stage A (live)

`config/model-router.json` + `scripts/csd-model-router`.

- `local/code` sticky on 3090. Never dual 14B.
- `fits_5080` + `latency=ok-lan` helpers migrate to 5080 when a tight
  14B/heavy must load. Restore only after 600s dwell and only if it
  makes sense (5080 needs CUDA, or 3090 idle and preferred host is prime).
- 5080 `gguf_count=0` → migrate is **logical park**, not a fake load.
- `csd-model-router pack` fills leftover VRAM (embed on 5080 if lock idle).

## Stage B (not enabled)

Mixed backends (`llama.cpp` GGUF, `vllm` HF/MoE batch, `bitnet-cpp`
W1.58). `pool.enabled=false` until 5080 GGUFs + images exist.

**Why pool ~40 GiB:** run **larger MoE** than either card holds
(router + active experts tight; idle expert shards `ok-lan` on the
other card). Alias `pool/large` / `pool/moe` (~36 GiB). Tight decode
never splits a dense 14B. Never preempt autodev unless steer says so.

## Stage C (horizon, do not start)

Intern + parallel share: up to 256 microscopic specialists **and**,
once Phase 3 CSD exists, **a bunch of small CSD region models** on the
same interned pool (one mind, many regions — not a swarm).
Pack: [GPU-SHARE.md](GPU-SHARE.md) (`G-SHARE`, `G-1080`).

## Product next

**G-QD met** (PR #9 `a823268`). Next board **P1-08** tiered store.
Do not start G-TRAIN, G-SPLIT, G-SHARE, G-1080, or physical RPC.

## Commands

```bash
./scripts/csd-gpu-plan
./scripts/csd-model-router status
./scripts/csd-model-router request --alias local/code --why autodev --noping
./scripts/csd-model-router pack
./scripts/csd-autodev-loop --once
```

Lab UI: https://code.vectorweight.com/lab — tabs **Live feed** (loaded
models only) and **Pool**.
