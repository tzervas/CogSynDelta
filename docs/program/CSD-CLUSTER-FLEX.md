# Cluster flex — heterogeneous GPU fit (3090 + 5080 + 1080 Ti)

Not `STATUS.md`. Not MIG. Not a speedup claim. Operator policy for
**what FITs** across akula-prime (3090 Ti) and gpu5080 (5080; 1080 Ti
when a VFIO guest or host `nvidia-smi` lists it). Source of truth:
`config/model-router.json` schema `csd-model-router/v2`.

Identity: **autodev**. Branch: `feat/agent-harness` (never `main` /
`staging` / `develop` / `dev`). gpu5080 LAN: **`192.168.1.251`**
(never `.252`).

Autodev **may** combine cards for helpers, extra inference, RAG, and
light GGUF — any mix that **FITs** leftover VRAM and respects arch
tags. Latency-**tight** stays on one card. `ok-lan` may split or
migrate. The pool is **heterogeneous** (~24 + 16 + 11 GiB when the
1080 Ti is live), not one MIG slice.

## Hard rules

- Never dual 14B.
- Never pause 3090 LocalAI (`docker stop akula-localai`) for RAG
  search. Keyword-cpu stays `:8091` / `:8092`.
- Never bind `0.0.0.0` on WAN. LocalAI stays `127.0.0.1:8080` (+ LAN
  socat on `192.168.1.98`).
- Never GitHub bot push.
- Never schedule `sm_120` / FP8 / FP4 CUDA on Ampere or Pascal.
- Never put a 14B 32k GGUF on the 5080 or the 1080 Ti.
- 5080 exclusive-seq (`gpu5080.lock` / CUDA CI / train) **preempts
  helpers on that card**. Share-small embed only when the lock is
  idle.
- Do not treat the 1080 Ti as a 5080 substitute. Do not call MIG
  APIs. Do not invent 10× compression or 10× throughput.

If the 1080 Ti guest is **not live**, still enable **3090 + 5080**
helper policy (Stage A place/migrate/pack). Router ignores
`future_hosts` until promoted.

## Cards (tags, not one pool)

| Host id | Box / IP | GPU | Arch tags | VRAM (json) | Safeguard | Live |
|---|---|---|---|---|---|---|
| `akula-prime` | akula-prime `192.168.1.98` | RTX 3090 Ti | `ampere` `sm_86` `fp16` `bf16` `gguf-infer` `high-vram` `localai` `decode-tight` | 23028 MiB | 20480 MiB | yes (`hosts`) |
| `gpu5080` | gpu5080 `192.168.1.251` | RTX 5080 | `blackwell` `sm_120` `fp8` `fp4` `fp16` `bf16` `cuda-eval` `embed` `gpu-ci` `parallel-batch` | 16303 MiB | 14336 MiB | yes (`hosts`; `needs_lock: gpu5080`) |
| `gpu5080-1080ti` | same box `192.168.1.251` PCI `06:00.0` | GTX 1080 Ti | `pascal` `sm_61` `gguf-infer-legacy` `11gb` | 11264 MiB | 10240 MiB | **no** (`future_hosts`; VFIO; no UUID) |

Lacks (do not schedule against them):

| Host | Lacks |
|---|---|
| `akula-prime` | `native-fp8` `fp4` `sm_120` `blackwell` `parallel-batch` |
| `gpu5080` | `high-vram` `localai` |
| `gpu5080-1080ti` | tensor cores, native FP16 tensor, `sm_86`, `sm_120`, `fp8`, `fp4`, `high-vram`, `blackwell`, `ampere` |

3090: `local/code` 14B **may stay loaded**. Leftover VRAM may hold
embed / 8B helpers. 5080: CUDA/eval; share-small embed when
`gpu5080.lock` is idle. 1080 Ti (when live): RAG retrieve/index,
keyword-cpu assist, small embed, light GGUF — never exclusive-seq,
never FP8/FP4, never `sm_120`.

## Flex (FIT + tags)

`never_dual_14b: true`. `autodev_default: local/code`.
`autodev_sticky: true`. `pool.enabled: false` (Stage B parked).

| Latency | Placement |
|---|---|
| `tight` | One card, one PCIe domain. Do not RPC-split. |
| `ok-lan` | May migrate or (later) split across hosts that FIT. |

Pack leftover, do not OOM:

| Card | Resident | Leftover may hold (if FIT) | Never |
|---|---|---|---|
| 3090 Ti | `local/code` (~16500 MiB) | `embed-qwen3-0.6b` (~4000) and/or `local/uncensored-fast` (~6000) | second 14B; pause for RAG |
| 5080 | idle **or** exclusive-seq job | share-small embed (~4000) when lock idle | 14B; helpers while `gpu5080.lock` held |
| 1080 Ti | nothing until UUID | small embed / light GGUF / retrieve (when live) | `sm_120`, FP8/FP4, 14B, exclusive-seq |

Combined raw: 23028 + 16303 = **39331 MiB** (3090+5080). Plus 11264
when the 1080 Ti is promoted ≈ **50595 MiB**. Usable pooled GGUF on
the two live cards is **39331 − `headroom_mib` 1536** (~37 GiB),
ratio `24:16`. Three-way split is later and only for workloads that
run Pascal + Ampere + Blackwell together.

## Fit table (v2 aliases)

Min VRAM is `aliases.*.vram_mib` from `config/model-router.json`.
Arch tags are `needs_caps` plus hard SM filters. Hosts allowed are
`host_pref` intersected with `fits_*` / `future_hosts.live`.
**exclusive-seq conflict** is vs `gpu5080.lock` / CUDA CI / train on
the 5080 (that lock still preempts helpers **on that card**).

| Alias | Min VRAM | Arch tags | Hosts allowed | exclusive-seq conflict |
|---|---|---|---|---|
| `local/code` | 16500 MiB | `14b` `gguf-infer` `high-vram` `decode-tight` `localai` `sm_86` | `akula-prime` only (`fits_5080: false`) | none (never on 5080) |
| `local/fast` | 12000 MiB | `14b` `gguf-infer` `localai` `sm_86` | `akula-prime` only | none (never on 5080) |
| `local/general` | 12000 MiB | `14b` `gguf-infer` `localai` `sm_86` | `akula-prime` only | none (never on 5080) |
| `local/code-alt` | 16500 MiB | `14b` `gguf-infer` `high-vram` `localai` `sm_86` | `akula-prime` only | none (never on 5080) |
| `local/reasoning` | 18000 MiB | `heavy` `gguf-infer` `high-vram` `decode-tight` `sm_86` | `akula-prime` only | none (never on 5080) |
| `local/code-heavy` | 20000 MiB | `heavy` `gguf-infer` `high-vram` `decode-tight` `sm_86` | `akula-prime` only | none (never on 5080) |
| `local/vision` | 9000 MiB | `vision` `gguf-infer` (`ok-lan`) | `akula-prime`, `gpu5080` if lock idle; 1080 Ti only if live **and** leftover FITs (10240 safeguard) | **preempted** on 5080 when lock held |
| `local/uncensored-fast` | 6000 MiB | `8b` `gguf-infer` (`ok-lan`) | `akula-prime` leftover, `gpu5080` if lock idle; 1080 Ti when live (light GGUF) | **preempted** on 5080 when lock held |
| `embed-qwen3-0.6b` | 4000 MiB | `embed` `fp16` prefer `sm_120` `cuda-eval` `embed` | `gpu5080` (pref, lock idle), `akula-prime` leftover; 1080 Ti when live (small-embed) | **preempted** on 5080 when lock held (`needs_lock: gpu5080`) |
| `cuda-eval` | 8192 MiB | `cuda` `cuda-eval` `sm_120` `blackwell` | `gpu5080` only (`fits_prime: false`) | **holds** exclusive-seq (`needs_lock: gpu5080`); preempts helpers |
| `pool/large` | 36000 MiB | `pooled` `gguf-infer` prefer `high-vram` (`ok-lan`, `pool_ok`) | `pool` (3090+5080 RPC) when `pool.enabled`; not 1080 Ti | **blocked** with `cuda` class / exclusive-seq; `pool.enabled=false` today |
| `pool/moe` | 36000 MiB | `pooled` `gguf-infer` prefer `parallel-batch` `high-vram` | `pool` when enabled; vLLM HF only on `gpu5080` | **blocked** with `cuda` class / exclusive-seq; `pool.enabled=false` today |

Class exclusivity on a **single card** (`exclusive_with` in v2):

| Class on card | Cannot co-reside |
|---|---|
| `14b` / `heavy` / `vision` | each other, plus `pooled` |
| `8b` | `heavy` only (may share leftover with a 14B if FIT) |
| `embed` | (empty) — leftover OK |
| `cuda` | other `cuda`, `pooled` |
| `pooled` | `14b` `heavy` `vision` `pooled` `cuda` |

`preempt_autodev` is **false** on pool aliases. Never steal
`local/code` for a pooled load.

## Runtimes (placement, not a 10× claim)

From v2 `runtimes`. Images/digests stay in the json.

| Runtime | Hosts | Do not |
|---|---|---|
| `llama.cpp` / `localai` | `akula-prime` (LocalAI wrap); `gpu5080` when GGUFs exist | dual 14B; pause LocalAI for RAG |
| `vllm` | `gpu5080` podman only | load on 3090 (LocalAI stays); Pascal 1080 Ti |
| `bitnet-cpp` | `akula-prime`, `gpu5080` when an image exists | force into vLLM/llama.cpp without a measured row |

## Exclusive-seq vs helpers (5080)

`gpu5080.lock` is flock, not an empty file. CUDA CI, train, Triton,
and `cuda-eval` take exclusive-seq and **unload 5080 helpers**.

| 5080 lock | Helpers on 5080 | 3090 LocalAI | 1080 Ti (when live) |
|---|---|---|---|
| idle | share-small embed / 8B if FIT (`helper_cap` ~6 GiB) | stay loaded | RAG / small embed / light GGUF (own device; do not take `gpu5080.lock`) |
| held (exclusive-seq) | **preempted** | stay loaded | still own-device RAG; must not borrow the 5080 |

Homelab (`192.168.1.170`) is CPU Actions only. No GPU alias lands
there.

## 1080 Ti guest not live

Catalog: `future_hosts.gpu5080-1080ti` (`live: false`,
`installed: false`). Seated at PCI `06:00.0` (`vfio-pci`) on
**192.168.1.251**. Host `nvidia-smi -L` lists only RTX 5080
`GPU-087267a6-14fb-0af3-da30-9a1a18523106`. No qemu guest as of
the 1080 Ti RAG catalog.

Until a guest or host lists the Pascal UUID:

1. Router scheduling reads `hosts` only (`akula-prime` + `gpu5080`).
2. **3090 + 5080 helper policy stays on** (sticky `local/code`,
   leftover embed/8B, 5080 share-small when lock idle).
3. Do not start `G-1080`. Do not power off gpu5080 to “fix” VFIO.

Promote only after `nvidia-smi` lists the 1080 Ti UUID. Role then:
`retrieve-index-light`. Still no FP8/FP4, no `sm_120`, no dual 14B,
no exclusive-seq on Pascal.

## Related

- Catalog: `config/model-router.json` (`csd-model-router/v2`)
- Placement: [GPU-POOL.md](GPU-POOL.md) · ADR-0015
- Horizon intern / 1080 Ti: [GPU-SHARE.md](GPU-SHARE.md) · [CSD-1080TI-RAG.md](CSD-1080TI-RAG.md)
- Ops: [../CODEX-OPS.md](../CODEX-OPS.md) · who runs what: [WHO-RUNS-WHAT.md](WHO-RUNS-WHAT.md)
- Eco: [CSD-GPU-ECO.md](CSD-GPU-ECO.md)
