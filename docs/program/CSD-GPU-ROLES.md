# GPU roles — 3090 autodev, 5080 chat+Comfy, 1080 Ti RAG

Not `STATUS.md`. Not MIG. Not tensor-wire `G-POOL`. Not 256 specialists.
Operator **fit snapshot** plus role table. Source of truth for aliases:
`config/model-router.json` schema `csd-model-router/v2`. Keep-up mutex:
[CSD-GPU-KEEPUP.md](CSD-GPU-KEEPUP.md). Flex/FIT tags:
[CSD-CLUSTER-FLEX.md](CSD-CLUSTER-FLEX.md). Ops: [../CODEX-OPS.md](../CODEX-OPS.md).

Identity: **autodev**. Branch: `feat/agent-harness` (never `main` /
`staging` / `develop` / `dev`). gpu5080 LAN: **`192.168.1.251`**
(never `.252`). Sampled **2026-09-01T00:09:13Z** (`nvidia-smi` on
prime, SSH `.251`, SSH guest `.243`).

## Priority (always)

Highest first. Queue / `gpu5080.lock`, **not** `docker stop` /
`systemctl mask`. Autodev and `local/code` **always** preempt Open
WebUI chat and Comfy.

1. **autodev / `local/code`** (3090 resident; 5080 CUDA/CI exclusive-seq)
2. **OWUI text chat** (smaller GGUF on 5080 only if FIT; else 3090)
3. **Comfy / media** (5080; waits on the same lock)
4. **idle share-small** helpers (`helper_cap` 6144 MiB; free ≥ 8192 MiB)

Services **stay up**. Comfy HTTP may sit idle; GPU alloc waits on
`gpu5080-lock`. keepup-queue may still be wiring that wrap — do not
fight it; set prio in timeshare JSON on gpu5080
(`AKULA_TIMESHARE=/home/tzervas/akula-harness/state/gpu-timeshare.json`).

## Hard rules

- Never dual 14B.
- Never pause 3090 LocalAI (`docker stop akula-localai`) for RAG.
- Never bind `0.0.0.0` on WAN.
- Never GitHub bot push.
- Do **not** move a 14B onto the 5080 unless live `nvidia-smi` free
  **after load** is honest **and** 14B + Comfy `helper_cap` still
  `<=` `safeguard_mib`. Today that is **false**.
- 1080 Ti guest: `retrieve-index-light` only. Not 14B, not Comfy,
  not `sm_120` / FP8 / FP4, not exclusive-seq.
- 3090 is never given to Comfy.

## Live `nvidia-smi` (this sample)

| Box | IP | GPU | UUID | Driver | Total | Used | Free | Util | P-state | PL |
|---|---|---|---|---|---|---|---|---|---|---|
| akula-prime | `192.168.1.98` | RTX 3090 Ti | `GPU-642a8f8b-7a80-6b5e-76d4-034b742a1480` | 610.57.04 | **23028** | **15988** | **6575** | 9% | P8 | 14 / 450 W |
| gpu5080 | `192.168.1.251` | RTX 5080 | `GPU-087267a6-14fb-0af3-da30-9a1a18523106` | 610.57.04 | **16303** | **10** | **15864** | 0% | P8 | 12 / 360 W |
| 1080 Ti guest | `192.168.1.243` | GTX 1080 Ti | `GPU-4df3ba11-fd12-3550-bb97-ad00b0b00569` | 535.274.02 | **11264** | **8** | **11163** | 0% | P8 | 9 / 250 W |

Earlier in the same sample window (~20:07–20:08 local) 3090 used was
**15966** MiB free **6597** (llama.cpp LocalAI **14752** MiB + desktop).
5080 had **no** compute processes. Guest had Xorg **5** MiB only.

5080 lock / queue at sample:

| Signal | Value |
|---|---|
| `gpu5080.lock` flock | **idle-no-flock** (empty file, no holder) |
| timeshare `current` | `null` |
| timeshare `queue` | `[]` |
| `akula-comfyui.service` | **active** / `generated` / lock wrap in ExecStart (`akula-comfyui-locked`). HTTP up; GPU not allocated (used 10 MiB) |

Do not mask Comfy. Wrapper drop-in:
`/etc/systemd/system/akula-comfyui.service.d/gpu5080-lock.conf`.

## Catalog (v2) vs live

`hosts.*.vram_mib` / `safeguard_mib` from `config/model-router.json`.

| Host id | Role | Catalog VRAM | Safeguard | Live used | Honest leftover for new load |
|---|---|---|---|---|---|
| `akula-prime` | **autodev** `local/code` 14B Q4 32k home | 23028 | **20480** | 15988 | `min(6575, 20480−15988)` = **4492** MiB |
| `gpu5080` | **chat + Comfy**; CUDA/CI via lock | 16303 | **14336** | 10 | `min(15864, 14336−10)` = **14326** MiB before reserving Comfy |
| `gpu5080-1080ti` | **retrieve-index-light** | 11264 | **10240** | 8 | `min(11163, 10240−8)` = **10232** MiB |

`local/code` catalog `vram_mib` **16500**. Live llama.cpp **14752** plus
desktop ~1.2 GiB. Resident stays on the 3090. Swap other 3090 aliases
only if they FIT that **4492** leftover (safeguard-capped, not raw free).

## `fits_5080_chat`

Rule: **true** only if a catalog **chat** alias with `class` **not**
`14b` has `vram_mib + Comfy helper_cap <= gpu5080.safeguard_mib`.

Constants:

| Term | MiB | Why |
|---|---|---|
| 5080 `safeguard_mib` | **14336** | catalog; ~14 GiB cap, not the 16303 sticker |
| Comfy `helper_cap` | **6144** | ~6 GiB (`6 × 1024`). Measured media receipt `vram_used_mib` 5872 < 6144 |
| Chat budget under safeguard | **8192** | `14336 − 6144` |
| 5080 live free | **15864** | honest empty card; still use safeguard for admission |

Chat-shaped aliases (jobs `chat` / `chat-small`):

| Alias | class | `vram_mib` | + helper_cap | `<= 14336`? | Notes |
|---|---|---|---|---|---|
| `local/uncensored-fast` | **8b** | **6000** | **12144** | **yes** (margin 2192) | only non-14B chat alias |
| `local/fast` | 14b | 12000 | 18144 | no | dual-14B + no Comfy room |
| `local/general` | 14b | 12000 | 18144 | no | same |
| `local/code` | 14b | 16500 | 22644 | no | **16500 > 16303** total; **16500 > 15864** live free |

Other 14B/heavy (not chat, still refuse on 5080): `local/code-alt`
16500, `local/reasoning` 18000, `local/code-heavy` 20000.

**`fits_5080_chat`: true** — resident 5080 chat is
`local/uncensored-fast` (8B, 6000 MiB) **with** Comfy 6144 reserved.
Not a 14B. 14B does **not** FIT this card with Comfy.

5080 after that pair (catalog): `12144 / 14336` safeguard,
`12144 / 16303` sticker. Live headroom if both load as cataloged:
`16303 − 10 − 12144 ≈ 4149` free (still under safeguard by 2192).

If `gpu5080.lock` is held (autodev CUDA/CI), **both** 5080 chat and
Comfy GPU alloc yield. HTTP stays up; weights do not steal exclusive-seq.

## Role table

| Card | Default home | May hold if FIT | Never |
|---|---|---|---|
| 3090 Ti | `local/code` 14B Q4 32k (~16.5 GiB) autodev | leftover helpers `<= 4492` MiB now (`embed-qwen3-0.6b` 4000 **yes**; 8B 6000 **no** while 14B resident) | second 14B; Comfy; pause for RAG |
| RTX 5080 | smaller chat (`local/uncensored-fast`) **+** Comfy image/video | share-small embed 4000 when lock idle; `cuda-eval` exclusive-seq (preempts chat/Comfy GPU) | 14B; dual-14B; mask-to-schedule |
| GTX 1080 Ti guest | RAG retrieve / `csd-kb-index` | small embed / light GGUF / keyword-cpu | 14B, Comfy, `sm_120`, exclusive-seq |

Fan-out (do not start intern-pool): one latency-tight worker stays
3090 `local/code`; parallel / `ok-lan` helpers → 3090 leftover, then
5080 share-small (lock idle), then 1080 Ti. Need `sm_120` / FP8 /
CUDA → 5080 exclusive-seq (Comfy waits).

## Related

- Catalog: `config/model-router.json` (`csd-model-router/v2`)
- Keep-up: [CSD-GPU-KEEPUP.md](CSD-GPU-KEEPUP.md)
- Flex: [CSD-CLUSTER-FLEX.md](CSD-CLUSTER-FLEX.md)
- Eco: [CSD-GPU-ECO.md](CSD-GPU-ECO.md)
- 1080 Ti RAG: [CSD-1080TI-RAG.md](CSD-1080TI-RAG.md)
- Pool (Stage B parked): [GPU-POOL.md](GPU-POOL.md)
- Horizon intern: [GPU-SHARE.md](GPU-SHARE.md)
