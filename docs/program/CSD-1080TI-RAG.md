# 1080 Ti RAG host — future third GPU

Not `STATUS.md`. Not live. Catalog stub only until `nvidia-smi` on
**gpu5080** lists the card. Router must ignore `future_hosts` until
promoted. Autodev must not start `G-1080` and must not power off the
host to seat hardware.

Identity: **autodev**. Branch: `feat/agent-harness` (never `main` /
`staging` / `develop` / `dev`).

## Hard rules

- Do **not** power off gpu5080 from this path.
- Do **not** unmask Comfy.
- Do **not** pause 3090 LocalAI.
- Do **not** treat the 1080 Ti as a 5080 substitute for PoC CUDA,
  FP8/FP4, dual-14B, or exclusive-seq train/eval.
- Never bind `0.0.0.0` on WAN.
- Never GitHub bot push.
- Never write operator or model vaults.
- Never mix 384-d Qdrant (`akula-csd-kb` stays 1024-d).

## Catalog

`config/model-router.json` `future_hosts` id **`gpu5080-1080ti`**.
Stay in `future_hosts` (`live: false`, `installed: false`) until the
operator seats the card and `nvidia-smi` on gpu5080 shows both the 5080
and the 1080 Ti. Then promote to `hosts` with role retrieve-index-light
and set `live: true`. Router scheduling reads `hosts` only.

| Field | Value |
|---|---|
| future_hosts id | `gpu5080-1080ti` |
| live | `false` (args.live=false; router ignores for scheduling) |
| installed | `false` |
| role | `retrieve-index-light` (when promoted) |
| host | `gpu5080` |
| IP (LAN) | `192.168.1.252` |
| Box | Same workstation as the RTX 5080 |
| PCI | Below the 5080 (lower slot; 5080 keeps the primary x16) |
| GPU | GTX 1080 Ti (EVGA reference / factory cooler first) |
| Arch / SM | Pascal `sm_61` |
| VRAM | ~11 GiB GDDR5X (`vram_mib` 11264, `safeguard_mib` 10240) |
| TDP | ~250 W (5080 + 1080 Ti + CPU on the 1300 W PSU is likely OK) |
| Caps | `pascal` `sm_61` `gguf-infer-legacy` `11gb` |
| prefer_jobs | `rag-index` `retrieve` `keyword-cpu` `small-embed` `light-gguf` |
| Lacks | tensor cores, native FP16 tensor, `sm_86`, `sm_120`, `fp8`, `fp4`, `high-vram`, `blackwell`, `ampere` |

## Role (when live)

RAG **retrieve / index**, keyword-cpu assist, small embed, light
legacy GGUF. **Not** train/eval exclusive-seq. **Not** PoC CUDA on
Blackwell kernels. **Not** a second 14B.

| Workload | Card |
|---|---|
| `csd-kb-index` + retrieve (prefer) | 1080 Ti |
| Keyword-cpu assist (`:8091` / `:8092`) | 1080 Ti overflow; CPU remains valid |
| Small embed / light GGUF | 1080 Ti |
| Train / eval / GPU CI / FP8 / FP4 / sm_120 | **5080** exclusive-seq |
| Comfy / media (operator unmask only) | **5080** |
| `local/code` autodev | **3090** LocalAI (stay loaded) |

When live: `./scripts/csd-kb-index` and retrieve **prefer the 1080 Ti**.
The 5080 stays train/eval/Comfy. The 3090 stays `local/code`. Do not
stop LocalAI for RAG. Do not steal 5080 exclusive-seq VRAM for index.

## Isolation

Same box, two devices. **No MIG** (GeForce; none of 3090 / 5080 /
1080 Ti implement MIG). Do not share VRAM with 5080 jobs.

| Mechanism | Policy |
|---|---|
| `CUDA_VISIBLE_DEVICES` | Separate from 5080 exclusive-seq. RAG/index process sees **only** the 1080 Ti. Train/eval/Comfy sees **only** the 5080. |
| MIG | Do not call MIG APIs. Do not claim MIG. |
| VRAM | Do not share VRAM with 5080 jobs. No CUDA IPC / MPS across the two cards for this role. |
| Lock | 5080 `gpu5080.lock` / exclusive-seq stays the 5080. 1080 Ti RAG must not take that lock to “borrow” the 5080. |
| Router | Ignore `future_hosts.gpu5080-1080ti` while `live: false`. Never schedule `sm_120` or FP8/FP4 on Pascal. |

Pin by PCI / UUID after `nvidia-smi -L`, not by assuming index 0 is
the 5080. Factory cooler first; do not plan a 1080 Ti loop until
factory air is measured.

## Labels (o11y)

Device-qualified path. `host` stays **`gpu5080`** (the machine). Do
not emit a live host value `gpu5080-1080ti` until `nvidia-smi` lists
the card.

| Key | Value |
|---|---|
| `env` | `lab` |
| `host` | `gpu5080` |
| `ns` | `index` |
| `kind` | `gpu` |
| `group` | `akula-rag` |
| `path` | `lab.gpu5080.index.1080ti` |

Tuple: `env=lab host=gpu5080 ns=index kind=gpu group=akula-rag path=lab.gpu5080.index.1080ti`.

Qdrant collection remains `akula-csd-kb` at **1024** dimensions.
Never mix 384-d collections.

## Promote (operator, not autodev)

1. Card seated. Factory cooler. Host stays up (no remote power-off).
2. `nvidia-smi` on gpu5080 lists RTX 5080 **and** GTX 1080 Ti.
3. Record PCI / UUID. 1080 Ti is the device **below** the 5080.
4. Promote `future_hosts.gpu5080-1080ti` → `hosts` with
   retrieve-index-light. Wire `csd-kb-index` host/device override to
   that UUID via `CUDA_VISIBLE_DEVICES`.
5. Stamp the labels above. 5080 exclusive-seq unchanged. 3090 LocalAI
   unchanged. Comfy stays masked unless the operator unmasks.

Until then: observe only. JSON + this doc must both show `live: false`.
`ok` is that match (no smi until live).

## References

- Router stub: `config/model-router.json` `future_hosts.gpu5080-1080ti`
- ADR-0016 consumer GPU share · [GPU-SHARE.md](GPU-SHARE.md) (`G-1080`)
- Placement: [CODEX-OPS.md](../CODEX-OPS.md) · [WHO-RUNS-WHAT.md](WHO-RUNS-WHAT.md)
- Index script today: `scripts/csd-kb-index` (`CSD_INDEX_HOST` default `gpu5080`)
