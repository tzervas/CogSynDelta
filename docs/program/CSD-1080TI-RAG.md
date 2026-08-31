# 1080 Ti RAG host — live retrieve-index-light guest

Not `STATUS.md`. Guest `nvidia-smi` lists the GTX 1080 Ti. Catalog
`hosts.gpu5080-1080ti` is `live: true`. Autodev must not start `G-1080`
and must not power off the host.

Identity: **autodev**. Branch: `feat/agent-harness` (never `main` /
`staging` / `develop` / `dev`).

Driver split and sandbox: [CSD-1080TI-ISOLATE.md](CSD-1080TI-ISOLATE.md)
(VFIO `06:00.0` only; 5080 stays host).

## Hard rules

- Do **not** power off gpu5080 from this path.
- Do **not** unmask Comfy.
- Do **not** pause 3090 LocalAI.
- Do **not** VFIO the RTX 5080.
- Do **not** treat the 1080 Ti as a 5080 substitute for PoC CUDA,
  FP8/FP4, dual-14B, or exclusive-seq train/eval.
- Never bind `0.0.0.0` on WAN.
- Never GitHub bot push.
- Never write operator or model vaults.
- Never mix 384-d Qdrant (`akula-csd-kb` stays 1024-d).

## Catalog

`config/model-router.json` `hosts` id **`gpu5080-1080ti`**.
Promoted 2026-08-31 after guest `nvidia-smi -L` listed
`GPU-4df3ba11-fd12-3550-bb97-ad00b0b00569`. Host `nvidia-smi -L` is
still RTX 5080 only (VFIO). Router scheduling reads `hosts` only.

| Field | Value |
|---|---|
| hosts id | `gpu5080-1080ti` |
| live | `true` (matches guest `nvidia-smi -L`) |
| installed | `true` (guest Tesla 535.274.02, UUID listed) |
| role | `retrieve-index-light` |
| host | `gpu5080` |
| guest IP / endpoint | `192.168.1.243` / `http://192.168.1.243` (LAN only) |
| IP (LAN host) | `192.168.1.251` (router static; was `.252`) |
| Box | Same workstation as the RTX 5080 |
| PCI | **`06:00.0`** (below the 5080; 5080 keeps the primary x16) |
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

Same box, two devices. **No MIG**. Do not share VRAM with 5080 jobs.

**Driver split is VFIO, not a container.** Blackwell 610 (host 5080)
and a Pascal driver cannot share one `nvidia.ko`. Docker/Podman GPU
passthrough uses the host module. Decision:
[CSD-1080TI-ISOLATE.md](CSD-1080TI-ISOLATE.md).

| Mechanism | Policy |
|---|---|
| VFIO | Bind **`06:00.0`** (1080 Ti) only. 5080 stays host `nvidia.ko` for CUDA CI / `gpu-5080.yml`. Guest: older Pascal driver, llama.cpp/embed. |
| CDI / UUID container | Only **after** host or guest actually binds the 1080 Ti. UUID only; no 5080. Not a second driver. |
| `CUDA_VISIBLE_DEVICES` | Process visibility only. Does not split kernel modules. |
| MIG | Do not call MIG APIs. Do not claim MIG. |
| VRAM | Do not share VRAM with 5080 jobs. No CUDA IPC / MPS across the two cards for this role. |
| Lock | 5080 `gpu5080.lock` / exclusive-seq stays the 5080. 1080 Ti RAG must not take that lock to “borrow” the 5080. |
| Router | Ignore `future_hosts.gpu5080-1080ti` while `live: false`. Never schedule `sm_120` or FP8/FP4 on Pascal. |

Pin by PCI / UUID after `nvidia-smi -L`, not by assuming index 0 is
the 5080. Factory cooler first; factory eco PL when bound. Do not plan
a 1080 Ti loop until factory air is measured. Preserve `/models`
(bind of `/bulk/models-hdd` on md127 RAID0). Do not wipe NVMe OS.
Do not reshape the healthy RAID0 that already uses both 3 TB HDDs.

Host verify (2026-08-31T20:58:23Z):
host `nvidia-smi -L` lists **only** RTX 5080
`GPU-087267a6-14fb-0af3-da30-9a1a18523106` (P8). `lspci -k` shows
`06:00.0`/`06:00.1` **vfio-pci**. Guest `tzervas@192.168.1.243`
`nvidia-smi -L` lists GTX 1080 Ti
`GPU-4df3ba11-fd12-3550-bb97-ad00b0b00569` (Tesla 535.274.02, PL 250 W).
Domain `gpu5080-1080ti-rag` running, virsh autostart. RAID `md127`
`/models` still mounted. Comfy masked. 5080 stays host nvidia 610.

## Labels (o11y)

Device-qualified path. `host` stays **`gpu5080`** (the machine).
Guest endpoint is `192.168.1.243`; do not scrape guest Prom until an
exporter exists.

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

## Promote (done 2026-08-31)

1. Card seated. Factory cooler. Host stays up. **Done.**
2. PCI **`06:00.0`** is the 1080 Ti. Guest `nvidia-smi` lists UUID.
   **Done.**
3. VFIO `06:00.0` only. 5080 remains host nvidia 610. **Done.**
4. Promoted `hosts.gpu5080-1080ti` retrieve-index-light `live: true`.
   Endpoint `http://192.168.1.243`. Index prefers the guest. **Done.**
5. Labels above. 5080 exclusive-seq CUDA CI unchanged. 3090 LocalAI
   unchanged. Comfy stays masked.

JSON + this doc both show `live: true`. `ok` is that match against
**guest** `nvidia-smi -L` (1080 Ti listed → `live: true`).

## References

- Isolate / VFIO: [CSD-1080TI-ISOLATE.md](CSD-1080TI-ISOLATE.md)
- Router: `config/model-router.json` `hosts.gpu5080-1080ti`
- ADR-0016 consumer GPU share · [GPU-SHARE.md](GPU-SHARE.md) (`G-1080`)
- Placement: [CODEX-OPS.md](../CODEX-OPS.md) · [WHO-RUNS-WHAT.md](WHO-RUNS-WHAT.md)
- Index script: `scripts/csd-kb-index` (`CSD_INDEX_HOST` default `gpu5080-1080ti`)
