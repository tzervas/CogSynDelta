# Deployed cluster pack — 2026-08-31

Not `STATUS.md`. Live capture **2026-08-31T22:32:27Z**. Identity **autodev**.
Branch `feat/agent-harness` (never `main` / `staging` / `develop` / `dev`).
Vault twin: `/akula-data/obsidian/akula-csd-kb/Program/CONTEXT-PACK-DEPLOYED-2026-08-31.md`.

Three GPU backends are **live**. 256 specialists (ADR-0016 / GPU-SHARE) is a
**future intern-pool example**, not a deployed target. Pool RPC is off.

## Hard never

- GitHub bot push. Forgejo `tzervas/*` only.
- Bind `0.0.0.0` WAN. Listen is `lan-only`.
- Pause 3090 LocalAI (`akula-localai`). Dual 14B.
- Mix 384-d Qdrant. Collection `akula-csd-kb` stays **1024-d** Cosine.
- Write `tzervas-dev-kb` or `akula-model-kb`. This pack is experiment plane only.
- Treat 1080 Ti as a 5080 substitute (`sm_120` / FP8 / FP4 / exclusive-seq).
- Take `gpu5080.lock` for RAG index. CUDA CI stays 5080 exclusive-seq.
- Unmask Comfy. Power off gpu5080. Start `G-1080` / `G-SHARE` / `G-TRAIN`.

## Three backends (FIT + tags, not MIG)

| Id | Box / IP | GPU / UUID | Role | Live |
|---|---|---|---|---|
| `akula-prime` | `192.168.1.98` | RTX 3090 Ti `GPU-642a8f8b-7a80-6b5e-76d4-034b742a1480` | `local/code` implementer | yes |
| `gpu5080` | `192.168.1.251` (never `.252`) | RTX 5080 `GPU-087267a6-14fb-0af3-da30-9a1a18523106` | exclusive-seq CUDA / CI | yes (`needs_lock: gpu5080`) |
| `gpu5080-1080ti` | guest `192.168.1.243` host `.251` PCI `06:00.0` | GTX 1080 Ti `GPU-4df3ba11-fd12-3550-bb97-ad00b0b00569` | **retrieve-index-light** | **yes** (`hosts.live: true`) |

Router `config/model-router.json` schema `csd-model-router/v2`:
`never_dual_14b: true`, `pool.enabled: false`, `future_hosts: {}`,
all hosts `listen: lan-only`. `hosts.gpu5080-1080ti.live=true` matches
**guest** `nvidia-smi -L`. Lab `GET /api/gpu` agrees (`live: true`).

## VFIO guest (retrieve-index-light)

Guest SSH `tzervas@192.168.1.243` `nvidia-smi -L`:

```
GPU 0: NVIDIA GeForce GTX 1080 Ti (UUID: GPU-4df3ba11-fd12-3550-bb97-ad00b0b00569)
```

Tesla **535.274.02**, persistence Enabled, **P8**, PL **250 W** = default,
util 0%. Role: RAG retrieve/index, keyword-cpu assist, small embed, light
GGUF. Endpoint `http://192.168.1.243` LAN only. Prefer this card for
`./scripts/csd-kb-index`. Do **not** train / PoC CUDA / exclusive-seq here.

Host `tzervas@192.168.1.251` `nvidia-smi -L` lists **only** the 5080
(driver 610.57.04). `lspci -k`: `01:00.0` nvidia, `06:00.0`/`06:00.1`
**vfio-pci**. Domain `gpu5080-1080ti-rag` running, virsh autostart.
Comfy **masked** / inactive. `GET /api/gpu/lock` → `lock=idle`.

## RAID0 `/bulk`

Host md127 `gpu5080:bulk` RAID0 `sda1`+`sdb1`, super 1.2, 512k chunks,
**clean 2/2**, 5.46 TiB. Mount `/bulk` ext4 UUID
`d942aafc-aa92-4fe4-8a04-732a5eda8809` `rw,noatime,stripe=256`
(`nofail`). `/models` is bind of `/bulk/models-hdd`. Preserve it. Do not
wipe NVMe OS. Do not reshape the healthy array. Do not use `/models` as
guest root.

## Eco (factory clocks + default PL)

Persistence on. `-pl` = `power.default_limit`. Idle P-states allowed.
Not an undervolt / OC BIOS.

| GPU | persistence | PL vs default | Idle this capture |
|---|---|---|---|
| 3090 Ti | Enabled | 450 = 450 W | P5, ~30 W, LocalAI resident (~16 GiB) |
| 5080 | Enabled | 360 = 360 W | P3, ~40 W, util 0%, Comfy masked |
| 1080 Ti (guest) | Enabled | 250 = 250 W | P8, ~8.5 W, util 0% |

`nvidia-factory-limits.service` enabled+active on gpu5080. Do not drop 3090
PL below default. Do not raise to max. Guest eco is **in-guest** after VFIO.

## `need_grok` mailbox

File `/akula-data/cabal/csd-need-grok.json`. Lab
`http://192.168.1.98:9118/api/grok-need` (LAN, not WAN).

This capture: `need: false`. Hosted Grok is **not** in the loop. Autodev
does not poll xAI. Operator opens a chat only when `need=true`.

## Next: `region-pretrain`

Steer `/akula-data/cabal/csd-steer.json` + `GET /api/goals`:
`next_goal: region-pretrain`, `pause: false`.

One existing region (LatentVAE / `stream_vae`) on
`Salesforce/wikitext` `wikitext-2-raw-v1` **train** (rev
`b08601e04326c79dfdd32d625aee71d232d685c3`). Failing pytest first:
`tests/test_poc_region_pretrain.py`. Pack:
[CONTEXT-PACK-REGION-PRETRAIN.md](CONTEXT-PACK-REGION-PRETRAIN.md).
Not 14B. Not `G-TRAIN`. P1-09 parked. Implement on 3090 `local/code`.
Tiny CUDA later on 5080 exclusive-seq only.

## `hf/autodev` missing

CSD vault `/akula-data/cabal/csd-vault` `secret ls`: `csd/apply-token`,
`git/autodev`, `gpu/localai-api-key` only. `hf/` exists **empty**.
Lab `GET /api/goals` → `hf.autodev: false`, `hf.gap: "mint HF"`.
No Hub publish until operator mints a fine-grained write token on
`tzervas/cogsyndelta*` and `secret set hf/autodev`. Never copy
`gpu/huggingface-token`.

## Qdrant / vault (experiment plane)

| Piece | Value |
|---|---|
| Obsidian | `/akula-data/obsidian/akula-csd-kb` (`csd-kb-setup`) |
| Qdrant | `127.0.0.1:6333` collection `akula-csd-kb` |
| Vectors | size **1024**, Cosine, status green, points 83 |
| Index | prefer 1080 Ti guest; never 3090 pause; never `gpu5080.lock` |

Keyword-cpu stays `:8091` / `:8092`. LocalAI `127.0.0.1:8080` + LAN socat
`192.168.1.98:8080` (healthy, 401 without key = still up).

## Related

- Flex: [CSD-CLUSTER-FLEX.md](CSD-CLUSTER-FLEX.md)
- Guest RAG: [CSD-1080TI-RAG.md](CSD-1080TI-RAG.md)
- VFIO: [CSD-1080TI-ISOLATE.md](CSD-1080TI-ISOLATE.md)
- Eco: [CSD-GPU-ECO.md](CSD-GPU-ECO.md)
- Ping: [CSD-GROK-PING.md](CSD-GROK-PING.md)
- Autodev: [CSD-AUTODEV-HANDOFF.md](CSD-AUTODEV-HANDOFF.md)
- Ladder: [CSD-SCALE-LADDER.md](CSD-SCALE-LADDER.md)
- Regions: [CSD-BRAIN-REGIONS.md](CSD-BRAIN-REGIONS.md)
