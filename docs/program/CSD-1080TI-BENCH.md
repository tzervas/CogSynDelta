# 1080 Ti bench — host CUDA bind failed; RAID0 smoke ok

Not `STATUS.md`. Measured on **gpu5080** `192.168.1.251` (enp5s0,
hostname `gpu5080`). Identity: **autodev**. Branch: `feat/agent-harness`
(never `main` / `staging` / `develop` / `dev`).

**`ok=false`.** The GTX 1080 Ti is seated at PCI `06:00.0` but did
**not** run a CUDA op. Host `nvidia.ko` 610.57.04 (open / Dual MIT/GPL)
rejects Pascal (no GSP). RAID write on the mechanical stripe **did**
work. No 10× claims.

## Hard rules this run

- SSH `tzervas@192.168.1.251` / `Host gpu5080`. Not `.252`.
- Did **not** power off. Did **not** pause 3090 LocalAI. Did **not**
  unmask Comfy. Did **not** run a 14B. Did **not** start
  `./scripts/csd-kb-index` (no 1080 Ti flag; that script still enqueues
  5080 exclusive-seq).
- Did **not** steal 5080 exclusive-seq. 5080 stayed idle (P8, 0 % util).
- Did **not** wipe NVMe/SSD OS. Did **not** reshape the healthy RAID0
  that already uses both 3 TB HDDs.
- Never `0.0.0.0` WAN. Never GitHub. Never operator/model vaults.
  Never mix 384-d Qdrant.

## Host

| Field | Value |
|---|---|
| Capture | 2026-08-31T20:19:38Z … 2026-08-31T20:21:54Z |
| Uptime at SSH | 49 min (boot ~15:30 EDT) |
| Driver | NVIDIA UNIX Open Kernel Module **610.57.04** (`nvidia.ko.xz` DKMS, license Dual MIT/GPL) |
| CUDA UMD | 13.3 |
| `nvidia-smi -L` | **only** RTX 5080 `GPU-087267a6-14fb-0af3-da30-9a1a18523106` |

## 1080 Ti bind

`lspci` already listed GP102. `nvidia-smi` listed only the 5080. Live
bind **before** this bench: `06:00.0`+`06:00.1` = `vfio-pci` (IOMMU
group 18). Unbound **those two BDFs only**, cleared `driver_override`,
wrote `0000:06:00.0` into `/sys/bus/pci/drivers/nvidia/bind`. Never
touched `01:00.*`.

| Device | PCI | ID | After restore |
|---|---|---|---|
| RTX 5080 | `01:00.0` | `10de:2c02` | `nvidia` (unchanged) |
| 5080 HD audio | `01:00.1` | `10de:22e9` | `snd_hda_intel` |
| GTX 1080 Ti | `06:00.0` | `10de:1b06` EVGA `3842:6390` | `vfio-pci` (restored) |
| 1080 Ti HDMI audio | `06:00.1` | `10de:10ef` | `vfio-pci` (restored) |

Probe at **16:20:53 EDT** (also boot 15:30:10 and earlier 15:43:37):

```
NVRM: The NVIDIA GPU 0000:06:00.0 (PCI ID: 10de:1b06)
NVRM: installed in this system is not supported by open
NVRM: nvidia.ko because it does not include the required GPU
NVRM: System Processor (GSP).
nvidia 0000:06:00.0: probe with driver nvidia failed with error -1
```

`tee` to `nvidia/bind` returned `Operation not permitted` after that
failed probe. `nvidia-smi -L` still one GPU. Restored with
`/usr/local/sbin/vfio-bind-1080ti.sh` (`/dev/vfio/18` present). 5080
driver stayed loaded.

No 1080 Ti UUID → no `nvidia-smi -i`, no VRAM query, no CUDA context on
Pascal. Catalog `future_hosts.gpu5080-1080ti` stays `live: false`.

### Link (sysfs, not a throughput bench)

| GPU | current | max (sysfs) |
|---|---|---|
| 1080 Ti `06:00.0` | **2.5 GT/s ×4** | 8.0 GT/s ×16 |
| 5080 `01:00.0` (idle P8) | 2.5 GT/s ×8 | not used this run |

dmesg: 1080 Ti “8.000 Gb/s available PCIe bandwidth, limited by 2.5
GT/s PCIe x4 link at `00:1c.4`”. Slot is not x16. Do not treat this
card as a 5080-class bus.

## VRAM

| GPU | total | free | used | util | power | P-state | temp |
|---|---|---|---|---|---|---|---|
| RTX 5080 (measured) | 16303 MiB | 15864 MiB | 10 MiB | 0 % | 12.4 W | P8 | 36 °C |
| GTX 1080 Ti | **not measured** | — | — | — | — | — | — |

Pascal VRAM is catalog **11264 MiB** (~11 GiB GDDR5X). This run has no
UUID, so those numbers are **not** from `nvidia-smi`.

## Embed / retrieve — skipped

`./scripts/csd-kb-index` has **no** 1080 Ti flag; it SSHes gpu5080
timeshare (5080 exclusive-seq). Not started.

On-disk small embed (no download this run):

| Path | Size | Used? |
|---|---|---|
| `~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B` | 1.2 G | **no** — no Pascal CUDA |
| snapshot `97b0c614…` `model.safetensors` | 1.2 G | no |
| `/bulk/rag/index`, `cold-gguf`, `corpora` | empty | n/a |

Host `/usr/bin/python3` has **no** `torch`. Did not load Qwen3-Embedding
on the 5080. Did not fetch a 14B.

## RAID smoke (mechanical stripe)

Live array is **healthy RAID0**, both 3 TB HDDs, not the old degraded
RAID1. Do not wipe. Do not add/remove members.

| Field | Value |
|---|---|
| Array | `md127` `gpu5080:bulk` RAID0 |
| UUID | `8d85a4cc:d1690f1e:5b147e0f:767d6d3e` |
| Members | `sda1` + `sdb1` (HGST HDN724030ALE640, 2.7 T each) |
| Size | 5860265984 blocks = **5.46 TiB** (512k chunks) |
| State | clean, 2/2, 0 failed |
| FS | ext4 UUID `d942aafc-aa92-4fe4-8a04-732a5eda8809` `LABEL=bulk` |
| Mount | `/bulk` `rw,noatime,stripe=256` — 5.5 T, 94 G used, 5.3 T free |
| OS | NVMe `nvme0n1` LVM root (untouched) |
| `/models` | bind of `/bulk/models-hdd` (same md127) |

1 MiB `os.urandom` write + `fsync` + read-back on `/bulk/rag/smoke/`:

| Field | Value |
|---|---|
| File | `/bulk/rag/smoke/csd-1080ti-bench-20260831T202122Z.txt.1mib` |
| bytes | 1048576 |
| fsync wall | **0.028607 s** |
| sha256 | `4e3416ddf13f0be98b937c7ba285c3668895bf17aa43cf9273a932a6334a3bbb` |
| roundtrip | True |

That is a **smoke**, not sequential bandwidth. One 1 MiB fsync is not
an array performance rating.

## What would make `ok=true`

1. A kernel that actually binds Pascal (guest R535/R550/R570 after VFIO,
   not host open 610) so `nvidia-smi -L` lists a 1080 Ti UUID.
2. One CUDA op **on that UUID** (tiny torch or cuda query).
3. RAID write still succeeding (already true).

Do not install a second `nvidia.ko` on this host kernel. Do not VFIO the
5080. See [CSD-1080TI-ISOLATE.md](CSD-1080TI-ISOLATE.md).

## Re-verify (2026-08-31T20:25:07Z)

SSH `tzervas@192.168.1.251` (enp5s0, hostname `gpu5080`). Uptime 53 min.
Did **not** re-bind Pascal. Did **not** steal 5080 exclusive-seq.
Did **not** start qemu. Comfy stayed masked.

```
GPU 0: NVIDIA GeForce RTX 5080 (UUID: GPU-087267a6-14fb-0af3-da30-9a1a18523106)
```

`06:00.0` still `vfio-pci`, link 2.5 GT/s ×4. 5080 P8 0 % 12.48 W 10 MiB.
`future_hosts.gpu5080-1080ti.live` stays `false` (matches smi).

## Refs

- Role catalog (still `live: false`): [CSD-1080TI-RAG.md](CSD-1080TI-RAG.md)
- VFIO decision: [CSD-1080TI-ISOLATE.md](CSD-1080TI-ISOLATE.md)
- RAID notes: live 2026-08-31T16:11 EDT is RAID0 both 3 TB HDDs (`/bulk`).
