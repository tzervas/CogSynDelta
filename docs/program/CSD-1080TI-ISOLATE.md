# 1080 Ti isolate — VFIO on gpu5080 only

Not `STATUS.md`. Decision for the **gpu5080 box** (RTX 5080 + GTX 1080 Ti).
Role catalog: [CSD-1080TI-RAG.md](CSD-1080TI-RAG.md). Horizon tag: `G-1080`.
Autodev must not start this, must not power off the host, and must not
touch the 3090 on akula-prime.

Identity: **autodev**. Branch: `feat/agent-harness` (never `main` /
`staging` / `develop` / `dev`).

## Decision

**VFIO the GTX 1080 Ti at PCI `06:00.0` only. Leave the RTX 5080 on the
host NVIDIA driver.** The 5080 stays host CUDA CI / `gpu-5080.yml` /
`gpu5080.lock`. The 1080 Ti gets a guest kernel with an **older NVIDIA
driver that still supports Pascal**. Guest workload: llama.cpp / embed /
RAG retrieve-index-light. **Not** a second 14B.

Containers (Podman/Docker CDI, `--device nvidia.com/gpu=<UUID>`) are
**not** the driver split. They may pin a UUID **after** a kernel (host
or guest) actually binds that GPU. They cannot load two `nvidia.ko`
versions on one kernel.

`ok` for this doc: the decision above, plus why containers cannot split
kernel modules.

## Why containers cannot split kernel modules

Docker and Podman GPU passthrough uses the **host** `nvidia.ko`. The
NVIDIA Container Toolkit / CDI injects device nodes and libraries into
the container namespace. The kernel module stays the host's. One Linux
kernel loads **one** NVIDIA driver.

| Want | Container (CDI / UUID) | VFIO + VM |
|---|---|---|
| Hide the 5080 from a RAG process | Yes, **after** host `nvidia.ko` already bound both cards | Yes — 5080 never enters the guest |
| Run Pascal driver **and** Blackwell 610 on the same box | **No** | Yes — guest `nvidia.ko` is a different kernel |
| Two `nvidia.ko` versions on one kernel | Impossible | Not needed (two kernels) |

Blackwell 610 (host, RTX 5080) does not drive Pascal. Pascal needs an
older branch (last host-side Pascal line is R570-class; 610 is the
5080 host). Two driver versions on one kernel is **not a container
fix**. VFIO unbinds `06:00.0` from the host driver, assigns it to
`vfio-pci`, and the guest loads the Pascal-capable module.

`CUDA_VISIBLE_DEVICES` is process visibility only. It does not load a
second module, does not sandbox the rest of gpu5080, and does not make
a dropped-arch card work on 610.

## Hard rules

- Host is **gpu5080** at **`192.168.1.251`** (router static; was `.252`).
  SSH: `ssh tzervas@192.168.1.251` / `Host gpu5080`. Never `.252`.
- Scope is **this box only** (5080 + 1080 Ti). Do **not** VFIO or
  containerize the 3090 on akula-prime. 3090 LocalAI stays a **separate
  host**.
- Do **not** VFIO the RTX 5080. Host CUDA CI and `.github/workflows/gpu-5080.yml`
  need it on the host `nvidia.ko`.
- Do **not** pause 3090 LocalAI.
- Do **not** unmask Comfy.
- Do **not** power off gpu5080 from this path.
- Never GitHub bot push.
- Never commit or push `main` / `staging` / `develop` / `dev`.
- Never bind `0.0.0.0` on WAN.
- 1080 Ti role: **retrieve / index / light only**. Not train/eval
  exclusive-seq. Not PoC CUDA. Not FP8/FP4. Not `sm_120`. **Not a
  second 14B.**
- Factory eco **power limit** when the 1080 Ti is bound (guest after
  VFIO, or host if it still binds). Not max TDP, not an OC BIOS.
- Preserve **`/models` on `sda2`** if touching disks.

## Scope

| Machine | IP | GPUs | This doc |
|---|---|---|---|
| gpu5080 | `192.168.1.251` | RTX 5080 (host) + GTX 1080 Ti (`06:00.0`) | Isolate **here** |
| akula-prime | `192.168.1.98` | RTX 3090 Ti | Out of scope. Do not VFIO. Do not containerize. Do not pause LocalAI. |
| homelab | `192.168.1.170` | none | Out of scope |

Even if a future host driver still binds **both** cards on gpu5080,
isolate **on that box**: 1080 Ti RAG in its own container/VM (device
UUID only, **no** 5080), 5080 CUDA CI stays on the host/lock.

## PCI / bind

| Device | PCI | IDs | IOMMU | Driver after isolate | Owner |
|---|---|---|---|---|---|
| GTX 1080 Ti | **`06:00.0`** | `10de:1b06` EVGA `3842:6390` | **group 18** | `vfio-pci` on host; Pascal NVIDIA in **guest** | RAG guest |
| 1080 Ti HDMI audio | `06:00.1` | `10de:10ef` | **group 18** (same) | `vfio-pci` with the GPU | Guest |
| RTX 5080 | **`01:00.0`** (primary x16) | `10de:2c02` | **group 15** | host `nvidia.ko` (Blackwell 610) | Host CUDA CI / lock / Comfy (masked) |
| 5080 HD audio | `01:00.1` | `10de:22e9` | **group 15** | host `snd_hda_intel` | Host (do not stub) |

Pin by PCI / GPU UUID after `lspci -nn` / `nvidia-smi -L` on gpu5080.
Do not assume index 0 is the 5080. Do not VFIO any 5080 function.

Groups **15 and 18 are split**. Live bind of `06:00.0`+`06:00.1` does
**not** require a reboot (Intel VT-d already on; `dmar0`/`dmar1`).
Autodev must still **not** power off. Do not start `G-1080` from an
autoloop. Do not add `10de:2c02` / `10de:22e9` to `vfio-pci.ids`.

## Live bind (2026-08-31)

Host `nvidia-smi` 610.57.04 lists **only** the 5080
(`GPU-087267a6-14fb-0af3-da30-9a1a18523106`, PL 360 W, P8). Pascal is
dropped: `NVRM: nvidia.ko because it does not include the required GPU`
on `06:00.0` (`10de:1b06`). `need_vfio=true`.

IOMMU check before stub:

- group 15: `01:00.0` + `01:00.1` (5080) — **leave**
- group 18: `06:00.0` + `06:00.1` (1080 Ti) — **stub these only**

Applied without reboot:

- `modprobe vfio vfio_iommu_type1 vfio-pci`
- unbind `06:00.1` from `snd_hda_intel`; `06:00.0` was already unbound
  (610 probe failed)
- `driver_override=vfio-pci` then bind both BDFs
- `/dev/vfio/18` present; 5080 remains `nvidia`

`vfio-bind-1080ti.service` is `Type=oneshot` `RemainAfterExit=yes`.
`systemctl is-active` can stay **active** after a later unbind. Trust
`lspci -k` / `/sys/bus/pci/drivers/vfio-pci/0000:06:00.*`, not the
unit state. Re-run `/usr/local/sbin/vfio-bind-1080ti.sh` if override
is set but no kernel driver is in use.

Re-verify **2026-08-31T20:03:42Z** (`ssh tzervas@192.168.1.251`):

```
nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 5080 (UUID: GPU-087267a6-14fb-0af3-da30-9a1a18523106)

lspci -nnk -d 10de:
01:00.0 … GB203 [GeForce RTX 5080] [10de:2c02]  Kernel driver in use: nvidia
01:00.1 … GB203 HD Audio [10de:22e9]            Kernel driver in use: snd_hda_intel
06:00.0 … GP102 [GeForce GTX 1080 Ti] [10de:1b06] Kernel driver in use: vfio-pci
06:00.1 … GP102 HDMI Audio [10de:10ef]          Kernel driver in use: vfio-pci
```

`/dev/vfio/18` present. Comfy **masked** / inactive. `/models` still
`/dev/sda2` ext4 `rw,noatime`. `gpu5080.lock` idle. 3090 LocalAI on
akula-prime left running (`akula-localai` healthy). Did not power off.
Did not VFIO `01:00.*`.

Persist (ids are 1080 Ti only):

| Path | Role |
|---|---|
| `/usr/local/sbin/vfio-bind-1080ti.sh` | ID-checked bind; refuses 5080 IDs |
| `/etc/systemd/system/vfio-bind-1080ti.service` | enabled oneshot |
| `/etc/modprobe.d/vfio-1080ti.conf` | `ids=10de:1b06,10de:10ef` |
| `/etc/modules-load.d/vfio-1080ti.conf` | vfio modules |
| `/etc/udev/rules.d/10-vfio-1080ti.rules` | `driver_override` on `06:00.*` only |

Git CaC (same files): `deploy/gpu5080/vfio/`. Re-apply from
[deploy/README.md](../../deploy/README.md) if the host drifts.

Libvirt XML stub (no disk image this run; qemu/ovmf **not** installed):

- Canonical: [deploy/gpu5080/libvirt/gpu5080-1080ti-rag.xml](../../deploy/gpu5080/libvirt/gpu5080-1080ti-rag.xml)
- Host working copy (not a git remote): `/home/tzervas/akula-harness/config/qemu/gpu5080-1080ti-rag.xml`
- Docs mirror: [gpu5080-1080ti-rag.xml](gpu5080-1080ti-rag.xml)

Guest NVIDIA: R535 / R550 / last R570 (Pascal `sm_61`). **Not** host 610.
Factory eco PL **in the guest** after that driver binds:
`nvidia-smi -pl` at `power.default_limit`. Host cannot `-pl` a vfio
device. Do not flash an OC BIOS. `/models` on `sda2` left mounted.

Guest **must** run stealth-leds (OpenRGB oneshot, never `--server`) and
`GPULogoBrightness=0` at boot. Host stealth does not survive a guest
Pascal bind — the GeForce logo comes back. XML stub comments that
requirement; no guest image this run.

`live: false` until the **guest** `nvidia-smi` lists the 1080 Ti.

## Host vs guest

```
gpu5080 host (192.168.1.251)
  ├── RTX 5080  → host nvidia.ko 610
  │                 gpu5080.lock / exclusive-seq
  │                 Forgejo GPU Actions (gpu-5080.yml)
  │                 Comfy stays masked
  └── GTX 1080 Ti 06:00.0 → vfio-pci
        └── guest (own kernel)
              older NVIDIA driver (Pascal)
              llama.cpp / embed
              RAG retrieve-index-light
              factory eco PL
              stealth-leds oneshot (never OpenRGB --server)
              RAG store on mechanical volume once RAID exists
```

Guest stack:

- Older NVIDIA userspace + `nvidia.ko` that **still lists Pascal**
  (`sm_61`). Not the host 610 tree.
- llama.cpp CUDA built for `sm_61` (or CPU-offload). Small embed.
  Light legacy GGUF only.
- RAG store on the **mechanical volume once RAID exists**. Until then,
  do not invent a second store on the 5080 SSD layout.
- **Not** LocalAI 14B. **Not** vLLM. **Not** PoC CUDA. **Not**
  `gpu-5080.yml`.
- Stealth: same `cabal-stealth-leds` oneshot as the host, **inside the
  guest**, never OpenRGB `--server` (CVE-2026-59682 / 59683).

When bound: set factory eco PL on the 1080 Ti (`nvidia-smi -pl` at the
card's factory ECO wattage). Measure; do not flash an OC BIOS from this
path.

## Sandbox

RAG must not see 5080 VRAM or the rest of gpu5080 except the RAG HDD
mount.

| Surface | Policy |
|---|---|
| 5080 VRAM / CUDA | Guest never receives the 5080. No `--device nvidia.com/gpu=all` on RAG. No CUDA IPC / MPS across cards. |
| Host processes / lock | RAG does not take `gpu5080.lock` to “borrow” the 5080. Host CI stays exclusive-seq on the 5080. |
| Disk | RAG HDD mount only (mechanical RAID when it exists). **Preserve `/models` on `sda2`.** Do not reformat, do not steal that mount for the guest root. |
| Network | LAN only. No `0.0.0.0` WAN. 3090 LocalAI remains on akula-prime; do not proxy it into the RAG guest. |
| Qdrant | Collection `akula-csd-kb` stays **1024-d**. Never mix 384-d. |

No MIG (none of 3090 / 5080 / 1080 Ti implement it). Do not claim MIG.

## Containers only after bind

CDI / `--device nvidia.com/gpu=<UUID>` is a **visibility** control on
whatever `nvidia.ko` is already loaded in **that** kernel.

1. **Recommended:** VFIO `06:00.0` → guest Pascal driver binds the 1080
   Ti. Then, *inside the guest*, a container may take that UUID only.
2. **Fallback if host 610 is not yet in play and host `nvidia.ko` still
   binds Pascal:** RAG container on the host with **1080 Ti UUID only**
   (no 5080). 5080 CUDA CI stays on the host/lock. This still does not
   run two driver versions; it only hides a device the host already
   drives.
3. **Forbidden:** a host container “with an old Pascal driver image”
   while host 610 owns the module. The image’s userspace cannot revive
   a dropped arch. Do not ship that as a fix.

Do not write CDI unit files, compose files, or `--gpus all` for the
1080 Ti until `nvidia-smi` (host **or** guest) lists that UUID as
bound.

## Role vs 5080 vs 3090

| Workload | Where |
|---|---|
| RAG retrieve / index / small embed / light GGUF | 1080 Ti guest (this isolate) |
| Keyword-cpu assist | 1080 Ti overflow; CPU remains valid |
| Train / eval / GPU CI / FP8 / FP4 / `sm_120` | **5080 host** exclusive-seq |
| Comfy / media | **5080**, operator unmask only — do not unmask from this path |
| `local/code` autodev 14B | **3090** LocalAI on akula-prime (stay loaded) |

Do not stop LocalAI for RAG. Do not steal 5080 exclusive-seq VRAM for
index. Combined 24+16+11 GiB is **not** a license to dual-14B or to
split a dense 14B onto Pascal.

## Promote (operator, not autodev)

1. Card seated. Factory cooler. Host stays up (no remote power-off).
2. Confirm PCI **`06:00.0`** is the 1080 Ti (`lspci -nn`). 5080 is not
   that slot. **Done** (GP102 EVGA `3842:6390`).
3. VFIO bind of `06:00.0` (+ group audio). 5080 remains on host
   `nvidia`. **Done** 2026-08-31 (live bind + persist; no reboot;
   re-verified `lspci -k` `vfio-pci`).
4. Guest: install qemu/libvirt/ovmf, define
   [gpu5080-1080ti-rag.xml](gpu5080-1080ti-rag.xml), Pascal driver,
   llama.cpp/embed, factory eco PL, guest stealth-leds oneshot, RAG
   HDD mount only. Preserve `/models` on `sda2`. **Not this run**
   (XML stub only; qemu/ovmf not installed).
5. Only after guest bind: optional guest container with **that UUID**.
   Then promote `future_hosts.gpu5080-1080ti` per
   [CSD-1080TI-RAG.md](CSD-1080TI-RAG.md).

Until `nvidia-smi` on gpu5080 (or the guest) lists the 1080 Ti, keep
`live: false`. Observe only.

## References

- Role catalog: [CSD-1080TI-RAG.md](CSD-1080TI-RAG.md)
- Libvirt stub: [deploy/gpu5080/libvirt/gpu5080-1080ti-rag.xml](../../deploy/gpu5080/libvirt/gpu5080-1080ti-rag.xml)
- ADR-0016 / horizon: [GPU-SHARE.md](GPU-SHARE.md) (`G-1080`)
- Placement: [CODEX-OPS.md](../CODEX-OPS.md) · [WHO-RUNS-WHAT.md](WHO-RUNS-WHAT.md)
- 5080 host CI: `.github/workflows/gpu-5080.yml` (must keep the 5080
  on the host driver)
- Router stub: `config/model-router.json` `future_hosts.gpu5080-1080ti`
