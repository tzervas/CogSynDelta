# 1080 Ti RAG guest (gpu5080-1080ti-rag)

Debian 13 genericcloud on NVMe (`/var/lib/libvirt/images/gpu5080-1080ti-rag.qcow2`).
VFIO `06:00.0`+`06:00.1` only. Host RTX 5080 stays `nvidia` at `01:00.0`.
`/models` is untouched (`md127[/models-hdd]`).

## SSH

| | |
|---|---|
| User | `tzervas` |
| Key | same `id_ed25519` as gpu5080 (`comfyui-homelab`) |
| LAN IP | **`192.168.1.243`** (macvtap `enp5s0`, MAC `52:54:00:10:80:71`) |
| Hostname | `gpu5080-1080ti-rag` |
| From LAN | `ssh tzervas@192.168.1.243` |
| Alias | `ssh gpu5080-1080ti` (prime `~/.ssh/config`) |
| Host console | `ssh gpu5080` then `sudo virsh console gpu5080-1080ti-rag` |

Passwordless sudo in the guest. No password auth. Do not bind `0.0.0.0` on WAN.

NAT `ennat0` (`52:54:00:10:80:72`, libvirt `default` / `virbr0`) is a host-only fallback; LAN is the documented path.

## Guest NVIDIA

Tesla **535.274.02** proprietary (`nvidia-tesla-535-kernel-dkms` + `nvidia-tesla-535-driver`).
Not host 610, not `nvidia-open`. Secure Boot off so DKMS can load.

```
nvidia-smi -L
GPU 0: NVIDIA GeForce GTX 1080 Ti (UUID: GPU-4df3ba11-fd12-3550-bb97-ad00b0b00569)
```

Factory eco PL 250 W (`power.default_limit`). Role: retrieve-index-light / RAG only.

## Stealth (boot oneshot)

`cabal-stealth-leds.service` is enabled. OpenRGB CLI Off/Direct `000000`, **never** `--server`.
`GPULogoBrightness=0` is attempted after bind. Copy of `akula-ai-platform/scripts/stealth-leds`.

## Autostart

`virsh autostart gpu5080-1080ti-rag` is on. libvirt nets `default` and `lan-enp5s0` autostart.

## Rebuild

`sudo /home/tzervas/akula-harness/config/qemu/1080ti-rag/provision-1080ti-rag.sh`
(from this tree: `deploy/gpu5080/guest/provision-1080ti-rag.sh`). Do not wipe `/models`.
