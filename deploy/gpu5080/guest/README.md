# 1080 Ti guest stealth (not live)

Host-only OpenRGB Off does not keep the EVGA 1080 Ti GeForce logo dark.
`06:00.0` is `vfio-pci`; the guest Pascal `nvidia.ko` re-lights the logo
on bind. Same oneshot as the host, **inside the guest**, never
`OpenRGB --server` (CVE-2026-59682 / 59683).

| File | Role |
|---|---|
| `cabal-stealth-leds.service` | Guest systemd oneshot. `After=nvidia-persistenced`, `Before=gpu5080-1080ti-rag`. |
| `cloud-init-stealth.yaml` | Seed the unit + i2c-dev + enable at first boot. |
| `gpu5080-1080ti-rag.container.d/stealth.conf` | Quadlet drop-in so RAG CUDA waits for stealth. |
| `../stealth-leds` | Shared script (`install` to `/usr/local/sbin/cabal-stealth-leds`). |

qemu/ovmf/libvirt are installed on gpu5080 (2026-08-31, no reboot).
Still no guest image. Do not `virsh define` until a dedicated qcow2
exists (not `/models`). Do not copy the quadlet drop-in onto the
**host** `/etc/containers/systemd/`. Do not unbind the RTX 5080
(`01:00.0`). Operator confirms the logo visually; SSH cannot see it.
