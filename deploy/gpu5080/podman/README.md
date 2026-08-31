# 1080 Ti RAG Podman / CDI — **disabled stubs**

Not live. Do **not** copy these onto gpu5080 host
`/etc/containers/systemd/` or `/etc/cdi/`.

Host `nvidia-smi -L` on `192.168.1.251` lists **only** the RTX 5080
(`GPU-087267a6-14fb-0af3-da30-9a1a18523106`). Host CDI (`nvidia-ctk
cdi list`, 2026-08-31):

- `nvidia.com/gpu=0`
- `nvidia.com/gpu=GPU-087267a6-14fb-0af3-da30-9a1a18523106`
- `nvidia.com/gpu=all`

Every one of those is the **5080**. The GTX 1080 Ti is `vfio-pci` at
`06:00.0` (IOMMU group 18). Podman cannot load a second `nvidia.ko`.
`--device nvidia.com/gpu=all` on a RAG container would pass the 5080.

| File | Use |
|---|---|
| `gpu5080-1080ti-rag.container.disabled` | Guest quadlet template. Rename to `.container` **inside the VFIO guest** after that kernel’s `nvidia-smi` lists Pascal. Pin `Device=nvidia.com/gpu=<GUEST-UUID>`. |
| `nvidia-1080ti.cdi.json.disabled` | Guest CDI placeholder (`kind` is **not** `nvidia.com/gpu`). Empty `devices` until a UUID exists. |

Never: host install, `nvidia.com/gpu=all`, 5080 UUID, `0.0.0.0` WAN,
LocalAI 14B image, pause 3090 LocalAI, unmask Comfy.
