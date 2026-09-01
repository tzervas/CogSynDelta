# Comfy queue hook (gpu5080 lock wrap)

Not tensor-wire G-POOL. Not 256 specialists. Not hard-pause 3090.

Wrapping the whole `akula-comfyui` ExecStart with `gpu5080-lock` is
**too coarse**: the flock would be held for the HTTP daemon lifetime
and autodev CUDA CI (`gpu-5080.yml` `LOCK=…`) would wait until Comfy
is stopped.

## What runs

| Piece | Path |
|---|---|
| Mutex helper (same as CUDA CI) | `/home/tzervas/akula-harness/scripts/gpu5080-lock` |
| Queue-at-start wrapper | `/home/tzervas/akula-harness/scripts/akula-comfyui-locked` |
| systemd drop-in | `/etc/systemd/system/akula-comfyui.service.d/gpu5080-lock.conf` |
| LAN HTTP (OWUI / Caddy) | `192.168.1.251:8188` only |
| GPU Comfy (after lock) | `127.0.0.1:8189` (podman publish, never WAN) |

## Hook

1. systemd starts the wrapper. HTTP is up immediately. **No CUDA.**
2. Idle GET `/system_stats` / `/queue` / `/healthz` → stub JSON.
3. POST `/prompt` (and `/ws`, `/object_info`, …) → `gpu5080-lock -- sleep infinity`, then `podman run` with `--device nvidia.com/gpu=all`.
4. If autodev holds the lock, the HTTP process **waits** (finish-then-free). TimeoutStart does not need to cover that wait.
5. After queue empty + `COMFY_IDLE_SEC` (90s) and no spliced clients: `podman rm -f` and kill the lock helper.

OWUI `IMAGE_GENERATION_ENGINE=comfyui` still uses the Comfy LAN URL
(`COMFYUI_BASE_URL=http://172.32.0.1:8188` → homelab media-proxy →
`192.168.1.251:8188`). Do not switch engines. Do not restart OWUI
unless that env must change.
