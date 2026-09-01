# GPU keep-up — services stay up; lock is the mutex

Not `STATUS.md`. Not tensor-wire **G-POOL**. Not 256 specialists.
Operator policy so LocalAI, Open WebUI, CUDA CI, and Comfy **keep
running**. Contention is a **queue**, not `docker stop` / `systemctl
mask`.

Identity: **autodev**. Branch: `feat/agent-harness` (never `main` /
`staging` / `develop` / `dev`). gpu5080 LAN: **`192.168.1.251`**
(never `.252`).

## Policy

Services **stay up**. `gpu5080.lock` is the mutex. Timeshare JSON on
**gpu5080** is the queue.

| Mechanism | Path | Role |
|---|---|---|
| Mutex | `gpu5080.lock` via `/home/tzervas/akula-harness/scripts/gpu5080-lock` | flock; one exclusive-seq holder |
| Queue | `AKULA_TIMESHARE=/home/tzervas/akula-harness/state/gpu-timeshare.json` on gpu5080 | worker queue |

Prime `/akula-data/cabal/gpu-timeshare.json` is **not** the 5080
worker queue. Enqueue on gpu5080, or the job never claims.

Priority on the 5080 (highest first):

1. **autodev** CUDA / Forgejo GPU CI (`with-gpu-5080 --mode exclusive-seq`)
2. **OWUI image / edit** (Comfy) — waits on the lock
3. **idle share-small** helpers (`helper_cap` ~6 GiB; free ≥ 8192 MiB)

**Finish-then-free.** The holder runs to completion. When the lock
releases, Comfy or the next autodev job acquires the same wrap.

## Route

| Box | IP | GPU | Job |
|---|---|---|---|
| akula-prime | `192.168.1.98` | RTX 3090 Ti | `local/code` + **OWUI chat**. One GGUF. |
| gpu5080 | **`192.168.1.251`** | RTX 5080 | CUDA / CI / **Comfy** exclusive-seq via lock |
| 1080 Ti guest | `192.168.1.243` | GTX 1080 Ti | RAG / retrieve-index-light. Does **not** take `gpu5080.lock`. |
| homelab | `192.168.1.170` | none | Open WebUI `ai.vectorweight.com` + Forgejo CPU Actions |

OWUI **chat** uses the 3090 and **must not take the 5080**.
OWUI **image / edit** waits on `gpu5080-lock`. Do not route chat
completions onto Comfy or 5080 CUDA.

Parallelism = 3090 inference **plus** one 5080 exclusive-seq job.
Not two jobs on one card. Not dual 14B.

## Hard rules

- Do **not** hard-pause 3090 LocalAI.
- Do **not** `docker stop akula-localai`.
- Open WebUI at https://ai.vectorweight.com **stays up**.
- Do **not** `systemctl mask` / `stop` `akula-comfyui.service` to
  “free” the 5080.
- Do **not** unmask / start Comfy until **ExecStart** (or a drop-in)
  acquires `/home/tzervas/akula-harness/scripts/gpu5080-lock` the
  **same wrap** as CUDA CI (`with-gpu-5080`).
- Autodev CUDA/CI still uses that lock. Comfy **waits**. When the
  lock releases, Comfy or the next autodev job runs.
- Never dual 14B.
- Never bind `0.0.0.0` on WAN. LocalAI stays `127.0.0.1:8080` (+ LAN
  socat on `192.168.1.98`).
- Never GitHub bot push.
- Never treat tensor-wire G-POOL or 256 specialists as live.

## Lock wrap (Comfy = CUDA CI)

CUDA CI already wraps:

```bash
AKULA=/home/kang/code/personal/tzervas/akula-ai-platform
"$AKULA/scripts/with-gpu-5080" --mode exclusive-seq -- \
  bash -lc '…'
# ssh gpu5080 → /home/tzervas/akula-harness/scripts/gpu5080-lock -- bash -lc …
```

`.github/workflows/gpu-5080.yml` keeps `LOCK=/home/tzervas/akula-harness/scripts/gpu5080-lock`.

Wrapping the **whole** quadlet ExecStart with `gpu5080-lock` is too
coarse (flock held for the HTTP lifetime; autodev starves). Queue
hook instead: HTTP stays up, GPU alloc on first prompt.

```ini
# /etc/systemd/system/akula-comfyui.service.d/gpu5080-lock.conf
[Service]
Type=simple
TimeoutStartSec=90
ExecStart=
ExecStart=/home/tzervas/akula-harness/scripts/akula-comfyui-locked
```

Wrapper: `/home/tzervas/akula-harness/scripts/akula-comfyui-locked`
(CaC: `deploy/gpu5080/akula-comfyui-locked`). LAN `192.168.1.251:8188`.
GPU Comfy publishes `127.0.0.1:8189` only after
`gpu5080-lock -- sleep infinity`. Idle (90s empty queue) stops the
container and releases the flock. Notes:
[deploy/gpu5080/comfy-queue-hook.md](../../deploy/gpu5080/comfy-queue-hook.md).

Unmask only after that wrap exists. Until the wrap is in ExecStart
(or a drop-in), leave Comfy disabled **because the wrap is missing**,
not because autodev prefers mask as a scheduler.

Image/edit from OWUI: enqueue timeshare on gpu5080; wait for the
lock; do not kill an in-flight autodev job; do not steal VRAM.

Share-small only when the lock is idle **and** `nvidia-smi` free ≥
8192 MiB and the helper is ≤ ~6 GiB. Otherwise refuse. No MIG.

## What stays loaded

| Service | Host | Keep-up |
|---|---|---|
| LocalAI `akula-localai` (`local/code`) | akula-prime | Always. Chat + autodev code. |
| Open WebUI `ai.vectorweight.com` | homelab | Always. Chat → 3090. Image → 5080 lock. |
| Forgejo GPU runner `gpu5080-tzervas` | gpu5080 | Queued on the lock. |
| `akula-comfyui.service` | gpu5080 | Enabled only with lock wrap; waits when autodev holds. |
| 1080 Ti RAG guest | `192.168.1.243` | Own device. No 5080 lock. |

## Not this work

- Tensor-wire **G-POOL** (Stage B `pool.enabled=false`). See
  [GPU-POOL.md](GPU-POOL.md).
- **256 specialists** intern pool. Not live. See
  [GPU-SHARE.md](GPU-SHARE.md) (horizon).
- Hard-pause / mask as a scheduling tool (superseded by this file).

## Related

- Ops: [CODEX-OPS.md](../CODEX-OPS.md) (lock wrap, not mask)
- Eco (clocks/PL; still no docker-stop): [CSD-GPU-ECO.md](CSD-GPU-ECO.md)
- Placement: [GPU-POOL.md](GPU-POOL.md) · flex: [CSD-CLUSTER-FLEX.md](CSD-CLUSTER-FLEX.md)
- 1080 Ti RAG: [CSD-1080TI-RAG.md](CSD-1080TI-RAG.md)
- Who runs what: [WHO-RUNS-WHAT.md](WHO-RUNS-WHAT.md)
