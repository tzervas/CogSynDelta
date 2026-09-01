# Factory GPU eco (clocks + default power limits)

Not a custom undervolt. Not `STATUS.md`. Operator policy for **idle
watts** on akula-prime (3090 Ti) and gpu5080 (5080; 1080 Ti when
`nvidia-smi` lists it).

Identity: **autodev**. Branch: `feat/agent-harness` (never `main` /
`staging` / `develop` / `dev`). Workflow: `.grok/workflows/csd-gpu-eco.rhai`.

## Policy

Factory **clocks** and factory **power limits** only:

- Persistence on: `nvidia-smi -pm 1`
- Unlock `--lock-gpu-clocks` / `--lock-memory-clocks` / `-ac`
- `-pl` = that GPU’s `power.default_limit` (query, do not invent)
- Idle must be allowed to drop P-states (**P8 / P5**). High wattage
  only when CUDA or LocalAI is actually running.
- Do **not** drop the 3090 PL below default so `local/code` cannot boost.
- Do **not** raise PL to `power.max_limit` “for headroom.”

## Hard rules

- Do **not** pause 3090 LocalAI (`docker stop akula-localai`).
- Do **not** unmask Comfy (`akula-comfyui.service`).
- Do **not** power off either host.
- Do **not** install the `nvidia-settings` GUI.
- Never bind `0.0.0.0` on WAN. LocalAI stays `127.0.0.1:8080` (+ LAN
  socat on `192.168.1.98`, not WAN).
- Never GitHub bot push.
- If `192.168.1.251` SSH fails, still eco the 3090 and record the gap.
- Persistence + default PL on **each UUID `nvidia-smi` lists**.
  lspci-only cards (1080 Ti until the driver binds) are a gap, not a
  target for `nvidia-smi -i`.

## Hosts

| Host | IP | GPU | UUID (when listed) | default PL |
|---|---|---|---|---|
| akula-prime | 192.168.1.98 | RTX 3090 Ti | `GPU-642a8f8b-7a80-6b5e-76d4-034b742a1480` | 450 W |
| gpu5080 | 192.168.1.251 | RTX 5080 | `GPU-087267a6-14fb-0af3-da30-9a1a18523106` | 360 W |
| gpu5080 | 192.168.1.251 | GTX 1080 Ti (PCI `06:00.0` GP102) | **not in `nvidia-smi`** | n/a |

SSH: `Host gpu5080` / `tzervas@192.168.1.251`. Not `.252`.

## Apply (per visible UUID)

Run as root (or passwordless `sudo -n` on gpu5080). Do not stop Docker.

```bash
# persistence
nvidia-smi -pm 1

# each UUID nvidia-smi currently lists
nvidia-smi --query-gpu=uuid,power.default_limit --format=csv,noheader \
  | while IFS=, read -r uuid defpl; do
      uuid="${uuid#"${uuid%%[![:space:]]*}"}"
      uuid="${uuid%"${uuid##*[![:space:]]}"}"
      defpl=$(echo "$defpl" | awk '{print $1}')
      [ -n "$uuid" ] || continue
      nvidia-smi --reset-gpu-clocks -i "$uuid" || true
      nvidia-smi --reset-memory-clocks -i "$uuid" || true
      nvidia-smi --reset-applications-clocks -i "$uuid" || true
      nvidia-smi -pl "$defpl" -i "$uuid" || true
    done
```

Do **not** use `nvidia-smi -ac …` or `--lock-gpu-clocks`. `-ac` is
deprecated on current drivers; reset is the unlock.

gpu5080 (passwordless sudo, Comfy lock-wrap not mask):

```bash
ssh -o BatchMode=yes gpu5080 \
  'sudo -n nvidia-smi -pm 1
   nvidia-smi --query-gpu=uuid,power.default_limit --format=csv,noheader
   sudo -n systemctl start nvidia-factory-limits.service
   systemctl is-enabled akula-comfyui.service   # expect: generated/wrapped, not masked'
```

Skip exclusive CUDA work if `gpu5080.lock` is **flock-held**. An empty
lock file with no flock is idle; eco (`-pm` / `-pl` / clock reset) is
not exclusive-seq and may run at util=0.

## Durable oneshot (already on both hosts)

Do not install `nvidia-settings`. The oneshot is:

- script: `/usr/local/sbin/nvidia-factory-limits.sh`
- unit: `/etc/systemd/system/nvidia-factory-limits.service`
- enable: `WantedBy=multi-user.target`

Unit:

```ini
[Unit]
Description=NVIDIA factory clocks and default power limits
Documentation=man:nvidia-smi(1)
After=systemd-modules-load.service nvidia-persistenced.service
ConditionPathExists=/usr/bin/nvidia-smi

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/nvidia-factory-limits.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Script body is the apply loop above (`-pm 1`, reset clocks, `-pl`
default). gpu5080 also has `akula-gpu-perf.service` with drop-in
`factory-pl.conf` so that unit does **not** `-pl 396` (max). Prime
keeps `nvidia-persistenced.service` **masked**; persistence is still
`nvidia-smi -pm 1` (Enabled). Do not unmask persistenced on prime as
part of this path.

## Verify

```bash
nvidia-smi --query-gpu=index,uuid,name,persistence_mode,pstate,power.draw,power.limit,power.default_limit,clocks.gr,clocks.mem,utilization.gpu --format=csv

# LocalAI must still answer (do not print the key)
secret exec TOKEN=gpu/localai-api-key -- \
  sh -c 'curl -sS -m 8 -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8080/v1/models'
```

Expect:

| Host | persistence | PL vs default | Idle (util=0) | Notes |
|---|---|---|---|---|
| 3090 Ti | Enabled | equal (450 W) | P5/P8 when CUDA idle | Resident LocalAI may sit P0; draw must stay **well below 450 W** |
| 5080 | Enabled | equal (360 W) | **not forced P0** (P8 observed) | util=0 → ~12 W |
| 1080 Ti | — | — | — | lspci only until bound |

LocalAI bind: `127.0.0.1:8080` (docker) + `192.168.1.98:8080` (socat).
Not `0.0.0.0`.

## Gaps (2026-08-31)

- **1080 Ti** is PCI `06:00.0` (`10DE:1B06` GP102, EVGA `3842:6390`)
  on gpu5080 but **unbound** (`enable=0`, no `driver` symlink, not in
  `nvidia-smi -L`). Eco the 5080 UUID only. Do not power off to seat.
  Do not bind the Pascal device from this path. Catalog:
  [CSD-1080TI-RAG.md](CSD-1080TI-RAG.md) (`future_hosts`, `live: false`).
- Prime clock reset / `-pl` as user `kang` returns “Insufficient
  Permissions”; factory oneshot already applied as root. `-pm 1`
  succeeds without sudo. Re-run `systemctl start nvidia-factory-limits`
  after a driver reload if clocks are locked again.

## Related

- Placement: [GPU-POOL.md](GPU-POOL.md) · ops: [CODEX-OPS.md](../CODEX-OPS.md)
- 1080 Ti catalog: [CSD-1080TI-RAG.md](CSD-1080TI-RAG.md)
- Who runs what: [WHO-RUNS-WHAT.md](WHO-RUNS-WHAT.md)
