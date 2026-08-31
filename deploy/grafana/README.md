# Grafana provisioning (homelab unifi-grafana)

CaC for unified alerting on the existing UniFi Grafana 11.4
(`172.30.0.13:3000`, https://grafana.vectorweight.com).

Do **not** add Tempo. SMTP stays send-only `127.0.0.1` + `172.30.0.1:25`.
Contact point `maintainers-email` only. Never bind extra `0.0.0.0` WAN ports.
Never pause 3090 LocalAI. Never unmask Comfy.

## Alerting

| File | What |
|---|---|
| `provisioning/alerting/contact-points.yaml` | `maintainers-email` → `maintainers@vectorweight.com` |
| `provisioning/alerting/policies.yaml` | root receiver that contact point; `group_by: [group, host]` |
| `provisioning/alerting/rules.yaml` | taxonomy-labeled rules; skip series that are not in VM |

Live rules (only if the Prom series exist):

- `CSDServiceDownByGroup` — `akula_backend_up{backend=~"localai|gpu|nl-router"} == 0`
- `CSDDiskLow` — gpu5080 root `< 10%` via `last_over_time(node_filesystem_*[7d])`
- `CSDGPULockStale` — `akula_timeshare_busy{host="gpu5080"} == 1` for 2h
- `CSDScaleLadderRungGreen` — `csd_scale_ladder_rung_green == 1` (from `benchmark_results/scale_ladder.json` / lab `/metrics`; `noDataState: OK`; all rungs `green: false` today — not a fake sat)
- `CSDNeedGrok` — `csd_need_grok == 1` (from `/akula-data/cabal/csd-need-grok.json` / lab `/metrics` + mtime; `for: 2m`; `noDataState: OK`; pages `maintainers-email` once per firing, not per loop tick; does **not** call hosted Grok)

Not provisioned (no Prom series): postfix deferred, Forgejo runner offline.
`up{group,ns,host}` is scraped from security-net and is currently `0` (LAN
exporters unreachable from `172.30.0.12`); do not alert on it (flood).

## Install on homelab

```bash
sudo mkdir -p /data/unifi-security/deploy/config/grafana/provisioning/alerting
sudo install -m 644 deploy/grafana/provisioning/alerting/*.yaml \
  /data/unifi-security/deploy/config/grafana/provisioning/alerting/
sudo systemctl restart unifi-grafana.service
```

Do not restart LocalAI. SMTP drop-in stays `unifi-grafana.container.d/smtp.conf`.
One contact-point test via Grafana API after reload, not a flood.

## Dashboards / library / playlist (API apply)

Folder **CSD lab** live uid `ffwv6ar8e06pse`. Template vars **env, host, ns,
group, service, path** are `label_values` queries (not IP lists).
Do not add `path` to every PromQL selector: `akula-node` omits taxonomy
`path` (collides with `node_filesystem` mount `path`). No Tempo. SMTP bind
unchanged (`GF_SMTP_HOST=172.30.0.1:25`).

```bash
secret exec TOKEN=homelab/grafana-sa-token -- python3 deploy/grafana/apply.py
```

`--dry-run` writes `deploy/grafana/dashboards/*.json` only.

| Kind | UID |
|---|---|
| Dashboards | `csd-service-map`, `csd-gpus`, `csd-logs`, `csd-traces`, `csd-investigate` |
| Library panels | `csd-lib-gpu-vram-3090`, `csd-lib-gpu-vram-5080`, `csd-lib-loki-errors`, `csd-lib-vm-up` |
| Playlist | **CSD-investigate** `csd-investigate-pl` (60s) |
| Jaeger DS | `csd-jaeger` → `http://172.30.0.15:16686` (LAN proxy, no Tempo) |
| Correlations | Loki `trace_id`→Jaeger; Loki `service`→VM `up`; VM `service`→Loki; VM `service`→Jaeger |

GPU panels use live `akula_gpu_*`. Investigate does not invent RED
(`http_requests_total` is absent). Loki DS is file-provisioned read-only,
so the `trace_id` derived field could not be patched (403); correlations
and dashboard data links still open Jaeger.
