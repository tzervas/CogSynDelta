# CSD o11y taxonomy overlay (homelab unifi-grafana stack)

CaC for scrape / Loki / Jaeger labels. Allowlist:
[`docs/program/CSD-O11Y-TAXONOMY.md`](../../docs/program/CSD-O11Y-TAXONOMY.md).

Do **not** add Tempo. Traces = Jaeger Apache-2.0 `172.30.0.15:16686`
(UI published `127.0.0.1:16686`). SMTP stays send-only
`127.0.0.1` + `172.30.0.1:25`. Never bind extra `0.0.0.0` WAN ports.
Never pause 3090 LocalAI. Never unmask Comfy.

## What was live (2026-08-31)

| Piece | Homelab path | Notes |
|---|---|---|
| VictoriaMetrics scrape | `/etc/akula/vm-scrape.yml` | jobs `akula-health` (`:9108`), `akula-node` (`:9100`) |
| NVIDIA | same jobs | series `akula_gpu_*`, not `nvidia_*` / DCGM |
| Promtail | `/data/unifi-security/deploy/config/promtail.yaml` | Suricata/Zeek/UniFi syslog only until this overlay |
| Loki | `/data/unifi-security/deploy/config/loki.yaml` | unchanged; labels come from Promtail |
| Alloy | **absent** | do not invent |
| Jaeger | `akula-jaeger.container` | live unit `akula-jaeger`, not `unifi-jaeger` |
| node_exporter | prime + gpu5080 `:9100` | unit `prometheus-node-exporter`; often down |

Do not add scrape targets that were not already jobs.

## Install on homelab

```bash
sudo install -m 644 deploy/o11y/vm-scrape.yml /etc/akula/vm-scrape.yml
sudo install -m 644 deploy/o11y/promtail.yaml \
  /data/unifi-security/deploy/config/promtail.yaml
sudo install -m 644 deploy/o11y/otel-collector.yaml \
  /etc/akula/otel-collector.yaml
sudo mkdir -p /etc/containers/systemd/unifi-promtail.container.d \
  /etc/containers/systemd/akula-jaeger.container.d
sudo install -m 644 deploy/o11y/unifi-promtail.container.d/journald.conf \
  /etc/containers/systemd/unifi-promtail.container.d/journald.conf
sudo install -m 644 deploy/o11y/akula-jaeger.container.d/process-tags.conf \
  /etc/containers/systemd/akula-jaeger.container.d/process-tags.conf
sudo systemctl daemon-reload
sudo systemctl restart unifi-victoria.service unifi-promtail.service \
  akula-jaeger.service akula-otel.service
```

Do not restart LocalAI. Do not change Grafana SMTP drop-in.
Do not set `GroupAdd=systemd-journal` on unifi-promtail: the image
has no such group (podman exit 126).
