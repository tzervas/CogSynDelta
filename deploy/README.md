# CSD lab IaC / CaC

Git-tracked install files for the homelab Grafana stack and gpu5080
host units. Apply from a `feat/agent-harness` checkout as **autodev**.
Never `main` / `staging` / `develop` / `dev`. Never GitHub.

| Host | LAN | This tree |
|---|---|---|
| homelab | `192.168.1.170` | Grafana / Loki / Victoria / Promtail / postfix |
| gpu5080 | **`192.168.1.251`** (router static; never `.252`) | VFIO 1080 Ti, factory GPU limits, RAID notes |
| akula-prime | `192.168.1.98` | CSD user units + factory GPU limits. **Do not pause 3090 LocalAI.** |

`akula-harness` on gpu5080 (`/home/tzervas/akula-harness`) is a host
working copy, **not** a Forgejo remote. Canonical git is this repo.

Hard rules: never bind `0.0.0.0` on WAN, never VFIO the RTX 5080,
never unmask Comfy from autodev, never dump vault tokens into these
files. `config/model-router.json` `future_hosts.gpu5080-1080ti` stays
`live: false` until a guest (or host) `nvidia-smi` lists the 1080 Ti.

## Homelab (Grafana + o11y)

Install on **homelab only**. Quadlets already live on security-net
`172.30.0.x`. Do not restart LocalAI. Do not change Grafana SMTP bind
(`172.30.0.1:25`).

1. Scrape / Promtail / OTEL — [o11y/README.md](o11y/README.md)

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

   gpu5080 scrape targets are **`192.168.1.251`**. 1080 Ti is a
   `metric_relabel` on `akula-health` if `name` matches `1080 Ti` —
   do not add a scrape job while `nvidia-smi` is 5080-only.

2. Grafana alerting + dashboards — [grafana/README.md](grafana/README.md)

   ```bash
   sudo mkdir -p /data/unifi-security/deploy/config/grafana/provisioning/alerting
   sudo install -m 644 deploy/grafana/provisioning/alerting/*.yaml \
     /data/unifi-security/deploy/config/grafana/provisioning/alerting/
   sudo systemctl restart unifi-grafana.service
   secret exec TOKEN=homelab/grafana-sa-token -- python3 deploy/grafana/apply.py
   ```

3. Send-only postfix — [mail/README.md](mail/README.md)

CSD user units on homelab (Open WebUI bind `192.168.1.170`, not WAN):

```bash
./scripts/csd-install-units homelab
```

## gpu5080 (`.251`)

SSH: `ssh tzervas@192.168.1.251` / `Host gpu5080`. Capture below matches
live bind **2026-08-31**: 1080 Ti `06:00.0`+`06:00.1` = `vfio-pci`, RTX
5080 `01:00.0` = host `nvidia`. Do **not** VFIO `01:00.*`.

### Already live (re-install only if drifted)

Factory eco (persistence + `power.default_limit`; script is shared with
prime):

```bash
sudo install -m 755 deploy/nvidia-factory-limits.sh \
  /usr/local/sbin/nvidia-factory-limits.sh
sudo install -m 644 deploy/gpu5080/nvidia-factory-limits.service \
  /etc/systemd/system/nvidia-factory-limits.service
sudo install -m 644 deploy/gpu5080/akula-gpu-perf.service.d/factory-pl.conf \
  /etc/systemd/system/akula-gpu-perf.service.d/factory-pl.conf
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-factory-limits.service
# drop-in must stay: the unit file still contains -pl 396 (max); drop-in
# clears ExecStart and applies default PL + performance governor
```

VFIO 1080 Ti only (`ids=10de:1b06,10de:10ef` — never `10de:2c02`):

```bash
sudo install -m 755 deploy/gpu5080/vfio/vfio-bind-1080ti.sh \
  /usr/local/sbin/vfio-bind-1080ti.sh
sudo install -m 644 deploy/gpu5080/vfio/vfio-bind-1080ti.service \
  /etc/systemd/system/vfio-bind-1080ti.service
sudo install -m 644 deploy/gpu5080/vfio/vfio-1080ti.conf \
  /etc/modprobe.d/vfio-1080ti.conf
sudo install -m 644 deploy/gpu5080/vfio/vfio-1080ti.modules \
  /etc/modules-load.d/vfio-1080ti.conf
sudo install -m 644 deploy/gpu5080/vfio/10-vfio-1080ti.rules \
  /etc/udev/rules.d/10-vfio-1080ti.rules
sudo systemctl daemon-reload
sudo systemctl enable vfio-bind-1080ti.service
# Trust lspci -k / /dev/vfio/18, not RemainAfterExit active-after-unbind.
```

RAID / `/models` — **notes only**, do not rewrite the live array:

- [gpu5080/storage/fstab.notes.md](gpu5080/storage/fstab.notes.md)
- [gpu5080/storage/mdadm.conf](gpu5080/storage/mdadm.conf) (`gpu5080:bulk`,
  degraded `[2/1] [_U]`, read-only, live mount `/mnt/bulk-old`)
- Preserve `/models` on `sda2`. Do not steal that mount for a guest root.

### Not live — do not apply yet

| File | Why it stays off |
|---|---|
| [gpu5080/libvirt/gpu5080-1080ti-rag.xml](gpu5080/libvirt/gpu5080-1080ti-rag.xml) | qemu/ovmf not installed; placeholder qcow2; `virsh define` later |
| [gpu5080/podman/](gpu5080/podman/) `*.disabled` | Host CDI `nvidia.com/gpu=0` **is the 5080**. 1080 Ti is vfio-pci, not in `nvidia-smi` |

Do not `virsh define` until operator installs qemu/libvirt/ovmf and
creates a dedicated qcow2 (not `/models`). Do not copy
`*.container.disabled` onto `/etc/containers/systemd/` on the **host**.
Guest CDI/quadlet only after that kernel's `nvidia-smi -L` lists Pascal.

## akula-prime (3090 LocalAI)

```bash
sudo install -m 755 deploy/nvidia-factory-limits.sh \
  /usr/local/sbin/nvidia-factory-limits.sh
sudo install -m 644 deploy/prime/nvidia-factory-limits.service \
  /etc/systemd/system/nvidia-factory-limits.service
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-factory-limits.service
./scripts/csd-install-units prime
```

`nvidia-persistenced` stays **masked** on prime; the oneshot still runs
`nvidia-smi -pm 1`. Do not `docker stop akula-localai`.

## Map

| Path | Role |
|---|---|
| `deploy/o11y/` | Victoria scrape, Promtail, OTEL, quadlet drop-ins |
| `deploy/grafana/` | Dashboards, alerting, SMTP drop-in |
| `deploy/mail/` | Send-only postfix |
| `deploy/systemd/` | CSD user units (console / gateway / autodev) |
| `deploy/nvidia-factory-limits.sh` | Shared factory eco script |
| `deploy/gpu5080/` | Host units, VFIO, libvirt stub, RAID notes, disabled CDI |
| `deploy/prime/` | Prime factory-limits unit (`After=` without persistenced) |
| `config/model-router.json` | `future_hosts.gpu5080-1080ti` (`live: false`) |
