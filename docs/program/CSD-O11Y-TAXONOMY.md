# CSD observability label taxonomy

Stable, low-cardinality labels for Grafana / Loki / VictoriaMetrics / Jaeger.
This file is the **allowlist**. Do not invent keys, values, VLANs, or series.

**Taxonomy only.** Do not provision dashboards, datasources, or alerts here.
Later: `/csd-grafana-o11y` then `args.apply=true` (see
[HANDOFF-NEXT.md](HANDOFF-NEXT.md)). Stack facts: [O11Y.md](O11Y.md).
Hosts/GPUs: [CODEX-OPS.md](../CODEX-OPS.md). Coupling: [HARNESS.md](HARNESS.md).
Who runs what: [WHO-RUNS-WHAT.md](WHO-RUNS-WHAT.md). Vault:
[CSD-SECRETS.md](CSD-SECRETS.md).

## Hard rules

- Never bind `0.0.0.0` on WAN. Lab HTTP binds the host LAN IP or loopback.
- Never Tempo (AGPL). Traces = Jaeger Apache-2.0, LAN `172.30.0.15:16686`.
- SMTP send-only `127.0.0.1` + `172.30.0.1:25`. Contact `maintainers-email`.
- Never pause 3090 LocalAI. Never unmask Comfy. Never GitHub bot push.
- Never `main` / `staging` / `develop` / `dev` as a working branch.
- Never write operator or model vaults. Never mix 384-d Qdrant.
- Do not duplicate the UniFi security-net stack. Do not add a VLAN this run.
- Metric / log labels are **bounded** and contain **no user content** (P1-10).

Forgejo Actions `runs-on` labels (`self-hosted`, `linux`, `x64`, `podman`,
`compute-cpu`, `host-homelab`, `gpu`, `5080`, `host-gpu5080`) are a **different
vocabulary**. Do not copy them into Prometheus / Loki.

---

## Keys and allowed values

Seven keys. All required on CSD-owned series. Values are closed sets except
`service` (closed catalog below) and `path` (derived).

| Key | Allowed values | Cardinality | Why |
|---|---|---|---|
| `env` | `lab` \| `ci` \| `prod-intent` | 3 | Split day-to-day lab from Actions from “must stay up” public frontends |
| `host` | `prime` \| `gpu5080` \| `homelab` | 3 | Short form of akula-prime / gpu5080 / homelab. Not FQDNs, not IPs |
| `ns` | `csd` \| `webui` \| `media` \| `mail` \| `git` \| `o11y` \| `runner` | 7 | Logical namespace, not a Kubernetes ns, not a new VLAN |
| `service` | full systemd **or** compose/quadlet **or** Forgejo runner name | catalog | One string per process. No aliases |
| `kind` | `api` \| `gpu` \| `ui` \| `mta` \| `runner` \| `index` \| `gateway` | 7 | What it *is*, so filters survive rename |
| `group` | `csd-autodev` \| `akula-chat` \| `akula-media` \| `forgejo-ci` \| `grafana-stack` | 5 | Set that fails/recovers together |
| `path` | `{env}.{ns}.{kind}` or `{env}.{host}.{kind}` when `ns=runner` | derived | Grafana/Loki filter; never free-form |

Do **not** add `instance`, `job`, `pod`, `container_id`, `commit`, `pr`,
`user`, `model`, `prompt`, or file paths as label keys. Those belong in
log **fields** or trace **tags** with explicit bounds, not index labels.

### `env`

| Value | Use |
|---|---|
| `lab` | Autodev, LocalAI, lab console, gateway, RAG index, GPU helpers |
| `ci` | Forgejo Actions runners and the jobs they execute |
| `prod-intent` | Public hostnames and alert path (Grafana, Loki, VM, Jaeger UI, status, Forgejo, mail, Open WebUI, Caddy). Homelab is not “prod”; treat these as if they were for uptime and SMTP |

### `host`

| Value | Machine | IP (LAN) | GPU |
|---|---|---|---|
| `prime` | akula-prime | `192.168.1.98` | RTX 3090 Ti. One LocalAI GGUF (`local/code`) |
| `gpu5080` | gpu5080 | `192.168.1.252` | RTX 5080. Exclusive-seq CUDA/index/GPU CI. Comfy **masked** |
| `homelab` | homelab | `192.168.1.170` | none. UI + Forgejo CPU + o11y + send-only MTA |

`host=prime` is the label. Router JSON still says `akula-prime`. Do not emit
both. Future `gpu5080-1080ti` is **not** a host value until `nvidia-smi` lists
the card (`G-1080` blocked).

### `ns`

| Value | Owns |
|---|---|
| `csd` | CogSynDelta harness: autodev, lab console, OpenAI gateway, CSD vault, `akula-csd-kb` index |
| `webui` | Open WebUI `ai.vectorweight.com` (homelab only). Not a second WebUI on `code.vectorweight.com` |
| `media` | `akula-comfyui.service` on gpu5080. **Masked.** Label if a leftover log line appears; do not start it |
| `mail` | Homelab Postfix send-only |
| `git` | Forgejo `git.vectorweight.com` |
| `o11y` | Grafana, Loki, VictoriaMetrics, Jaeger, OpenSearch (internal), status aggregator |
| `runner` | Forgejo self-hosted runners |

### `kind`

| Value | Means |
|---|---|
| `api` | HTTP JSON API (lab `/api/apply`, LocalAI `/v1`, Qdrant, keyword-cpu, Forgejo, status) |
| `gpu` | Process that occupies a GPU (LocalAI GGUF, Comfy, CUDA job) |
| `ui` | Browser frontend (lab console, Open WebUI, Grafana, Jaeger UI) |
| `mta` | Send-only mail |
| `runner` | Forgejo Actions worker |
| `index` | RAG/Qdrant index job (`csd-kb-index`, 5080 exclusive-seq) |
| `gateway` | Reverse proxy / OpenAI-compat forwarder / Caddy |

Pick **one** primary `kind`. LocalAI is `gpu` (the card owner); its HTTP is
still that same series. Lab console on homelab is `ui`; the prime process
that serves `/api/apply` is `api`.

### `service` catalog

`service` = the unit/container/runner name as systemd, compose, or Forgejo
sees it. Keep the `.service` suffix for systemd. Do not invent names that
are not in this table or in a unit file under `deploy/`.

**This repo (`deploy/`):**

| `service` | Host | Notes |
|---|---|---|
| `csd-autodev-loop.service` | prime | Worker. CSD vault only |
| `csd-lab-console.service` | prime | `:9118` GPU APIs + apply |
| `csd-lab-console.homelab.service` | homelab | UI; proxies to prime `:9118` |
| `csd-openai-gateway.service` | prime | `:9120` router |
| `csd-openai-gateway.homelab.service` | homelab | `:9120` forward-only to prime LocalAI |
| `csd-lab.target` | prime | Wants console + gateway + autodev. Not a scrape target |
| `postfix` | homelab | Send-only. Bind `127.0.0.1` + `172.30.0.1:25` |

**Referenced, not owned by this repo** (do not rename here):

| `service` | Host | Kind of unit | Evidence |
|---|---|---|---|
| `akula-localai` | prime | docker | [CODEX-OPS.md](../CODEX-OPS.md) |
| `akula-comfyui.service` | gpu5080 | systemd, **masked** | CODEX-OPS |
| `forgejo-runner-cpu` | homelab | systemd | CODEX-OPS. Forgejo name `homelab-cpu` |
| `homelab-cpu` | homelab | Forgejo runner | WHO-RUNS-WHAT. Use this for CI job series |
| `gpu5080-tzervas` | gpu5080 | Forgejo runner | WHO-RUNS-WHAT / `G-GPUCI` |
| `akula-prime-cpu` | prime | Forgejo runner (optional) | CODEX-OPS. Must stay `compute-cpu` |
| `unifi-grafana` | homelab | quadlet | [O11Y.md](O11Y.md), `deploy/grafana/` |
| `unifi-loki` | homelab | quadlet | O11Y.md |
| `unifi-opensearch` | homelab | quadlet, internal | O11Y.md. Caddy commented |
| `gpu-timeshare-5080.timer` | gpu5080 | systemd timer | PHASE-0-KEEP-DROP |
| `csd-kb-index` | gpu5080 | oneshot script `./scripts/csd-kb-index` | CODEX-OPS. Not a long-run unit |
| `akula-jaeger` | homelab | quadlet | live `/etc/containers/systemd/akula-jaeger.container` |
| `unifi-victoria` | homelab | quadlet | live `unifi-victoria.service` (not `unifi-vm`) |
| `unifi-promtail` | homelab | quadlet | live; journald overlay `deploy/o11y/` |
| `akula-otel` | homelab | quadlet | live collector `172.30.0.14` |
| `akula-open-webui` | homelab | quadlet | `ai.vectorweight.com` |
| `akula-status-aggregator` | homelab | systemd | `:9109` |
| `akula-health-exporter` | prime, gpu5080 | systemd/python `:9108` | VM job `akula-health`; series `akula_gpu_*` |
| `prometheus-node-exporter` | prime, gpu5080 | systemd | VM job `akula-node`; NVIDIA textfile `akula-nvidia-textfile.service` |
| `akula-nvidia-textfile.service` | prime, gpu5080 | oneshot | node_exporter textfile; not a DCGM job |

Live names read off homelab 2026-08-31. Do **not** invent `unifi-jaeger` /
`unifi-vm`. Alloy is **not** installed. Do not attach taxonomy `path` to
`akula-node` (collides with `node_filesystem` mount `path`).

### `path` (derived)

Default:

```
path = {env}.{ns}.{kind}
```

When `ns=runner`, middle segment is `host` so CPU and GPU CI do not collide:

```
path = {env}.{host}.{kind}
```

Examples (these are the only patterns):

| Series | `path` |
|---|---|
| Homelab OpenAI gateway | `lab.csd.gateway` |
| Prime lab apply API | `lab.csd.api` |
| Homelab lab UI | `lab.csd.ui` |
| 3090 LocalAI | `lab.csd.gpu` (one series; chat and autodev share the GGUF) |
| 5080 CUDA / index | `lab.csd.index` (index job) / `lab.csd.gpu` (card occupant) |
| Homelab CPU Actions | `ci.homelab.runner` |
| 5080 GPU Actions | `ci.gpu5080.runner` |
| Grafana | `prod-intent.o11y.ui` |
| Loki | `prod-intent.o11y.api` |
| VictoriaMetrics | `prod-intent.o11y.api` |
| Jaeger UI | `prod-intent.o11y.ui` |
| Postfix | `prod-intent.mail.mta` |
| Forgejo | `prod-intent.git.api` |
| Open WebUI | `prod-intent.webui.ui` |
| Caddy `code.vectorweight.com` | `prod-intent.csd.gateway` |
| Masked Comfy | `lab.media.gpu` (must stay silent) |

Do not encode user, PR, branch, or SHA in `path`.

---

## Group map

A `group` is the set you page/filter together. Members keep their own
`service` / `kind` / `path`.

### `csd-autodev`

Self-hosted implement loop. Hosted Grok is **not** a member.

| service | env | host | ns | kind | path |
|---|---|---|---|---|---|
| `csd-autodev-loop.service` | lab | prime | csd | api | lab.csd.api |
| `csd-lab-console.service` | lab | prime | csd | api | lab.csd.api |
| `csd-lab-console.homelab.service` | lab | homelab | csd | ui | lab.csd.ui |
| `csd-openai-gateway.service` | lab | prime | csd | gateway | lab.csd.gateway |
| `csd-openai-gateway.homelab.service` | lab | homelab | csd | gateway | lab.csd.gateway |
| `akula-localai` | lab | prime | csd | gpu | lab.csd.gpu |
| `csd-kb-index` (oneshot on 5080) | lab | gpu5080 | csd | index | lab.csd.index |

Depends on GPU lock (layer 5) and CSD vault (layer 4). Steer:
`/akula-data/cabal/csd-steer.json`. Apply token: prime `POST /api/apply`.

### `akula-chat`

Human chat. WebUI on **homelab only**. GPU decode on prime.

| service | env | host | ns | kind | path |
|---|---|---|---|---|---|
| Open WebUI (`ai.vectorweight.com`) | prod-intent | homelab | webui | ui | prod-intent.webui.ui |
| `csd-openai-gateway.homelab.service` | lab | homelab | csd | gateway | lab.csd.gateway |
| `akula-localai` | lab | prime | csd | gpu | lab.csd.gpu |

Do not run a second WebUI on `code.vectorweight.com`. `/chat` redirects.
Autodev has priority on `local/code` over WebUI chat
(`./scripts/csd-autodev-priority`).

### `akula-media`

| service | env | host | ns | kind | path |
|---|---|---|---|---|---|
| `akula-comfyui.service` | lab | gpu5080 | media | gpu | lab.media.gpu |

**Masked** (operator grant 2026-08-31). Unmask only with operator
`args.comfy=true` when `gpu5080.lock` is idle. Do not start from CSD.

### `forgejo-ci`

| service | env | host | ns | kind | path |
|---|---|---|---|---|---|
| `homelab-cpu` (`forgejo-runner-cpu`) | ci | homelab | runner | runner | ci.homelab.runner |
| `gpu5080-tzervas` | ci | gpu5080 | runner | runner | ci.gpu5080.runner |
| `akula-prime-cpu` | ci | prime | runner | runner | ci.prime.runner |
| Forgejo (`git.vectorweight.com`) | prod-intent | homelab | git | api | prod-intent.git.api |

CPU runners stay `compute-cpu`, never `gpu`. GPU runner takes `gpu5080.lock`
vs timeshare. Identity for CSD pushes: Forgejo user `autodev` (CSD vault
`git/autodev`), not operator admin.

### `grafana-stack`

Existing UniFi security-net o11y. **Do not add Tempo.**

| service | env | host | ns | kind | path |
|---|---|---|---|---|---|
| `unifi-grafana` | prod-intent | homelab | o11y | ui | prod-intent.o11y.ui |
| `unifi-loki` | prod-intent | homelab | o11y | api | prod-intent.o11y.api |
| `unifi-victoria` (`vm.vectorweight.com`) | prod-intent | homelab | o11y | api | prod-intent.o11y.api |
| `akula-jaeger` (`172.30.0.15:16686`) | prod-intent | homelab | o11y | ui | prod-intent.o11y.ui |
| `unifi-opensearch` | prod-intent | homelab | o11y | api | prod-intent.o11y.api |
| `akula-status-aggregator` (`:9109`) | prod-intent | homelab | o11y | api | prod-intent.o11y.api |
| `unifi-promtail` | prod-intent | homelab | o11y | api | prod-intent.o11y.api |
| `akula-otel` | prod-intent | homelab | o11y | api | prod-intent.o11y.api |
| `postfix` | prod-intent | homelab | mail | mta | prod-intent.mail.mta |

Frontends: `https://grafana.vectorweight.com`, `https://loki.vectorweight.com`,
`https://status.vectorweight.com`. Grafana SMTP:
`GF_SMTP_HOST=172.30.0.1:25`, contact point `maintainers-email` →
`maintainers@vectorweight.com`. Autodev is **not** Grafana admin.

OTLP from CSD is **not** enabled yet (`intern_parallel` flag). When it is,
use these same keys as resource attributes.

---

## Isolation environments (long-term)

Five layers. Innermost first. **Do not invent a new VLAN this run.** Existing
nets only.

### 1. Process / user unit

One systemd unit, docker container, or quadlet per `service`. Do not share a
process or user session across `group`s.

- CSD units: `deploy/systemd/` (`csd-lab.target` wants console + gateway +
  autodev on prime).
- Homelab UI units bind `192.168.1.170`, not `0.0.0.0`.
- Prime apply API is HTTP to `192.168.1.98:9118`, not SSH into worktrees.
- Allowlisted worktrees only; never `kang-main-wip`.

Failure of one unit must not require opening another group's ports.

### 2. Host

| Host | May run | Must not run |
|---|---|---|
| `prime` | LocalAI (stay loaded), autodev worker, lab API, optional CPU runner | Open WebUI, Forgejo GPU jobs, `csd-kb-index` CUDA |
| `gpu5080` | Exclusive-seq CUDA, index, GPU CI, share-small embed when lock idle | Second 14B, `compute-cpu` workflows, Comfy while autodev owns the card |
| `homelab` | WebUIs, Caddy, Forgejo CPU runner, grafana-stack, postfix | GPU jobs, LocalAI GGUF |

Parallelism is **across hosts** (3090 text + 5080 CUDA), not two 14Bs.

### 3. LAN ns / quadlet (existing nets only)

No new VLAN, bridge, or security-net subnet in this change.

| Net | Use | Documented members |
|---|---|---|
| LAN `192.168.1.0/24` | Host front-doors | prime `.98`, gpu5080 `.252`, homelab `.170` |
| UniFi security-net `172.30.0.x` (existing quadlets; prefix not in this repo) | o11y + SMTP | gateway `.1` (postfix), Grafana `.13`, Jaeger `.15`, OpenSearch `.21` |

Caddy `code.vectorweight.com` → `192.168.1.98:9118`. Open WebUI stays
`ai.vectorweight.com`. Quadlets stay on security-net; CSD lab units stay on
LAN IPs. OTLP collector address already on homelab (`192.168.1.170`).

Do not port-forward `25` / `587` / `465` / `16686` / Qdrant `6333` to WAN.

### 4. Vault (CSD vs operator)

| Plane | Path | Who |
|---|---|---|
| CSD secrets | `/akula-data/cabal/csd-vault` (prime); `~/.config/csd/vault` (homelab replica, same age key) | autodev / lab units via `secret exec` |
| Operator secrets | `~/.secrets` | operator only |
| Obsidian operator | `tzervas-dev-kb` | **RO** |
| Obsidian model | `akula-model-kb` | **RO** |
| Process memory | `akula-gap-kb` `:8092` | RW, not chat RAG |
| CSD experiment | `akula-csd-kb` | RW + 5080 reindex |

Allowlist in CSD vault: `gpu/localai-api-key`, `csd/apply-token`,
`git/autodev`, `hf/autodev` (when minted). Never copy operator Forgejo admin,
Cloudflare, GPU console, or `gpu/huggingface-token`.

Qdrant: 1024-d Qwen3 collections only. Never mix 384-d. Keyword retrieve
`:8091` / `:8092` is CPU; never pause LocalAI for search.

### 5. GPU lock

| Card | Lock | Rule |
|---|---|---|
| 3090 Ti on prime | Stay loaded. Prime timeshare `/akula-data/cabal/gpu-timeshare.json` only if CSD must **own the whole card** (rare; restore LocalAI on EXIT) | Never pause for RAG or `csd-kb-index` |
| 5080 | `gpu5080.lock` + timeshare `AKULA_TIMESHARE=/home/tzervas/akula-harness/state/gpu-timeshare.json` (not the prime file) | One exclusive-seq job. `share-small` only if free ≥ 8192 MiB and helper ≤ ~6 GiB |
| Comfy | masked `akula-comfyui.service` | Do not unmask from CSD |

Prime timeshare state is **not** the 5080 worker queue. Enqueue on the 5080
file or the job never claims.

---

## Bind / SMTP / traces (do not “fix” by opening WAN)

| Thing | Bind | Not |
|---|---|---|
| Lab console / gateway on homelab | `192.168.1.170` | `0.0.0.0` |
| Lab console / gateway on prime | host LAN / documented port | WAN |
| Postfix | `127.0.0.1:25`, `172.30.0.1:25` | `0.0.0.0`, `192.168.1.170:25`, `:587`, `:465` |
| Grafana SMTP client | `172.30.0.13` → `172.30.0.1:25` | auth, WAN relay unless operator Workspace creds |
| Jaeger UI | LAN `172.30.0.15:16686` | Tempo, WAN scrape |
| Qdrant | prime loopback `:6333`; 5080 via SSH reverse tunnel | public `:6333` |

Sender `grafana@vectorweight.com`. Recipients `maintainers@` /
`alerts@vectorweight.com` only. MX/SPF stay Google Workspace.

---

## Cardinality budget

Worst case today: `3 env × 3 host × 7 ns × ~20 service × 7 kind × 5 group`
would explode if cross-producted. **Do not** attach unused keys at scrape
time as empty. Emit the seven keys with the **one** allowed tuple from the
group map.

Refuse a new `service` value until it has a row in the catalog and a unit
file or runner registration. Refuse a new `group` until two or more units
fail together.

Prom/Loki reserved: do not override `job` / `instance` meaning. Put CSD
taxonomy **next to** them, not instead of `__address__`.

---

## Not this run

- No Grafana dashboard / playlist / alert provision (`args.apply` later).
- No new security-net IPs, VLANs, or quadlets.
- No Tempo.
- No OTLP from CSD until `intern_parallel`.
- No Comfy unmask.
- No 1080 Ti host value.
- No Plane / OpenProject install (Forgejo Projects API is 404; milestones
  stay git-side).

When provisioning later, stamp these labels on: systemd `Environment=`,
Prom scrape relabel, Loki pipeline stages, Jaeger process tags. Same keys,
same values, same group map.
