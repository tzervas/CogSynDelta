# CSD Grok ping (operator-wakes mailbox)

Autodev is **local**: 3090 `local/code` + Forgejo. Hosted Grok is not in
the loop. Autodev writes a small mailbox when it is stuck. Grok reads
that file **only when the operator opens a chat**. Do not auto-call the
hosted API.

This is **not** a scheduler. Do **not** poll hosted Grok. Polling and
timers burn quota. There is no systemd unit, cron, `csd-escalate`
`CSD_ESCALATE=1` fire, or lab watch that invokes `grok` / xAI.

## Hard never

| Rule | Value |
|---|---|
| Identity | Forgejo **`autodev`** (`git/autodev`) |
| Implementer | 3090 LocalAI `local/code` + Forgejo workflows |
| Grok wake | Operator opens a session **after** `need=true` |
| Ping budget | **≤ ~2 KiB** total JSON |
| Transcripts | **Never** (no chat dumps, no autodev-out, no CI logs) |
| GitHub | **Never** (Forgejo `tzervas/*` only) |
| Dual 14B | **Never** |
| 3090 LocalAI | **Never pause** |
| Merge | Only when required Forgejo checks **ran and succeeded** |
| Hosted Grok API | **Never** from autodev, lab, timers, or hooks |

Skip / `|| true` / `continue-on-error` / fallback `echo` / missing
runner is **not** green. Do not wait on this chat to merge.

## Mailbox

File: `/akula-data/cabal/csd-need-grok.json` (`CSD_GROK_NEED`). Same cabal
dir as `csd-steer.json`. LAN lab only — never WAN `0.0.0.0`.

```json
{
  "need": true,
  "question": "one question the operator/Grok can answer",
  "evidence": "blocker + paths + what was tried (compact)",
  "goal": "P1-09"
}
```

| Field | Required | Notes |
|---|---|---|
| `need` | yes | `true` only under [When to ping](#when-to-ping). `false` / omit = no wake. |
| `question` | when `need` | One question. Not a transcript. |
| `evidence` | when `need` | Blocker, paths, what was tried. Compact. |
| `goal` | when `need` | Board id (`P1-09`) or short goal name. |

Optional keys **inside the same 2 KiB cap** (do not add a dump):

| Field | Notes |
|---|---|
| `blocker` | One-line stall reason |
| `paths` | Few repo-relative paths (not whole trees) |
| `tried` | Few command/CI names already attempted |
| `identity` | Always `autodev` if present |
| `t` | Unix seconds when the ping was set |

Truncate to 2 KiB. Drop `tried`/`paths` before `question`. Never attach
pytest output, journald, Open WebUI turns, or `autodev-out/*.json`.

Idle / clear:

```json
{
  "need": false
}
```

Operator or Grok clears `need` after the session answers. Autodev does
not flip `need` back to `true` for the same fingerprint until a new
stall or a new missing capability.

## Lab API

Prime lab `http://192.168.1.98:9118` (LAN bind, not WAN). Same agent as
steer.

| Method | Path | Body / result |
|---|---|---|
| `GET` | `/api/grok-need` | Current mailbox JSON |
| `POST` | `/api/grok-need` | Write/merge mailbox JSON |

```bash
# autodev (stuck): set the ping
curl -sS -X POST http://192.168.1.98:9118/api/grok-need \
  -H 'Content-Type: application/json' \
  -d '{"need":true,"question":"…","evidence":"…","goal":"P1-09"}'

# operator / Grok session: read then clear
curl -sS http://192.168.1.98:9118/api/grok-need
curl -sS -X POST http://192.168.1.98:9118/api/grok-need \
  -H 'Content-Type: application/json' \
  -d '{"need":false}'
```

`GET`/`POST` only rewrite the mailbox file. They **must not** spawn
`grok`, hit xAI, or enqueue a hosted job.

## When to ping

Autodev sets `need=true` **only** after local work is exhausted:

1. **Stuck after N failed CI** — same required-check fingerprint failed
   **N=2** times (matches [GOAL-LOOP.md](GOAL-LOOP.md) stall). Not one
   flake. Not skip-theatre.
2. **Missing capability** autodev cannot do locally:
   - **Web research** (named-version GHSA/PyPI/docs the sandbox denies)
   - **HF mint** (`hf/autodev` still empty; operator-only token create)
   - **Hardware** (GPU down, VFIO guest dark, lock stuck, runner missing)

Do **not** ping for: every implement tick, G-TRAIN / weight training,
dual 14B, pause LocalAI, Jules/`develop`/GitHub merge, `--always-approve`.

Keep implementing other non-blocked goals while `need=true`. Ping is
asynchronous. Do not idle the 3090 waiting for Grok.

## Who reads it

| Actor | Job |
|---|---|
| Autodev (`local/code`) | Implement. Write mailbox when stuck. Never call hosted Grok. |
| Lab `/api/grok-need` | Read/write the file. No side effects. |
| Operator | Opens a Grok session **after** seeing `need=true` (lab, ping file, or note). |
| Hosted Grok | Reads the mailbox at session start. Answers the question. Clears `need` when done. |

Grok does not poll. If `need` is false, the session is a normal operator
chat — do not invent a stall.

## GPUs while waiting

Autodev stays on **3090**. CUDA CI stays on **5080**. LocalAI stays
loaded.

**1080 Ti** (`hosts.gpu5080-1080ti`): use **only** when **both** are
true:

1. Router catalog `live=true`
2. Guest `nvidia-smi` lists the GTX 1080 Ti

Else **3090 + 5080 only**. Do not start `G-1080`. Do not power off
gpu5080. Do not treat Pascal as a 5080 substitute.

## Not this mechanism

| Thing | Why not |
|---|---|
| `csd-steer.json` / `POST /api/steer` | Operator next-goal / pause. Not a Grok wake. |
| `scripts/csd-escalate` + `CSD_ESCALATE=1` | Would call hosted `grok --print`. Forbidden as a ping path. Leave dry-run. |
| `/csd-python-first-drive` | Burns hosted quota. Not implement. |
| Workflow host `grok-4.5` / `grok-4.6` spawn | Cannot run `local/code`. Do not schedule it to “check the ping.” |
| GitHub.com bots / PRs | Out of program. |

## Example (budget)

```json
{
  "need": true,
  "identity": "autodev",
  "goal": "P1-09",
  "question": "Forgejo required pytest failed twice on domain isolation; is the oracle contract wrong or the gateway?",
  "evidence": "blocker: tests/test_gateway_retrieve.py::test_domain_isolation; paths: src/memory_gate/gateway.py tests/test_gateway_retrieve.py; tried: local pytest x2, forgejo wait x2, no WAN",
  "blocker": "required pytest red N=2",
  "paths": [
    "src/memory_gate/gateway.py",
    "tests/test_gateway_retrieve.py"
  ],
  "tried": ["pytest x2", "forgejo/wait x2"]
}
```

## Alert (Grafana → send-only postfix)

One page to `maintainers@vectorweight.com` when the mailbox is `need=true`.
Not a Grok scheduler. Not a 30s loop-tick mail.

| Piece | Role |
|---|---|
| Tiny exporter | Lab `GET /metrics` gauges `csd_need_grok` and `csd_need_grok_mtime_seconds` |
| Scrape | `deploy/o11y/vm-scrape.yml` job `csd-lab` → prime `:9118` (live) |
| Rule | `CSDNeedGrok` (`csd_need_grok == 1`, `for: 2m`, `noDataState: OK`) |
| Mail | Grafana contact `maintainers-email` → postfix send-only → `maintainers@` |

`csd_need_grok` is 1 only while JSON `need` is true. Mtime changes **only**
when the file is rewritten (fingerprint in the loop). Grafana
`repeat_interval: 4h` so a stuck ping is one mail, not one per scrape.
Clear `need=false` after the operator/Grok session.

Do **not**: poll hosted Grok, `CSD_ESCALATE=1`, cron/`csd-escalate`, or
email on every `csd-autodev-loop` tick.

## References

- Autodev vs Grok: [CSD-AUTODEV-HANDOFF.md](CSD-AUTODEV-HANDOFF.md)
- Visibility: [WHO-RUNS-WHAT.md](WHO-RUNS-WHAT.md)
- Loop / stall: [GOAL-LOOP.md](GOAL-LOOP.md)
- Secrets / HF mint: [CSD-SECRETS.md](CSD-SECRETS.md)
- 1080 Ti live gate: [CSD-1080TI-RAG.md](CSD-1080TI-RAG.md)
- Lab harness: [HARNESS.md](HARNESS.md)
