# Who runs what — hosted Grok vs self-hosted LocalAI

Living visibility for the operator. Not `STATUS.md`.

## Intent

Drive CogSynDelta / memory-gate **maximally on self-hosted models**. Hosted
Grok is the control plane: plans, WAN, merge policy, research feed,
context packs. It does **not** implement.

## What actually runs today (2026-08-31)

| Path | Where | Model | Why |
|---|---|---|---|
| This Grok TUI | akula-prime session | Hosted grok-4.6 | Control plane. You are here. |
| `/csd-autodev-loop` | akula-prime | **local/code via queue** | Default implementer. Router packs helpers. No hosted Grok tokens |
| `/csd-python-first-drive` | Grok Build | **do not use for implement** | Burns hosted quota; host cannot spawn `local/code` |
| 3090 LocalAI `local/code` | akula-prime `:8080` | Qwen2.5-Coder-14B Q4 **32k** | Resident assist (~16.5 GiB). Chat via Open WebUI / lab **Live feed** |
| 5080 | gpu5080 | helpers / CUDA (Comfy masked) | Exclusive CUDA/index/GPU CI. Share-small embed when lock idle. Not a second 14B |
| `csd-model-router` | both | placement + migrate | Caps: Ampere vs Blackwell. Stage A live; Stage B pool parked |
| Homelab `homelab-cpu` | 192.168.1.170 | none | Forgejo CPU Actions |
| `gpu5080-tzervas` | gpu5080 | none | Forgejo GPU Actions, queued on `gpu5080.lock` |

**WebUIs live on homelab only.** GPU backends are akula-prime (3090 LocalAI)
and gpu5080 (CUDA / helpers / GPU CI). Prime must not run Open WebUI.

Workflows still **orchestrate** as grok-4.5/4.6 (host slugs), but **code
generation goes through** `./scripts/csd-localai-queue` → 3090 `local/code`
(autodev priority over WebUI chat). 5080 is tests/helpers behind `gpu5080.lock`.

## Operator UI

- Chat: https://ai.vectorweight.com (existing Open WebUI → 3090 LocalAI
  `:8080` / `:8079`; Comfy/media via that instance). Do **not** run a
  second WebUI on `code.vectorweight.com`.
- Lab: https://code.vectorweight.com — **Live feed** (3090+5080 loaded
  models), **Pool**, **History**, **Steer**. `/chat` redirects to AI WebUI.
- Autodev priority: `./scripts/csd-autodev-priority on` (Comfy lock-wrap; prime WebUI off)

## How to invert it (self-hosted lift)

1. Implement agents call `csd-localai-queue complete --kind autodev`.
2. 5080: GPU pytest + helper models, never a second 14B.
3. Hosted Grok: pick, verify, WAN, merge-when-green.

## Monitor

| Signal | Command |
|---|---|
| 3090 VRAM + LocalAI | `csd-lab-console` / `nvidia-smi` on prime |
| 5080 VRAM + lock + runner | same (SSH `gpu5080`) |
| Forgejo CI | `scripts/forgejo-pr-watch.py` |
| Grok workflows | `/workflow runs` in this TUI |

Steer file: `/akula-data/cabal/csd-steer.json` (`pause`, `note`, `next_goal`).

## Long-term: autodev escalates to Grok (minimal perms)

Autodev (local/code + 5080 helpers) **decides** when hosted Grok is worth a
token. Operator is not in the loop for every stall.

| May escalate (`csd-escalate --reason`) | Must not escalate |
|---|---|
| `wan` — GHSA/PyPI/HF named-version check | Every implement tick |
| `stall` — same fingerprint twice | G-TRAIN / weight training |
| `adr` — one architecture decision | Dual 14B, pause LocalAI for RAG |
| `merge-policy` — red vs skip-theatre | Jules/`develop`/GitHub bot merge |
| `safety` — quota/OOM/collision | `--always-approve` grok CLI |

Default **dry-run** (log `/akula-data/cabal/escalate-log.jsonl`). Fire only if
`CSD_ESCALATE=1`. CLI: `grok --print --prompt-file` with `--allow Read` and
`--deny` WAN curl/wget. Other providers later, same allowlist.

**UX:** homelab https://code.vectorweight.com is the chat entry;
https://code.vectorweight.com/lab is the harness (GPUs, queue, steer). GPU
backends stay on 3090/5080. That is the DI/DX target: self-hosted loop first,
tiny Grok assist when the loop itself asks.
