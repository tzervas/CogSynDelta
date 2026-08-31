# Who runs what — hosted Grok vs self-hosted LocalAI

Living visibility for the operator. Not `STATUS.md`.

## Intent

Drive CogSynDelta / memory-gate **maximally on self-hosted models**. Hosted
Grok is the control plane: plans, WAN, merge policy, research feed.

## What actually runs today (2026-08-31)

| Path | Where | Model | Why |
|---|---|---|---|
| This Grok TUI | akula-prime session | Hosted grok-4.6 | Control plane. You are here. |
| `/csd-goal-loop` pick/verify | Grok Build workflow host | **grok-4.5** | Workflow runtime cannot spawn `local/code` |
| `/csd-python-first-drive` implement + merge | same | **grok-4.6** | Same host limit. `local/code` is loaded but not the workflow implementer |
| 3090 LocalAI `local/code` | akula-prime `:8080` | Qwen2.5-Coder-14B Q4 **32k** | Resident assist (~16.5 GiB). Chat via Open WebUI / lab console |
| 5080 | gpu5080 | **no LLM** (Comfy masked) | Exclusive CUDA/index/GPU CI. Not a second 14B |
| Homelab `homelab-cpu` | 192.168.1.170 | none | Forgejo CPU Actions |
| `gpu5080-tzervas` | gpu5080 | none | Forgejo GPU Actions, queued on `gpu5080.lock` |

**WebUIs live on homelab only.** GPU backends are akula-prime (3090 LocalAI)
and gpu5080 (CUDA / helpers / GPU CI). Prime must not run Open WebUI.

Workflows still **orchestrate** as grok-4.5/4.6 (host slugs), but **code
generation goes through** `./scripts/csd-localai-queue` → 3090 `local/code`
(autodev priority over WebUI chat). 5080 is tests/helpers behind `gpu5080.lock`.

## Operator UI

- Chat: https://code.vectorweight.com (homelab Open WebUI → 3090 `:8080`)
- Lab/steer: https://code.vectorweight.com/lab (`csd-lab-console`)
- Autodev priority: `./scripts/csd-autodev-priority on` (Comfy masked; prime WebUI off)

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
