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

So: autodev **workflows are hosted Grok**, not the 3090 coder, until a local
harness calls LocalAI. That gap is what `scripts/csd-lab-console` and
`code.vectorweight.com` are for.

## How to invert it (self-hosted lift)

1. Chat / steer / watch: `./scripts/csd-lab-console` or https://code.vectorweight.com
   (Open WebUI → LocalAI `local/code`).
2. Local implement loop (Cabal `cabal-dev-loop` pattern): one file, pytest,
   Forgejo PR — **LocalAI writes, Grok reviews**. Not wired as the default
   `/csd-python-first-drive` child yet (workflow host slugs are grok-4.5/4.6
   only).
3. 5080: GPU tests + CUDA jobs, not text implement.

## Monitor

| Signal | Command |
|---|---|
| 3090 VRAM + LocalAI | `csd-lab-console` / `nvidia-smi` on prime |
| 5080 VRAM + lock + runner | same (SSH `gpu5080`) |
| Forgejo CI | `scripts/forgejo-pr-watch.py` |
| Grok workflows | `/workflow runs` in this TUI |

Steer file: `/akula-data/cabal/csd-steer.json` (`pause`, `note`, `next_goal`).
