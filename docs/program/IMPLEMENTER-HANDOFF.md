# Implementer handoff package (self-hosted + autoloops)

Cold-start package for `local/code` and `csd-autodev-loop`. Hosted Grok
does not implement. Do not reconstruct the parent chat. Read these files.

## Read in this order

1. This file
2. `docs/program/GOAL-LOOP.md` (do-while on **goals**, stall/switch/unblock)
3. `docs/program/GOALS.md` (ledger)
4. `docs/program/PYTHON-FIRST-PLAN.md`
5. `docs/program/PHASE-1-TASK-BOARD.md` (current board)
6. `docs/program/SELF-HOSTED-DRIVE-TARGETS.md` (horizon — **do not start training**)
7. `docs/program/HANDOFF-NEXT.md` (live PR SHAs)
8. `STATUS.md` for any performance claim
9. `docs/program/GPU-POOL.md` (placement, caps, ~40 GiB pool — do not enable RPC)
10. `docs/program/GPU-SHARE.md` (horizon only — do not start `G-SHARE` / `G-1080`)
11. Vault (read-only unless you are the gatekeeper indexer):
   `/akula-data/obsidian/akula-csd-kb/Program/`

## Who does what

| Actor | Job |
|---|---|
| Hosted Grok (this control plane) | Planner / safety / unblock / context packs only. Merge is **not** gated on this chat |
| Self-hosted `local/code` (3090, 32k Q4 + share-small leftover) | Implement: one failing test + one function. No WAN |
| `csd-autodev-loop` + `csd-model-router` | Default implementer + GPU pack/migrate. No hosted tokens |
| Autoloop `/csd-python-first-drive` | **Do not use for implement** (burns hosted quota) |
| Second Grok (akula-ai-platform) | WebUI/Comfy. Do not touch from CSD |

Workflow host **cannot** spawn model slug `local/code` (only `grok-4.5` /
`grok-4.6`). Implement via `csd-autodev-loop` + `csd-localai-queue`. Keep
LocalAI loaded. Visibility: `docs/program/WHO-RUNS-WHAT.md` and
`./scripts/csd-lab-console` (tabs: Live feed, Pool, Steer). Chat UI:
https://code.vectorweight.com (Open WebUI → loaded 3090 alias) and `/lab`.

## Current closeable work (2026-08-31)

Phase 1 only. Horizon training is **blocked** until `P1-16` / `G-LIFE`.

| ID | Repo / worktree | Notes |
|---|---|---|
| P1-00 | PR #1 | **Merged** `7c1cdeba` |
| P1-01 | PR #2 | **Merged** `91cc0f79` |
| P1-02 | PR #4 | **Merged** `81af7f98` |
| G-FLEET | PR #5 | **Merged** `7595bd53` |
| P1-04 | PR #6 | **Merged** `58934e6` |
| P1-03 chroma | PR #3 | Red. Not a wheel patch. Do not merge. |
| P1-03 durable | PR #7 | **Merged** `6c0c6fb5` |
| P1-06 | PR #8 | **Merged** `a1f648d` (head `9c6c09d`) |
| P1-07 | PR #9 | **Merged** `a823268` (head `5546ad7`). Next is **P1-08**. |
| P1-08 | `memory-gate-wt-p1-08` | `feat/tiered-memory-policy`. One failing pytest + one function. |

**Never** reset `python-ai/memory-gate` (`local/kang-main-wip` `691bb85`).
Sibling worktrees from `forgejo/main`.

## Invariants

- One brain. Regions ≠ agents. Personas select basins.
- Python-first. `memory-gate-rs` is reference until program Phase 3 exits.
- Stores: SQLite+sqlite-vec local; Qdrant 1024-d Akula. No Chroma `1.5.10.dev*` /
  tag `latest`. Reintegrate only on a **named** patched RC/stable in GHSA.
- GPU: 3090 `local/code` Q4 native 32k + share-small leftover. Never dual 14B.
  5080 exclusive CUDA/index; Comfy masked for autodev. Never pause LocalAI for RAG.
  Never mix 384-d Qdrant.
- Git: autodev loop pushes as Forgejo identity **`autodev`** via
  `./scripts/csd-autodev-git` (`TOKEN=git/autodev` in the CSD vault).
  Operator admin token stays in `~/.secrets`. Never GitHub bot push.
- Branches: cut from `main`. After merge, delete the head. Keep only `main`,
  `release/*`, tags, and `local/kang-main-wip`. Drop Jules/Copilot heads.
  Merge API: `delete_branch_after_merge=true`.
- Merge: required checks **ran and succeeded**. Skip / `|| true` / missing
  runner is not green. Workflows merge themselves; do not wait on chat.
- WAN: workers deny. Gatekeeper may fetch chroma GHSA, PyPI, HF.

## Commands

```bash
cd /home/kang/code/personal/tzervas/CogSynDelta
./scripts/grok-csd
# one increment:
# /csd-python-first-drive  args.id=next
```

Push:

```bash
./scripts/csd-autodev-git push forgejo HEAD:<branch>
```

Index (5080 timeshare, never 3090):

```bash
./scripts/csd-kb-index
```

## Return shape

Changed paths, command, result, PR URL, residual risk. No whole-repo dumps.
If blocked (WAN needed, not closeable), say so and stop. Comfy is masked;
do not unmask. Do not start G-TRAIN.
