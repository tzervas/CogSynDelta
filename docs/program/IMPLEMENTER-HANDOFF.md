# Implementer handoff package (self-hosted + autoloops)

Cold-start package for `local/code`, Grok child agents, and `/csd-python-first-drive`.
Do not reconstruct the parent chat. Read these files with tools.

## Read in this order

1. This file
2. `docs/program/GOAL-LOOP.md` (do-while on **goals**, stall/switch/unblock)
3. `docs/program/GOALS.md` (ledger)
4. `docs/program/PYTHON-FIRST-PLAN.md`
5. `docs/program/PHASE-1-TASK-BOARD.md` (current board)
6. `docs/program/SELF-HOSTED-DRIVE-TARGETS.md` (horizon — **do not start training**)
7. `docs/program/HANDOFF-NEXT.md` (live PR SHAs)
6. `STATUS.md` for any performance claim
7. Vault (read-only unless you are the gatekeeper indexer):
   `/akula-data/obsidian/akula-csd-kb/Program/`

## Who does what

| Actor | Job |
|---|---|
| Hosted Grok (this control plane) | Plans, ADRs, WAN, research feed. Merge is **not** gated on this chat |
| Self-hosted `local/code` (3090, 32k Q4 + share-small leftover) | One failing test + one function. No WAN |
| Autoloop `/csd-goal-loop` | Drive GOALS.md; **Merge phase** waits honest CI then merges |
| Autoloop `/csd-python-first-drive` | One board ID; open PR; Merge phase waits honest CI then merges |
| Second Grok (akula-ai-platform) | WebUI/Comfy. Do not touch from CSD |

Workflow host **cannot** spawn model slug `local/code` (only `grok-4.5` /
`grok-4.6`). Implement steps use `grok-4.6` until that changes. Keep LocalAI
loaded anyway.

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
| P1-03 durable | PR #7 | Head `7bac0e3`. Merge iff legitimately green. Then P1-06 SQLite. |

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
- Git: Forgejo `tzervas/*` as `GIT_USERNAME=tzervas` + `git-askpass-token` +
  `credential.helper=` disabled. Never GitHub bot push.
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
secret exec TOKEN=git/cabal-forgejo-admin -- \
  env GIT_USERNAME=tzervas GIT_ASKPASS=/home/kang/code/personal/tzervas/akula-ai-platform/scripts/git-askpass-token \
  GIT_TERMINAL_PROMPT=0 \
  git -c credential.helper= -c credential.https://git.vectorweight.com.helper= \
  push forgejo HEAD:<branch>
```

Index (5080 timeshare, never 3090):

```bash
./scripts/csd-kb-index
```

## Return shape

Changed paths, command, result, PR URL, residual risk. No whole-repo dumps.
If blocked (WAN needed, not closeable), say so and stop. Comfy is masked;
do not unmask. Do not start G-TRAIN.
