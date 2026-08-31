# Goal loop (autoloop control)

Workers do **not** reconstruct the parent chat. They load context, RAG/Obsidian
refs, and a **goal list**, then iterate until goals are met, stalled, or blocked.

This is a **do-while on goals**, not a one-shot todo.

## Loop

```
load IMPLEMENTER-HANDOFF + GOALS.md + csd-kb Program/ notes
while unmet goals remain AND not forced-stop:
  pick the highest-priority goal that is not blocked
  take one sandbox-safe increment toward that goal
  if the goal is met: mark met, continue
  if the increment made no progress (same fingerprint twice): mark stalled,
      switch to the next non-blocked goal
  if every remaining goal is blocked:
      draft an unblock path
      if the path is sandbox-safe: ask hosted Grok (gatekeeper) then do it
      if unsafe but Grok can help (WAN, HF, 5080 vs Comfy): ask Grok
      if truly unsafe or needs the operator: pause and inform the operator
```

Run via `/csd-goal-loop`. Bounded: a few increments per run; the 2h scheduler
is the backup sequence. Resume a paused run after the gatekeeper answers.

## Sandbox (may proceed without the operator)

Lab only: Forgejo `tzervas/*`, localhost Qdrant/LocalAI, `akula-csd-kb` reads,
homelab CPU CI, sibling git worktrees, `uv`/`pytest`/`ruff` on the worktree.
Merge our PRs when required jobs **ran and succeeded** without waiting for this
chat. Do not schedule cargo jobs on Python-only repos (no `if: rust` skip
theatre). If `Cargo.toml` appears later, fail closed and add a real rust job.

**Not sandbox** (gatekeeper or operator): WAN/GitHub/PyPI/HF downloads, pause
3090 LocalAI, preempt Comfy/video on the 5080, git history rewrite, merge of
red checks, writes to `tzervas-dev-kb` / `akula-model-kb`, Jules/`develop`,
**registering a new Forgejo runner**. `homelab-cpu` id 5 already has the six
labels and runs `tzervas/*`. Asking to add `tzervas-homelab-cpu` is an
operator pause, not a sandbox unblock.

## Context pack (always)

| Kind | Where |
|---|---|
| Cold start | `docs/program/IMPLEMENTER-HANDOFF.md` |
| Goal ledger | `docs/program/GOALS.md` |
| Phase 1 board | `docs/program/PHASE-1-TASK-BOARD.md` |
| Horizon | `docs/program/SELF-HOSTED-DRIVE-TARGETS.md` (blocked until P1-16) |
| Live PRs | `docs/program/HANDOFF-NEXT.md` |
| Measured claims | `STATUS.md` |
| Obsidian | `/akula-data/obsidian/akula-csd-kb/Program/` |
| RAG | collection `akula-csd-kb` 1024-d (may be 0 points until 5080 index) |

Point the worker at **specific files**, not “search the internet.”

## Stall vs block

| State | Meaning | Next |
|---|---|---|
| Met | DoD evidenced in git/CI | Next goal |
| Open | Increment possible in sandbox | Drive it |
| Stalled | Two rounds, same fingerprint | Switch goal |
| Blocked | Depends on WAN, GPU timeshare, operator, or unpatched CVE | Unblock path |

Do not spin a stalled goal. Do not fake green to “complete” a goal.
