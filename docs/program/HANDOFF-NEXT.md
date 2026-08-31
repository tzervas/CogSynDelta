# Handoff — this CogSynDelta session vs WebUI session

**Split (operator 2026-08-30):**

| Session | Owns |
|---|---|
| **This repo / `feat/agent-harness`** | Python-first plan, memory-gate P1-00/P1-01, 3090 `local/code` assist, HF datasets *if/when* |
| **Second Grok, cwd akula-ai-platform** | `https://ai.vectorweight.com` catalog/routing and `https://media.vectorweight.com` Comfy |

WebUI brief: `akula-ai-platform/docs/operations/HANDOFF-WEBUI-2026-08-30.md`.

Do not implement LocalAI/Comfy catalog fixes in CogSynDelta.

## Resume here

```bash
cd /home/kang/code/personal/tzervas/CogSynDelta   # feat/agent-harness
./scripts/grok-csd
```

Read `docs/program/IMPLEMENTER-HANDOFF.md` then `PYTHON-FIRST-PLAN.md` and
`PHASE-1-TASK-BOARD.md`. Horizon only: `SELF-HOSTED-DRIVE-TARGETS.md` (do not train).

### Open Forgejo PRs (memory-gate, not merged)

- P1-00 https://git.vectorweight.com/tzervas/memory-gate/pulls/1 — **merged** `7c1cdeba` (head `500039e`). Fail-closed CI on `main`. Local `python-ai/memory-gate` `691bb85` **untouched**.
- P1-01 https://git.vectorweight.com/tzervas/memory-gate/pulls/2 — **merged** `91cc0f79` (head `8b47943`). Quality/security/fleet-ci/tests ran.
- P1-03 chroma mitigation https://git.vectorweight.com/tzervas/memory-gate/pulls/3 — `fix/chroma-cve-client-only` @ `1b07c56`. Client-only refuse. **Not** an upstream wheel patch. Red; not mergeable.
- P1-02 https://git.vectorweight.com/tzervas/memory-gate/pulls/4 — **merged** `81af7f98`.
- G-FLEET https://git.vectorweight.com/tzervas/memory-gate/pulls/5 — **merged** `7595bd53`. Python-only fleet-ci; no cargo jobs.
- P1-04 https://git.vectorweight.com/tzervas/memory-gate/pulls/6 — **merged** `58934e6` (head `38156b3`). 15 jobs ran.
- P1-03 https://git.vectorweight.com/tzervas/memory-gate/pulls/7 — **merged** `6c0c6fb5` (head `7bac0e3`).
- P1-06 https://git.vectorweight.com/tzervas/memory-gate/pulls/8 — **merged** `a1f648d` (head `9c6c09d`). 15 jobs ran and succeeded; head deleted.

Worktrees: `...-wt-p1-00`, `...-wt-p1-01`, `...-wt-chroma-cve`, `...-wt-p1-02`, `...-wt-fleet-py`, `...-wt-p1-04`, `...-wt-p1-06`. **Never** reset `.../python-ai/memory-gate` (`local/kang-main-wip`).

### Merge grant (operator 2026-08-31)

Grok and self-hosted `local/code` **may merge their own** Forgejo PRs when required
checks are **legitimately green** (jobs ran and succeeded) and the row bar is met.
Skip / `|| true` / `continue-on-error` / fallback `echo` / missing runner is **not**
green. Honest red is not mergeable. Never merge Jules/`develop`/GitHub bots.
Never merge GitHub.com without the operator.

Do not merge until Forgejo required checks **ran and succeeded** on the new heads.

### Next closeable on this side

1. Met: G-CI, G-SPEC, G-ERR, G-FLEET, G-STORE, G-ACK, G-SQL, G-RAG, G-GPU, G-CAP. P1-06 `a1f648d`.
2. **Met:** P1-07 / `G-QD` PR #9 merged `a823268` (head `5546ad7`, 15 jobs ran).
   **Next product:** P1-08 bounded tiered store (`feat/tiered-memory-policy`). Not G-LIFE yet.
3. Autoloop: `scripts/csd-autodev-loop --worker` (self-hosted). Do **not** `/csd-python-first-drive` for implement. Lab tabs: feed + pool. `G-POOL` Stage A is the router; Stage B RPC stays off. `G-SHARE` / `G-1080` are horizon only (`docs/program/GPU-SHARE.md`).
4. **Chroma:** do **not** pin `1.5.10.dev266` / `latest`. SQLite+vec + Qdrant until a named patched RC/stable that GHSA lists. PR #3 stays unmerged while red.
5. HF: private `tzervas/cogsyndelta-eval` only at P1-15.

Push on Forgejo as **`autodev`** via `./scripts/csd-autodev-git`.
Operator admin stays in `~/.secrets`. Never GitHub bot push.
