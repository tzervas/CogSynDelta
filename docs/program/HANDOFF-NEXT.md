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

Read `docs/program/PYTHON-FIRST-PLAN.md` then `PHASE-1-TASK-BOARD.md`.

### Open Forgejo PRs (memory-gate, not merged)

- P1-00 https://git.vectorweight.com/tzervas/memory-gate/pulls/1 — `ci/forgejo-cpu-fail-closed` @ `a66f730`. **Pytest ran** (unit 8 failed / 121 passed / 1 error in 132s; fleet-ci 8 failed / 126 passed). Ruff/cz green. Honest leftover reds: unawaited + agent-feedback (P1-03), gitleaks 14 historical hits, trivy chromadb 1.5.9 + cryptography. Drop `upload-artifact@v4` (Forgejo GHESNotSupported). P1-02 unblocked.
- P1-01 https://git.vectorweight.com/tzervas/memory-gate/pulls/2 — `docs/memory-lifecycle-contract` @ `7b5c877`. Docs only.
- P1-03 chroma mitigation https://git.vectorweight.com/tzervas/memory-gate/pulls/3 — `fix/chroma-cve-client-only` @ `1b07c56`. Client-only refuse. **Not** an upstream wheel patch.
- P1-02 https://git.vectorweight.com/tzervas/memory-gate/pulls/4 — `feat/public-error-contract` @ `90a37ce`. `memory_gate.errors` + nine named types + `map_backend_error`. **60** public-error tests passed. Mapper **not** wired into gateway/store raise sites yet (VectorStore* still leak there). Local main `691bb85` untouched.

Worktrees: `...-wt-p1-00`, `...-wt-p1-01`, `...-wt-chroma-cve`, `...-wt-p1-02`. **Never** reset `.../python-ai/memory-gate` (`local/kang-main-wip`).

### Merge grant (operator 2026-08-31)

Grok and self-hosted `local/code` **may merge their own** Forgejo PRs when required
checks are **legitimately green** (jobs ran and succeeded) and the row bar is met.
Skip / `|| true` / `continue-on-error` / fallback `echo` / missing runner is **not**
green. Honest red is not mergeable. Never merge Jules/`develop`/GitHub bots.
Never merge GitHub.com without the operator.

**None of PR #1–#4 are legitimately green today** (pytest unawaited, gitleaks history,
trivy chromadb 1.5.9). Do not merge them.

### Next closeable on this side

1. P1-03 durable `learn` / unawaited coroutines (pytest reds are the honest leftover).
2. **Chroma:** do **not** pin `1.5.10.dev266` / `latest`. SQLite+vec + Qdrant until a named patched RC/stable that GHSA lists.
3. 5080 index: still **1024-d / 0 points**. Do not kill Comfy. Do not pause 3090 LocalAI.
4. HF: private `tzervas/cogsyndelta-eval` only at P1-15.

Push/merge on Forgejo as `tzervas` via `secret exec TOKEN=git/cabal-forgejo-admin` +
`git-askpass-token` (`credential.helper` disabled). Never GitHub bot push.
