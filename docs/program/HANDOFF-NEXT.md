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
- P1-03 chroma mitigation https://git.vectorweight.com/tzervas/memory-gate/pulls/3 — `fix/chroma-cve-client-only` @ `5a9a13f`. Client-only refuse (`chroma_server_*` / `trust_remote_code` / `HttpClient`). **Not** an upstream wheel patch.

Worktrees: `.../python-ai/memory-gate-wt-p1-00`, `...-wt-p1-01`, `...-wt-chroma-cve`. **Never** reset `.../python-ai/memory-gate` (`local/kang-main-wip`).

### Next closeable on this side

1. **Ruff as a tool, not a hope.** CSD already uses `uvx ruff@PIN`. Memory-gate `lint.yml` / `fleet-ci.yml` still `uv run ruff` after `uv sync`. Pin `RUFF_VERSION` and `uvx ruff@…` so a missing project extra cannot hide the binary. Keep fail-closed (no `|| true`).
2. Read job logs for P1-00 12s code-quality fail vs 3m12s fleet-ci python fail; fix install vs findings separately.
3. Do not start P1-02 product errors until P1-00 has an honest Forgejo result you can explain (red is OK).
4. **Chroma:** do **not** pin GitHub nightly `1.5.10.dev266` / tag `latest`. No named RC exists (checked 2026-08-30). Keep SQLite+vec + Qdrant; reintegrate only on a named patched RC/stable that GHSA lists. Criteria: `PHASE-1-TASK-BOARD.md` and memory-gate `SECURITY.md`.
5. 5080 index: launcher enqueues; collection still **1024-d / 0 points**. Do not kill Comfy. Do not pause 3090 LocalAI.
6. HF: private `tzervas/cogsyndelta-eval` only at P1-15.

Push CSD/memory-gate with `secret exec TOKEN=git/cabal-forgejo-admin` + `akula-ai-platform/scripts/git-askpass-token`, `GIT_USERNAME=tzervas`. Never GitHub bot push.
