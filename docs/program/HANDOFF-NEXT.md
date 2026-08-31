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

- P1-00 https://git.vectorweight.com/tzervas/memory-gate/pulls/1 — `ci/forgejo-cpu-fail-closed`. Local pytest **134 passed / 1 gpu deselected** after await-learn + ruff 0.16.5 + cryptography 50.0.1 + current-tree gitleaks + chroma `.trivyignore`. Push and wait Forgejo; merge only if **legitimately green**.
- P1-01 https://git.vectorweight.com/tzervas/memory-gate/pulls/2 — `docs/memory-lifecycle-contract` @ `7b5c877`. Docs only.
- P1-03 chroma mitigation https://git.vectorweight.com/tzervas/memory-gate/pulls/3 — `fix/chroma-cve-client-only` @ `1b07c56`. Client-only refuse. **Not** an upstream wheel patch.
- P1-02 https://git.vectorweight.com/tzervas/memory-gate/pulls/4 — `feat/public-error-contract` @ `52e8aac`. Types + `raise_mapped` self-chain fix. **Not merged**: Forgejo 1 ok / 14 fail (not legitimately green). Uncommitted mapper wiring may be in-flight on the p1-02 worktree (`csd-goal-loop`). Local main `691bb85` untouched.

Worktrees: `...-wt-p1-00`, `...-wt-p1-01`, `...-wt-chroma-cve`, `...-wt-p1-02`. **Never** reset `.../python-ai/memory-gate` (`local/kang-main-wip`).

### Merge grant (operator 2026-08-31)

Grok and self-hosted `local/code` **may merge their own** Forgejo PRs when required
checks are **legitimately green** (jobs ran and succeeded) and the row bar is met.
Skip / `|| true` / `continue-on-error` / fallback `echo` / missing runner is **not**
green. Honest red is not mergeable. Never merge Jules/`develop`/GitHub bots.
Never merge GitHub.com without the operator.

Do not merge until Forgejo required checks **ran and succeeded** on the new heads.

### Next closeable on this side

1. Push P1-00 await-learn/CI pins; wait legitimately green Forgejo; then merge PR #1.
2. Autoloop: `/csd-drive-bootstrap` then `/csd-python-first-drive` `id=next`.
2. **Chroma:** do **not** pin `1.5.10.dev266` / `latest`. SQLite+vec + Qdrant until a named patched RC/stable that GHSA lists.
3. 5080 index: still **1024-d / 0 points**. Do not kill Comfy. Do not pause 3090 LocalAI.
4. HF: private `tzervas/cogsyndelta-eval` only at P1-15.

Push/merge on Forgejo as `tzervas` via `secret exec TOKEN=git/cabal-forgejo-admin` +
`git-askpass-token` (`credential.helper` disabled). Never GitHub bot push.
