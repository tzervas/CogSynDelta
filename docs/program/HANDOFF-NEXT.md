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

- P1-00 https://git.vectorweight.com/tzervas/memory-gate/pulls/1 — `ci/forgejo-cpu-fail-closed` @ `1c22643`. `578c66b` ran honest red. Local: unit 124 + integration 5 + regression 5; gitleaks clean with `.gitleaks.toml`; setuptools 84.0.0 offline. Safety still red on onnxruntime 1.22.0 (no 1.24 wheel in cache). Merge only if **legitimately green**.
- P1-01 https://git.vectorweight.com/tzervas/memory-gate/pulls/2 — `docs/memory-lifecycle-contract` @ `7b5c877`. Docs only.
- P1-03 chroma mitigation https://git.vectorweight.com/tzervas/memory-gate/pulls/3 — `fix/chroma-cve-client-only` @ `1b07c56`. Client-only refuse. **Not** an upstream wheel patch.
- P1-02 https://git.vectorweight.com/tzervas/memory-gate/pulls/4 — `feat/public-error-contract` @ `7a30b7b`. Mapper on store/gateway; CI restacked from P1-00. **Not merged**. Local main `691bb85` untouched.

Worktrees: `...-wt-p1-00`, `...-wt-p1-01`, `...-wt-chroma-cve`, `...-wt-p1-02`. **Never** reset `.../python-ai/memory-gate` (`local/kang-main-wip`).

### Merge grant (operator 2026-08-31)

Grok and self-hosted `local/code` **may merge their own** Forgejo PRs when required
checks are **legitimately green** (jobs ran and succeeded) and the row bar is met.
Skip / `|| true` / `continue-on-error` / fallback `echo` / missing runner is **not**
green. Honest red is not mergeable. Never merge Jules/`develop`/GitHub bots.
Never merge GitHub.com without the operator.

Do not merge until Forgejo required checks **ran and succeeded** on the new heads.

### Next closeable on this side

1. G-CI `1c22643`: only Safety/onnxruntime 1.22.0 is red — WAN lock bump ≥1.24.1 (not sandbox). Merge PR #1 only if **legitimately green**.
1b. G-ERR PR #4 `7a30b7b`: wait Forgejo after CI restack; same onnxruntime residual.
2. Autoloop: `/csd-drive-bootstrap` then `/csd-python-first-drive` `id=next`.
2. **Chroma:** do **not** pin `1.5.10.dev266` / `latest`. SQLite+vec + Qdrant until a named patched RC/stable that GHSA lists.
3. 5080 index: still **1024-d / 0 points**. Do not kill Comfy. Do not pause 3090 LocalAI.
4. HF: private `tzervas/cogsyndelta-eval` only at P1-15.

Push/merge on Forgejo as `tzervas` via `secret exec TOKEN=git/cabal-forgejo-admin` +
`git-askpass-token` (`credential.helper` disabled). Never GitHub bot push.
