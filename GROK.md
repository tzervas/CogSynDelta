# Grok harness — CogSynDelta

Working branch in **`tzervas/CogSynDelta`** (`feat/agent-harness`). Codex: `docs/CODEX.md` + `./scripts/codex-csd`. Persistent program: `docs/PROGRAM-GOALS.md`. `docs/HANDOFF-ARCHITECTURE.md` is unvetted ChatGPT context — vet against live **Python** source. Python-first (memory-gate, then CSD); Rust rewrite later for speed/safety.

## Where to sit

```
/home/kang/code/personal/tzervas/CogSynDelta
branch: feat/agent-harness   (from origin/main PoC HEAD)
Forgejo deep tree: https://git.vectorweight.com/tzervas/CogSynDelta
```

Launch:

```bash
./scripts/grok-csd
# or:  cd this worktree && grok
```

`scripts/grok-csd` refuses protected trunks (`main`, `staging`, `develop`, `dev`) and injects `LOCALAI_API_KEY` from the secret toolchain (never argv).

## Truth order (do not invert)

1. `STATUS.md` — measured PoC-1/2/3 claims
2. `pyproject.toml` `description` — MoE-adjacent one-mind; Python `>=3.12,<3.14`
3. `src/cogsyndelta/poc/` — live measured slice
4. `docs/adr/` — decisions
5. `memory/constitution.md` — governance
6. Vault hub `Projects/Repositories/GitHub/tzervas/CogSynDelta/CogSynDelta.md` (ingest may lag HEAD)
7. `README.md` — **source text, often broader than the PoC**. Do not treat “10–100×”, VL-JEPA, quantum, or agent fleets as measured unless STATUS/tests say so.

CogSynDelta is **MoE-adjacent, not multi-agent**. Regions are experts of one mind. Softmax top-k + Switch aux LB is the gate. mHC / JEPA / quantum are later or stubbed.

## Skills / workflows

| Invoke | What |
|---|---|
| `/csd-context` | Load STATUS + PoC layout + GPU/branch rules before edits |
| `/code-thropology` | Strata map: README vs STATUS vs poc vs archive vs ADRs |
| `/csd-code-thropology` | Workflow: bounded parallel read-only dig |
| `/csd-branch-pr-review` | Workflow: open PRs × main/develop — fits vs sediment |
| `/csd-poc-gpu-drive` | CUDA measure on 5080 (exclusive-seq; do not pause LocalAI) |
| `/csd-ci-unblock` | Merge-stack CI |
| `/csd-python-first-drive` | One closeable Python-first increment (`P1-00`…`P1-17`, `ops-index`, `ops-runner`) |
| `/csd-drive-bootstrap` | Mirror program docs into `akula-csd-kb`, enqueue 5080 index, name next ID |

WebUI/Comfy (`ai.vectorweight.com`, `media.vectorweight.com`): **other Grok session**, cwd akula-ai-platform. Brief: `docs/program/HANDOFF-NEXT.md`.

## Self-hosted models

Copy tables from `config/clients/grok.toml.example` into `~/.grok/config.toml` by hand. Default for this dig: **hosted Grok** for orchestration; **`local/code`** (Qwen2.5-Coder-14B on akula-prime `:8080`) on the 3090 for implementation slices. One GGUF. Do not dual-load with `local/code-heavy`. Do not CUDA-index RAG while LocalAI holds the 3090. Private HF datasets (`tzervas/cogsyndelta-eval`) only if/when a board row needs a corpus.

## Hard rules

- Isolated worktree. Never commit or push `main` / `staging` / `develop` / `dev`.
- Conventional commits. Google-style docstrings. `uv`, not pip.
- GPU: 3090 = one LocalAI GGUF. 5080 = exclusive PoC CUDA. Keyword-cpu retrieve.
- Vault writes: `/update-kb` only after durable findings. Never mix operator/model KBs.
- Forgejo: Grok and self-hosted `local/code` **may merge their own**
  `tzervas/CogSynDelta` / `tzervas/memory-gate` PRs when required checks are
  **legitimately green** (ran and succeeded; skip/`|| true`/missing runner is
  not green) and the board-row bar is met. Never merge Jules/`develop`/GitHub
  bots. Never merge GitHub.com without the operator.

Related: memory-gate predecessor (Python), memory-gate-rs successor. Overlap is research (VSA/compression), not a shared impl.

Branch/PR fit: `docs/GROK-STRATA.md`. PoC lives on **main**. `develop` + Jules quality PRs are a second program — do not merge them to “sync.” Open #59 (`ci/akula-gpu-runner` → main) is the one PR that fits, after a rebase onto #66.
