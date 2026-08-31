---
name: csd-context
description: Load CogSynDelta working context (STATUS, PoC slice, branch/GPU rules) before substantial work. Use when starting in CogSynDelta, the user runs /csd-context, or asks what is actually implemented vs README.
---

# CogSynDelta context

Checkout: isolated worktree. Protected trunks: `main`, `staging`, `develop`, `dev`.

1. Read `GROK.md` then `STATUS.md` (measured PoC-1/2/3). Do not start from README claims.
2. Live code is `src/cogsyndelta/poc/` (`vae`, `regions`, `router`, `train`, `train_route`, `compress`, `cli`). Older `core/` `agents/` `api/` `quantum/` are sediment unless STATUS or tests prove otherwise.
3. `pyproject.toml`: Python `>=3.12,<3.14` (README python-3.14 badge is stale). Description: MoE-adjacent one-mind.
4. Vault hub: `Projects/Repositories/GitHub/tzervas/CogSynDelta/CogSynDelta.md`. Note `source_commit` vs `git rev-parse HEAD`. Related: memory-gate / memory-gate-rs (research overlap, not shared impl).
4b. Branches/PRs: `docs/GROK-STRATA.md`. Program A = main PoC. Program B = develop + Jules PRs (do not merge to sync).
5. GPU: 3090 LocalAI one GGUF; 5080 exclusive PoC CUDA; never pause LocalAI for RAG; never 0.0.0.0 WAN.
6. Tests: `uv run pytest tests/test_poc_*.py`. Claims need STATUS rows or new measurements.
