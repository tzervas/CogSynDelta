---
name: code-thropology
description: Map CogSynDelta sedimentary layers (README vs STATUS vs poc vs archive vs ADRs) with provenance. Use for a code-thropology dig, /code-thropology, or when the user asks what is live vs fossil vs marketing.
---

# Code-thropology (CogSynDelta)

Produce a **strata map**, not a rewrite. Read-only unless the user asks to land a harness/doc fix.

## Layers (top = current)

| Stratum | Where | How to treat |
|---|---|---|
| Measured PoC | `STATUS.md`, `tests/test_poc_*.py`, `src/cogsyndelta/poc/` | Source of truth |
| Package contract | `pyproject.toml` | Python floor, deps, honest description |
| Decisions | `docs/adr/` | Binding if still referenced by PoC |
| Governance | `memory/constitution.md`, `AGENTS.md` | Process |
| Vault | hub + `CogSynDelta-codebase` / `-api` | May lag HEAD (`source_commit`) |
| Marketing/README | `README.md` badges and feature list | Claims until evidenced |
| Archive | `docs/archive/`, `docs/2026-01-archive.tar.gz`, Jules/copilot branches | Fossil; cite path, do not revive blindly |
| Adjacent research | memory-gate, memory-gate-rs, ternary-inference-rs | Overlap in VSA/compression ideas |
| Branch/PR map | `docs/GROK-STRATA.md`, vault `CogSynDelta-branches` | Two programs: PoC-on-main vs develop sediment. Jules PRs do not fit. |

## Procedure

1. `git log --oneline -20` and current branch. If on a protected trunk, stop authoring.
2. Diff README “Overview” bullets against STATUS in-scope/out-of-scope tables.
3. List `src/cogsyndelta/` top-level packages; mark each **live / sediment / stub** with a file you actually opened.
4. Skim ADR titles (`docs/adr/README.md` or the files). Note ADR-0010 ternary, ADR-0012 VSA, ADR-0013 V-JEPA — later than PoC-3 unless tests exist.
5. Quote measured numbers only from STATUS or a command you ran.
6. Read `docs/GROK-STRATA.md`. `gh pr list` if it may be stale. Classify each open PR **fits / does not fit** Program A (PoC-on-main).
7. Write a short strata table + branch/PR fit table + “next honest increment” (one slice, not a fleet).

## Do not

- Restate README as architecture.
- Claim 10× compression (STATUS reports ~1.3–1.6× bytes).
- Treat quantum, VL-JEPA, AgentFleet, or mHC as implemented.
- CUDA-index the vault; keyword-cpu retrieve.
