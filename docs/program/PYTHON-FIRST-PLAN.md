# CogSynDelta Python-first development plan

**Status**: accepted 2026-08-30 (operator: proceed out of planning; implement).
**Source**: Codex Phase 0 artifacts + proposed plan. ChatGPT/Codex then stalled
on empty `akula-csd-kb` Qdrant and resident Comfy VRAM; this file is the
official execution contract.

Measured truth remains `STATUS.md`. `docs/HANDOFF-ARCHITECTURE.md` is unvetted
ChatGPT context and is not an implementation spec.

## Canonical artifacts

| Artifact | Path |
|---|---|
| Keep/drop audit | [PHASE-0-KEEP-DROP.md](PHASE-0-KEEP-DROP.md) |
| Python vs Rust gap map | [PYTHON-MEMORY-GATE-GAP-MAP.md](PYTHON-MEMORY-GATE-GAP-MAP.md) |
| Sequenced Phase 1 board | [PHASE-1-TASK-BOARD.md](PHASE-1-TASK-BOARD.md) |
| Program allocation | [../PROGRAM-GOALS.md](../PROGRAM-GOALS.md) |
| Lab ops | [../CODEX-OPS.md](../CODEX-OPS.md) |

Vault copies live under `akula-csd-kb` (RW experiment plane). Git copies in this
directory are what agents implement against.

## Invariants

- One brain with specialized centers. Personas select basins. Neither is an
  independent agent or MoE expert.
- Python-first. `memory-gate-rs` is read-only until program Phase 3 exits.
- One architecture change per Forgejo PR. Conventional Commits.
- Self-review COMMENT; merge only on honestly green Forgejo checks.
- Never bot-push GitHub.com. Never write `tzervas-dev-kb` or `akula-model-kb`.
- Never pause 3090 LocalAI for RAG search. Never mix 384-d into shared Qdrant.
- Shared Akula Qdrant is pinned Qwen3-Embedding-0.6B at **1024** dimensions.
- Pruning is not consolidation. A skip is not a green check.

## Goal 0 — trustworthy operations (in flight)

1. Repair the 5080 `akula-csd-kb` index path:
   - Remote worker must read the remote timeshare queue, not prime-only paths.
   - Serialize against resident ComfyUI; enqueue, do not preempt.
   - Pinned Qwen3-Embedding-0.6B CUDA runtime + indexer on gpu5080.
   - Read-only staged vault + authenticated Qdrant route.
   - Require dimension 1024 and a nonzero point count.
   - Never fall back to the 3090.
2. Restore fail-closed Forgejo CPU CI (`P1-00`):
   - Discover whether `homelab-cpu` or `akula-prime-cpu` is the live runner.
   - Register and verify it.
   - Every product workflow: `runs-on: [self-hosted, linux, x64, podman, compute-cpu, host-homelab]`.
   - No `ubuntu-latest`, `.forgejo/workflows`, GPU labels, `continue-on-error`,
     fallback `echo` / `|| true`, or GitHub bot push.
   - Unawaited-coroutine `RuntimeWarning` must fail tests.
3. Preserve Phase 0 verdicts: CSD `main`, `feat/agent-harness`, rebase
   `ci/akula-gpu-runner`. Do not merge `develop` / `dev` / Jules / Claude /
   Copilot / maintenance sediment. Preserve Python `local/kang-main-wip`.
   Rust `main` + `feat/hypha-kv-tiers` are references only.

## Goal 1 — Python `memory-gate` complete

Work tree: `/home/kang/code/personal/tzervas/python-ai/memory-gate`.
Always branch from current `forgejo/main`. Never reset local `main`
(`local/kang-main-wip`). Use sibling worktrees.

IDs and gates: [PHASE-1-TASK-BOARD.md](PHASE-1-TASK-BOARD.md). Exit when
`learn → retrieve → consolidate → persona instantiate` survives restart and
every REQUIRED gap-map row cites a passing test. Then **stop and re-plan
Phase 2**.

`P1-05` / `P1-06` / `P1-07` may run in parallel after `P1-04`. All other
dependencies are hard gates. Do not start product-code PRs until `P1-00`
has produced a real failing-when-broken Forgejo run. Docs/ADR (`P1-01`) may
land in parallel.

## Later gates (do not start)

- **Phase 2**: wire completed Python memory-gate into CogSynDelta `memory/`.
  Keep PoC-1..3 green. Personas select basins.
- **Phase 3**: more regions (one brain), cull, meta-opt, Triton on 5080,
  honest STATUS + HF model-index.
- **Phase 4**: progressive Rust rewrite, dual-run vs Python.

## Allocation

| Slice | Who |
|---|---|
| Plan, ADR, review, Forgejo/HF/GPU ops | Grok 4.5 / Codex (high) |
| One failing test + one function | `local/code` on resident 3090 |
| Qwen3-1024 measure, later Triton | Medium + 5080 `exclusive-seq` |
| Jules CLOSE recommendation batches | Small; operator closes GitHub |

Drive one closeable ID at a time with `/csd-python-first-drive`.
