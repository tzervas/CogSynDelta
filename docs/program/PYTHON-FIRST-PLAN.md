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

## Compute split (until a CSD model is actually deployed)

Do **not** train or ship CogSynDelta weights yet. Use the lab as assist:

| Resource | Until CSD deploy | Do not |
|---|---|---|
| Hosted Grok 4.6 / 4.5 | Orchestration, ADRs, review, Forgejo ops, swarms/workflows | Stuff implementation into the parent chat |
| **3090 Ti LocalAI** `local/code` (Qwen2.5-Coder-14B) | Keep loaded. Implementation slices: one failing test + one function | Pause for RAG; dual-load a second 14B; swap to `local/reasoning` unless a plan-only exclusive job |
| **5080** | Enqueue CUDA / `csd-kb-index` / later Qwen3-1024 measure | Steal from healthy Comfy; index with the 3090; mix 384-d |
| Homelab CPU | Forgejo product CI once a **tzervas-scoped** runner exists | Pretend cabal-collective runners cover `tzervas/*` |

Parallelism is **3090 inference while a 5080 job runs**, not two jobs on one card.

## Hugging Face datasets (if and when)

Private Hub only, when a board row actually needs a corpus:

- Model checkpoints later: `tzervas/cogsyndelta`
- Eval dumps / golden sets: `tzervas/cogsyndelta-eval`
- First consumer is **P1-15** (golden recall). Until then do not invent Hub numbers or upload empty cards.
- Public datasets may be **copied into** the private eval repo if/when a test needs them. Pin revision, license, and split in the card YAML `datasets:` list. Never treat a skipped download as a pass.

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

Chroma is unpatched through PyPI/GitHub `1.5.9`. There is no named GitHub RC
(checked 2026-08-30); nightly `1.5.10.dev266` is not a patch. Phase 1 stores are
SQLite+sqlite-vec (local) and Qdrant (Akula). Reintegrate Chroma only after a
named patched RC/stable that GHSA lists; then roll that pin until PyPI ships the
same cut. See [PHASE-1-TASK-BOARD.md](PHASE-1-TASK-BOARD.md).

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
