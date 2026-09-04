# Python `memory-gate` gap map — 2026-08-30

> Official git copy of the 2026-08-30 Codex Phase 0 gap map. Vault twin:
> `akula-csd-kb/Architecture/Python-Memory-Gate-Gap-Map-2026-08-30.md`.
> Required rows are the Phase 1 exit ledger. Rust is reference only.

## Decision

`tzervas/memory-gate` Python is the Phase 1 vehicle. `memory-gate-rs` is a read-only capability
reference, not an implementation target. The Python exit condition is one durable, testable path:

`learn → retrieve → consolidate → persona instantiate`

The current Python implementation does not meet that condition. The largest correctness defect is
that `MemoryGateway.learn_from_interaction()` starts an untracked `asyncio.create_task()` and returns
success before persistence has completed. A process exit, cancellation, or backend failure can lose
the write after the caller was told learning succeeded.

## Truth reconciliation

- Python baseline: Forgejo `main` `95f24f3`; live files under `src/memory_gate/` and its tests.
- Rust reference: Forgejo `main` `02bcced`, plus the one-commit
  `feat/hypha-kv-tiers` reference branch.
- CogSynDelta integration is Phase 2. The older `src/cogsyndelta/memory/` research package is not a
  Phase 1 implementation source; `STATUS.md` validates the PoC compactor, not that whole package.
- `docs/HANDOFF-ARCHITECTURE.md` was treated only as unvetted hypotheses. Claims not present in
  source/tests were discarded, including any implication that true CLS, persona instantiation, MCP,
  Qwen3-1024, or a production tiered backend already exists.

## Required gap matrix

| Capability | Python evidence | Rust/reference evidence | Phase 1 verdict |
|---|---|---|---|
| Gateway lifecycle | `memory_gateway.py` adapts and learns; no retrieve façade, start/stop, drain, or flush contract. | `gateway.rs` exposes learn/retrieve/start/stop/manual consolidation/count/clear. | **REQUIRED**: define a single lifecycle façade and typed failure behavior. |
| Durable learn acknowledgement | `learn_from_interaction()` uses untracked `asyncio.create_task(store_experience(...))`; metrics mark success immediately. | Rust awaits `store()` and propagates failure. | **REQUIRED / P0**: await persistence or track tasks with backpressure, error capture, and shutdown drain. |
| Store contract | `memory_protocols.py` exposes only store/retrieve. Consolidation relies on methods outside that protocol. | `traits.rs` covers keyed read/write/delete/enumerate/count/clear plus filter/batch/vector traits. | **REQUIRED**: small explicit CRUD/query protocol with a conformance suite. |
| In-memory store | `storage/in_memory.py` supports basic store/retrieve and domain filtering. | `storage/in_memory.rs` implements the full keyed contract. | **PARTIAL**: bring it to the conformance contract; use as deterministic oracle. |
| Chroma vector store | `storage/vector_store.py` exists; PR #3 refuses server/`HttpClient`/`trust_remote_code` (client-only; not an upstream patch). PyPI and GitHub stable are `1.5.9` (CVE-2026-45829 unpatched). No named RC; do not pin nightly `1.5.10.dev266`. | No Chroma equivalent required. | **DROP-UNLESS-PATCHED**: keep client-only refuse; drop the extra at P1-05 if conformance needs a server. Reintegrate only on a **named** patched RC/stable that GHSA lists, then PyPI. SQLite+vec + Qdrant are the Phase 1 stores. |
| SQLite durability | No Python implementation despite an optional storage dependency group. | `storage/sqlite_vec.rs` provides a durable backend. | **REQUIRED**: CPU-first durable/reference backend with restart tests. |
| Qdrant durability | No Python implementation. | `storage/qdrant.rs` provides a network vector backend and binding checks. | **REQUIRED** for shared Akula deployment; namespace and dimension must fail closed. |
| Tier policy | No bounded RAM→durable spill/promotion store. | `feat/hypha-kv-tiers:storage/tiered.rs` has tested spill, promotion, pruning, and residency metadata. | **REQUIRED AFTER BACKENDS**: port policy semantics, not llama-specific flags. |
| Embedding identity | `embedding_catalog.py` binds stable model ids for 384/768-d models. | Rust catalog/binding reference is also 384/768-d. | **PARTIAL**: preserve binding behavior, add the selected Qwen3 1024-d profile. |
| Akula 1024-d isolation | No Qwen3-1024 profile or 1024-d collection contract. | Also absent from retained Rust main. | **REQUIRED**: never write 384-d data to shared Akula Qdrant; fail closed before first write. |
| Domain filter | Store/agent paths can filter a domain, but the gateway has no retrieval façade or namespace policy. | Traits/facade carry filters through the gateway. | **PARTIAL**: make domain a typed, tested end-to-end query constraint. |
| Fast CLS | `consolidation.py` deletes old low-importance rows; similarity merge/new abstractions are comments. | Rust consolidation also chiefly prunes; it is not proof of CLS. | **REQUIRED NEW DESIGN**: deterministic episode clustering/merge with provenance and golden tests. |
| Slow CLS | No separate scheduler, promotion policy, checkpoint, retry, or idempotence contract. | No production-grade proof in retained Rust source. | **REQUIRED NEW DESIGN**: bounded resumable slow pass; do not market pruning as consolidation. |
| Metrics truth | Prometheus instrumentation exists, but learn/consolidation can record success before durable completion. | `metrics.rs` offers useful names around awaited operations. | **REQUIRED**: success only after commit; add failure, queue depth, latency, retry, and consolidation-outcome metrics. |
| Persona backend | No persona repository adapter or namespace. | No operator-compatible Akula persona backend. | **REQUIRED**: read `/akula-data/obsidian/akula-personas/Personas`; keep it separate from every KB. |
| Persona semantics | No basin-selection object or instantiation result. | Rust agent/facade taxonomy is not an operator contract. | **REQUIRED**: persona selects a memory basin/behavior configuration; it is not an MoE expert or agent. |
| MCP/integration boundary | No MCP-facing implementation or stable application façade. | Rust facade is design reference only. | **REQUIRED BOUNDARY**: expose the lifecycle through a small Python API; wire external transports separately. |
| Golden recall | No pinned corpus or recall@k helper in Python. | `tests/fixtures/golden_corpus.json` and `tests/golden_recall.rs` exist, but the model-download test is ignored. | **REQUIRED**: CPU deterministic fixture plus measured Qwen3-1024 run; no claim from an ignored test. |
| Async test/benchmark validity | The suite passes while emitting unawaited-coroutine warnings; all three benchmark cases time coroutine creation instead of awaiting the operation. | Rust ignored tests are reported separately and do not masquerade as measurements. | **REQUIRED / P0**: RuntimeWarnings fail CI; rewrite async benchmarks before retaining any latency/throughput claim. |
| Restart/end-to-end proof | No single test covers learn, process/backend restart, retrieve, consolidate, persona instantiate. | Rust gateway/storage tests are separate; no matching persona lifecycle. | **REQUIRED EXIT TEST**: one hermetic lifecycle test plus a real Qdrant variant. |
| Typed public errors | Broad exceptions and asynchronous failures leak or disappear across layers. | `error.rs` is a useful enum/reference. | **REQUIRED**: stable errors for validation, dimension mismatch, unavailable backend, timeout, cancellation, and durability failure. |
| Adapters | `agent_interface.py` provides partial agent integration. | `adapters/` has a passthrough shape. | **KEEP NARROW**: adapters consume the gateway; they must not bypass durability or become a swarm. |
| VSA/holographic store | Absent from Python Phase 1 vehicle. | Optional Rust research modules exist. | **NOT A GAP**: defer; no Phase 1 feature request. |
| CogSynDelta memory wiring | Not present in the Python vehicle. | Not applicable. | **PHASE 2**, only after the Phase 1 exit test is green. |

## Source-level findings that must become regression tests

1. **Acknowledgement-before-write:** force a store failure after `learn_from_interaction()` is called;
   the current method can return success and lose the task exception. The repaired API must either
   return only after commit or return an explicit durable receipt whose completion is awaited.
2. **Shutdown drain:** queue multiple writes, stop immediately, restart the durable backend, and
   prove every acknowledged write is present exactly once.
3. **Dimension/model fail-closed:** reopen an existing collection with a different stable model id
   or dimension and prove initialization fails before data mutation. Add a direct 384↔1024 case.
4. **Consolidation honesty:** a failed scan/merge/delete must increment failure, not success. A fast
   CLS output must retain source ids, timestamps, domain, persona/basin, and transformation version.
5. **Domain isolation:** identical text in two domains must never cross a domain-filtered retrieval.
6. **Persona/KB isolation:** persona definitions come only from `akula-personas`; program memories
   go only to the configured process-memory store; `tzervas-dev-kb`, `akula-model-kb`, and
   `akula-csd-kb` are never persona stores.
7. **Golden recall:** pin corpus, queries, relevance judgments, embedding model revision, distance
   metric, `k`, and threshold. A skipped/download-gated test is not a passing quality gate.

## Validation evidence

- CSD PoC CPU truth: **21 passed, 5 skipped**. Only PoC-1/2/3 are supported by this result.
- Rust reference, all features: **121 passed, 32 ignored** across unit, integration, and doctests.
  Backend/model-download and golden-recall coverage remains ignored, so it supports API archaeology,
  not a production-readiness claim.
- Python Forgejo `main`, CPU-hidden run: **135 passed, 8 warnings** in 65.62 s with 97.73%
  reported coverage. This is not an honest clean gate: three benchmark tests emitted “coroutine was
  never awaited” and benchmarked coroutine construction rather than storage/GPU embedding work;
  shutdown, consolidation, and gateway tests emitted five more unawaited-coroutine warnings. No
  benchmark or performance claim may use these results, and RuntimeWarnings must become CI failures.
- Private HF repos `tzervas/cogsyndelta` and `tzervas/cogsyndelta-eval` were accessible and private;
  no artifact was changed.

## Exit gate

Phase 1 is complete only when all **REQUIRED** rows above have executable acceptance tests and the
end-to-end lifecycle succeeds through a restart. Stop and re-plan at that point. Do not begin CSD
memory wiring or Rust feature development early.
