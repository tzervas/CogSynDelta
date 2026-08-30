# Phase 1 task board — Python `memory-gate`

> Official git copy of the 2026-08-30 Codex Phase 1 board. Vault twin:
> `akula-csd-kb/Program/Phase-1-Python-Memory-Gate-Task-Board-2026-08-30.md`.
> Drive with `/csd-python-first-drive`. One architecture change per PR.

## Outcome and stop rule

Make the Python repository complete enough that one honest test proves:

`learn → retrieve → consolidate → persona instantiate`

including durable restart behavior. Then stop and re-plan Phase 2. Do not add Rust features, wire
CogSynDelta early, or market pruning as CLS.

Every row below is one architecture change and one Forgejo PR. Branch from the current Forgejo
`main`, use Conventional Commits, self-review, and merge only when required Forgejo checks are truly
green. GitHub remains a read-only operator mirror.

## Sequenced board

| ID | Depends on | Atomic PR / branch | Acceptance gate | Allocation |
|---|---|---|---|---|
| `P1-00` | Phase 0 | **Restore honest CI** — `ci/forgejo-cpu-fail-closed` | `homelab-cpu` is active and registered; every workflow uses `[self-hosted, linux, x64, podman, compute-cpu, host-homelab]`; dependency install, pytest, ruff, mypy, scans, and unawaited-coroutine RuntimeWarnings propagate non-zero exit; no `ubuntu-latest`, `.forgejo/workflows`, GPU label, `continue-on-error`, fallback `echo`, or GitHub push. Recreate the selector; selectively take `ci/false-green-sweep`. | High-reasoning review + homelab CPU ops. No GPU. |
| `P1-01` | `P1-00` | **Specify lifecycle and data contract** — `docs/memory-lifecycle-contract` | SDD spec, ADR, and task acceptance fixtures define durable acknowledgement, CRUD/query protocol, domain, persona/basin, provenance, typed errors, cancellation, shutdown drain, stable embedding identity, and 1024-d fail-closed rules. No code. | High-reasoning Codex/Grok; adversarial self-review. |
| `P1-02` | `P1-01` | **Implement typed public errors** — `feat/public-error-contract` | Stable public exception types cover validation, dimension/model mismatch, unavailable backend, timeout, cancellation, durability failure, and consolidation failure. Backend-specific exceptions are chained but never leak as the public contract; exhaustive tests prove each mapping. | `local/code`: one error type + one mapping test per slice; high-reasoning API review. |
| `P1-03` | `P1-02` | **Make gateway writes and lifecycle durable** — `fix/gateway-durable-lifecycle` | A failing test first demonstrates lost/unobserved `create_task`; repaired learn awaits commit or returns an explicit awaited receipt. `start`, `stop`, `drain`, and `flush` have typed cancellation/failure behavior; bounded backpressure and shutdown/restart tests pass. Rewrite the three async benchmarks to await real operations and eliminate all eight baseline unawaited-coroutine warnings. | `local/code` on 3090: one failing test + one function slice; high-reasoning lifecycle review. LocalAI stays resident. |
| `P1-04` | `P1-03` | **Complete store protocol and in-memory oracle** — `feat/store-conformance-contract` | Typed store protocol covers put/get/query/delete/enumerate/count/clear and required metadata; a backend-neutral conformance suite passes against deterministic in-memory storage, including domain isolation and idempotent keys. | `local/code`: one protocol/test slice at a time; high-reasoning contract review. |
| `P1-05` | `P1-04` | **Adjudicate Chroma backend** — `test/chroma-conformance` | Run the common contract and close/reopen/restart suite against Chroma. Retain it only if clean; otherwise remove it and its dependency in this PR with a migration note. No backend may remain nominally supported without conformance evidence. | `local/code` test slices; high-reasoning keep/drop review. CPU only. |
| `P1-06` | `P1-04` | **Add durable SQLite backend** — `feat/sqlite-durable-store` | Full conformance suite passes; acknowledged rows survive close/reopen and process-style restart; migrations are versioned; interruption cannot leave a reported-success write absent; CPU-only CI is sufficient. | `local/code` implementation slices; homelab CPU integration. |
| `P1-07` | `P1-04` | **Add Qdrant backend and Qwen3-1024 binding** — `feat/qdrant-qwen3-1024` | Separate test collection uses the selected pinned Qwen3 revision at 1024 dimensions; stable model id, dimension, metric, and namespace are stored and validated before mutation; 384↔1024 and model-id mismatch tests fail closed; shared operator/model KBs are never written. | High-reasoning schema/ADR; Medium model with 5080 `exclusive-seq` only for embedding measurement; CPU tests use a fake/local service. |
| `P1-08` | `P1-06`, `P1-07` | **Add bounded tiered store policy** — `feat/tiered-memory-policy` | Policy tests prove RAM→durable spill, promotion, eviction, pruning, restart recovery, and explicit residency metadata without pretending to own llama.cpp KV bytes. Backend failures never discard the only durable copy. | `local/code` policy slices; high-reasoning safety review. No Rust implementation. |
| `P1-09` | `P1-04` | **Add retrieval and domain isolation** — `feat/gateway-retrieve-domain` | Gateway retrieve uses the typed store protocol and typed errors; domain is mandatory or explicitly global; identical text in different domains cannot cross; ranking/limits are deterministic for the in-memory oracle. | `local/code`: one query/test slice; high-reasoning public API review. |
| `P1-10` | `P1-03`, `P1-09` | **Make metrics reflect commits** — `fix/metrics-durability-truth` | Success increments only after durable completion. Tests cover failure, cancellation, queue depth/backpressure, retries, store latency, retrieval outcome, and consolidation outcome. Metric labels are bounded and contain no user content. | `local/code` metrics slices; high-reasoning observability review. |
| `P1-11` | `P1-08`, `P1-09` | **Implement fast CLS transform** — `feat/cls-fast-pass` | Specified deterministic clustering/merge produces a new representation with source ids, timestamps, domain, persona/basin, transform version, and reversible audit link; it does more than delete low-importance rows; retry is idempotent. | High-reasoning algorithm/ADR; `local/code` one golden test + transform function. |
| `P1-12` | `P1-11` | **Implement resumable slow CLS** — `feat/cls-slow-pass` | Bounded scheduler/checkpoint processes a known window, can stop/restart without double promotion, reports partial/failure honestly, and never runs uncontrolled background work. Slow promotion policy has measured CPU baseline before any GPU experiment. | High-reasoning scheduling; `local/code` checkpoint slices; homelab CPU soak. |
| `P1-13` | `P1-09`, `P1-11` | **Add Akula persona backend** — `feat/akula-persona-backend` | Read-only adapter loads versioned definitions only from `/akula-data/obsidian/akula-personas/Personas`; validates schema; instantiates a basin-selection/configuration object; never writes or queries a KB and never models personas as MoE experts/agents. Fixture tests use a temporary persona directory. | High-reasoning semantics/security; `local/code` parser/test slices. |
| `P1-14` | `P1-10`, `P1-12`, `P1-13` | **Implement stable application façade and adapter boundary** — `feat/memory-application-facade` | One public Python façade exposes typed `start`, `stop`, `drain`, `flush`, `learn`, `retrieve`, `consolidate`, and `instantiate_persona`. Existing agent adapters consume only this façade and cannot call stores directly. Transport/MCP bindings remain separate thin consumers; tests prove no durability bypass and no swarm semantics. | High-reasoning boundary design/review; `local/code` one adapter test + one façade method per slice. |
| `P1-15` | `P1-07`, `P1-09` | **Add golden recall evaluation** — `test/golden-recall-qwen3` | Corpus, queries, relevance labels, model revision, distance metric, `k`, and threshold are pinned. Deterministic ranking logic runs on CPU; a 5080 exclusive-seq job records actual Qwen3-1024 recall and latency. A skip cannot satisfy the required check. | High-reasoning eval design; Medium + 5080 for measured run; Small model may format results only. |
| `P1-16` | `P1-14`, `P1-15` | **Prove the lifecycle through restart** — `test/memory-gate-lifecycle` | Hermetic SQLite scenario and real Qdrant scenario both execute learn → close/restart → retrieve → fast/slow consolidate → persona instantiate through the public façade. All acknowledged input remains traceable, domain-isolated, and correctly dimensioned; adapter-bypass and failure injection tests pass. | High-reasoning integration/review; homelab CPU, then Medium + 5080 only for real embedding leg. |
| `P1-17` | `P1-16` | **Close the Python-vs-Rust gap ledger** — `docs/phase1-exit` | Every REQUIRED row in the Phase 0 gap map links to a passing test; STATUS and private HF model-index/card contain only measured claims; Rust-only/VSA items remain explicitly deferred. Publish no checkpoint unless one was actually produced. **Stop and re-plan Phase 2.** | High-reasoning Codex/Grok; HF/Forgejo ops under operator policy. |

`P1-05`, `P1-06`, and `P1-07` may execute concurrently after `P1-04`, but their PRs stay
independent. All other
dependencies are hard sequencing gates.

## Work-slice and context policy

- High-reasoning agents own plans, ADRs, API/schema decisions, strata adjudication, adversarial
  review, Forgejo/HF operations, and exit-claim review.
- `local/code` owns exactly one failing test plus one implementation function per prompt. It uses the
  already resident 3090 LocalAI model; do not pause it for RAG or CPU tests.
- Medium reasoning gets the 5080 only through `with-gpu-5080 --mode exclusive-seq` for Qwen3
  calibration/recall measurement. If Comfy is active, enqueue rather than preempt.
- Small models prepare GitHub/Jules CLOSE recommendation batches and mechanical report formatting;
  an operator performs any GitHub closure.
- Use tightly scoped prompts containing only the contract, one failing test, the target function,
  and its acceptance command. Return a compact handoff with changed paths, command, result, and
  unresolved risk; do not paste whole-repository context into implementation agents.

## Non-negotiable gates

1. Do not start feature PRs until `P1-00` has produced a real failing-when-broken Forgejo run.
2. Preserve local `main` divergence through `local/kang-main-wip`; never reset or overwrite it.
3. No collection may change its embedding dimension in place. Shared Akula Qdrant means 1024-d
   Qwen3 or a separate, explicitly named non-shared collection.
4. Personas select a basin; specialized CogSynDelta centers remain parts of one brain. Neither is a
   multi-agent swarm.
5. No Rust feature work before **program Phase 3** exits—that is, after CogSynDelta Python is
   production-ready, not merely after this Phase 1 board. `feat/hypha-kv-tiers` is reference only.
6. A green check obtained through skip, `|| true`, `continue-on-error`, fallback `echo`, or missing
   runner is not green.

## Current operational blockers

- Before `P1-00`, finish the 5080 KB-index **runtime** deploy. Launcher path is fixed to enqueue
  `rag-index`/`csd-kb` onto the 5080 **local** timeshare state (not prime-only venv SSH, not prime
  queue file). Still blocked for nonempty `akula-csd-kb`: no indexer/torch on 5080, vault not
  mounted, Qdrant bound `127.0.0.1` on prime only, Comfy holds ~11 GiB. Do not use the 3090 fallback.
- Forgejo CPU runners (live 2026-08-30): `homelab-cpu` (id 4) on homelab **and** `akula-prime-cpu`
  (id 2) on prime — both user-unit `forgejo-runner-cpu`, labels
  `self-hosted,linux,x64,podman,compute-cpu,host-homelab`, both picking jobs. Product CI host of
  record remains either labeled host; prefer always-up **homelab** for long jobs. `P1-00` is no
  longer blocked on “runner missing.”
- Current Python workflows omit `compute-cpu` and `host-homelab`; one branch adds the noncanonical
  `scribe-cpu-build`. Re-author the selector instead of merging that branch.
- Current Python fleet CI masks test failures with fallback commands. Land the focused fail-closed
  behavior before trusting any product gate.
