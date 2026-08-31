# Goal ledger — CogSynDelta / memory-gate

Living list for `/csd-goal-loop`. Update status when evidence exists.
Horizon rows stay **blocked** until Phase 1 exit. Not `STATUS.md`.

## Active (Phase 1)

| ID | Goal | Status | Blocker | Sandbox? | Refs |
|---|---|---|---|---|---|
| G-CI | Forgejo required checks **legitimately green** on memory-gate PR #1, then merge | open | `1c22643` ran: quality/unit/integration/regression/gitleaks/trivy **green**. Remaining honest red: Safety onnxruntime 1.22.0 (WAN lock ≥1.24.1). Do not merge red. Do not register a new runner. | wait WAN/gatekeeper | HANDOFF-NEXT; PR #1 |
| G-ERR | Public `memory_gate.errors` in use; mapper at store/gateway or documented leftover closed | open | Mapper on store+gateway. CI restack pushed `7a30b7b`. Merge blocked until required checks legitimately green (same onnxruntime WAN as G-CI). | wait CI after restack | P1-02 |
| G-STORE | In-memory oracle + store protocol conformance | open | After typed errors | yes | P1-04 |
| G-SQL | Durable SQLite+sqlite-vec backend | open | After G-STORE | yes | P1-06; SELF-HOSTED-DRIVE-TARGETS storage |
| G-QD | Qdrant 1024-d Qwen3 binding, fail-closed 384 | open | After G-STORE; 5080 only for measure | CPU tests yes; embeddings no if Comfy | P1-07 |
| G-LIFE | learn → retrieve → consolidate → persona through restart | open | After G-SQL, G-QD, CLS, persona | yes once deps met | P1-16 |

## Blocked / horizon

| ID | Goal | Status | Unblock |
|---|---|---|---|
| G-CHROMA | Reintegrate Chroma | blocked | Named patched RC/stable in GHSA — gatekeeper WAN check |
| G-RAG | `akula-csd-kb` 1024-d points > 0 | blocked | 5080 timeshare (video/Comfy). Enqueue only. Never 3090 |
| G-TRAIN | Region specialization + interconnect train/eval | blocked | Phase 1 exit (`G-LIFE`). Then SELF-HOSTED-DRIVE-TARGETS |

## Priority

`G-CI` → `G-ERR` → `G-STORE` → `G-SQL` / `G-QD` → … → `G-LIFE`. Skip stalled.
Never start `G-TRAIN` from an autoloop.
