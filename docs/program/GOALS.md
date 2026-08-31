# Goal ledger — CogSynDelta / memory-gate

Living list for `/csd-goal-loop`. Update status when evidence exists.
Horizon rows stay **blocked** until Phase 1 exit. Not `STATUS.md`.

## Active (Phase 1)

| ID | Goal | Status | Blocker | Sandbox? | Refs |
|---|---|---|---|---|---|
| G-CI | Forgejo required checks **legitimately green** on memory-gate PR #1, then merge | met | Merged `500039e` as tzervas. Merge commit `7c1cdeba`. Jobs ran: quality, security, unit, integration, regression, gitleaks, trivy, fleet-ci python. Skips were rust-only `if:` (not fake green). | n/a | PR #1 |
| G-ERR | Public `memory_gate.errors` in use; mapper at store/gateway or documented leftover closed | open | Merged `forgejo/main` (`7c1cdeba`) + `raise_mapped`. Local trivy clean (`cryptography` 50.0.1). Head `f586dde` pushed. Merge only if Forgejo required checks **ran and succeeded**. | wait CI | P1-02 |
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
