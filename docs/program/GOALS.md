# Goal ledger — CogSynDelta / memory-gate

Living list for `/csd-goal-loop`. Update status when evidence exists.
Horizon rows stay **blocked** until Phase 1 exit. Not `STATUS.md`.

## Active (Phase 1)

| ID | Goal | Status | Blocker | Sandbox? | Refs |
|---|---|---|---|---|---|
| G-CI | Forgejo required checks **legitimately green** on memory-gate PR #1, then merge | met | Merged `500039e` as tzervas. Merge commit `7c1cdeba`. Jobs ran: quality, security, unit, integration, regression, gitleaks, trivy, fleet-ci python. Skips were rust-only `if:` (not fake green). | n/a | PR #1 |
| G-ERR | Public `memory_gate.errors` in use; mapper at store/gateway or documented leftover closed | met | Merged as tzervas PR #4 `81af7f98`. Taxonomy + `raise_mapped` on store/gateway. | n/a | P1-02 |
| G-FLEET | memory-gate `fleet-ci.yml` must not schedule cargo/rust jobs | met | Merged PR #5 `7595bd53` (head `7ecf2dc`). Python job **ran** (4m52s). No cargo job queued. `Cargo.toml` presence fails closed. Skip-if-rust was not kept. | n/a | fleet-ci.yml |
| G-STORE | In-memory oracle + store protocol conformance | met | Merged PR #6 `58934e6` (head `38156b3`). Combined status success: 15 jobs ran (quality, security, tests, fleet-ci python, gitleaks, trivy, commitizen). | n/a | P1-04 |
| G-SQL | Durable SQLite+sqlite-vec backend | open | After G-STORE (PR #6 merged). Next closeable. | yes | P1-06; SELF-HOSTED-DRIVE-TARGETS storage |
| G-QD | Qdrant 1024-d Qwen3 binding, fail-closed 384 | open | After G-STORE; 5080 only for measure | CPU tests yes; 5080 CUDA ok (Comfy masked) | P1-07 |
| G-LIFE | learn → retrieve → consolidate → persona through restart | open | After G-SQL, G-QD, CLS, persona | yes once deps met | P1-16 |
| G-RAG | `akula-csd-kb` 1024-d points > 0 | met | 83 points / 20 files, 1024-d Qwen3-Embedding-0.6B on 5080 python3.13 (Comfy masked). Never 3090 | n/a | vault `akula-csd-kb` |

## Blocked / horizon

| ID | Goal | Status | Unblock |
|---|---|---|---|
| G-CHROMA | Reintegrate Chroma | blocked | Named patched RC/stable in GHSA — gatekeeper WAN check |
| G-TRAIN | Region specialization + interconnect train/eval | blocked | Phase 1 exit (`G-LIFE`). Then SELF-HOSTED-DRIVE-TARGETS |

## Priority

`G-CI` → `G-ERR` → `G-FLEET` → `G-STORE` → `G-SQL` / `G-QD` → … → `G-LIFE`.
P1-01 docs PR #2 restacked; merge when legitimately green. Skip stalled.
Never start `G-TRAIN` from an autoloop.
