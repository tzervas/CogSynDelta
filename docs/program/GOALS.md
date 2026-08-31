# Goal ledger — CogSynDelta / memory-gate

Living list for `/csd-goal-loop`. Update status when evidence exists.
Horizon rows stay **blocked** until Phase 1 exit. Not `STATUS.md`.

## Active (Phase 1)

| ID | Goal | Status | Blocker | Sandbox? | Refs |
|---|---|---|---|---|---|
| G-CI | Forgejo required checks **legitimately green** on memory-gate PR #1, then merge | open | Head `1c22643`. **Do not register a new runner** — `homelab-cpu` id 5 is already active with the six labels. Wait for that run. onnxruntime 1.22.0 bump is WAN/gatekeeper. Do not merge red. | wait existing runner | HANDOFF-NEXT; PR #1 |
| G-ERR | Public `memory_gate.errors` in use; mapper at store/gateway or documented leftover closed | open | **Code on PR #4 `b80cd13`**: store+gateway `raise_mapped`; VectorStore* is `__cause__`. Consolidation leftover documented. **Merge blocked**: Forgejo 1 ok / 14 fail (not legitimately green). Restack CI pins from PR #1 rather than a new runner | wait CI / restack | P1-02 |
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
