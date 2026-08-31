# G-QD / P1-07 — autodev slice pack

Repo: `/home/kang/code/personal/tzervas/python-ai/memory-gate-wt-p1-07`
Branch: `feat/qdrant-qwen3-1024` (Forgejo PR **#9**).
**Never** touch `/home/kang/code/personal/tzervas/python-ai/memory-gate`
(`kang-main-wip` @ `691bb85`).

## Already landed (do not rewrite)

`QdrantMemoryStore`, `InProcessQdrant`, pinned Qwen3-1024 Cosine
(`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`), fail-closed 384,
shared-KB refuse, ADR-0003, tests in `tests/storage/test_qdrant_qwen3.py`.

## Your job this tick

If PR #9 CI is still running or green: return
`{"path": null, "content": null, "note": "wait-ci"}`.
If a **required** check **ran and failed**: one failing test + one
function in this worktree. CPU fake Qdrant only. 5080 exclusive-seq
only for a later embed measure (P1-15). Never write tzervas-dev-kb /
akula-model-kb. Never mix 384-d into shared collections.

Return JSON only: `{"path": "src/...", "content": "..."}` or wait-ci.
