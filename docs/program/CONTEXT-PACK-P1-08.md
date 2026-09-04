# P1-08 — bounded tiered store policy

Worktree: `/home/kang/code/personal/tzervas/python-ai/memory-gate-wt-p1-08`
Branch: `feat/tiered-memory-policy` from Forgejo `main` (`a823268`).
**Never** touch `python-ai/memory-gate` (`kang-main-wip`).

G-QD / P1-07 is **done**. Do not reopen Qdrant. This row is the policy
layer on top of existing `MemoryStore` backends.

## DoD (whole PR; not this tick)

RAM→durable spill, promotion, eviction, pruning, restart recovery,
residency metadata. Do not own llama.cpp KV. Backend failure must not
drop the only durable copy.

## This tick — one failing test only

Copy style from `tests/storage/test_sqlite_durable.py`.

- `pytest` + `@pytest.mark.asyncio`. **No** `unittest`, **no** `pass`.
- Import `memory_gate.storage.sqlite.SqliteDurableStore` and
  `memory_gate.memory_protocols.MemoryRecord`.
- Path **must** be `tests/storage/test_tiered_memory_policy.py`.
- One test: `test_hot_cap_spills_to_durable_and_survives_hot_eviction`.
- Arrange: wrap `SqliteDurableStore(tmp_path/"t.sqlite")` in a policy
  with `hot_cap=1`. `put` two records in the same domain. Assert the
  first is gone from hot/RAM and still `get`s from durable after a new
  `SqliteDurableStore` on the same path (restart).
- The test **must fail** because
  `memory_gate.storage.tiered.TieredMemoryStore` does not exist yet.
- Include `assert` on identity + content. Empty stubs are rejected.

Next tick (not now): implement
`src/memory_gate/storage/tiered.py` class `TieredMemoryStore` wrapping
a hot `InMemoryKnowledgeStore` + durable `SqliteDurableStore`.

## Forbidden

- `from src.…` or `src/tiered_memory_policy.py` or `src/memory_policy.py`
- Rewriting the same pass-stub `tests/test_tiered_memory_policy.py`
- G-TRAIN, WAN, dual 14B, Chroma, llama.cpp KV claims
