"""Row E1 gate (i): E0's nine named fixtures, green against the BUILT store, on both oracles.

WHY THIS FILE EXISTS INSTEAD OF AN EDIT TO E0's FILE. Gate (i) is *"E0's nine named tests go
green, and the diff that makes them green touches no test file -- a fix that edits its own
gate is refused."* E0 encoded the red half of its own gate as `pytest.raises(ContractGap)`
against `InMemoryStoreStub`, so `tests/interconnect/test_episodic_store.py` is a gate on the
STUB: making the stub implement DEC-63 would turn that file red, which is the same forbidden
move by a different door. So E1 adds a second implementation of E0's protocol beside the stub
and re-runs the nine fixtures against it here, verbatim in assertion and in name, minus the
`ContractGap` wrapper that only ever described the stub. `git diff` on E1's OWN branch showed
`tests/interconnect/test_episodic_store.py` and `src/cogsyndelta/interconnect/episodic_store.py`
untouched; that was gate (i)'s evidence, and it was checkable rather than asserted. Later
branches may edit both -- gate (i) forbids editing a gate to make it pass, not fixing a
measured defect in the contract both files implement (see `episodic/__init__.py`'s 2026-09-07
note); the nine fixtures below are unchanged by any of it.

BOTH ORACLES, ONE SUITE. Every test here is parameterised over `InMemoryBackend` (the
conformance oracle) and a file-backed `SqliteBackend` (the durability oracle), because row E1
names both and because a contract that holds on a dict and not on a database was a property of
the dict.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch

from cogsyndelta.interconnect.episodic import (
    GPU_RESIDENT_BONUS,
    MAX_IN_FLIGHT,
    EpisodicStoreImpl,
    InMemoryBackend,
    Lifecycle,
    Residency,
    SqliteBackend,
    StoreBackend,
    StoreBackpressureError,
    StoreDurabilityError,
    StoreLifecycleError,
    TierBudget,
    partition_scope_key,
)
from cogsyndelta.interconnect.episodic.backends import SqliteDurabilityError
from cogsyndelta.interconnect.episodic_store import (
    ContractGap,
    EpisodicStore,
    Scope,
    StoreScopeError,
    derive_scope,
)

pytestmark = pytest.mark.cpu

DOMAINS = frozenset({"chat", "agent", "code_session"})
BIG = 1 << 40
"""A capacity no test working set can reach, for the tests that are not about capacity."""


@pytest.fixture(params=["in_memory", "sqlite"])
def backend_factory(request, tmp_path: Path) -> Callable[[], StoreBackend]:
    """Build a fresh backend of the parameterised kind.

    Args:
        request: pytest's parameter carrier.
        tmp_path: pytest's per-test temp directory (private per ci_local run).

    Returns:
        A zero-argument factory producing a new backend each call.
    """
    if request.param == "in_memory":
        return InMemoryBackend
    counter = {"n": 0}

    def _sqlite() -> StoreBackend:
        counter["n"] += 1
        return SqliteBackend(tmp_path / f"store-{counter['n']}.db")

    return _sqlite


def _latent(dim: int = 8, fill: float = 1.0) -> torch.Tensor:
    return torch.full((dim,), fill)


def _store(
    backend_factory: Callable[[], StoreBackend],
    *,
    capacity: int = BIG,
    tier_budget: TierBudget | None = None,
    half_life_s: float = 3600.0,
) -> EpisodicStoreImpl:
    store = EpisodicStoreImpl(
        DOMAINS,
        backend=backend_factory(),
        capacity_provider=lambda: capacity,
        host="test-host",
        half_life_s=half_life_s,
        tier_budget=tier_budget or TierBudget(ram_max_items=1024, disk_max_items=4096),
    )
    store.start()
    return store


# ---------------------------------------------------------------------------------------
# Protocol conformance: the built store is a drop-in for the stub.
# ---------------------------------------------------------------------------------------


def test_built_store_satisfies_the_e0_protocol(backend_factory) -> None:
    """`mind.py` types against `EpisodicStore`; E1's store must be substitutable for E0's stub."""
    store = _store(backend_factory)
    assert isinstance(store, EpisodicStore)


# ---------------------------------------------------------------------------------------
# E0 fixture 1/9 -- domain isolation (`test_domain_isolation_same_logical_key`).
# ---------------------------------------------------------------------------------------


def test_e0_domain_isolation_same_logical_key(backend_factory) -> None:
    """The same logical key in two domains yields two records and never cross-leaks on read."""
    store = _store(backend_factory)
    scope = derive_scope("alice")
    store.learn(scope, "chat", "shared-key", _latent(fill=1.0))
    store.learn(scope, "agent", "shared-key", _latent(fill=2.0))

    chat_latents, chat_mask = store.retrieve(scope, domain="chat", b_store=1)
    agent_latents, agent_mask = store.retrieve(scope, domain="agent", b_store=1)
    assert chat_mask[0] and agent_mask[0]
    assert torch.allclose(chat_latents[0], _latent(fill=1.0))
    assert torch.allclose(agent_latents[0], _latent(fill=2.0))


# ---------------------------------------------------------------------------------------
# E0 fixture 2/9 -- `test_double_put_same_identity_last_write_wins`.
# ---------------------------------------------------------------------------------------


def test_e0_double_put_same_identity_last_write_wins(backend_factory) -> None:
    """A second write at one identity replaces the first rather than becoming a second row."""
    store = _store(backend_factory)
    scope = derive_scope("alice")
    store.learn(scope, "chat", "k", _latent(fill=1.0))
    store.learn(scope, "chat", "k", _latent(fill=2.0))

    latents, mask = store.retrieve(scope, domain="chat", b_store=2)
    assert mask.sum().item() == 1, "a double put at one identity must not become two records"
    assert torch.allclose(latents[0], _latent(fill=2.0))


# ---------------------------------------------------------------------------------------
# E0 fixture 3/9 -- `test_query_requires_domain_or_global_flag`.
# ---------------------------------------------------------------------------------------


def test_e0_query_requires_domain_or_global_flag(backend_factory) -> None:
    """A query names a domain or passes the global flag; omitting both is refused."""
    store = _store(backend_factory)
    scope = derive_scope("alice")
    store.learn(scope, "chat", "k", _latent())

    with pytest.raises(ValueError, match="query_requires_domain_or_global_flag"):
        store.retrieve(scope, b_store=4)

    latents, mask = store.retrieve(scope, global_query=True, b_store=4)
    assert mask.sum().item() == 1
    assert torch.allclose(latents[0], _latent())


# ---------------------------------------------------------------------------------------
# E0 fixture 4/9 -- caller cannot mutate the store through a read result.
# ---------------------------------------------------------------------------------------


def test_e0_caller_cannot_mutate(backend_factory) -> None:
    """Mutating a read result must not reach the stored record."""
    store = _store(backend_factory)
    scope = derive_scope("alice")
    store.learn(scope, "chat", "k", _latent(fill=1.0))

    latents, _mask = store.retrieve(scope, domain="chat", b_store=1)
    latents[0].fill_(999.0)

    again, _mask = store.retrieve(scope, domain="chat", b_store=1)
    assert torch.allclose(again[0], _latent(fill=1.0)), (
        "mutating a read result corrupted the store's own record"
    )


# ---------------------------------------------------------------------------------------
# E0 fixture 5/9 -- `admit_spills_lowest_importance`. Now a real assertion about ORDER, not
# a ContractGap: the store admits, the byte capacity binds, and the lowest-scored record is
# the one that leaves the resident set.
# ---------------------------------------------------------------------------------------


def test_e0_admit_spills_lowest_importance(backend_factory) -> None:
    """Admission past the byte capacity spills the lowest-importance record, not an arbitrary one."""
    span = 1000
    store = _store(backend_factory, capacity=2 * span)
    scope = derive_scope("alice")
    for name, importance in (("low", 0.1), ("mid", 0.5), ("high", 0.9)):
        store.learn(scope, "chat", name, _latent(), importance=importance, span_bytes=span)

    resident = {key[2] for key in store.resident_keys()}
    assert resident == {"mid", "high"}, f"expected the lowest importance to spill, got {resident}"
    assert store.resident_bytes <= 2 * span


# ---------------------------------------------------------------------------------------
# E0 fixture 6/9 -- `gpu_hint_protects_from_spill`.
# ---------------------------------------------------------------------------------------


def test_e0_gpu_hint_protects_from_spill(backend_factory) -> None:
    """A GPU-tagged span outscores a host-resident one of higher importance, within the bonus."""
    span = 1000
    store = _store(backend_factory, capacity=1 * span)
    scope = derive_scope("alice")
    store.learn(scope, "chat", "gpu-low", _latent(), importance=0.10, span_bytes=span)
    store.mark_gpu(scope, "chat", "gpu-low")
    store.learn(
        scope,
        "chat",
        "host-high",
        _latent(),
        importance=0.10 + GPU_RESIDENT_BONUS - 0.05,
        span_bytes=span,
    )

    resident = {key[2] for key in store.resident_keys()}
    assert resident == {"gpu-low"}, f"the GPU hint did not protect the span: {resident}"


# ---------------------------------------------------------------------------------------
# E0 fixture 7/9 -- `promote_returns_span_to_ram`.
# ---------------------------------------------------------------------------------------


def test_e0_promote_returns_span_to_ram(backend_factory) -> None:
    """A spilled span comes back to RAM when the caller promotes it."""
    span = 1000
    capacity = {"value": span}
    store = EpisodicStoreImpl(
        DOMAINS,
        backend=backend_factory(),
        capacity_provider=lambda: capacity["value"],
        host="test-host",
    )
    store.start()
    scope = derive_scope("alice")
    store.learn(scope, "chat", "a", _latent(), importance=0.1, span_bytes=span)
    store.learn(scope, "chat", "b", _latent(), importance=0.9, span_bytes=span)
    assert {k[2] for k in store.resident_keys()} == {"b"}

    capacity["value"] = 4 * span  # the tick moved; the store re-reads it, never caches it
    store.promote(scope, "chat", "a")
    assert {k[2] for k in store.resident_keys()} == {"a", "b"}


# ---------------------------------------------------------------------------------------
# E0 fixture 8/9 -- `retrieve_merges_tiers`.
# ---------------------------------------------------------------------------------------


def test_e0_retrieve_merges_tiers(backend_factory) -> None:
    """A read returns resident and spilled records together; residency is not a read filter."""
    span = 1000
    store = _store(backend_factory, capacity=span)
    scope = derive_scope("alice")
    store.learn(scope, "chat", "spilled", _latent(fill=1.0), importance=0.1, span_bytes=span)
    store.learn(scope, "chat", "resident", _latent(fill=2.0), importance=0.9, span_bytes=span)
    assert {k[2] for k in store.resident_keys()} == {"resident"}

    _latents, mask = store.retrieve(scope, domain="chat", b_store=4)
    assert mask.sum().item() == 2, "the read did not merge the spilled tier"


# ---------------------------------------------------------------------------------------
# E0 fixture 9/9 -- `disk_prune_drops_lowest`.
# ---------------------------------------------------------------------------------------


def test_e0_disk_prune_drops_lowest(backend_factory) -> None:
    """Disk-tier overflow hard-deletes the lowest-importance rows -- the only data-loss path."""
    store = _store(
        backend_factory,
        capacity=0,
        tier_budget=TierBudget(ram_max_items=1, disk_max_items=2),
    )
    scope = derive_scope("alice")
    for i in range(5):
        store.learn(scope, "chat", f"k{i}", _latent(), importance=float(i), span_bytes=100)

    surviving = {key[2] for key in store._index}
    assert surviving == {"k3", "k4"}, f"disk prune kept the wrong rows: {surviving}"


# ---------------------------------------------------------------------------------------
# The read tie-break: newest first, so a resident set is not a wall (2026-09-07).
# ---------------------------------------------------------------------------------------


def _tied_pair(backend_factory) -> tuple[EpisodicStoreImpl, Scope]:
    """Two records whose eviction SCORE is exactly equal, so only the tie-break can decide.

    Equal importance, equal residency and a `last_accessed` pinned to one instant on both,
    which zeroes the staleness term's difference -- the same pinning
    `test_e1_eviction_order.py` uses to isolate one term of the score at a time.
    """
    store = _store(backend_factory)
    scope = derive_scope("alice")
    store.learn(scope, "chat", "older", _latent(fill=1.0), importance=0.5)
    store.learn(scope, "chat", "newer", _latent(fill=2.0), importance=0.5)
    now = time.time()
    for logical_key in ("older", "newer"):
        store._index[(partition_scope_key(scope), "chat", logical_key)].last_accessed = now
    return store, scope


def test_a_tied_read_prefers_the_newer_record(backend_factory) -> None:
    """`episodic_store.py::_tie_break`'s repair, in the store that replaces that stub.

    The stub's oldest-first tie-break made a set of equally-important primers permanently
    outrank everything written later; this store inherited the same direction through
    `_rank_key`. Preferring the newer record also puts the read back in agreement with
    `_enforce_capacity`, which spills the OLDER record when scores tie.
    """
    store, scope = _tied_pair(backend_factory)
    latents, _mask = store.retrieve(scope, domain="chat", b_store=1)
    assert torch.allclose(latents[0], _latent(fill=2.0))


def test_the_old_ascending_tie_break_would_have_returned_the_older_record(
    backend_factory,
) -> None:
    """The can-fail control: the same records, ranked by the expression that was wrong."""
    store, scope = _tied_pair(backend_factory)
    now = time.time()
    matches = store._backend.partition(partition_scope_key(scope), "chat")
    scores = {r.key: store._rank_key(r, now)[0] for r in matches}
    assert len(set(scores.values())) == 1, "the fixture no longer ties the score"

    old_first = sorted(matches, key=lambda r: (scores[r.key], r.written_at, r.key))[0]
    assert old_first.logical_key == "older"
    new_first = sorted(matches, key=lambda r: (scores[r.key], -r.written_at, r.key))[0]
    assert new_first.logical_key == "newer"


# ---------------------------------------------------------------------------------------
# The query (`episodic_store.py` ambiguity note 6), on the built store and both oracles.
# ---------------------------------------------------------------------------------------


def _basis_store(backend_factory) -> tuple[EpisodicStoreImpl, Scope]:
    """Four records at the four coordinate axes of a `D = 4` space, uniform importance."""
    store = _store(backend_factory)
    scope = derive_scope("alice")
    for axis in range(4):
        record = torch.zeros(4)
        record[axis] = 1.0
        store.learn(scope, "chat", f"axis-{axis}", record)
    return store, scope


def test_the_query_reaches_the_built_store_through_both_verb_names(backend_factory) -> None:
    """`retrieve` and its protocol alias `read` must rank identically for one query.

    `mind.py` is typed against `EpisodicStore` and calls `read`; this store's `read` is a
    thin forward to `retrieve`, and a forward that dropped the new argument would silently
    put the constant back for every caller holding the real store instead of the stub.
    """
    store, scope = _basis_store(backend_factory)
    query = torch.tensor([0.0, 0.0, 1.0, 0.0])
    through_retrieve, _mask = store.retrieve(scope, domain="chat", b_store=4, query=query)
    through_read, _mask = store.read(scope, domain="chat", b_store=4, query=query)

    assert torch.equal(through_retrieve, through_read)
    assert torch.allclose(through_read[0], query)


def test_the_built_store_read_is_not_constant_across_queries(backend_factory) -> None:
    """The repair's own assertion, on the store that replaces the stub in production."""
    store, scope = _basis_store(backend_factory)
    bank_a, _mask = store.read(
        scope, domain="chat", b_store=4, query=torch.tensor([1.0, 0.0, 0.0, 0.0])
    )
    bank_b, _mask = store.read(
        scope, domain="chat", b_store=4, query=torch.tensor([0.0, 1.0, 0.0, 0.0])
    )
    assert float((bank_a - bank_b).abs().max()) > 0.0


def test_the_built_store_read_stays_constant_with_no_query(backend_factory) -> None:
    """The compatibility control: `query=None` is the merged contract, unchanged."""
    store, scope = _basis_store(backend_factory)
    first, _mask = store.read(scope, domain="chat", b_store=4)
    second, _mask = store.read(scope, domain="chat", b_store=4)
    assert float((first - second).abs().max()) == 0.0


def test_a_query_does_not_cross_a_scope_on_the_built_store(backend_factory) -> None:
    """Scope stays the isolation axis: a query pointing at another principal reads nothing."""
    store = _store(backend_factory)
    alice = derive_scope("alice")
    bob = derive_scope("bob")
    secret = torch.tensor([9.0, 9.0, 9.0, 9.0])
    store.learn(bob, "chat", "bobs-episode", secret)

    _latents, mask = store.read(alice, domain="chat", b_store=4, query=secret)
    assert not bool(mask.any())


def test_a_query_of_the_wrong_width_is_refused_by_the_built_store(backend_factory) -> None:
    """One `D` per store, checked on the read edge as well as the write edge."""
    store, scope = _basis_store(backend_factory)
    with pytest.raises(ValueError, match="width"):
        store.read(scope, domain="chat", b_store=4, query=torch.zeros(7))


# ---------------------------------------------------------------------------------------
# Identity: the scope segment is derived, and the derivation is injective.
# ---------------------------------------------------------------------------------------


def test_partition_scope_key_refuses_a_request_supplied_scope() -> None:
    """G32: only a `derive_scope` product may become a key segment."""
    with pytest.raises(StoreScopeError, match="derive_scope"):
        partition_scope_key({"principal": "alice"})  # type: ignore[arg-type]


def test_partition_scope_key_is_injective_across_separator_collisions() -> None:
    """`("a/b", None)` and `("a", "b")` must not name the same partition.

    A naive join makes them equal, which is a cross-scope read that needs no attacker -- only
    a principal with a separator in it. The escaping is what makes gate (iii)'s fuzz a test of
    the store rather than of the test's own key choices.
    """
    collide_a = partition_scope_key(derive_scope("a/b"))
    collide_b = partition_scope_key(derive_scope("a", session="b"))
    assert collide_a != collide_b
    assert partition_scope_key(derive_scope("a")) != partition_scope_key(
        derive_scope("a", session="")
    )


def test_a_scope_with_no_principal_has_no_partition() -> None:
    """An empty principal is refused at derivation and again at key time."""
    with pytest.raises(ValueError, match="principal"):
        derive_scope("")
    with pytest.raises(StoreScopeError, match="principal"):
        partition_scope_key(Scope(principal=""))


def test_g32_learn_with_no_scope_is_refused(backend_factory) -> None:
    """An unknown principal has no legal partition to write into."""
    store = _store(backend_factory)
    with pytest.raises(StoreScopeError, match="no server-derived scope"):
        store.learn(None, "chat", "k", _latent())


def test_g32_read_with_no_scope_degrades_to_an_empty_partition(backend_factory) -> None:
    """A caller with no identity sees nothing rather than crashing the request."""
    store = _store(backend_factory)
    store.learn(derive_scope("alice"), "chat", "k", _latent())
    _latents, mask = store.retrieve(None, domain="chat", b_store=4)
    assert mask.sum().item() == 0


def test_domain_outside_the_enum_is_refused(backend_factory) -> None:
    """The task axis is a closed enum, not a free string."""
    store = _store(backend_factory)
    scope = derive_scope("alice")
    with pytest.raises(ValueError, match="domain_enum"):
        store.learn(scope, "not-a-real-domain", "k", _latent())
    with pytest.raises(ValueError, match="domain_enum"):
        store.retrieve(scope, domain="not-a-real-domain", b_store=4)


# ---------------------------------------------------------------------------------------
# The six lifecycle verbs, and refusing backpressure.
# ---------------------------------------------------------------------------------------


def test_data_verbs_refuse_before_start_and_after_stop(backend_factory) -> None:
    """`start()` is not decorative: a store accepts work between start and stop, and not outside."""
    store = EpisodicStoreImpl(
        DOMAINS,
        backend=backend_factory(),
        capacity_provider=lambda: BIG,
        host="test-host",
    )
    scope = derive_scope("alice")
    assert store.state is Lifecycle.NEW
    with pytest.raises(StoreLifecycleError, match="not running"):
        store.learn(scope, "chat", "k", _latent())

    store.start()
    store.learn(scope, "chat", "k", _latent())
    store.stop()
    assert store.state is Lifecycle.STOPPED
    with pytest.raises(StoreLifecycleError, match="not running"):
        store.retrieve(scope, domain="chat", b_store=1)
    with pytest.raises(StoreLifecycleError, match="terminal"):
        store.start()


def test_backpressure_refuses_rather_than_queues(backend_factory) -> None:
    """At `max_in_flight`, the next verb is REFUSED synchronously -- nothing is enqueued."""
    store = _store(backend_factory)
    scope = derive_scope("alice")
    with ExitStack() as stack:
        for i in range(MAX_IN_FLIGHT):
            stack.enter_context(store.admission_slot(f"holder-{i}"))
        assert store.in_flight == MAX_IN_FLIGHT
        with pytest.raises(StoreBackpressureError, match="max_in_flight is 32"):
            store.learn(scope, "chat", "k", _latent())
        with pytest.raises(StoreBackpressureError):
            store.retrieve(scope, domain="chat", b_store=1)
        with pytest.raises(StoreBackpressureError):
            store.flush()
        # drain is deliberately NOT bounded by backpressure: it is reached, it WAITS for the
        # held slots, and it reports a timeout -- it is never refused at submit the way the
        # three verbs above are. That distinction is the divergence the module docstring
        # names, pinned here rather than merely documented.
        with pytest.raises(StoreDurabilityError, match="still in flight"):
            store.drain(timeout_s=0.05)

    assert store.in_flight == 0
    store.learn(scope, "chat", "k", _latent())


def test_backpressure_bound_is_thirty_two() -> None:
    """The number is the contract's, not a tunable that drifted."""
    assert MAX_IN_FLIGHT == 32


def test_stop_is_refuse_new_then_drain_then_flush(backend_factory) -> None:
    """Lifecycle clause (3)'s order, observable from the outside."""
    store = _store(backend_factory)
    scope = derive_scope("alice")
    store.learn(scope, "chat", "k", _latent())
    store.stop()
    assert store.state is Lifecycle.STOPPED
    store.stop()  # idempotent


# ---------------------------------------------------------------------------------------
# The durability oracle is a durability oracle.
# ---------------------------------------------------------------------------------------


def test_sqlite_is_the_durability_oracle_and_says_so(tmp_path: Path) -> None:
    """WAL + `synchronous=FULL` are enforced at connect, and a file-backed store is durable."""
    backend = SqliteBackend(tmp_path / "d.db")
    assert backend.durability == "durable"
    assert int(backend._conn.execute("PRAGMA synchronous").fetchone()[0]) == 2
    assert str(backend._conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
    backend.close()


def test_in_memory_is_the_conformance_oracle_and_makes_no_durability_claim() -> None:
    """Section 1.3 clause (4), stated in the code rather than only in the design."""
    assert InMemoryBackend().durability == "conformance"
    assert SqliteBackend(":memory:").durability == "conformance"


def test_acked_writes_survive_reopening_the_database(tmp_path: Path) -> None:
    """Durable-first: a receipt means committed, so a reopened store finds the record."""
    path = tmp_path / "survive.db"
    scope = derive_scope("alice")
    first = EpisodicStoreImpl(
        DOMAINS, backend=SqliteBackend(path), capacity_provider=lambda: BIG, host="test-host"
    )
    first.start()
    first.learn(scope, "chat", "k", _latent(fill=7.0))
    first.stop()

    second = EpisodicStoreImpl(
        DOMAINS, backend=SqliteBackend(path), capacity_provider=lambda: BIG, host="test-host"
    )
    second.start()
    latents, mask = second.retrieve(scope, domain="chat", b_store=1)
    assert mask[0]
    assert torch.allclose(latents[0], _latent(fill=7.0))
    assert second.resident_bytes > 0, "the index was not rebuilt from the database"
    second.stop()


def test_a_sqlite_backend_that_cannot_hold_its_pragmas_is_refused() -> None:
    """The durability check FIRES: a connection reporting synchronous=NORMAL is refused.

    Constructed rather than observed, because SQLite will honour the pragma on any healthy
    build -- so the only way to know this check can fail is to feed it the failing answer. A
    guard never seen to fire is a guard nobody has evidence for.
    """

    class _Row(list):
        def fetchone(self):
            return self

    class _FakeConn:
        def __init__(self, synchronous: int, journal: str) -> None:
            self.synchronous = synchronous
            self.journal = journal

        def execute(self, sql: str):
            key = sql.strip().upper()
            if key == "PRAGMA SYNCHRONOUS":
                return _Row([self.synchronous])
            if key == "PRAGMA JOURNAL_MODE":
                return _Row([self.journal])
            raise AssertionError(f"unexpected sql {sql!r}")

    backend = SqliteBackend.__new__(SqliteBackend)
    backend._path = "degraded.db"
    backend._conn = _FakeConn(1, "wal")
    with pytest.raises(SqliteDurabilityError, match="not 2"):
        backend._assert_pragmas()

    backend._conn = _FakeConn(2, "delete")
    with pytest.raises(SqliteDurabilityError, match=r"not 'wal'"):
        backend._assert_pragmas()

    backend._conn = _FakeConn(2, "wal")
    backend._assert_pragmas()


# ---------------------------------------------------------------------------------------
# DEC-66 is still deferred, and says which decision deferred it.
# ---------------------------------------------------------------------------------------


def test_consolidate_remains_a_named_contract_gap(backend_factory) -> None:
    """E1 builds the container; DEC-66 defers learned consolidation to phase 3."""
    store = _store(backend_factory)
    with pytest.raises(ContractGap) as exc_info:
        store.consolidate()
    assert exc_info.value.dec == "DEC-66"


def test_bind_session_bounds_an_unbounded_client_string(backend_factory) -> None:
    """DEC-64's "the server MAY bound `session`" -- implemented, no longer a gap."""
    store = _store(backend_factory)
    bound = store.bind_session(derive_scope("alice", session="../x" * 100))
    assert bound.principal == "alice"
    assert bound.session is not None
    assert len(bound.session) <= 64
    assert "/" not in bound.session
    assert store.bind_session(derive_scope("alice")).session is None


def test_capacity_bytes_refuses_to_answer_for_another_host(backend_factory) -> None:
    """Capacity is per host; answering for the wrong card is gate (ii)'s failure mode."""
    store = _store(backend_factory)
    assert store.capacity_bytes("test-host") == BIG
    with pytest.raises(ValueError, match="per host"):
        store.capacity_bytes("some-other-host")


def test_residency_tags_move_no_bytes(backend_factory) -> None:
    """`mark_gpu` records residency and moves nothing -- DEC-65's model, asserted."""
    store = _store(backend_factory)
    scope = derive_scope("alice")
    store.learn(scope, "chat", "k", _latent(), span_bytes=64)
    before = store.resident_bytes
    store.mark_gpu(scope, "chat", "k")
    assert store.resident_bytes == before
    key = (partition_scope_key(scope), "chat", "k")
    assert store._index[key].residency is Residency.GPU
