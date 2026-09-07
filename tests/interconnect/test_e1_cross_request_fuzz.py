"""Row E1 gate (iii): the cross-request poisoning fuzz, and the negative test that gives it teeth.

THE GATE, verbatim: *"CROSS-REQUEST FUZZ: >=10,000 interleaved writes across >=100 partitions,
then a read from every partition; ZERO reads return a value written under another scope, and
the negative test is a deliberately mis-derived scope key that MUST produce a cross-partition
read so the fuzz is shown to be capable of catching one."*

WHY THE NEGATIVE TEST IS THE LOAD-BEARING HALF. A fuzz that finds nothing is consistent with
two very different worlds: the store isolates, or the fuzz cannot see a crossing. Section 9.9
B2 calls the store *"the only writable cross-request object in the architecture"*, so the
difference matters. `test_the_fuzz_catches_a_crossing_when_the_key_is_mis_derived` runs the
IDENTICAL workload through a store whose scope derivation has been broken in the one specific
way that matters -- the scope segment dropped from the key -- and asserts the same check that
passes above FAILS there. The broken derivation is injected into the store under test rather
than simulated in a local dict, so what is shown to catch a crossing is the real assertion
against the real read path.

WHY THE PARTITIONS ARE ADVERSARIAL RATHER THAN `p0..p119`. Sequential principal names cannot
produce a key collision under any join, so a fuzz over them would pass against a store with no
escaping at all. The partition set below includes principals containing the key separator and
the escape character, and pairs like `("a/b", None)` against `("a", "b")` whose keys collide
under the obvious join -- the crossings a real principal space would eventually contain.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from cogsyndelta.interconnect.episodic import (
    EpisodicStoreImpl,
    InMemoryBackend,
    SqliteBackend,
    StoreBackend,
    TierBudget,
)
from cogsyndelta.interconnect.episodic_store import Scope, derive_scope

pytestmark = pytest.mark.cpu

DOMAINS = frozenset({"chat", "agent", "code_session"})
DOMAIN_LIST = sorted(DOMAINS)
WRITES = 10_000
"""The gate's floor, taken literally."""

PARTITIONS = 120
"""The gate's floor is 100; 120 leaves room for the adversarial pairs below without dropping
below it if one is ever removed."""

BIG = 1 << 40
SEED = 20260906


def _partitions() -> list[Scope]:
    """Build `PARTITIONS` scopes, including pairs that collide under a naive key join.

    Returns:
        A list of `Scope`s, all distinct as partitions.
    """
    scopes: list[Scope] = [
        derive_scope("a/b"),
        derive_scope("a", session="b"),
        derive_scope("a"),
        derive_scope("a", session=""),
        derive_scope("pct%2F"),
        derive_scope("pct", session="2F"),
    ]
    i = 0
    while len(scopes) < PARTITIONS:
        scopes.append(derive_scope(f"principal-{i}", session=None if i % 3 else f"sess-{i}"))
        i += 1
    assert len({(s.principal, s.session) for s in scopes}) == PARTITIONS
    return scopes


def _backends(tmp_path: Path) -> dict[str, Callable[[], StoreBackend]]:
    """Both durability oracles, as factories."""
    return {
        "in_memory": InMemoryBackend,
        "sqlite": lambda: SqliteBackend(tmp_path / "fuzz.db"),
    }


def _fuzz_store(backend: StoreBackend) -> EpisodicStoreImpl:
    """A store with capacity and tier budgets that cannot bind: this gate is about isolation.

    Eviction is gate (iv)'s subject. Letting the byte capacity bind here would mean a read
    that found nothing could be a spill rather than an isolation property, which would make a
    passing fuzz weaker evidence, not stronger.
    """
    store = EpisodicStoreImpl(
        DOMAINS,
        backend=backend,
        capacity_provider=lambda: BIG,
        host="fuzz-host",
        tier_budget=TierBudget(ram_max_items=WRITES * 2, disk_max_items=WRITES * 2),
    )
    store.start()
    return store


def _run_fuzz(
    store: EpisodicStoreImpl,
    scopes: list[Scope],
    *,
    read_partitions: int | None = None,
) -> list[tuple[int, float]]:
    """Interleave `WRITES` writes across `scopes`, then read partitions back.

    Every record's latent is filled with its OWN partition index, so a crossing is visible in
    the returned tensor rather than only in a key the test would have to trust.

    Args:
        store: The store under test.
        scopes: The partitions to write across.
        read_partitions: How many partitions to read back. `None` (the gate's shape) reads
            every one of them. The negative arm passes a small number for cost only: under a
            mis-derived key EVERY partition holds all 10,000 records, so reading all 120 of
            them re-materialises 1.2 million records to demonstrate a crossing that the first
            partition already shows.

    Returns:
        One `(expected_index, observed_value)` pair per unmasked slot returned by the reads.
    """
    rng = random.Random(SEED)  # noqa: S311 -- workload shuffling, not a security primitive
    latents = [torch.full((8,), float(i)) for i in range(len(scopes))]
    for n in range(WRITES):
        index = rng.randrange(len(scopes))
        domain = DOMAIN_LIST[n % len(DOMAIN_LIST)]
        store.learn(scopes[index], domain, f"episode-{n}", latents[index])

    observed: list[tuple[int, float]] = []
    for index, scope in enumerate(scopes[: read_partitions or len(scopes)]):
        values, mask = store.retrieve(scope, global_query=True, b_store=WRITES)
        for slot in range(int(mask.sum().item())):
            observed.append((index, float(values[slot][0].item())))
    return observed


@pytest.mark.parametrize("backend_name", ["in_memory", "sqlite"])
def test_cross_request_fuzz_returns_no_value_from_another_scope(
    backend_name: str, tmp_path: Path
) -> None:
    """10,000 interleaved writes over 120 partitions; every read stays inside its own scope."""
    scopes = _partitions()
    store = _fuzz_store(_backends(tmp_path)[backend_name]())
    observed = _run_fuzz(store, scopes)

    assert len(observed) == WRITES, (
        f"the fuzz read back {len(observed)} records, not the {WRITES} it wrote -- a read that "
        "loses records cannot prove anything about crossings"
    )
    crossings = [(expected, value) for expected, value in observed if value != float(expected)]
    assert not crossings, f"{len(crossings)} reads crossed scopes, e.g. {crossings[:5]}"


def test_the_fuzz_catches_a_crossing_when_the_key_is_mis_derived(tmp_path: Path) -> None:
    """THE NEGATIVE TEST: break the scope derivation and the same check must FAIL.

    The break is the specific one the design names -- *"partition key derived server-side...
    never from the request"* -- inverted: the scope segment is dropped from the key entirely,
    so every principal shares one partition. It is injected into the store class under test,
    so the crossing is produced by the real write and read paths.
    """
    scopes = _partitions()

    import cogsyndelta.interconnect.episodic.store as store_module

    original = store_module.partition_scope_key
    try:
        store_module.partition_scope_key = lambda scope: "SHARED"
        store = EpisodicStoreImpl(
            DOMAINS,
            backend=InMemoryBackend(),
            capacity_provider=lambda: BIG,
            host="fuzz-host",
            tier_budget=TierBudget(ram_max_items=WRITES * 2, disk_max_items=WRITES * 2),
        )
        store.start()
        observed = _run_fuzz(store, scopes, read_partitions=4)
    finally:
        store_module.partition_scope_key = original

    crossings = [(expected, value) for expected, value in observed if value != float(expected)]
    assert crossings, (
        "the mis-derived key produced NO crossing -- the fuzz above is therefore not shown to "
        "be capable of catching one, and gate (iii) has not fired"
    )


def test_the_healthy_store_uses_the_real_derivation_after_the_negative_test() -> None:
    """Positive control: the module-level patch above was undone, so the real key is back."""
    from cogsyndelta.interconnect.episodic.store import partition_scope_key

    assert partition_scope_key(derive_scope("alice")) != partition_scope_key(derive_scope("bob"))
