"""Tests for `cogsyndelta.interconnect.episodic_store` (spec Table 9, `episodic_store.py` row).

Table 9's own words: *"scope isolation on the stub; a mis-derived scope key produces a
crossing in the negative test; each `ContractGap` names its DEC; E0's nine ported fixtures
run and fail red on the clauses the stub does not implement."* Four things, four sections
below (plus shapes/masks/determinism, required by every component's row in Table 9's own
header).

Every guard test here follows `tests/test_guards_can_fail.py`'s pattern: construct the exact
condition the guard exists to catch, and assert it IS caught, alongside a positive control
so a guard that fires on everything is not mistaken for one that works.
"""

from __future__ import annotations

import pytest
import torch

from cogsyndelta.interconnect.episodic_store import (
    ContractGap,
    EpisodicStore,
    InMemoryStoreStub,
    Scope,
    StoreScopeError,
    WriteReceipt,
    derive_scope,
)

pytestmark = pytest.mark.cpu

DOMAINS = frozenset({"chat", "agent", "code_session"})


def _store(*, half_life_s: float = 3600.0, importance_default: float = 0.5) -> InMemoryStoreStub:
    return InMemoryStoreStub(
        domain_enum=DOMAINS, half_life_s=half_life_s, importance_default=importance_default
    )


def _latent(dim: int = 8, fill: float = 1.0) -> torch.Tensor:
    return torch.full((dim,), fill)


# ---------------------------------------------------------------------------------------
# Protocol conformance.
# ---------------------------------------------------------------------------------------


def test_stub_satisfies_the_episodic_store_protocol() -> None:
    """`mind.py` (IC-8) will type against `EpisodicStore`, not this stub -- the
    `@runtime_checkable` Protocol must actually recognise it."""
    assert isinstance(_store(), EpisodicStore)


# ---------------------------------------------------------------------------------------
# Shapes and masks (every Table 9 row's baseline requirement).
# ---------------------------------------------------------------------------------------


def test_read_shape_matches_b_store_regardless_of_how_many_records_exist() -> None:
    store = _store()
    scope = derive_scope("alice")
    for i in range(3):
        store.write(scope, "chat", f"turn-{i}", _latent())

    latents, mask = store.read(scope, domain="chat", b_store=8)
    assert latents.shape == (8, 8)
    assert mask.shape == (8,)
    assert mask.dtype == torch.bool
    assert mask.sum().item() == 3


def test_read_truncates_to_b_store_when_more_records_exist_than_slots() -> None:
    store = _store()
    scope = derive_scope("alice")
    for i in range(5):
        store.write(scope, "chat", f"turn-{i}", _latent(fill=float(i)), importance=float(i))

    latents, mask = store.read(scope, domain="chat", b_store=2)
    assert latents.shape == (2, 8)
    assert mask.tolist() == [True, True]
    # Ranked by descending importance (ambiguity note 3): turns 4 and 3 win, in that order.
    assert torch.allclose(latents[0], _latent(fill=4.0))
    assert torch.allclose(latents[1], _latent(fill=3.0))


def test_empty_partition_yields_zero_unmasked_slots() -> None:
    """Spec step 8: "an empty partition yields zero unmasked store slots, which is the
    no-store configuration." A scope that has genuinely never written anything."""
    store = _store()
    latents, mask = store.read(derive_scope("nobody-wrote-yet"), domain="chat", b_store=8)
    assert latents.shape == (8, 0), "no write has ever set a width, so D is honestly 0"
    assert not mask.any()


def test_determinism_under_a_fixed_seed() -> None:
    """No RNG is used anywhere in this module -- ranking is `(-importance, -written_at,
    logical_key)`, a stable, seed-free order (ambiguity note 3) -- so two identically-seeded
    runs building identical stores must read back bitwise identical tensors."""

    def build_and_read() -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        store = _store()
        scope = derive_scope("alice")
        for i in range(4):
            store.write(scope, "chat", f"turn-{i}", torch.randn(6), importance=float(i))
        return store.read(scope, domain="chat", b_store=4)

    latents_a, mask_a = build_and_read()
    latents_b, mask_b = build_and_read()
    assert torch.equal(latents_a, latents_b)
    assert torch.equal(mask_a, mask_b)


# ---------------------------------------------------------------------------------------
# The tie-break: a resident set must not become a wall (measured defect, 2026-09-07).
# ---------------------------------------------------------------------------------------


def _wall_fixture() -> tuple[InMemoryStoreStub, Scope]:
    """Eight primers, then four later writes, all at the default (uniform) importance.

    This is the phase-A shape exactly: `cli.prime_store` writes `records` episodes before
    step 0 and the model then writes one record per turn, none of them overriding
    `importance_default`, so `importance` cannot separate any of them.
    """
    store = _store()
    scope = derive_scope("alice")
    for i in range(8):
        store.write(scope, "chat", f"primer-{i}", _latent(fill=float(i)))
    for i in range(4):
        store.write(scope, "chat", f"later-{i}", _latent(fill=100.0 + i))
    return store, scope


def test_a_read_reaches_records_written_after_the_resident_set() -> None:
    """With importance uniform, the newest writes must be readable, not queued behind primers.

    Measured on `main`: with the tie-break at `written_at` ASCENDING, `b_store = 8` over 8
    primers returned the 8 primers forever. An 8-record store and a 32-record store trained
    to bit-identical results to 17 significant figures, because nothing written after the
    primers could ever be read.
    """
    store, scope = _wall_fixture()
    latents, mask = store.read(scope, domain="chat", b_store=8)

    assert bool(mask.all()), "eight slots, twelve records: every slot must be filled"
    fills = [float(row[0]) for row in latents]
    assert sum(1 for f in fills if f >= 100.0) == 4, (
        f"the four post-primer writes are still unreachable: {fills}"
    )


def test_the_oldest_first_tie_break_is_what_walled_the_store_off() -> None:
    """The can-fail control: rank the SAME records by the old key and the wall comes back.

    `MEMORY.md`'s "verify guards by making them fail" applied to a fixed ordering bug -- the
    green test above only means something beside a red one built from the exact expression
    that was wrong.
    """
    store, scope = _wall_fixture()
    records = [r for r in store._records.values() if r.scope == scope]
    assert len(records) == 12

    old_order = sorted(records, key=lambda r: (-r.importance, r.written_at))[:8]
    assert all(r.logical_key.startswith("primer-") for r in old_order), (
        "the fixture no longer reproduces the uniform-importance tie"
    )

    new_order = sorted(records, key=lambda r: (-r.importance, -r.written_at))[:8]
    assert sum(1 for r in new_order if r.logical_key.startswith("later-")) == 4


def test_importance_still_outranks_recency() -> None:
    """Recency is the TIE-break, not the ranking: a deliberate importance still wins.

    Guards the other direction -- a repair that made the read purely recency-ordered would
    have thrown away DEC-63's importance axis rather than fixing its tie-break.
    """
    store = _store()
    scope = derive_scope("alice")
    store.write(scope, "chat", "old-but-important", _latent(fill=1.0), importance=0.9)
    store.write(scope, "chat", "new-but-dull", _latent(fill=2.0), importance=0.1)

    latents, _mask = store.read(scope, domain="chat", b_store=1)
    assert torch.allclose(latents[0], _latent(fill=1.0))


# ---------------------------------------------------------------------------------------
# G31-shaped boundary check: an episodic latent is never an integer-typed payload.
# ---------------------------------------------------------------------------------------


def test_write_refuses_an_integer_typed_latent() -> None:
    store = _store()
    with pytest.raises(ValueError, match="not floating point"):
        store.write(derive_scope("alice"), "chat", "k", torch.zeros(4, dtype=torch.long))


def test_write_refuses_a_latent_width_that_disagrees_with_the_stores_established_width() -> None:
    store = _store()
    scope = derive_scope("alice")
    store.write(scope, "chat", "first", _latent(dim=8))
    with pytest.raises(ValueError, match="disagrees with"):
        store.write(scope, "chat", "second", _latent(dim=16))


# ---------------------------------------------------------------------------------------
# G32 -- store scope (Table 8). Two failure shapes: no scope at all, and a request-supplied
# one. `write` refuses both; `read` refuses only the second and degrades on the first.
# ---------------------------------------------------------------------------------------


def test_g32_write_with_no_scope_is_refused() -> None:
    """ "No server-derived scope" -- an unknown principal. There is no legal partition to
    write an anonymous record into, so unlike `read`, this must raise."""
    store = _store()
    with pytest.raises(StoreScopeError, match="G32"):
        store.write(None, "chat", "k", _latent())


def test_g32_read_with_no_scope_degrades_to_an_empty_partition_rather_than_raising() -> None:
    """ "An unknown principal reads an empty partition" (Table 8's own phrase for this
    exact case) -- a request with no identity sees nothing, but the request does not crash."""
    store = _store()
    scope = derive_scope("alice")
    store.write(scope, "chat", "k", _latent())  # someone else's real data

    latents, mask = store.read(None, domain="chat", b_store=4)
    assert not mask.any(), "an unknown principal must never see alice's record"
    assert latents.shape == (4, 8)


def test_g32_a_request_supplying_its_own_scope_is_refused_on_write_and_read() -> None:
    """The other failure shape: a value that never went through `derive_scope` -- e.g. a
    handler that forwarded `request.json["scope"]` verbatim instead of deriving one. Table
    8's guard-can-fail case: "a request carrying its own scope (G32)"."""
    store = _store()
    scope = derive_scope("alice")
    store.write(scope, "chat", "k", _latent())
    forged = "alice"  # a bare string: the exact shape a naive handler would forward

    with pytest.raises(StoreScopeError, match="G32"):
        store.write(forged, "chat", "other", _latent())
    with pytest.raises(StoreScopeError, match="G32"):
        store.read(forged, domain="chat", b_store=4)


def test_g32_positive_control_a_genuinely_derived_scope_is_accepted() -> None:
    """A guard that refuses everything is as useless as one that refuses nothing."""
    store = _store()
    scope = derive_scope("alice", session="s1")
    receipt = store.write(scope, "chat", "k", _latent())
    assert isinstance(receipt, WriteReceipt)
    assert receipt.committed is True
    latents, mask = store.read(scope, domain="chat", b_store=4)
    assert mask.sum().item() == 1


def test_a_mis_derived_scope_key_produces_a_crossing_in_the_negative_test() -> None:
    """Table 9: "a mis-derived scope key produces a crossing in the negative test." This
    proves the isolation assertion above has teeth -- that it is actually checking scope,
    not passing by accident -- by replaying the same two writes through a DELIBERATELY
    broken key function that drops `scope` from the partition tuple, and showing that
    broken function DOES leak bob's record into alice's read. The real store (asserted
    directly below) must not.
    """
    alice, bob = derive_scope("alice"), derive_scope("bob")

    def broken_key(scope: Scope, domain: str, logical_key: str) -> tuple[str, str]:
        del scope  # the bug under test: scope is dropped from the partition key
        return (domain, logical_key)

    broken_records: dict[tuple[str, str], str] = {}
    broken_records[broken_key(alice, "chat", "shared-key")] = "alice's secret"
    broken_records[broken_key(bob, "chat", "shared-key")] = "bob's secret"
    # Same logical_key, same domain, different scopes -- the broken key collides them.
    assert broken_records[broken_key(alice, "chat", "shared-key")] == "bob's secret", (
        "the broken key must actually demonstrate a cross-scope collision, or this test "
        "is not exercising the failure it exists to catch"
    )

    # The real store, given the identical inputs, must not exhibit that collision.
    store = _store()
    store.write(alice, "chat", "shared-key", _latent(fill=1.0))
    store.write(bob, "chat", "shared-key", _latent(fill=2.0))
    latents, mask = store.read(alice, domain="chat", b_store=1)
    assert mask[0]
    assert torch.allclose(latents[0], _latent(fill=1.0)), "alice's read crossed into bob's write"


# ---------------------------------------------------------------------------------------
# E0 fixture 1/9 -- domain isolation (ported from
# tests/storage/test_store_conformance.py:150, `test_domain_isolation_same_logical_key`).
# Genuinely implementable without any DEC-63..66 machinery: PASSES.
# ---------------------------------------------------------------------------------------


def test_e0_domain_isolation_same_logical_key() -> None:
    """The same logical key in two domains must yield two distinct records and never
    cross-leak on read."""
    store = _store()
    scope = derive_scope("alice")
    store.write(scope, "chat", "shared-key", _latent(fill=1.0))
    store.write(scope, "agent", "shared-key", _latent(fill=2.0))

    chat_latents, chat_mask = store.read(scope, domain="chat", b_store=1)
    agent_latents, agent_mask = store.read(scope, domain="agent", b_store=1)
    assert chat_mask[0] and agent_mask[0]
    assert torch.allclose(chat_latents[0], _latent(fill=1.0))
    assert torch.allclose(agent_latents[0], _latent(fill=2.0))


# ---------------------------------------------------------------------------------------
# E0 fixture 2/9 -- last-write-wins
# (`test_double_put_same_identity_last_write_wins`). PASSES.
# ---------------------------------------------------------------------------------------


def test_e0_double_put_same_identity_last_write_wins() -> None:
    store = _store()
    scope = derive_scope("alice")
    store.write(scope, "chat", "k", _latent(fill=1.0))
    store.write(scope, "chat", "k", _latent(fill=2.0))

    latents, mask = store.read(scope, domain="chat", b_store=2)
    assert mask.sum().item() == 1, "a double put at one identity must not become two records"
    assert torch.allclose(latents[0], _latent(fill=2.0))


# ---------------------------------------------------------------------------------------
# E0 fixture 3/9 -- query requires domain or the global flag
# (`test_query_requires_domain_or_global_flag`). PASSES.
# ---------------------------------------------------------------------------------------


def test_e0_query_requires_domain_or_global_flag() -> None:
    store = _store()
    scope = derive_scope("alice")
    store.write(scope, "chat", "k", _latent())

    with pytest.raises(ValueError, match="query_requires_domain_or_global_flag"):
        store.read(scope, b_store=4)  # domain=None, global_query=False (the default)

    # The escape hatch works and still respects scope.
    latents, mask = store.read(scope, global_query=True, b_store=4)
    assert mask.sum().item() == 1
    assert torch.allclose(latents[0], _latent())


# ---------------------------------------------------------------------------------------
# E0 fixture 4/9 -- caller cannot mutate the store through a read result. PASSES.
# ---------------------------------------------------------------------------------------


def test_e0_caller_cannot_mutate() -> None:
    store = _store()
    scope = derive_scope("alice")
    store.write(scope, "chat", "k", _latent(fill=1.0))

    latents, _mask = store.read(scope, domain="chat", b_store=1)
    latents[0].fill_(999.0)  # mutate the caller's copy

    latents_again, _mask = store.read(scope, domain="chat", b_store=1)
    assert torch.allclose(latents_again[0], _latent(fill=1.0)), (
        "mutating a read result corrupted the store's own record"
    )


# ---------------------------------------------------------------------------------------
# E0 fixtures 5-9/9 -- the five eviction/tiering behaviours TAX:683 names to port. These
# need byte capacity, staleness-scored eviction, and GPU/RAM/disk tiering -- none of which
# exist in a flat in-memory dict. Each FAILS RED here: not a silent no-op, but a raised,
# named `ContractGap` (module docstring, ambiguity note 1, for the DEC-63/DEC-65 split).
# ---------------------------------------------------------------------------------------


def test_e0_admit_spills_lowest_importance_fails_red_on_dec_63() -> None:
    store = _store()
    scope = derive_scope("alice")
    store.write(scope, "chat", "low", _latent(), importance=0.1)
    store.write(scope, "chat", "high", _latent(), importance=0.9)

    with pytest.raises(ContractGap) as exc_info:
        store.evict()
    assert exc_info.value.dec == "DEC-63"


def test_e0_gpu_hint_protects_from_spill_fails_red_on_dec_63() -> None:
    store = _store()
    with pytest.raises(ContractGap) as exc_info:
        store.evict()
    assert exc_info.value.dec == "DEC-63"


def test_e0_disk_prune_drops_lowest_fails_red_on_dec_63() -> None:
    store = _store()
    with pytest.raises(ContractGap) as exc_info:
        store.evict()
    assert exc_info.value.dec == "DEC-63"


def test_e0_promote_returns_span_to_ram_fails_red_on_dec_65() -> None:
    store = _store()
    scope = derive_scope("alice")
    store.write(scope, "chat", "k", _latent())

    with pytest.raises(ContractGap) as exc_info:
        store.promote(scope, "chat", "k")
    assert exc_info.value.dec == "DEC-65"


def test_e0_retrieve_merges_tiers_fails_red_on_dec_65() -> None:
    """The stub has one flat tier, so there is nothing to merge across -- the same missing
    tiering system `promote` names (module docstring, ambiguity note 1)."""
    store = _store()
    scope = derive_scope("alice")
    with pytest.raises(ContractGap) as exc_info:
        store.promote(scope, "chat", "k")
    assert exc_info.value.dec == "DEC-65"


# ---------------------------------------------------------------------------------------
# Every ContractGap names its DEC (Table 9's third clause, checked directly against every
# gap method this module ships, not only the five E0 fixtures above).
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "expected_dec"),
    [
        (lambda s: s.capacity_bytes("5080"), "DEC-63"),
        (lambda s: s.evict(), "DEC-63"),
        (lambda s: s.promote(derive_scope("alice"), "chat", "k"), "DEC-65"),
        (lambda s: s.bind_session(derive_scope("alice")), "DEC-64"),
        (lambda s: s.consolidate(), "DEC-66"),
    ],
)
def test_every_gap_method_raises_contract_gap_naming_its_dec(call, expected_dec: str) -> None:
    store = _store()
    with pytest.raises(ContractGap) as exc_info:
        call(store)
    assert exc_info.value.dec == expected_dec
    assert expected_dec in str(exc_info.value)


def test_domain_outside_the_enum_is_refused_on_write_and_read() -> None:
    """Not a DEC gap -- ordinary schema validation against `domain_enum` (DEC-64's base
    axis, adopted and implemented, module docstring)."""
    store = _store()
    scope = derive_scope("alice")
    with pytest.raises(ValueError, match="domain_enum"):
        store.write(scope, "not-a-real-domain", "k", _latent())
    with pytest.raises(ValueError, match="domain_enum"):
        store.read(scope, domain="not-a-real-domain", b_store=4)
