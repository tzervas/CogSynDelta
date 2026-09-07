"""Row E1 gate (iv): eviction order under a constructed overflow, with known scores.

THE GATE, verbatim: *"EVICTION ORDER UNDER A CONSTRUCTED OVERFLOW: build a working set whose
scores are known and whose total exceeds capacity, then assert the survivors are exactly the
top-scored set AND that a GPU-resident low-importance entry outlives a host-resident
higher-importance one by exactly the `+1.0` bonus -- the behaviour `gpu_hint_protects_from_spill`
pins upstream."*

WHY THE TIMESTAMPS ARE PINNED RATHER THAN LEFT TO THE CLOCK. The score is
`importance + gpu_resident_bonus - staleness(last_accessed)`, so two records written a
microsecond apart do NOT have equal scores -- the later one is a hair less stale. That is
correct behaviour, and it makes "exactly the +1.0 bonus" and "ties break by older timestamp"
untestable against a free-running clock, because there are then no exact ties and no exact
boundary. Every test here writes its working set and then pins `written_at` and
`last_accessed` on the index directly, relative to one instant captured when the fixture was
built, so the constructed scores ARE the scores and a boundary assertion is about the policy
rather than about scheduling noise.

WHAT "SURVIVORS" MEANS. The resident set -- GPU- and RAM-tagged records. A spilled record is
still readable (`retrieve_merges_tiers`); what it has lost is its claim on the byte capacity,
which is what section 8 gap (a)'s number bounds.
"""

from __future__ import annotations

import time

import pytest
import torch

from cogsyndelta.interconnect.episodic import (
    GPU_RESIDENT_BONUS,
    EpisodicStoreImpl,
    InMemoryBackend,
    Residency,
    TierBudget,
    partition_scope_key,
    staleness_penalty,
)
from cogsyndelta.interconnect.episodic_store import derive_scope

pytestmark = pytest.mark.cpu

DOMAINS = frozenset({"chat"})
SPAN = 1000
BIG = 1 << 40


class _Fixture:
    """A store with a mutable capacity and pinned record timestamps."""

    def __init__(
        self,
        *,
        half_life_s: float = 3600.0,
        capacity: int = BIG,
        ram_max_items: int = 10_000,
        max_records: int = 10_000,
    ) -> None:
        self.capacity = {"value": capacity}
        self.half_life_s = half_life_s
        self.now = time.time()
        self.store = EpisodicStoreImpl(
            DOMAINS,
            backend=InMemoryBackend(),
            capacity_provider=lambda: self.capacity["value"],
            host="test-host",
            half_life_s=half_life_s,
            tier_budget=TierBudget(ram_max_items=ram_max_items, max_records=max_records),
        )
        self.store.start()
        self.scope = derive_scope("alice")
        self.scope_key = partition_scope_key(self.scope)

    def write(
        self,
        logical_key: str,
        importance: float,
        *,
        gpu: bool = False,
        age_s: float = 0.0,
    ) -> None:
        """Admit one span and pin its timestamps `age_s` seconds before the fixture's instant."""
        self.store.learn(
            self.scope,
            "chat",
            logical_key,
            torch.ones(8),
            importance=importance,
            span_bytes=SPAN,
        )
        if gpu:
            self.store.mark_gpu(self.scope, "chat", logical_key)
        self.pin(logical_key, age_s)

    def pin(self, logical_key: str, age_s: float) -> None:
        """Set one record's timestamps to `age_s` seconds before the fixture's instant."""
        meta = self.store._index[(self.scope_key, "chat", logical_key)]
        meta.written_at = self.now - age_s
        meta.last_accessed = self.now - age_s

    def survivors(self) -> set[str]:
        """Logical keys still holding a claim on the byte capacity."""
        return {key[2] for key in self.store.resident_keys()}

    def members(self) -> set[str]:
        """Logical keys the store still HOLDS, at any residency -- membership, not placement."""
        return {key[2] for key in self.store._index}

    def evict_at(self, capacity: int) -> int:
        """Set the capacity for this tick and run one eviction pass."""
        self.capacity["value"] = capacity
        return self.store.evict()


def test_survivors_are_exactly_the_top_scored_set() -> None:
    """Ten known scores, capacity for four: the four highest survive and no others."""
    fixture = _Fixture()
    importances = [0.05, 0.95, 0.35, 0.75, 0.15, 0.85, 0.45, 0.65, 0.25, 0.55]
    for i, importance in enumerate(importances):
        fixture.write(f"k{i}", importance)

    ranked = sorted(enumerate(importances), key=lambda pair: -pair[1])
    expected = {f"k{i}" for i, _ in ranked[:4]}
    evicted = fixture.evict_at(4 * SPAN)

    assert fixture.survivors() == expected, (
        f"survivors {sorted(fixture.survivors())} are not the top-scored four {sorted(expected)}"
    )
    assert evicted == 6
    assert fixture.store.resident_bytes == 4 * SPAN


def test_a_gpu_resident_low_importance_entry_outlives_a_higher_host_resident_one() -> None:
    """The GPU bonus is worth exactly `+1.0` of importance, checked on both sides of the line."""
    epsilon = 0.05
    base = 0.10

    below = _Fixture()
    below.write("gpu-low", base, gpu=True)
    below.write("host-high", base + GPU_RESIDENT_BONUS - epsilon)
    below.evict_at(SPAN)
    assert below.survivors() == {"gpu-low"}, (
        "a host-resident record less than the bonus above the GPU record must lose"
    )

    above = _Fixture()
    above.write("gpu-low", base, gpu=True)
    above.write("host-high", base + GPU_RESIDENT_BONUS + epsilon)
    above.evict_at(SPAN)
    assert above.survivors() == {"host-high"}, (
        "a host-resident record more than the bonus above the GPU record must win -- a bonus "
        "that protected unconditionally would not be worth 1.0, it would be worth infinity"
    )


def test_the_bonus_is_worth_exactly_one_point_zero_at_the_boundary() -> None:
    """At exactly `+1.0` the residency term has bought one unit of importance and nothing more.

    This is what makes "by exactly the `+1.0` bonus" a measurement rather than a direction: at
    the boundary the two records differ only in staleness, so the outcome is decided by the
    documented tie-break (older loses), not by residency.
    """
    fixture = _Fixture()
    fixture.write("gpu-low", 0.10, gpu=True, age_s=10.0)
    fixture.write("host-high", 0.10 + GPU_RESIDENT_BONUS, age_s=0.0)

    gpu_key = (fixture.scope_key, "chat", "gpu-low")
    host_key = (fixture.scope_key, "chat", "host-high")
    now = fixture.now
    gpu_score = fixture.store.score(gpu_key, now=now)
    host_score = fixture.store.score(host_key, now=now)
    without_staleness_gap = gpu_score + staleness_penalty(now - 10.0, now, fixture.half_life_s)
    assert without_staleness_gap == pytest.approx(0.10 + GPU_RESIDENT_BONUS, abs=1e-12), (
        "the GPU record's score is not importance + exactly the bonus"
    )
    assert gpu_score < host_score, "the older of two equal-importance-plus-bonus records must lose"

    fixture.evict_at(SPAN)
    assert fixture.survivors() == {"host-high"}


def test_ties_break_by_older_timestamp_then_by_key() -> None:
    """Equal score and equal timestamp: the lexicographically smaller key spills first."""
    fixture = _Fixture()
    fixture.write("bbb", 0.5)
    fixture.write("aaa", 0.5)
    fixture.pin("aaa", 0.0)
    fixture.pin("bbb", 0.0)

    fixture.evict_at(SPAN)
    assert fixture.survivors() == {"bbb"}, (
        "with score and timestamp tied, the smaller key must spill first"
    )


def test_older_timestamp_loses_before_the_key_is_consulted() -> None:
    """The first tie-break is the timestamp: the older record spills even with the larger key."""
    fixture = _Fixture()
    fixture.write("aaa", 0.5, age_s=0.0)
    fixture.write("zzz", 0.5, age_s=0.0)
    # Same score; make `zzz` the older WRITE while leaving last_accessed equal, so staleness
    # cannot decide and only the written_at tie-break can.
    fixture.store._index[(fixture.scope_key, "chat", "zzz")].written_at = fixture.now - 100.0

    fixture.evict_at(SPAN)
    assert fixture.survivors() == {"aaa"}


def test_the_staleness_term_moves_the_order() -> None:
    """A stale high-importance record loses to a fresh lower-importance one, as designed.

    Section 8 gap (a) extended asks for the decay function to be exercised rather than merely
    configured. With a one-second half-life and a ten-second age the stale record has accrued
    a penalty of `1 - 2**-10`, which is more than enough to cost it 0.4 of importance.
    """
    fixture = _Fixture(half_life_s=1.0)
    fixture.write("stale-important", 0.9, age_s=10.0)
    fixture.write("fresh-modest", 0.5, age_s=0.0)

    penalty = staleness_penalty(fixture.now - 10.0, fixture.now, 1.0)
    assert 0.9 - penalty < 0.5, "the constructed working set does not actually exercise decay"

    fixture.evict_at(SPAN)
    assert fixture.survivors() == {"fresh-modest"}


def test_an_overflowing_write_evicts_rather_than_refusing() -> None:
    """Gate (ii)'s phrasing: an over-capacity write triggers EVICTION, not a refusal.

    A store that refused the write would also keep `resident_bytes <= capacity` and would be a
    different contract -- durable-first ordering says the write commits and something else
    leaves.
    """
    fixture = _Fixture()
    fixture.capacity["value"] = 2 * SPAN
    for i in range(5):
        fixture.store.learn(
            fixture.scope,
            "chat",
            f"k{i}",
            torch.ones(8),
            importance=float(i),
            span_bytes=SPAN,
        )
    assert fixture.store.resident_bytes <= 2 * SPAN
    assert fixture.survivors() == {"k3", "k4"}
    _latents, mask = fixture.store.retrieve(fixture.scope, domain="chat", b_store=8)
    assert mask.sum().item() == 5, "spilled records are still readable; nothing was refused"


def test_spilled_records_carry_the_disk_tag_rather_than_disappearing() -> None:
    """Eviction from the resident set is a residency change, not a delete."""
    fixture = _Fixture()
    fixture.write("keep", 0.9)
    fixture.write("spill", 0.1)
    fixture.evict_at(SPAN)
    meta = fixture.store._index[(fixture.scope_key, "chat", "spill")]
    assert meta.residency is Residency.DISK


# ---------------------------------------------------------------------------------------
# Membership is host-invariant (DEC-63 as amended 2026-09-07).
#
# THE DEFECT. `_enforce_capacity` always ended in `_prune_disk()`, and the spill rate that fed
# it follows `capacity_bytes`, which follows live VRAM. A smaller card spilled more, reached
# the fixed `disk_max_items` at a smaller working set, and HARD-DELETED rows a larger card
# still held: the same input sequence produced a different store depending on which card ran
# it, through the one path the store documents as its only data-loss path. The deletion SCORE
# was already clean -- importance only, GPU bonus deliberately excluded. The TRIGGER was not.
# Spilling is placement and may depend on the host; deletion is membership and must not.
# ---------------------------------------------------------------------------------------

MEMBERSHIP_CEILING = 6
"""Declared record ceiling for the two-card arms below -- small enough to bite at ten writes."""


def _ten_writes(*, capacity: int) -> _Fixture:
    """Ten records, distinct importances, one declared ceiling, one caller-chosen capacity.

    `capacity` is the ONLY difference between the two arms: it stands for the card. Everything
    the store is told about what it should HOLD is identical.
    """
    fixture = _Fixture(capacity=capacity, ram_max_items=10_000, max_records=MEMBERSHIP_CEILING)
    for i in range(10):
        fixture.write(f"k{i}", importance=i / 10.0)
    return fixture


def test_membership_is_identical_on_a_big_card_and_a_small_one() -> None:
    """Same writes, wildly different capacities: the store HOLDS the same records.

    The two arms must genuinely diverge in placement for this to mean anything, so the second
    assertion checks that they did -- otherwise a store that simply never spills would pass.
    """
    big = _ten_writes(capacity=BIG)
    small = _ten_writes(capacity=2 * SPAN)

    expected = {f"k{i}" for i in range(4, 10)}
    assert big.members() == expected, f"big card holds {big.members()}"
    assert small.members() == expected, f"small card holds {small.members()}"
    assert big.survivors() != small.survivors(), (
        "the two arms placed records identically, so this test never exercised the divergence "
        "it exists to bound"
    )


def _old_trigger_victims(fixture: _Fixture) -> set[str]:
    """The PRE-AMENDMENT deletion set: overflow of the SPILLED tier against the ceiling.

    Reconstructed here rather than described, so the can-fail control below is the expression
    that was wrong, applied to the two arms' real, divergent residency.
    """
    disk = [(k, m) for k, m in fixture.store._index.items() if m.residency is Residency.DISK]
    overflow = len(disk) - MEMBERSHIP_CEILING
    if overflow <= 0:
        return set()
    order = sorted(disk, key=lambda km: (km[1].importance, km[1].written_at, km[0]))
    return {key[2] for key, _meta in order[:overflow]}


def test_the_old_spill_fed_trigger_deleted_different_rows_on_different_cards(monkeypatch) -> None:
    """The can-fail control: the trigger that was replaced, on the same two arms.

    With membership enforcement stubbed out, both arms hold all ten records and differ only in
    residency. Applying the old expression to each then deletes DIFFERENT rows -- nothing on
    the big card, real rows on the small one. That difference is the defect, and it is what
    the test above bounds.
    """
    monkeypatch.setattr(EpisodicStoreImpl, "_enforce_membership", lambda self: 0)
    big = _ten_writes(capacity=BIG)
    small = _ten_writes(capacity=2 * SPAN)
    assert big.members() == small.members(), "the stub did not disable deletion"

    big_victims = _old_trigger_victims(big)
    small_victims = _old_trigger_victims(small)
    assert big_victims == set(), "the big card should have spilled too little to trigger a prune"
    assert small_victims, "the small card should have spilled past the ceiling"
    assert big_victims != small_victims, (
        "the old trigger did not diverge across cards here, so it is not reconstructed"
    )


def test_a_spill_deletes_nothing() -> None:
    """`_enforce_placement` demotes and never deletes -- the two paths are unhooked."""
    fixture = _Fixture(capacity=BIG, max_records=10_000)
    for i in range(10):
        fixture.write(f"k{i}", importance=i / 10.0)
    before = fixture.members()

    fixture.evict_at(2 * SPAN)
    assert fixture.members() == before, "a spill removed a record from the store"
    assert len(fixture.survivors()) < len(before), "nothing spilled; the arm proves nothing"


def test_demotion_under_capacity_pressure_still_happens() -> None:
    """DEC-70's positive control: the fix must not have removed demotion.

    Demotion costs latency, not membership, and it is the correct response to VRAM pressure.
    A repair that made the store host-invariant by refusing to spill at all would pass every
    membership assertion above and be a worse store.
    """
    fixture = _Fixture(capacity=BIG, max_records=10_000)
    for i in range(10):
        fixture.write(f"k{i}", importance=i / 10.0)
    assert len(fixture.survivors()) == 10

    fixture.evict_at(3 * SPAN)
    assert len(fixture.survivors()) == 3, "the store did not demote under capacity pressure"
    assert fixture.survivors() == {"k7", "k8", "k9"}, "demotion kept the wrong three"


def test_the_membership_budget_is_counted_in_records_not_bytes() -> None:
    """A narrower latent must not change WHAT the store holds.

    Quantisation makes records smaller. Under a byte ceiling that silently admits more of them,
    what the store answers would change out of a step that was supposed to preserve behaviour
    -- and re-quantisation is already a memory-gate migration trigger. Two arms, same records,
    one at a quarter the span: the membership must be identical.
    """
    wide = _Fixture(capacity=BIG, max_records=MEMBERSHIP_CEILING)
    narrow = _Fixture(capacity=BIG, max_records=MEMBERSHIP_CEILING)
    for i in range(10):
        wide.store.learn(
            wide.scope, "chat", f"k{i}", torch.ones(8), importance=i / 10.0, span_bytes=SPAN
        )
        narrow.store.learn(
            narrow.scope,
            "chat",
            f"k{i}",
            torch.ones(2),
            importance=i / 10.0,
            span_bytes=SPAN // 4,
        )

    assert wide.members() == narrow.members() == {f"k{i}" for i in range(4, 10)}
