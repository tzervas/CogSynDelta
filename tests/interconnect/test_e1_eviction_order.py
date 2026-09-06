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

    def __init__(self, *, half_life_s: float = 3600.0) -> None:
        self.capacity = {"value": BIG}
        self.half_life_s = half_life_s
        self.now = time.time()
        self.store = EpisodicStoreImpl(
            DOMAINS,
            backend=InMemoryBackend(),
            capacity_provider=lambda: self.capacity["value"],
            host="test-host",
            half_life_s=half_life_s,
            tier_budget=TierBudget(ram_max_items=10_000, disk_max_items=10_000),
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
