"""Row E1: proof that the three constructed gates FIRE when the thing they guard is broken.

The row asks for *"four, three of them constructed to fire."* A green gate is evidence only if
it could have been red, and the repo already keeps that discipline in
`tests/test_guards_can_fail.py` after three CSD guards turned out to be structurally incapable
of firing. This file is that pass for E1: for each of gates (ii), (iii) and (iv), break exactly
the property the gate exists to protect and assert the gate's own assertion then fails.

Each break is a MINIMAL, NAMED defect -- the bonus set to zero, the eviction pass made a no-op,
the score made constant, the scope segment dropped -- not a scrambled implementation, so what
is demonstrated is that the gate is sensitive to THAT property rather than to any change at
all. Gate (iii)'s negative test lives beside its fuzz (a crossing has to be produced by the
same 10,000-write workload to mean anything), and is re-asserted here in miniature.

Gate (i) is the one gate not constructed to fire, and cannot be: its subject is a property of
the DIFF -- that no test file was edited to make E0's fixtures green -- which `git diff`
answers and no test can. That is stated rather than papered over with a test that would only
be checking itself.
"""

from __future__ import annotations

import pytest
import torch

import cogsyndelta.interconnect.episodic.store as store_module
from cogsyndelta.interconnect.episodic import (
    GIB,
    GPU_RESIDENT_BONUS,
    EpisodicStoreImpl,
    HostVram,
    InMemoryBackend,
    SchedulerBudgets,
    TierBudget,
    capacity_decision,
)
from cogsyndelta.interconnect.episodic.capacity import MIB
from cogsyndelta.interconnect.episodic_store import derive_scope

pytestmark = pytest.mark.cpu

DOMAINS = frozenset({"chat"})
SPAN = 1000
BIG = 1 << 40


def _store(capacity: dict) -> EpisodicStoreImpl:
    store = EpisodicStoreImpl(
        DOMAINS,
        backend=InMemoryBackend(),
        capacity_provider=lambda: capacity["value"],
        host="test-host",
        tier_budget=TierBudget(ram_max_items=10_000, max_records=10_000),
    )
    store.start()
    return store


# ---------------------------------------------------------------------------------------
# Gate (ii) fires: the capacity gate catches a zero, a constant, and an unenforced bound.
# ---------------------------------------------------------------------------------------


def test_gate_ii_fires_when_the_floor_branch_is_removed() -> None:
    """Without the pre-committed floor, the 5080's long-context tick yields a ZERO capacity.

    Exactly section 9.11's case, and exactly the outcome the gate calls a failure: *"a probe
    returning two DIFFERENT numbers where one is zero has also failed."* The broken formula
    here is the one the design explicitly rejected -- the bare residual, floored at zero, with
    no alternative branch.
    """
    vram = HostVram(host="gpu5080", device_name="RTX 5080", total_bytes=16_303 * MIB)
    budgets = SchedulerBudgets.from_context(2_000_000, ("language", "reason", "retrieve", "visual"))

    broken = max(
        0,
        vram.total_bytes - budgets.kv_reserved_bytes - budgets.activation_reserve_bytes - 2 * GIB,
    )
    assert broken == 0, "this tick was supposed to starve the residual on the 5080"

    healthy = capacity_decision(vram, budgets)
    assert healthy.capacity_bytes > 0, "the floor branch did not save the deployment card"
    assert healthy.branch.value == "floor"


def test_gate_ii_fires_when_the_capacity_ignores_the_active_region_set() -> None:
    """A per-host constant passes "the two cards differ" and still fails the dynamism clause."""
    vram = HostVram(host="h", device_name="fake", total_bytes=24_000 * MIB)
    four = SchedulerBudgets(kv_reserved_bytes=1 * GIB, activation_reserve_bytes=1 * GIB)
    five_but_broken = SchedulerBudgets(kv_reserved_bytes=1 * GIB, activation_reserve_bytes=1 * GIB)

    assert (
        capacity_decision(vram, four).capacity_bytes
        == capacity_decision(vram, five_but_broken).capacity_bytes
    ), "a KV reserve that ignores regions_active must produce the constant the gate rejects"

    real_four = SchedulerBudgets.from_context(8192, ("a", "b", "c", "d"))
    real_five = SchedulerBudgets.from_context(8192, ("a", "b", "c", "d", "e"))
    assert (
        capacity_decision(vram, real_four).capacity_bytes
        != capacity_decision(vram, real_five).capacity_bytes
    )


def test_gate_ii_fires_when_the_capacity_is_not_enforced_on_admission(monkeypatch) -> None:
    """With the eviction pass made a no-op, the store blows through its own capacity."""
    capacity = {"value": 2 * SPAN}
    scope = derive_scope("alice")

    monkeypatch.setattr(EpisodicStoreImpl, "_enforce_placement", lambda self, now: 0)
    broken = _store(capacity)
    for i in range(5):
        broken.learn(scope, "chat", f"k{i}", torch.ones(8), importance=float(i), span_bytes=SPAN)
    assert broken.resident_bytes > capacity["value"], (
        "the unenforced store stayed under capacity by accident; the gate below proves nothing"
    )

    monkeypatch.undo()
    healthy = _store({"value": 2 * SPAN})
    for i in range(5):
        healthy.learn(scope, "chat", f"k{i}", torch.ones(8), importance=float(i), span_bytes=SPAN)
    assert healthy.resident_bytes <= 2 * SPAN


# ---------------------------------------------------------------------------------------
# Gate (iii) fires: the isolation check catches a dropped scope segment.
# ---------------------------------------------------------------------------------------


def test_gate_iii_fires_when_the_scope_segment_is_dropped_from_the_key(monkeypatch) -> None:
    """The miniature of the fuzz's negative arm: two principals, one shared key, one crossing.

    The full-scale version (10,000 writes, 120 partitions) is
    `test_e1_cross_request_fuzz.py::test_the_fuzz_catches_a_crossing_when_the_key_is_mis_derived`;
    this is the same defect at a size a reader can hold in their head.
    """
    alice, bob = derive_scope("alice"), derive_scope("bob")

    monkeypatch.setattr(store_module, "partition_scope_key", lambda scope: "SHARED")
    broken = _store({"value": BIG})
    broken.learn(alice, "chat", "k", torch.full((8,), 1.0))
    broken.learn(bob, "chat", "k", torch.full((8,), 2.0))
    values, mask = broken.retrieve(alice, domain="chat", b_store=4)
    assert mask[0] and float(values[0][0].item()) == 2.0, (
        "the dropped scope segment did not produce a crossing, so the fuzz is not shown to be "
        "capable of catching one"
    )

    monkeypatch.undo()
    healthy = _store({"value": BIG})
    healthy.learn(alice, "chat", "k", torch.full((8,), 1.0))
    healthy.learn(bob, "chat", "k", torch.full((8,), 2.0))
    values, mask = healthy.retrieve(alice, domain="chat", b_store=4)
    assert mask.sum().item() == 1
    assert float(values[0][0].item()) == 1.0


# ---------------------------------------------------------------------------------------
# Gate (iv) fires: the eviction-order gate catches a missing bonus and a missing score.
# ---------------------------------------------------------------------------------------


def test_gate_iv_fires_when_the_gpu_residency_bonus_is_removed(monkeypatch) -> None:
    """With the bonus at zero, the GPU-tagged low-importance span is the one that spills."""
    scope = derive_scope("alice")

    monkeypatch.setattr(store_module, "GPU_RESIDENT_BONUS", 0.0)
    broken = _store({"value": SPAN})
    broken.learn(scope, "chat", "gpu-low", torch.ones(8), importance=0.10, span_bytes=SPAN)
    broken.mark_gpu(scope, "chat", "gpu-low")
    broken.learn(
        scope,
        "chat",
        "host-high",
        torch.ones(8),
        importance=0.10 + GPU_RESIDENT_BONUS - 0.05,
        span_bytes=SPAN,
    )
    assert {key[2] for key in broken.resident_keys()} == {"host-high"}, (
        "removing the bonus did not change the outcome, so gate (iv)'s bonus assertion is not "
        "actually measuring the bonus"
    )

    monkeypatch.undo()
    healthy = _store({"value": SPAN})
    healthy.learn(scope, "chat", "gpu-low", torch.ones(8), importance=0.10, span_bytes=SPAN)
    healthy.mark_gpu(scope, "chat", "gpu-low")
    healthy.learn(
        scope,
        "chat",
        "host-high",
        torch.ones(8),
        importance=0.10 + GPU_RESIDENT_BONUS - 0.05,
        span_bytes=SPAN,
    )
    assert {key[2] for key in healthy.resident_keys()} == {"gpu-low"}


def test_gate_iv_fires_when_eviction_stops_scoring(monkeypatch) -> None:
    """A constant score degrades eviction to write order, and the survivors are then wrong."""
    scope = derive_scope("alice")
    importances = [0.05, 0.95, 0.35, 0.75]
    expected = {"k1", "k3"}  # the two highest importances

    monkeypatch.setattr(EpisodicStoreImpl, "_score_meta", lambda self, meta, now: 0.0)
    broken = _store({"value": 2 * SPAN})
    for i, importance in enumerate(importances):
        broken.learn(scope, "chat", f"k{i}", torch.ones(8), importance=importance, span_bytes=SPAN)
    assert {key[2] for key in broken.resident_keys()} != expected, (
        "an unscored eviction happened to keep the top-scored set; gate (iv) would not have "
        "noticed the difference"
    )

    monkeypatch.undo()
    healthy = _store({"value": 2 * SPAN})
    for i, importance in enumerate(importances):
        healthy.learn(scope, "chat", f"k{i}", torch.ones(8), importance=importance, span_bytes=SPAN)
    assert {key[2] for key in healthy.resident_keys()} == expected
