"""Row E1 gate (ii): the dynamic-capacity formula, its floor branch, and the two-card receipts.

THE GATE, verbatim: *"compute capacity from a live `nvidia-smi` plus the scheduler's current
KV and activation budgets on the 3090 Ti (24 GiB) and the 5080 (16 GiB), assert the two
DIFFER, assert each is `> 0`, assert each is RESPECTED -- a write that would exceed it triggers
eviction rather than an allocation -- and assert the value CHANGES when the active-region set
changes... A probe returning one number for both cards has measured a constant and failed; a
probe returning two DIFFERENT numbers where one is zero has also failed."*

HOW THE TWO HALVES MEET. No host can see both cards, and CI's own runners have no GPU at all
(`scripts/ci_local.sh` exports `CUDA_VISIBLE_DEVICES=""` precisely so that local and CI runs
agree). So the live half runs on each card via `scripts/run_store_capacity_probe.py` and
commits a receipt; the cross-card assertions the gate names are made HERE, over the two
committed receipts, so they run in every CI job on every push rather than only on a GPU host.
The 5080's half is additionally re-run on every PR by the GPU CI runner
(`.github/workflows/gpu-5080.yml`), and `test_the_5080_probe_is_wired_to_the_gpu_runner`
asserts that job exists, carries the 5080 labels, and cannot pass by absence.

WHAT THE RECEIPTS SAY, at the revision this row landed:

  3090 Ti  20,858,273,792 B (19,891 MiB)  branch: residual
  RTX 5080 13,806,600,192 B (13,167 MiB)  branch: residual

and, on the long-context arm that section 9.11 warns about, the 5080 residual DOES round to
zero and the pre-committed floor fires while the 3090 Ti stays on the residual. That contrast
is the evidence that the floor is wired rather than written down.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from cogsyndelta.interconnect.episodic import (
    DEFAULT_STORE_FLOOR_BYTES,
    GIB,
    CapacityBranch,
    EpisodicStoreImpl,
    HostVram,
    InMemoryBackend,
    SchedulerBudgets,
    capacity_decision,
)
from cogsyndelta.interconnect.episodic.capacity import MIB, kv_reserved_bytes
from cogsyndelta.interconnect.episodic_store import derive_scope

pytestmark = pytest.mark.cpu

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = REPO_ROOT / "docs/design/evidence/e1-episodic-store-2026-09-06"
RECEIPTS = {
    "3090ti": EVIDENCE / "capacity-probe-3090ti.json",
    "5080": EVIDENCE / "capacity-probe-5080.json",
}
GPU_WORKFLOW = REPO_ROOT / ".github/workflows/gpu-5080.yml"
REGIONS = ("language", "reason", "retrieve", "visual")


def _vram(total_mib: int, host: str = "test-host") -> HostVram:
    return HostVram(host=host, device_name=f"fake-{total_mib}MiB", total_bytes=total_mib * MIB)


# ---------------------------------------------------------------------------------------
# The formula.
# ---------------------------------------------------------------------------------------


def test_the_formula_is_the_documented_subtraction() -> None:
    """`max(0, VRAM_total - KV_reserved - activation_reserve - safety_margin)`, exactly."""
    vram = _vram(24_000)
    budgets = SchedulerBudgets(kv_reserved_bytes=3 * GIB, activation_reserve_bytes=1 * GIB)
    decision = capacity_decision(vram, budgets, safety_margin_bytes=2 * GIB)
    assert decision.capacity_bytes == 24_000 * MIB - 3 * GIB - 1 * GIB - 2 * GIB
    assert decision.branch is CapacityBranch.RESIDUAL


def test_capacity_changes_when_the_active_region_set_changes() -> None:
    """The gate's dynamism clause: a per-host constant has measured the card, not the tick."""
    vram = _vram(24_000)
    four = SchedulerBudgets.from_context(8192, REGIONS)
    five = SchedulerBudgets.from_context(8192, (*REGIONS, "episodic_store"))
    assert (
        capacity_decision(vram, four).capacity_bytes != capacity_decision(vram, five).capacity_bytes
    )


def test_capacity_changes_when_the_context_length_changes() -> None:
    """The other half of `KV_reserved(context, regions_active)`."""
    vram = _vram(24_000)
    short = SchedulerBudgets.from_context(4096, REGIONS)
    long = SchedulerBudgets.from_context(8192, REGIONS)
    assert (
        capacity_decision(vram, short).capacity_bytes > capacity_decision(vram, long).capacity_bytes
    )


def test_kv_reserve_is_linear_in_both_arguments() -> None:
    """A model that ignored `regions_active` would still differ per card and still be wrong."""
    assert kv_reserved_bytes(1000, ("a",), 2048) == 1000 * 2048
    assert kv_reserved_bytes(1000, ("a", "b"), 2048) == 2 * kv_reserved_bytes(1000, ("a",), 2048)


# ---------------------------------------------------------------------------------------
# The pre-committed floor branch. Constructed, because section 9.11 says it will be reached.
# ---------------------------------------------------------------------------------------


def test_a_zero_residual_fires_the_floor_rather_than_yielding_a_zero_capacity() -> None:
    """*"A probe returning two DIFFERENT numbers where one is zero has also failed."*"""
    vram = _vram(16_303)  # the 5080, test-pinned in the design at this exact size
    starved = SchedulerBudgets(kv_reserved_bytes=14 * GIB, activation_reserve_bytes=1 * GIB)
    decision = capacity_decision(vram, starved, safety_margin_bytes=2 * GIB)

    assert decision.residual_bytes == 0, "this working set was supposed to starve the residual"
    assert decision.branch is CapacityBranch.FLOOR
    assert decision.capacity_bytes == DEFAULT_STORE_FLOOR_BYTES
    assert decision.capacity_bytes > 0


def test_the_floor_is_reserved_before_the_kv_budget_not_bolted_on_after() -> None:
    """A `max(residual, floor)` would overcommit the card; the floor takes its slice first.

    `kv_headroom_bytes` is what the KV cache is left with once the floor is taken, which is
    the context-for-recall trade section 8 says this branch is. It must be strictly less than
    what the scheduler asked for, or the floor cost nothing and came from nowhere.
    """
    vram = _vram(16_303)
    asked = 14 * GIB
    starved = SchedulerBudgets(kv_reserved_bytes=asked, activation_reserve_bytes=1 * GIB)
    decision = capacity_decision(vram, starved, safety_margin_bytes=2 * GIB)
    assert decision.kv_headroom_bytes < asked
    assert (
        decision.capacity_bytes
        + decision.kv_headroom_bytes
        + decision.activation_reserve_bytes
        + decision.safety_margin_bytes
        == decision.vram_total_bytes
    )


def test_a_card_too_small_for_the_floor_is_refused_rather_than_given_a_fiction() -> None:
    """The floor is a policy, not a promise the hardware can always keep."""
    vram = _vram(2_048)
    starved = SchedulerBudgets(kv_reserved_bytes=8 * GIB, activation_reserve_bytes=1 * GIB)
    with pytest.raises(ValueError, match="does not fit"):
        capacity_decision(vram, starved, safety_margin_bytes=2 * GIB)


def test_the_residual_branch_costs_the_context_nothing() -> None:
    """On the residual branch the store took leftovers, so KV headroom is what it asked for."""
    decision = capacity_decision(_vram(24_000), SchedulerBudgets.from_context(8192, REGIONS))
    assert decision.branch is CapacityBranch.RESIDUAL
    assert decision.kv_headroom_bytes == decision.kv_reserved_bytes


# ---------------------------------------------------------------------------------------
# "Respected": an over-capacity write evicts, and the capacity is re-read every tick.
# ---------------------------------------------------------------------------------------


def _store(provider) -> EpisodicStoreImpl:
    store = EpisodicStoreImpl(
        {"chat"},
        backend=InMemoryBackend(),
        capacity_provider=provider,
        host="test-host",
    )
    store.start()
    return store


def test_the_capacity_is_re_read_on_every_admission_and_never_cached() -> None:
    """DEC-63: *"computed... at scheduler-tick time and NEVER CACHED."*"""
    calls = {"n": 0}

    def provider() -> int:
        calls["n"] += 1
        return 1 << 30

    store = _store(provider)
    scope = derive_scope("alice")
    for i in range(5):
        store.learn(scope, "chat", f"k{i}", torch.ones(8))
    assert calls["n"] >= 5, "the store answered from a cached capacity"


def test_a_capacity_that_shrinks_between_ticks_is_obeyed_at_the_next_tick() -> None:
    """The 1080 Ti is preemptible: the number moves under the store and the store must follow."""
    capacity = {"value": 10_000}
    store = _store(lambda: capacity["value"])
    scope = derive_scope("alice")
    for i in range(5):
        store.learn(scope, "chat", f"k{i}", torch.ones(8), importance=float(i), span_bytes=1_000)
    assert len(store.resident_keys()) == 5

    capacity["value"] = 2_000
    store.evict()
    assert len(store.resident_keys()) == 2
    assert store.resident_bytes <= 2_000


def test_a_large_capacity_is_exercised_without_allocating_it() -> None:
    """DEC-65's index model is what makes the gate's own phrasing testable.

    *"a write that would exceed it triggers eviction rather than an ALLOCATION."* The store
    indexes runtime-owned spans, so a 13 GiB bound is exercised with a handful of 8-float
    tensors carrying real `span_bytes`. Allocating 13 GiB to prove a claim about not
    allocating would be a strange way to test it -- and impossible on a CI runner.
    """
    capacity = 13 * GIB
    store = _store(lambda: capacity)
    scope = derive_scope("alice")
    span = capacity // 4
    for i in range(6):
        store.learn(scope, "chat", f"k{i}", torch.ones(8), importance=float(i), span_bytes=span)
    assert store.resident_bytes <= capacity
    assert len(store.resident_keys()) == 4


# ---------------------------------------------------------------------------------------
# The two committed receipts, and the cross-card assertions no single host can make.
# ---------------------------------------------------------------------------------------


def _receipt(name: str) -> dict:
    path = RECEIPTS[name]
    assert path.is_file(), f"gate (ii) receipt missing: {path}"
    return json.loads(path.read_text())


def test_both_cards_produced_a_receipt_and_both_passed_their_own_checks() -> None:
    """Neither half of the probe is allowed to be absent, and neither is allowed to be red."""
    for name in RECEIPTS:
        receipt = _receipt(name)
        assert receipt["ok"] is True, f"{name} probe failed: {receipt.get('errors')}"
        assert receipt["row"] == "E1"
        assert receipt["checks"]["capacity_respected"] is True
        assert receipt["checks"]["region_set_changes_capacity"] is True


def test_the_two_cards_report_two_different_positive_capacities() -> None:
    """The gate's headline: differ, and neither is zero."""
    a = _receipt("3090ti")
    b = _receipt("5080")
    assert "3090 Ti" in a["device_name"]
    assert "5080" in b["device_name"]
    assert a["capacity_bytes"] > 0 and b["capacity_bytes"] > 0
    assert a["capacity_bytes"] != b["capacity_bytes"], (
        "one number for both cards is a measured constant, not a dynamic capacity"
    )


def test_each_receipt_records_which_branch_was_active_on_that_card() -> None:
    """*"E1's receipt records which of the two (residual or floor) is active on each card."*"""
    valid = {branch.value for branch in CapacityBranch}
    for name in RECEIPTS:
        receipt = _receipt(name)
        assert receipt["branch"] in valid
        assert receipt["arms"]["deployment"]["branch"] in valid
        assert receipt["arms"]["long_context"]["branch"] in valid


def test_the_floor_branch_fired_on_the_deployment_card_under_a_long_context() -> None:
    """Section 9.11's case, measured rather than predicted -- and its contrast on the 24 GiB card.

    The 5080's residual rounds to zero at a two-million-token context and the pre-committed
    floor takes over; the 3090 Ti, given the identical budgets, stays on the residual. A floor
    that fired on both cards, or on neither, would not show that the branch is selected by the
    card's own arithmetic.
    """
    assert _receipt("5080")["arms"]["long_context"]["branch"] == CapacityBranch.FLOOR.value
    assert _receipt("5080")["arms"]["long_context"]["capacity_bytes"] > 0
    assert _receipt("3090ti")["arms"]["long_context"]["branch"] == CapacityBranch.RESIDUAL.value


def test_each_receipt_shows_the_capacity_moving_with_the_active_region_set() -> None:
    """Measured on real cards, not only in the unit test above."""
    for name in RECEIPTS:
        receipt = _receipt(name)
        deployment = receipt["arms"]["deployment"]["capacity_bytes"]
        changed = receipt["arms"]["region_set_changed"]["capacity_bytes"]
        assert changed != deployment
        assert changed > 0


def test_each_receipt_shows_the_capacity_respected_by_eviction() -> None:
    """The respect arm ran on the card, against that card's own measured number."""
    for name in RECEIPTS:
        arm = _receipt(name)["arms"]["respected"]
        assert arm["peak_resident_bytes"] <= arm["capacity_bytes"]
        assert arm["evicted"] > 0, "nothing was evicted, so the bound never bound"
        assert arm["allocated_bytes_for_spans"] < arm["capacity_bytes"] // 1000, (
            "the respect arm allocated real memory instead of indexing declared spans"
        )


# ---------------------------------------------------------------------------------------
# The 5080 half must be re-run by CI, and must fail visibly if no GPU runner takes it.
# ---------------------------------------------------------------------------------------


def test_the_5080_probe_is_wired_to_the_gpu_runner() -> None:
    """A GPU gate that silently skips when no runner picks it up is a gate that passes by absence.

    So the wiring is asserted structurally: the job exists, it demands the 5080 runner's
    labels, it runs this row's probe script, and it carries neither `continue-on-error` nor an
    `if:` that could make it evaporate. It also has a timeout, so a job nothing picks up ends
    red rather than pending forever.
    """
    workflow = yaml.safe_load(GPU_WORKFLOW.read_text())
    job = workflow["jobs"]["e1-store-capacity-probe"]

    labels = {str(label) for label in job["runs-on"]}  # YAML parses the 5080 label as an int
    assert labels >= {"self-hosted", "gpu", "5080", "host-gpu5080"}
    assert "continue-on-error" not in job, "a GPU gate must not be allowed to pass while failing"
    assert "if" not in job, "a conditional GPU gate can disappear silently"
    assert isinstance(job.get("timeout-minutes"), int), (
        "without a timeout, a job no runner claims stays pending rather than failing visibly"
    )
    body = "\n".join(str(step.get("run", "")) for step in job["steps"])
    assert "scripts/run_store_capacity_probe.py" in body
    assert "capacity-probe-5080.json" in body, (
        "the CI run must compare against the committed receipt, not merely produce a number"
    )
