"""Tests for `cogsyndelta.interconnect.gates` -- spec section 5, Table 9's `gates.py` row:
"each gate passes its positive control", following the `tests/test_guards_can_fail.py`
pattern of constructing the exact condition a guard exists to catch and asserting it
fires.
"""

from __future__ import annotations

import math
import statistics

import pytest

from cogsyndelta.interconnect.gates import (
    POINT,
    GateFailure,
    GateInconclusive,
    GateReport,
    check_attention_mass_floor,
    check_frozen_set_identity,
    check_null_gate,
    check_overfit_gate,
    check_phase_d_revert,
    check_topology_agreement,
    check_write_back_gate,
    evaluate_attention_mass_floor,
    evaluate_overfit_gate,
    replicate_verdict,
    require_conclusive,
)

pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------------------
# G27 -- write-back gate
# ---------------------------------------------------------------------------------------


def test_g27_positive_control_passes() -> None:
    check_write_back_gate(
        "language",
        own_bin_before=0.800,
        own_bin_after=0.805,
        composed_before=0.700,
        composed_after=0.710,
    )


def test_g27_fires_on_own_bin_drop_over_one_point() -> None:
    with pytest.raises(GateFailure, match="G27"):
        check_write_back_gate(
            "language",
            own_bin_before=0.800,
            own_bin_after=0.800 - 1.5 * POINT,
            composed_before=0.700,
            composed_after=0.710,
        )


def test_g27_own_bin_drop_of_exactly_one_point_passes() -> None:
    """The boundary is "> 1 point", not ">=" -- exactly one point must not fire."""
    check_write_back_gate(
        "language",
        own_bin_before=0.800,
        own_bin_after=0.800 - POINT,
        composed_before=0.700,
        composed_after=0.710,
    )


def test_g27_fires_when_composed_metric_does_not_improve() -> None:
    with pytest.raises(GateFailure, match="G27"):
        check_write_back_gate(
            "language",
            own_bin_before=0.800,
            own_bin_after=0.805,
            composed_before=0.700,
            composed_after=0.700,
        )


# ---------------------------------------------------------------------------------------
# G28 -- topology agreement
# ---------------------------------------------------------------------------------------


def test_g28_positive_control_passes() -> None:
    check_topology_agreement(0.97, item_count=200)


def test_g28_fires_when_the_two_derivations_disagree_on_10_percent() -> None:
    with pytest.raises(GateFailure, match="G28"):
        check_topology_agreement(0.90, item_count=200)


def test_g28_boundary_at_exactly_95_percent_passes() -> None:
    check_topology_agreement(0.95)


# ---------------------------------------------------------------------------------------
# G29 -- attention-mass floor
# ---------------------------------------------------------------------------------------


def test_g29_positive_control_passes() -> None:
    check_attention_mass_floor("language", mean_a=0.05, eta=0.15, r=5, printed_floor=0.03)


def test_g29_fires_when_a_region_is_collapsed() -> None:
    """One region's keys masked so its mass is zero -- the spec's own G29 test shape."""
    with pytest.raises(GateFailure, match="G29"):
        check_attention_mass_floor("visual", mean_a=0.0, eta=0.15, r=5)


def test_g29_fires_on_the_r5_store_falsifier() -> None:
    """`b_store` pinned at its legal floor of 8 on episodes with no recall dependency:
    the in-contract way `mean(a_store) < eta/R = 3.0%` at R=5 (TAX:2662)."""
    with pytest.raises(GateFailure, match="G29"):
        check_attention_mass_floor("store", mean_a=0.0375 - 0.01, eta=0.15, r=5)


def test_g29_fires_when_the_printed_floor_disagrees_with_eta_over_r() -> None:
    """A receipt printing 3.75% (eta/R at R=4) while actually measured at R=5."""
    with pytest.raises(GateFailure, match="G29"):
        check_attention_mass_floor("language", mean_a=0.10, eta=0.15, r=5, printed_floor=0.0375)


# ---------------------------------------------------------------------------------------
# G33 -- frozen-set identity
# ---------------------------------------------------------------------------------------


def _participant(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "language",
        "checkpoint_sha256": "a" * 64,
        "receipt_checkpoint_sha256": "a" * 64,
        "kind": "contrastive_encoder",
        "parametric": True,
    }
    base.update(overrides)
    return base


def test_g33_positive_control_passes() -> None:
    check_frozen_set_identity([_participant(), _participant(name="memory")])


def test_g33_fires_on_a_flipped_checkpoint_byte() -> None:
    """A checkpoint with one flipped byte against its receipt -- the spec's own case."""
    flipped = "a" * 63 + "b"
    with pytest.raises(GateFailure, match="G33"):
        check_frozen_set_identity([_participant(checkpoint_sha256=flipped)])


def test_g33_fires_on_a_parametric_region_claiming_nonparametric_store() -> None:
    with pytest.raises(GateFailure, match="G33"):
        check_frozen_set_identity(
            [_participant(name="episodic_store", kind="nonparametric_store", parametric=True)]
        )


def test_g33_nonparametric_store_declaring_its_own_kind_passes() -> None:
    check_frozen_set_identity(
        [_participant(name="episodic_store", kind="nonparametric_store", parametric=False)]
    )


# ---------------------------------------------------------------------------------------
# G34 -- phase-D revert
# ---------------------------------------------------------------------------------------


def test_g34_positive_control_passes() -> None:
    check_phase_d_revert(d_metric=0.75, c_metric=0.70, d_flops=900.0, c_flops=1000.0)


def test_g34_fires_when_d_is_worse_than_c() -> None:
    with pytest.raises(GateFailure, match="G34"):
        check_phase_d_revert(d_metric=0.65, c_metric=0.70, d_flops=900.0, c_flops=1000.0)


def test_g34_fires_when_d_beats_c_but_costs_more_flops() -> None:
    with pytest.raises(GateFailure, match="G34"):
        check_phase_d_revert(d_metric=0.75, c_metric=0.70, d_flops=1100.0, c_flops=1000.0)


# ---------------------------------------------------------------------------------------
# G35 -- overfit gate
# ---------------------------------------------------------------------------------------


def test_g35_positive_control_passes() -> None:
    check_overfit_gate(train_metric=0.73, held_out_metric=0.70)


def test_g35_fires_on_a_six_point_gap() -> None:
    with pytest.raises(GateFailure, match="G35"):
        check_overfit_gate(train_metric=0.76, held_out_metric=0.70)


def test_g35_fires_at_exactly_the_five_point_boundary() -> None:
    """ "gap >= 5 points" -- exactly five points must fire, not just above it."""
    with pytest.raises(GateFailure, match="G35"):
        check_overfit_gate(train_metric=0.75, held_out_metric=0.70)


# ---------------------------------------------------------------------------------------
# G36 -- NULL gate
# ---------------------------------------------------------------------------------------


def test_g36_positive_control_passes() -> None:
    check_null_gate(general_null_recall=0.60, per_bin_null_fpr={"code": 0.02, "memory": 0.01})


def test_g36_fires_on_low_general_bin_null_recall() -> None:
    """A general-bin NULL recall of 0.40 -- the spec's own G36 test shape."""
    with pytest.raises(GateFailure, match="G36"):
        check_null_gate(general_null_recall=0.40, per_bin_null_fpr={"code": 0.01})


def test_g36_fires_on_high_off_bin_null_false_positive_rate() -> None:
    """An off-bin NULL false-positive rate of 0.10 -- the spec's own G36 test shape."""
    with pytest.raises(GateFailure, match="G36"):
        check_null_gate(general_null_recall=0.60, per_bin_null_fpr={"code": 0.10})


# ---------------------------------------------------------------------------------------
# The third verdict -- `evaluate_*`, `replicate_verdict`, `require_conclusive`
#
# Two of these gates were measured not to discriminate on phase A's synthetic stream
# (`/akula-data/session-backup-staging/notes/GATE-DISCRIMINATION-2026-09-07.md`). The
# tests below pin the invariant that makes the fix a repair rather than a loosening --
# `"inconclusive"` may replace a `"pass"`, never a `"fail"` -- and then replay the two
# real sweeps to show what the replicated verdict says about them.
# ---------------------------------------------------------------------------------------

# `mean(a_store)` at `OMP_NUM_THREADS` 8, 800 steps, seeds 0-7, phase A's synthetic
# stream at R=3, eta=0.15 (floor 0.05):
# `/akula-data/scratch/csd-g29-diag-evidence/out/cli_seed{0..7}.json`.
G29_EIGHT_SEED_SWEEP = (0.0468, 0.0783, 0.1500, 0.1351, 0.1952, 0.2263, 0.1476, 0.0463)

# `dev_recall_at_1` over 24 seeds at 200 steps with `torch.set_num_threads` pinned;
# `train_metric` was 1.0 in every one of them:
# `/akula-data/scratch/csd-det/runs/seeds24-db8-pin1/s{0..23}/meta.json`.
# Written as MISSED held-out items rather than as recalls, because the miss count is what
# the gate is really reading: with `train_metric` pinned at 1.0 the gap is `misses / 64`.
G35_MISSES_PER_SEED = (0, 0, 4, 0, 0, 7, 0, 0) + (0, 1, 3, 1, 3, 1, 2, 1) + (3, 1, 2, 4, 0, 0, 1, 0)
G35_TWENTY_FOUR_SEED_DEV = tuple(1.0 - m / 64 for m in G35_MISSES_PER_SEED)


def test_gate_inconclusive_is_not_a_gate_failure() -> None:
    """A caller catching `GateFailure` must not silently absorb "could not tell"."""
    assert not issubclass(GateInconclusive, GateFailure)
    assert not issubclass(GateFailure, GateInconclusive)


# --------------------------------------------------------------- the invariant, both gates


def test_g29_a_fired_gate_still_fails_however_large_the_spread() -> None:
    """`"inconclusive"` may replace a `"pass"`, never a `"fail"` -- G29 half."""
    report = evaluate_attention_mass_floor("store", mean_a=0.01, eta=0.15, r=3, spread=100.0)
    assert report.fired is True
    assert report.verdict == "fail"
    with pytest.raises(GateFailure, match="G29"):
        require_conclusive(report)


def test_g35_a_fired_gate_still_fails_however_large_the_spread() -> None:
    """`"inconclusive"` may replace a `"pass"`, never a `"fail"` -- G35 half."""
    report = evaluate_overfit_gate(1.0, 0.80, held_out_n=64, spread=100.0)
    assert report.fired is True
    assert report.verdict == "fail"
    with pytest.raises(GateFailure, match="G35"):
        require_conclusive(report)


def test_evaluate_g29_fires_exactly_where_the_raising_guard_does() -> None:
    """The reporter and the ratified guard share one predicate; prove they cannot drift."""
    for mean_a in (0.0, 0.0316, 0.0468, 0.049, 0.05, 0.0500001, 0.1, 0.9):
        report = evaluate_attention_mass_floor("store", mean_a=mean_a, eta=0.15, r=3)
        raised = False
        try:
            check_attention_mass_floor("store", mean_a, 0.15, 3)
        except GateFailure:
            raised = True
        assert report.fired is raised, mean_a


def test_evaluate_g35_fires_exactly_where_the_raising_guard_does() -> None:
    """Including the exact-boundary case the `isclose` half exists for."""
    for train, held in ((0.73, 0.70), (0.75, 0.70), (0.76, 0.70), (0.35, 0.30), (1.0, 1.0)):
        report = evaluate_overfit_gate(train, held)
        raised = False
        try:
            check_overfit_gate(train, held)
        except GateFailure:
            raised = True
        assert report.fired is raised, (train, held)


# ------------------------------------------------------------------- refusing to evaluate


def test_evaluate_g29_without_a_spread_cannot_certify_a_pass() -> None:
    """One draw of a statistic that flips on thread count is not a pass."""
    report = evaluate_attention_mass_floor("store", mean_a=0.20, eta=0.15, r=3)
    assert report.fired is False
    assert report.verdict == "inconclusive"
    assert "no spread supplied" in report.summary()
    with pytest.raises(GateInconclusive, match="G29"):
        require_conclusive(report)


def test_evaluate_g29_margin_inside_the_spread_is_inconclusive() -> None:
    report = evaluate_attention_mass_floor("store", mean_a=0.0783, eta=0.15, r=3, spread=0.0664)
    assert report.fired is False
    assert report.verdict == "inconclusive"
    assert report.margin_over_spread is not None
    assert report.margin_over_spread < 1.0


def test_evaluate_g29_margin_clear_of_the_spread_is_a_pass() -> None:
    """The `memory_only` arm, where the store is the only route to the label: measured
    `mean(a_store)` 0.19-0.64 against the same 0.05 floor."""
    report = evaluate_attention_mass_floor("store", mean_a=0.2673, eta=0.15, r=3, spread=0.1025)
    assert report.verdict == "pass"
    require_conclusive(report)


def test_evaluate_g35_margin_below_one_quantum_is_inconclusive() -> None:
    """`recall@1` over 64 items moves in 1.5625-point steps: a margin under one step is a
    verdict one held-out item from flipping. Measured on 3 of the 24 pinned seeds."""
    report = evaluate_overfit_gate(1.0, 1.0 - 0.046875, held_out_n=64, spread=0.0)
    assert report.fired is False
    assert report.verdict == "inconclusive"
    assert any("resolution" in note for note in report.notes)


def test_evaluate_g35_reports_the_ceiling_it_actually_enforced() -> None:
    """The spec's ceiling is 5.00 points and is not moved. At n=64 the metric cannot
    express it, so the enforced ceiling is 6.25; the report says so rather than pretending."""
    report = evaluate_overfit_gate(1.0, 1.0, held_out_n=64, spread=0.001)
    assert report.threshold == pytest.approx(5 * POINT)
    assert any("6.2500 points" in note for note in report.notes)
    assert any("1.25x the spec" in note for note in report.notes)


def test_evaluate_g35_flags_a_saturated_train_operand() -> None:
    """`train_metric` was 1.0 in all 147 measured 200-step runs, which makes the "gap"
    exactly `1 - dev`: a dev-only bar. A note, not a verdict -- it changes what the number
    means, not how precisely it is known."""
    saturated = evaluate_overfit_gate(1.0, 0.9844, held_out_n=64, spread=0.001)
    assert any("saturated at 1.0" in note for note in saturated.notes)
    honest = evaluate_overfit_gate(0.9844, 0.9688, held_out_n=64, spread=0.001)
    assert not any("saturated" in note for note in honest.notes)


# ----------------------------------------------------------------- the margin as a number


def test_margin_over_spread_is_scale_invariant() -> None:
    """Measured exactly invariant under a global rescale; the property is why it is the
    statistic to consume. Scaling the mass, the floor and the spread alike must not move it."""
    base = evaluate_attention_mass_floor("store", mean_a=0.20, eta=0.15, r=3, spread=0.05)
    for factor in (10.0, 100.0):
        scaled = evaluate_attention_mass_floor(
            "store", mean_a=0.20 * factor, eta=0.15 * factor, r=3, spread=0.05 * factor
        )
        assert scaled.margin_over_spread == pytest.approx(base.margin_over_spread)


def test_margin_over_spread_is_none_without_a_spread_and_infinite_without_variance() -> None:
    assert evaluate_attention_mass_floor("store", 0.2, 0.15, 3).margin_over_spread is None
    zero_spread = evaluate_attention_mass_floor("store", 0.2, 0.15, 3, spread=0.0)
    assert zero_spread.margin_over_spread == math.inf
    assert zero_spread.verdict == "pass"


# --------------------------------------------------------------------- seed replication


def _g29_reports(masses: tuple[float, ...]) -> list[GateReport]:
    return [evaluate_attention_mass_floor("store", m, 0.15, 3) for m in masses]


def test_replicate_verdict_refuses_a_single_draw() -> None:
    with pytest.raises(ValueError, match="at least 2 seeds"):
        replicate_verdict(_g29_reports((0.2,)))


def test_replicate_verdict_refuses_reports_from_different_gates() -> None:
    mixed = [
        evaluate_attention_mass_floor("store", 0.2, 0.15, 3),
        evaluate_overfit_gate(1.0, 1.0),
    ]
    with pytest.raises(ValueError, match="more than one gate"):
        replicate_verdict(mixed)


def test_replicate_verdict_refuses_replicates_gated_against_different_thresholds() -> None:
    differing = [
        evaluate_attention_mass_floor("store", 0.2, 0.15, 3),
        evaluate_attention_mass_floor("store", 0.2, 0.15, 5),
    ]
    with pytest.raises(ValueError, match="different thresholds"):
        replicate_verdict(differing)


def test_replicate_verdict_unanimous_fail_is_a_fail() -> None:
    combined = replicate_verdict(_g29_reports((0.010, 0.012, 0.009, 0.011)))
    assert combined.fired is True
    assert combined.verdict == "fail"
    with pytest.raises(GateFailure, match="G29"):
        require_conclusive(combined)


def test_replicate_verdict_unanimous_pass_with_a_tight_spread_is_a_pass() -> None:
    combined = replicate_verdict(_g29_reports((0.300, 0.305, 0.298, 0.302)))
    assert combined.verdict == "pass"
    require_conclusive(combined)


def test_replicate_verdict_mixed_is_inconclusive_and_never_a_pass() -> None:
    """A minority of firing seeds must not be voted away, and must not be promoted into a
    hard FAIL either: on this stream the firings flip on `OMP_NUM_THREADS` alone."""
    combined = replicate_verdict(_g29_reports((0.0468, 0.20, 0.21, 0.22)))
    assert combined.verdict == "inconclusive"
    assert combined.verdict != "pass"
    assert "1 of 4 replicate seeds fired" in combined.summary()
    with pytest.raises(GateInconclusive, match="G29"):
        require_conclusive(combined)


def test_replicate_verdict_reports_the_seed_axis_spread_it_measured() -> None:
    masses = (0.10, 0.20, 0.30, 0.40)
    combined = replicate_verdict(_g29_reports(masses))
    assert combined.spread == pytest.approx(statistics.stdev(masses))
    assert combined.statistic == pytest.approx(statistics.fmean(masses))


# ------------------------------------------------- the two measured sweeps, replayed


def test_measured_g29_eight_seed_sweep_reports_inconclusive_not_a_coin_flip() -> None:
    """Phase A's synthetic stream, 800 steps, `OMP_NUM_THREADS=8`, seeds 0-7. Two seeds
    fire and six do not, on a store whose read is item-invariant. Run one at a time this
    is a PASS or a FAIL depending on which seed you drew; replicated it is neither."""
    combined = replicate_verdict(_g29_reports(G29_EIGHT_SEED_SWEEP))
    assert combined.verdict == "inconclusive"
    assert "2 of 8 replicate seeds fired" in combined.summary()


def test_measured_g35_twenty_four_seed_sweep_reports_inconclusive() -> None:
    """24 pinned-thread seeds at 200 steps. Three fire; the gap's seed-axis spread is 2.80
    points against the 5.00-point ceiling it is compared against, so the mean margin does
    not clear its own spread either. Both routes reach the same honest answer."""
    reports = [evaluate_overfit_gate(1.0, dev, held_out_n=64) for dev in G35_TWENTY_FOUR_SEED_DEV]
    assert sum(1 for r in reports if r.fired) == 3
    combined = replicate_verdict(reports)
    assert combined.verdict == "inconclusive"
    assert combined.spread is not None
    assert combined.spread / POINT == pytest.approx(2.80, abs=0.01)
    assert combined.margin_over_spread is not None
    assert combined.margin_over_spread < 1.0


def test_measured_g29_useful_store_arm_is_the_positive_control_that_passes() -> None:
    """The same gate, same code, on a stream where the store IS the only route to the
    label: `mean(a_store)` never leaves [0.19, 0.64] over 800 steps at either seed, 3.8x
    the floor. G29 discriminates; phase A's stream is what does not exercise it."""
    combined = replicate_verdict(_g29_reports((0.1948, 0.3398, 0.2616, 0.3122)))
    assert combined.verdict == "pass"
    require_conclusive(combined)
