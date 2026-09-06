"""Tests for `cogsyndelta.interconnect.gates` -- spec section 5, Table 9's `gates.py` row:
"each gate passes its positive control", following the `tests/test_guards_can_fail.py`
pattern of constructing the exact condition a guard exists to catch and asserting it
fires.
"""

from __future__ import annotations

import pytest

from cogsyndelta.interconnect.gates import (
    POINT,
    GateFailure,
    check_attention_mass_floor,
    check_frozen_set_identity,
    check_null_gate,
    check_overfit_gate,
    check_phase_d_revert,
    check_topology_agreement,
    check_write_back_gate,
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
