"""Every phase-A guard, shown firing on a constructed failure -- spec section 5's style.

`tests/test_interconnect_guards_can_fail.py` already proves that `gates.py`'s ten functions
fire when called with bad numbers. That is a different claim from this one. What is checked
here is that PHASE A ACTUALLY CALLS THEM, and that each guard's stated EFFECT happens: G33
refuses to start before a parameter is touched, G29 names the collapsed region in the
receipt, G35 and G36 mark the receipt FAIL. A guard that is implemented perfectly and never
reached is a guard that cannot fire, which is the failure mode memory note
`verify-guards-by-making-them-fail` records: three CSD guards were structurally incapable
of firing and reading them confirmed intent, not behaviour.

Every guard therefore gets both arms: a constructed failure, and a positive control on the
same path, so a guard that refuses everything is caught too.

Run directly:
    CUDA_VISIBLE_DEVICES="" python -m pytest tests/interconnect/test_phase_a_guards_can_fail.py -q
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from cogsyndelta.interconnect import phase_a
from cogsyndelta.interconnect.gates import GateFailure
from cogsyndelta.interconnect.mind import WhiteMatter
from cogsyndelta.interconnect.phase_a import (
    PhaseAConfig,
    PhaseATrainer,
    check_phase_a_frozen_set,
    check_receipt_collapse_floor,
    phase_a_guards,
)
from tests.interconnect.test_phase_a import honest_frozen_set, identity_for, toy_batch

# ----------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------


@pytest.fixture
def trainer(white_matter: WhiteMatter) -> PhaseATrainer:
    """A `PhaseATrainer` over the toy `white_matter`, with a frozen set G33 accepts.

    Defined here rather than imported from `test_phase_a.py`: a fixture crosses files only
    through a `conftest.py`, and the toy `conftest.py` belongs to lane IC-8.
    """
    return PhaseATrainer(white_matter, PhaseAConfig(lr=3e-3), honest_frozen_set(white_matter))


def _healthy_masses() -> dict[str, float]:
    """Three regions well clear of `eta/R = 0.15/3 = 0.05`."""
    return {"language": 0.4, "visual": 0.35, "episodic_store": 0.25}


def _built_receipt(
    trainer: PhaseATrainer, white_matter: WhiteMatter, toy_scope: Any, *, steps: int = 3
) -> dict[str, Any]:
    """Run phase A briefly and return the built (unwritten) receipt dict."""
    batch = toy_batch(white_matter, toy_scope)
    result = trainer.run([batch], [batch], steps=steps)
    return trainer.build_receipt(result, identity_for(white_matter)).build()


def _mask_region_keys(white_matter: WhiteMatter, region: str) -> Any:
    """Mask every one of `region`'s bank slots, so it can receive no attention mass.

    Spec section 5's own G29 construction: "a phase-A run with one region's keys masked so
    its mass is zero". A forward hook on `KVBank` is used rather than an edit to the module
    under test, so the run is otherwise the real one -- the same optimizer, the same loss,
    the same receipt path.
    """
    index = white_matter.participant_names.index(region)

    def hook(_module: Any, _args: Any, output: Any) -> Any:
        bank_k, bank_v, key_mask, slot_region = output
        return bank_k, bank_v, key_mask & (slot_region != index), slot_region

    return white_matter.kv_bank.register_forward_hook(hook)


# ----------------------------------------------------------------------------------
# G33 -- frozen-set identity. Effect: phase A refuses to START.
# ----------------------------------------------------------------------------------


def test_g33_positive_control_an_honest_frozen_set_constructs(
    white_matter: WhiteMatter,
) -> None:
    """The control: matching shas, every field present, the module's own participants."""
    trainer = PhaseATrainer(white_matter, PhaseAConfig(), honest_frozen_set(white_matter))
    assert trainer.trainable_parameter_count() > 0


def test_g33_fires_on_a_checkpoint_sha_that_disagrees_with_its_receipt(
    white_matter: WhiteMatter,
) -> None:
    """Spec section 5: "a checkpoint with one flipped byte against its receipt"."""
    rows = honest_frozen_set(white_matter)
    measured = rows[0]["checkpoint_sha256"]
    flipped = measured[:-1] + ("0" if measured[-1] != "0" else "1")
    assert flipped != measured, "the constructed failure must actually differ"
    rows[0]["checkpoint_sha256"] = flipped
    with pytest.raises(GateFailure, match=r"G33"):
        PhaseATrainer(white_matter, PhaseAConfig(), rows)


def test_g33_fires_on_a_parametric_region_claiming_nonparametric_store(
    white_matter: WhiteMatter,
) -> None:
    """Spec section 5: "a parametric region claiming `nonparametric_store`"."""
    rows = honest_frozen_set(white_matter)
    rows[0]["kind"] = "nonparametric_store"
    with pytest.raises(GateFailure, match=r"G33"):
        PhaseATrainer(white_matter, PhaseAConfig(), rows)


@pytest.mark.parametrize("field", ["receipt_path", "status"])
def test_g33_fires_when_a_row_omits_a_table_7_field(white_matter: WhiteMatter, field: str) -> None:
    """Table 7's frozen-set group requires `receipt_path` and `status` per region.

    The identity half alone passes this row -- its two shas agree and its `kind` is honest
    -- so without the extra check a run could start against a region nobody can trace back
    to a receipt.
    """
    rows = honest_frozen_set(white_matter)
    del rows[0][field]
    with pytest.raises(GateFailure, match=f"G33.*{field}"):
        PhaseATrainer(white_matter, PhaseAConfig(), rows)


def test_g33_fires_when_the_participant_list_disagrees_with_the_module(
    white_matter: WhiteMatter,
) -> None:
    """A frozen set for four regions against a module built for five is a mismatch.

    Every row is internally consistent, so the identity half passes it; what fails is that
    the run would be describing a different composition from the one about to execute.
    """
    rows = honest_frozen_set(white_matter)[:-1]
    with pytest.raises(GateFailure, match=r"G33.*participants"):
        PhaseATrainer(white_matter, PhaseAConfig(), rows)


def test_g33_fires_when_r_disagrees_with_the_participant_count() -> None:
    """`R` and the participant list must agree -- the receipt prints both."""
    rows = [
        {
            "name": "language",
            "checkpoint_sha256": "a" * 64,
            "receipt_checkpoint_sha256": "a" * 64,
            "receipt_path": "receipts/language.json",
            "status": "frozen",
            "kind": "contrastive_encoder",
            "parametric": True,
        }
    ]
    with pytest.raises(GateFailure, match=r"G33.*R=5"):
        check_phase_a_frozen_set(rows, expected_participants=["language"], expected_r=5)


def test_g33_refuses_before_any_parameter_is_touched(white_matter: WhiteMatter) -> None:
    """ "Refuses to START" means the module is left exactly as it was found.

    A refusal that had already flipped `requires_grad` on half the module would leave a
    caller holding a half-configured `WhiteMatter` that looks trained-ready and is not.
    """
    for param in white_matter.parameters():
        param.requires_grad_(False)
    for faculty in white_matter.faculties.values():
        for param in faculty.parameters():
            param.requires_grad_(True)
    before = {name: p.requires_grad for name, p in white_matter.named_parameters()}
    faculties_before = [
        p.requires_grad for f in white_matter.faculties.values() for p in f.parameters()
    ]

    rows = honest_frozen_set(white_matter)
    rows[0]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(GateFailure, match=r"G33"):
        PhaseATrainer(white_matter, PhaseAConfig(), rows)

    assert {name: p.requires_grad for name, p in white_matter.named_parameters()} == before
    assert [
        p.requires_grad for f in white_matter.faculties.values() for p in f.parameters()
    ] == faculties_before


# ----------------------------------------------------------------------------------
# G29 -- attention-mass floor. Effects: the region is NAMED; a bad printed floor REFUSES.
# ----------------------------------------------------------------------------------


def test_g29_positive_control_a_real_run_names_nothing_and_prints_the_right_floor(
    trainer: PhaseATrainer, white_matter: WhiteMatter, toy_scope: Any
) -> None:
    """The control, end to end: no region collapses and the receipt's own floor validates."""
    receipt = _built_receipt(trainer, white_matter, toy_scope)
    assert receipt["attention_mass"]["collapsed_in_phase_A"] == []
    assert receipt["budgets"]["collapse_floor"] == {
        "expression": "eta/R",
        "R": 3,
        "value": pytest.approx(0.05),
    }
    check_receipt_collapse_floor(receipt)


def test_g29_fires_end_to_end_when_a_regions_keys_are_masked(
    white_matter: WhiteMatter, toy_scope: Any
) -> None:
    """Spec section 5: "one region's keys masked so its mass is zero".

    The whole path runs -- optimizer, loss, gates, receipt -- and the effect Table 8 states
    is what is asserted: the region is named in `collapsed_in_phase_A`, so phase B will
    exclude it from its distillation targets.
    """
    handle = _mask_region_keys(white_matter, "language")
    try:
        trainer = PhaseATrainer(
            white_matter, PhaseAConfig(lr=3e-3), honest_frozen_set(white_matter)
        )
        batch = toy_batch(white_matter, toy_scope)
        result = trainer.run([batch], [batch], steps=3)
    finally:
        handle.remove()

    assert result.mean_per_region["language"] == pytest.approx(0.0, abs=1e-9)
    assert "language" in result.guards.collapsed_in_phase_A
    assert "G29" in result.guards.floor_failures["language"]
    assert not result.guards.passed

    receipt = trainer.build_receipt(result, identity_for(white_matter)).build()
    assert "language" in receipt["attention_mass"]["collapsed_in_phase_A"]
    assert receipt["verdicts"]["integration"].startswith("FAIL")
    assert "G29" in receipt["verdicts"]["integration"]


def test_g29_fires_on_a_receipt_printing_3_75_percent_at_r_5(
    trainer: PhaseATrainer, white_matter: WhiteMatter, toy_scope: Any
) -> None:
    """Spec section 5's own case: "a receipt printing 3.75% at `R = 5`".

    `eta/5 = 0.15/5 = 0.03`, so 3.75% is not this receipt's floor at this receipt's `R`.
    The effect is that the receipt is refused, not that a region is degraded.
    """
    receipt = _built_receipt(trainer, white_matter, toy_scope)
    receipt["budgets"]["collapse_floor"] = {"expression": "eta/R", "R": 5, "value": 0.0375}
    with pytest.raises(GateFailure, match=r"G29.*printed floor"):
        check_receipt_collapse_floor(receipt)


def test_g29_fires_on_a_stale_printed_floor_at_the_receipts_own_r(
    trainer: PhaseATrainer, white_matter: WhiteMatter, toy_scope: Any
) -> None:
    """The same failure with only ONE field tampered: `R` is honest, the value is stale.

    This is the shape the guard exists for -- a floor computed at one `R` and printed
    beside another -- with no second edit that could be catching it instead.
    """
    receipt = _built_receipt(trainer, white_matter, toy_scope)
    receipt["budgets"]["collapse_floor"]["value"] = 0.0375
    with pytest.raises(GateFailure, match=r"G29.*printed floor"):
        check_receipt_collapse_floor(receipt)


def test_g29_fires_on_a_receipt_hiding_a_collapsed_region(
    trainer: PhaseATrainer, white_matter: WhiteMatter, toy_scope: Any
) -> None:
    """A region below the floor that the receipt did NOT name must still be caught.

    `collapsed_in_phase_A` is the receipt's own claim about which regions were degraded; a
    reader must not have to take it on trust, or a receipt could pass by omission.
    """
    receipt = _built_receipt(trainer, white_matter, toy_scope)
    receipt["attention_mass"]["mean_per_region"]["visual"] = 0.001
    with pytest.raises(GateFailure, match=r"G29.*'visual'"):
        check_receipt_collapse_floor(receipt)


def test_g29_fires_when_every_region_is_named_collapsed(
    trainer: PhaseATrainer, white_matter: WhiteMatter, toy_scope: Any
) -> None:
    """With no surviving region there is nothing honest left to check, so refuse."""
    receipt = _built_receipt(trainer, white_matter, toy_scope)
    receipt["attention_mass"]["collapsed_in_phase_A"] = list(
        receipt["attention_mass"]["mean_per_region"]
    )
    with pytest.raises(GateFailure, match=r"G29.*every one"):
        check_receipt_collapse_floor(receipt)


def test_g29_receipt_floor_survives_a_json_round_trip(
    trainer: PhaseATrainer, white_matter: WhiteMatter, toy_scope: Any, tmp_path: Path
) -> None:
    """A reader with nothing but the file on disk reaches the same verdict as the run."""
    batch = toy_batch(white_matter, toy_scope)
    result = trainer.run([batch], [batch], steps=3)
    path = trainer.write_receipt(result, identity_for(white_matter), tmp_path)
    check_receipt_collapse_floor(json.loads(path.read_text()))


# ----------------------------------------------------------------------------------
# G35 -- overfit gate. Effect: the phase-A receipt is marked FAIL.
# ----------------------------------------------------------------------------------


def test_g35_positive_control_a_three_point_gap_passes() -> None:
    """The control: under five points, and the verdict is a PASS."""
    report = phase_a_guards(
        mean_per_region=_healthy_masses(),
        eta=0.15,
        r=3,
        train_metric=0.80,
        held_out_metric=0.77,
        general_null_recall=0.90,
        per_bin_null_fpr={"content": 0.01},
    )
    assert report.overfit_failure is None
    assert report.passed
    assert report.verdict().startswith("PASS")


def test_g35_fires_on_a_six_point_gap() -> None:
    """Spec section 5: "a 6-point gap (G35)". The gate is "gap < 5 points"."""
    report = phase_a_guards(
        mean_per_region=_healthy_masses(),
        eta=0.15,
        r=3,
        train_metric=0.80,
        held_out_metric=0.74,
        general_null_recall=0.90,
        per_bin_null_fpr={"content": 0.01},
    )
    assert report.overfit_failure is not None
    assert "G35" in report.overfit_failure
    assert report.overfit_gap == pytest.approx(0.06)
    assert not report.passed
    assert report.verdict().startswith("FAIL")


def test_g35_marks_the_receipt_fail_end_to_end(
    trainer: PhaseATrainer, white_matter: WhiteMatter, toy_scope: Any
) -> None:
    """Table 8 row G35's stated effect, checked on a real built receipt.

    The run itself is real; only the two metrics the gate reads are replaced, which is the
    smallest construction that reaches the effect without faking the receipt.
    """
    batch = toy_batch(white_matter, toy_scope)
    result = trainer.run([batch], [batch], steps=3)
    result.guards = phase_a_guards(
        mean_per_region=result.mean_per_region,
        eta=white_matter.config.floor_eta,
        r=len(white_matter.participant_names),
        train_metric=0.80,
        held_out_metric=0.74,
        general_null_recall=0.90,
        per_bin_null_fpr={"content": 0.01},
    )
    receipt = trainer.build_receipt(result, identity_for(white_matter)).build()
    assert receipt["verdicts"]["integration"].startswith("FAIL")
    assert "G35" in receipt["verdicts"]["integration"]


# ----------------------------------------------------------------------------------
# G36 -- NULL gate. Effect: the phase-A receipt is marked FAIL per bin.
# ----------------------------------------------------------------------------------


def test_g36_positive_control_healthy_recall_and_fpr_pass() -> None:
    """The control: recall above 0.50, every off-bin false-positive rate under 0.05."""
    report = phase_a_guards(
        mean_per_region=_healthy_masses(),
        eta=0.15,
        r=3,
        train_metric=0.80,
        held_out_metric=0.79,
        general_null_recall=0.72,
        per_bin_null_fpr={"content": 0.02, "code": 0.01},
    )
    assert report.null_failure is None
    assert report.passed


@pytest.mark.parametrize(
    ("recall", "fpr"),
    [(0.40, {"content": 0.01}), (0.90, {"content": 0.10})],
    ids=["low-general-recall", "high-off-bin-fpr"],
)
def test_g36_fires_on_either_half(recall: float, fpr: dict[str, float]) -> None:
    """Spec section 5: "a general-bin `NULL` recall of 0.40 and an off-bin FPR of 0.10"."""
    report = phase_a_guards(
        mean_per_region=_healthy_masses(),
        eta=0.15,
        r=3,
        train_metric=0.80,
        held_out_metric=0.79,
        general_null_recall=recall,
        per_bin_null_fpr=fpr,
    )
    assert report.null_failure is not None
    assert "G36" in report.null_failure
    assert not report.passed
    assert "G36" in report.verdict()


# ----------------------------------------------------------------------------------
# The partition itself -- a guard over the definition of the trainable set
# ----------------------------------------------------------------------------------


def test_the_partition_refuses_a_parameter_claimed_by_both_columns(
    white_matter: WhiteMatter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A submodule listed in BOTH Table 6 columns must be refused, not silently resolved.

    Constructed by adding `workspace.` to the frozen list, which the trainable list already
    claims -- the shape a future edit that moved a submodule between columns and forgot to
    remove the old entry would take. Whichever way such a tie were broken silently, half
    the module would be trained or frozen against Table 6 with nothing to say so.
    """
    monkeypatch.setattr(
        phase_a, "PHASE_A_FROZEN_PREFIXES", ("controller.", "workspace."), raising=True
    )
    with pytest.raises(GateFailure, match=r"matched both Table 6 columns"):
        PhaseATrainer(white_matter, PhaseAConfig(), honest_frozen_set(white_matter))


def test_guard_report_records_the_gates_it_could_not_evaluate() -> None:
    """Table 6 row A's gate column also names G0, G1 and G2; none exists in `gates.py`.

    They are declared unevaluated rather than omitted, so a receipt cannot be read as
    "every gate in the row passed" when three of them were never checked.
    """
    report = phase_a_guards(
        mean_per_region=_healthy_masses(),
        eta=0.15,
        r=3,
        train_metric=0.80,
        held_out_metric=0.79,
        general_null_recall=0.72,
        per_bin_null_fpr={"content": 0.02},
    )
    assert report.unimplemented_gates == ("G0", "G1", "G2")


def test_phase_a_guards_refuses_an_empty_measurement() -> None:
    """No regions measured means no floor was checked; that must not read as a pass."""
    with pytest.raises(ValueError, match="mean_per_region is empty"):
        phase_a_guards(
            mean_per_region={},
            eta=0.15,
            r=3,
            train_metric=0.8,
            held_out_metric=0.8,
            general_null_recall=0.9,
            per_bin_null_fpr={},
        )


def test_a_deep_copy_of_the_healthy_report_is_not_accidentally_shared() -> None:
    """`PhaseAGuardReport`'s mutable defaults must be per-instance, not class-level."""
    first = phase_a_guards(
        mean_per_region={"language": 0.02, "visual": 0.98},
        eta=0.15,
        r=2,
        train_metric=0.8,
        held_out_metric=0.8,
        general_null_recall=0.9,
        per_bin_null_fpr={},
    )
    second = phase_a_guards(
        mean_per_region=_healthy_masses(),
        eta=0.15,
        r=3,
        train_metric=0.8,
        held_out_metric=0.8,
        general_null_recall=0.9,
        per_bin_null_fpr={},
    )
    assert first.collapsed_in_phase_A == ["language"]
    assert second.collapsed_in_phase_A == []
    assert copy.deepcopy(first).collapsed_in_phase_A == ["language"]
