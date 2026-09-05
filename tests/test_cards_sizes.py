"""Tests for `cogsyndelta.cards.sizes`.

Covers: every number is either read straight off a receipt or off a budgets file, and
`training_peak`/`width_histogram` REFUSE (`CardError`) rather than print an unmeasured
number under a MEASURED label -- mutation-proofed by writing a budget file whose
`source` is not `"report_peak"` and confirming the refusal fires.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cogsyndelta.cards.methodology import CardError
from cogsyndelta.cards.sizes import (
    build_size_report,
    compression_ratio,
    deployed_parameter_count,
    disk_bytes_per_state,
    eval_peak_vram,
    parameter_count,
    training_peak,
    width_histogram,
)

pytestmark = pytest.mark.cpu


def test_parameter_count_reads_the_receipt_field() -> None:
    assert parameter_count({"parameters": 16021248}) == 16021248


def test_parameter_count_none_when_absent() -> None:
    assert parameter_count({}) is None


def test_deployed_parameter_count_prefers_eval_provenance() -> None:
    train = {"parameters": 22905216}
    eval_receipt = {"provenance": {"parameters": 10712448}}
    assert deployed_parameter_count(train, eval_receipt) == 10712448


def test_deployed_parameter_count_falls_back_to_train() -> None:
    assert deployed_parameter_count({"parameters": 16021248}, None) == 16021248
    assert deployed_parameter_count({"parameters": 16021248}, {"provenance": {}}) == 16021248


def test_width_histogram_none_quant_receipt_returns_none() -> None:
    assert width_histogram(None) is None


def test_width_histogram_reads_and_stringifies_keys() -> None:
    hist = width_histogram({"width_histogram": {3: 17}})
    assert hist == {"3": 17}


def test_width_histogram_raises_when_quant_receipt_has_no_histogram() -> None:
    with pytest.raises(CardError, match="width_histogram"):
        width_histogram({"stored_bytes": 100})


def test_disk_bytes_per_state_fp32_and_csd_ptq_v1() -> None:
    quant = {"fp32_bytes": 64084992, "stored_bytes": 6519016}
    out = disk_bytes_per_state({}, quant)
    assert out == {"fp32": 64084992, "csd-ptq-v1": 6519016}


def test_disk_bytes_per_state_no_quant_receipt_is_empty() -> None:
    assert disk_bytes_per_state({}, None) == {}


def test_compression_ratio_v2_name_preferred_over_v1() -> None:
    assert compression_ratio({"quant.compression_ratio": 9.83, "compression_ratio": 1.0}) == 9.83


def test_compression_ratio_falls_back_to_v1_name() -> None:
    assert compression_ratio({"compression_ratio": 9.83}) == 9.83


def test_compression_ratio_none_receipt() -> None:
    assert compression_ratio(None) is None


def test_eval_peak_vram_labels_with_batch_and_max_len() -> None:
    eval_receipt = {"metrics": {"eff.peak_vram_mb": 694.02, "rank.candidates": 512}}
    out = eval_peak_vram(eval_receipt, None, max_len=96)
    assert out["fp32"].value == 694.02
    assert "eval batch 512" in out["fp32"].label
    assert "max_len 96" in out["fp32"].label
    assert "quantized" not in out


def test_eval_peak_vram_both_fp32_and_quantized() -> None:
    fp32 = {"metrics": {"eff.peak_vram_mb": 694.0}}
    quantized = {"metrics": {"eff.peak_vram_mb": 498.8}}
    out = eval_peak_vram(fp32, quantized)
    assert out["fp32"].value == 694.0
    assert out["quantized"].value == 498.8


def test_eval_peak_vram_missing_field_is_absent_not_zero() -> None:
    out = eval_peak_vram({"metrics": {}}, None)
    assert "fp32" not in out


# --------------------------------------------------------------------------- training_peak / budgets


def test_training_peak_none_when_batch_size_unknown() -> None:
    assert training_peak("code", None) is None


def test_training_peak_none_when_no_budget_file(tmp_path: Path) -> None:
    assert training_peak("code", 1280, root=tmp_path) is None


def test_training_peak_reads_a_real_budget_file(tmp_path: Path) -> None:
    region_dir = tmp_path / "code"
    region_dir.mkdir()
    (region_dir / "batch=1280.json").write_text(
        json.dumps(
            {"cell_id": "code-b1280-s1-7bc2699-20260904", "mib": 11594.0, "source": "report_peak"}
        )
    )
    mv = training_peak("code", 1280, max_len=96, root=tmp_path)
    assert mv is not None
    assert mv.value == 11594.0
    assert mv.unit == "MiB"
    assert "batch 1280" in mv.label
    assert "max_len 96" in mv.label
    assert "code-b1280-s1-7bc2699-20260904" in mv.label


def test_training_peak_refuses_when_source_is_not_report_peak(tmp_path: Path) -> None:
    """Mutation proof: a budget file whose `source` is NOT `report_peak` (an estimate,
    or a config-time projection that happens to live at the same path) must never be
    labelled MEASURED -- `training_peak` refuses outright rather than printing it."""
    region_dir = tmp_path / "code"
    region_dir.mkdir()
    (region_dir / "batch=1280.json").write_text(json.dumps({"mib": 99999.0, "source": "estimated"}))
    with pytest.raises(CardError, match="report_peak"):
        training_peak("code", 1280, root=tmp_path)


def test_training_peak_mutation_proof_restores_on_report_peak(tmp_path: Path) -> None:
    """Break it, see it fail; restore, see it pass again -- the guard is checking the
    field's value, not merely the file's existence."""
    region_dir = tmp_path / "code"
    region_dir.mkdir()
    path = region_dir / "batch=1280.json"

    path.write_text(json.dumps({"mib": 1.0, "source": "not_measured"}))
    with pytest.raises(CardError):
        training_peak("code", 1280, root=tmp_path)

    path.write_text(json.dumps({"mib": 1.0, "source": "report_peak"}))
    mv = training_peak("code", 1280, root=tmp_path)
    assert mv is not None and mv.value == 1.0


# --------------------------------------------------------------------------- build_size_report


def test_build_size_report_assembles_every_field(tmp_path: Path) -> None:
    region_dir = tmp_path / "code"
    region_dir.mkdir()
    (region_dir / "batch=1280.json").write_text(
        json.dumps({"cell_id": "cell-x", "mib": 11594.0, "source": "report_peak"})
    )
    train_receipt = {
        "parameters": 16021248,
        "config": {"batch_size": 1280, "max_len": 96, "holdout_pairs": 512},
    }
    quant_receipt = {
        "fp32_bytes": 64084992,
        "stored_bytes": 6519016,
        "width_histogram": {"3": 17},
        "quant.compression_ratio": 9.83,
    }
    eval_receipt = {"metrics": {"eff.peak_vram_mb": 694.0, "rank.candidates": 512}}
    eval_quantized_receipt = {"metrics": {"eff.peak_vram_mb": 498.8, "rank.candidates": 512}}

    report = build_size_report(
        region="code",
        train_receipt=train_receipt,
        quant_receipt=quant_receipt,
        eval_receipt=eval_receipt,
        eval_quantized_receipt=eval_quantized_receipt,
        budgets_root=tmp_path,
    )
    assert report.parameters == 16021248
    assert report.fp32_bytes == 64084992
    assert report.quantized_stored_bytes == 6519016
    assert report.compression_ratio == 9.83
    assert report.width_histogram == {"3": 17}
    assert report.eval_peak_vram["fp32"].value == 694.0
    assert report.eval_peak_vram["quantized"].value == 498.8
    assert report.training_peak is not None
    assert report.training_peak.value == 11594.0
    assert report.training_parameters is None  # same as deployed when no eval provenance


def test_build_size_report_visual_splits_deployed_from_training_count(tmp_path: Path) -> None:
    train_receipt = {"parameters": 22905216, "config": {"batch_size": 16}}
    eval_receipt = {"provenance": {"parameters": 10712448}}
    report = build_size_report(
        region="visual",
        train_receipt=train_receipt,
        eval_receipt=eval_receipt,
        budgets_root=tmp_path,
    )
    assert report.parameters == 10712448
    assert report.training_parameters == 22905216


def test_build_size_report_no_optional_receipts(tmp_path: Path) -> None:
    report = build_size_report(
        region="code", train_receipt={"parameters": 100}, budgets_root=tmp_path
    )
    assert report.parameters == 100
    assert report.fp32_bytes is None
    assert report.width_histogram is None
    assert report.eval_peak_vram == {}
    assert report.training_peak is None
