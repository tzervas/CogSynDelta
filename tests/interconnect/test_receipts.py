"""Tests for `cogsyndelta.interconnect.receipts.ComposeReceipt` -- spec section 5, Table 9's
`receipts.py` row: "`ComposeReceipt` writes every Table 7 group and a receipt missing any
group fails a schema test."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cogsyndelta.interconnect.receipts import ComposeReceipt
from cogsyndelta.pipeline.receipt import SCHEMA as PIPELINE_SCHEMA
from cogsyndelta.pipeline.receipt import STAGES
from cogsyndelta.regions._receipt import METRICS_SCHEMA_V2

pytestmark = pytest.mark.cpu


def _identity(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "corpus_fingerprint": "f" * 64,
        "fingerprint_scheme": "csd-corpus-fingerprint/v1",
        "battery_id": "compose-dev/v1",
        "k": 32,
        "pooling": "frontal",
        "checkpoint_sha256": "c" * 64,
        "region": "white_matter",
        "seed": 0,
        "split_sha256": "s" * 64,
    }
    base.update(overrides)
    return base


def _fully_built(stage: str = "compose") -> ComposeReceipt:
    """A `ComposeReceipt` with every Table 7 group attached, ready to `build()`."""
    r = ComposeReceipt(stage, _identity())
    r.set_frozen_set(
        {
            "regions": [
                {
                    "name": "language",
                    "checkpoint_sha256": "a" * 64,
                    "receipt_path": "receipts/language-train-20260101T000000Z.json",
                    "status": "frozen",
                }
            ],
            "participants": ["language", "memory", "reasoning", "visual", "episodic_store"],
            "R": 5,
        }
    )
    r.set_budgets(
        {
            "B_read": 256,
            "B_kv": 3221225472,
            "eta": 0.15,
            "collapse_floor": {"expression": "eta/R", "R": 5, "value": 0.03},
            "token_budget": {"language": {"min": 1, "default": 64, "max": 96}},
            "ctx_min": {"language": 8},
            "ctx_max": {"language": 96},
            "b": {"language": 52},
            "ctx": {"language": 96},
        }
    )
    r.set_composed_metric(
        {
            "recall_at_1": {"overall": 0.71, "bins": {"general": 0.75}},
            "null_recall": 0.60,
            "null_fpr": {"code": 0.01},
            "pair_deltas": {},
        }
    )
    r.set_baselines(
        {
            "B0": {"value": 0.10, "source": "graded"},
            "B1": {"value": 0.50, "source": "graded"},
        }
    )
    r.set_attention_mass(
        {
            "mean_per_region": {"language": 0.22},
            "histogram_per_region": {"language": [0.1, 0.2, 0.3, 0.4]},
            "collapsed_in_phase_A": [],
            "a_store": 0.05,
        }
    )
    r.set_write_back_topology(
        {
            "enabled": False,
            "own_bin_delta": {"language": 0.0},
            "topology": {"agreement": None, "status": "not demonstrated"},
        }
    )
    r.set_scheduler(
        {
            "rho": 0.7,
            "flops_ratio": 0.45,
            "sparse_rel_delta": 0.01,
            "phase_d_vs_c": None,
            "s_stats": {},
            "mean_iters": 3.2,
            "halt_at_histogram": {"1": 0, "2": 1, "3": 2, "4": 5},
            "budget_as_tag_gap": 0.0,
        }
    )
    r.set_latency(
        {
            "wall_ms_measured": 812.0,
            "first_token_ms_measured": None,
            "speech_frame_miss_fraction": None,
        }
    )
    r.set_store(
        {"b_store": 8, "capacity_bytes": {"5080": 0}, "occupancy_bytes": 0, "active": "floor"}
    )
    r.set_verdicts({"integration": "holds", "scheduling": "n/a", "trigger_sensitivity": "n/a"})
    r.set_placement_knobs({"placement": {}, "knobs": {}})
    return r


# ---------------------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------------------


def test_stage_must_be_compose_or_schedule() -> None:
    with pytest.raises(ValueError, match="stage"):
        ComposeReceipt("pretrain", _identity())


def test_schedule_is_a_member_of_pipeline_stages() -> None:
    """This lane's own addition to `pipeline/receipt.py` (spec section 4)."""
    assert "schedule" in STAGES
    ComposeReceipt("schedule", _identity())  # does not raise


def test_identity_missing_a_required_key_raises() -> None:
    identity = _identity()
    del identity["split_sha256"]
    with pytest.raises(ValueError, match="split_sha256"):
        ComposeReceipt("compose", identity)


# ---------------------------------------------------------------------------------------
# Group completeness -- "a receipt missing any group fails a schema test"
# ---------------------------------------------------------------------------------------


def test_build_refuses_with_no_groups_attached() -> None:
    r = ComposeReceipt("compose", _identity())
    with pytest.raises(ValueError, match="missing required Table 7 groups"):
        r.build()


def test_build_refuses_with_exactly_one_group_missing() -> None:
    r = _fully_built()
    r._groups.pop("verdicts")  # simulate the one omitted set_* call
    with pytest.raises(ValueError, match="verdicts"):
        r.build()


def test_set_method_rejects_a_payload_missing_its_own_keys() -> None:
    r = ComposeReceipt("compose", _identity())
    with pytest.raises(ValueError, match="R"):
        r.set_frozen_set({"regions": [], "participants": []})


# ---------------------------------------------------------------------------------------
# build() shape
# ---------------------------------------------------------------------------------------


def test_build_produces_the_full_envelope_and_every_group() -> None:
    receipt = _fully_built().build()
    assert receipt["schema"] == PIPELINE_SCHEMA
    assert receipt["stage"] == "compose"
    assert receipt["metrics_schema"] == METRICS_SCHEMA_V2
    assert receipt["corpus"]["fingerprint"] == "f" * 64
    assert receipt["artifacts"]["checkpoint_sha256"] == "c" * 64
    assert receipt["split"]["sha256"] == "s" * 64
    for group in (
        "frozen_set",
        "budgets",
        "composed_metric",
        "baselines",
        "attention_mass",
        "write_back_topology",
        "scheduler",
        "latency",
        "store",
        "verdicts",
        "placement_knobs",
    ):
        assert group in receipt


def test_build_is_deterministic() -> None:
    """Table 9 "all" row's determinism requirement, applied to receipt assembly."""
    assert _fully_built().build() == _fully_built().build()


def test_edges_is_omitted_unless_the_caller_supplied_it() -> None:
    receipt = _fully_built().build()
    assert "edges" not in receipt["write_back_topology"]

    with_edges = _fully_built()
    with_edges._groups["write_back_topology"]["edges"] = [{"from": "language", "to": "memory"}]
    assert "edges" in with_edges.build()["write_back_topology"]


# ---------------------------------------------------------------------------------------
# write() -- stamps code_revision and metrics_schema through write_receipt
# ---------------------------------------------------------------------------------------


def test_write_stamps_code_revision_and_writes_json(tmp_path: Path) -> None:
    path = _fully_built().write(tmp_path, "cogsyndelta-white_matter-compose-test.json")
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk["metrics_schema"] == METRICS_SCHEMA_V2
    assert "code_revision" in on_disk
    assert "git_sha" in on_disk["code_revision"]


def test_write_refuses_when_code_revision_capture_returns_nothing(tmp_path: Path) -> None:
    """Mirrors `write_receipt`'s own refusal test -- proven to fire through this method too."""
    with pytest.raises(RuntimeError, match="code_revision"):
        _fully_built().write(
            tmp_path,
            "should-not-be-written.json",
            capture=lambda _root: None,
        )
    assert not (tmp_path / "should-not-be-written.json").exists()
