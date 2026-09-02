"""Tests for the model-agnostic pipeline receipt envelope.

The envelope's only job is to let one console read pipelines it knows nothing about. So
what is worth testing is that it survives shapes it was not designed against, and that it
refuses to call a run successful when nothing was measured.
"""

from __future__ import annotations

import json
from pathlib import Path

from cogsyndelta.pipeline.receipt import SCHEMA, Producer, Receipt, adapt, load_all


def test_no_gates_is_not_a_pass() -> None:
    """A run that measured nothing has demonstrated nothing.

    Treating an empty gate set as success is how an empty pipeline shows green.
    """
    assert not Receipt(Producer("p", "c"), "pretrain", metrics={"x": 1.0}).passed


def test_all_gates_must_hold() -> None:
    r = Receipt(Producer("p", "c"), "pretrain", gates={"a": True, "b": False})
    assert not r.passed
    r.gates["b"] = True
    assert r.passed


def test_adapts_a_foreign_architecture() -> None:
    """A ternary model's receipt must read without the envelope knowing what ternary is."""
    rec = Receipt(
        producer=Producer("tritter", "attn-block", "ternary"),
        stage="pretrain",
        metrics={"perplexity": 12.5, "bits_per_weight": 1.58},
        gates={"beats_baseline": True},
        detail={"ternary_sparsity": 0.42},
    )
    back = adapt(json.loads(json.dumps(rec.__dict__, default=lambda o: o.__dict__)), Path("x"))
    assert back is not None
    assert back.producer.architecture == "ternary"
    assert back.metrics["bits_per_weight"] == 1.58
    assert back.detail["ternary_sparsity"] == 0.42


def test_adapts_legacy_pretrain_receipt() -> None:
    """Receipts written before the envelope existed still have to be readable."""
    raw = {
        "region": "code",
        "held_out": {"recall@1": 0.95, "recall@10": 0.99},
        "untrained_baseline": {"recall@1": 0.40},
        "beats_untrained": {"recall@1": True, "recall@10": True},
        "parameters": 16_021_248,
        "elapsed_s": 1056.0,
    }
    rec = adapt(raw, Path("code-2026.json"))
    assert rec is not None
    assert rec.stage == "pretrain"
    assert rec.metrics["recall@1"] == 0.95
    assert rec.baseline["recall@1"] == 0.40
    assert rec.passed


def test_adapts_legacy_quant_receipt() -> None:
    raw = {
        "region": "code",
        "compression_ratio": 9.79,
        "quantized_metric": 0.955,
        "fp32_metric_recomputed": 0.959,
        "stored_bytes": 6_500_000,
        "drop": 0.004,
        "within_budget": True,
    }
    rec = adapt(raw, Path("code-quant.json"))
    assert rec is not None
    assert rec.stage == "quantize"
    assert rec.metrics["compression_ratio"] == 9.79
    assert rec.passed


def test_non_numeric_metrics_are_dropped_not_crashed() -> None:
    """A string where a number was expected must not take the whole console down."""
    raw = {
        "region": "x",
        "held_out": {"recall@1": 0.5, "note": "manual run", "flag": True},
        "untrained_baseline": {},
        "beats_untrained": {"recall@1": True},
    }
    rec = adapt(raw, Path("x.json"))
    assert rec is not None
    assert rec.metrics == {"recall@1": 0.5}


def test_unreadable_files_are_skipped(tmp_path: Path) -> None:
    """One malformed receipt must not hide the other forty."""
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "irrelevant.json").write_text('{"hello": "world"}')
    good = {
        "region": "code",
        "held_out": {"recall@1": 0.9},
        "untrained_baseline": {"recall@1": 0.4},
        "beats_untrained": {"recall@1": True},
    }
    (tmp_path / "good.json").write_text(json.dumps(good))
    found = load_all([tmp_path])
    assert len(found) == 1
    assert found[0].producer.component == "code"


def test_schema_constant_is_architecture_neutral() -> None:
    """The name is load-bearing: a CSD-specific schema invites CSD-specific readers."""
    assert "csd" not in SCHEMA.lower()
    assert "cogsyndelta" not in SCHEMA.lower()
