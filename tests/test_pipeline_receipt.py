"""Tests for the model-agnostic pipeline receipt envelope.

The envelope's only job is to let one console read pipelines it knows nothing about. So
what is worth testing is that it survives shapes it was not designed against, and that it
refuses to call a run successful when nothing was measured.
"""

from __future__ import annotations

import json
from pathlib import Path

from cogsyndelta.pipeline.receipt import (
    METRICS_SCHEMA,
    METRICS_SCHEMA_V1_LEGACY,
    SCHEMA,
    Producer,
    Receipt,
    adapt,
    load_all,
)


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


def test_adapts_pretrain_receipt_carries_checkpoint_sha256() -> None:
    """R9: a receipt written after checkpoint fingerprinting carries a content hash of
    the exact file `checkpoint` names, not just the path -- so a reader is not trusting
    the path alone, and a checkpoint silently swapped or truncated on disk after the
    receipt was written no longer passes as a match. `adapt()` has to carry that hash
    through into `artifacts['checkpoint_sha256']` for it to reach a reader at all.
    Verified by mutation: deleting the `checkpoint_sha256` line from the pretrain branch
    of `adapt()` left every test in this file green before this one was added."""
    raw = {
        "region": "code",
        "held_out": {"recall@1": 0.95},
        "untrained_baseline": {"recall@1": 0.40},
        "beats_untrained": {"recall@1": True},
        "checkpoint": "/akula-data/csd/receipts/code-checkpoints/deadbeef/final.pt",
        "checkpoint_sha256": "a" * 64,
    }
    rec = adapt(raw, Path("code-2026.json"))
    assert rec is not None
    assert rec.artifacts["checkpoint"] == raw["checkpoint"]
    assert rec.artifacts["checkpoint_sha256"] == "a" * 64


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
    # No metrics_schema key on this raw receipt at all -- predates the g7 stamp, so
    # this must read as the legacy table, never silently default to the current one.
    assert rec.metrics_schema == METRICS_SCHEMA_V1_LEGACY


def test_adapts_v2_quant_receipt_under_its_new_field_names() -> None:
    """`scripts/csd-quantize.py` writes these three names now (g7 §3.1); `adapt()`
    must land the same values in the envelope's own generic `metrics` keys as the v1
    branch above does -- an architecture-agnostic reader's own naming stays stable
    across a CSD-internal rename."""
    raw = {
        "region": "code",
        "metrics_schema": "csd-metrics/v2",
        "quant.compression_ratio": 9.79,
        "quant.plan_recall@1": 0.955,
        "fp32_metric_recomputed": 0.959,
        "stored_bytes": 6_500_000,
        "quant.drop_recall@1": 0.004,
        "within_budget": True,
    }
    rec = adapt(raw, Path("code-quant.json"))
    assert rec is not None
    assert rec.stage == "quantize"
    assert rec.metrics["compression_ratio"] == 9.79
    assert rec.metrics["metric"] == 0.955
    assert rec.metrics["drop"] == 0.004
    assert rec.passed
    assert rec.metrics_schema == METRICS_SCHEMA == "csd-metrics/v2"


def test_a_fresh_receipt_defaults_to_the_current_metrics_schema() -> None:
    assert Receipt(Producer("p", "c"), "eval").metrics_schema == METRICS_SCHEMA


def test_adapts_envelope_receipt_carries_its_own_metrics_schema_through() -> None:
    raw = {
        "schema": SCHEMA,
        "producer": {"project": "cogsyndelta", "component": "code"},
        "stage": "eval",
        "metrics": {"rank.recall@1": 0.9},
        "metrics_schema": "csd-metrics/v2",
    }
    rec = adapt(raw, Path("x.json"))
    assert rec is not None
    assert rec.metrics_schema == "csd-metrics/v2"


def test_adapts_envelope_receipt_with_no_metrics_schema_reads_as_legacy() -> None:
    """An eval/eval-quantized receipt written before the g7 stamp existed has no
    `metrics_schema` key at all -- must not be mistaken for a v2 receipt just because
    its envelope `schema` is still `model-pipeline-receipt/v1` (that field is
    unrelated; see the `Receipt.metrics_schema` docstring)."""
    raw = {
        "schema": SCHEMA,
        "producer": {"project": "cogsyndelta", "component": "code"},
        "stage": "eval",
        "metrics": {"rank.recall@1": 0.9},
    }
    rec = adapt(raw, Path("x.json"))
    assert rec is not None
    assert rec.metrics_schema == METRICS_SCHEMA_V1_LEGACY
    assert rec.metrics_schema != METRICS_SCHEMA


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


def _console():
    """Load the console script, which has no .py extension."""
    import importlib.util as u
    from importlib.machinery import SourceFileLoader

    loader = SourceFileLoader("console", "scripts/model-pipeline-console")
    spec = u.spec_from_loader("console", loader)
    mod = u.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_prometheus_escapes_label_values() -> None:
    """An unescaped quote makes the WHOLE scrape unparseable, not just one line.

    Component names are free text from a producer, so they are escaped rather than
    trusted.
    """
    c = _console()
    assert c._escape('a"b') == 'a\\"b'
    assert c._escape("a\\b") == "a\\\\b"
    assert c._escape("a\nb") == "a\\nb"


def test_prometheus_puts_metric_name_in_a_label() -> None:
    """`recall@1` is not a legal Prometheus metric name, and minting one series per
    metric name would make the schema change whenever a new architecture reports
    something novel."""
    c = _console()
    rec = Receipt(
        producer=Producer("tritter", "attn", "ternary"),
        stage="pretrain",
        metrics={"recall@1": 0.5, "bits_per_weight": 1.58},
        gates={"ok": True},
    )
    text = c.prometheus_text([rec])
    assert 'metric="recall@1"' in text
    assert 'metric="bits_per_weight"' in text
    assert "model_pipeline_metric{" in text
    # The architecture must survive into the labels, or per-architecture queries break.
    assert 'architecture="ternary"' in text


def test_prometheus_exports_only_the_newest_run_per_stage() -> None:
    """Re-exporting past runs would rewrite history on every scrape with the scrape's
    own timestamp. VictoriaMetrics keeps the history; the exporter reports current state."""
    c = _console()
    newer = Receipt(
        Producer("p", "c"),
        "pretrain",
        metrics={"m": 2.0},
        gates={"g": True},
        started_utc="2026-09-02T10:00:00Z",
    )
    older = Receipt(
        Producer("p", "c"),
        "pretrain",
        metrics={"m": 1.0},
        gates={"g": True},
        started_utc="2026-09-01T10:00:00Z",
    )
    text = c.prometheus_text([newer, older])  # load_all yields newest first
    series = [ln for ln in text.splitlines() if ln.startswith("model_pipeline_metric{")]
    assert len(series) == 1
    assert series[0].endswith(" 2.0")


def test_prometheus_skips_non_numeric_metrics() -> None:
    """A bool or string in metrics must not emit a line Prometheus cannot parse."""
    c = _console()
    rec = Receipt(
        Producer("p", "c"),
        "eval",
        metrics={"good": 1.0, "flag": True, "note": "text"},  # type: ignore[dict-item]
        gates={"g": True},
    )
    text = c.prometheus_text([rec])
    assert 'metric="good"' in text
    assert 'metric="flag"' not in text
    assert 'metric="note"' not in text


# --------------------------------------------------------------------- code provenance
#
# H1. Training and quantization receipts have carried a `code_revision` block since
# `regions/_receipt.write_receipt` existed -- that helper REFUSES to write without one.
# Eval receipts, which go through `Receipt.write` instead, carried none, so the matrix
# harness's G5b gate ("receipt code_revision.git_sha == run.code.sha and dirty == false")
# could not fire on the `eval-quantized` receipt whose entire purpose is to prove a
# published artifact was scored by known code. These tests pin both halves: the stamp,
# and the refusal when the capture mechanism is broken.

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_write_stamps_code_revision_from_the_same_helper_training_receipts_use() -> None:
    """The block must be real -- this checkout's actual sha -- and must reach the file,
    not just the in-memory object."""
    import tempfile

    from cogsyndelta.regions._receipt import capture_code_revision

    rec = Receipt(
        producer=Producer("cogsyndelta", "memory", "dense-transformer"),
        stage="eval",
        kind="eval-quantized",
        metrics={"rank.recall@1": 0.85},
        provenance={"eval_target": "quantized"},
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = rec.write(Path(tmp))
        on_disk = json.loads(path.read_text())

    assert set(rec.code_revision) == {"git_sha", "dirty", "branch", "describe"}
    assert on_disk["code_revision"] == rec.code_revision
    # Not a fabricated block: it agrees with what the helper reports for this tree.
    assert rec.code_revision["git_sha"] == capture_code_revision(_REPO_ROOT)["git_sha"]
    assert rec.code_revision["git_sha"] != "unknown"
    assert isinstance(rec.code_revision["dirty"], bool)


def test_a_stamped_eval_quant_receipt_satisfies_the_harness_g5b_shape() -> None:
    """G5b reads `code_revision.git_sha` and `code_revision.dirty` off the receipt after
    the harness's own `adapt` (which copies unknown top-level keys through). Assert the
    two fields it needs are present and typed, on the exact receipt kind it gates."""
    import tempfile

    rec = Receipt(
        producer=Producer("cogsyndelta", "memory", "dense-transformer"),
        stage="eval",
        kind="eval-quantized",
        provenance={"eval_target": "quantized"},
    )
    with tempfile.TemporaryDirectory() as tmp:
        raw = json.loads(rec.write(Path(tmp)).read_text())

    assert isinstance(raw["code_revision"]["git_sha"], str)
    assert isinstance(raw["code_revision"]["dirty"], bool)


def test_write_refuses_when_the_capture_mechanism_returns_nothing(tmp_path: Path) -> None:
    """MUTATION. The real `capture_code_revision` never returns falsy -- it degrades to
    an honest "unknown" block -- so a falsy return means capture itself is broken. The
    receipt must not reach disk: this is the same fail-closed contract
    `regions/_receipt.write_receipt` has had all along."""
    import pytest

    rec = Receipt(producer=Producer("cogsyndelta", "memory"), stage="eval", kind="eval")
    out_dir = tmp_path / "receipts"

    with pytest.raises(RuntimeError, match="code_revision"):
        rec.write(out_dir, capture=lambda repo_root: None)

    assert not out_dir.exists() or list(out_dir.iterdir()) == []
    assert rec.code_revision == {}


def test_write_refuses_on_an_empty_capture_too(tmp_path: Path) -> None:
    """`{}` is exactly as falsy as `None`, and exactly as broken."""
    import pytest

    rec = Receipt(producer=Producer("cogsyndelta", "memory"), stage="eval", kind="eval")
    with pytest.raises(RuntimeError, match="code_revision"):
        rec.write(tmp_path / "receipts", capture=lambda repo_root: {})


def test_a_receipt_without_code_revision_fails_the_gate_that_needs_it() -> None:
    """MUTATION, from the reader's side: reproduce the pre-fix payload (no
    `code_revision` key at all) and assert the G5b-shaped check cannot be satisfied --
    which is what "the gate could not fire" meant in practice."""
    import tempfile

    rec = Receipt(producer=Producer("cogsyndelta", "memory"), stage="eval", kind="eval")
    with tempfile.TemporaryDirectory() as tmp:
        raw = json.loads(rec.write(Path(tmp)).read_text())
    del raw["code_revision"]

    assert raw.get("code_revision", {}).get("git_sha") is None
