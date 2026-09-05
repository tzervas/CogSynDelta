"""Tests for `cogsyndelta.cards.tables`.

Covers: `metrics_schema_of`/`assert_schemas_agree` (agreement, disagreement ->
CardError with both stamps named, all-v1-fallback agreement, mutation proof that the
refusal is load-bearing); `build_eval_tables` (v1 legacy key mapped to its v2 name and
marked, retired `rank.map`/`rank.precision@10` dropped rather than raising,
undocumented-metric refusal, category grouping, `heading_suffix`, comparator columns);
`build_training_table`/`build_gate_table`/`build_quant_table`; `_best_column`'s
lower-is-better exception list.
"""

from __future__ import annotations

import pytest

from cogsyndelta.cards.methodology import CardError
from cogsyndelta.cards.tables import (
    V1_LEGACY_SCHEMA,
    MetricRow,
    _best_column,
    assert_schemas_agree,
    build_eval_tables,
    build_gate_table,
    build_quant_table,
    build_training_table,
    metrics_schema_of,
    render_table_markdown,
)

pytestmark = pytest.mark.cpu


def _eval_receipt(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "producer": {"project": "cogsyndelta", "component": "compress"},
        "stage": "eval",
        "metrics": {
            "rank.recall@1": 0.49,
            "repr.anisotropy": 0.1306,
            "repr.effective_rank_ratio": 0.459,
        },
        "gates": {"beats_untrained": True, "not_anisotropic": True, "uses_its_dimensions": True},
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- schema agreement


def test_metrics_schema_of_none_receipt_is_none() -> None:
    assert metrics_schema_of(None) is None


def test_metrics_schema_of_unstamped_receipt_is_v1_fallback() -> None:
    assert metrics_schema_of({"metrics": {}}) == V1_LEGACY_SCHEMA


def test_metrics_schema_of_stamped_receipt_is_its_own_stamp() -> None:
    assert metrics_schema_of({"metrics_schema": "csd-metrics/v2"}) == "csd-metrics/v2"


def test_assert_schemas_agree_all_unstamped() -> None:
    """Every real fixture receipt in this project predates the stamp -- the common
    case must agree with itself under the v1 fallback, not spuriously refuse."""
    schema = assert_schemas_agree({"train": {}, "eval": {}, "quant": {}})
    assert schema == V1_LEGACY_SCHEMA


def test_assert_schemas_agree_all_stamped_v2() -> None:
    schema = assert_schemas_agree(
        {
            "train": {"metrics_schema": "csd-metrics/v2"},
            "eval": {"metrics_schema": "csd-metrics/v2"},
        }
    )
    assert schema == "csd-metrics/v2"


def test_assert_schemas_agree_ignores_none_entries() -> None:
    schema = assert_schemas_agree({"train": {"metrics_schema": "csd-metrics/v2"}, "eval": None})
    assert schema == "csd-metrics/v2"


def test_assert_schemas_agree_raises_on_disagreement_naming_both() -> None:
    with pytest.raises(CardError) as exc_info:
        assert_schemas_agree({"train": {}, "eval": {"metrics_schema": "csd-metrics/v2"}})
    msg = str(exc_info.value)
    assert "csd-metrics/v1" in msg
    assert "csd-metrics/v2" in msg
    assert "train=" in msg
    assert "eval=" in msg


def test_assert_schemas_agree_raises_on_no_receipts() -> None:
    with pytest.raises(CardError):
        assert_schemas_agree({"train": None, "eval": None})


def test_assert_schemas_agree_mutation_proof() -> None:
    """The refusal is load-bearing: two receipts that plainly disagree must not slip
    through as 'agreeing' under any code path."""
    agreeing = {"a": {"metrics_schema": "X"}, "b": {"metrics_schema": "X"}}
    assert assert_schemas_agree(agreeing) == "X"
    disagreeing = {"a": {"metrics_schema": "X"}, "b": {"metrics_schema": "Y"}}
    with pytest.raises(CardError):
        assert_schemas_agree(disagreeing)


# --------------------------------------------------------------------------- build_eval_tables


def test_build_eval_tables_none_receipt_returns_empty() -> None:
    assert build_eval_tables(eval_receipt=None) == []


def test_build_eval_tables_groups_by_category() -> None:
    tables = build_eval_tables(eval_receipt=_eval_receipt())
    categories = {t.category for t in tables}
    assert categories == {"rank", "repr"}


def test_build_eval_tables_v1_legacy_key_mapped_and_marked() -> None:
    """`repr.effective_rank_ratio` (v1) must render under `repr.effective_rank_entropy_ratio`
    (v2), marked `is_v1_mapped=True` -- CARD SPEC: 'v1 receipt renders through the alias
    map with an explicit footnote'."""
    tables = build_eval_tables(eval_receipt=_eval_receipt())
    repr_table = next(t for t in tables if t.category == "repr")
    keys = {r.key: r for r in repr_table.rows}
    assert "repr.effective_rank_entropy_ratio" in keys
    assert "repr.effective_rank_ratio" not in keys  # v1 bare key dropped once mapped
    assert keys["repr.effective_rank_entropy_ratio"].is_v1_mapped is True
    # anisotropy is a plain v2-shaped key already -- never marked v1
    assert keys["repr.anisotropy"].is_v1_mapped is False


def test_build_eval_tables_retired_rank_metrics_dropped_not_refused() -> None:
    """`rank.map`/`rank.precision@10` are RETIRED, not undocumented -- a legacy
    receipt carrying them must render (dropped), never raise CardError for them."""
    receipt = _eval_receipt(
        metrics={
            "rank.recall@1": 0.49,
            "rank.map": 0.49,  # == recall@1 by construction pre-retirement
            "rank.precision@10": 0.1,
        }
    )
    tables = build_eval_tables(eval_receipt=receipt)
    rank_table = next(t for t in tables if t.category == "rank")
    keys = {r.key for r in rank_table.rows}
    assert "rank.recall@1" in keys
    assert "rank.map" not in keys
    assert "rank.precision@10" not in keys


def test_build_eval_tables_undocumented_metric_raises_card_error() -> None:
    receipt = _eval_receipt(metrics={"rank.totally_made_up_metric": 1.0})
    with pytest.raises(CardError, match=r"totally_made_up_metric"):
        build_eval_tables(eval_receipt=receipt)


def test_build_eval_tables_baseline_column_from_baseline_receipt() -> None:
    receipt = _eval_receipt()
    baseline = _eval_receipt(metrics={"rank.recall@1": 0.05})
    tables = build_eval_tables(eval_receipt=receipt, baseline_eval_receipt=baseline)
    rank_table = next(t for t in tables if t.category == "rank")
    row = next(r for r in rank_table.rows if r.key == "rank.recall@1")
    assert row.variant == 0.49
    assert row.baseline == 0.05


def test_build_eval_tables_comparator_columns() -> None:
    receipt = _eval_receipt()
    other = _eval_receipt(metrics={"rank.recall@1": 0.6, "repr.anisotropy": 0.2})
    tables = build_eval_tables(eval_receipt=receipt, comparators={"b512-s2": other})
    rank_table = next(t for t in tables if t.category == "rank")
    row = next(r for r in rank_table.rows if r.key == "rank.recall@1")
    assert row.comparators == {"b512-s2": 0.6}


def test_build_eval_tables_heading_suffix() -> None:
    tables = build_eval_tables(eval_receipt=_eval_receipt(), heading_suffix=" (fp32)")
    assert all(t.heading.endswith(" (fp32)") for t in tables)


# --------------------------------------------------------------------------- other table builders


def test_build_training_table_from_held_out_and_baseline() -> None:
    train_receipt = {
        "held_out": {"n_pairs": 512, "recall@1": 0.707, "recall@10": 0.9199},
        "untrained_baseline": {"n_pairs": 512, "recall@1": 0.037, "recall@10": 0.068},
    }
    table = build_training_table(train_receipt)
    assert table.category == "held_out"
    keys = {r.key: r for r in table.rows}
    assert keys["recall@1"].variant == 0.707
    assert keys["recall@1"].baseline == 0.037


def test_build_training_table_undocumented_key_raises() -> None:
    with pytest.raises(CardError):
        build_training_table({"held_out": {"totally_unknown_field": 1.0}})


def test_build_gate_table_none_or_empty_returns_none() -> None:
    assert build_gate_table(None) is None
    assert build_gate_table({"gates": {}}) is None


def test_build_gate_table_v1_gate_name_mapped() -> None:
    table = build_gate_table({"gates": {"beats_untrained": True}})
    assert table is not None
    row = table.rows[0]
    assert row.key == "beats_untrained_eval"
    assert row.is_v1_mapped is True


def test_build_quant_table_none_returns_none() -> None:
    assert build_quant_table(None) is None


def test_build_quant_table_reads_only_the_known_fields() -> None:
    quant = {
        "fp32_metric_recomputed": 0.496,
        "quant.plan_recall@1": 0.488,
        "quant.drop_recall@1": 0.0078,
        "tolerance": 0.01,
        "within_budget": True,
        "quant.compression_ratio": 9.83,
        "fp32_bytes": 1000,
        "stored_bytes": 100,
        "some_unrelated_field": "ignored",
    }
    table = build_quant_table(quant)
    assert table is not None
    keys = {r.key for r in table.rows}
    assert keys == {
        "fp32_metric_recomputed",
        "quant.plan_recall@1",
        "quant.drop_recall@1",
        "tolerance",
        "within_budget",
        "quant.compression_ratio",
        "fp32_bytes",
        "stored_bytes",
    }


# --------------------------------------------------------------------------- best-column bolding


def test_best_column_higher_is_better_default() -> None:
    row = MetricRow(key="recall@1", variant=0.9, baseline=0.1)
    assert _best_column(row) == "variant"


def test_best_column_lower_is_better_for_latency() -> None:
    row = MetricRow(
        key="eff.latency_p50_ms",
        variant=5.0,
        baseline=1.0,
        comparators={"other": 2.0},  # gitleaks:allow
    )
    assert _best_column(row) == "baseline"


def test_best_column_none_for_boolean_row() -> None:
    row = MetricRow(key="within_budget", variant=True, baseline=True)
    assert _best_column(row) is None


def test_best_column_none_when_fewer_than_two_values() -> None:
    row = MetricRow(key="recall@1", variant=0.9)
    assert _best_column(row) is None


def test_render_table_markdown_bolds_the_best_value() -> None:
    from cogsyndelta.cards.methodology import METRIC_METHODOLOGY
    from cogsyndelta.cards.tables import CATEGORY_HEADINGS, MetricTable

    row = MetricRow(key="rank.recall@1", variant=0.9, baseline=0.1)
    table = MetricTable(category="rank", heading=CATEGORY_HEADINGS["rank"], rows=[row])
    m = METRIC_METHODOLOGY["recall@1"]
    fn_key = (m.definition, m.battery_id, m.pooling, m.source)
    md = render_table_markdown(table, footnote_numbers={fn_key: 1})
    assert "**0.9**" in md
    assert "0.1" in md and "**0.1**" not in md
