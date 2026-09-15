"""Tests for the "Representation geometry after quantization" row group
(`cogsyndelta.cards.tables.build_quant_geometry_table`) and its wiring into
`render_card`.

Kept as its own file rather than folded into `tests/test_cards_render.py` /
`tests/test_cards_tables.py`: this feature's fixtures (an eval-quantized receipt with
and without the `quant.geometry.*` fields) are self-contained and do not need those
files' larger `make_*_receipt` machinery. `tests/test_cards_render.py`'s golden
snapshot never supplies an `eval_quantized` receipt (confirmed by reading
`_full_fixture_receipts`), so it is untouched by this feature and needed no update.
"""

from __future__ import annotations

import pytest

from cogsyndelta.cards.methodology import CardError
from cogsyndelta.cards.render import render_card
from cogsyndelta.cards.tables import (
    LEXICAL_NOT_MEASURED,
    QUANT_GEOMETRY_KEYS,
    build_eval_tables,
    build_quant_geometry_table,
)

pytestmark = pytest.mark.cpu

_GEOMETRY_VALUES: dict[str, float] = {
    "quant.geometry.mean_cosine": 0.952436,
    "quant.geometry.min_cosine": 0.914011,
    "quant.geometry.p05_cosine": 0.93,
    "quant.geometry.nn_agreement_at_10": 0.9085,
    "quant.geometry.latent_std_ratio": 1.060542,
}


def _eval_quantized_receipt(*, with_geometry: bool) -> dict[str, object]:
    metrics: dict[str, object] = {"quant.artifact_probe_top1": 0.7243, "repr.rep_std": 0.4017}
    if with_geometry:
        metrics.update(_GEOMETRY_VALUES)
    return {
        "producer": {"project": "cogsyndelta", "component": "visual"},
        "stage": "eval",
        "kind": "eval-quantized",
        "metrics": metrics,
        "provenance": {"eval_target": "quantized"},
    }


def test_build_quant_geometry_table_none_receipt_returns_none() -> None:
    assert build_quant_geometry_table(None) is None


def test_build_quant_geometry_table_populated() -> None:
    table = build_quant_geometry_table(_eval_quantized_receipt(with_geometry=True))
    assert table is not None
    assert table.category == "quant_geometry"
    assert table.heading == "Representation geometry after quantization"
    values = {row.key: row.variant for row in table.rows}
    assert values == _GEOMETRY_VALUES
    assert [row.key for row in table.rows] == list(QUANT_GEOMETRY_KEYS)


def test_build_quant_geometry_table_absent_fields_read_not_measured() -> None:
    """The fail-closed-for-cards half: a receipt written before this feature (or one
    where the fp32 reference could not be established) must show every row as
    `"not measured"`, never a number invented in its place, and never a silently
    omitted row."""
    table = build_quant_geometry_table(_eval_quantized_receipt(with_geometry=False))
    assert table is not None
    assert len(table.rows) == len(QUANT_GEOMETRY_KEYS)
    for row in table.rows:
        assert row.variant == LEXICAL_NOT_MEASURED, row.key


def test_build_quant_geometry_table_partial_fields_mix_measured_and_not() -> None:
    receipt = _eval_quantized_receipt(with_geometry=True)
    del receipt["metrics"]["quant.geometry.min_cosine"]  # type: ignore[index]
    table = build_quant_geometry_table(receipt)
    assert table is not None
    by_key = {row.key: row.variant for row in table.rows}
    assert by_key["quant.geometry.min_cosine"] == LEXICAL_NOT_MEASURED
    assert by_key["quant.geometry.mean_cosine"] == pytest.approx(0.952436)


def test_build_quant_geometry_table_ignores_a_non_numeric_stray_value() -> None:
    """A stray non-numeric value under a geometry key (should never happen in a real
    receipt, but this table must not crash or render it as a number) reads as
    "not measured" rather than propagating garbage to the card."""
    receipt = _eval_quantized_receipt(with_geometry=True)
    receipt["metrics"]["quant.geometry.mean_cosine"] = "oops"  # type: ignore[index]
    table = build_quant_geometry_table(receipt)
    assert table is not None
    by_key = {row.key: row.variant for row in table.rows}
    assert by_key["quant.geometry.mean_cosine"] == LEXICAL_NOT_MEASURED


def test_build_quant_geometry_table_requires_documented_methodology() -> None:
    """Mutation proof: an empty methodology table must refuse to render these keys,
    same discipline every other card table refuses undocumented metrics under."""
    with pytest.raises(CardError, match=r"quant\.geometry"):
        build_quant_geometry_table(_eval_quantized_receipt(with_geometry=True), methodology={})


def test_generic_eval_tables_never_double_print_geometry_keys() -> None:
    """`build_eval_tables` (the generic per-present-key `quant` grouping) must not
    ALSO print these five keys -- they belong exclusively to the dedicated row group,
    or a card would show every geometry number twice under two different headings."""
    tables = build_eval_tables(eval_receipt=_eval_quantized_receipt(with_geometry=True))
    printed_keys = {row.key for table in tables for row in table.rows}
    assert printed_keys.isdisjoint(QUANT_GEOMETRY_KEYS)
    # The non-geometry quant.* field is still there, unaffected by the exclusion.
    assert "quant.artifact_probe_top1" in printed_keys


_REGION_CFG: dict[str, object] = {
    "kind": "i-jepa",
    "modality": "vision",
    "role": "test",
    "router_trigger": "test",
    "licence_tier": "mit",
    "licence_why": "test fixture",
}


def test_render_card_prints_the_geometry_section_when_eval_quantized_is_supplied() -> None:
    card = render_card(
        "region_variant",
        region="visual",
        region_cfg=_REGION_CFG,
        receipts={"eval_quantized": _eval_quantized_receipt(with_geometry=True)},
        files={},
    )
    assert "### Representation geometry after quantization" in card
    assert "`quant.geometry.mean_cosine`" in card
    assert "0.952436" in card


def test_render_card_prints_not_measured_when_eval_quantized_lacks_geometry() -> None:
    card = render_card(
        "region_variant",
        region="visual",
        region_cfg=_REGION_CFG,
        receipts={"eval_quantized": _eval_quantized_receipt(with_geometry=False)},
        files={},
    )
    assert "### Representation geometry after quantization" in card
    assert card.count(LEXICAL_NOT_MEASURED) >= len(QUANT_GEOMETRY_KEYS)


def test_render_card_omits_the_geometry_section_with_no_eval_quantized_receipt() -> None:
    card = render_card(
        "region_variant",
        region="visual",
        region_cfg=_REGION_CFG,
        receipts={},
        files={},
    )
    assert "Representation geometry after quantization" not in card
