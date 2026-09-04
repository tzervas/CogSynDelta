"""Tests for `cogsyndelta.cards.metadata`.

Covers: `region_repo_tags` (base tags, `csd-ptq-v1` only when quantized);
`build_eval_results` (one `EvalResult` per numeric metric, non-numeric skipped,
`verified=False` on every one, empty list for no eval receipt); `build_card_data`
(front matter round-trips through `ModelCard(...).data`, model-index present with the
receipt's own metric names, `composed` kind has no `region:` tag).
"""

from __future__ import annotations

import pytest
from huggingface_hub import ModelCard

from cogsyndelta.cards.metadata import build_card_data, build_eval_results, region_repo_tags

pytestmark = pytest.mark.cpu


def test_region_repo_tags_base() -> None:
    assert region_repo_tags("code", quantized=False) == ["cogsyndelta", "region:code"]


def test_region_repo_tags_quantized_adds_csd_ptq_v1() -> None:
    assert region_repo_tags("code", quantized=True) == ["cogsyndelta", "region:code", "csd-ptq-v1"]


def test_build_eval_results_none_receipt_is_empty() -> None:
    assert build_eval_results(region="code", eval_receipt=None) == []


def test_build_eval_results_one_per_numeric_metric() -> None:
    receipt = {"metrics": {"rank.recall@1": 0.99, "repr.anisotropy": 0.003, "not_a_number": "x"}}
    results = build_eval_results(region="code", eval_receipt=receipt)
    types = {r.metric_type for r in results}
    assert types == {"rank.recall@1", "repr.anisotropy"}  # non-numeric skipped


def test_build_eval_results_boolean_metrics_are_skipped() -> None:
    """`isinstance(True, int)` is True in Python -- a boolean metric (there are none
    today, but `gates` dicts are booleans) must not silently become `metric_value=1.0`."""
    receipt = {"metrics": {"some_flag": True, "rank.recall@1": 0.5}}
    results = build_eval_results(region="code", eval_receipt=receipt)
    types = {r.metric_type for r in results}
    assert types == {"rank.recall@1"}


def test_build_eval_results_every_result_is_unverified() -> None:
    receipt = {"metrics": {"rank.recall@1": 0.5}}
    results = build_eval_results(region="code", eval_receipt=receipt)
    assert all(r.verified is False for r in results)


def test_build_eval_results_dataset_name_default_and_override() -> None:
    receipt = {"metrics": {"rank.recall@1": 0.5}}
    default = build_eval_results(region="code", eval_receipt=receipt)
    assert default[0].dataset_name == "cogsyndelta-code-holdout"
    overridden = build_eval_results(region="code", eval_receipt=receipt, dataset_name="custom-set")
    assert overridden[0].dataset_name == "custom-set"


def test_build_card_data_front_matter_round_trips() -> None:
    receipt = {"metrics": {"rank.recall@1": 0.99}}
    card_data = build_card_data(
        kind="region_variant", region="code", tier="mit", eval_receipt=receipt
    )
    yaml_text = card_data.to_yaml()
    card = ModelCard(f"---\n{yaml_text}\n---\n\nbody\n")
    assert card.data.license == "mit"
    assert card.data.pipeline_tag == "feature-extraction"
    assert "region:code" in card.data.tags


def test_build_card_data_model_index_present_with_v2_names() -> None:
    receipt = {"metrics": {"rank.recall@1": 0.99, "quant.plan_recall@1": 0.98}}
    card_data = build_card_data(
        kind="region_variant", region="code", tier="mit", eval_receipt=receipt
    )
    assert card_data.eval_results is not None
    metric_types = {r.metric_type for r in card_data.eval_results}
    assert "rank.recall@1" in metric_types
    assert "quant.plan_recall@1" in metric_types


def test_build_card_data_no_eval_receipt_has_no_model_index() -> None:
    card_data = build_card_data(kind="placeholder", region="stream_vae", tier="mit")
    assert card_data.eval_results is None


def test_build_card_data_composed_kind_has_no_region_tag() -> None:
    card_data = build_card_data(kind="composed", region="cogsyndelta", tier="cc-by-nc-sa-4.0")
    assert card_data.tags == ["cogsyndelta"]
    assert card_data.model_name == "cogsyndelta"


def test_build_card_data_quantized_tag() -> None:
    card_data = build_card_data(kind="region_variant", region="code", tier="mit", quantized=True)
    assert "csd-ptq-v1" in card_data.tags


def test_build_card_data_datasets_field_omitted_when_unknown() -> None:
    card_data = build_card_data(kind="region_variant", region="code", tier="mit")
    assert card_data.datasets is None


def test_build_card_data_datasets_field_set_when_known() -> None:
    card_data = build_card_data(
        kind="region_variant", region="code", tier="mit", datasets=["some/corpus"]
    )
    assert card_data.datasets == ["some/corpus"]
