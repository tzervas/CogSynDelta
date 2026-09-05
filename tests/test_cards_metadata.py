"""Tests for `cogsyndelta.cards.metadata`.

Covers: `region_repo_tags` (base tags, `csd-ptq-v1` only when quantized);
`build_eval_results` (one `EvalResult` per numeric metric, non-numeric skipped,
`verified=False` on every one, empty list for no eval receipt, RETIRED rank metrics
dropped not refused, an undocumented metric raises `CardError`); `build_card_data`
(front matter round-trips through `ModelCard(...).data`, model-index present with the
receipt's own metric names, `composed` kind has no `region:` tag).
"""

from __future__ import annotations

import pytest
from huggingface_hub import ModelCard

from cogsyndelta.cards.metadata import (
    build_card_data,
    build_eval_results,
    datasets_from_train_receipt,
    region_repo_tags,
)
from cogsyndelta.cards.methodology import METRIC_METHODOLOGY, CardError

pytestmark = pytest.mark.cpu


def test_datasets_from_train_receipt_converts_shard_landings() -> None:
    rec = {
        "corpus": {
            "shards": [
                "nyuuzyou__pxhere/processed/20260905T035325Z/train.zip",
                "facebookresearch__clevr/processed/20260905T063733Z/train.zip",
                "nyuuzyou__pxhere/processed/20260905T035325Z/train.zip",
            ]
        }
    }
    assert datasets_from_train_receipt(rec) == [
        "nyuuzyou/pxhere",
        "facebookresearch/clevr",
    ]


def test_datasets_from_train_receipt_none_when_absent() -> None:
    assert datasets_from_train_receipt(None) is None
    assert datasets_from_train_receipt({}) is None
    assert datasets_from_train_receipt({"corpus": {}}) is None


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


def test_build_eval_results_retired_rank_metrics_dropped_not_refused() -> None:
    """`rank.map`/`rank.precision@10` are RETIRED (METRICS-METHODOLOGY.md Sec 13) as
    independently displayed values -- a model-index `EvalResult` IS that, so these two
    must never appear here, the same drop `cogsyndelta.cards.tables.build_eval_tables`
    already applies to its rank table (mirrors
    `test_build_eval_tables_retired_rank_metrics_dropped_not_refused` in
    `tests/test_cards_tables.py`). Dropped, not refused: a legacy receipt carrying them
    must still render a model-index for its other metrics."""
    receipt = {
        "metrics": {
            "rank.recall@1": 0.49,
            "rank.map": 0.9922266602516174,
            "rank.precision@10": 0.099609375,
        }
    }
    results = build_eval_results(region="code", eval_receipt=receipt)
    types = {r.metric_type for r in results}
    assert types == {"rank.recall@1"}
    assert "rank.map" not in types
    assert "rank.precision@10" not in types


def test_build_eval_results_undocumented_metric_raises_card_error() -> None:
    """The front matter must not publish a number the card's own tables would refuse
    to print -- `require_documented` runs here exactly as it does in every
    `cogsyndelta.cards.tables.build_*` function."""
    receipt = {"metrics": {"rank.totally_made_up_metric": 1.0}}
    with pytest.raises(CardError, match=r"totally_made_up_metric"):
        build_eval_results(region="code", eval_receipt=receipt)


def test_build_eval_results_undocumented_metric_mutation_proof_stubbed_methodology() -> None:
    """A render that succeeds with the real table must fail once the table is emptied
    -- proves `require_documented` is load-bearing on this path, not merely present."""
    receipt = {"metrics": {"rank.recall@1": 0.5}}
    build_eval_results(region="code", eval_receipt=receipt)  # sanity: succeeds
    with pytest.raises(CardError):
        build_eval_results(region="code", eval_receipt=receipt, methodology={})
    assert METRIC_METHODOLOGY  # the real, un-stubbed table is untouched


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


def test_build_card_data_model_index_drops_retired_rank_metrics() -> None:
    """Integration-level counterpart of the `build_eval_results` unit test above: the
    same drop must hold through `build_card_data`, the function `render.py` actually
    calls, not only the lower-level helper."""
    receipt = {
        "metrics": {
            "rank.recall@1": 0.49,
            "rank.map": 0.99,
            "rank.precision@10": 0.1,
        }
    }
    card_data = build_card_data(
        kind="region_variant", region="code", tier="mit", eval_receipt=receipt
    )
    assert card_data.eval_results is not None
    metric_types = {r.metric_type for r in card_data.eval_results}
    assert metric_types == {"rank.recall@1"}


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
