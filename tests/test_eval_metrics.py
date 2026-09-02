"""Evaluation metric tests.

The constraint these serve: a change must give a net gain "without decreasing quality
and/or increasing perplexity". That is only checkable against an uncontaminated baseline,
so the contamination guard is tested harder than the metrics themselves.
"""

from __future__ import annotations

import math

import pytest
import torch

from cogsyndelta.eval import (
    assert_no_contamination,
    compare,
    contamination_report,
    mean_reciprocal_rank,
    recall_at_k,
    representation_std,
    token_weighted_perplexity,
)


@pytest.mark.cpu
def test_contamination_guard_catches_reformatted_duplicates() -> None:
    """Contamination usually arrives as a reformatted copy, not a byte-identical one.

    A guard that only catches exact matches gives false confidence, which is worse than
    no guard: it makes an inflated number look audited.
    """
    train = ["The cat sat on the mat.", "Robert Boulter is an English actor."]
    reformatted = ["  the   CAT sat on the MAT.  "]

    report = contamination_report(train, reformatted)
    assert report["overlap"] == 1
    assert report["eval_fraction_contaminated"] == 1.0

    with pytest.raises(ValueError, match="contaminated"):
        assert_no_contamination(train, reformatted)


@pytest.mark.cpu
def test_clean_split_passes_and_reports_counts() -> None:
    train = ["alpha beta", "gamma delta"]
    held_out = ["epsilon zeta"]
    report = assert_no_contamination(train, held_out)
    assert report["overlap"] == 0
    assert report["train_unique"] == 2
    assert report["eval_unique"] == 1


@pytest.mark.cpu
def test_tolerance_must_be_deliberate_not_a_workaround() -> None:
    """A non-zero tolerance still fails when exceeded, so it cannot be used to wave a
    genuinely contaminated set through."""
    train = ["a b c", "d e f"]
    evalset = ["a b c", "x y z"]  # 50% contaminated
    with pytest.raises(ValueError):
        assert_no_contamination(train, evalset, tolerance=0.1)
    assert_no_contamination(train, evalset, tolerance=0.5)


@pytest.mark.cpu
def test_perplexity_is_token_weighted_not_batch_averaged() -> None:
    """A short trailing batch must not carry the weight of a full one.

    With losses [2.0, 3.0] over [1000, 10] tokens the correct answer is ~7.46; averaging
    the batch losses gives ~12.18, a 63% error driven purely by batch shape.
    """
    got = token_weighted_perplexity([2.0, 3.0], [1000, 10])
    naive = math.exp((2.0 + 3.0) / 2)
    assert abs(got - math.exp((2.0 * 1000 + 3.0 * 10) / 1010)) < 1e-9
    assert got < naive * 0.7


@pytest.mark.cpu
def test_perplexity_refuses_an_empty_set() -> None:
    """Reporting a perplexity over zero tokens would be a number with no referent."""
    with pytest.raises(ValueError, match="no tokens"):
        token_weighted_perplexity([], [])
    with pytest.raises(ValueError, match="losses vs"):
        token_weighted_perplexity([1.0], [1, 2])


@pytest.mark.cpu
def test_ranking_metrics() -> None:
    scores = torch.tensor([[0.1, 0.9, 0.3], [0.8, 0.2, 0.5]])
    relevant = torch.tensor([1, 0])
    assert recall_at_k(scores, relevant, 1) == 1.0
    assert mean_reciprocal_rank(scores, relevant) == 1.0

    worst = torch.tensor([[0.9, 0.1]])
    assert recall_at_k(worst, torch.tensor([1]), 1) == 0.0
    assert mean_reciprocal_rank(worst, torch.tensor([1])) == 0.5


@pytest.mark.cpu
def test_representation_std_detects_collapse() -> None:
    """The collapse signal, shared by JEPA and any embedding model."""
    collapsed = torch.ones(8, 16)
    assert representation_std(collapsed) < 1e-6
    assert representation_std(torch.randn(8, 16)) > 0.1
    with pytest.raises(ValueError, match="at least 2"):
        representation_std(torch.randn(1, 16))


@pytest.mark.cpu
def test_compare_names_regressions_instead_of_averaging_them_away() -> None:
    """A change that improves one metric while degrading another is not a win.

    A blended score would report this case as an improvement, which is exactly how a
    quality regression gets shipped as an efficiency gain.
    """
    result = compare(
        {"perplexity": 30.0, "recall@1": 0.60},
        {"perplexity": 28.0, "recall@1": 0.55},
        lower_is_better={"perplexity"},
    )
    assert result["regressions"] == ["recall@1"]
    assert "regressed" in str(result["verdict"])
    assert result["metrics"]["perplexity"]["improved"] is True
    assert result["metrics"]["recall@1"]["improved"] is False


@pytest.mark.cpu
def test_compare_reports_a_clean_win() -> None:
    result = compare(
        {"perplexity": 30.0, "recall@1": 0.60},
        {"perplexity": 28.0, "recall@1": 0.65},
        lower_is_better={"perplexity"},
    )
    assert result["regressions"] == []
    assert result["verdict"] == "no regression"
