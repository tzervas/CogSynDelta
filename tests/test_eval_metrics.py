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
    spearman_correlation,
    token_weighted_perplexity,
)
from cogsyndelta.eval.metrics import (
    METRIC_ALIASES_V1,
    MetricGroup,
    MetricIdentity,
    assert_sameness,
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


def _identity(**overrides: object) -> MetricIdentity:
    """A baseline `MetricIdentity` every field of which matches its own defaults --
    tests mutate exactly one field at a time off this so a refusal can be pinned to it."""
    base: dict[str, object] = {
        "metrics_schema": "csd-metrics/v2",
        "corpus_fingerprint": "fp-code-holdout-abc123",
        "fingerprint_scheme": "csd-corpus-fp/v2",
        "battery_id": "eval_holdout",
        "k": None,
        "pooling": "pooled_both",
        "checkpoint_sha256": "127adeba58e39a1a0211e185adad74586d08b9fd0bdda8e2da4f5614f49ad8e1",
        "region": "code",
        "git_sha": "a7694090903664bc256b4b96d998b37cacd316cf",
        "seed": 0,
    }
    base.update(overrides)
    return MetricIdentity(**base)  # type: ignore[arg-type]


def _group(values: dict[str, float], **identity_overrides: object) -> MetricGroup:
    return MetricGroup(identity=_identity(**identity_overrides), values=values)


@pytest.mark.cpu
def test_compare_names_regressions_instead_of_averaging_them_away() -> None:
    """A change that improves one metric while degrading another is not a win.

    A blended score would report this case as an improvement, which is exactly how a
    quality regression gets shipped as an efficiency gain.
    """
    result = compare(
        _group({"perplexity": 30.0, "recall@1": 0.60}),
        _group({"perplexity": 28.0, "recall@1": 0.55}),
        lower_is_better={"perplexity"},
    )
    assert result["refused"] is False
    assert result["regressions"] == ["recall@1"]
    assert "regressed" in str(result["verdict"])
    assert result["metrics"]["perplexity"]["improved"] is True
    assert result["metrics"]["recall@1"]["improved"] is False


@pytest.mark.cpu
def test_compare_reports_a_clean_win() -> None:
    result = compare(
        _group({"perplexity": 30.0, "recall@1": 0.60}),
        _group({"perplexity": 28.0, "recall@1": 0.65}),
        lower_is_better={"perplexity"},
    )
    assert result["refused"] is False
    assert result["regressions"] == []
    assert result["verdict"] == "no regression"


@pytest.mark.cpu
def test_compare_matching_identity_with_k_none_on_both_sides_is_not_refused() -> None:
    """`k` is `None` for a metric with no `@k` (e.g. `mrr`). `None == None` must not, on
    its own, be read as a mismatch -- this is the explicit case the spec calls out."""
    result = compare(
        _group({"mrr": 0.90}, k=None),
        _group({"mrr": 0.95}, k=None),
        lower_is_better=set(),
    )
    assert result["refused"] is False


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("field", "candidate_override", "expected_receipt_name"),
    [
        ("metrics_schema", {"metrics_schema": "csd-metrics/v1"}, "metrics_schema"),
        ("corpus_fingerprint", {"corpus_fingerprint": "fp-different"}, "corpus.fingerprint"),
        (
            "fingerprint_scheme",
            {"fingerprint_scheme": "csd-corpus-fp/v1"},
            "corpus.fingerprint_scheme",
        ),
        ("battery_id", {"battery_id": "eval_quantized_holdout"}, "battery_id"),
        ("k", {"k": 10}, "k"),
        ("pooling", {"pooling": "anchor"}, "pooling"),
        (
            "checkpoint_sha256",
            {"checkpoint_sha256": "deadbeef" * 8},
            "artifacts.checkpoint_sha256",
        ),
        ("region", {"region": "retrieve"}, "region / producer.component"),
        ("git_sha", {"git_sha": "0" * 40}, "code_revision.git_sha"),
        ("seed", {"seed": 1}, "seed"),
    ],
)
def test_compare_refuses_on_each_identity_key_independently(
    field: str, candidate_override: dict[str, object], expected_receipt_name: str
) -> None:
    """MUTATION PROOF target: one test per identity key, each changing exactly ONE field
    off an otherwise-matching pair. If the refuse-predicate silently dropped a key (a
    field listed in `MetricIdentity` but never checked), the matching test for THAT field
    would pass with `refused: False` -- this is deliberately one test per key rather than
    a single loop-and-assert, so a broken single key fails its own test, not the whole
    suite as one undifferentiated failure.
    """
    result = compare(
        _group({"recall@1": 0.99}),
        _group({"recall@1": 0.99}, **candidate_override),
        lower_is_better=set(),
    )
    assert result["refused"] is True, f"expected a refusal when {field!r} differs"
    assert result["mismatched_key"] == expected_receipt_name
    assert result["reason"]  # non-empty, human-readable


@pytest.mark.cpu
def test_compare_names_the_first_mismatching_key_in_identity_order_not_alphabetical() -> None:
    """When several identity fields differ at once, the refusal names the FIRST one in
    `MetricIdentity`'s declared field order. `git_sha` sorts before `metrics_schema`
    alphabetically but `metrics_schema` is checked first -- this pins the check order
    against a future refactor that iterates the fields in a different sequence."""
    result = compare(
        _group({"recall@1": 0.99}),
        _group({"recall@1": 0.99}, metrics_schema="csd-metrics/v1", git_sha="0" * 40),
        lower_is_better=set(),
    )
    assert result["mismatched_key"] == "metrics_schema"


@pytest.mark.cpu
def test_compare_refuses_rather_than_diffing_shared_keys_across_batteries() -> None:
    """The v1 defect this replaces: a battery_id mismatch alone (e.g. a training
    `held_out.*` battery vs an eval `rank.*` battery) must refuse even though both sides
    happen to share the metric name `recall@1` and a plausible-looking value -- diffing
    shared keys across two different batteries is exactly what let a plan-vs-artifact
    quantized_metric get misread as comparable to an eval-quantized rank.recall@1
    (MM §4)."""
    result = compare(
        _group({"recall@1": 0.9902}, battery_id="train_holdout"),
        _group({"recall@1": 0.9902}, battery_id="eval_holdout"),
        lower_is_better=set(),
    )
    assert result["refused"] is True
    assert result["mismatched_key"] == "battery_id"
    assert "metrics" not in result


@pytest.mark.cpu
def test_assert_sameness_passes_the_real_map_equals_mrr_identity() -> None:
    """Grounded in a real receipt: `code-b1280-s1-7bc2699-20260904`'s eval-quantized
    receipt records `rank.map == rank.mrr == 0.9911115169525146` exactly, the single-
    relevant-item identity MM §3.4 documents."""
    assert_sameness("map==mrr", 0.9911115169525146, 0.9911115169525146)


@pytest.mark.cpu
def test_assert_sameness_passes_the_real_precision_equals_recall_over_k_identity() -> None:
    """Same receipt: `rank.precision@10 == 0.0994140625 == rank.recall@10 / 10
    (0.994140625 / 10)`."""
    assert_sameness("p@10==r@10/10", 0.0994140625, 0.994140625 / 10)


@pytest.mark.cpu
def test_assert_sameness_passes_the_real_plan_vs_artifact_recall_pair() -> None:
    """Grounded in the real quant + eval-quantized receipt pair for the same checkpoint
    sha (`127adeba...`): quant.plan_recall@1 (`quantized_metric` in the v1 quant receipt)
    and quant.artifact_recall@1 (`rank.recall@1` in the v1 eval-quantized receipt) agreed
    exactly, `0.98828125`. This is the pair MM §4 explicitly says the refuse-function must
    NOT reject -- and it does not, because this goes through `assert_sameness()`, never
    `compare()`."""
    sha = "127adeba58e39a1a0211e185adad74586d08b9fd0bdda8e2da4f5614f49ad8e1"
    assert_sameness(
        "quant.plan_recall@1 vs quant.artifact_recall@1",
        0.98828125,
        0.98828125,
        baseline_checkpoint_sha256=sha,
        candidate_checkpoint_sha256=sha,
    )


@pytest.mark.cpu
def test_assert_sameness_rejects_a_checkpoint_sha_mismatch_even_if_values_agree() -> None:
    """Two numbers that happen to be numerically equal are NOT a legitimate sameness pair
    if they were measured against different checkpoints -- MM §4's pairing is "on the same
    sha", not "on any two receipts with equal recall@1"."""
    with pytest.raises(ValueError, match="checkpoint sha differs"):
        assert_sameness(
            "quant.plan_recall@1 vs quant.artifact_recall@1",
            0.98828125,
            0.98828125,
            baseline_checkpoint_sha256="a" * 64,
            candidate_checkpoint_sha256="b" * 64,
        )


@pytest.mark.cpu
def test_assert_sameness_fails_when_the_identity_no_longer_holds() -> None:
    """MUTATION PROOF: this is the case the guard exists to catch -- a pairing that was
    supposed to be an identity but has drifted. Using the real `code` region's fp32 vs
    quantized MRR (`0.9922266602516174` vs `0.9911115169525146`) as the "drifted" values:
    these are NOT the blessed map==mrr pair (that compares within ONE receipt), so
    asserting sameness between them must fail."""
    with pytest.raises(ValueError, match="sameness guard failed"):
        assert_sameness("fp32 mrr vs quantized mrr", 0.9922266602516174, 0.9911115169525146)


@pytest.mark.cpu
def test_assert_sameness_tolerance_is_configurable_and_still_a_real_check() -> None:
    assert_sameness("within tolerance", 1.0, 1.0 + 1e-9)
    with pytest.raises(ValueError, match="sameness guard failed"):
        assert_sameness("within tolerance", 1.0, 1.1, tolerance=1e-6)
    assert_sameness("within tolerance", 1.0, 1.1, tolerance=0.2)


@pytest.mark.cpu
def test_metric_aliases_v1_map_the_examples_the_spec_names() -> None:
    """The three v1 names the unification-rules memo names explicitly by name (§3.1/§4),
    resolving to the v2 canonical dotted names this module and `cogsyndelta.eval.benchmark`
    now write."""
    assert METRIC_ALIASES_V1["effective_rank"] == "repr.effective_rank_entropy"
    assert METRIC_ALIASES_V1["emb_std"] == "repr.emb_std_anchor"
    assert METRIC_ALIASES_V1["quantized_metric"] == "quant.plan_recall@1"


@pytest.mark.cpu
def test_metric_aliases_v1_never_aliases_the_forbidden_effective_rank_pr_name() -> None:
    """`repr.effective_rank_pr` must never be invented -- it is not a valid v2 target for
    anything, so it must not appear anywhere in the alias table's values."""
    assert "repr.effective_rank_pr" not in METRIC_ALIASES_V1.values()


@pytest.mark.cpu
def test_spearman_is_monotone_not_linear() -> None:
    """Rank correlation must be blind to the scale a model happens to use.

    This is the whole reason the compress region is judged on Spearman rather than
    Pearson: a small encoder that ranks every pair correctly still squeezes its cosines
    into a narrow band, and Pearson would score that squeeze as a failure.
    """
    gold = [0.0, 0.25, 0.5, 0.75, 1.0]
    squeezed = [0.80, 0.81, 0.82, 0.83, 0.84]
    assert spearman_correlation(squeezed, gold) == pytest.approx(1.0)
    assert spearman_correlation(list(reversed(squeezed)), gold) == pytest.approx(-1.0)


@pytest.mark.cpu
def test_spearman_averages_tied_ranks() -> None:
    """Ties are most of the data, not an edge case.

    STS-B validation has 1500 pairs over 64 distinct scores; 139 of them are annotated
    0.0. Breaking those ties by array position invents an ordering the annotators never
    gave. Checked against the closed form for this case: with gold ranks [1.5, 1.5, 3, 4]
    against predicted ranks [1, 2, 3, 4], rho = 0.9486832980505138.
    """
    gold = [1.0, 1.0, 2.0, 3.0]
    predicted = [0.1, 0.2, 0.3, 0.4]
    assert spearman_correlation(predicted, gold) == pytest.approx(0.9486832980505138)

    # Position-based tie breaking would give exactly 1.0 here. It must not.
    assert spearman_correlation(predicted, gold) < 1.0


@pytest.mark.cpu
def test_spearman_reports_zero_for_a_collapsed_prediction() -> None:
    """A constant prediction is not a perfect correlation.

    A fully collapsed encoder scores every pair identically. Returning 0.0 rather than
    raising keeps a long training run alive so the collapse is recorded next to emb_std
    instead of aborting the run that was measuring it.
    """
    assert spearman_correlation([0.5] * 6, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) == 0.0


@pytest.mark.cpu
def test_spearman_refuses_input_it_cannot_correlate() -> None:
    with pytest.raises(ValueError, match="predictions vs"):
        spearman_correlation([0.1, 0.2], [0.1])
    with pytest.raises(ValueError, match="at least 2"):
        spearman_correlation([0.1], [0.1])
