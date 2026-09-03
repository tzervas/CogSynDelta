"""`cogsyndelta.eval.beir_fiqa`: the ported BEIR-style FiQA eval (DEC-09) and the five
pre-registered W4 gates (docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md, row W4).

Every gate function is a pure function over already-measured numbers -- no I/O, no model
-- so each one below is constructed BOTH to pass and to FAIL, per the row's own rule:
"Each gate must be constructible-to-fail in tests." The BM25-beats-the-model case (gate
c) gets special attention because it is the one the module docstring names by name:
"a BM25 that beats the model must FAIL gate (c)".
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("pyarrow", reason="train group not installed")

from cogsyndelta.eval import beir_fiqa

pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------------------
# BM25 -- ported verbatim from the parked module's own tests.
# ---------------------------------------------------------------------------------------


def test_bm25_ranks_the_matching_document_first() -> None:
    docs = [
        "the capital gains tax on a long term stock sale",
        "how to bake sourdough bread at home",
        "mortgage interest deduction and property tax",
    ]
    bm25 = beir_fiqa.BM25(docs)
    scores = bm25.scores("long term capital gains on a stock")
    assert int(scores.argmax()) == 0
    assert scores[1] < scores[0]


def test_bm25_scores_zero_for_out_of_vocabulary_query() -> None:
    bm25 = beir_fiqa.BM25(["alpha beta", "gamma delta"])
    assert float(bm25.scores("zzzz qqqq").sum()) == 0.0


def test_bm25_score_matrix_shape() -> None:
    bm25 = beir_fiqa.BM25(["alpha beta", "gamma delta", "beta gamma"])
    assert tuple(bm25.score_matrix(["alpha", "gamma"]).shape) == (2, 3)


# ---------------------------------------------------------------------------------------
# rank_metrics -- multi-relevant ranking.
# ---------------------------------------------------------------------------------------


def _toy_scores() -> tuple[torch.Tensor, list[list[int]]]:
    scores = torch.full((2, 12), -1.0)
    scores[0, 1] = 0.9  # an irrelevant document outranks both golds
    scores[0, 2] = 0.5  # best gold, rank 2
    scores[0, 3] = 0.2  # second gold, further down
    scores[1, 0] = 0.9  # only gold, rank 1
    return scores, [[2, 3], [0]]


def test_rank_metrics_scores_first_relevant_not_the_diagonal() -> None:
    scores, gold = _toy_scores()
    metrics = beir_fiqa.rank_metrics(scores, gold)
    assert metrics["recall@1"] == pytest.approx(0.5)
    assert metrics["recall@10"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(0.75)


def test_rank_metrics_skips_k_larger_than_the_pool() -> None:
    scores, gold = _toy_scores()
    assert "recall@100" not in beir_fiqa.rank_metrics(scores, gold)


def test_rank_metrics_rejects_a_query_with_no_relevant_document() -> None:
    scores, gold = _toy_scores()
    with pytest.raises(ValueError, match="at least one relevant"):
        beir_fiqa.rank_metrics(scores, [gold[0], []])


def test_rank_metrics_rejects_mismatched_gold_length() -> None:
    scores, gold = _toy_scores()
    with pytest.raises(ValueError, match="score rows"):
        beir_fiqa.rank_metrics(scores, gold[:1])


# ---------------------------------------------------------------------------------------
# Path resolution -- an unmounted export must name itself.
# ---------------------------------------------------------------------------------------


def test_missing_root_raises_clearly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="not present"):
        beir_fiqa.resolve_paths(tmp_path / "definitely-absent")


def test_unbuilt_split_names_the_builder(tmp_path) -> None:
    (tmp_path / "fiqa-pairs").mkdir()
    (tmp_path / "fiqa" / "corpus").mkdir(parents=True)
    paths = beir_fiqa.resolve_paths(tmp_path)
    with pytest.raises(FileNotFoundError, match="csd-fetch-fiqa-qrels"):
        paths.split("train")


def test_unknown_pool_rejected() -> None:
    with pytest.raises(ValueError, match="pool must be"):
        beir_fiqa.build_ranking_task("dev", pool="everything")


# ---------------------------------------------------------------------------------------
# Gate (a): matches or beats both parents on both parents' own gates.
# ---------------------------------------------------------------------------------------


def test_gate_a_passes_when_memory_matches_both_parents_exactly() -> None:
    result = beir_fiqa.gate_a_beats_both_parents(
        memory_held_out_recall_at_1=beir_fiqa.COMPRESS_PARENT_RECALL_AT_1,
        memory_graded_spearman=beir_fiqa.COMPRESS_PARENT_GRADED_SPEARMAN,
    )
    assert result["passed"] is True


def test_gate_a_fails_when_memory_recall_is_below_the_harder_parent() -> None:
    """Constructed to fail: recall@1 clears retrieve's floor but not compress's (the
    harder of the two), and the graded spearman clears compress's floor."""
    result = beir_fiqa.gate_a_beats_both_parents(
        memory_held_out_recall_at_1=beir_fiqa.RETRIEVE_PARENT_RECALL_AT_1,  # < compress's
        memory_graded_spearman=beir_fiqa.COMPRESS_PARENT_GRADED_SPEARMAN,
    )
    assert result["passed_recall"] is False
    assert result["passed"] is False


def test_gate_a_fails_when_memory_graded_spearman_regresses() -> None:
    """Constructed to fail: recall clears both parents, spearman does not clear compress's."""
    result = beir_fiqa.gate_a_beats_both_parents(
        memory_held_out_recall_at_1=beir_fiqa.COMPRESS_PARENT_RECALL_AT_1 + 0.05,
        memory_graded_spearman=beir_fiqa.COMPRESS_PARENT_GRADED_SPEARMAN - 0.10,
    )
    assert result["passed_graded"] is False
    assert result["passed"] is False


# ---------------------------------------------------------------------------------------
# Gate (b): recall@10 > 0.20 and MRR > 0.10 on the full pool.
# ---------------------------------------------------------------------------------------


def test_gate_b_passes_above_both_floors() -> None:
    result = beir_fiqa.gate_b_full_pool_thresholds({"recall@10": 0.30, "mrr": 0.15})
    assert result["passed"] is True


@pytest.mark.parametrize(
    "metrics",
    [
        {"recall@10": 0.10, "mrr": 0.15},  # recall too low
        {"recall@10": 0.30, "mrr": 0.05},  # mrr too low
        {},  # nothing measured
    ],
)
def test_gate_b_fails_below_either_floor(metrics: dict[str, float]) -> None:
    assert beir_fiqa.gate_b_full_pool_thresholds(metrics)["passed"] is False


# ---------------------------------------------------------------------------------------
# Gate (c): memory > BM25 on the same pool/qrels/code path.
# ---------------------------------------------------------------------------------------


def test_gate_c_passes_when_the_model_beats_bm25() -> None:
    result = beir_fiqa.gate_c_beats_bm25(
        full_pool_trained={"recall@10": 0.35}, full_pool_bm25={"recall@10": 0.25}
    )
    assert result["passed"] is True


def test_gate_c_fails_when_bm25_beats_the_model() -> None:
    """The module docstring's own example, constructed exactly: "a BM25 that beats the
    model must FAIL gate (c)"."""
    result = beir_fiqa.gate_c_beats_bm25(
        full_pool_trained={"recall@10": 0.22}, full_pool_bm25={"recall@10": 0.31}
    )
    assert result["passed"] is False


def test_gate_c_fails_on_an_exact_tie() -> None:
    """ "Beaten", not "matched or beaten" -- (a) uses `>=`, (c) deliberately does not."""
    result = beir_fiqa.gate_c_beats_bm25(
        full_pool_trained={"recall@10": 0.25}, full_pool_bm25={"recall@10": 0.25}
    )
    assert result["passed"] is False


# ---------------------------------------------------------------------------------------
# Gate (d): memory > its own random-init baseline on the full pool.
# ---------------------------------------------------------------------------------------


def test_gate_d_passes_when_trained_beats_untrained() -> None:
    result = beir_fiqa.gate_d_beats_random_init(
        full_pool_trained={"recall@10": 0.30}, full_pool_untrained={"recall@10": 0.02}
    )
    assert result["passed"] is True


def test_gate_d_fails_when_the_model_has_not_learned_anything() -> None:
    """Constructed to fail: a trained model no better than random init."""
    result = beir_fiqa.gate_d_beats_random_init(
        full_pool_trained={"recall@10": 0.02}, full_pool_untrained={"recall@10": 0.02}
    )
    assert result["passed"] is False


# ---------------------------------------------------------------------------------------
# Gate (e): §4.0's retrain gate, both clauses.
# ---------------------------------------------------------------------------------------


def test_gate_e_passes_with_a_healthy_rank_ratio_and_no_regression() -> None:
    result = beir_fiqa.gate_e_retrain_gate(
        token_global_pr_rank=40.0,
        pooled_pr_rank=15.0,  # ratio 2.67 >= 2.0
        memory_held_out_recall_at_1=beir_fiqa.COMPRESS_PARENT_RECALL_AT_1,
        memory_graded_spearman=beir_fiqa.COMPRESS_PARENT_GRADED_SPEARMAN,
    )
    assert result["pr_rank_clause"]["passed"] is True
    assert result["receipt_regression_clause"]["passed"] is True
    assert result["passed"] is True


def test_gate_e_fails_the_rank_clause_when_the_ratio_is_below_threshold() -> None:
    """Hand-fed arithmetic test of the clause's own ratio>=2.0 comparison, NOT a model
    of what a no-op L_token/L_decorr measurably leaves behind: the real W4 control arm
    (both terms OFF, docs/design/evidence/w4-control-arm-2026-09-03/) measures
    token_global_pr_rank/pooled_pr_rank = 2.0191x on the real corpus -- ABOVE this
    clause's threshold, not below it (see `_PR_RANK_CLAUSE_NOTE` in beir_fiqa.py; the
    clause is known NOT DISCRIMINATING at 50 steps for that reason). This test only
    checks that `gate_e_retrain_gate` correctly fails a ratio below 2.0 when handed one
    -- 16.0/15.0 = 1.07 is a value chosen to be below threshold, not a claim about what
    an untrained or no-op state actually measures."""
    result = beir_fiqa.gate_e_retrain_gate(
        token_global_pr_rank=16.0,
        pooled_pr_rank=15.0,  # ratio 1.07 < 2.0
        memory_held_out_recall_at_1=beir_fiqa.COMPRESS_PARENT_RECALL_AT_1,
        memory_graded_spearman=beir_fiqa.COMPRESS_PARENT_GRADED_SPEARMAN,
    )
    assert result["pr_rank_clause"]["passed"] is False
    assert result["passed"] is False


def test_gate_e_fails_the_regression_clause_when_the_receipt_metric_drops_too_far() -> None:
    """Constructed to fail: the rank clause clears, but recall@1 sits 2 points under
    compress's own -- outside the 1-point margin §4.0 sets."""
    result = beir_fiqa.gate_e_retrain_gate(
        token_global_pr_rank=40.0,
        pooled_pr_rank=15.0,
        memory_held_out_recall_at_1=beir_fiqa.COMPRESS_PARENT_RECALL_AT_1 - 0.02,
        memory_graded_spearman=beir_fiqa.COMPRESS_PARENT_GRADED_SPEARMAN,
    )
    assert result["receipt_regression_clause"]["passed"] is False
    assert result["passed"] is False


def test_gate_e_verify_the_gate_can_fail_against_a_pre_retrain_checkpoint() -> None:
    """§4.0's own instruction: "run it against the pre-retrain checkpoint and assert it
    reports FAIL -- the harness that says 0.66x today must still say FAIL when handed
    0.66x." A pre-retrain (untrained-for-the-token-terms) checkpoint is exactly W1's
    measured 0.66x-1.30x pooled/token-global ratio band -- i.e. token_global BELOW or
    barely above pooled, never at 2x. 0.78x (W1's `compress` ratio) is used here."""
    result = beir_fiqa.gate_e_retrain_gate(
        token_global_pr_rank=0.78 * 15.0,
        pooled_pr_rank=15.0,
        memory_held_out_recall_at_1=beir_fiqa.COMPRESS_PARENT_RECALL_AT_1,
        memory_graded_spearman=beir_fiqa.COMPRESS_PARENT_GRADED_SPEARMAN,
    )
    assert result["passed"] is False


# ---------------------------------------------------------------------------------------
# w4_gates -- the aggregate, read from a receipt-shaped dict.
# ---------------------------------------------------------------------------------------


def _passing_receipt() -> dict:
    return {
        "held_out": {"recall@1": beir_fiqa.COMPRESS_PARENT_RECALL_AT_1 + 0.01},
        "graded_held_out": {"spearman": beir_fiqa.COMPRESS_PARENT_GRADED_SPEARMAN + 0.01},
        "token_aware": {"final_block_rank": {"token_global_pr_rank": 40.0, "pooled_pr_rank": 15.0}},
    }


def test_w4_gates_all_pass_on_a_constructed_passing_case() -> None:
    gates = beir_fiqa.w4_gates(
        memory_receipt=_passing_receipt(),
        full_pool_trained={"recall@10": 0.30, "mrr": 0.15},
        full_pool_bm25={"recall@10": 0.22, "mrr": 0.09},
        full_pool_untrained={"recall@10": 0.01, "mrr": 0.02},
    )
    assert gates["passed"] is True
    assert all(gates[k]["passed"] for k in gates if k != "passed")


def test_w4_gates_overall_fails_when_only_bm25_wins() -> None:
    """One gate failing (c) must sink the conjunction even when the other four pass."""
    gates = beir_fiqa.w4_gates(
        memory_receipt=_passing_receipt(),
        full_pool_trained={"recall@10": 0.22, "mrr": 0.15},
        full_pool_bm25={"recall@10": 0.31, "mrr": 0.09},
        full_pool_untrained={"recall@10": 0.01, "mrr": 0.02},
    )
    assert gates["c_beats_bm25"]["passed"] is False
    assert gates["passed"] is False


def test_w4_gates_requires_graded_held_out() -> None:
    receipt = _passing_receipt()
    del receipt["graded_held_out"]
    with pytest.raises(KeyError, match="graded_held_out"):
        beir_fiqa.w4_gates(
            memory_receipt=receipt,
            full_pool_trained={"recall@10": 0.30, "mrr": 0.15},
            full_pool_bm25={"recall@10": 0.22, "mrr": 0.09},
            full_pool_untrained={"recall@10": 0.01, "mrr": 0.02},
        )


def test_w4_gates_requires_token_aware_final_block_rank() -> None:
    receipt = _passing_receipt()
    del receipt["token_aware"]
    with pytest.raises(KeyError, match="token_aware"):
        beir_fiqa.w4_gates(
            memory_receipt=receipt,
            full_pool_trained={"recall@10": 0.30, "mrr": 0.15},
            full_pool_bm25={"recall@10": 0.22, "mrr": 0.09},
            full_pool_untrained={"recall@10": 0.01, "mrr": 0.02},
        )
