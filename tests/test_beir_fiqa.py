"""`cogsyndelta.eval.beir_fiqa`: the ported BEIR-style FiQA eval (DEC-09) and the five
pre-registered W4 gates (docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md, row W4).

Every gate function is a pure function over already-measured numbers -- no I/O, no model
-- so each one below is constructed BOTH to pass and to FAIL, per the row's own rule:
"Each gate must be constructible-to-fail in tests." The BM25-beats-the-model case (gate
c) gets special attention because it is the one the module docstring names by name:
"a BM25 that beats the model must FAIL gate (c)".
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

pytest.importorskip("pyarrow", reason="train group not installed")

from cogsyndelta.eval import beir_fiqa

pytestmark = pytest.mark.cpu

_REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_gate_e_stamps_the_canonical_token_pr_rank_ratio_name() -> None:
    """§3.1's canonical name, additive beside the existing `ratio` key -- same value,
    plus the formula string, never a silent rename that could break an existing reader
    of `pr_rank_clause["ratio"]`."""
    result = beir_fiqa.gate_e_retrain_gate(
        token_global_pr_rank=40.0,
        pooled_pr_rank=15.0,
        memory_held_out_recall_at_1=beir_fiqa.COMPRESS_PARENT_RECALL_AT_1,
        memory_graded_spearman=beir_fiqa.COMPRESS_PARENT_GRADED_SPEARMAN,
    )
    clause = result["pr_rank_clause"]
    assert clause["token.pr_rank_ratio"] == pytest.approx(clause["ratio"])
    assert clause["token.pr_rank_ratio"] == pytest.approx(40.0 / 15.0)
    assert clause["token.pr_rank_ratio_formula"] == "token_global_pr_rank / pooled_pr_rank"


# ---------------------------------------------------------------------------------------
# `to_beir_metrics` -- explicit `beir.*` names, pooling/battery_id provenance.
# ---------------------------------------------------------------------------------------


def test_to_beir_metrics_prefixes_recall_and_mrr_for_the_corpus_pool() -> None:
    out = beir_fiqa.to_beir_metrics("corpus", {"recall@1": 0.4, "recall@10": 0.6, "mrr": 0.5})
    assert out["beir.recall@1"] == pytest.approx(0.4)
    assert out["beir.recall@10"] == pytest.approx(0.6)
    assert out["beir.mrr"] == pytest.approx(0.5)
    assert out["pooling"] == "fiqa_corpus"
    assert out["battery_id"] == "beir_fiqa_corpus"


def test_to_beir_metrics_tags_the_split_pool_distinctly() -> None:
    out = beir_fiqa.to_beir_metrics("split", {"recall@10": 0.6, "mrr": 0.5})
    assert out["pooling"] == "fiqa_split"
    assert out["battery_id"] == "beir_fiqa_split"
    # Distinct from the corpus pool's tags -- a schema-v2 reader must never equate them.
    corpus_out = beir_fiqa.to_beir_metrics("corpus", {"recall@10": 0.6, "mrr": 0.5})
    assert out["pooling"] != corpus_out["pooling"]
    assert out["battery_id"] != corpus_out["battery_id"]


def test_to_beir_metrics_drops_non_metric_keys_like_bm25_index_s() -> None:
    out = beir_fiqa.to_beir_metrics("corpus", {"recall@10": 0.3, "mrr": 0.2, "index_s": 1.7})
    assert "beir.index_s" not in out
    assert "index_s" not in out


def test_to_beir_metrics_never_writes_a_fake_ndcg() -> None:
    """§4's placeholder rule: `beir.ndcg@10` is NOT implemented here -- never fabricated."""
    out = beir_fiqa.to_beir_metrics("corpus", {"recall@1": 0.4, "recall@10": 0.6, "mrr": 0.5})
    assert "beir.ndcg@10" not in out


def test_to_beir_metrics_rejects_an_unknown_pool() -> None:
    with pytest.raises(ValueError, match="pool must be"):
        beir_fiqa.to_beir_metrics("everything", {"recall@10": 0.3, "mrr": 0.1})


# ---------------------------------------------------------------------------------------
# `refuse_cross_family_rank_ratio` / `token_rank_surfaces` -- explicit `token.*` names,
# and the PR-vs-entropy conflation guard.
# ---------------------------------------------------------------------------------------

# Frozen W1 fixture (docs/design/evidence/w1-token-rank-2026-09-02/results.json,
# regions.code.regions.trained -- read 2026-09-04, never regenerated): the two rank
# families disagree in SIGN on this project's own production `code` region. PR ratio
# 28.091115489593022 / 42.633552623237634 ~= 0.659 (< 1); entropy ratio
# 138.874267578125 / 114.9809799194336 ~= 1.208 (> 1).
_W1_CODE_TRAINED_POOLED_PR_RANK = 42.633552623237634
_W1_CODE_TRAINED_GLOBAL_PR_RANK = 28.091115489593022
_W1_CODE_TRAINED_POOLED_ENTROPY_RANK = 114.9809799194336
_W1_CODE_TRAINED_GLOBAL_ENTROPY_RANK = 138.874267578125

_W1_FIXTURE = _REPO_ROOT / "docs/design/evidence/w1-token-rank-2026-09-02/results.json"


def test_w1_pin_matches_the_frozen_fixture_it_was_transcribed_from() -> None:
    """The four `_W1_CODE_TRAINED_*` constants above are transcribed, not read, from
    `_W1_FIXTURE` -- transcribed so the sign-disagreement regression below has no
    filesystem dependency of its own. That transcription could silently drift from the
    frozen file (a typo, someone regenerating the fixture without updating the pin);
    this test is the one place that would catch it."""
    if not _W1_FIXTURE.is_file():
        pytest.skip(f"frozen W1 fixture not present in this checkout: {_W1_FIXTURE}")
    trained = json.loads(_W1_FIXTURE.read_text())["regions"]["code"]["regions"]["trained"]
    assert trained["pooled_pr_rank"] == _W1_CODE_TRAINED_POOLED_PR_RANK
    assert trained["token_global_pr_rank"] == _W1_CODE_TRAINED_GLOBAL_PR_RANK
    assert trained["pooled_entropy_rank"] == _W1_CODE_TRAINED_POOLED_ENTROPY_RANK
    assert trained["token_global_entropy_rank"] == _W1_CODE_TRAINED_GLOBAL_ENTROPY_RANK


def test_w1_sign_disagreement_regression_pr_and_entropy_ratios_stay_separate() -> None:
    """Regression pin: on the frozen W1 `code` fixture, PR ratio < 1 while entropy ratio
    > 1 -- METRICS-METHODOLOGY.md §9's sign-disagreement, reproduced from
    `token_rank_surfaces`'s own output so a future change that quietly merges or
    reorders the two families is caught here, not only in the design doc's prose."""
    surfaces = beir_fiqa.token_rank_surfaces(
        {
            "pooled_pr_rank": _W1_CODE_TRAINED_POOLED_PR_RANK,
            "token_global_pr_rank": _W1_CODE_TRAINED_GLOBAL_PR_RANK,
            "pooled_entropy_rank": _W1_CODE_TRAINED_POOLED_ENTROPY_RANK,
            "token_global_entropy_rank": _W1_CODE_TRAINED_GLOBAL_ENTROPY_RANK,
            "n_tokens": 26757.0,
        }
    )
    pr_ratio = surfaces["token.pr_rank_ratio"]
    entropy_ratio = (
        surfaces["global"]["token.global_entropy_rank"]
        / surfaces["pooled"]["token.pooled_entropy_rank"]
    )
    assert pr_ratio < 1.0
    assert entropy_ratio > 1.0
    # The two families never collapsed into one field: `token.pr_rank_ratio` exists,
    # there is no `token.entropy_rank_ratio` counterpart written by this function (an
    # entropy ratio, if one is ever needed, is the caller's own computation on the two
    # `*_entropy_rank` fields -- never this function's `token.pr_rank_ratio`).
    assert "token.entropy_rank_ratio" not in surfaces


def test_token_rank_surfaces_uses_the_explicit_canonical_names() -> None:
    surfaces = beir_fiqa.token_rank_surfaces(
        {
            "pooled_pr_rank": 15.0,
            "token_global_pr_rank": 40.0,
            "pooled_entropy_rank": 100.0,
            "token_global_entropy_rank": 130.0,
            "n_tokens": 5000.0,
        }
    )
    assert surfaces["pooled"]["token.pooled_pr_rank"] == pytest.approx(15.0)
    assert surfaces["pooled"]["token.pooled_entropy_rank"] == pytest.approx(100.0)
    assert surfaces["pooled"]["pooling"] == "anchor_pooled"
    assert surfaces["pooled"]["battery_id"] == "train_token_rank"
    assert surfaces["global"]["token.global_pr_rank"] == pytest.approx(40.0)
    assert surfaces["global"]["token.global_entropy_rank"] == pytest.approx(130.0)
    assert surfaces["global"]["pooling"] == "anchor_token_global"
    assert surfaces["global"]["battery_id"] == "train_token_rank"
    assert surfaces["token.pr_rank_ratio"] == pytest.approx(40.0 / 15.0)
    assert surfaces["n_tokens"] == pytest.approx(5000.0)
    # Distinct pooling tags between the pooled and token-global groups -- never merged.
    assert surfaces["pooled"]["pooling"] != surfaces["global"]["pooling"]


def test_token_rank_surfaces_pooled_zero_gives_nan_ratio_not_a_false_zero() -> None:
    surfaces = beir_fiqa.token_rank_surfaces(
        {
            "pooled_pr_rank": 0.0,
            "token_global_pr_rank": 12.0,
            "pooled_entropy_rank": 0.0,
            "token_global_entropy_rank": 12.0,
            "n_tokens": 0.0,
        }
    )
    assert math.isnan(surfaces["token.pr_rank_ratio"])


def test_refuse_cross_family_rank_ratio_accepts_matching_pr_family() -> None:
    ratio = beir_fiqa.refuse_cross_family_rank_ratio(
        "token.global_pr_rank", 40.0, "token.pooled_pr_rank", 15.0
    )
    assert ratio == pytest.approx(40.0 / 15.0)


def test_refuse_cross_family_rank_ratio_accepts_matching_entropy_family() -> None:
    ratio = beir_fiqa.refuse_cross_family_rank_ratio(
        "token.global_entropy_rank", 130.0, "token.pooled_entropy_rank", 100.0
    )
    assert ratio == pytest.approx(1.3)


def test_refuse_cross_family_rank_ratio_mutation_proof_a_reader_mixing_families_is_refused() -> (
    None
):
    """Mutation proof: construct the exact defect the guard exists to catch -- a "reader"
    that takes a `_pr_rank` numerator over an `_entropy_rank` denominator (or vice versa)
    -- and assert it is refused rather than silently returning a number. Both directions
    of the mistake are checked; a guard that only caught one direction would still let
    half of this conflation through."""
    with pytest.raises(ValueError, match="PR-rank and entropy-rank"):
        beir_fiqa.refuse_cross_family_rank_ratio(
            "token.global_pr_rank",
            _W1_CODE_TRAINED_GLOBAL_PR_RANK,
            "token.pooled_entropy_rank",
            _W1_CODE_TRAINED_POOLED_ENTROPY_RANK,
        )
    with pytest.raises(ValueError, match="PR-rank and entropy-rank"):
        beir_fiqa.refuse_cross_family_rank_ratio(
            "token.global_entropy_rank",
            _W1_CODE_TRAINED_GLOBAL_ENTROPY_RANK,
            "token.pooled_pr_rank",
            _W1_CODE_TRAINED_POOLED_PR_RANK,
        )


def test_refuse_cross_family_rank_ratio_rejects_an_unrecognised_field_name() -> None:
    with pytest.raises(ValueError, match="not a recognised token-rank field"):
        beir_fiqa.refuse_cross_family_rank_ratio(
            "token.pooled_pr_rank", 15.0, "repr.effective_rank", 100.0
        )


# ---------------------------------------------------------------------------------------
# rank_metrics_per_query -- the paired-test surface. The pre-registered rule for the
# memory region negative-set round is a paired bootstrap over the 500 shared FiQA dev
# queries, so the per-query terms must survive AND must be the same numbers the receipt
# already stores. These pin both halves of that.
# ---------------------------------------------------------------------------------------


def _multi_relevant_ranking(
    queries: int = 48, pool: int = 60
) -> tuple[torch.Tensor, list[list[int]]]:
    """A non-trivial multi-relevant case: 1-4 golds per query, at FiQA-ish density."""
    generator = torch.Generator().manual_seed(20260906)
    scores = torch.randn(queries, pool, generator=generator)
    gold: list[list[int]] = []
    for row in range(queries):
        count = 1 + (row % 4)
        picks = torch.randperm(pool, generator=generator)[:count]
        gold.append(sorted(int(p) for p in picks))
    return scores, gold


def test_rank_metrics_per_query_aggregates_are_exactly_rank_metrics() -> None:
    """EXACT equality with the aggregate-only path, on a non-trivial multi-relevant case.

    This is the property that makes the refactor safe to land under an already-published
    receipt: `rank_metrics` now delegates here, so if the two ever disagreed the receipt
    and the paired test would be describing different measurements. `==` rather than
    `approx` because any difference at all is the bug.
    """
    scores, gold = _multi_relevant_ranking()
    aggregates, per_query = beir_fiqa.rank_metrics_per_query(scores, gold)
    assert aggregates == beir_fiqa.rank_metrics(scores, gold)
    assert list(aggregates) == list(per_query)


def test_rank_metrics_per_query_vectors_average_to_the_aggregate() -> None:
    """Each vector re-averaged in float32 reproduces its own aggregate, bit for bit.

    Aggregation stays float32 deliberately (see `metrics.recall_at_k`): the published
    memory-region receipt stores Success@10 = 0.2 as 0.20000000298023224, and a float64
    re-aggregation would return 0.2 and stop matching it.
    """
    scores, gold = _multi_relevant_ranking()
    aggregates, per_query = beir_fiqa.rank_metrics_per_query(scores, gold)
    for key, values in per_query.items():
        assert len(values) == scores.size(0)
        assert torch.tensor(values, dtype=torch.float32).mean().item() == aggregates[key]


def test_rank_metrics_per_query_hand_checked_tiny_case() -> None:
    """The `_toy_scores` case, read off per query rather than averaged.

    Query 0 has two golds; its best one sits at rank 2 behind an irrelevant document, so
    it misses @1 and scores 1/2. Query 1's only gold is first. Those are exactly the 0.5
    recall@1 and 0.75 MRR the aggregate test above asserts -- shown here as the two
    numbers they are averaged from, which is what a paired test consumes.
    """
    scores, gold = _toy_scores()
    aggregates, per_query = beir_fiqa.rank_metrics_per_query(scores, gold)
    assert per_query["recall@1"] == [0.0, 1.0]
    assert per_query["recall@10"] == [1.0, 1.0]
    assert per_query["mrr"] == [0.5, 1.0]
    assert aggregates["recall@1"] == pytest.approx(0.5)
    assert "recall@100" not in per_query  # capped at the pool size, same as the aggregate


def test_bm25_metrics_per_query_carries_both_halves_unchanged() -> None:
    """The BM25 reference emits per-query terms without changing what it reports.

    `bm25_metrics` keeps stamping `index_s`, which stays an aggregate: it is wall-clock
    for the whole pass and has no per-query meaning. The metric keys, and only those,
    appear on both sides.
    """
    docs = [
        "the capital gains tax on a long term stock sale",
        "how to bake sourdough bread at home",
        "mortgage interest deduction and property tax",
        "index funds and expense ratios explained",
    ]
    task = beir_fiqa.RankingTask(
        queries=["capital gains tax on stock", "sourdough bread"],
        query_ids=["q0", "q1"],
        pool_ids=[f"d{i}" for i in range(len(docs))],
        pool_texts=docs,
        gold=[[0], [1]],
    )
    aggregates, per_query = beir_fiqa.bm25_metrics_per_query(task)
    assert "index_s" in aggregates
    assert "index_s" not in per_query
    for key, values in per_query.items():
        assert len(values) == len(task.queries)
        assert torch.tensor(values, dtype=torch.float32).mean().item() == aggregates[key]

    aggregate_only = beir_fiqa.bm25_metrics(task)
    assert {k: v for k, v in aggregate_only.items() if k != "index_s"} == {
        k: v for k, v in aggregates.items() if k != "index_s"
    }
