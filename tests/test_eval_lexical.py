"""Lexical baseline scorer and the G26 split-sha refuse (g49)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cogsyndelta.eval.lexical import (
    SCORER_VERSION,
    build_lexical_baseline,
    score_holdout,
    verify_lexical_baseline_split,
)
from cogsyndelta.pipeline.receipt import Producer, Receipt
from cogsyndelta.splits import SplitGuardError

pytestmark = pytest.mark.cpu


def test_score_holdout_is_deterministic_and_diagonal_perfect_on_unique_tokens() -> None:
    """Each positive restates its anchor's unique token -- TF-IDF r@1 is 1.0 both times."""
    holdout = [(f"query alpha{i} unique", f"doc alpha{i} unique") for i in range(8)]
    first = score_holdout(holdout)
    second = score_holdout(holdout)
    assert first == second
    assert first["tfidf"]["recall@1"] == 1.0
    assert first["bm25"]["recall@1"] == 1.0
    for family in ("tfidf", "bm25"):
        for key in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"):
            assert 0.0 <= first[family][key] <= 1.0


def test_build_lexical_baseline_carries_scorer_version_and_split_sha() -> None:
    holdout = [("cat sat", "the cat sat"), ("dog ran", "the dog ran")]
    block = build_lexical_baseline(holdout, "abc123")
    assert block["scorer_version"] == SCORER_VERSION
    assert block["split_sha256"] == "abc123"
    assert block["n_pairs"] == 2
    assert "recall@1" in block["tfidf"]
    assert "recall@1" in block["bm25"]


def test_build_lexical_baseline_refuses_blank_split_sha() -> None:
    with pytest.raises(ValueError, match="split_sha256"):
        build_lexical_baseline([("a", "b")], "")


def test_verify_lexical_baseline_absent_is_not_measured() -> None:
    verify_lexical_baseline_split({"metrics": {"rank.recall@1": 0.1}})


def test_verify_lexical_baseline_split_sha_mismatch_refuses() -> None:
    """G26 fail-closed: a ceiling from a different holdout is not this receipt's number."""
    receipt = {
        "split": {"sha256": "aaa"},
        "lexical_baseline": {
            "scorer_version": SCORER_VERSION,
            "split_sha256": "bbb",
            "tfidf": {"recall@1": 0.9},
        },
    }
    with pytest.raises(SplitGuardError, match=r"split\.sha256"):
        verify_lexical_baseline_split(receipt)


def test_verify_lexical_baseline_present_without_receipt_split_refuses() -> None:
    receipt = {
        "lexical_baseline": {
            "scorer_version": SCORER_VERSION,
            "split_sha256": "aaa",
            "tfidf": {"recall@1": 0.9},
        }
    }
    with pytest.raises(SplitGuardError, match=r"no split\.sha256"):
        verify_lexical_baseline_split(receipt)


def test_verify_lexical_baseline_matching_sha_is_accepted() -> None:
    receipt = {
        "provenance": {"split": {"sha256": "aaa"}},
        "lexical_baseline": {
            "scorer_version": SCORER_VERSION,
            "split_sha256": "aaa",
            "tfidf": {"recall@1": 0.9},
        },
    }
    verify_lexical_baseline_split(receipt)


def test_receipt_write_refuses_mismatched_lexical_split_sha(tmp_path: Path) -> None:
    rec = Receipt(
        producer=Producer("cogsyndelta", "compress"),
        stage="eval",
        kind="eval",
        provenance={"split": {"sha256": "aaa"}},
        lexical_baseline={
            "scorer_version": SCORER_VERSION,
            "split_sha256": "bbb",
            "tfidf": {"recall@1": 0.5},
            "bm25": {"recall@1": 0.4},
        },
    )
    with pytest.raises(SplitGuardError, match=r"split\.sha256"):
        rec.write(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_receipt_write_omits_empty_lexical_baseline(tmp_path: Path) -> None:
    rec = Receipt(
        producer=Producer("cogsyndelta", "visual"),
        stage="eval",
        kind="eval",
    )
    path = rec.write(tmp_path)
    raw = json.loads(path.read_text())
    assert "lexical_baseline" not in raw
