"""Closed-pool TF-IDF / BM25 lexical ceiling on a text holdout.

WHY THIS EXISTS
A trained encoder's recall@1 next to only the untrained baseline overstates what was
learned when the battery is solvable by bag-of-words. The 2026-09-06 reason-region
diagnosis scored TF-IDF cosine and BM25 on the exact seed-0 holdouts and found code /
compress / retrieve sitting at that ceiling. This module is that scorer, lifted into
the eval path so every text eval receipt carries `lexical_baseline.{tfidf,bm25}.*`
instead of a card having to cite an offline JSON.

THE SCORER MUST NOT DRIFT
Tokenisation, IDF, BM25 (k1=1.5, b=0.75), and the seed-0 uniform tie-break are copied
from `docs/design/evidence/reason-region-diagnosis-2026-09-06/scripts/cross_region_lexical.py`.
Changing any of those changes the number; bump `SCORER_VERSION` and re-file the
evidence if you must. Do not substitute `eval.beir_fiqa.BM25` -- that uses BEIR's
k1=0.9/b=0.4 and a different word regex.

G26
`lexical_baseline.split_sha256` is the holdout that was scored. A receipt whose
`split.sha256` disagrees is a different eval set; `verify_lexical_baseline_split`
refuses rather than let a card print a ceiling from the wrong items.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cogsyndelta.splits import SplitGuardError

SCORER_VERSION = "csd-lexical/v1"
"""Version stamp written next to the scores. Bump when the formula changes."""

WORD = re.compile(r"[a-z]+|\d+(?:\.\d+)?")
"""Diagnosis tokenizer: lowercased alphabetic runs, or a number with optional decimal."""

BM25_K1 = 1.5
BM25_B = 0.75
TIE_SEED = 0
TIE_EPS = 1e-9

RANK_KEYS: tuple[str, ...] = ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10")
"""The ranking metrics the eval battery reports (plus recall@5, which it also writes)."""


def tokenize(text: str) -> list[str]:
    """Split ``text`` with the diagnosis word regex.

    Args:
        text: Raw pair side (anchor or positive).

    Returns:
        Lowercased tokens. Empty if the string has no matching runs.
    """
    return WORD.findall(text.lower())


def rank_metrics(scores: np.ndarray) -> dict[str, float]:
    """Closed-pool ranking metrics with the relevant item on the diagonal.

    Args:
        scores: ``[N, N]``, higher is better. Row i's relevant candidate is column i.

    Returns:
        ``recall@1``, ``recall@5``, ``recall@10``, ``mrr``, ``ndcg@10``.

    Raises:
        ValueError: If ``scores`` is not a square 2-d array.
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"expected square [N, N] scores, got {tuple(scores.shape)}")
    n = scores.shape[0]
    if n == 0:
        return dict.fromkeys(RANK_KEYS, 0.0)
    order = np.argsort(-scores, axis=1)
    ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(n)])
    ndcg = np.where(ranks <= 10, 1.0 / np.log2(ranks.astype(np.float64) + 1.0), 0.0)
    return {
        "recall@1": float((ranks == 1).mean()),
        "recall@5": float((ranks <= 5).mean()),
        "recall@10": float((ranks <= 10).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "ndcg@10": float(ndcg.mean()),
    }


def _tie_break(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Seed-0 uniform noise in ``(0, 1e-9)`` so argsort is deterministic on ties."""
    return rng.uniform(0, TIE_EPS, shape)


def score_tfidf(anchors: Sequence[list[str]], positives: Sequence[list[str]]) -> np.ndarray:
    """Log-tf × smoothed-idf cosine, IDF over anchors+positives combined.

    Args:
        anchors: Tokenised queries, one list per holdout row.
        positives: Tokenised candidates, aligned with ``anchors``.

    Returns:
        ``[N, N]`` cosine matrix.
    """
    docs = list(anchors) + list(positives)
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))
    n_docs = len(docs)
    vocab = {word: i for i, word in enumerate(df)}
    idf = np.array([math.log((n_docs + 1) / (df[word] + 1)) + 1 for word in vocab])

    def tfidf(doc: list[str]) -> np.ndarray:
        """Log-tf × smoothed-idf vector for one tokenised document, L2-normalised."""
        vec = np.zeros(len(vocab))
        for word, count in Counter(doc).items():
            vec[vocab[word]] = (1 + math.log(count)) * idf[vocab[word]]
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    matrix_a = np.stack([tfidf(doc) for doc in anchors]) if anchors else np.zeros((0, 0))
    matrix_p = np.stack([tfidf(doc) for doc in positives]) if positives else np.zeros((0, 0))
    if matrix_a.size == 0 or matrix_p.size == 0:
        return np.zeros((len(anchors), len(positives)))
    return matrix_a @ matrix_p.T


def score_bm25(anchors: Sequence[list[str]], positives: Sequence[list[str]]) -> np.ndarray:
    """Okapi BM25 over the positive side, diagnosis k1/b and positives-only IDF.

    Args:
        anchors: Tokenised queries.
        positives: Tokenised documents (the candidate pool).

    Returns:
        ``[N, N]`` BM25 matrix.
    """
    n_pos = len(positives)
    n_q = len(anchors)
    scores = np.zeros((n_q, n_pos))
    if n_pos == 0 or n_q == 0:
        return scores
    pdocs = [Counter(doc) for doc in positives]
    dl = np.array([sum(doc.values()) for doc in pdocs], dtype=np.float64)
    avgdl = float(dl.mean())
    dfp: Counter[str] = Counter()
    for doc in pdocs:
        dfp.update(doc.keys())
    for i, query in enumerate(anchors):
        for word in set(query):
            if word not in dfp:
                continue
            idf_w = math.log(1 + (n_pos - dfp[word] + 0.5) / (dfp[word] + 0.5))
            for j, doc in enumerate(pdocs):
                freq = doc.get(word, 0)
                if freq:
                    denom = freq + BM25_K1 * (1 - BM25_B + BM25_B * dl[j] / avgdl)
                    scores[i, j] += idf_w * freq * (BM25_K1 + 1) / denom
    return scores


def score_holdout(holdout: Sequence[tuple[str, str]]) -> dict[str, dict[str, float]]:
    """TF-IDF cosine and BM25 ranking metrics on a closed holdout.

    Args:
        holdout: ``(anchor, positive)`` pairs; the candidate pool is the positives,
            relevant item i is pair i (same diagonal as ``evaluate()`` / the eval battery).

    Returns:
        ``{"tfidf": {metric: float}, "bm25": {metric: float}}``.

    Raises:
        ValueError: If the holdout is empty.
    """
    if not holdout:
        raise ValueError("refusing to score an empty holdout")
    anchors = [tokenize(anchor) for anchor, _ in holdout]
    positives = [tokenize(positive) for _, positive in holdout]
    rng = np.random.default_rng(TIE_SEED)
    n = len(holdout)
    tfidf = score_tfidf(anchors, positives)
    bm25 = score_bm25(anchors, positives)
    return {
        "tfidf": rank_metrics(tfidf + _tie_break((n, n), rng)),
        "bm25": rank_metrics(bm25 + _tie_break((n, n), rng)),
    }


def build_lexical_baseline(holdout: Sequence[tuple[str, str]], split_sha256: str) -> dict[str, Any]:
    """First-class ``lexical_baseline`` block for an eval (or sidecar) receipt.

    Args:
        holdout: The manifested held-out pairs this eval scored.
        split_sha256: ``split.sha256`` of that holdout (G26). Written onto the block
            so a later reader can refuse a mismatch.

    Returns:
        ``{scorer_version, split_sha256, n_pairs, tfidf, bm25}``.

    Raises:
        ValueError: Empty holdout, or ``split_sha256`` is blank.
    """
    sha = str(split_sha256 or "").strip()
    if not sha:
        raise ValueError(
            "refusing to write lexical_baseline without split_sha256 "
            "(fail closed, G26: a ceiling with no split is not a measurement)"
        )
    scored = score_holdout(holdout)
    return {
        "scorer_version": SCORER_VERSION,
        "split_sha256": sha,
        "n_pairs": len(holdout),
        "tfidf": scored["tfidf"],
        "bm25": scored["bm25"],
    }


def receipt_split_sha256(receipt: Mapping[str, Any]) -> str | None:
    """The receipt's own ``split.sha256``, train-top-level or eval ``provenance.split``.

    Args:
        receipt: A train, eval, or lexical sidecar dict.

    Returns:
        The hex digest, or ``None`` when neither location carries one.
    """
    split = receipt.get("split")
    if isinstance(split, dict):
        sha = split.get("sha256")
        if sha:
            return str(sha)
    provenance = receipt.get("provenance")
    if isinstance(provenance, dict):
        nested = provenance.get("split")
        if isinstance(nested, dict) and nested.get("sha256"):
            return str(nested["sha256"])
    return None


def verify_lexical_baseline_split(receipt: Mapping[str, Any]) -> None:
    """Refuse a ``lexical_baseline`` whose split sha is not this receipt's.

    Absent field: not measured, accepted. Present field with no sha, or a sha that
    disagrees with ``split.sha256``: ``SplitGuardError`` (fail closed, G26).

    Args:
        receipt: Eval / lexical sidecar (or any dict carrying both fields).

    Raises:
        SplitGuardError: The field is present and cannot be bound to this receipt's split.
    """
    field = receipt.get("lexical_baseline")
    if not field:
        return
    if not isinstance(field, dict):
        raise SplitGuardError("G26: lexical_baseline is present but is not an object -- refusing")
    field_sha = str(field.get("split_sha256") or "")
    if not field_sha:
        raise SplitGuardError(
            "G26: lexical_baseline is present but split_sha256 is missing -- refusing"
        )
    receipt_sha = receipt_split_sha256(receipt)
    if not receipt_sha:
        raise SplitGuardError(
            "G26: lexical_baseline is present but the receipt has no split.sha256 -- "
            "refusing (a ceiling that cannot name its holdout is not a measurement)"
        )
    if field_sha != receipt_sha:
        raise SplitGuardError(
            f"G26: lexical_baseline.split_sha256 {field_sha} != receipt split.sha256 {receipt_sha}"
        )
