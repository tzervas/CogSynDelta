"""Evaluation metrics, and the contamination guards that make them mean anything.

WHY THIS EXISTS
The operator's constraint is that a change must give a net efficiency gain "without
decreasing quality and/or increasing perplexity". That is only checkable against a
baseline, and this repo has none: every entry in benchmark_results carries
``"quality": {}``. Without a measurement, "no quality loss" is not a claim -- it is a
hope.

THE FAILURE MODE THIS IS BUILT AGAINST
The easiest way to poison a model is not a broken layer; it is a contaminated eval. If
even a fraction of the eval set appears in training, every number improves, the model
looks better than it is, and nothing in the training loop can detect it. That is
indistinguishable from real progress right up until the model meets data it has genuinely
not seen.

So :func:`assert_no_contamination` is not a nicety here. It is the difference between a
measurement and a story.

WHAT IS DELIBERATELY NOT HERE
No "quality score" that blends unrelated numbers into one figure. A single blended score
hides exactly the trade a small model must be judged on -- perplexity moving one way while
retrieval moves the other is the interesting case, and averaging destroys it.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from typing import TypedDict

import torch


class ContaminationReport(TypedDict):
    """Overlap between a train and an eval set."""

    train_unique: int
    eval_unique: int
    overlap: int
    eval_fraction_contaminated: float
    examples: list[str]


def token_weighted_perplexity(losses: Sequence[float], token_counts: Sequence[int]) -> float:
    """Perplexity weighted by tokens, not by batch.

    Averaging per-batch losses lets a short trailing batch carry the same weight as a full
    one, which shifts the number for a reason that has nothing to do with the model.

    Args:
        losses: Mean cross-entropy per batch, in nats.
        token_counts: Target token count per batch.

    Returns:
        Perplexity over all tokens.

    Raises:
        ValueError: If the sequences disagree in length or no tokens were seen.
    """
    if len(losses) != len(token_counts):
        raise ValueError(f"{len(losses)} losses vs {len(token_counts)} counts")
    total = sum(token_counts)
    if total == 0:
        raise ValueError("no tokens: refusing to report a perplexity over an empty set")
    nll = sum(loss * n for loss, n in zip(losses, token_counts, strict=True))
    return math.exp(nll / total)


def _fingerprint(text: str) -> str:
    """Stable hash of normalised text, for overlap detection.

    Normalises whitespace and case so that trivial reformatting cannot hide a duplicate --
    contamination usually arrives via a reformatted copy, not a byte-identical one.
    """
    normalised = " ".join(text.split()).lower()
    return hashlib.blake2b(normalised.encode("utf-8", "replace"), digest_size=16).hexdigest()


def contamination_report(
    train_texts: Iterable[str], eval_texts: Iterable[str]
) -> ContaminationReport:
    """Measure overlap between a train and an eval set.

    Args:
        train_texts: Training documents.
        eval_texts: Held-out documents.

    Returns:
        Counts, the overlap fraction of the eval set, and up to five example
        fingerprints so a hit can actually be chased down rather than merely reported.
    """
    train_fp = {_fingerprint(t) for t in train_texts}
    eval_fp_list = [_fingerprint(t) for t in eval_texts]
    eval_fp = set(eval_fp_list)
    overlap = train_fp & eval_fp
    return ContaminationReport(
        train_unique=len(train_fp),
        eval_unique=len(eval_fp),
        overlap=len(overlap),
        eval_fraction_contaminated=(len(overlap) / len(eval_fp)) if eval_fp else 0.0,
        examples=sorted(overlap)[:5],
    )


def assert_no_contamination(
    train_texts: Iterable[str], eval_texts: Iterable[str], *, tolerance: float = 0.0
) -> ContaminationReport:
    """Raise if the eval set overlaps training beyond ``tolerance``.

    Default tolerance is zero. A non-zero tolerance should be a deliberate, argued choice
    for a specific corpus -- not a way to make a failing check pass.

    Args:
        train_texts: Training documents.
        eval_texts: Held-out documents.
        tolerance: Maximum acceptable contaminated fraction of the eval set.

    Returns:
        The contamination report, when within tolerance.

    Raises:
        ValueError: If contamination exceeds ``tolerance``.
    """
    report = contamination_report(train_texts, eval_texts)
    fraction = report["eval_fraction_contaminated"]
    if fraction > tolerance:
        raise ValueError(
            f"eval set is {fraction:.2%} contaminated ({report['overlap']} of "
            f"{report['eval_unique']} documents also appear in training; tolerance "
            f"{tolerance:.2%}). Every metric measured against it is inflated."
        )
    return report


def recall_at_k(scores: torch.Tensor, relevant: torch.Tensor, k: int) -> float:
    """Fraction of queries whose relevant item appears in the top ``k``.

    Args:
        scores: ``[B, N]``, higher is better.
        relevant: ``[B]`` index of the relevant candidate per query.
        k: Cutoff.

    Returns:
        Recall in ``[0, 1]``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, scores.size(1))
    top = scores.topk(k, dim=-1).indices
    return (top == relevant.unsqueeze(-1)).any(dim=-1).float().mean().item()


def mean_reciprocal_rank(scores: torch.Tensor, relevant: torch.Tensor) -> float:
    """Mean of 1/rank of the relevant candidate.

    Args:
        scores: ``[B, N]``, higher is better.
        relevant: ``[B]`` index of the relevant candidate.

    Returns:
        MRR in ``(0, 1]``.
    """
    order = scores.argsort(dim=-1, descending=True)
    ranks = (order == relevant.unsqueeze(-1)).float().argmax(dim=-1) + 1
    return (1.0 / ranks.float()).mean().item()


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Ascending ranks, with tied values sharing their average rank.

    Ties are not an edge case for graded similarity data, they are most of it: STS-B
    validation carries 1500 pairs over 64 distinct scores, and its largest tie group is
    139 pairs all annotated 0.0. Ranking those by array position would invent an ordering
    the annotators never gave and move the correlation for a reason that has nothing to
    do with the model.

    Args:
        values: Numbers to rank.

    Returns:
        One rank per input, in input order, 1-based.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        shared = (start + stop) / 2.0 + 1.0
        for pos in range(start, stop + 1):
            ranks[order[pos]] = shared
        start = stop + 1
    return ranks


def spearman_correlation(predicted: Sequence[float], gold: Sequence[float]) -> float:
    """Spearman rank correlation: Pearson over average ranks.

    This is the held-out metric for any region judged on GRADED similarity rather than
    retrieval. Cosine similarity and a human 0-1 score do not share a scale and are not
    linearly related, so Pearson on the raw values would penalise a model that ranks every
    pair correctly but compresses its cosines into a narrow band -- which small encoders
    always do. Rank correlation asks only the question the region's job actually poses:
    do neighbours stay neighbours, in order.

    Implemented here rather than pulled from scipy: this is the whole of it, and scipy
    would be a new dependency on every host and CI job for one function.

    Args:
        predicted: Model scores, e.g. cosine similarities.
        gold: Human scores, aligned with ``predicted``.

    Returns:
        Correlation in ``[-1, 1]``. Returns 0.0 when either side is constant -- see below.

    Raises:
        ValueError: If the sequences differ in length or hold fewer than two points.
    """
    if len(predicted) != len(gold):
        raise ValueError(f"{len(predicted)} predictions vs {len(gold)} gold scores")
    if len(predicted) < 2:
        raise ValueError("rank correlation needs at least 2 points")

    rank_p = _average_ranks(predicted)
    rank_g = _average_ranks(gold)
    n = len(rank_p)
    mean_p = sum(rank_p) / n
    mean_g = sum(rank_g) / n
    cov = sum((a - mean_p) * (b - mean_g) for a, b in zip(rank_p, rank_g, strict=True))
    var_p = sum((a - mean_p) ** 2 for a in rank_p)
    var_g = sum((b - mean_g) ** 2 for b in rank_g)

    # A constant side has no ranking to correlate with, so the coefficient is undefined.
    # 0.0 is the honest report -- "no monotone relationship detectable" -- and it is what
    # a fully collapsed encoder deserves: every pair scored identically is not a perfect
    # correlation. Raising instead would abort a training run at exactly the moment the
    # collapse it is meant to detect had happened. Report emb_std alongside this to tell
    # "collapsed" apart from "uncorrelated".
    if var_p <= 0.0 or var_g <= 0.0:
        return 0.0
    return cov / math.sqrt(var_p * var_g)


def representation_std(embeddings: torch.Tensor) -> float:
    """Per-feature standard deviation across the batch, averaged.

    The collapse signal for any embedding model. Near zero means every input maps to the
    same vector, which produces excellent-looking losses on several objectives while the
    representation carries no information at all.

    Args:
        embeddings: ``[B, D]``.

    Returns:
        Mean per-feature std.
    """
    if embeddings.dim() != 2:
        raise ValueError(f"expected [B, D], got {tuple(embeddings.shape)}")
    if embeddings.size(0) < 2:
        raise ValueError("need at least 2 examples to measure variance across a batch")
    return embeddings.std(dim=0).mean().item()


def compare(
    baseline: dict[str, float], candidate: dict[str, float], *, lower_is_better: set[str]
) -> dict[str, object]:
    """Compare a candidate against a baseline, naming regressions explicitly.

    Exists so that "net gain without quality loss" is evaluated per metric rather than by
    a blended score. A change that improves throughput while raising perplexity is not a
    win, and averaging the two would report it as one.

    Args:
        baseline: Metric name to value.
        candidate: Same keys.
        lower_is_better: Metrics where a decrease is an improvement, e.g. perplexity.

    Returns:
        Per-metric deltas, a list of regressions, and an overall verdict.
    """
    shared = sorted(set(baseline) & set(candidate))
    deltas, regressions = {}, []
    for name in shared:
        before, after = baseline[name], candidate[name]
        improved = after < before if name in lower_is_better else after > before
        rel = ((after - before) / before) if before else float("nan")
        deltas[name] = {"before": before, "after": after, "rel_change": rel, "improved": improved}
        if not improved and before != after:
            regressions.append(name)
    return {
        "metrics": deltas,
        "regressions": regressions,
        "missing": sorted(set(baseline) ^ set(candidate)),
        "verdict": "no regression" if not regressions else f"regressed: {', '.join(regressions)}",
    }
