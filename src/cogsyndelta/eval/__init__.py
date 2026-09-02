"""Evaluation: metrics, and the contamination guards that make them mean anything."""

from cogsyndelta.eval.metrics import (
    ChannelOverlap,
    ContaminationReport,
    PairContaminationReport,
    assert_no_contamination,
    assert_no_pair_contamination,
    compare,
    contamination_report,
    mean_reciprocal_rank,
    pair_contamination_report,
    pair_fingerprint,
    recall_at_k,
    representation_std,
    spearman_correlation,
    token_weighted_perplexity,
)

__all__ = [
    "ChannelOverlap",
    "ContaminationReport",
    "PairContaminationReport",
    "assert_no_contamination",
    "assert_no_pair_contamination",
    "compare",
    "contamination_report",
    "mean_reciprocal_rank",
    "pair_contamination_report",
    "pair_fingerprint",
    "recall_at_k",
    "representation_std",
    "spearman_correlation",
    "token_weighted_perplexity",
]
