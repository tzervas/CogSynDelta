"""Evaluation: metrics, and the contamination guards that make them mean anything."""

from cogsyndelta.eval.metrics import (
    ContaminationReport,
    assert_no_contamination,
    compare,
    contamination_report,
    mean_reciprocal_rank,
    recall_at_k,
    representation_std,
    spearman_correlation,
    token_weighted_perplexity,
)

__all__ = [
    "ContaminationReport",
    "assert_no_contamination",
    "compare",
    "contamination_report",
    "mean_reciprocal_rank",
    "recall_at_k",
    "representation_std",
    "spearman_correlation",
    "token_weighted_perplexity",
]
