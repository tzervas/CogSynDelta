"""Evaluation: metrics, and the contamination guards that make them mean anything."""

from cogsyndelta.eval.geometry import (
    GeometryReference,
    GeometryReferenceError,
    compute_geometry,
    verify_geometry_reference,
)
from cogsyndelta.eval.lexical import (
    SCORER_VERSION,
    build_lexical_baseline,
    score_holdout,
    verify_lexical_baseline_split,
)
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
    screen_pair_contamination,
    spearman_correlation,
    token_weighted_perplexity,
)

__all__ = [
    "SCORER_VERSION",
    "ChannelOverlap",
    "ContaminationReport",
    "GeometryReference",
    "GeometryReferenceError",
    "PairContaminationReport",
    "assert_no_contamination",
    "assert_no_pair_contamination",
    "build_lexical_baseline",
    "compare",
    "compute_geometry",
    "contamination_report",
    "mean_reciprocal_rank",
    "pair_contamination_report",
    "pair_fingerprint",
    "recall_at_k",
    "representation_std",
    "score_holdout",
    "screen_pair_contamination",
    "spearman_correlation",
    "token_weighted_perplexity",
    "verify_geometry_reference",
    "verify_lexical_baseline_split",
]
