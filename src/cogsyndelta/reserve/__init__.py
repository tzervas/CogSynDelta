"""The constructed cross-faculty reserve: item shapes, their guards, and their ledger.

The reserve is *constructed*, not fetched (design §5.5), so every guard here keys on the
SOURCE ROW rather than on the derived item -- DEC-38's correction. Nothing in this package
reads a corpus: parquet I/O lives in ``scripts/csd-build-x7-episodes.py`` so this package
imports cleanly in a venv without the ``train`` dependency group (see
``tests/test_import_hygiene.py`` for why that constraint is load-bearing).
"""

from cogsyndelta.reserve.episodes import (
    EPISODE_SCHEMA,
    GENERATOR_NAME,
    REJECTION_REASONS,
    SPLIT_KEY_SCHEME,
    EpisodeRejectedError,
    FactDonor,
    ProbeSource,
    SplitKeyMissingError,
    build_episode,
    content_hash,
    episode_content_payload,
    generator_identity,
    reference_solver_rank1,
    source_row_split,
    split_key_id,
)

__all__ = [
    "EPISODE_SCHEMA",
    "GENERATOR_NAME",
    "REJECTION_REASONS",
    "SPLIT_KEY_SCHEME",
    "EpisodeRejectedError",
    "FactDonor",
    "ProbeSource",
    "SplitKeyMissingError",
    "build_episode",
    "content_hash",
    "episode_content_payload",
    "generator_identity",
    "reference_solver_rank1",
    "source_row_split",
    "split_key_id",
]
