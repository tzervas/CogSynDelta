"""Regression test for the unconditional-corpus-shuffle fix in `build_splits`.

`build_splits` used to shuffle the loaded corpus only when
`len(cfg.extra_sources) > 1 or cfg.extra_sources` was true -- i.e. only when combining
MULTIPLE sources. That is always false for a single-source region, so `code` and
`compress` (each backed by exactly one parquet source) were never shuffled at all:

  - in-batch InfoNCE negatives came from contiguous, same-domain blocks (on `code`,
    contiguous runs of one GitHub repository), which is an easier task than the recall
    numbers implied.
  - `holdout = all_pairs[:cfg.holdout_pairs]` takes the HEAD of that same unshuffled
    list, so the entire held-out eval set was whatever group happened to sort first --
    on `code`, 512 held-out pairs spanning only 2 of the corpus's 13,581 repositories.

The fix makes the shuffle unconditional (still seeded on `cfg.seed`, so it stays
deterministic and reproducible). This test would have failed against the pre-fix code:
it builds a small single-source corpus whose on-disk order is grouped -- exactly the
shape a real single-source corpus has -- and asserts the resulting holdout is not
confined to a single group. Asserting "shuffle was called" would not catch the original
bug, since the shuffle call itself was correct; it was simply unreachable for a single
source. The property that has to hold is holdout diversity, so that is what is checked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# MUST precede the import below: cogsyndelta.regions.pretrain imports tokenizers at
# module scope, so without the train group installed the import errors during
# collection and the whole file fails instead of skipping (same reasoning as
# tests/test_region_compress.py and tests/test_pretrain_resume.py).
pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

import pyarrow as pa
import pyarrow.parquet as pq

from cogsyndelta.regions.pretrain import PretrainConfig, build_splits

pytestmark = pytest.mark.cpu

_GROUPS = ("groupA", "groupB", "groupC", "groupD")
_PER_GROUP = 60
_HOLDOUT_PAIRS = 40


def _group_of(anchor: str) -> str:
    """The group a synthetic anchor came from, e.g. 'groupA-anchor-3' -> 'groupA'."""
    return anchor.split("-", 1)[0]


def _write_grouped_corpus(path: Path) -> None:
    """A single parquet shard with `_PER_GROUP` rows per group, ALL of one group's rows
    contiguous before the next starts -- the same shape CodeSearchNet's `repo` column
    has (long contiguous runs of one repository) and the shape that revealed the bug."""
    anchors: list[str] = []
    positives: list[str] = []
    for group in _GROUPS:
        for i in range(_PER_GROUP):
            anchors.append(f"{group}-anchor-{i}")
            positives.append(f"{group}-positive-{i}")
    pq.write_table(pa.table({"a": anchors, "b": positives}), path)


@pytest.fixture
def grouped_cfg(tmp_path: Path) -> PretrainConfig:
    shard = tmp_path / "grouped.parquet"
    _write_grouped_corpus(shard)
    total = len(_GROUPS) * _PER_GROUP
    return PretrainConfig(
        region="grouped-test",
        pair_columns=("a", "b"),
        shards=[str(shard)],
        # Deliberately empty: this is exactly the single-source shape of `code` and
        # `compress`, the two regions the original bug affected. `retrieve` (multiple
        # sources) already exercised the shuffle before this fix and is covered
        # separately below.
        extra_sources=[],
        steps=1,
        batch_size=total,  # budget = steps*batch_size + holdout_pairs > total corpus
        holdout_pairs=_HOLDOUT_PAIRS,
        seed=0,
    )


def test_holdout_is_not_confined_to_a_single_group(grouped_cfg: PretrainConfig) -> None:
    holdout, train_pairs, _ = build_splits(grouped_cfg)
    assert len(holdout) == grouped_cfg.holdout_pairs

    groups_in_holdout = {_group_of(a) for a, _ in holdout}
    # An unshuffled, grouped, single-source corpus puts the ENTIRE holdout in
    # `_GROUPS[0]` (the head of the list) -- this is the exact failure this test exists
    # to catch, and it is what the pre-fix `build_splits` produced here.
    assert len(groups_in_holdout) > 1, (
        f"holdout drawn from only {groups_in_holdout} out of {_GROUPS}; a single-source "
        f"corpus is not being shuffled"
    )
    # Stronger than "more than one": with four equally-sized groups and a seeded
    # near-uniform shuffle, the holdout should draw from most or all of them, not just
    # barely escape a single group.
    assert len(groups_in_holdout) >= len(_GROUPS) - 1, (
        f"holdout drawn from only {groups_in_holdout}; expected broader coverage of "
        f"{_GROUPS} from a properly shuffled corpus"
    )

    # The split must still be a real, disjoint partition.
    holdout_anchors = {a for a, _ in holdout}
    train_anchors = {a for a, _ in train_pairs}
    assert holdout_anchors.isdisjoint(train_anchors)


def test_shuffle_is_deterministic_across_repeated_calls(grouped_cfg: PretrainConfig) -> None:
    """Reproducibility is a hard requirement (receipts record a corpus fingerprint and
    two runs must be comparable), so the same seed must produce the same split every
    time -- this must stay true after making the shuffle unconditional."""
    holdout_1, train_1, _ = build_splits(grouped_cfg)
    holdout_2, train_2, _ = build_splits(grouped_cfg)
    assert holdout_1 == holdout_2
    assert train_1 == train_2


def test_a_genuinely_multi_source_region_was_already_shuffled(tmp_path: Path) -> None:
    """`retrieve` (unlike `code`/`compress`) has always had >1 extra source and so was
    already shuffled before this fix. Confirm that directly rather than assuming it: the
    same grouped-corpus setup, but split into a primary shard plus one extra source,
    must also escape single-group holdout -- and did even under the old gating
    condition, since `len(cfg.extra_sources) > 1 or cfg.extra_sources` is true here."""
    primary = tmp_path / "primary.parquet"
    extra = tmp_path / "extra.parquet"
    half = len(_GROUPS) // 2
    _write_grouped_corpus_subset(primary, _GROUPS[:half])
    _write_grouped_corpus_subset(extra, _GROUPS[half:])

    cfg = PretrainConfig(
        region="grouped-multi-source-test",
        pair_columns=("a", "b"),
        shards=[str(primary)],
        extra_sources=[{"shards": [str(extra)], "columns": ["a", "b"], "limit": 0}],
        steps=1,
        batch_size=len(_GROUPS) * _PER_GROUP,
        holdout_pairs=_HOLDOUT_PAIRS,
        seed=0,
    )
    holdout, _, _ = build_splits(cfg)
    groups_in_holdout = {_group_of(a) for a, _ in holdout}
    assert len(groups_in_holdout) > 1


def _write_grouped_corpus_subset(path: Path, groups: tuple[str, ...]) -> None:
    anchors: list[str] = []
    positives: list[str] = []
    for group in groups:
        for i in range(_PER_GROUP):
            anchors.append(f"{group}-anchor-{i}")
            positives.append(f"{group}-positive-{i}")
    pq.write_table(pa.table({"a": anchors, "b": positives}), path)
