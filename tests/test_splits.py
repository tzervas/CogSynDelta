"""G26 split-manifest guards, stdlib-only (CI does not install the train group).

The load-bearing checks -- sha of membership, corpus-fingerprint binding, receipt
split.sha256, held-out leak -- must be testable without pyarrow/tokenizers. Drawing a
real parquet split lives in tests/test_split_manifests.py.
"""

from __future__ import annotations

import json

import pytest

from cogsyndelta.splits import (
    SPLIT_MANIFEST_SCHEMA,
    SplitGuardError,
    assert_no_held_out_in_pairs,
    build_split_manifest,
    item_id,
    membership_sha256,
    permute_train_pairs,
    split_manifest_path,
    verify_receipt_split,
    verify_split_manifest,
)

pytestmark = pytest.mark.cpu


def test_item_id_is_ordered_and_normalised() -> None:
    assert item_id("Hello  World", "Yes") == item_id("hello world", "yes")
    assert item_id("a", "b") != item_id("b", "a")


def test_membership_sha256_is_order_independent() -> None:
    ids = [item_id(f"a{i}", f"b{i}") for i in range(5)]
    assert membership_sha256(ids) == membership_sha256(list(reversed(ids)))


def test_split_manifest_path_uses_fp8_and_seed(tmp_path) -> None:
    path = split_manifest_path("reason", "ca364a92d2c6c5fd", 0, splits_dir=tmp_path)
    assert path.name == "reason-ca364a92-split0.json"


def test_doctored_membership_list_fails_sha_check() -> None:
    holdout = [("q1", "a1"), ("q2", "a2")]
    train = [("q3", "a3")]
    manifest = build_split_manifest(
        region="t",
        corpus_fingerprint="fp" * 16,
        split_seed=0,
        holdout=holdout,
        train_pairs=train,
        generator={"algorithm": "csd-split-draw/v1"},
        counts={},
    )
    manifest["holdout_ids"][0] = "0" * 64
    with pytest.raises(SplitGuardError, match="doctored"):
        verify_split_manifest(manifest, corpus_fingerprint="fp" * 16)


def test_fingerprint_mismatch_refuses() -> None:
    holdout = [("q1", "a1"), ("q2", "a2")]
    manifest = build_split_manifest(
        region="t",
        corpus_fingerprint="aa" * 16,
        split_seed=0,
        holdout=holdout,
        train_pairs=[],
        generator={},
        counts={},
    )
    with pytest.raises(SplitGuardError, match="corpus fingerprint"):
        verify_split_manifest(manifest, corpus_fingerprint="bb" * 16, holdout=holdout)


def test_one_pair_swapped_in_draw_refuses() -> None:
    holdout = [("q1", "a1"), ("q2", "a2")]
    manifest = build_split_manifest(
        region="t",
        corpus_fingerprint="cc" * 16,
        split_seed=0,
        holdout=holdout,
        train_pairs=[],
        generator={},
        counts={},
    )
    swapped = [("q1", "a1"), ("q9", "a9")]
    with pytest.raises(SplitGuardError, match="does not match the split manifest"):
        verify_split_manifest(manifest, corpus_fingerprint="cc" * 16, holdout=swapped)


def test_held_out_item_in_train_refuses() -> None:
    holdout = [("q1", "a1"), ("q2", "a2")]
    train = [("q3", "a3"), ("q1", "a1")]
    with pytest.raises(SplitGuardError, match="held-out item"):
        assert_no_held_out_in_pairs(holdout, train)


def test_receipt_split_sha_mismatch_refuses() -> None:
    manifest = {"sha256": "aaa", "seed": 0, "schema": SPLIT_MANIFEST_SCHEMA}
    receipt = {"split": {"sha256": "bbb", "seed": 0, "manifest": "x.json"}}
    with pytest.raises(SplitGuardError, match=r"split\.sha256"):
        verify_receipt_split(receipt, manifest)


def test_legacy_seed1_receipt_without_split_block_is_retired() -> None:
    manifest = {"sha256": "aaa", "seed": 0}
    receipt = {"config": {"seed": 1}}
    with pytest.raises(SplitGuardError, match="retired"):
        verify_receipt_split(receipt, manifest)


def test_legacy_seed0_receipt_without_split_block_is_accepted() -> None:
    manifest = {"sha256": "aaa", "seed": 0}
    receipt = {"config": {"seed": 0}}
    verify_receipt_split(receipt, manifest)


def test_committed_seed0_manifests_exist_and_hash() -> None:
    """The six text-region seed-0 files in config/mind/splits/ are load-bearing.

    CI has no corpus, so this only checks the committed artefacts parse and that
    ``sha256`` matches the listed holdout ids -- a truncated or hand-edited file
    fails closed here rather than at the next GPU run.
    """
    from cogsyndelta.splits import DEFAULT_SPLITS_DIR, SPLIT_MANIFEST_SCHEMA

    names = (
        "code-e493273b-split0.json",
        "language-e493273b-split0.json",
        "compress-65a99381-split0.json",
        "retrieve-af4bb36f-split0.json",
        "reason-ca364a92-split0.json",
        "memory-6fc0cf23-split0.json",
    )
    for name in names:
        path = DEFAULT_SPLITS_DIR / name
        assert path.is_file(), f"missing committed split manifest {path}"
        payload = json.loads(path.read_text())
        assert payload["schema"] == SPLIT_MANIFEST_SCHEMA
        assert payload["seed"] == 0
        assert payload["sha256"] == membership_sha256(payload["holdout_ids"])
        assert len(payload["holdout_ids"]) == payload["counts"]["holdout_pairs"] == 512


def test_order_seed_zero_is_identity() -> None:
    pairs = [("a", "1"), ("b", "2"), ("c", "3")]
    assert permute_train_pairs(pairs, 0) == pairs
    shuffled = permute_train_pairs(pairs, 7)
    assert sorted(shuffled) == sorted(pairs)
    assert shuffled != pairs
