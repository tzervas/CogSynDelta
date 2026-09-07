"""G41: the gsm8k `test` split is a holdout, and the guard that makes it one must fire.

WHY THIS FILE EXISTS
`openai/gsm8k:main` `test` (1,319 rows, MIT) was landed 2026-09-06 as the population of
the pre-registered reasoning battery. Its entire value is that no region has seen it. A
holdout that is merely *intended* to be held out is not a holdout, so the property is
enforced in two places and this file constructs the failure each one exists to catch:

  1. **By item id** -- `assert_no_reserved_holdout_in_pairs`, called from `build_splits`
     on the realised training pairs, so it covers every region and every composite phase
     regardless of which shards a config names.
  2. **By shard path** -- `csd-train-all.py`'s `HELD_OUT_SHARDS`, checked at
     shard-resolution time, so a glob widened from `train.parquet` to `*.parquet` refuses
     to start rather than quietly absorbing the battery's rows.

Per tests/test_guards_can_fail.py's lesson, "the guard exists" is not the claim. Every
positive assertion below has a paired control: the leak IS detected with the guard in
place, and the same leak is NOT detected once the call site is neutered
(`test_neutering_the_call_site_lets_the_leak_through`), which is what proves the raise
came from the guard rather than from something else in the path.

Stdlib-only above the `importorskip` line, so CI (no train group, no corpus) still runs
the manifest-integrity and guard-firing checks.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from cogsyndelta.splits import (
    DEFAULT_SPLITS_DIR,
    REQUIRED_RESERVED_HOLDOUTS,
    RESERVED_HOLDOUT_SCHEMA,
    ReservedHoldoutError,
    assert_no_reserved_holdout_in_pairs,
    build_reserved_holdout_manifest,
    item_id,
    load_reserved_holdout_ids,
    verify_reserved_holdout_manifest,
    write_json_manifest,
)

pytestmark = pytest.mark.cpu

MANIFEST_NAME = "reason-gsm8k-test-holdout.json"
EXPECTED_ROWS = 1319
EXPECTED_MEMBERSHIP_SHA = "8aeb05279e083e2926958f644d7b72fd09dfd81bf45a2726cfeb5e6ebb3fa593"
GSM8K_TEST_SHARD = Path("/mnt/bulk/csd-corpus/reason/gsm8k-main/test.parquet")


def _committed() -> dict:
    return json.loads((DEFAULT_SPLITS_DIR / MANIFEST_NAME).read_text())


# ---------------------------------------------------------------------------------------
# The committed manifest itself.
# ---------------------------------------------------------------------------------------


def test_committed_holdout_manifest_is_intact() -> None:
    """1,319 unique ids that hash to the recorded sha, scoped to every region."""
    manifest = _committed()
    assert manifest["schema"] == RESERVED_HOLDOUT_SCHEMA
    assert manifest["source"] == "openai/gsm8k"
    assert manifest["split"] == "test"
    # `all`, not `reason`: a holdout reserved from one region only is still training
    # material for the composed model, which is exactly the leak this prevents.
    assert manifest["region_scope"] == "all"
    assert manifest["rows"] == EXPECTED_ROWS
    ids = verify_reserved_holdout_manifest(manifest, path=DEFAULT_SPLITS_DIR / MANIFEST_NAME)
    assert len(ids) == EXPECTED_ROWS
    assert len(set(ids)) == EXPECTED_ROWS
    assert manifest["sha256"] == EXPECTED_MEMBERSHIP_SHA


def test_committed_manifest_records_the_licence_verified_at_the_primary_source() -> None:
    """The contract's mirror->upstream discipline has to survive in the artefact."""
    manifest = _committed()
    assert "MIT" in manifest["licence"]
    assert "github.com/openai/grade-school-math" in manifest["upstream"]
    assert manifest["shard_sha256"] and len(manifest["shard_sha256"]) == 64


def test_load_reserved_holdout_ids_returns_the_committed_membership() -> None:
    assert len(load_reserved_holdout_ids()) == EXPECTED_ROWS


# ---------------------------------------------------------------------------------------
# The guard fires -- each with its control.
# ---------------------------------------------------------------------------------------


def _fixture_dir(tmp_path: Path, reserved: list[tuple[str, str]]) -> Path:
    """A splits dir holding one reserved-holdout manifest over `reserved`."""
    payload = build_reserved_holdout_manifest(
        name="fixture-holdout",
        source="fixture/source",
        split="test",
        region_scope="all",
        pair_columns=["question", "answer"],
        shard=str(tmp_path / "test.parquet"),
        shard_sha256="0" * 64,
        rows=len(reserved),
        item_ids=[item_id(a, b) for a, b in reserved],
        licence="MIT",
        upstream="https://example.invalid/LICENSE",
        notes="fixture",
    )
    write_json_manifest(tmp_path / "fixture-holdout.json", payload)
    for required in REQUIRED_RESERVED_HOLDOUTS:
        write_json_manifest(tmp_path / required, payload | {"name": required[: -len(".json")]})
    return tmp_path


def test_a_reserved_item_in_training_pairs_is_refused(tmp_path: Path) -> None:
    """THE LEAK TEST: one reserved row hidden among clean ones must refuse."""
    reserved = [("held out question", "held out answer #### 42")]
    splits_dir = _fixture_dir(tmp_path, reserved)
    clean = [(f"q{i}", f"a{i}") for i in range(50)]
    with pytest.raises(ReservedHoldoutError, match="reserved-holdout item"):
        assert_no_reserved_holdout_in_pairs(clean + reserved, splits_dir=splits_dir)


def test_clean_training_pairs_pass(tmp_path: Path) -> None:
    """The control for the test above: without the reserved row, nothing raises."""
    splits_dir = _fixture_dir(tmp_path, [("held out question", "held out answer #### 42")])
    assert_no_reserved_holdout_in_pairs(
        [(f"q{i}", f"a{i}") for i in range(50)], splits_dir=splits_dir
    )


def test_reformatted_reserved_item_is_still_caught(tmp_path: Path) -> None:
    """Whitespace and case cannot launder a reserved row past the id.

    `item_id` normalises before hashing, so re-wrapping a derivation -- which is exactly
    what a corpus re-export does -- does not produce a new item.
    """
    reserved = [("Held Out Question", "held out answer #### 42")]
    splits_dir = _fixture_dir(tmp_path, reserved)
    laundered = [("held   out\nquestion", "HELD OUT ANSWER   #### 42")]
    with pytest.raises(ReservedHoldoutError):
        assert_no_reserved_holdout_in_pairs(laundered, splits_dir=splits_dir)


def test_a_missing_required_manifest_refuses_rather_than_loading_nothing(tmp_path: Path) -> None:
    """Deleting the manifest must break the build, not silently disable the guard."""
    with pytest.raises(ReservedHoldoutError, match="required reserved-holdout manifest missing"):
        load_reserved_holdout_ids(splits_dir=tmp_path)


def test_a_doctored_membership_list_refuses(tmp_path: Path) -> None:
    """Dropping an id to make a row trainable must fail the sha, not pass quietly."""
    splits_dir = _fixture_dir(tmp_path, [("q", "a"), ("q2", "a2")])
    path = splits_dir / "fixture-holdout.json"
    payload = json.loads(path.read_text())
    payload["item_ids"] = payload["item_ids"][:1]
    path.write_text(json.dumps(payload))
    with pytest.raises(ReservedHoldoutError, match="doctored or truncated"):
        load_reserved_holdout_ids(splits_dir=splits_dir)


# ---------------------------------------------------------------------------------------
# The shard half: csd-train-all.py's HELD_OUT_SHARDS.
# ---------------------------------------------------------------------------------------


def _load_csd_train_all():
    """Import scripts/csd-train-all.py by path, as every consumer in this repo does."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "csd-train-all.py"
    spec = importlib.util.spec_from_file_location("csd_train_all_holdout_test", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "shard",
    [
        "/mnt/bulk/csd-corpus/reason/gsm8k-main/test.parquet",
        "/bulk/csd-corpus/reason/gsm8k-main/test.parquet",
        "/mnt/fleet-datasets/csd/region/reason/gsm8k-main/test.parquet",
    ],
)
def test_held_out_shard_is_refused_under_any_root(shard: str) -> None:
    """The reservation is on the trailing path, so the mount it resolved against is moot."""
    mod = _load_csd_train_all()
    with pytest.raises(mod.ReservedSourceError, match="held-out shard"):
        mod._refuse_reserved_shards("reason", "reason/gsm8k-main/*.parquet", [shard])


def test_the_train_split_of_the_same_directory_is_still_allowed() -> None:
    """The control: reserving by directory would have refused the region's own corpus."""
    mod = _load_csd_train_all()
    mod._refuse_reserved_shards(
        "reason",
        "reason/gsm8k-main/train.parquet",
        ["/mnt/bulk/csd-corpus/reason/gsm8k-main/train.parquet"],
    )


def test_a_widened_glob_that_picks_up_both_shards_still_refuses() -> None:
    """The realistic failure: `*.parquet` resolves train AND test; one bad shard is enough."""
    mod = _load_csd_train_all()
    with pytest.raises(mod.ReservedSourceError, match="held-out shard"):
        mod._refuse_reserved_shards(
            "reason",
            "reason/gsm8k-main/*.parquet",
            [
                "/mnt/bulk/csd-corpus/reason/gsm8k-main/train.parquet",
                "/mnt/bulk/csd-corpus/reason/gsm8k-main/test.parquet",
            ],
        )


def test_the_reason_region_does_not_declare_the_held_out_shard() -> None:
    """`REGIONS['reason']` must name `train.parquet` explicitly, never a glob."""
    mod = _load_csd_train_all()
    globs = [g for g, _cols, _cap in mod.region_spec("reason").sources]
    assert any(g.endswith("gsm8k-main/train.parquet") for g in globs)
    assert not any("test.parquet" in g or "gsm8k-main/*" in g for g in globs)


# ---------------------------------------------------------------------------------------
# Wiring into build_splits, plus the neutered-guard control.
# ---------------------------------------------------------------------------------------

pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from cogsyndelta.regions import pretrain as _pretrain  # noqa: E402
from cogsyndelta.regions.pretrain import PretrainConfig, build_splits  # noqa: E402


def _corpus_with(
    tmp_path: Path, extra: list[tuple[str, str]], n_clean: int = 200
) -> PretrainConfig:
    """A tiny training corpus, optionally seeded with rows that must never be trainable.

    `steps * batch_size` is deliberately larger than the row count: `_draw_split`'s budget
    caps how many pairs are loaded at all, and a reserved row that is never loaded proves
    nothing about the guard.
    """
    anchors = [f"clean question {i}" for i in range(n_clean)] + [a for a, _ in extra]
    positives = [f"clean answer {i}" for i in range(n_clean)] + [b for _, b in extra]
    shard = tmp_path / "pairs.parquet"
    pq.write_table(pa.table({"a": anchors, "b": positives}), shard)
    return PretrainConfig(
        region="holdout-g38-test",
        pair_columns=("a", "b"),
        shards=[str(shard)],
        steps=64,
        batch_size=8,
        holdout_pairs=8,
        seed=0,
        split_seed=0,
        order_seed=0,
    )


def _one_real_reserved_pair() -> tuple[str, str]:
    table = pq.read_table(str(GSM8K_TEST_SHARD), columns=["question", "answer"])
    return str(table.column("question")[0]), str(table.column("answer")[0])


needs_shard = pytest.mark.skipif(
    not GSM8K_TEST_SHARD.is_file(), reason="gsm8k test shard not on this host"
)


@needs_shard
def test_the_real_holdout_shard_is_exactly_the_committed_membership() -> None:
    """Every row of the landed shard is reserved -- no row is trainable by omission."""
    table = pq.read_table(str(GSM8K_TEST_SHARD), columns=["question", "answer"])
    ids = {
        item_id(str(q), str(a))
        for q, a in zip(
            table.column("question").to_pylist(),
            table.column("answer").to_pylist(),
            strict=True,
        )
    }
    assert len(ids) == EXPECTED_ROWS
    assert ids <= load_reserved_holdout_ids()


@needs_shard
def test_build_splits_refuses_a_corpus_containing_a_real_gsm8k_test_row(tmp_path: Path) -> None:
    """END TO END: a real held-out row reaching a training corpus refuses to train."""
    cfg = _corpus_with(tmp_path, [_one_real_reserved_pair()])
    with pytest.raises(ReservedHoldoutError, match="reserved-holdout item"):
        build_splits(cfg)


@needs_shard
def test_neutering_the_call_site_lets_the_leak_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE CONTROL. With the guard stubbed out, the identical corpus trains happily.

    This is the evidence that the test above is measuring the guard. Without it, a
    `build_splits` that raised for some unrelated reason -- a malformed shard, a dedup
    error -- would look exactly like a working holdout.
    """
    cfg = _corpus_with(tmp_path, [_one_real_reserved_pair()])
    monkeypatch.setattr(
        _pretrain, "assert_no_reserved_holdout_in_pairs", lambda *a, **k: None, raising=True
    )
    _holdout, train_pairs, _meta = build_splits(cfg)
    reserved = load_reserved_holdout_ids()
    leaked = [item_id(a, b) for a, b in train_pairs if item_id(a, b) in reserved]
    assert leaked, "expected the neutered guard to let the reserved row into training"


@needs_shard
def test_a_corpus_without_the_reserved_row_still_builds(tmp_path: Path) -> None:
    """The other control: the guard does not refuse a clean corpus."""
    _holdout, train_pairs, _meta = build_splits(_corpus_with(tmp_path, []))
    assert train_pairs
