"""E0 / G26: held-out membership is a loaded manifest, not the training seed.

Acceptance (reason-region diagnosis E0):
  1. two training seeds report byte-identical held-out membership and split.sha256
  2. a doctored manifest refuses
  3. a held-out item leaked into training refuses
  4. mutation proof: drawing with cfg.seed (bypassing the manifest / split_seed path)
     makes the identical-membership test go red
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

import pyarrow as pa
import pyarrow.parquet as pq

from cogsyndelta.regions.pretrain import (
    PretrainConfig,
    _draw_split,
    build_splits,
    write_split_and_order_manifests,
)
from cogsyndelta.splits import (
    SplitGuardError,
    item_id,
    load_json_manifest,
    membership_sha256,
    write_json_manifest,
)

pytestmark = pytest.mark.cpu


def _write_pairs(path: Path, n: int) -> None:
    anchors = [f"question number {i} about topic {i}" for i in range(n)]
    positives = [f"answer body {i} describing outcome {i}" for i in range(n)]
    pq.write_table(pa.table({"a": anchors, "b": positives}), path)


def _cfg(tmp_path: Path, *, seed: int, n: int = 80, **overrides: object) -> PretrainConfig:
    shard = tmp_path / "pairs.parquet"
    if not shard.is_file():
        _write_pairs(shard, n)
    kwargs: dict[str, object] = {
        "region": "split-g26-test",
        "pair_columns": ("a", "b"),
        "shards": [str(shard)],
        "steps": 4,
        "batch_size": 8,
        "holdout_pairs": 8,
        "seed": seed,
        "split_seed": 0,
        "order_seed": 0,
    }
    kwargs.update(overrides)
    return PretrainConfig(**kwargs)  # type: ignore[arg-type]


def test_two_training_seeds_share_byte_identical_holdout_membership(tmp_path: Path) -> None:
    """The E0 acceptance test: training seed must not touch membership."""
    holdout_0, train_0, meta_0 = build_splits(_cfg(tmp_path, seed=0))
    holdout_1, train_1, meta_1 = build_splits(_cfg(tmp_path, seed=1))
    ids_0 = [item_id(a, b) for a, b in holdout_0]
    ids_1 = [item_id(a, b) for a, b in holdout_1]
    assert ids_0 == ids_1
    assert holdout_0 == holdout_1
    assert meta_0["split"]["sha256"] == meta_1["split"]["sha256"]
    assert meta_0["split"]["sha256"] == membership_sha256(ids_0)
    # Train leftover is the same membership too (order_seed 0 = identity).
    assert [item_id(a, b) for a, b in train_0] == [item_id(a, b) for a, b in train_1]


def test_bypassing_manifest_load_makes_identical_membership_test_red(tmp_path: Path) -> None:
    """MUTATION PROOF: the pre-E0 draw used cfg.seed for the shuffle. With that path
    restored (membership_seed=cfg.seed, no manifest), two training seeds no longer share
    a holdout -- so the identical-membership test above goes red. If this assertion
    ever fails (the two seed-keyed draws agree), the acceptance test is tautological.
    """
    holdout_0, _, _ = _draw_split(_cfg(tmp_path, seed=0), membership_seed=0)
    holdout_1, _, _ = _draw_split(_cfg(tmp_path, seed=1), membership_seed=1)
    assert [item_id(a, b) for a, b in holdout_0] != [item_id(a, b) for a, b in holdout_1]


def test_doctored_manifest_refuses(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, seed=0)
    split_path, _order_path, _meta = write_split_and_order_manifests(
        cfg, splits_dir=tmp_path / "splits"
    )
    payload = load_json_manifest(split_path)
    payload["holdout_ids"][0] = "ab" * 32
    write_json_manifest(split_path, payload)
    cfg_req = _cfg(
        tmp_path,
        seed=0,
        split_manifest=str(split_path),
        require_split_manifest=True,
    )
    with pytest.raises(SplitGuardError, match="G26"):
        build_splits(cfg_req)


def test_doctored_manifest_with_recomputed_sha_still_refuses(tmp_path: Path) -> None:
    """A doctor who rewrites sha256 after swapping one pair is still caught: the draw
    from the corpus does not match the file."""
    cfg = _cfg(tmp_path, seed=0)
    split_path, _, _ = write_split_and_order_manifests(cfg, splits_dir=tmp_path / "splits")
    payload = load_json_manifest(split_path)
    payload["holdout_ids"][0] = "cd" * 32
    payload["sha256"] = membership_sha256(payload["holdout_ids"])
    write_json_manifest(split_path, payload)
    cfg_req = _cfg(
        tmp_path,
        seed=0,
        split_manifest=str(split_path),
        require_split_manifest=True,
    )
    with pytest.raises(SplitGuardError, match="does not match the split manifest"):
        build_splits(cfg_req)


def test_missing_required_manifest_refuses(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        seed=0,
        split_manifest=str(tmp_path / "no-such.json"),
        require_split_manifest=True,
    )
    with pytest.raises(SplitGuardError, match="missing"):
        build_splits(cfg)


def test_committed_manifest_is_loaded_and_stamped(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, seed=0)
    split_path, order_path, written = write_split_and_order_manifests(
        cfg, splits_dir=tmp_path / "splits"
    )
    loaded_cfg = _cfg(
        tmp_path,
        seed=9,
        split_manifest=str(split_path),
        order_manifest=str(order_path),
        require_split_manifest=True,
    )
    holdout, _train, meta = build_splits(loaded_cfg)
    assert meta["split"]["manifest"] == str(split_path)
    assert meta["split"]["sha256"] == written["split"]["sha256"]
    assert meta["split"]["seed"] == 0
    assert [item_id(a, b) for a, b in holdout] == load_json_manifest(split_path)["holdout_ids"]


def test_receipt_from_pretrain_stamps_split_fields(tmp_path: Path) -> None:
    pytest.importorskip("torch", reason="train group not installed")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer

    from cogsyndelta.regions.pretrain import pretrain_region
    from cogsyndelta.regions.text_encoder import TextEncoderConfig

    n = 40
    texts = [f"anchor number {i} word" for i in range(n)] + [
        f"anchor number {i} match" for i in range(n)
    ]
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = Whitespace()
    tok.train_from_iterator(texts, trainer=WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"]))
    tok_path = tmp_path / "tok.json"
    tok.save(str(tok_path))
    shard = tmp_path / "pairs.parquet"
    pq.write_table(
        pa.table({"anchor": texts[:n], "positive": texts[n:]}),
        shard,
    )
    receipts = []
    for seed, run_id in ((0, "s0"), (1, "s1")):
        cfg = PretrainConfig(
            region="split-receipt-test",
            pair_columns=("anchor", "positive"),
            shards=[str(shard)],
            steps=2,
            batch_size=4,
            holdout_pairs=4,
            eval_every=2,
            checkpoint_every=0,
            max_len=16,
            seed=seed,
            split_seed=0,
            device="cpu",
            encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=16),
            tokenizer_path=str(tok_path),
            out_dir=str(tmp_path / f"out-{run_id}"),
        )
        receipts.append(pretrain_region(cfg))
    assert receipts[0]["split"]["sha256"] == receipts[1]["split"]["sha256"]
    assert receipts[0]["split"]["seed"] == 0
    assert receipts[0]["config"]["seed"] == 0
    assert receipts[1]["config"]["seed"] == 1
    assert receipts[0]["batch_order"]["sha256"] == receipts[1]["batch_order"]["sha256"]


def test_benchmark_refuses_receipt_whose_split_sha_differs(tmp_path: Path) -> None:
    """G26 benchmark side: scoring a receipt against a different split.sha256 refuses."""
    pytest.importorskip("torch", reason="train group not installed")
    import importlib.util
    import json
    import sys
    from typing import ClassVar

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer

    from cogsyndelta.regions.pretrain import pretrain_region
    from cogsyndelta.regions.text_encoder import TextEncoderConfig
    from cogsyndelta.splits import SplitGuardError

    n = 40
    texts = [f"anchor number {i} word" for i in range(n)] + [
        f"anchor number {i} match" for i in range(n)
    ]
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = Whitespace()
    tok.train_from_iterator(texts, trainer=WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"]))
    tok_path = tmp_path / "tok.json"
    tok.save(str(tok_path))
    shard = tmp_path / "pairs.parquet"
    pq.write_table(pa.table({"anchor": texts[:n], "positive": texts[n:]}), shard)
    cfg = PretrainConfig(
        region="bench-g26-test",
        pair_columns=("anchor", "positive"),
        shards=[str(shard)],
        steps=2,
        batch_size=4,
        holdout_pairs=4,
        eval_every=2,
        checkpoint_every=2,
        max_len=16,
        seed=0,
        device="cpu",
        encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=16),
        tokenizer_path=str(tok_path),
        out_dir=str(tmp_path / "receipts"),
    )
    receipt = pretrain_region(cfg)
    receipt_path = Path(receipt["receipt_path"])
    body = json.loads(receipt_path.read_text())
    body["split"]["sha256"] = "00" * 32
    receipt_path.write_text(json.dumps(body, indent=2) + "\n")

    bench_path = Path(__file__).resolve().parents[1] / "scripts" / "csd-benchmark.py"
    spec = importlib.util.spec_from_file_location("csd_benchmark_g26", bench_path)
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)

    class _Entry:
        sources: ClassVar = [("pairs.parquet", ("anchor", "positive"), 0)]
        root = tmp_path

    bench._regions_spec = lambda: {  # type: ignore[method-assign]
        "REGIONS": {},
        "_shards": lambda *a, **k: [str(shard)],
        "region_spec": lambda name: _Entry(),
    }
    with pytest.raises(SplitGuardError, match=r"split\.sha256"):
        bench.benchmark_region("bench-g26-test", tmp_path)
