"""W7v-cfg (4): the visual receipt stamps `corpus.fingerprint`/`fingerprint_scheme`
like the text regions, and states the truth about masking.

Reuses `cogsyndelta.corpus.fingerprint_corpus`/`CORPUS_FINGERPRINT_SCHEME` -- the same
helper and scheme `regions/pretrain.py` stamps on text receipts -- rather than inventing
a visual-only scheme. `masking: random-permutation` states what `sample_masks`
(`vl_jepa.py`) actually does (`torch.randperm`), not I-JEPA's paper spatial multi-block
sampling; this file does NOT implement multi-block masking (`g8-visual/S02.md` §9).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow", reason="train group not installed")

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from cogsyndelta.corpus import CORPUS_FINGERPRINT_SCHEME, fingerprint_corpus
from cogsyndelta.model.vl_jepa import JEPAConfig
from cogsyndelta.regions.vl_pretrain import VLPretrainConfig, pretrain_vl_region

pytestmark = pytest.mark.cpu


def _png_bytes(size: int, fill: int) -> bytes:
    arr = np.full((size, size, 3), fill, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _write_image_parquet(path: Path, n: int, size: int, n_classes: int = 2) -> None:
    images = [_png_bytes(size, (i * 40) % 256) for i in range(n)]
    labels = [i % n_classes for i in range(n)]
    pq.write_table(pa.table({"image": images, "label": labels}), path)


@pytest.fixture
def tiny_cfg(tmp_path: Path) -> VLPretrainConfig:
    shard = tmp_path / "images.parquet"
    _write_image_parquet(shard, n=8, size=16)
    jepa_cfg = JEPAConfig(
        image_size=16, patch_size=4, dim=8, depth=1, n_heads=2, predictor_dim=4, predictor_depth=1
    )
    return VLPretrainConfig(
        region="vl-fingerprint-test",
        train_shards=[str(shard)],
        probe_train_shards=[str(shard)],
        probe_eval_shards=[str(shard)],
        transfer_shards=[],
        steps=2,
        batch_size=4,
        warmup_steps=1,
        eval_every=1,
        checkpoint_every=1,
        probe_steps=2,
        probe_limit=8,
        seed=0,
        device="cpu",
        jepa=jepa_cfg,
        cache_dir=str(tmp_path / "vl-cache"),
        out_dir=str(tmp_path / "run"),
    )


def test_receipt_stamps_corpus_fingerprint_and_scheme(tiny_cfg: VLPretrainConfig) -> None:
    receipt = pretrain_vl_region(tiny_cfg)
    assert "corpus" in receipt
    assert receipt["corpus"]["fingerprint_scheme"] == CORPUS_FINGERPRINT_SCHEME
    expected = fingerprint_corpus(
        tiny_cfg.train_shards, columns=[tiny_cfg.image_column, tiny_cfg.label_column]
    )
    assert receipt["corpus"]["fingerprint"] == expected


def test_corpus_fingerprint_uses_the_shared_helper_no_new_scheme(
    tiny_cfg: VLPretrainConfig,
) -> None:
    """Same scheme string the text harness stamps (regions/pretrain.py) -- proves this
    is the reused helper, not a visual-only reinvention."""
    receipt = pretrain_vl_region(tiny_cfg)
    assert receipt["corpus"]["fingerprint_scheme"] == "csd-corpus-fp/v2"


def test_receipt_prints_the_true_masking_scheme(tiny_cfg: VLPretrainConfig) -> None:
    receipt = pretrain_vl_region(tiny_cfg)
    assert receipt["masking"] == "random-permutation"


def test_fingerprint_changes_when_the_train_shard_content_changes(
    tmp_path: Path, tiny_cfg: VLPretrainConfig
) -> None:
    """A real property of `fingerprint_corpus`, exercised through the receipt path:
    different bytes at the same path name a different corpus."""
    receipt_a = pretrain_vl_region(tiny_cfg)

    other_shard = tmp_path / "images2.parquet"
    _write_image_parquet(other_shard, n=8, size=16, n_classes=3)
    from dataclasses import replace

    cfg_b = replace(
        tiny_cfg,
        region="vl-fingerprint-test-b",
        train_shards=[str(other_shard)],
        probe_train_shards=[str(other_shard)],
        probe_eval_shards=[str(other_shard)],
        out_dir=str(tmp_path / "run-b"),
        cache_dir=str(tmp_path / "vl-cache-b"),
    )
    receipt_b = pretrain_vl_region(cfg_b)

    assert receipt_a["corpus"]["fingerprint"] != receipt_b["corpus"]["fingerprint"]
