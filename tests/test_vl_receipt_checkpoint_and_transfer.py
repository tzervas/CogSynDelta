"""Visual train receipts name the transfer set and bind the checkpoint bytes."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from cogsyndelta.model.vl_jepa import JEPAConfig
from cogsyndelta.regions.vl_pretrain import VLPretrainConfig, pretrain_vl_region

pytestmark = pytest.mark.cpu


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (32, 64, 128)).save(buf, format="PNG")
    return buf.getvalue()


_TINY_JEPA = JEPAConfig(
    image_size=32,
    patch_size=8,
    dim=32,
    depth=1,
    n_heads=4,
    predictor_dim=16,
    predictor_depth=1,
    n_target_blocks=1,
)


def _zip_pngs(path: Path, n: int, folder: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _png_bytes()
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(n):
            zf.writestr(f"{folder}/{i:02d}.png", payload)
    return path


def _cfg(tmp: Path) -> VLPretrainConfig:
    train = _zip_pngs(tmp / "train.zip", n=4, folder="cls")
    eurosat = _zip_pngs(tmp / "eurosat.zip", n=4, folder="forest")
    fashion = _zip_pngs(tmp / "fashion.zip", n=4, folder="coat")
    return VLPretrainConfig(
        region="visual",
        train_shards=[str(train)],
        probe_train_shards=[str(eurosat)],
        probe_eval_shards=[str(eurosat)],
        transfer_shards=[str(fashion)],
        steps=2,
        batch_size=2,
        seed=0,
        device="cpu",
        warmup_steps=1,
        eval_every=1,
        checkpoint_every=1,
        probe_steps=2,
        probe_limit=4,
        jepa=_TINY_JEPA,
        out_dir=str(tmp / "receipts"),
        cache_dir=str(tmp / "cache"),
        image_backend="png_zip",
        corpus_source="visual-clean-v1",
        probe_set_names=["eurosat-test", "fashion-t10k"],
        probe_sets=[
            {
                "name": "eurosat-test",
                "source": "phelber/eurosat-rgb-128",
                "role": "primary",
            },
            {
                "name": "fashion-t10k",
                "source": "zalando/fashion-mnist",
                "role": "transfer",
            },
        ],
    )


def test_receipt_binds_checkpoint_and_names_transfer(tmp_path: Path) -> None:
    receipt = pretrain_vl_region(_cfg(tmp_path))
    ckpt = Path(receipt["checkpoint"])
    assert ckpt.is_file()
    assert ckpt.name == "step-2.pt"
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert receipt["checkpoint_sha256"] == digest

    assert receipt["held_out"]["source"] == "phelber/eurosat-rgb-128"
    assert receipt["held_out"]["name"] == "eurosat-test"
    assert receipt["untrained_baseline"]["name"] == "eurosat-test"
    assert receipt["transfer"]["source"] == "zalando/fashion-mnist"
    assert receipt["transfer"]["name"] == "fashion-t10k"
    assert receipt["untrained_transfer"]["name"] == "fashion-t10k"


def test_parquet_fixture_without_probe_sets_does_not_invent_cifar(tmp_path: Path) -> None:
    """Empty probe_sets must not stamp a stale cifar100 label."""
    train = _zip_pngs(tmp_path / "train.zip", n=4, folder="cls")
    cfg = VLPretrainConfig(
        region="visual",
        train_shards=[str(train)],
        probe_train_shards=[str(train)],
        probe_eval_shards=[str(train)],
        steps=2,
        batch_size=2,
        seed=0,
        device="cpu",
        warmup_steps=1,
        eval_every=1,
        checkpoint_every=1,
        probe_steps=2,
        jepa=_TINY_JEPA,
        out_dir=str(tmp_path / "receipts"),
        cache_dir=str(tmp_path / "cache"),
        image_backend="png_zip",
    )
    receipt = pretrain_vl_region(cfg)
    assert "source" not in receipt["held_out"]
    assert "name" not in receipt["held_out"]
    assert receipt["transfer"] is None
