"""Visual eval/quantize stages: harness-shaped receipts, sha bind, mutation KeyError."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from cogsyndelta.model.vl_jepa import JEPAConfig
from cogsyndelta.regions._checkpoint import ChecksumMismatchError
from cogsyndelta.regions.vl_pretrain import VLPretrainConfig, pretrain_vl_region

pytestmark = pytest.mark.cpu

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG")
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


def _train(tmp: Path) -> tuple[dict, Path, Path]:
    state = tmp / "state"
    receipts = state / "receipts"
    train = _zip_pngs(tmp / "train.zip", n=4, folder="cls")
    eurosat = _zip_pngs(tmp / "eurosat.zip", n=4, folder="forest")
    fashion = _zip_pngs(tmp / "fashion.zip", n=4, folder="coat")
    cfg = VLPretrainConfig(
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
        out_dir=str(receipts),
        cache_dir=str(state / "vl-cache"),
        image_backend="png_zip",
        corpus_source="visual-clean-v1",
        probe_sets=[
            {"name": "eurosat-test", "source": "phelber/eurosat-rgb-128", "role": "primary"},
            {"name": "fashion-t10k", "source": "zalando/fashion-mnist", "role": "transfer"},
        ],
    )
    receipt = pretrain_vl_region(cfg)
    train_path = Path(receipt["receipt_path"])
    return receipt, state, train_path


def _load_benchmark():
    path = SCRIPTS / "csd-benchmark.py"
    spec = importlib.util.spec_from_file_location("csd_benchmark_visual_eval_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_quantize():
    path = SCRIPTS / "csd-quantize.py"
    spec = importlib.util.spec_from_file_location("csd_quantize_visual_eval_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_visual_eval_writes_gates_and_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Harness-shaped invocation. Mutation: drop `_is_visual_region` dispatch in
    `benchmark_region` and this raises KeyError: no REGIONS entry for 'visual'."""
    receipt, state, train_path = _train(tmp_path)
    assert "probe_protocol" in receipt
    assert receipt.get("probe") is not True
    bench = _load_benchmark()
    rec = bench.benchmark_region("visual", state, train_receipt_path=train_path)
    assert rec is not None
    assert rec.kind == "eval"
    assert rec.stage == "eval"
    assert "beats_untrained_eval" in rec.gates
    assert "not_collapsed" in rec.gates
    assert (
        rec.artifacts["checkpoint_sha256"]
        == hashlib.sha256(Path(rec.artifacts["checkpoint"]).read_bytes()).hexdigest()
    )
    assert "path" in rec.artifacts["source_training_receipt"]
    assert rec.metrics["probe.top1"] >= 0.0
    assert rec.provenance["probe_repeatability"] == "not-bitwise"
    printed = capsys.readouterr().out
    assert "untrained top1" in printed
    assert "rep_std" in printed
    out = rec.write(state / "receipts")
    assert out.name.startswith("cogsyndelta-visual-eval-")


def test_wrong_checkpoint_sha256_refuses(tmp_path: Path) -> None:
    receipt, state, train_path = _train(tmp_path)
    receipt["checkpoint_sha256"] = "0" * 64
    train_path.write_text(json.dumps(receipt))
    bench = _load_benchmark()
    with pytest.raises(ChecksumMismatchError):
        bench.benchmark_region("visual", state, train_receipt_path=train_path)


def test_text_region_spec_still_refuses_visual() -> None:
    bench = _load_benchmark()
    with pytest.raises(KeyError, match="no REGIONS entry for 'visual'"):
        bench._regions_spec()["region_spec"]("visual")


def test_visual_quantize_writes_within_budget(tmp_path: Path) -> None:
    _receipt, state, train_path = _train(tmp_path)
    quant = _load_quantize()
    rec = quant.quantize_visual_region(
        "visual", state, tolerance=1.0, aggressive=8, max_bits=8, train_receipt_path=train_path
    )
    assert rec["within_budget"] is True
    assert Path(rec["artifacts"]["quantized_path"]).is_file()
    assert rec["artifacts"]["checkpoint_sha256"]
