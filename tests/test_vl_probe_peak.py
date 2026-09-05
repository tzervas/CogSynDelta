"""GPU_PACK_PROBE on the visual trainer: short train loop, no receipt, no checkpoint.

CPU only. Synthetic 1x1 PNG zip; tiny JEPA so the 20-step probe is a unit test, not a
GPU job. Mutation: drop probe_requested() and this writes a receipt + checkpoint.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from cogsyndelta.model.vl_jepa import JEPAConfig
from cogsyndelta.regions.vl_pretrain import VLPretrainConfig, pretrain_vl_region
from cogsyndelta.util.gpu_budget import PROBE_STEPS

pytestmark = pytest.mark.cpu


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (128, 64, 32)).save(buf, format="PNG")
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


def _zip_pngs(path: Path, n: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _png_bytes()
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(n):
            zf.writestr(f"cls/{i:02d}.png", payload)
    return path


def _cfg(tmp: Path, *, steps: int = 4000, batch: int = 2) -> VLPretrainConfig:
    train = _zip_pngs(tmp / "train.zip", n=4)
    return VLPretrainConfig(
        region="visual",
        train_shards=[str(train)],
        probe_train_shards=[str(train)],
        probe_eval_shards=[str(train)],
        steps=steps,
        batch_size=batch,
        seed=0,
        device="cpu",
        jepa=_TINY_JEPA,
        out_dir=str(tmp / "receipts"),
        cache_dir=str(tmp / "cache"),
        image_backend="png_zip",
        corpus_source="visual-clean-v1",
    )


def test_probe_env_caps_steps_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_PACK_PROBE", "1")
    cfg = _cfg(tmp_path, steps=4000)
    result = pretrain_vl_region(cfg)
    assert result["probe"] is True
    assert result["steps"] == PROBE_STEPS
    assert result["steps"] < 4000
    out = Path(cfg.out_dir)
    assert not out.exists() or list(out.glob("*.json")) == []
    ckpt = out / "visual-checkpoints"
    assert not ckpt.exists() or list(ckpt.glob("*.pt")) == []


def test_without_probe_env_a_short_run_still_writes_a_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation guard: the probe skip is env-gated, not an unconditional early return."""
    monkeypatch.delenv("GPU_PACK_PROBE", raising=False)
    cfg = _cfg(tmp_path, steps=1, batch=2)
    cfg.probe_steps = 1
    cfg.probe_limit = 4
    result = pretrain_vl_region(cfg)
    assert "probe" not in result or result.get("probe") is not True
    assert "untrained_baseline" in result
    receipts = list(Path(cfg.out_dir).glob("visual-*.json"))
    assert len(receipts) == 1


def test_run_vl_region_probe_does_not_print_train_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib.util
    import sys

    monkeypatch.setenv("GPU_PACK_PROBE", "1")
    path = Path(__file__).resolve().parents[1] / "scripts" / "csd-train-all.py"
    spec = importlib.util.spec_from_file_location("csd_train_all_probe_test", path)
    assert spec is not None and spec.loader is not None
    train_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = train_mod
    spec.loader.exec_module(train_mod)

    captured: list[object] = []

    def fake_pretrain(cfg: object) -> dict:
        captured.append(cfg)
        return {"probe": True, "region": "visual", "steps": PROBE_STEPS, "batch_size": 2}

    monkeypatch.setattr("cogsyndelta.regions.vl_pretrain.pretrain_vl_region", fake_pretrain)
    monkeypatch.setitem(train_mod.VL_REGIONS["visual"], "manifest", None)
    monkeypatch.setitem(train_mod.VL_REGIONS["visual"], "corpus_source", "visual-clean-v1")
    monkeypatch.setattr(
        train_mod, "_shards", lambda pattern, root=None: [str(tmp_path / "x.parquet")]
    )

    receipt = train_mod.run_vl_region(
        name="visual", state=tmp_path, steps=4000, batch=128, dry=False, seed=0
    )
    assert receipt is not None
    assert receipt["probe"] is True
    printed = capsys.readouterr().out
    assert "probe_peak" in printed
    assert "untrained top1" not in printed
    assert captured, "run_vl_region must still construct VLPretrainConfig and call pretrain"
