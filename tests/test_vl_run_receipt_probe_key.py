"""run_vl_region must not treat a full receipt's protocol block as GPU_PACK_PROBE.

The peak marker is top-level ``probe: True``. A dict on that key is truthy and used to
read ``receipt['steps']``, which only a peak receipt has.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.cpu

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_train_all():
    path = SCRIPTS / "csd-train-all.py"
    spec = importlib.util.spec_from_file_location("csd_train_all_probe_key_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _full_receipt(*, protocol_key: str = "probe_protocol") -> dict[str, Any]:
    """A full-shaped visual train receipt: no top-level ``steps``, not a peak marker."""
    block = {
        "jepa_train_shards": [],
        "linear_train_shards": [],
        "linear_eval_shards": [],
        "transfer_shards": [],
    }
    return {
        "region": "visual",
        protocol_key: block,
        "untrained_baseline": {"top1": 0.10, "top5": 0.30, "rep_std": 0.02},
        "held_out": {"top1": 0.50, "top5": 0.80, "rep_std": 0.05},
        "untrained_transfer": {"top1": 0.20, "name": "fashion-t10k"},
        "transfer": {"top1": 0.40, "name": "fashion-t10k"},
        "collapse_ratio": 2.5,
        "collapsed": False,
        "beats_untrained": {"probe_top1": True, "not_collapsed": True},
        "parameters": 1_000_000,
        "checkpoint": "visual-checkpoints/step-24.pt",
        "checkpoint_sha256": "ab" * 32,
    }


def _wire_fake(mod: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, receipt: dict) -> None:
    monkeypatch.setitem(mod.VL_REGIONS["visual"], "corpus_source", "test-corpus")
    monkeypatch.setitem(mod.VL_REGIONS["visual"], "manifest", None)
    monkeypatch.setattr(mod, "_shards", lambda pattern, root=None: [str(tmp_path / "x.parquet")])
    monkeypatch.setattr("cogsyndelta.regions.vl_pretrain.pretrain_vl_region", lambda cfg: receipt)


def test_full_receipt_prints_and_returns_beats_untrained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mutation: write the protocol block as top-level `probe` and this raises KeyError steps."""
    mod = _load_train_all()
    receipt = _full_receipt(protocol_key="probe_protocol")
    _wire_fake(mod, monkeypatch, tmp_path, receipt)
    out = mod.run_vl_region(name="visual", state=tmp_path, steps=24, batch=16, dry=False, seed=0)
    assert out is not None
    assert out["beats_untrained"]["probe_top1"] is True
    printed = capsys.readouterr().out
    assert "probe_peak" not in printed
    assert "untrained top1" in printed
    assert "beats_untrained=" in printed
    assert "KeyError" not in printed


def test_probe_true_still_short_circuits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load_train_all()
    peak = {"probe": True, "region": "visual", "steps": 20, "batch_size": 2}
    _wire_fake(mod, monkeypatch, tmp_path, peak)
    out = mod.run_vl_region(name="visual", state=tmp_path, steps=4000, batch=128, dry=False, seed=0)
    assert out is not None
    assert out["probe"] is True
    printed = capsys.readouterr().out
    assert "probe_peak" in printed
    assert "untrained top1" not in printed


def test_old_probe_dict_key_raises_keyerror_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-fix receipt key. ``if receipt.get("probe"):`` is truthy for a dict."""
    mod = _load_train_all()
    receipt = _full_receipt(protocol_key="probe")
    _wire_fake(mod, monkeypatch, tmp_path, receipt)
    with pytest.raises(KeyError, match="steps"):
        mod.run_vl_region(name="visual", state=tmp_path, steps=24, batch=16, dry=False, seed=0)
