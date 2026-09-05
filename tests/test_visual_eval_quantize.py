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
import torch
from PIL import Image

from cogsyndelta.model.vl_jepa import IJEPA, JEPAConfig
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
    assert rec.provenance["quantized_module"] == "target_encoder"
    ijepa = IJEPA(JEPAConfig(**receipt["config"]["jepa"]))
    assert rec.provenance["parameters"] == sum(p.numel() for p in ijepa.target_encoder.parameters())
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
    assert rec["artifacts"]["quantized_module"] == "target_encoder"


_VISUAL_QUANT_PROBE_KEYS = frozenset(
    {"quant.plan_probe_top1", "quant.drop_probe_top1", "quant.compression_ratio"}
)


def test_visual_quant_receipt_keys_are_probe_named_not_recall(tmp_path: Path) -> None:
    """Visual quant receipts name probe top-1. Mutation: write quant.plan_recall@1
    again and this finds a recall@1 key."""
    _receipt, state, train_path = _train(tmp_path)
    quant = _load_quantize()
    rec = quant.quantize_visual_region(
        "visual", state, tolerance=1.0, aggressive=8, max_bits=8, train_receipt_path=train_path
    )
    assert rec.keys() >= _VISUAL_QUANT_PROBE_KEYS
    assert not any("recall@1" in k for k in rec)


def test_visual_eval_quantized_uses_artifact_probe_top1(tmp_path: Path) -> None:
    _receipt, state, train_path = _train(tmp_path)
    quant = _load_quantize()
    qrec = quant.quantize_visual_region(
        "visual", state, tolerance=1.0, aggressive=8, max_bits=8, train_receipt_path=train_path
    )
    bench = _load_benchmark()
    rec = bench.benchmark_visual_region_quantized(
        "visual",
        state,
        Path(qrec["artifacts"]["quantized_path"]),
        train_receipt_path=train_path,
    )
    assert "quant.artifact_probe_top1" in rec.metrics
    assert "quant.artifact_recall@1" not in rec.metrics
    assert rec.metrics["quant.artifact_probe_top1"] == rec.metrics["probe.top1"]


def test_mixed_quantize_summary_print_does_not_raise() -> None:
    """A mixed text+visual results list must not KeyError on the plan key."""
    quant = _load_quantize()
    text = {
        "region": "language",
        "quant.compression_ratio": 2.0,
        "fp32_metric_recomputed": 0.9,
        "quant.plan_recall@1": 0.88,
        "within_budget": True,
    }
    vis = {
        "region": "visual",
        "quant.compression_ratio": 3.0,
        "fp32_metric_recomputed": 0.7,
        "quant.plan_probe_top1": 0.65,
        "within_budget": True,
    }
    assert "0.8800" in quant._format_quant_summary(text)
    assert "0.6500" in quant._format_quant_summary(vis)
    with pytest.raises(KeyError):
        quant._format_quant_summary(
            {
                "region": "visual",
                "quant.compression_ratio": 3.0,
                "fp32_metric_recomputed": 0.7,
                "within_budget": True,
            }
        )


def _packed_param_names(path: Path) -> set[str]:
    packed = torch.load(path, weights_only=True, map_location="cpu")
    return set(packed["fp32"]) | set(packed["bits"])


def test_quantized_artifact_contains_only_target_encoder_keys(tmp_path: Path) -> None:
    """Packed artifact is the EMA target encoder. Mutation: pack the full IJEPA
    and this finds encoder.* / predictor.* keys."""
    _receipt, state, train_path = _train(tmp_path)
    quant = _load_quantize()
    rec = quant.quantize_visual_region(
        "visual", state, tolerance=1.0, aggressive=8, max_bits=8, train_receipt_path=train_path
    )
    names = _packed_param_names(Path(rec["artifacts"]["quantized_path"]))
    assert names
    assert all(n.startswith("target_encoder.") for n in names)
    assert not any(n.startswith("encoder.") or n.startswith("predictor.") for n in names)


def test_quantized_eval_reproduces_probe_and_parameter_count(tmp_path: Path) -> None:
    receipt, state, train_path = _train(tmp_path)
    quant = _load_quantize()
    qrec = quant.quantize_visual_region(
        "visual", state, tolerance=1.0, aggressive=8, max_bits=8, train_receipt_path=train_path
    )
    bench = _load_benchmark()
    rec = bench.benchmark_visual_region_quantized(
        "visual",
        state,
        Path(qrec["artifacts"]["quantized_path"]),
        train_receipt_path=train_path,
    )
    assert rec.kind == "eval-quantized"
    assert "probe.top1" in rec.metrics
    ijepa = IJEPA(JEPAConfig(**receipt["config"]["jepa"]))
    expected = sum(p.numel() for p in ijepa.target_encoder.parameters())
    full = sum(p.numel() for p in ijepa.parameters())
    assert expected < full
    assert rec.provenance["parameters"] == expected
    assert rec.provenance["quantized_module"] == "target_encoder"
    fp32 = bench.benchmark_visual_region("visual", state, train_receipt_path=train_path)
    assert fp32 is not None
    assert fp32.provenance["parameters"] == expected
    assert fp32.provenance["fp32_reference_bytes"] == expected * 4


def test_packed_artifact_does_not_load_onto_full_ijepa(tmp_path: Path) -> None:
    receipt, state, train_path = _train(tmp_path)
    quant = _load_quantize()
    qrec = quant.quantize_visual_region(
        "visual", state, tolerance=1.0, aggressive=8, max_bits=8, train_receipt_path=train_path
    )
    from cogsyndelta.quant.ptq import load_packed_artifact, unpack_state_dict

    sd = unpack_state_dict(load_packed_artifact(qrec["artifacts"]["quantized_path"]))
    model = IJEPA(JEPAConfig(**receipt["config"]["jepa"]))
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(sd)


def _counting_splits(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    import cogsyndelta.regions.vl_pretrain as vl

    counts = {"n": 0}
    orig = vl._load_visual_splits

    def counted(*args: object, **kwargs: object) -> object:
        counts["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(vl, "_load_visual_splits", counted)
    return counts


def _cpu_visual_model(receipt: dict):
    from cogsyndelta.model.vl_jepa import IJEPA
    from cogsyndelta.regions._checkpoint import load_checkpoint

    cfg_jepa = receipt["config"]["jepa"]
    from cogsyndelta.model.vl_jepa import JEPAConfig

    model = IJEPA(JEPAConfig(**cfg_jepa))
    ck = load_checkpoint(
        receipt["checkpoint"],
        expected_sha256=receipt["checkpoint_sha256"],
        map_location="cpu",
    )
    model.load_state_dict(ck["model"])
    return model


def test_splits_loaded_once_across_eval_fn_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N eval_fn calls share one load. Mutation: drop the splits argument so
    `_measure_visual_probes` reloads, and counts['n'] becomes N."""
    receipt, _state, _train_path = _train(tmp_path)
    bench = _load_benchmark()
    counts = _counting_splits(monkeypatch)
    cfg = bench._vl_cfg_from_train_receipt("visual", receipt)
    splits = bench.load_visual_splits(cfg)
    assert counts["n"] == 1
    model = _cpu_visual_model(receipt)
    device = torch.device("cpu")
    n_eval = 3
    for _ in range(n_eval):
        bench._measure_visual_probes(model, cfg, device, splits)
    assert counts["n"] == 1


def test_mutation_per_call_load_is_n(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt, _state, _train_path = _train(tmp_path)
    bench = _load_benchmark()
    counts = _counting_splits(monkeypatch)
    cfg = bench._vl_cfg_from_train_receipt("visual", receipt)
    model = _cpu_visual_model(receipt)
    device = torch.device("cpu")
    n_eval = 3
    for _ in range(n_eval):
        bench._measure_visual_probes(model, cfg, device)
    assert counts["n"] == n_eval
