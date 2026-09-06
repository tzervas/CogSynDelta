"""End-to-end coverage for `_quant_geometry_metrics` wired into BOTH
`benchmark_region_quantized` (text) and `benchmark_visual_region_quantized` (visual).

Three layers, cheapest first:

1. `_quant_geometry_metrics` in isolation -- tiny synthetic latents, fabricated shas,
   NO checkpoint file anywhere, no real model. `test_benchmark_metrics_v2_receipt.py` /
   `test_visual_eval_quantize.py` already cover the surrounding receipt shape for
   OTHER fields; this module's own `tests/test_eval_geometry.py` already covers
   `compute_geometry`/`verify_geometry_reference` in isolation. What was missing --
   the gap this file closes -- is a test of the WIRING FUNCTION `scripts/csd-
   benchmark.py` actually defines and both branches call, proving its return shape is
   exactly what a receipt's `metrics`/`provenance["quant.geometry.reference"]` get.
2. A real, tiny (CPU, sub-second) `pretrain_region` + `quantize_text_region` +
   `benchmark_region_quantized` pass for the TEXT branch -- the ACTUAL Receipt object,
   not a stand-in -- proving the wiring merges `quant.geometry.*` into a real receipt,
   and that stripping the checkpoint_sha256 the wiring binds to (the
   `--allow-unbound-train-receipt` case) skips the block cleanly rather than crashing
   or loading an unverified checkpoint.
3. The visual-branch equivalent of (2). Visual has no "unbound" case to test --
   `require_bound_visual_train_receipt` never returns without a sha (MM §23(j)) -- so
   layer 1's shared-helper coverage plus this populated case is what "both call
   sites" means for visual.
"""

from __future__ import annotations

import importlib.util
import inspect
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import ClassVar

import pytest
import torch

pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from cogsyndelta.model.vl_jepa import JEPAConfig
from cogsyndelta.regions.pretrain import PretrainConfig, pretrain_region
from cogsyndelta.regions.text_encoder import TextEncoderConfig
from cogsyndelta.regions.vl_pretrain import VLPretrainConfig, pretrain_vl_region

pytestmark = pytest.mark.cpu

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

_GEOMETRY_KEYS = {
    "quant.geometry.mean_cosine",
    "quant.geometry.min_cosine",
    "quant.geometry.p05_cosine",
    "quant.geometry.nn_agreement_at_10",
    "quant.geometry.latent_std_ratio",
}


def _load(name: str, filename: str):
    """Import a hyphenated `scripts/*.py` module -- see
    tests/test_benchmark_checkpoint_sha_receipt.py for why this indirection exists."""
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bench = _load("csd_benchmark_geometry_wiring_test", "csd-benchmark.py")
quant = _load("csd_quantize_geometry_wiring_test", "csd-quantize.py")


# =====================================================================================
# Layer 1: the shared helper, synthetic latents only.
# =====================================================================================


def test_both_benchmark_branches_call_the_same_geometry_helper() -> None:
    """Guards the premise every test below relies on: if a future refactor gives each
    branch its own private geometry wiring, this stops silently covering only one."""
    text_src = inspect.getsource(bench.benchmark_region_quantized)
    visual_src = inspect.getsource(bench.benchmark_visual_region_quantized)
    assert "_quant_geometry_metrics(" in text_src
    assert "_quant_geometry_metrics(" in visual_src


def test_quant_geometry_metrics_populates_receipt_shaped_fields() -> None:
    """Tiny synthetic latents (n=20, over DEFAULT_NN_K), no checkpoint file, no real
    model -- `metrics`/`reference` here are exactly what BOTH call sites merge
    verbatim into a receipt (`metrics.update(geometry_metrics)`;
    `provenance["quant.geometry.reference"] = geometry_reference`)."""
    torch.manual_seed(0)
    fp32_latents = torch.randn(20, 6)
    quantized_latents = fp32_latents + 0.01 * torch.randn(20, 6)

    metrics, reference = bench._quant_geometry_metrics(
        fp32_latents=fp32_latents,
        quantized_latents=quantized_latents,
        split_sha256="synthetic-split-abc",
        checkpoint_sha256="synthetic-checkpoint-sha",
        quantized_sha256="synthetic-quantized-sha",
        source_training_receipt={"path": "nowhere.json", "sha256": "0" * 64},
    )

    assert set(metrics) == _GEOMETRY_KEYS
    assert 0.9 < metrics["quant.geometry.mean_cosine"] <= 1.0
    assert reference == {
        "checkpoint_sha256": "synthetic-checkpoint-sha",
        "quantized_sha256": "synthetic-quantized-sha",
        "source_training_receipt": {"path": "nowhere.json", "sha256": "0" * 64},
        "split_sha256": "synthetic-split-abc",
        "n_items": 20,
    }


def test_quant_geometry_metrics_skips_cleanly_below_nn_k(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A held-out set no bigger than DEFAULT_NN_K (fixture-scale, not production) must
    not crash the whole receipt -- `({}, None)`, printed rather than silent."""
    fp32_latents = torch.randn(5, 4)
    quantized_latents = torch.randn(5, 4)

    metrics, reference = bench._quant_geometry_metrics(
        fp32_latents=fp32_latents,
        quantized_latents=quantized_latents,
        split_sha256="synthetic-split-tiny",
        checkpoint_sha256="synthetic-checkpoint-sha",
        quantized_sha256="synthetic-quantized-sha",
        source_training_receipt={"path": "nowhere.json", "sha256": "0" * 64},
    )

    assert metrics == {}
    assert reference is None
    assert "skipped" in capsys.readouterr().out


# =====================================================================================
# Layer 2: TEXT branch, real tiny pipeline.
# =====================================================================================


def _build_tokenizer(path: Path, n_pairs: int) -> None:
    texts = [f"anchor number {i} word" for i in range(n_pairs)] + [
        f"anchor number {i} match" for i in range(n_pairs)
    ]
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106 -- a token id name, not a secret
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(path))


def _build_pairs_parquet(path: Path, n_pairs: int) -> None:
    anchors = [f"anchor number {i} word" for i in range(n_pairs)]
    positives = [f"anchor number {i} match" for i in range(n_pairs)]
    pq.write_table(pa.table({"anchor": anchors, "positive": positives}), path)


@pytest.fixture
def text_trained_receipt(tmp_path: Path) -> dict:
    """A real, tiny (dim=8, depth=1, 2 steps) `pretrain_region` receipt with
    `holdout_pairs=6` -- `_embed_holdout_pooled_both` concatenates anchor+positive, so
    this gives 12 held-out items, one more than `DEFAULT_NN_K` needs to populate
    rather than skip the geometry block (same fixture shape as
    `tests/test_benchmark_metrics_v2_receipt.py`'s `trained_receipt`, tuned for item
    count instead of raw speed)."""
    tok_path = tmp_path / "tokenizer.json"
    shard_path = tmp_path / "pairs.parquet"
    _build_tokenizer(tok_path, 40)
    _build_pairs_parquet(shard_path, 40)

    cfg = PretrainConfig(
        region="geo-wiring-text-test",
        pair_columns=("anchor", "positive"),
        shards=[str(shard_path)],
        steps=2,
        batch_size=4,
        holdout_pairs=6,
        eval_every=2,
        checkpoint_every=2,
        max_len=16,
        seed=3,
        device="cpu",
        encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=16),
        tokenizer_path=str(tok_path),
        out_dir=str(tmp_path / "receipts"),
    )
    receipt = pretrain_region(cfg)

    def fake_regions_spec() -> dict:
        def _shards(glob_pat: str, root: Path | None = None) -> list[str]:
            return [str(shard_path)]

        class _Entry:
            sources: ClassVar = [("pairs.parquet", ("anchor", "positive"), 0)]
            root = tmp_path

        return {"REGIONS": {}, "_shards": _shards, "region_spec": lambda name: _Entry()}

    bench._regions_spec = fake_regions_spec
    quant._load_regions_spec = fake_regions_spec
    return receipt


@pytest.fixture(autouse=True)
def _restore_regions_spec():
    orig_bench, orig_quant = bench._regions_spec, quant._load_regions_spec
    yield
    bench._regions_spec = orig_bench
    quant._load_regions_spec = orig_quant


def test_text_eval_quantized_receipt_carries_quant_geometry_fields(
    tmp_path: Path, text_trained_receipt: dict
) -> None:
    quant_rec = quant.quantize_text_region(
        "geo-wiring-text-test", tmp_path, tolerance=1.0, aggressive=3, max_bits=8
    )
    rec = bench.benchmark_region_quantized(
        "geo-wiring-text-test", tmp_path, Path(quant_rec["artifacts"]["quantized_path"])
    )

    geometry_keys = {k for k in rec.metrics if k.startswith("quant.geometry.")}
    assert geometry_keys == _GEOMETRY_KEYS
    reference = rec.provenance["quant.geometry.reference"]
    assert reference["n_items"] == 12
    assert reference["checkpoint_sha256"] == rec.artifacts["checkpoint_sha256"]
    assert reference["quantized_sha256"] == rec.artifacts["quantized_sha256"]


def test_text_unbound_train_receipt_skips_geometry_cleanly(
    tmp_path: Path, text_trained_receipt: dict
) -> None:
    """The wiring's own guard
    (`if checkpoint_sha256 and train_receipt.get("checkpoint"):`): a training receipt
    with no `checkpoint_sha256` anywhere must still produce a full receipt -- no
    crash, no unverified checkpoint load -- just no `quant.geometry.*`. The quantized
    artifact is built FIRST, from the intact receipt (quantizing does not go through
    this guard); only the on-disk TRAINING receipt is stripped afterward, matching
    `--allow-unbound-train-receipt`'s real trigger (a pre-R9 receipt, not a corrupted
    quantize step)."""
    quant_rec = quant.quantize_text_region(
        "geo-wiring-text-test", tmp_path, tolerance=1.0, aggressive=3, max_bits=8
    )

    receipt_path = Path(text_trained_receipt["receipt_path"])
    on_disk = json.loads(receipt_path.read_text())
    on_disk.pop("checkpoint_sha256", None)
    on_disk.get("artifacts", {}).pop("checkpoint_sha256", None)
    receipt_path.write_text(json.dumps(on_disk))

    rec = bench.benchmark_region_quantized(
        "geo-wiring-text-test",
        tmp_path,
        Path(quant_rec["artifacts"]["quantized_path"]),
        allow_unbound_train_receipt=True,
    )

    assert not any(k.startswith("quant.geometry.") for k in rec.metrics)
    assert "quant.geometry.reference" not in rec.provenance


# =====================================================================================
# Layer 3: VISUAL branch, real tiny pipeline. No "unbound" case -- see module docstring.
# =====================================================================================

_GEOMETRY_TINY_JEPA = JEPAConfig(
    image_size=32,
    patch_size=8,
    dim=32,
    depth=1,
    n_heads=4,
    predictor_dim=16,
    predictor_depth=1,
    n_target_blocks=1,
)


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _zip_pngs(path: Path, n: int, folder: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _png_bytes()
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(n):
            zf.writestr(f"{folder}/{i:02d}.png", payload)
    return path


@pytest.fixture
def visual_trained_receipt(tmp_path: Path) -> tuple[Path, Path]:
    """A real, tiny `pretrain_vl_region` receipt with 12 eurosat probe-eval images --
    one more than `DEFAULT_NN_K` needs to populate rather than skip the geometry
    block. Visual's `probe_eval_shards` decode is UNCAPPED regardless of
    `probe_limit` (`_decode_png_stores(..., limit=0, ...)`), so the item count is
    exactly the zip's own image count."""
    state = tmp_path / "state"
    receipts = state / "receipts"
    train = _zip_pngs(tmp_path / "train.zip", n=4, folder="cls")
    eurosat = _zip_pngs(tmp_path / "eurosat.zip", n=12, folder="forest")
    cfg = VLPretrainConfig(
        region="visual",
        train_shards=[str(train)],
        probe_train_shards=[str(eurosat)],
        probe_eval_shards=[str(eurosat)],
        transfer_shards=[],
        steps=2,
        batch_size=2,
        seed=0,
        device="cpu",
        warmup_steps=1,
        eval_every=1,
        checkpoint_every=1,
        probe_steps=2,
        probe_limit=12,
        jepa=_GEOMETRY_TINY_JEPA,
        out_dir=str(receipts),
        cache_dir=str(state / "vl-cache"),
        image_backend="png_zip",
        corpus_source="visual-clean-v1",
        probe_sets=[
            {"name": "eurosat-test", "source": "phelber/eurosat-rgb-128", "role": "primary"},
        ],
    )
    receipt = pretrain_vl_region(cfg)
    train_path = Path(receipt["receipt_path"])
    return state, train_path


def test_visual_eval_quantized_receipt_carries_quant_geometry_fields(
    visual_trained_receipt: tuple[Path, Path],
) -> None:
    state, train_path = visual_trained_receipt
    qrec = quant.quantize_visual_region(
        "visual", state, tolerance=1.0, aggressive=8, max_bits=8, train_receipt_path=train_path
    )
    rec = bench.benchmark_visual_region_quantized(
        "visual",
        state,
        Path(qrec["artifacts"]["quantized_path"]),
        train_receipt_path=train_path,
    )

    geometry_keys = {k for k in rec.metrics if k.startswith("quant.geometry.")}
    assert geometry_keys == _GEOMETRY_KEYS
    reference = rec.provenance["quant.geometry.reference"]
    assert reference["n_items"] == 12
    assert reference["checkpoint_sha256"] == rec.artifacts["checkpoint_sha256"]
    assert reference["quantized_sha256"] == rec.artifacts["quantized_sha256"]
