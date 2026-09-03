"""N5: the fp32 eval receipt must report the fp32 CHECKPOINT's own bytes on disk, never
a quantized artifact's -- `benchmark_region` used to prefer a same-region
``{region}-quant-*.json`` receipt's ``stored_bytes`` when one existed, so an fp32 receipt
scored right next to a quantized one silently reported the SMALLER, quantized size. Only
`benchmark_region_quantized` (the ``--quantized`` / ``kind="eval-quantized"`` path) may
report the packed artifact's size.

Two tests:

1. A real tiny CPU pretrain -> quantize pass (same fixture shape as
   `test_benchmark_quantized_artifact.py`), with the fp32 pass run AFTER the quantized
   receipt already exists on disk under the same state root -- exactly the condition
   that used to trigger the bug (`benchmark_region`'s old `{region}-quant-*.json` glob
   would have matched). Asserts the fp32 receipt's `eff.stored_mb` is derived from
   the checkpoint FILE's own `stat().st_size`, not the quantized artifact's smaller
   `stored_bytes`.

2. The exact acceptance case named in the matrix-harness operator brief: today's memory
   V2 pilot checkpoint + packed artifact under
   `/akula-data/csd/receipts/memory-checkpoints/369351bf-b1280/`. Skipped when those
   host-local files are absent (a fresh clone, CI, another host). Asserts the fp32
   receipt's stored bytes equal `final.pt`'s file size and the eval-quantized receipt's
   stored bytes equal the packed artifact's own `packed_stored_bytes` (what
   `save_packed_artifact` counted, not raw file bytes -- see `ptq.packed_stored_bytes`'s
   docstring for why those differ).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from cogsyndelta.regions._receipt import write_receipt
from cogsyndelta.regions.pretrain import PretrainConfig, pretrain_region
from cogsyndelta.regions.text_encoder import TextEncoderConfig

pytestmark = pytest.mark.cpu

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# Today's memory V2 pilot artifacts (2026-09-03), host-local -- see the matrix-harness
# operator brief and test_benchmark_quantized_artifact.py, which uses the same files.
MEMORY_CHECKPOINT_DIR = Path("/akula-data/csd/receipts/memory-checkpoints/369351bf-b1280")
MEMORY_CHECKPOINT = MEMORY_CHECKPOINT_DIR / "final.pt"
MEMORY_QUANTIZED = MEMORY_CHECKPOINT_DIR / "final.ptq.pt"
MEMORY_QUANT_RECEIPT = Path(
    "/akula-data/session-backup-staging/memory-variants/state-b1280/receipts/"
    "memory-quant-20260903T203342Z.json"
)
MEMORY_TRAIN_RECEIPT = Path(
    "/akula-data/session-backup-staging/memory-variants/state-b1280/receipts/"
    "memory-20260903T184441Z.json"
)


def _load_module(name: str, filename: str) -> Any:
    """Import a hyphenated `scripts/*.py` file -- see tests/test_region_spec_consumers.py
    for why every consumer in this repo loads it this way."""
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


benchmark_mod = _load_module("csd_benchmark_stored_bytes_test", "csd-benchmark.py")
quantize_mod = _load_module("csd_quantize_stored_bytes_test", "csd-quantize.py")


def _build_tokenizer(path: Path, n_pairs: int) -> None:
    texts = [f"anchor number {i} word extra padding token" for i in range(n_pairs)] + [
        f"anchor number {i} match extra padding token" for i in range(n_pairs)
    ]
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106 -- a token id name, not a secret
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(path))


def _build_pairs_parquet(path: Path, n_pairs: int) -> None:
    anchors = [f"anchor number {i} word extra padding token" for i in range(n_pairs)]
    positives = [f"anchor number {i} match extra padding token" for i in range(n_pairs)]
    pq.write_table(pa.table({"anchor": anchors, "positive": positives}), path)


@pytest.fixture(autouse=True)
def _restore_spec_monkeypatches() -> Any:
    orig_bench = benchmark_mod._regions_spec
    orig_quant = quantize_mod._load_regions_spec
    yield
    benchmark_mod._regions_spec = orig_bench
    quantize_mod._load_regions_spec = orig_quant


def test_fp32_receipt_reports_checkpoint_file_size_even_with_a_quant_receipt_present(
    tmp_path: Path,
) -> None:
    """Reproduces the exact condition the old code mishandled: a `{region}-quant-*.json`
    receipt already sitting under the same state root when the fp32 pass runs."""
    tok_path = tmp_path / "tokenizer.json"
    shard_path = tmp_path / "pairs.parquet"
    _build_tokenizer(tok_path, 48)
    _build_pairs_parquet(shard_path, 48)

    cfg = PretrainConfig(
        region="benchq-n5",
        pair_columns=("anchor", "positive"),
        shards=[str(shard_path)],
        steps=2,
        batch_size=8,
        holdout_pairs=8,
        eval_every=2,
        checkpoint_every=2,
        max_len=16,
        seed=0,
        device="cpu",
        encoder=TextEncoderConfig(dim=64, depth=1, n_heads=2, max_len=16),
        tokenizer_path=str(tok_path),
        out_dir=str(tmp_path / "receipts"),
    )
    train_receipt = pretrain_region(cfg)
    train_receipt_path = Path(train_receipt["receipt_path"])

    def _shards(glob_pat: str, root: Path | None = None) -> list[str]:
        return [str(shard_path)]

    class _Entry:
        sources: ClassVar = [("pairs.parquet", ("anchor", "positive"), 0)]
        root = tmp_path

    fake_spec = {"REGIONS": {}, "_shards": _shards, "region_spec": lambda name: _Entry()}
    benchmark_mod._regions_spec = lambda: fake_spec
    quantize_mod._load_regions_spec = lambda: fake_spec

    quant_receipt = quantize_mod.quantize_text_region(
        "benchq-n5",
        tmp_path,
        tolerance=1.0,
        aggressive=3,
        max_bits=8,
        train_receipt_path=train_receipt_path,
    )
    assert quant_receipt["bits"], "fixture produced no quantized tensors -- widen the encoder"
    quantized_path = Path(quant_receipt["artifacts"]["quantized_path"])

    # Land it exactly where the old buggy glob (`{region}-quant-*.json` under
    # `state/receipts`) would have matched it -- the fp32 pass below runs against the
    # SAME `tmp_path` state root.
    write_receipt(quant_receipt, tmp_path / "receipts", "benchq-n5-quant-fixture.json")

    checkpoint_path = Path(train_receipt["checkpoint"])
    fp32_file_size = checkpoint_path.stat().st_size
    quantized_file_size = quantized_path.stat().st_size
    assert quantized_file_size < fp32_file_size, (
        "fixture must actually shrink on disk or this test cannot distinguish the two sizes"
    )

    rec_fp32 = benchmark_mod.benchmark_region(
        "benchq-n5", tmp_path, train_receipt_path=train_receipt_path
    )
    assert rec_fp32 is not None
    assert rec_fp32.kind == "eval"

    stored_mb = rec_fp32.metrics["eff.stored_mb"]
    assert stored_mb == pytest.approx(fp32_file_size / 1e6), (
        f"fp32 receipt reported stored_mb={stored_mb!r} but the checkpoint file "
        f"{checkpoint_path} is {fp32_file_size} bytes -- a quant receipt sitting in the "
        "same state root must not change what the fp32 pass reports"
    )
    assert stored_mb != pytest.approx(quantized_file_size / 1e6), (
        "fp32 receipt's stored_mb must not equal the quantized artifact's size"
    )
    assert "quantized_size" not in rec_fp32.provenance


@pytest.mark.skipif(
    not (
        MEMORY_CHECKPOINT.is_file()
        and MEMORY_QUANTIZED.is_file()
        and MEMORY_QUANT_RECEIPT.is_file()
        and MEMORY_TRAIN_RECEIPT.is_file()
    ),
    reason="today's memory V2 pilot artifacts are host-local, not part of the repo fixture set",
)
def test_memory_v2_stored_bytes_match_each_artifacts_own_size() -> None:
    from cogsyndelta.quant.ptq import load_packed_artifact, packed_stored_bytes

    quant_receipt = json.loads(MEMORY_QUANT_RECEIPT.read_text())

    rec_fp32 = benchmark_mod.benchmark_region(
        "memory", Path("/akula-data/csd"), train_receipt_path=MEMORY_TRAIN_RECEIPT
    )
    assert rec_fp32 is not None
    fp32_stored_mb = rec_fp32.metrics["eff.stored_mb"]
    assert fp32_stored_mb == pytest.approx(MEMORY_CHECKPOINT.stat().st_size / 1e6), (
        f"fp32 receipt stored_mb={fp32_stored_mb!r} must equal {MEMORY_CHECKPOINT}'s own "
        f"file size ({MEMORY_CHECKPOINT.stat().st_size} bytes), not any quantized number"
    )

    rec_q = benchmark_mod.benchmark_region_quantized(
        "memory",
        Path("/akula-data/csd"),
        MEMORY_QUANTIZED,
        quant_receipt_path=MEMORY_QUANT_RECEIPT,
        train_receipt_path=MEMORY_TRAIN_RECEIPT,
    )
    assert rec_q.artifacts["quantized_sha256"] == quant_receipt["artifacts"]["quantized_sha256"]

    packed = load_packed_artifact(MEMORY_QUANTIZED)
    expected_quantized_mb = packed_stored_bytes(packed) / 1e6
    q_stored_mb = rec_q.metrics["eff.stored_mb"]
    assert q_stored_mb == pytest.approx(expected_quantized_mb), (
        f"eval-quantized receipt stored_mb={q_stored_mb!r} must equal the artifact's own "
        f"packed_stored_bytes ({expected_quantized_mb!r} MB), not the fp32 checkpoint's size"
    )
    assert q_stored_mb != pytest.approx(fp32_stored_mb), (
        "the two receipts must report DIFFERENT stored sizes -- the packed artifact is "
        "meant to be smaller than the fp32 checkpoint"
    )
