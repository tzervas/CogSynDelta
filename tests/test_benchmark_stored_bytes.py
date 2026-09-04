"""N5: the fp32 eval receipt and the eval-quantized receipt must report WEIGHTS-ONLY bytes,
using the SAME definition the quantizer itself uses for `compression_ratio`'s denominator
(`cogsyndelta.quant.ptq.fp32_reference_bytes`) -- never a checkpoint FILE's raw
`stat().st_size`. A resumable training checkpoint also carries the Adam optimizer's
momentum and variance buffers (`opt.state_dict()`, see `regions/pretrain.py`'s checkpoint
dict), routinely ~2x the weights themselves, so the file's size and the model's fp32 weight
bytes are two different numbers -- an earlier fix (N5 round 1) reported the FILE's size,
which fixed the "picked up a smaller quantized receipt" bug but introduced a new mismatch:
the file-size number does not divide into `benchmark_region_quantized`'s
`packed_stored_bytes` at the ratio `csd-quantize.py`'s own `compression_ratio` measured.

Three tests:

1. A real tiny CPU pretrain -> quantize pass (same fixture shape as
   `test_benchmark_quantized_artifact.py`), with the fp32 pass run AFTER the quantized
   receipt already exists on disk under the same state root -- exactly the condition the
   original N5 bug mishandled (a `{region}-quant-*.json` receipt's `stored_bytes`
   contaminating the fp32 pass). Asserts the fp32 receipt's `eff.stored_mb` equals
   `fp32_reference_bytes(model)` (weights only), is strictly less than the checkpoint
   FILE's `stat().st_size` (proof the optimizer-state overhead is excluded), and that both
   receipts carry `provenance.stored_bytes_definition == "weights-only"`.

2. The ratio between the fp32 receipt's `eff.stored_mb` and the eval-quantized receipt's
   `eff.stored_mb` equals the quant receipt's own `compression_ratio` -- the acceptance
   property the round-2 review named: "the ratio in the matrix equals the quant receipt's
   compression_ratio".

3. The exact acceptance case named in the matrix-harness operator brief: today's memory V2
   pilot checkpoint + packed artifact under
   `/akula-data/csd/receipts/memory-checkpoints/369351bf-b1280/`. Skipped when those
   host-local files are absent (a fresh clone, CI, another host). Asserts the fp32 receipt's
   stored bytes equal the quantizer's own `fp32_reference_bytes` for that checkpoint, the
   eval-quantized receipt's stored bytes equal the packed artifact's own
   `packed_stored_bytes`, and their ratio equals the quant receipt's `compression_ratio`
   within 1e-6.
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


def _train_and_quantize_fixture(tmp_path: Path) -> tuple[dict, dict, Path]:
    """A real tiny CPU pretrain -> quantize pass. Returns (train_receipt, quant_receipt,
    quantized_path)."""
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

    # Land it exactly where the original N5 bug's glob (`{region}-quant-*.json` under
    # `state/receipts`) would have matched it -- the fp32 pass runs against the SAME
    # `tmp_path` state root.
    write_receipt(quant_receipt, tmp_path / "receipts", "benchq-n5-quant-fixture.json")
    return train_receipt, quant_receipt, quantized_path


def test_fp32_receipt_reports_weights_only_bytes_even_with_a_quant_receipt_present(
    tmp_path: Path,
) -> None:
    """Reproduces the exact condition the original N5 bug mishandled, and asserts the fp32
    receipt reports the quantizer's own `fp32_reference_bytes` -- weights only, never the
    checkpoint FILE's `stat().st_size` (which also carries Adam optimizer state)."""
    from cogsyndelta.quant.ptq import fp32_reference_bytes

    train_receipt, quant_receipt, quantized_path = _train_and_quantize_fixture(tmp_path)
    train_receipt_path = Path(train_receipt["receipt_path"])

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

    # Rebuild the model exactly as benchmark_region does, to get the same weights-only
    # reference figure independently of the production code path under test.
    from cogsyndelta.regions._checkpoint import load_checkpoint
    from cogsyndelta.regions.text_encoder import TextEncoder

    enc_cfg = TextEncoderConfig(**train_receipt["config"]["encoder"])
    model = TextEncoder(enc_cfg, name="benchq-n5")
    ck = load_checkpoint(train_receipt["checkpoint"], map_location="cpu")
    model.load_state_dict(ck["model"])
    expected_fp32_bytes = fp32_reference_bytes(model)

    stored_mb = rec_fp32.metrics["eff.stored_mb"]
    assert stored_mb == pytest.approx(expected_fp32_bytes / 1e6), (
        f"fp32 receipt reported stored_mb={stored_mb!r} but the quantizer's own "
        f"fp32_reference_bytes for this checkpoint is {expected_fp32_bytes} bytes -- the "
        "fp32 eval pass must use the same weights-only definition compression_ratio does"
    )
    assert stored_mb != pytest.approx(quantized_file_size / 1e6), (
        "fp32 receipt's stored_mb must not equal the quantized artifact's size"
    )
    assert stored_mb < fp32_file_size / 1e6, (
        "fp32 receipt's stored_mb must be strictly smaller than the checkpoint FILE's own "
        "size -- the file also carries Adam optimizer state the weights-only figure excludes"
    )
    assert "quantized_size" not in rec_fp32.provenance
    assert rec_fp32.provenance["stored_bytes_definition"] == "weights-only"

    rec_q = benchmark_mod.benchmark_region_quantized(
        "benchq-n5",
        tmp_path,
        quantized_path,
        quant_receipt_path=tmp_path / "receipts" / "benchq-n5-quant-fixture.json",
        train_receipt_path=train_receipt_path,
    )
    assert rec_q.provenance["stored_bytes_definition"] == "weights-only"

    matrix_ratio = rec_fp32.metrics["eff.stored_mb"] / rec_q.metrics["eff.stored_mb"]
    assert matrix_ratio == pytest.approx(quant_receipt["quant.compression_ratio"], rel=1e-6), (
        f"fp32/eval-quantized stored_mb ratio ({matrix_ratio!r}) must equal the quant "
        f"receipt's own quant.compression_ratio ({quant_receipt['quant.compression_ratio']!r}) -- "
        "both receipts must count bytes the same way the quantizer does"
    )


@pytest.mark.skipif(
    not (
        MEMORY_CHECKPOINT.is_file()
        and MEMORY_QUANTIZED.is_file()
        and MEMORY_QUANT_RECEIPT.is_file()
        and MEMORY_TRAIN_RECEIPT.is_file()
    ),
    reason="today's memory V2 pilot artifacts are host-local, not part of the repo fixture set",
)
def test_memory_v2_stored_bytes_match_quantizers_own_definition_and_ratio() -> None:
    from cogsyndelta.quant.ptq import (
        fp32_reference_bytes,
        load_packed_artifact,
        packed_stored_bytes,
    )
    from cogsyndelta.regions._checkpoint import load_checkpoint
    from cogsyndelta.regions.text_encoder import TextEncoder

    quant_receipt = json.loads(MEMORY_QUANT_RECEIPT.read_text())
    train_receipt = json.loads(MEMORY_TRAIN_RECEIPT.read_text())

    enc_cfg = TextEncoderConfig(**train_receipt["config"]["encoder"])
    model = TextEncoder(enc_cfg, name="memory")
    ck = load_checkpoint(train_receipt["checkpoint"], map_location="cpu")
    model.load_state_dict(ck["model"])
    expected_fp32_bytes = fp32_reference_bytes(model)

    rec_fp32 = benchmark_mod.benchmark_region(
        "memory", Path("/akula-data/csd"), train_receipt_path=MEMORY_TRAIN_RECEIPT
    )
    assert rec_fp32 is not None
    fp32_stored_mb = rec_fp32.metrics["eff.stored_mb"]
    assert fp32_stored_mb == pytest.approx(expected_fp32_bytes / 1e6), (
        f"fp32 receipt stored_mb={fp32_stored_mb!r} must equal the quantizer's own "
        f"fp32_reference_bytes ({expected_fp32_bytes} bytes), not the checkpoint file's "
        f"raw size ({MEMORY_CHECKPOINT.stat().st_size} bytes) and not any quantized number"
    )
    assert rec_fp32.provenance["stored_bytes_definition"] == "weights-only"

    rec_q = benchmark_mod.benchmark_region_quantized(
        "memory",
        Path("/akula-data/csd"),
        MEMORY_QUANTIZED,
        quant_receipt_path=MEMORY_QUANT_RECEIPT,
        train_receipt_path=MEMORY_TRAIN_RECEIPT,
    )
    assert rec_q.artifacts["quantized_sha256"] == quant_receipt["artifacts"]["quantized_sha256"]
    assert rec_q.provenance["stored_bytes_definition"] == "weights-only"

    packed = load_packed_artifact(MEMORY_QUANTIZED)
    expected_quantized_mb = packed_stored_bytes(packed) / 1e6
    q_stored_mb = rec_q.metrics["eff.stored_mb"]
    assert q_stored_mb == pytest.approx(expected_quantized_mb), (
        f"eval-quantized receipt stored_mb={q_stored_mb!r} must equal the artifact's own "
        f"packed_stored_bytes ({expected_quantized_mb!r} MB), not the fp32 checkpoint's size"
    )
    assert q_stored_mb != pytest.approx(fp32_stored_mb), (
        "the two receipts must report DIFFERENT stored sizes -- the packed artifact is "
        "meant to be smaller than the fp32 model"
    )

    # This fixture is the frozen real 2026-09-03 receipt (v1 field names, no `quant.`
    # prefix) -- unlike the live-generated receipt above, it predates the metrics-v2
    # rename and is never regenerated, so it stays read as `compression_ratio`.
    matrix_ratio = fp32_stored_mb / q_stored_mb
    assert matrix_ratio == pytest.approx(quant_receipt["compression_ratio"], rel=1e-6), (
        f"fp32/eval-quantized stored_mb ratio ({matrix_ratio!r}) must equal the quant "
        f"receipt's own compression_ratio ({quant_receipt['compression_ratio']!r})"
    )
