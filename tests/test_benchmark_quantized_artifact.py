"""A6 (DESIGN.v2 §6.2, closing C1 structurally): `scripts/csd-benchmark.py --quantized`
must score the PACKED ARTIFACT `csd-quantize.py` actually wrote, not re-report the
in-memory plan metric under a quantized label.

Three properties, each with its own constructed failing case:

1. The fp32 pass (`kind="eval"`) and the quantized-artifact pass (`kind="eval-quantized"`)
   are DIFFERENT receipt kinds with DIFFERENT bindings -- the quantized receipt carries
   both `artifacts.quantized_sha256` (the file this process actually opened) and
   `artifacts.checkpoint_sha256` (the fp32 parent), while the fp32 receipt carries only
   the latter. Two receipts sharing a region and a `stage` must still be distinguishable
   without inspecting `provenance`.
2. A `--quant-receipt` naming a DIFFERENT packed file (mismatched `quantized_sha256`) is
   refused outright -- the binding is a fact checked against the bytes on disk, not an
   unread claim carried across from the quant receipt.
3. Corrupting one packed tensor's stored bytes changes the reported metric -- proof this
   function actually forwarded the artifact's own tensors through the model, rather than
   e.g. silently falling back to an untouched fp32 checkpoint.

A fourth test drives the exact acceptance case named in the matrix-harness operator brief:
that this function's `rank.recall@1` on TODAY's memory V2 production artifact reproduces
`csd-quantize.py`'s own `quantized_metric` for that same artifact within 1e-4. It is guarded
by the presence of the specific host-local pilot files it names (see MEMORY_* below) and
skips cleanly everywhere else, including CI -- those files are today's measured pilot
output, not a repo fixture.
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
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from cogsyndelta.regions._checkpoint import sha256_file
from cogsyndelta.regions.pretrain import PretrainConfig, pretrain_region
from cogsyndelta.regions.text_encoder import TextEncoderConfig

pytestmark = pytest.mark.cpu

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# Today's memory V2 pilot artifacts (2026-09-03), host-local -- see the matrix-harness
# operator brief. Not part of this repo's fixture set; the live-reproduction test below
# skips when they are absent (a fresh clone, CI, another host).
MEMORY_CHECKPOINT_DIR = Path("/akula-data/csd/receipts/memory-checkpoints/369351bf-b1280")
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


benchmark_mod = _load_module("csd_benchmark_quantized_artifact_test", "csd-benchmark.py")
quantize_mod = _load_module("csd_quantize_quantized_artifact_test", "csd-quantize.py")


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


@pytest.fixture
def quantized_fixture(tmp_path: Path) -> dict[str, Any]:
    """A REAL tiny CPU pretrain -> quantize pass, exactly the shape production runs:
    `pretrain_region` writes a training receipt and checkpoint; `quantize_text_region`
    (the real entry point `csd-quantize.py` uses) reads it, builds a quantization plan
    and writes a packed `final.ptq.pt` plus its own quant receipt.

    `dim=64` (not the `dim=8` other fixtures in this repo use) so at least one tensor
    clears `quantizable()`'s 4096-element floor and is genuinely packed at sub-byte
    width -- a plan that quantizes nothing would make test 3 (corrupt a packed tensor)
    vacuous.
    """
    tok_path = tmp_path / "tokenizer.json"
    shard_path = tmp_path / "pairs.parquet"
    _build_tokenizer(tok_path, 48)
    _build_pairs_parquet(shard_path, 48)

    cfg = PretrainConfig(
        region="benchq-test",
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

    def _shards(glob_pat: str, root: Path | None = None) -> list[str]:
        return [str(shard_path)]

    class _Entry:
        sources: ClassVar = [("pairs.parquet", ("anchor", "positive"), 0)]
        root = tmp_path

    fake_spec = {"REGIONS": {}, "_shards": _shards, "region_spec": lambda name: _Entry()}
    benchmark_mod._regions_spec = lambda: fake_spec
    quantize_mod._load_regions_spec = lambda: fake_spec

    quant_receipt = quantize_mod.quantize_text_region(
        "benchq-test",
        tmp_path,
        tolerance=1.0,
        aggressive=3,
        max_bits=8,
        train_receipt_path=Path(train_receipt["receipt_path"]),
    )
    assert quant_receipt["bits"], "fixture produced no quantized tensors -- widen the encoder"

    from cogsyndelta.regions._receipt import write_receipt

    quant_receipt_path = write_receipt(
        quant_receipt, tmp_path / "receipts", "benchq-test-quant-fixture.json"
    )

    return {
        "tmp_path": tmp_path,
        "train_receipt": train_receipt,
        "train_receipt_path": Path(train_receipt["receipt_path"]),
        "quant_receipt": quant_receipt,
        "quant_receipt_path": quant_receipt_path,
        "quantized_path": Path(quant_receipt["artifacts"]["quantized_path"]),
    }


@pytest.fixture(autouse=True)
def _restore_spec_monkeypatches() -> Any:
    orig_bench = benchmark_mod._regions_spec
    orig_quant = quantize_mod._load_regions_spec
    yield
    benchmark_mod._regions_spec = orig_bench
    quantize_mod._load_regions_spec = orig_quant


def test_fp32_and_quantized_receipts_differ_in_kind_and_binding(
    quantized_fixture: dict[str, Any],
) -> None:
    tmp_path = quantized_fixture["tmp_path"]
    train_receipt_path = quantized_fixture["train_receipt_path"]

    rec_fp32 = benchmark_mod.benchmark_region(
        "benchq-test", tmp_path, train_receipt_path=train_receipt_path
    )
    assert rec_fp32 is not None

    rec_q = benchmark_mod.benchmark_region_quantized(
        "benchq-test",
        tmp_path,
        quantized_fixture["quantized_path"],
        quant_receipt_path=quantized_fixture["quant_receipt_path"],
        train_receipt_path=train_receipt_path,
    )

    # kind differs -- the whole point: `stage` is "eval" for both.
    assert rec_fp32.stage == rec_q.stage == "eval"
    assert rec_fp32.kind == "eval"
    assert rec_q.kind == "eval-quantized"
    assert rec_fp32.kind != rec_q.kind

    # binding differs -- the quantized receipt names an artifact the fp32 one does not,
    # and the two AGREE on which fp32 checkpoint sits behind both of them.
    assert "quantized_sha256" not in rec_fp32.artifacts
    assert "quantized_sha256" in rec_q.artifacts
    assert rec_q.artifacts["quantized_sha256"] == sha256_file(quantized_fixture["quantized_path"])
    assert rec_q.artifacts["checkpoint_sha256"] == rec_fp32.artifacts["checkpoint_sha256"]

    assert rec_fp32.provenance["eval_target"] == "fp32"
    assert rec_q.provenance["eval_target"] == "quantized"

    # Filenames must not collide either (C2 defence in depth): `Receipt.write()` now
    # names the file from `kind`, not bare `stage`.
    path_fp32 = rec_fp32.write(tmp_path / "receipts")
    path_q = rec_q.write(tmp_path / "receipts")
    assert path_fp32 != path_q
    assert "eval-quantized" in path_q.name
    assert "eval-quantized" not in path_fp32.name


def test_mismatched_quant_receipt_sha_is_refused(quantized_fixture: dict[str, Any]) -> None:
    """A `--quant-receipt` naming a `quantized_sha256` that does not match the actual
    file on disk must be refused, not silently trusted -- the file is not the one that
    receipt describes."""
    tmp_path = quantized_fixture["tmp_path"]
    bad_receipt = dict(quantized_fixture["quant_receipt"])
    bad_receipt["artifacts"] = dict(bad_receipt["artifacts"])
    bad_receipt["artifacts"]["quantized_sha256"] = "0" * 64
    bad_path = tmp_path / "bad-quant-receipt.json"
    bad_path.write_text(json.dumps(bad_receipt))

    with pytest.raises(ValueError, match="does not match"):
        benchmark_mod.benchmark_region_quantized(
            "benchq-test",
            tmp_path,
            quantized_fixture["quantized_path"],
            quant_receipt_path=bad_path,
            train_receipt_path=quantized_fixture["train_receipt_path"],
        )


def test_corrupting_the_packed_artifact_changes_the_reported_metric(
    quantized_fixture: dict[str, Any],
) -> None:
    """Mutation test (verify-guards-by-making-them-fail): if `benchmark_region_quantized`
    silently ignored the artifact and re-scored the fp32 checkpoint instead, corrupting
    the artifact's bytes would have no effect on the reported metric. It must."""
    tmp_path = quantized_fixture["tmp_path"]
    quantized_path = quantized_fixture["quantized_path"]
    train_receipt_path = quantized_fixture["train_receipt_path"]

    baseline = benchmark_mod.benchmark_region_quantized(
        "benchq-test", tmp_path, quantized_path, train_receipt_path=train_receipt_path
    )

    packed = torch.load(quantized_path, weights_only=True, map_location="cpu")
    name = next(iter(packed["codes"]))
    packed["codes"][name] = packed["codes"][name] ^ 0xFF  # flip every bit of every code byte

    corrupted_path = tmp_path / "corrupted.ptq.pt"
    torch.save(packed, corrupted_path)

    corrupted = benchmark_mod.benchmark_region_quantized(
        "benchq-test", tmp_path, corrupted_path, train_receipt_path=train_receipt_path
    )

    assert corrupted.artifacts["quantized_sha256"] != baseline.artifacts["quantized_sha256"]
    assert corrupted.metrics["rank.recall@1"] != baseline.metrics["rank.recall@1"] or (
        corrupted.metrics["repr.anisotropy"] != baseline.metrics["repr.anisotropy"]
    ), "corrupting a packed tensor's codes must move at least one reported metric"


@pytest.mark.skipif(
    not (
        MEMORY_QUANTIZED.is_file()
        and MEMORY_QUANT_RECEIPT.is_file()
        and MEMORY_TRAIN_RECEIPT.is_file()
    ),
    reason="today's memory V2 pilot artifacts are host-local, not part of the repo fixture set",
)
def test_quantized_recall_reproduces_todays_memory_v2_quant_receipt() -> None:
    """The exact acceptance case the matrix-harness operator brief names: scoring the
    real artifact at MEMORY_QUANTIZED must reproduce `csd-quantize.py`'s own
    `quantized_metric` for that artifact within 1e-4 -- proof that A6's "score the file"
    and the plan's own "score the in-memory model" measure the same quantized weights.
    """
    quant_receipt = json.loads(MEMORY_QUANT_RECEIPT.read_text())

    rec = benchmark_mod.benchmark_region_quantized(
        "memory",
        Path("/akula-data/csd"),
        MEMORY_QUANTIZED,
        quant_receipt_path=MEMORY_QUANT_RECEIPT,
        train_receipt_path=MEMORY_TRAIN_RECEIPT,
    )

    assert rec.kind == "eval-quantized"
    assert rec.artifacts["quantized_sha256"] == quant_receipt["artifacts"]["quantized_sha256"]
    assert rec.artifacts["checkpoint_sha256"] == quant_receipt["artifacts"]["checkpoint_sha256"]

    reproduced = rec.metrics["rank.recall@1"]
    expected = quant_receipt["quantized_metric"]
    assert abs(reproduced - expected) < 1e-4, (
        f"reproduced quantized recall@1={reproduced!r} vs quant receipt's "
        f"quantized_metric={expected!r}, diff={abs(reproduced - expected)!r} >= 1e-4"
    )
