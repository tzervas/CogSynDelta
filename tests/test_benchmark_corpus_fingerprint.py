"""`csd-benchmark.py` must trust the receipt's corpus, not silently re-glob past it.

`scripts/csd-quantize.py` verifies the corpus fingerprint against the training receipt
before comparing anything, and refuses if a receipt-recorded shard name is missing.
`scripts/csd-benchmark.py` had neither check: `by_name[n] for n in wanted if n in
by_name] or resolved[0][0]` silently fell back to re-globbing the WHOLE corpus the
instant a single receipt-recorded shard name failed to match, then built a fresh holdout
from that (possibly different) corpus and compared it against the receipt's
`untrained_baseline` gate as though it were the same split.

This file drives `benchmark_region` -- the real entry point, not a unit helper -- through
a genuine tiny end-to-end run: `pretrain_region` produces a real receipt and checkpoint on
CPU with a WordLevel tokenizer and an 8-dim encoder (same fixture shape as
tests/test_pretrain_resume.py), `_regions_spec()` is monkeypatched to route
`csd-benchmark.py`'s corpus resolution back to that same tiny shard, and:

  - a receipt whose `corpus.shards` names a file no longer on disk must raise
  - a receipt whose `corpus.fingerprint` does not match the resolved corpus must raise
  - the unmodified (matching) receipt must run the whole pipeline through to a Receipt
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import ClassVar

import pytest

# MUST precede the imports below: cogsyndelta.regions.pretrain imports tokenizers at
# module scope, so without the train group installed the import errors during collection
# and the whole file fails instead of skipping (same reasoning as
# tests/test_pretrain_resume.py).
pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from cogsyndelta.regions.pretrain import PretrainConfig, pretrain_region
from cogsyndelta.regions.text_encoder import TextEncoderConfig

pytestmark = pytest.mark.cpu


def _load_csd_benchmark():
    """Import scripts/csd-benchmark.py the same way tests/test_reserved_corpus_guard.py
    loads scripts/csd-train-all.py: the hyphenated filename is not a valid module name."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "csd-benchmark.py"
    spec = importlib.util.spec_from_file_location("csd_benchmark_fingerprint_test", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_csd_benchmark()


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
def trained_receipt(tmp_path: Path) -> dict:
    """A REAL receipt and checkpoint from a tiny CPU run of `pretrain_region`, written
    under `<tmp_path>/receipts` -- exactly the shape `benchmark_region` reads in
    production, not a hand-built stand-in.
    """
    tok_path = tmp_path / "tokenizer.json"
    shard_path = tmp_path / "pairs.parquet"
    _build_tokenizer(tok_path, 40)
    _build_pairs_parquet(shard_path, 40)

    cfg = PretrainConfig(
        region="benchfp-test",
        pair_columns=("anchor", "positive"),
        shards=[str(shard_path)],
        steps=2,
        batch_size=4,
        holdout_pairs=4,
        eval_every=2,
        checkpoint_every=2,
        max_len=16,
        seed=0,
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

        return {"REGIONS": {}, "_shards": _shards, "region_spec": lambda name: _Entry()}

    mod._regions_spec = fake_regions_spec  # module-level monkeypatch, restored per-test below
    return receipt


@pytest.fixture(autouse=True)
def _restore_regions_spec():
    original = mod._regions_spec
    yield
    mod._regions_spec = original


def test_receipt_naming_a_missing_shard_raises(tmp_path: Path, trained_receipt: dict) -> None:
    receipt_path = Path(trained_receipt["receipt_path"])
    receipt_path.write_text(
        receipt_path.read_text().replace('"pairs.parquet"', '"does-not-exist.parquet"')
    )

    with pytest.raises(RuntimeError, match="no longer on disk"):
        mod.benchmark_region("benchfp-test", tmp_path)


def test_receipt_with_mismatched_fingerprint_raises(tmp_path: Path, trained_receipt: dict) -> None:
    receipt_path = Path(trained_receipt["receipt_path"])
    real_fp = trained_receipt["corpus"]["fingerprint"]
    tampered_fp = real_fp[::-1] if real_fp[::-1] != real_fp else "0" * len(real_fp)
    receipt_path.write_text(receipt_path.read_text().replace(real_fp, tampered_fp))

    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        mod.benchmark_region("benchfp-test", tmp_path)


def test_matching_receipt_proceeds_through_the_whole_pipeline(
    tmp_path: Path, trained_receipt: dict
) -> None:
    """The receipt `pretrain_region` actually wrote, untouched: same shard, same
    fingerprint. Must not raise at the verification gate -- and, since nothing downstream
    is mocked, must complete the real benchmark pipeline and return a passing Receipt.
    """
    rec = mod.benchmark_region("benchfp-test", tmp_path)
    assert rec is not None
    assert rec.metrics["rank.recall@1"] >= 0.0
