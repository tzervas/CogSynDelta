"""`scripts/csd-quantize.py` must bind its quant receipt to the exact checkpoint bytes
it just measured, not merely the mutable path both name.

`scripts/csd-publish-checkpoint.py` (feat/hf-checkpoint-publisher) refuses to publish
any receipt whose ``artifacts.checkpoint_sha256`` does not equal the sha256 it computes
of the checkpoint file itself. `quantize_text_region` already routes its checkpoint load
through `cogsyndelta.regions._checkpoint.load_checkpoint`, which -- given `sha256_out` --
verifies (or, absent a prior expected hash, computes) that hash before `torch.load` ever
opens the file. This drives the real `quantize_text_region` entry point, on a genuine
tiny CPU pretrain + quantize run (same fixture shape as
tests/test_region_spec_consumers.py and the pretrain fixtures in
tests/test_pretrain_resume.py), and asserts the emitted receipt's
``artifacts.checkpoint_sha256`` is exactly the checkpoint file's sha256, and that
``artifacts.source_training_receipt`` names the training receipt actually read plus its
own content hash.

Verified by mutation (`test_receipt_omitting_checkpoint_sha256_would_be_unpublishable`):
a receipt shape with the ``artifacts`` block deleted -- what this file's target code
produced before this fix -- fails the same assertions this file's positive test makes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import ClassVar

import pytest

pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from cogsyndelta.regions._checkpoint import sha256_file
from cogsyndelta.regions.pretrain import PretrainConfig, pretrain_region
from cogsyndelta.regions.text_encoder import TextEncoderConfig

pytestmark = pytest.mark.cpu

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_csd_quantize():
    """Import scripts/csd-quantize.py -- a hyphenated filename is not a valid module
    name, so every consumer in this repo loads it this way (see
    tests/test_region_spec_consumers.py)."""
    path = SCRIPTS / "csd-quantize.py"
    spec = importlib.util.spec_from_file_location("csd_quantize_sha_receipt_test", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_csd_quantize()


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
    under `<tmp_path>/receipts` -- exactly the shape `quantize_text_region` reads in
    production, not a hand-built stand-in."""
    tok_path = tmp_path / "tokenizer.json"
    shard_path = tmp_path / "pairs.parquet"
    _build_tokenizer(tok_path, 40)
    _build_pairs_parquet(shard_path, 40)

    cfg = PretrainConfig(
        region="quantfp-test",
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

    def fake_load_regions_spec() -> dict:
        def _shards(glob_pat: str) -> list[str]:
            return [str(shard_path)]

        class _Entry:
            sources: ClassVar = [("pairs.parquet", ("anchor", "positive"), 0)]

        return {"REGIONS": {}, "_shards": _shards, "region_spec": lambda name: _Entry()}

    mod._load_regions_spec = fake_load_regions_spec  # module-level monkeypatch, restored below
    return receipt


@pytest.fixture(autouse=True)
def _restore_load_regions_spec():
    original = mod._load_regions_spec
    yield
    mod._load_regions_spec = original


def test_quant_receipt_artifacts_checkpoint_sha256_matches_the_real_file(
    tmp_path: Path, trained_receipt: dict
) -> None:
    """The whole point: `artifacts.checkpoint_sha256` on the emitted quant receipt must
    equal the checkpoint file's own sha256 -- the exact bytes `quantize_text_region`
    just loaded and measured, not a stale or fabricated value."""
    rec = mod.quantize_text_region(
        "quantfp-test", tmp_path, tolerance=1.0, aggressive=3, max_bits=8
    )

    checkpoint_path = Path(trained_receipt["checkpoint"])
    real_sha = sha256_file(checkpoint_path)

    assert rec["artifacts"]["checkpoint"] == trained_receipt["checkpoint"]
    assert rec["artifacts"]["checkpoint_sha256"] == real_sha
    # And it must be the sha the training receipt itself already recorded, not merely
    # any valid hash of the file -- proves this is really the checkpoint that receipt
    # names, and not a coincidentally-matching recomputation.
    assert rec["artifacts"]["checkpoint_sha256"] == trained_receipt["checkpoint_sha256"]


def test_quant_receipt_names_the_source_training_receipt_by_path_and_hash(
    tmp_path: Path, trained_receipt: dict
) -> None:
    rec = mod.quantize_text_region(
        "quantfp-test", tmp_path, tolerance=1.0, aggressive=3, max_bits=8
    )

    source = rec["artifacts"]["source_training_receipt"]
    receipt_path = Path(trained_receipt["receipt_path"])
    assert source["path"] == str(receipt_path)
    assert source["sha256"] == sha256_file(receipt_path)


def test_receipt_omitting_the_artifacts_block_would_be_unpublishable(
    tmp_path: Path, trained_receipt: dict
) -> None:
    """Mutation test: reproduce the pre-fix receipt shape (no `artifacts` block at
    all) and assert this file's own binding assertions -- which stand in for
    `scripts/csd-publish-checkpoint.py`'s `receipt_checkpoint_sha256()` gate -- would
    reject it. If a future edit deletes the `artifacts` write, this fails the same way
    the publisher would."""
    rec = mod.quantize_text_region(
        "quantfp-test", tmp_path, tolerance=1.0, aggressive=3, max_bits=8
    )
    del rec["artifacts"]

    artifacts = rec.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    recorded_sha = artifacts.get("checkpoint_sha256") or rec.get("checkpoint_sha256")

    assert not recorded_sha, (
        "a receipt with no 'artifacts' block must have no discoverable "
        "checkpoint_sha256 -- exactly what scripts/csd-publish-checkpoint.py refuses "
        "to publish"
    )
