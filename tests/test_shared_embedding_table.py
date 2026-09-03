"""DEC-24 (§6.2, docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md): `PretrainConfig.
init_embedding_from` -- copying another region's token embedding table into a fresh run
instead of a random init, through `_checkpoint.load_checkpoint` (DEC-40/W0c's one
sanctioned `torch.load` entry point).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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
from cogsyndelta.regions.pretrain import (
    PretrainConfig,
    _apply_init_embedding,
    _resume_fields,
    pretrain_region,
)
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig

pytestmark = pytest.mark.cpu


def _write_source_checkpoint(path: Path, table: torch.Tensor) -> None:
    torch.save({"model": {"embed.weight": table}}, path)


def test_apply_init_embedding_copies_the_source_table(tmp_path: Path) -> None:
    cfg = TextEncoderConfig(vocab_size=20, dim=8, depth=1, n_heads=2, max_len=8)
    model = TextEncoder(cfg)
    source_table = torch.randn(20, 8)
    ckpt = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt, source_table)

    before = model.embed.weight.detach().clone()
    result = _apply_init_embedding(model, str(ckpt), torch.device("cpu"))

    assert result["applied"] is True
    assert result["source"] == str(ckpt)
    assert result["source_sha256"] == sha256_file(ckpt)
    assert torch.allclose(model.embed.weight.detach(), source_table)
    assert not torch.allclose(model.embed.weight.detach(), before)


def test_apply_init_embedding_refuses_a_shape_mismatch(tmp_path: Path) -> None:
    cfg = TextEncoderConfig(vocab_size=20, dim=8, depth=1, n_heads=2, max_len=8)
    model = TextEncoder(cfg)
    wrong_shape_table = torch.randn(30, 8)  # different vocab_size
    ckpt = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt, wrong_shape_table)

    with pytest.raises(ValueError, match="cannot share a table"):
        _apply_init_embedding(model, str(ckpt), torch.device("cpu"))


def test_init_embedding_from_is_resume_relevant() -> None:
    """A resume against a DIFFERENT source table is a different run."""
    cfg = PretrainConfig(region="x", pair_columns=("a", "b"), shards=[])
    without = _resume_fields(cfg)
    with_source = _resume_fields(replace(cfg, init_embedding_from="/some/other/checkpoint.pt"))
    assert without["init_embedding_from"] != with_source["init_embedding_from"]


def _build_corpus(tmp_path: Path, n_pairs: int) -> tuple[Path, Path]:
    tok_path = tmp_path / "tokenizer.json"
    shard_path = tmp_path / "pairs.parquet"
    anchors = [f"anchor number {i} word" for i in range(n_pairs)]
    positives = [f"anchor number {i} match" for i in range(n_pairs)]
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tok.train_from_iterator(anchors + positives, trainer=trainer)
    tok.save(str(tok_path))
    pq.write_table(pa.table({"anchor": anchors, "positive": positives}), shard_path)
    return tok_path, shard_path


def test_pretrain_region_applies_and_records_the_inherited_table(tmp_path: Path) -> None:
    """End to end: a fresh run with `init_embedding_from` set both applies the table (the
    untrained baseline is measured AFTER it is copied in, not before) and records it."""
    tok_path, shard_path = _build_corpus(tmp_path, n_pairs=40)
    tok = Tokenizer.from_file(str(tok_path))
    encoder_cfg = TextEncoderConfig(
        vocab_size=tok.get_vocab_size(), dim=8, depth=1, n_heads=2, max_len=16
    )
    source_table = torch.randn(encoder_cfg.vocab_size, encoder_cfg.dim)
    source_ckpt = tmp_path / "source.pt"
    _write_source_checkpoint(source_ckpt, source_table)

    receipt = pretrain_region(
        PretrainConfig(
            region="inherit-test",
            pair_columns=("anchor", "positive"),
            shards=[str(shard_path)],
            steps=4,
            batch_size=4,
            holdout_pairs=4,
            eval_every=2,
            checkpoint_every=0,
            max_len=16,
            seed=0,
            device="cpu",
            bf16=False,
            encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=16),
            tokenizer_path=str(tok_path),
            out_dir=str(tmp_path / "run"),
            init_embedding_from=str(source_ckpt),
        )
    )
    block = receipt["shared_embedding_table"]
    assert block["applied"] is True
    assert block["source"] == str(source_ckpt)
    assert block["source_sha256"] == sha256_file(source_ckpt)


def test_pretrain_region_without_init_embedding_from_records_not_applied(tmp_path: Path) -> None:
    tok_path, shard_path = _build_corpus(tmp_path, n_pairs=40)
    receipt = pretrain_region(
        PretrainConfig(
            region="no-inherit-test",
            pair_columns=("anchor", "positive"),
            shards=[str(shard_path)],
            steps=4,
            batch_size=4,
            holdout_pairs=4,
            eval_every=2,
            checkpoint_every=0,
            max_len=16,
            seed=0,
            device="cpu",
            bf16=False,
            encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=16),
            tokenizer_path=str(tok_path),
            out_dir=str(tmp_path / "run"),
        )
    )
    block = receipt["shared_embedding_table"]
    assert block["applied"] is False
    assert block["source"] is None
