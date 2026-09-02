"""bf16 autocast must not reach the loss, the master weights, or the saved checkpoint.

Mixed precision is the one change in the training stack that can cost accuracy without
costing anything visible: the loss curve looks normal, the receipt looks normal, and
recall drops by an amount that reads as seed variance. Two specific mechanisms, both
tested here rather than trusted:

  - **The logits matmul.** `info_nce` divides `a @ p.T` by `temperature=0.05`, so a
    unit-cosine logit lands near +-20, where bf16's 8 mantissa bits give a quantum of
    ~0.125. A softmax over a batch of 1,280 classes on logits that coarse is a real
    perturbation of the ranking. `F.cross_entropy` upcasts itself under autocast; the
    matmul feeding it does not, so `info_nce` forces fp32 explicitly.
  - **The artifact.** Autocast must stay a CUDA-context-local decision. If a bf16
    `state_dict` ever reached disk, the checkpoint would stop being portable to the
    fleet's 1080 Ti (sm_61, no bf16 at all) and post-training quantization would be
    measuring a different tensor than it measured before.

The autocast context itself is CUDA-only, so the tests that need one skip without a
bf16-capable GPU. The two that matter most -- the fp32 loss and the fp32 checkpoint --
do not need one and always run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

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
from cogsyndelta.regions.text_encoder import TextEncoderConfig, info_nce

pytestmark = pytest.mark.cpu

_HAS_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def test_info_nce_returns_fp32_from_bf16_inputs() -> None:
    """bf16 embeddings must still produce an fp32 loss and fp32 statistics."""
    torch.manual_seed(0)
    anchors = torch.randn(8, 16, dtype=torch.bfloat16)
    positives = torch.randn(8, 16, dtype=torch.bfloat16)

    loss, stats = info_nce(anchors, positives)

    assert loss.dtype == torch.float32, "the loss must not be computed in bf16"
    assert isinstance(stats["emb_std"], float)
    assert torch.isfinite(loss)


def test_info_nce_is_unchanged_for_fp32_inputs() -> None:
    """The fp32 path must be bit-identical to what it was before the cast was added.

    `Tensor.float()` returns the same tensor when it is already fp32, so this is a
    property of the implementation rather than a tolerance -- and asserting exact
    equality is what would catch someone "simplifying" the cast into a `.to(torch.float32)`
    round trip through another dtype.
    """
    torch.manual_seed(0)
    anchors = torch.randn(8, 16)
    positives = torch.randn(8, 16)

    loss, _ = info_nce(anchors, positives)
    expected = 0.5 * (
        torch.nn.functional.cross_entropy(
            (
                torch.nn.functional.normalize(anchors, dim=-1)
                @ torch.nn.functional.normalize(positives, dim=-1).T
            )
            / 0.05,
            torch.arange(8),
        )
        + torch.nn.functional.cross_entropy(
            (
                torch.nn.functional.normalize(anchors, dim=-1)
                @ torch.nn.functional.normalize(positives, dim=-1).T
            ).T
            / 0.05,
            torch.arange(8),
        )
    )
    assert torch.equal(loss, expected)


@pytest.mark.skipif(not _HAS_BF16, reason="needs a bf16-capable CUDA device")
def test_logits_stay_fp32_under_autocast() -> None:
    """Inside a real bf16 autocast the loss still comes out fp32.

    The dtype check above uses bf16 tensors without an autocast context, which is not
    quite the production situation: autocast rewrites operators, so a cast that looks
    right on plain bf16 inputs could still be undone by the context. This is the one
    that proves it in situ.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    anchors = torch.randn(8, 16, device=device)
    positives = torch.randn(8, 16, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = info_nce(anchors, positives)
    assert loss.dtype == torch.float32


def _tiny_cfg(tmp_path: Path, *, bf16: bool) -> PretrainConfig:
    """A config small enough to train on CPU in a fraction of a second.

    Args:
        tmp_path: Directory for the tokenizer, the corpus shard and the receipt.
        bf16: What to set on the config under test.

    Returns:
        The config.
    """
    n_pairs = 40
    texts = [f"anchor number {i} word" for i in range(n_pairs)] + [
        f"anchor number {i} match" for i in range(n_pairs)
    ]
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106 -- a token id name
    tok.pre_tokenizer = Whitespace()
    tok.train_from_iterator(texts, trainer=WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"]))
    tok.save(str(tmp_path / "tokenizer.json"))
    pq.write_table(
        pa.table({"anchor": texts[:n_pairs], "positive": texts[n_pairs:]}),
        tmp_path / "pairs.parquet",
    )
    return PretrainConfig(
        region="tiny",
        pair_columns=("anchor", "positive"),
        shards=[str(tmp_path / "pairs.parquet")],
        steps=4,
        batch_size=4,
        holdout_pairs=4,
        max_len=8,
        eval_every=100,
        warmup_steps=1,
        checkpoint_every=0,
        bf16=bf16,
        device="cpu",
        encoder=TextEncoderConfig(dim=16, depth=1, n_heads=2, max_len=8),
        tokenizer_path=str(tmp_path / "tokenizer.json"),
        out_dir=str(tmp_path / "out"),
    )


@pytest.mark.parametrize("bf16", [True, False])
def test_saved_state_dict_is_always_fp32(tmp_path: Path, bf16: bool) -> None:
    """The checkpoint must be fp32 whatever `bf16` was asked for.

    Run with `bf16=True` on CPU this also covers the fallback path: a device without
    bf16 support must quietly train in fp32 rather than fail, because the fleet's 1080
    Ti is Pascal and one day something will train there.
    """
    receipt = pretrain_region(_tiny_cfg(tmp_path, bf16=bf16))
    payload = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)

    offenders = {
        name: str(tensor.dtype)
        for name, tensor in payload["model"].items()
        if tensor.is_floating_point() and tensor.dtype is not torch.float32
    }
    assert offenders == {}, f"non-fp32 tensors in the saved checkpoint: {offenders}"
    assert receipt["precision"]["master_weights"] == "fp32"
    assert receipt["precision"]["requested_bf16"] is bf16
    # On CPU there is no bf16 autocast to apply, so the receipt must say fp32 whatever
    # was requested -- a receipt claiming bf16 on a card that cannot do it is the exact
    # provenance error this block exists to prevent.
    assert receipt["precision"]["autocast"] == "fp32"


def test_precision_is_a_resume_relevant_field(tmp_path: Path) -> None:
    """Resuming an fp32 checkpoint under bf16 must be refused, not silently absorbed."""
    cfg = _tiny_cfg(tmp_path, bf16=False)
    pretrain_region(cfg)

    from dataclasses import replace

    with pytest.raises(ValueError, match="different config"):
        pretrain_region(replace(cfg, bf16=True))
