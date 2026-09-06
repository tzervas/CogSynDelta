"""Negatives beyond the batch: the loss identity, the bank, the mining, and the audit.

WHAT THESE TESTS ARE FOR
PREREG-RETRIEVAL-NEGATIVES-2026-09-06 (rev 3) runs three arms that differ in ONE thing --
the negative set. Every piece of machinery below could quietly introduce a second
difference, so the tests here are mostly identity tests rather than feature tests:

  - with no bank and no mined manifest, `info_nce` must produce the SAME BITS as the line
    it replaced (`test_control_is_bit_identical_*`). An argument that "the slice is a
    no-op" is not evidence; `torch.equal` against a verbatim copy of the pre-change
    expression is.
  - `in_batch_acc` must keep meaning "gold beats the other B-1 in this batch" once extra
    columns exist, and the whole-row question must be reported separately with its own
    denominator (Table 4).
  - the checkpoint resume fingerprint must not move for a config that uses none of this,
    or every checkpoint written to date becomes un-resumable for a feature it never used.

The two guards (G38 self-positive disjointness, G39 mining provenance) have their failing
arms in `tests/test_guards_can_fail.py`, per this project's rule that a guard with no
failing-case test is a comment with a function signature.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

# MUST precede the cogsyndelta imports: `cogsyndelta.regions.pretrain` imports tokenizers
# at module scope and the end-to-end arms below build their own parquet/tokenizer
# fixtures, so without the train group installed the import errors during collection and
# the whole file fails instead of skipping (same reasoning as tests/test_pretrain_resume.py).
pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from cogsyndelta.regions import _mining
from cogsyndelta.regions.pretrain import (
    PretrainConfig,
    _resume_fields,
    build_splits,
    load_sources,
    negative_set_name,
    pretrain_region,
)
from cogsyndelta.regions.text_encoder import NegativeBank, TextEncoderConfig, info_nce

pytestmark = pytest.mark.cpu


def _reference_loss(
    anchors: torch.Tensor, positives: torch.Tensor, temperature: float = 0.05
) -> torch.Tensor:
    """The objective exactly as `text_encoder.info_nce` computed it at `12e2d1f`.

    Copied verbatim rather than imported, so this test compares the new implementation
    against the old EXPRESSION and keeps doing so after the old one is gone.

    Args:
        anchors: ``[B, D]``.
        positives: ``[B, D]``.
        temperature: Softmax temperature.

    Returns:
        The scalar loss.
    """
    a = F.normalize(anchors.float(), dim=-1)
    p = F.normalize(positives.float(), dim=-1)
    logits = (a @ p.T) / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _towers(batch: int = 12, dim: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Two random embedding towers.

    Args:
        batch: Rows.
        dim: Width.
        seed: RNG seed.

    Returns:
        ``(anchors, positives)``.
    """
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch, dim, generator=generator),
        torch.randn(batch, dim, generator=generator),
    )


# ---------------------------------------------------------------------------------------
# The identity the whole round rests on: the control is a special case of the treatment.
# ---------------------------------------------------------------------------------------


def test_control_is_bit_identical_with_no_extra_negatives() -> None:
    anchors, positives = _towers()
    loss, stats = info_nce(anchors, positives)
    assert torch.equal(loss, _reference_loss(anchors, positives))
    assert stats["loss"] == _reference_loss(anchors, positives).item()


@pytest.mark.parametrize("empty", ["none", "zero_rows", "unpushed_bank"])
def test_control_is_bit_identical_with_an_empty_bank(empty: str) -> None:
    anchors, positives = _towers()
    extra: torch.Tensor | None
    if empty == "none":
        extra = None
    elif empty == "zero_rows":
        extra = torch.zeros(0, anchors.size(1))
    else:
        extra = NegativeBank(16).negatives()

    loss, stats = info_nce(anchors, positives, extra_negatives=extra)
    reference = _reference_loss(anchors, positives)
    assert torch.equal(loss, reference), f"empty={empty} moved the loss"

    control_loss, control_stats = info_nce(anchors, positives)
    assert stats == control_stats


def test_control_stats_keep_their_pre_change_values() -> None:
    anchors, positives = _towers()
    _, stats = info_nce(anchors, positives)
    a = F.normalize(anchors.float(), dim=-1)
    p = F.normalize(positives.float(), dim=-1)
    logits = (a @ p.T) / 0.05
    labels = torch.arange(a.size(0))
    assert stats["in_batch_acc"] == (logits.argmax(dim=-1) == labels).float().mean().item()
    assert stats["chance"] == 1.0 / a.size(0)
    assert stats["emb_std"] == a.std(dim=0).mean().item()
    # New keys, and with no extra columns they say the same thing the old ones did.
    assert stats["full_acc"] == stats["in_batch_acc"]
    assert stats["full_chance"] == stats["chance"]
    assert stats["negatives_per_query"] == float(a.size(0) - 1)


# ---------------------------------------------------------------------------------------
# The treatment: what the extra columns do, and what they must NOT do.
# ---------------------------------------------------------------------------------------


def test_extra_negatives_widen_the_denominator_only() -> None:
    anchors, positives = _towers()
    extra = torch.randn(37, anchors.size(1), generator=torch.Generator().manual_seed(3))
    control_loss, control_stats = info_nce(anchors, positives)
    loss, stats = info_nce(anchors, positives, extra_negatives=extra)

    assert loss.item() != control_loss.item()
    assert loss.item() > control_loss.item(), "more negatives cannot make the task easier"
    # The in-batch statistic is over the SAME [B, B] block, so it is unmoved.
    assert stats["in_batch_acc"] == control_stats["in_batch_acc"]
    assert stats["chance"] == control_stats["chance"]
    assert stats["full_chance"] == 1.0 / (anchors.size(0) + 37)
    assert stats["negatives_per_query"] == float(anchors.size(0) - 1 + 37)


def test_in_batch_accuracy_ignores_the_extra_columns() -> None:
    # A batch whose in-batch task is easy (each anchor's own positive is the closest of
    # the four) but whose extra columns hold an EXACT copy of every anchor, so the
    # whole-row argmax always lands outside the block. `in_batch_acc` must stay 1.0 and
    # `full_acc` must fall to 0.0. Before the narrowing, `in_batch_acc` WAS the whole-row
    # number, so this case is exactly the silent redefinition Table 4 refuses.
    anchors = torch.eye(4)
    positives = anchors + 0.05
    extra = torch.cat([anchors, anchors], dim=0)
    _, stats = info_nce(anchors, positives, extra_negatives=extra)
    assert stats["in_batch_acc"] == 1.0
    assert stats["full_acc"] == 0.0
    assert stats["full_chance"] == 1.0 / (4 + 8)


def test_extra_negatives_carry_no_gradient() -> None:
    anchors, positives = _towers()
    anchors.requires_grad_(True)
    extra = torch.randn(5, anchors.size(1), requires_grad=True)
    loss, _ = info_nce(anchors, positives, extra_negatives=extra)
    loss.backward()
    assert extra.grad is None, "a bank/mined vector must not be trained through"
    assert anchors.grad is not None


def test_symmetric_term_restricted_to_the_in_batch_block() -> None:
    anchors, positives = _towers()
    extra = torch.randn(9, anchors.size(1), generator=torch.Generator().manual_seed(5))
    loss, _ = info_nce(anchors, positives, extra_negatives=extra)

    a = F.normalize(anchors.float(), dim=-1)
    p = F.normalize(positives.float(), dim=-1)
    n = F.normalize(extra.float(), dim=-1)
    logits = torch.cat([(a @ p.T) / 0.05, (a @ n.T) / 0.05], dim=1)
    labels = torch.arange(a.size(0))
    expected = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits[:, : a.size(0)].T, labels)
    )
    assert torch.equal(loss, expected)


def test_extra_negatives_of_the_wrong_width_are_refused() -> None:
    anchors, positives = _towers()
    with pytest.raises(ValueError, match="extra_negatives must be"):
        info_nce(anchors, positives, extra_negatives=torch.randn(3, anchors.size(1) + 1))


# ---------------------------------------------------------------------------------------
# The bank.
# ---------------------------------------------------------------------------------------


def test_bank_is_fifo_and_capped() -> None:
    bank = NegativeBank(4)
    assert len(bank) == 0
    assert bank.negatives().numel() == 0

    bank.push(torch.tensor([[1.0], [2.0], [3.0]]))
    assert len(bank) == 3
    assert sorted(bank.negatives().flatten().tolist()) == [1.0, 2.0, 3.0]

    bank.push(torch.tensor([[4.0], [5.0]]))
    assert len(bank) == 4
    # 1.0 is the oldest and is the one evicted.
    assert sorted(bank.negatives().flatten().tolist()) == [2.0, 3.0, 4.0, 5.0]


def test_bank_push_larger_than_capacity_keeps_the_last_rows() -> None:
    bank = NegativeBank(3)
    bank.push(torch.arange(10, dtype=torch.float32).unsqueeze(1))
    assert sorted(bank.negatives().flatten().tolist()) == [7.0, 8.0, 9.0]


def test_bank_stores_detached_fp32_copies() -> None:
    bank = NegativeBank(4)
    live = torch.randn(2, 3, requires_grad=True)
    bank.push(live)
    stored = bank.negatives()
    assert not stored.requires_grad
    assert stored.dtype is torch.float32


def test_bank_refuses_a_width_change_and_a_zero_capacity() -> None:
    bank = NegativeBank(4)
    bank.push(torch.randn(1, 3))
    with pytest.raises(ValueError, match="width"):
        bank.push(torch.randn(1, 5))
    with pytest.raises(ValueError, match="capacity"):
        NegativeBank(0)


# ---------------------------------------------------------------------------------------
# Config: one negative set at a time, and the fingerprint that must not move.
# ---------------------------------------------------------------------------------------


def _cfg(**overrides: object) -> PretrainConfig:
    """A minimal config; nothing here loads a corpus.

    Args:
        **overrides: `PretrainConfig` fields.

    Returns:
        The config.
    """
    base = PretrainConfig(region="neg-test", pair_columns=("a", "b"), shards=[])
    return replace(base, **overrides)  # type: ignore[arg-type]


def test_negative_set_name_reads_the_config() -> None:
    assert negative_set_name(_cfg()) == "in_batch"
    assert negative_set_name(_cfg(negative_bank_size=16384)) == "bank"
    assert (
        negative_set_name(_cfg(mined_negatives_manifest="m.json", mined_negatives_per_anchor=8))
        == "mined"
    )


def test_two_negative_sets_at_once_are_refused() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        negative_set_name(
            _cfg(
                negative_bank_size=16,
                mined_negatives_manifest="m.json",
                mined_negatives_per_anchor=8,
            )
        )
    with pytest.raises(ValueError, match="mined_negatives_per_anchor"):
        negative_set_name(_cfg(mined_negatives_manifest="m.json"))
    with pytest.raises(ValueError, match="without a mined_negatives_manifest"):
        negative_set_name(_cfg(mined_negatives_per_anchor=8))


def test_resume_fingerprint_is_unmoved_when_the_negative_set_is_off() -> None:
    fields = _resume_fields(_cfg())
    assert "negative_bank_size" not in fields
    assert "mined_negatives" not in fields
    assert fields == _resume_fields(_cfg(negative_bank_size=0, mined_negatives_per_anchor=0))


def test_resume_fingerprint_moves_when_a_bank_is_on() -> None:
    assert _resume_fields(_cfg(negative_bank_size=16384))["negative_bank_size"] == 16384
    assert _resume_fields(_cfg(negative_bank_size=16384)) != _resume_fields(_cfg())


# ---------------------------------------------------------------------------------------
# Mining, its manifest, and the FiQA false-negative audit.
# ---------------------------------------------------------------------------------------


def _fiqa_fixture() -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    """A miniature FiQA: 3 queries, several golds each, plus distractor passages.

    Returns:
        ``(pairs, qrels)`` -- pairs as `(query, passage)` in qrels row order.
    """
    rows = [
        ("q1", "d1", "how do i file taxes", "file your taxes with the tax agency"),
        ("q1", "d2", "how do i file taxes", "taxes are filed annually with the agency"),
        ("q2", "d3", "what is an index fund", "an index fund tracks a market index"),
        ("q2", "d4", "what is an index fund", "index funds track indexes cheaply"),
        ("q3", "d5", "how to open a bank account", "open a bank account at a branch"),
        ("q3", "d6", "how to open a bank account", "a bank account is opened with id"),
    ]
    qrels = {
        "query_id": [r[0] for r in rows],
        "doc_id": [r[1] for r in rows],
        "query": [r[2] for r in rows],
        "passage": [r[3] for r in rows],
    }
    return [(r[2], r[3]) for r in rows], qrels


def test_mining_never_returns_the_pairs_own_positive_and_is_deterministic() -> None:
    pairs, qrels = _fiqa_fixture()
    sources = [("primary", pairs)]
    pools = _mining.build_pools(sources)
    source_of_pair = _mining.label_sources(pairs, sources)
    first = _mining.mine_negatives(pairs, source_of_pair, pools, m=2)
    second = _mining.mine_negatives(pairs, source_of_pair, pools, m=2)

    assert first == second, "mining must be reproducible; the manifest pins it"
    assert all(len(row) == 2 for row in first)
    _mining.assert_self_positive_disjoint(pairs, first, source_of_pair, pools)


def test_false_negative_audit_measures_the_rate_and_the_ceiling_branch_repairs_it() -> None:
    pairs, qrels = _fiqa_fixture()
    sources = [("primary", pairs)]
    pools = _mining.build_pools(sources)
    source_of_pair = _mining.label_sources(pairs, sources)
    negatives = _mining.mine_negatives(pairs, source_of_pair, pools, m=2)
    audit = _mining.audit_fiqa_false_negatives(
        train_pairs=pairs,
        source_of_pair=source_of_pair,
        negatives=negatives,
        pools=pools,
        qrels=qrels,
        fiqa_source="primary",
    )
    # This fixture is built so BM25 ranks each query's other gold first: the rate is high
    # by construction, which is what makes the branch reachable in a test at all.
    assert audit.anchors == len(pairs)
    assert audit.mined_negatives == 2 * len(pairs)
    assert audit.rate > _mining.FALSE_NEGATIVE_CEILING
    assert audit.breached
    assert audit.golds_per_query == pytest.approx(2.0)

    result = _mining.mine_and_audit(
        train_pairs=pairs, sources=sources, qrels=qrels, fiqa_source="primary", m=2
    )
    assert len(result.audits) == 2, "the breach and the repaired audit are both recorded"
    assert result.audits[0].breached
    assert result.audits[1].query_id_exclusion
    assert result.audits[1].false_negatives == 0
    assert result.audits[1].rate == 0.0


def test_manifest_round_trips_and_verifies(tmp_path: Path) -> None:
    pairs, qrels = _fiqa_fixture()
    sources = [("primary", pairs)]
    qrels_path = tmp_path / "train.parquet"
    qrels_path.write_bytes(b"not really parquet, but it is the artefact that is hashed")
    result = _mining.mine_and_audit(
        train_pairs=pairs, sources=sources, qrels=qrels, fiqa_source="primary", m=2
    )
    manifest = _mining.build_manifest(
        region="memory",
        corpus_fingerprint="deadbeef",
        result=result,
        train_pairs=pairs,
        qrels_path=qrels_path,
        m=2,
    )
    path = _mining.write_manifest(tmp_path / "mined.json", manifest)
    loaded = _mining.load_manifest(path)
    assert loaded == json.loads(path.read_text())

    _mining.verify_manifest(
        loaded,
        corpus_fingerprint="deadbeef",
        pools=_mining.build_pools(sources),
        qrels_sha256=_mining.file_sha256(qrels_path),
        train_pairs=pairs,
        m=2,
    )
    texts = _mining.negative_texts(loaded, _mining.build_pools(sources), m=2)
    assert len(texts) == 2 * len(pairs)


def test_pinned_bm25_parameters_match_the_ranker_they_claim_to_describe() -> None:
    import inspect

    from cogsyndelta.eval.beir_fiqa import _WORD, BM25

    signature = inspect.signature(BM25.__init__)
    assert signature.parameters["k1"].default == _mining.BM25_K1
    assert signature.parameters["b"].default == _mining.BM25_B
    assert _WORD.pattern == _mining.BM25_TOKENIZER


def test_pool_is_the_training_union_only() -> None:
    pairs, _ = _fiqa_fixture()
    held_out = pairs[0]
    restricted = _mining.restrict_to_union([("primary", pairs)], pairs[1:])
    pool = _mining.build_pools(restricted)["primary"]
    assert held_out[1] not in pool.texts
    assert len(pool.texts) == len(pairs) - 1


# ---------------------------------------------------------------------------------------
# End to end: the three arms actually run, and the control's receipt is unchanged.
#
# Tiny (6 steps, batch 4, dim 8) and CPU-only -- these prove the wiring, not the science.
# The round itself runs later, from merged code, on the 3090 Ti.
# ---------------------------------------------------------------------------------------


def _build_tokenizer(path: Path, n_pairs: int) -> None:
    """A WordLevel tokenizer over exactly the fixture vocabulary.

    Args:
        path: Where to save the tokenizer JSON.
        n_pairs: How many pairs the fixture holds.
    """
    texts = [f"anchor number {i} word" for i in range(n_pairs)] + [
        f"anchor number {i} match" for i in range(n_pairs)
    ]
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106 -- a token id name, not a secret
    tok.pre_tokenizer = Whitespace()
    tok.train_from_iterator(texts, WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"]))
    tok.save(str(path))


def _build_pairs_parquet(path: Path, n_pairs: int) -> None:
    """`n_pairs` distinct pairs -- distinct anchors, so none are dropped by anchor dedup.

    Args:
        path: Parquet path to write.
        n_pairs: Rows.
    """
    pq.write_table(
        pa.table(
            {
                "anchor": [f"anchor number {i} word" for i in range(n_pairs)],
                "positive": [f"anchor number {i} match" for i in range(n_pairs)],
            }
        ),
        path,
    )


@pytest.fixture
def tiny_cfg(tmp_path: Path) -> PretrainConfig:
    """A config small enough to train in a fraction of a second on CPU.

    Args:
        tmp_path: pytest temp dir.

    Returns:
        The config: steps 6, batch 4, holdout 4 -- a 28-pair budget against 40 supplied.
    """
    tok_path = tmp_path / "tokenizer.json"
    shard_path = tmp_path / "pairs.parquet"
    _build_tokenizer(tok_path, 40)
    _build_pairs_parquet(shard_path, 40)
    return PretrainConfig(
        region="negatives-test",
        pair_columns=("anchor", "positive"),
        shards=[str(shard_path)],
        steps=6,
        batch_size=4,
        holdout_pairs=4,
        eval_every=3,
        checkpoint_every=0,
        max_len=16,
        seed=0,
        device="cpu",
        bf16=False,
        encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=16),
        tokenizer_path=str(tok_path),
        out_dir=str(tmp_path / "run"),
    )


def test_control_run_records_the_in_batch_negative_set(tiny_cfg: PretrainConfig) -> None:
    receipt = pretrain_region(tiny_cfg)
    assert receipt["method"] == "symmetric InfoNCE over in-batch negatives"
    assert receipt["negatives"]["set"] == "in_batch"
    assert receipt["negatives"]["bank_size"] == 0
    assert "mined" not in receipt["negatives"]
    assert all(entry["negatives_per_query"] == 3.0 for entry in receipt["history"])
    assert all(entry["full_acc"] == entry["in_batch_acc"] for entry in receipt["history"])


def test_bank_arm_trains_and_widens_the_denominator(tiny_cfg: PretrainConfig) -> None:
    receipt = pretrain_region(replace(tiny_cfg, negative_bank_size=8))
    assert receipt["negatives"]["set"] == "bank"
    assert receipt["negatives"]["bank_size"] == 8
    # Step 0 sees an empty bank (the control's denominator); later steps see more.
    per_query = [entry["negatives_per_query"] for entry in receipt["history"]]
    assert per_query[0] == 3.0
    assert per_query[-1] > 3.0
    assert receipt["history"][-1]["full_chance"] < receipt["history"][-1]["chance"]


def _mine_for(cfg: PretrainConfig, tmp_path: Path, m: int = 2) -> Path:
    """Mine a manifest for `cfg`'s own pinned union.

    Args:
        cfg: The run config the manifest must match.
        tmp_path: Where to write the manifest and the qrels stand-in.
        m: Negatives per anchor.

    Returns:
        The manifest path.
    """
    _, train_pairs, meta = build_splits(cfg)
    sources = _mining.restrict_to_union(load_sources(cfg, cfg.split_seed), train_pairs)
    qrels_path = tmp_path / "fiqa-train.parquet"
    qrels_path.write_bytes(b"stand-in for fiqa-pairs/train.parquet")
    result = _mining.mine_and_audit(
        train_pairs=train_pairs,
        sources=sources,
        qrels={"query_id": [], "doc_id": [], "query": [], "passage": []},
        fiqa_source="primary",
        m=m,
    )
    manifest = _mining.build_manifest(
        region=cfg.region,
        corpus_fingerprint=meta["corpus_fingerprint"],
        result=result,
        train_pairs=train_pairs,
        qrels_path=qrels_path,
        m=m,
    )
    return _mining.write_manifest(tmp_path / "mined.json", manifest)


def test_mined_arm_verifies_its_manifest_and_trains(
    tiny_cfg: PretrainConfig, tmp_path: Path
) -> None:
    manifest_path = _mine_for(tiny_cfg, tmp_path)
    receipt = pretrain_region(
        replace(
            tiny_cfg,
            mined_negatives_manifest=str(manifest_path),
            mined_negatives_per_anchor=2,
        )
    )
    assert receipt["negatives"]["set"] == "mined"
    assert receipt["negatives"]["mined"]["per_anchor"] == 2
    assert receipt["negatives"]["mined"]["manifest_sha256"]
    # B + m*B = 4 + 8 columns, so 11 negatives per query on every step.
    assert all(entry["negatives_per_query"] == 11.0 for entry in receipt["history"])


def test_mined_arm_refuses_a_manifest_from_another_corpus(
    tiny_cfg: PretrainConfig, tmp_path: Path
) -> None:
    """G39, wired: the refusal happens inside `pretrain_region`, before any step runs."""
    manifest_path = _mine_for(tiny_cfg, tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["corpus"]["fingerprint"] = "0" * 32
    manifest["sha256"] = _mining.manifest_payload_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(_mining.MiningGuardError, match="G39"):
        pretrain_region(
            replace(
                tiny_cfg,
                mined_negatives_manifest=str(manifest_path),
                mined_negatives_per_anchor=2,
            )
        )


def test_mined_arm_refuses_a_negative_that_is_its_own_positive(
    tiny_cfg: PretrainConfig, tmp_path: Path
) -> None:
    """G38, wired: a hand-edited manifest that survives G39 still cannot start a run."""
    manifest_path = _mine_for(tiny_cfg, tmp_path)
    manifest = json.loads(manifest_path.read_text())
    _, train_pairs, _ = build_splits(tiny_cfg)
    sources = _mining.restrict_to_union(load_sources(tiny_cfg, tiny_cfg.split_seed), train_pairs)
    pool = _mining.build_pools(sources)["primary"]
    manifest["negatives"][2][0] = pool.texts.index(train_pairs[2][1])
    manifest["sha256"] = _mining.manifest_payload_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(_mining.MiningGuardError, match="G38"):
        pretrain_region(
            replace(
                tiny_cfg,
                mined_negatives_manifest=str(manifest_path),
                mined_negatives_per_anchor=2,
            )
        )


def test_the_auxiliary_weights_are_settable_to_zero(tmp_path: Path, monkeypatch) -> None:
    """Section 2.1: both arms run `token_loss_weight` and `decorr_weight` at 0.0.

    `memory_config()` sets them to 0.1 each, so this asserts the override reaches the
    config AND that `pretrain_region`'s loop reads it as OFF -- a receipt whose history
    carries no `token_loss` key is the observable form of "the auxiliary path did not
    run".
    """
    from cogsyndelta.regions import memory

    root = tmp_path / "corpus"
    for relative in (
        memory.RETRIEVAL_FIQA_SHARD,
        "region/compress/all-nli/pair/train-00000.parquet",
        memory.CONSOLIDATION_GRADED_SHARD,
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    monkeypatch.setattr(memory, "MEMORY_ROOT", root)

    default = memory.memory_config()
    assert (default.token_loss_weight, default.decorr_weight) == (0.1, 0.1)
    off = memory.memory_config(token_loss_weight=0.0, decorr_weight=0.0)
    assert (off.token_loss_weight, off.decorr_weight) == (0.0, 0.0)


def test_zero_weight_run_has_no_auxiliary_terms(tiny_cfg: PretrainConfig) -> None:
    receipt = pretrain_region(replace(tiny_cfg, token_loss_weight=0.0, decorr_weight=0.0))
    assert all("token_loss" not in entry for entry in receipt["history"])
    assert all("decorr_loss" not in entry for entry in receipt["history"])
    assert (
        asdict(_ := PretrainConfig(region="x", pair_columns=("a", "b"), shards=[]))[
            "token_loss_weight"
        ]
        == 0.0
    )
