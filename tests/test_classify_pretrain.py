"""Tests for the `classify` specialists' harness (`regions/classify_pretrain.py`).

Mirrors `tests/test_pretrain_resume.py`'s approach: everything runs on CPU with a tiny
encoder and a handful of synthetic rows, so it takes a fraction of a second and does not
touch the fleet's NFS-mounted corpus or a real gpt2 tokenizer. What is exercised here is
new code specific to this harness -- the label-shape resolution (single-label string vs
multi-label id-list-with-metadata) and the average-precision metric -- plus the same
resume/checkpoint contract `regions/pretrain.py` already has tests for, since
`ClassifyPretrainConfig`'s resume path is an independent (if parallel) implementation,
not a reuse of `pretrain.py`'s.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

# MUST precede the imports below: cogsyndelta.regions.classify_pretrain imports
# tokenizers at module scope and needs pyarrow to read parquet, so without the train
# group installed the import errors during collection and the whole file fails instead
# of skipping (same reasoning as tests/test_region_compress.py).
pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from cogsyndelta.regions.classify_pretrain import (
    ClassifyPretrainConfig,
    _average_precision_binary,
    _content_overlap_report,
    build_classify_splits,
    load_classify_rows,
    pretrain_classify_region,
)
from cogsyndelta.regions.text_encoder import TextEncoderConfig

pytestmark = pytest.mark.cpu


def _build_tokenizer(path: Path, texts: list[str]) -> None:
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106 -- a token id name
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(path))


_CLASSES = ["alpha", "beta", "gamma", "delta"]


def _single_label_texts(n_per_class: int) -> tuple[list[str], list[str]]:
    """Distinct texts across `_CLASSES`, each naming its own class so the classifier
    task is trivially learnable -- these tests check plumbing, not learning capacity."""
    texts, labels = [], []
    for cls in _CLASSES:
        for i in range(n_per_class):
            texts.append(f"this is {cls} example number {i}")
            labels.append(cls)
    return texts, labels


def _build_single_label_parquet(path: Path, n_per_class: int) -> None:
    texts, labels = _single_label_texts(n_per_class)
    pq.write_table(pa.table({"text": texts, "category": labels}), path)


_ML_LABEL_NAMES = ["red", "green", "blue"]


def _build_multi_label_parquet(path: Path, n_per_label: int) -> None:
    """Rows carry 1-2 of `_ML_LABEL_NAMES` by id, with HF ClassLabel metadata attached
    -- mirroring go_emotions' `labels` column exactly (see
    `_read_hf_classlabel_names`'s docstring for the shape being matched)."""
    texts: list[str] = []
    label_ids: list[list[int]] = []
    for idx, name in enumerate(_ML_LABEL_NAMES):
        for i in range(n_per_label):
            texts.append(f"this is {name} example number {i}")
            label_ids.append([idx])
    # A couple of genuinely multi-label rows, so the multihot/AP path sees >1 positive
    # per row at least once.
    texts.append("this is red and blue example")
    label_ids.append([0, 2])
    texts.append("this is green and blue example")
    label_ids.append([1, 2])

    table = pa.table({"text": texts, "labels": label_ids})
    hf_meta = {
        "info": {
            "features": {
                "labels": {
                    "feature": {"names": _ML_LABEL_NAMES, "_type": "ClassLabel"},
                    "_type": "List",
                }
            }
        }
    }
    table = table.replace_schema_metadata({"huggingface": json.dumps(hf_meta)})
    pq.write_table(table, path)


@pytest.fixture
def tiny_single_label_cfg(tmp_path: Path) -> ClassifyPretrainConfig:
    """steps=6, batch=4, holdout_rows=4 -> needs >= 8 rows; 4 classes * 6 = 24 supplied."""
    tok_path = tmp_path / "tokenizer.json"
    shard_path = tmp_path / "banking-like.parquet"
    texts, _ = _single_label_texts(6)
    _build_tokenizer(tok_path, texts)
    _build_single_label_parquet(shard_path, 6)
    return ClassifyPretrainConfig(
        region="classify-test-single",
        shards=[str(shard_path)],
        text_column="text",
        label_column="category",
        multi_label=False,
        steps=6,
        batch_size=4,
        holdout_rows=4,
        eval_every=2,
        checkpoint_every=2,
        max_len=16,
        seed=0,
        device="cpu",
        bf16=False,
        encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=16),
        tokenizer_path=str(tok_path),
        out_dir=str(tmp_path / "run"),
    )


@pytest.fixture
def tiny_multi_label_cfg(tmp_path: Path) -> ClassifyPretrainConfig:
    tok_path = tmp_path / "tokenizer.json"
    shard_path = tmp_path / "goemotions-like.parquet"
    texts = [f"this is {n} example number {i}" for n in _ML_LABEL_NAMES for i in range(8)]
    _build_tokenizer(tok_path, texts)
    _build_multi_label_parquet(shard_path, 8)
    return ClassifyPretrainConfig(
        region="classify-test-multi",
        shards=[str(shard_path)],
        text_column="text",
        label_column="labels",
        multi_label=True,
        label_names_from_metadata=True,
        min_holdout_positives=1,
        steps=6,
        batch_size=4,
        holdout_rows=4,
        eval_every=2,
        checkpoint_every=2,
        max_len=16,
        seed=0,
        device="cpu",
        bf16=False,
        encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=16),
        tokenizer_path=str(tok_path),
        out_dir=str(tmp_path / "run"),
    )


# ---------------------------------------------------------------------------------------
# _average_precision_binary: the metric go_emotions is actually gated on.
# ---------------------------------------------------------------------------------------


def test_average_precision_perfect_ranking_is_one() -> None:
    scores = torch.tensor([0.9, 0.8, 0.1, 0.05])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert _average_precision_binary(scores, labels) == pytest.approx(1.0)


def test_average_precision_worst_ranking_is_low() -> None:
    # Two positives ranked dead last out of four (rank order by score desc: neg, neg,
    # pos, pos): precision at each positive's rank is 1/3 (1 of the top 3) and 2/4 (both
    # of the top 4), AP = mean = (1/3 + 2/4) / 2.
    scores = torch.tensor([0.05, 0.1, 0.8, 0.9])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    expected = ((1 / 3) + (2 / 4)) / 2
    assert _average_precision_binary(scores, labels) == pytest.approx(expected)


def test_average_precision_zero_positives_is_none() -> None:
    """Undefined, not zero -- a 0.0 would silently misreport 'no signal' as 'failed'."""
    scores = torch.tensor([0.9, 0.1])
    labels = torch.tensor([0.0, 0.0])
    assert _average_precision_binary(scores, labels) is None


# ---------------------------------------------------------------------------------------
# load_classify_rows: single-label string column vs multi-label id-list + HF metadata.
# ---------------------------------------------------------------------------------------


def test_single_label_class_names_are_sorted_and_deterministic(tmp_path: Path) -> None:
    shard = tmp_path / "single.parquet"
    _build_single_label_parquet(shard, 3)
    texts, label_ids, class_names = load_classify_rows(
        [str(shard)], "text", "category", multi_label=False, label_names_from_metadata=False
    )
    assert class_names == sorted(_CLASSES)
    assert len(texts) == len(label_ids) == 3 * len(_CLASSES)
    # Every label id is a single-element list indexing into class_names correctly.
    for text, ids in zip(texts, label_ids, strict=True):
        assert len(ids) == 1
        cls = text.split()[2]  # "this is <cls> example number N"
        assert class_names[ids[0]] == cls


def test_multi_label_class_names_come_from_hf_metadata_not_derived(tmp_path: Path) -> None:
    shard = tmp_path / "multi.parquet"
    _build_multi_label_parquet(shard, 2)
    texts, label_ids, class_names = load_classify_rows(
        [str(shard)], "text", "labels", multi_label=True, label_names_from_metadata=True
    )
    assert class_names == _ML_LABEL_NAMES
    assert len(texts) == len(label_ids)
    multi = [ids for ids in label_ids if len(ids) > 1]
    assert multi, "expected at least one genuinely multi-label row in the fixture"


def test_multi_label_without_metadata_flag_raises(tmp_path: Path) -> None:
    """There is no safe data-derived class order for ids that may never appear as a
    positive -- see load_classify_rows' docstring. Refuse rather than guess."""
    shard = tmp_path / "multi.parquet"
    _build_multi_label_parquet(shard, 2)
    with pytest.raises(ValueError, match="label_names_from_metadata"):
        load_classify_rows(
            [str(shard)],
            "text",
            "labels",
            multi_label=True,
            label_names_from_metadata=False,
        )


# ---------------------------------------------------------------------------------------
# build_classify_splits: dedup and contamination, matching pretrain.py's build_splits.
# ---------------------------------------------------------------------------------------


def test_holdout_and_train_never_share_a_text(
    tiny_single_label_cfg: ClassifyPretrainConfig,
) -> None:
    (holdout_texts, _), (train_texts, _), _, meta = build_classify_splits(tiny_single_label_cfg)
    assert set(holdout_texts).isdisjoint(train_texts)
    assert meta["contamination"]["overlap"] == 0


def test_contamination_channel_is_tautologically_zero_but_content_overlap_can_fire() -> None:
    """`contamination` keys on the same normalisation dedup already partitioned on, so it
    is 0 whatever the data says (see `_content_overlap_report`'s docstring -- this is the
    single-text analogue of the bug `cogsyndelta.eval.metrics.pair_contamination_report`
    fixed for (anchor, positive) pairs). `content_overlap` keys on content words instead,
    so a paraphrase that survives exact-text dedup is still visible."""
    train = ["how do you know if a mango is ripe", "totally unrelated other sentence"]
    holdout = ["how to know if a mango is ripe"]  # same content words, different wording
    report = _content_overlap_report(train, holdout)
    assert report["overlap"] == 1
    assert report["holdout_fraction_overlapping"] == pytest.approx(1.0)
    assert "ripe" in report["examples"][0]


def test_content_overlap_is_zero_for_genuinely_distinct_texts() -> None:
    train = ["how do you know if a mango is ripe"]
    holdout = ["what time does the bank branch close today"]
    report = _content_overlap_report(train, holdout)
    assert report["overlap"] == 0
    assert report["holdout_fraction_overlapping"] == 0.0


def test_duplicate_rows_are_deduped_before_split(tmp_path: Path) -> None:
    tok_path = tmp_path / "tok.json"
    shard = tmp_path / "dupes.parquet"
    texts, labels = _single_label_texts(6)
    _build_tokenizer(tok_path, texts)
    # Duplicate the whole corpus (identical text+label rows) so post-dedup row count is
    # exactly what a single copy would give.
    pq.write_table(pa.table({"text": texts + texts, "category": labels + labels}), shard)
    cfg = ClassifyPretrainConfig(
        region="dedup-test",
        shards=[str(shard)],
        text_column="text",
        label_column="category",
        multi_label=False,
        steps=1,
        batch_size=4,
        holdout_rows=4,
        max_len=16,
        seed=0,
        device="cpu",
        encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=16),
        tokenizer_path=str(tok_path),
        out_dir=str(tmp_path / "run"),
    )
    (holdout_texts, _), (train_texts, _), _, meta = build_classify_splits(cfg)
    assert meta["duplicates_removed"] == len(texts)
    assert len(holdout_texts) + len(train_texts) == len(texts)


# ---------------------------------------------------------------------------------------
# End-to-end: single-label and multi-label both train and produce an honest receipt.
# ---------------------------------------------------------------------------------------


def test_single_label_receipt_has_chance_floor_and_gate(
    tiny_single_label_cfg: ClassifyPretrainConfig,
) -> None:
    receipt = pretrain_classify_region(tiny_single_label_cfg)
    assert receipt["schema"] == "csd-classify-receipt/v1"
    assert set(receipt["beats_untrained"]) == {"top1", "macro_f1"}
    assert receipt["untrained_baseline"]["top1_chance"] == pytest.approx(1 / len(_CLASSES))
    assert receipt["class_names"] == sorted(_CLASSES)


def test_multi_label_receipt_reports_macro_and_micro_ap(
    tiny_multi_label_cfg: ClassifyPretrainConfig,
) -> None:
    receipt = pretrain_classify_region(tiny_multi_label_cfg)
    assert receipt["schema"] == "csd-classify-receipt/v1"
    assert set(receipt["beats_untrained"]) == {"macro_ap", "micro_ap"}
    held = receipt["held_out"]
    assert 0.0 <= held["macro_ap"] <= 1.0
    assert 0.0 <= held["macro_ap_chance"] <= 1.0
    assert set(held["per_label"]) == set(_ML_LABEL_NAMES)


# ---------------------------------------------------------------------------------------
# Resuming produces the same result as an uninterrupted run (mirrors
# tests/test_pretrain_resume.py::test_resume_matches_uninterrupted_run).
# ---------------------------------------------------------------------------------------


def test_resume_matches_uninterrupted_run(
    tiny_single_label_cfg: ClassifyPretrainConfig, monkeypatch
) -> None:
    import cogsyndelta.regions.classify_pretrain as classify_mod

    uninterrupted_cfg = replace(
        tiny_single_label_cfg, out_dir=str(Path(tiny_single_label_cfg.out_dir) / "uninterrupted")
    )
    reference = pretrain_classify_region(uninterrupted_cfg)

    crashing_cfg = replace(
        tiny_single_label_cfg, out_dir=str(Path(tiny_single_label_cfg.out_dir) / "crashed")
    )

    orig_cross_entropy = classify_mod.F.cross_entropy
    call_count = {"n": 0}

    def flaky_cross_entropy(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 4:  # right after the step=2 checkpoint has been written
            raise RuntimeError("simulated crash")
        return orig_cross_entropy(*args, **kwargs)

    monkeypatch.setattr(classify_mod.F, "cross_entropy", flaky_cross_entropy)
    with pytest.raises(RuntimeError, match="simulated crash"):
        pretrain_classify_region(crashing_cfg)
    monkeypatch.undo()

    ckpt_dir = Path(crashing_cfg.out_dir) / f"{crashing_cfg.region}-checkpoints"
    assert (ckpt_dir / "step-000002.pt").is_file(), "expected a checkpoint before the crash"

    resumed = pretrain_classify_region(crashing_cfg)
    assert resumed["resumed"] is True
    assert resumed["resumed_from_step"] == 3
    assert resumed["untrained_baseline"] == reference["untrained_baseline"]

    ref_ckpt = torch.load(reference["checkpoint"], map_location="cpu", weights_only=True)
    res_ckpt = torch.load(resumed["checkpoint"], map_location="cpu", weights_only=True)
    for key, value in ref_ckpt["model"].items():
        assert torch.allclose(res_ckpt["model"][key], value, atol=1e-6, rtol=1e-5), (
            f"resumed run's final encoder weight for {key} diverged"
        )
    for key, value in ref_ckpt["head"].items():
        assert torch.allclose(res_ckpt["head"][key], value, atol=1e-6, rtol=1e-5), (
            f"resumed run's final head weight for {key} diverged"
        )
    assert resumed["held_out"]["top1"] == pytest.approx(reference["held_out"]["top1"], abs=1e-6)


def test_a_completed_run_resumed_again_trains_zero_more_steps(
    tiny_single_label_cfg: ClassifyPretrainConfig,
) -> None:
    first = pretrain_classify_region(tiny_single_label_cfg)
    second = pretrain_classify_region(tiny_single_label_cfg)
    assert second["resumed"] is True
    assert second["resumed_from_step"] == tiny_single_label_cfg.steps
    assert second["held_out"] == first["held_out"]


def test_config_mismatch_is_refused_not_silently_accepted(
    tiny_single_label_cfg: ClassifyPretrainConfig,
) -> None:
    pretrain_classify_region(tiny_single_label_cfg)
    changed = replace(tiny_single_label_cfg, batch_size=tiny_single_label_cfg.batch_size + 1)
    with pytest.raises(ValueError, match="refusing to resume"):
        pretrain_classify_region(changed)


def test_config_mismatch_error_names_the_differing_field(
    tiny_single_label_cfg: ClassifyPretrainConfig,
) -> None:
    pretrain_classify_region(tiny_single_label_cfg)
    changed = replace(tiny_single_label_cfg, lr=tiny_single_label_cfg.lr * 2)
    with pytest.raises(ValueError, match=re.escape("lr: checkpoint=")):
        pretrain_classify_region(changed)
