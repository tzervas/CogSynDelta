"""Compress-region wiring tests.

These skip when the fleet dataset export is not mounted, for the same reason
tests/test_corpus.py does: the corpus is an NFS export from another host and no CI runner
will ever have it. What CI still gets from this file is that the module imports, that the
gate arithmetic is right, and that a missing mount fails loudly instead of silently
training on nothing.

The gate logic is deliberately testable without the corpus. It is the part that decides
whether P1 advances, and a gate that can only be exercised on a training host is a gate
nobody checks.
"""

from __future__ import annotations

import pytest

# MUST precede the import below: cogsyndelta.regions.pretrain imports tokenizers at module
# scope and load_graded_pairs needs pyarrow, so without the train group installed the
# import errors during collection and the whole file fails instead of skipping.
pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

from cogsyndelta.regions import compress as compress_mod
from cogsyndelta.regions.pretrain import _pair_key, load_graded_pairs, pretrain_region
from cogsyndelta.regions.text_encoder import TextEncoderConfig

_HAVE_CORPUS = (compress_mod.COMPRESS_ROOT / compress_mod.TRAIN_SHARD).is_file()
needs_corpus = pytest.mark.skipif(
    not _HAVE_CORPUS,
    reason=f"compress corpus not mounted at {compress_mod.COMPRESS_ROOT}",
)


@pytest.mark.cpu
def test_missing_corpus_raises_instead_of_training_on_nothing(tmp_path, monkeypatch) -> None:
    """An unmounted export must name itself, not surface as an empty corpus.

    Globbing or opening a missing path is the single most likely operational failure here,
    and the default symptom -- zero pairs -- sends you to look at the data instead of at
    the mount.
    """
    monkeypatch.setattr(compress_mod, "COMPRESS_ROOT", tmp_path / "definitely-absent")
    with pytest.raises(FileNotFoundError, match="compress corpus shard missing"):
        compress_mod.compress_config()


@pytest.mark.cpu
def test_pair_key_is_order_independent() -> None:
    """Similarity is symmetric, so (a, b) in training and (b, a) in the eval set is one
    leak, not two unrelated pairs. An ordered key would miss half of them."""
    assert _pair_key("A man plays guitar.", "Someone plays a guitar.") == _pair_key(
        "  someone   PLAYS a guitar.  ", "a man plays guitar."
    )
    assert _pair_key("a b", "c d") != _pair_key("a b", "c e")


@pytest.mark.cpu
def test_gate_reports_a_pass_that_did_not_beat_the_untrained_model() -> None:
    """Clearing a threshold without beating random init is not a win.

    This is the lesson the `code` region paid for: a random-init encoder already scores
    recall@1 ~0.40 on CodeSearchNet from lexical overlap alone, so a threshold can be
    cleared by the corpus rather than by the training. The verdict must say so rather
    than reporting a bare PASS.
    """
    receipt = {
        "untrained_graded_baseline": {"spearman": 0.55, "emb_std": 0.02},
        "graded_held_out": {"spearman": 0.50, "emb_std": 0.02},
    }
    report = compress_mod.gate_report(receipt)
    assert report["passed"] is True
    assert report["beats_untrained"] is False
    assert report["verdict"] == "PASS (but does not beat untrained)"


@pytest.mark.cpu
def test_gate_fails_on_collapse_even_with_a_good_correlation() -> None:
    """Both conditions bind. emb_std is the collapse signal and a correlation measured on
    collapsed embeddings is decided by floating-point noise, not by the model."""
    receipt = {
        "untrained_graded_baseline": {"spearman": 0.10, "emb_std": 0.02},
        "graded_held_out": {"spearman": 0.61, "emb_std": 0.001},
    }
    report = compress_mod.gate_report(receipt)
    assert report["passed"] is False
    assert report["verdict"] == "FAIL"
    assert report["conditions"]["stsb_spearman"]["passed"] is True
    assert report["conditions"]["emb_std"]["passed"] is False


@pytest.mark.cpu
def test_gate_refuses_a_receipt_that_measured_no_spearman() -> None:
    """A run configured without a graded set cannot be judged against this gate. Reading
    a missing key as a failure would report 'FAIL' for a run that measured nothing."""
    with pytest.raises(KeyError, match="no graded_held_out"):
        compress_mod.gate_report({"held_out": {"recall@1": 0.9}})


@needs_corpus
def test_config_points_at_entailment_pairs_and_graded_stsb() -> None:
    """The training corpus must be the entailment-only `pair` config, not `pair-class`.

    InfoNCE needs a positive, and only entailment supplies one. Pointing this at
    pair-class would feed neutral and contradiction pairs in as positives -- sentences
    humans judged unrelated, taught as neighbours.
    """
    cfg = compress_mod.compress_config()
    assert cfg.pair_columns == ("anchor", "positive")
    assert cfg.shards[0].endswith("all-nli/pair/train-00000-of-00001.parquet")
    assert cfg.graded_columns == ("sentence1", "sentence2", "score")
    assert cfg.graded_shards[0].endswith("stsb/data/validation-00000-of-00001.parquet")


@needs_corpus
def test_stsb_scores_are_normalised_to_zero_one() -> None:
    """The dataset card says 0-5 divided by 5; the harness assumes nothing and checks.

    A 0-5 score would not change Spearman -- rank correlation is scale-free -- but it
    would silently break any later threshold or loss written against these values.
    """
    cfg = compress_mod.compress_config()
    graded = load_graded_pairs(cfg.graded_shards, cfg.graded_columns)
    assert len(graded) == 1500
    scores = [s for _, _, s in graded]
    assert min(scores) >= 0.0
    assert max(scores) <= 1.0
    assert max(scores) > 0.9, "a 0-1 range that never approaches 1 is a rescaling bug"


@needs_corpus
def test_graded_eval_has_the_ties_the_metric_must_handle() -> None:
    """Guards the assumption the Spearman implementation is built on.

    If a future corpus swap produced 1500 distinct scores, average-rank handling would
    stop mattering and the reasoning in _average_ranks would be stale rather than wrong.
    Better to be told.
    """
    cfg = compress_mod.compress_config()
    scores = [s for _, _, s in load_graded_pairs(cfg.graded_shards, cfg.graded_columns)]
    assert len(set(scores)) < len(scores) // 10, "expected heavy ties in graded similarity"


@needs_corpus
def test_short_run_produces_a_receipt_with_both_baselines(tmp_path) -> None:
    """End to end on the real corpus, tiny, so the plumbing is proven rather than assumed.

    Asserts the shape of the receipt, not the quality of the model: two steps train
    nothing. What must hold is that the untrained baseline was recorded BEFORE training
    and that the graded set was checked against the training pairs -- the two properties
    that make the eventual real number trustworthy.
    """
    receipt = pretrain_region(
        compress_mod.compress_config(
            steps=4,
            batch_size=16,
            holdout_pairs=16,
            eval_every=1,
            checkpoint_every=0,
            max_len=32,
            device="cpu",
            out_dir=str(tmp_path),
            encoder=TextEncoderConfig(dim=32, depth=1, n_heads=2),
        )
    )
    assert receipt["region"] == "compress"
    assert receipt["graded_corpus"]["name"] == "stsb-validation"
    assert receipt["graded_corpus"]["pairs_evaluated"] > 1000
    assert -1.0 <= receipt["untrained_graded_baseline"]["spearman"] <= 1.0
    assert -1.0 <= receipt["graded_held_out"]["spearman"] <= 1.0
    assert "spearman" in receipt["beats_untrained"]
    # The gate must be evaluable from the receipt alone; that is what makes program.json
    # fillable from a measurement rather than from someone's memory of a run.
    assert compress_mod.gate_report(receipt)["gate"] == "P1/compress"
