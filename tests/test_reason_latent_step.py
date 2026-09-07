"""CPU tests for E5 (latent-step prediction, diagnosis §4 E5).

Two layers, matching the operator's instructions:

1. Unit tests on `cogsyndelta.regions.reason_latent_step` -- step splitting, windowing
   (including the sequence-blind control), step corruption, the battery, and the
   `LatentStepModel`'s shapes/masking/gradient-flow/EMA update -- all on tiny synthetic
   derivations and a tiny trained tokenizer. No dependency on the real fleet corpus or
   the real GPT-2 tokenizer, exactly `tests/test_pretrain_resume.py`'s own reasoning for
   why its fixtures are synthetic.

2. An end-to-end `--dry-run` of `scripts/csd-train-reason-e5.py`'s `main()`, imported by
   path (hyphenated script names are not importable as modules -- same pattern
   `tests/test_csd_train_all_memory_dispatch.py` uses). `build_prereg_config` is
   monkeypatched to a synthetic corpus (mirroring `tiny_cfg`'s reasoning in
   `test_pretrain_resume.py`) with REAL split/order manifests pre-written via
   `write_split_and_order_manifests` to a scratch `splits_dir` -- G26 is exercised for
   real, against a corpus this test controls, never against `config/mind/splits/` (a
   different lane owns that directory; nothing here writes to it). The dry run must
   still produce a real `csd-metrics/v2` receipt with `split.sha256` populated and a
   pre-registration file written before it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

from cogsyndelta.eval.corrupted_derivation import w2c_region_seed as e1_w2c_region_seed
from cogsyndelta.regions.pretrain import PretrainConfig, write_split_and_order_manifests
from cogsyndelta.regions.reason_latent_step import (
    CHANCE,
    GO_MARGIN_OVER_BLIND,
    GO_RECALL_FLOOR,
    KILL_MARGIN_OVER_BLIND,
    MIN_HOLDOUT_STEPS,
    MIN_STEPS,
    LatentStepConfig,
    LatentStepModel,
    StepExample,
    build_step_battery,
    build_step_examples,
    collapse_stats,
    corrupt_step,
    enumerate_step_windows,
    go_kill_note,
    score_step_battery,
    split_derivation_steps,
    w2c_region_seed,
)
from cogsyndelta.regions.text_encoder import TextEncoderConfig

pytestmark = pytest.mark.cpu

NATALIA = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n"
    "#### 72"
)

# ---------------------------------------------------------------------------------------
# split_derivation_steps
# ---------------------------------------------------------------------------------------


def test_split_derivation_steps_keeps_the_final_hash_line_as_a_real_step() -> None:
    steps = split_derivation_steps(NATALIA)
    assert steps == [
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.",
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.",
        "#### 72",
    ]


def test_split_derivation_steps_drops_blank_lines() -> None:
    text = "step one\n\n\nstep two\n"
    assert split_derivation_steps(text) == ["step one", "step two"]


def test_split_derivation_steps_empty_string_is_no_steps() -> None:
    assert split_derivation_steps("") == []


# ---------------------------------------------------------------------------------------
# build_step_examples
# ---------------------------------------------------------------------------------------


def test_build_step_examples_keeps_only_gsm8k_shaped_pairs_with_enough_steps() -> None:
    pairs = [
        ("q1", NATALIA),  # gsm8k-shaped, 3 steps
        ("q2", "a rationale with no hash marker at all"),  # not gsm8k-shaped
        ("q3", "#### 5"),  # gsm8k-shaped, only 1 step
    ]
    examples = build_step_examples(pairs, min_steps=MIN_STEPS)
    assert [ex.question for ex in examples] == ["q1"]
    assert examples[0].steps == tuple(split_derivation_steps(NATALIA))


def test_build_step_examples_min_holdout_steps_floor_excludes_short_derivations() -> None:
    two_step = "line one\n#### 3"
    examples = build_step_examples([("q", two_step)], min_steps=MIN_HOLDOUT_STEPS)
    assert examples == []
    examples2 = build_step_examples([("q", two_step)], min_steps=MIN_STEPS)
    assert len(examples2) == 1


# ---------------------------------------------------------------------------------------
# enumerate_step_windows
# ---------------------------------------------------------------------------------------


def _natalia_example() -> StepExample:
    return build_step_examples([("What did Natalia sell?", NATALIA)], min_steps=MIN_STEPS)[0]


def test_enumerate_step_windows_one_per_prefix_length() -> None:
    ex = _natalia_example()
    windows = enumerate_step_windows([ex])
    assert [w.t for w in windows] == [1, 2]
    assert windows[0].target == ex.steps[1]
    assert windows[1].target == ex.steps[2]
    # Non-blind context carries the question AND the prior steps.
    assert ex.question in windows[0].context
    assert ex.steps[0] in windows[0].context
    assert ex.steps[1] not in windows[0].context  # steps[:1], not steps[:2]


def test_enumerate_step_windows_blind_context_never_contains_step_text() -> None:
    ex = _natalia_example()
    windows = enumerate_step_windows([ex], blind=True)
    assert len(windows) == 2
    for w in windows:
        assert w.context == ex.question
        for step in ex.steps:
            assert step not in w.context
    # The TARGET side is identical between arms -- only context differs.
    plain = enumerate_step_windows([ex], blind=False)
    assert [w.target for w in windows] == [w.target for w in plain]


def test_enumerate_step_windows_single_step_example_yields_nothing() -> None:
    ex = StepExample(item_id="x", question="q", steps=("only step",))
    assert enumerate_step_windows([ex]) == []


# ---------------------------------------------------------------------------------------
# corrupt_step
# ---------------------------------------------------------------------------------------


def test_corrupt_step_edits_the_annotation_and_stays_deterministic() -> None:
    step = "Natalia sold 48+24 = <<48+24=72>>72 clips altogether."
    a = corrupt_step(step, item_id="id1", t=1, corruption_seed=0)
    b = corrupt_step(step, item_id="id1", t=1, corruption_seed=0)
    assert a is not None
    assert a != step
    assert a == b  # deterministic given the same (seed, item_id, t)
    # A different t for the same item must be free to differ (independent RNG stream).
    c = corrupt_step(step, item_id="id1", t=2, corruption_seed=0)
    assert c is not None


def test_corrupt_step_returns_none_without_a_calculator_annotation() -> None:
    assert corrupt_step("#### 72", item_id="id1", t=2, corruption_seed=0) is None
    assert corrupt_step("a plain narrative line", item_id="id1", t=1, corruption_seed=0) is None


# ---------------------------------------------------------------------------------------
# build_step_battery
# ---------------------------------------------------------------------------------------


def _battery_examples() -> list[StepExample]:
    pairs = [
        (
            f"question {i}",
            f"line a {i} = <<{i}+{i}={2 * i}>>{2 * i}.\nline b {i} = <<{i}*2={2 * i}>>{2 * i}.\n#### {2 * i}",
        )
        for i in range(6)
    ]
    return build_step_examples(pairs, min_steps=MIN_HOLDOUT_STEPS)


def test_build_step_battery_shapes_and_no_self_leakage() -> None:
    examples = _battery_examples()
    assert len(examples) == 6
    items = build_step_battery(examples, corruption_seed=0)
    assert len(items) > 0
    for item in items:
        assert item.corrupted != item.target
        assert item.target not in item.others
        assert len(set(item.others)) == 3  # sampled without replacement
        # The distractors must not be this item's own derivation's steps.
        owner = next(ex for ex in examples if ex.item_id == item.item_id)
        assert not any(other in owner.steps for other in item.others)


def test_build_step_battery_is_deterministic() -> None:
    examples = _battery_examples()
    a = build_step_battery(examples, corruption_seed=0)
    b = build_step_battery(examples, corruption_seed=0)
    assert [(i.item_id, i.t, i.corrupted, i.others) for i in a] == [
        (i.item_id, i.t, i.corrupted, i.others) for i in b
    ]


# ---------------------------------------------------------------------------------------
# w2c_region_seed parity with the E1 module's own copy
# ---------------------------------------------------------------------------------------


def test_w2c_region_seed_matches_the_e1_modules_copy() -> None:
    assert w2c_region_seed("reason") == e1_w2c_region_seed("reason")


# ---------------------------------------------------------------------------------------
# LatentStepModel: shapes, masking, gradient flow, EMA update
# ---------------------------------------------------------------------------------------


def _tiny_tokenizer(tmp_path: Path) -> Path:
    texts = [
        "question zero plus zero equals zero",
        "line a 0 = << 0 + 0 = 0 >> 0",
        "line b 0 = << 0 * 2 = 0 >> 0",
        "#### 0",
        "question one plus one equals two",
        "line a 1 = << 1 + 1 = 2 >> 2",
        "#### 2",
    ]
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tok.train_from_iterator(texts, trainer=trainer)
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    return path


@pytest.fixture
def tiny_model_and_tok(tmp_path: Path) -> tuple[LatentStepModel, Tokenizer]:
    tok = Tokenizer.from_file(str(_tiny_tokenizer(tmp_path)))
    encoder_cfg = TextEncoderConfig(
        dim=8, depth=1, n_heads=2, max_len=32, vocab_size=tok.get_vocab_size()
    )
    predictor_cfg = LatentStepConfig(dim=8, hidden=4, depth=1)
    torch.manual_seed(0)
    model = LatentStepModel(encoder_cfg, predictor_cfg)
    return model, tok


def _tokenize(
    tok: Tokenizer, texts: list[str], max_len: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [tok.encode(t).ids[:max_len] for t in texts]
    width = max(1, *(len(e) for e in encoded))
    ids = torch.zeros(len(encoded), width, dtype=torch.long)
    mask = torch.zeros(len(encoded), width, dtype=torch.long)
    for i, seq in enumerate(encoded):
        if seq:
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, : len(seq)] = 1
    return ids, mask


def test_latent_step_model_forward_shapes_and_stats(tiny_model_and_tok) -> None:
    model, tok = tiny_model_and_tok
    ctx_ids, ctx_mask = _tokenize(
        tok, ["question zero plus zero equals zero", "question one plus one equals two"]
    )
    tgt_ids, tgt_mask = _tokenize(
        tok, ["line a 0 = << 0 + 0 = 0 >> 0", "line a 1 = << 1 + 1 = 2 >> 2"]
    )
    loss, stats = model(ctx_ids, ctx_mask, tgt_ids, tgt_mask)
    assert loss.dim() == 0
    assert loss.item() >= 0.0
    assert {"loss", "target_emb_std", "predicted_emb_std"} <= stats.keys()
    predicted = model.predict(ctx_ids, ctx_mask)
    assert predicted.shape == (2, 8)


def test_padding_is_masked_out_of_the_context_pool(tiny_model_and_tok) -> None:
    """Two paddings of the same short text must encode identically (module docstring's
    "padding is masked out of both attention and the pool" -- the same property
    `TextEncoder`'s own docstring asserts, inherited here since the trunk IS TextEncoder.
    """
    model, tok = tiny_model_and_tok
    model.eval()
    short_ids, short_mask = _tokenize(tok, ["question zero plus zero equals zero"], max_len=32)
    padded_ids = torch.nn.functional.pad(short_ids, (0, 10))
    padded_mask = torch.nn.functional.pad(short_mask, (0, 10))
    with torch.no_grad():
        a = model.encode_context(short_ids, short_mask)
        b = model.encode_context(padded_ids, padded_mask)
    assert torch.allclose(a, b, atol=1e-5)


def test_backward_updates_encoder_and_predictor_but_never_the_target(tiny_model_and_tok) -> None:
    model, tok = tiny_model_and_tok
    before_target = [p.clone() for p in model.target_encoder.parameters()]
    ctx_ids, ctx_mask = _tokenize(
        tok, ["question zero plus zero equals zero", "question one plus one equals two"]
    )
    tgt_ids, tgt_mask = _tokenize(
        tok, ["line a 0 = << 0 + 0 = 0 >> 0", "line a 1 = << 1 + 1 = 2 >> 2"]
    )
    loss, _ = model(ctx_ids, ctx_mask, tgt_ids, tgt_mask)
    loss.backward()

    assert all(p.grad is not None for p in model.encoder.parameters())
    assert all(p.grad is not None for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())
    # No optimizer step ran -- the target's VALUES (not just its grad) must be untouched.
    for before, after in zip(before_target, model.target_encoder.parameters(), strict=True):
        assert torch.equal(before, after)


def test_update_target_is_the_stated_ema_formula(tiny_model_and_tok) -> None:
    model, _ = tiny_model_and_tok
    before_ctx = [p.clone() for p in model.encoder.parameters()]
    before_tgt = [p.clone() for p in model.target_encoder.parameters()]
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)  # move the context encoder somewhere the target must chase
    momentum = 0.9
    model.update_target(momentum)
    for ctx_before, tgt_before, tgt_after in zip(
        before_ctx, before_tgt, model.target_encoder.parameters(), strict=True
    ):
        expected = momentum * tgt_before + (1.0 - momentum) * (ctx_before + 1.0)
        assert torch.allclose(tgt_after, expected, atol=1e-6)
    # And the target must still carry no gradient after an EMA update.
    assert all(not p.requires_grad for p in model.target_encoder.parameters())


def test_score_step_battery_is_within_unit_interval_and_deterministic(tiny_model_and_tok) -> None:
    model, tok = tiny_model_and_tok
    examples = build_step_examples(
        [
            (
                "question zero plus zero equals zero",
                "line a 0 = << 0 + 0 = 0 >> 0\nline b 0 = << 0 * 2 = 0 >> 0\n#### 0",
            ),
            (
                "question one plus one equals two",
                "line a 1 = << 1 + 1 = 2 >> 2\nline b 1 = << 1 * 2 = 2 >> 2\n#### 2",
            ),
            (
                "question two plus two",
                "line a 2 = << 2 + 2 = 4 >> 4\nline b 2 = << 2 * 2 = 4 >> 4\n#### 4",
            ),
        ],
        min_steps=MIN_HOLDOUT_STEPS,
    )
    battery = build_step_battery(examples, corruption_seed=0)
    assert battery, "fixture must produce at least one scoreable item"

    def tokenize(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        return _tokenize(tok, texts)

    a = score_step_battery(model, tokenize, battery, batch=4, tie_seed=0)
    b = score_step_battery(model, tokenize, battery, batch=4, tie_seed=0)
    assert a == b  # deterministic given the same weights and tie_seed
    assert 0.0 <= a["recall@1"] <= 1.0
    assert a["chance"] == CHANCE
    assert a["n_items"] == float(len(battery))


def test_collapse_stats_reports_zero_std_for_a_constant_embedding() -> None:
    constant = torch.ones(5, 4)
    stats = collapse_stats(constant)
    assert stats["emb_std"] == pytest.approx(0.0, abs=1e-6)
    assert stats["n"] == 5.0
    assert stats["dim"] == 4.0


# ---------------------------------------------------------------------------------------
# go_kill_note
# ---------------------------------------------------------------------------------------


def test_go_kill_note_pending_without_a_paired_blind_score() -> None:
    result = go_kill_note(0.5, None)
    assert result["verdict"] == "pending"


def test_go_kill_note_go_requires_both_the_floor_and_the_margin() -> None:
    go = go_kill_note(GO_RECALL_FLOOR + 0.02, GO_RECALL_FLOOR + 0.02 - GO_MARGIN_OVER_BLIND - 0.01)
    assert go["verdict"] == "go"
    # Floor met but margin not met -> not a go.
    not_go = go_kill_note(
        GO_RECALL_FLOOR + 0.01, GO_RECALL_FLOOR + 0.01 - GO_MARGIN_OVER_BLIND + 0.05
    )
    assert not_go["verdict"] != "go"


def test_go_kill_note_kill_on_a_small_margin_even_below_the_floor() -> None:
    kill = go_kill_note(0.10, 0.10 - KILL_MARGIN_OVER_BLIND)
    assert kill["verdict"] == "kill"


def test_go_kill_note_none_in_the_undecided_band() -> None:
    result = go_kill_note(0.20, 0.12)  # margin 0.08: > kill (0.05), < go (0.10)
    assert result["verdict"] == "none"


# =========================================================================================
# End-to-end: scripts/csd-train-reason-e5.py --dry-run, against a synthetic corpus.
# =========================================================================================

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "csd-train-reason-e5.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("csd_train_reason_e5", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_synthetic_gsm8k_shard(path: Path, n: int) -> None:
    questions = [f"synthetic question number {i}" for i in range(n)]
    answers = [
        f"line a {i} = <<{i}+{i}={2 * i}>>{2 * i}.\nline b {i} = <<{i}*2={2 * i}>>{2 * i}.\n#### {2 * i}"
        for i in range(n)
    ]
    pq.write_table(pa.table({"question": questions, "answer": answers}), path)


@pytest.fixture
def dry_run_env(tmp_path: Path):
    """A synthetic corpus, a tiny tokenizer, and real (G26) split/order manifests for
    it -- everything `main()` needs, without touching the real fleet corpus or the
    real `config/mind/splits/`.
    """
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    shard_path = corpus_dir / "train.parquet"
    _write_synthetic_gsm8k_shard(shard_path, 40)

    tokenizer_path = _tiny_tokenizer_full(tmp_path)

    steps, batch_size, holdout = 5, 8, 6
    draft_cfg = PretrainConfig(
        region="reason",
        pair_columns=("question", "answer"),
        shards=[str(shard_path)],
        steps=steps,
        batch_size=batch_size,
        max_len=64,
        holdout_pairs=holdout,
        seed=0,
        split_seed=0,
        order_seed=0,
        require_split_manifest=False,
        tokenizer_path=str(tokenizer_path),
        encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=64),
        out_dir=str(tmp_path / "receipts"),
    )
    splits_dir = tmp_path / "splits"
    split_path, order_path, meta = write_split_and_order_manifests(draft_cfg, splits_dir=splits_dir)
    n_train_pairs = json.loads(order_path.read_text())["n_train"]

    real_cfg = PretrainConfig(
        region="reason",
        pair_columns=("question", "answer"),
        shards=[str(shard_path)],
        steps=steps,
        batch_size=batch_size,
        max_len=64,
        holdout_pairs=holdout,
        seed=0,
        split_seed=0,
        order_seed=0,
        require_split_manifest=True,
        split_manifest=str(split_path),
        order_manifest=str(order_path),
        tokenizer_path=str(tokenizer_path),
        encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=64),
        out_dir=str(tmp_path / "receipts"),
    )
    return {
        "cfg": real_cfg,
        "corpus_fingerprint": meta["corpus_fingerprint"],
        "train_pairs": int(n_train_pairs),
        "duplicates_removed": int(meta["duplicates_removed"]),
        "steps": steps,
        "batch_size": batch_size,
        "out_dir": tmp_path / "receipts",
    }


def _tiny_tokenizer_full(tmp_path: Path) -> Path:
    texts = [f"synthetic question number {i}" for i in range(40)]
    texts += [
        f"line a {i} = << {i} + {i} = {2 * i} >> {2 * i} . line b {i} = << {i} * 2 = {2 * i} >> {2 * i} . #### {2 * i}"
        for i in range(40)
    ]
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tok.train_from_iterator(texts, trainer=trainer)
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    return path


def test_dry_run_writes_a_prereg_file_before_training_and_a_v2_receipt(
    dry_run_env, monkeypatch, capsys
) -> None:
    module = _load_script_module()
    env = dry_run_env
    monkeypatch.setattr(
        module, "build_prereg_config", lambda *, steps, batch_size, seed: env["cfg"]
    )
    monkeypatch.setattr(module, "EXPECTED_FP", env["corpus_fingerprint"])
    monkeypatch.setattr(module, "EXPECTED_TRAIN_PAIRS", env["train_pairs"])
    monkeypatch.setattr(module, "EXPECTED_DUPES", env["duplicates_removed"])

    argv = [
        "csd-train-reason-e5.py",
        "--dry-run",
        "--arm",
        "latent-step",
        "--seed",
        "0",
        "--steps",
        str(env["steps"]),
        "--batch-size",
        str(env["batch_size"]),
        "--out-dir",
        str(env["out_dir"]),
        "--eval-batch",
        "4",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    rc = module.main()
    assert rc == 0

    out_dir = env["out_dir"]
    prereg_files = sorted(out_dir.glob("reason-e5-prereg-*.json"))
    receipt_files = sorted(out_dir.glob("reason-e5-latent-step-s0-*.json"))
    assert len(prereg_files) == 1
    assert len(receipt_files) == 1

    prereg = json.loads(prereg_files[0].read_text())
    assert prereg["schema"] == "csd-e5-prereg/v1"
    assert prereg["experiment"] == "E5"
    assert prereg["arm"] == "latent-step"
    assert "predictor acc@1" in prereg["go"]
    assert prereg["split"]["sha256"]
    assert prereg["notes_written_before_training"] is True

    receipt = json.loads(receipt_files[0].read_text())
    assert receipt["schema"] == "csd-reason-e5-receipt/v1"
    assert receipt["kind"] == "train"
    assert receipt["metrics_schema"] == "csd-metrics/v2"  # stamped by write_receipt
    assert receipt["dry_run"] is True
    assert receipt["split"]["sha256"] == prereg["split"]["sha256"]
    assert "code_revision" in receipt and receipt["code_revision"]["git_sha"]
    for key in (
        "predictor_battery",
        "untrained_predictor_battery",
        "e1_regression_guard",
        "diagonal_recall_reference_only",
        "collapse",
        "go_kill_reference",
    ):
        assert key in receipt, f"missing receipt field: {key}"
    assert receipt["predictor_battery"]["chance"] == CHANCE
    # The prereg's pre-training battery score must equal the receipt's own record of it
    # -- proof the same measurement was not silently redone (or drifted) between the two
    # writes.
    assert (
        prereg["untrained_predictor_battery"]["recall@1"]
        == receipt["untrained_predictor_battery"]["recall@1"]
    )

    ckpt_path = Path(receipt["checkpoint"])
    assert ckpt_path.is_file()
    from cogsyndelta.regions._checkpoint import sha256_file

    assert sha256_file(ckpt_path) == receipt["checkpoint_sha256"]

    out = capsys.readouterr().out
    assert "pre-registration written BEFORE training" in out


def test_dry_run_sequence_blind_arm_is_flagged_in_the_receipt(dry_run_env, monkeypatch) -> None:
    module = _load_script_module()
    env = dry_run_env
    monkeypatch.setattr(
        module, "build_prereg_config", lambda *, steps, batch_size, seed: env["cfg"]
    )
    monkeypatch.setattr(module, "EXPECTED_FP", env["corpus_fingerprint"])
    monkeypatch.setattr(module, "EXPECTED_TRAIN_PAIRS", env["train_pairs"])
    monkeypatch.setattr(module, "EXPECTED_DUPES", env["duplicates_removed"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "csd-train-reason-e5.py",
            "--dry-run",
            "--arm",
            "sequence-blind",
            "--seed",
            "1",
            "--steps",
            str(env["steps"]),
            "--batch-size",
            str(env["batch_size"]),
            "--out-dir",
            str(env["out_dir"]),
            "--eval-batch",
            "4",
        ],
    )
    assert module.main() == 0
    receipt_files = sorted(env["out_dir"].glob("reason-e5-sequence-blind-s1-*.json"))
    assert len(receipt_files) == 1
    receipt = json.loads(receipt_files[0].read_text())
    assert receipt["arm"] == "sequence-blind"
    assert receipt["blind"] is True
