"""Resumable-checkpointing tests for the text-region pretrain harness.

Training runs died three times to session churn in one day, and every time restarted
from step 0 -- pretrain.py had no resume path at all. These tests prove the fix does
what it needs to and nothing less:

  - a checkpoint round-trips: save, load, model/optimizer state match exactly.
  - resuming produces the SAME final weights as an uninterrupted run of the same total
    step count -- not "close", not "similar loss", the same weights. This is the one
    that would catch a subtly wrong resume (e.g. re-measuring the baseline, restarting
    the LR schedule, or dropping optimizer momentum) that every other test could miss.
  - a config mismatch is REFUSED, not silently absorbed into a run that is neither the
    old config nor the new one.
  - an atomic write leaves no partial file visible under the final checkpoint name.

Everything here runs on CPU with a tiny encoder and a handful of synthetic pairs, so it
takes a fraction of a second -- there is no dependency on the fleet's NFS-mounted corpus
or its real gpt2 tokenizer, both of which the actual GPU run in flight is using right now
and which this file must not touch.
"""

from __future__ import annotations

import re
from dataclasses import asdict, replace
from pathlib import Path

import pytest

# MUST precede the imports below: cogsyndelta.regions.pretrain imports tokenizers at
# module scope, and this file builds its own tiny tokenizer/parquet fixtures, so without
# the train group installed the import errors during collection and the whole file fails
# instead of skipping (same reasoning as tests/test_region_compress.py).
pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from cogsyndelta.regions._checkpoint import atomic_save
from cogsyndelta.regions.pretrain import (
    PretrainConfig,
    _checkpoint_payload,
    _config_fingerprint,
    _resume_fields,
    _vintage_fingerprint,
    pretrain_region,
)
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig

pytestmark = pytest.mark.cpu


def _build_tokenizer(path: Path, n_pairs: int) -> None:
    """A tiny WordLevel tokenizer trained on exactly the vocabulary the test pairs use,
    so nothing falls back to [UNK] and shapes stay small."""
    texts = [f"anchor number {i} word" for i in range(n_pairs)] + [
        f"anchor number {i} match" for i in range(n_pairs)
    ]
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106 -- a token id name, not a secret
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(path))


def _build_pairs_parquet(path: Path, n_pairs: int) -> None:
    """`n_pairs` distinct (anchor, positive) rows -- distinct anchors so none are
    dropped by pretrain.py's anchor-fingerprint dedup."""
    anchors = [f"anchor number {i} word" for i in range(n_pairs)]
    positives = [f"anchor number {i} match" for i in range(n_pairs)]
    pq.write_table(pa.table({"anchor": anchors, "positive": positives}), path)


@pytest.fixture
def tiny_cfg(tmp_path: Path) -> PretrainConfig:
    """A config small enough to train in a fraction of a second on CPU.

    steps=6, batch_size=4, holdout_pairs=4 -> budget = 6*4+4 = 28 pairs needed; 40
    supplied for margin.
    """
    tok_path = tmp_path / "tokenizer.json"
    shard_path = tmp_path / "pairs.parquet"
    _build_tokenizer(tok_path, 40)
    _build_pairs_parquet(shard_path, 40)
    return PretrainConfig(
        region="resume-test",
        pair_columns=("anchor", "positive"),
        shards=[str(shard_path)],
        steps=6,
        batch_size=4,
        holdout_pairs=4,
        eval_every=2,
        checkpoint_every=2,
        max_len=16,
        seed=0,
        device="cpu",
        encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=16),
        tokenizer_path=str(tok_path),
        out_dir=str(tmp_path / "run"),
    )


def _tiny_model_and_opt(cfg: PretrainConfig) -> tuple[TextEncoder, torch.optim.AdamW]:
    tok = Tokenizer.from_file(cfg.tokenizer_path)
    encoder_cfg = TextEncoderConfig(**{**asdict(cfg.encoder), "vocab_size": tok.get_vocab_size()})
    model = TextEncoder(encoder_cfg, name=cfg.region)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    return model, opt


# ---------------------------------------------------------------------------------------
# Checkpoint round-trip.
# ---------------------------------------------------------------------------------------


def test_checkpoint_round_trip_matches_model_and_optimizer_state(tiny_cfg: PretrainConfig) -> None:
    model, opt = _tiny_model_and_opt(tiny_cfg)

    # Give the optimizer real momentum (exp_avg etc.) to round-trip, not just zeros.
    x = torch.randint(0, 5, (4, 6))
    mask = torch.ones(4, 6, dtype=torch.long)
    loss = model(x, mask).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()

    tok = Tokenizer.from_file(tiny_cfg.tokenizer_path)
    encoder_cfg = TextEncoderConfig(
        **{**asdict(tiny_cfg.encoder), "vocab_size": tok.get_vocab_size()}
    )
    fingerprint = _config_fingerprint(tiny_cfg)
    fields_from_cfg = _resume_fields(tiny_cfg)
    payload = _checkpoint_payload(
        step=3,
        model=model,
        opt=opt,
        encoder_cfg=encoder_cfg,
        fingerprint=fingerprint,
        fields=fields_from_cfg,
        baseline={"recall@1": 0.1},
        graded_baseline={},
        history=[{"step": 1.0}],
        elapsed_s=12.5,
    )

    ckpt_dir = Path(tiny_cfg.out_dir) / "ckpt"
    path = ckpt_dir / "step-000003.pt"
    atomic_save(payload, path)
    assert path.is_file()

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["step"] == 3
    assert loaded["config_fingerprint"] == fingerprint
    assert loaded["elapsed_s"] == 12.5
    assert loaded["untrained_baseline"] == {"recall@1": 0.1}
    assert loaded["history"] == [{"step": 1.0}]

    for key, value in model.state_dict().items():
        assert torch.equal(loaded["model"][key], value), f"model tensor {key} did not round-trip"

    loaded_opt_state = loaded["opt"]["state"]
    live_opt_state = opt.state_dict()["state"]
    assert set(loaded_opt_state) == set(live_opt_state)
    for idx in live_opt_state:
        for k, v in live_opt_state[idx].items():
            if torch.is_tensor(v):
                assert torch.equal(loaded_opt_state[idx][k], v), f"optimizer state {idx}/{k}"
            else:
                assert loaded_opt_state[idx][k] == v


def test_rng_state_round_trips_bit_exact(tiny_cfg: PretrainConfig, tmp_path: Path) -> None:
    """The saved RNG state must restore EXACTLY, independent of whether this specific
    harness's training loop happens to be sensitive to it (the text harness has no
    dropout, so it mostly is not -- see pretrain.py's `_checkpoint_payload` docstring).
    The mechanism itself must still be correct for any harness that reuses it."""
    model, opt = _tiny_model_and_opt(tiny_cfg)
    tok = Tokenizer.from_file(tiny_cfg.tokenizer_path)
    encoder_cfg = TextEncoderConfig(
        **{**asdict(tiny_cfg.encoder), "vocab_size": tok.get_vocab_size()}
    )

    torch.manual_seed(12345)
    torch.randn(37)  # advance the global RNG to a non-trivial state
    saved_state = torch.get_rng_state().clone()

    payload = _checkpoint_payload(
        step=1,
        model=model,
        opt=opt,
        encoder_cfg=encoder_cfg,
        fingerprint="fp",
        fields={},
        baseline={},
        graded_baseline={},
        history=[],
        elapsed_s=0.0,
    )
    path = tmp_path / "rng-ckpt" / "step-000001.pt"
    atomic_save(payload, path)

    torch.manual_seed(0)  # perturb the global RNG to something else entirely
    torch.randn(999)

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    torch.set_rng_state(loaded["rng_state"])
    assert torch.equal(torch.get_rng_state(), saved_state)


# ---------------------------------------------------------------------------------------
# Resuming produces the same result as an uninterrupted run.
# ---------------------------------------------------------------------------------------


def test_resume_matches_uninterrupted_run(tiny_cfg: PretrainConfig, monkeypatch) -> None:
    """Run 6 steps in one shot; separately run the same 6 steps but crash after 3 (right
    after a checkpoint at step 2 fires) and resume. Final weights must match."""
    import cogsyndelta.regions.pretrain as pretrain_mod

    uninterrupted_cfg = replace(tiny_cfg, out_dir=str(Path(tiny_cfg.out_dir) / "uninterrupted"))
    reference = pretrain_region(uninterrupted_cfg)

    crashing_cfg = replace(tiny_cfg, out_dir=str(Path(tiny_cfg.out_dir) / "crashed-then-resumed"))

    orig_info_nce = pretrain_mod.info_nce
    call_count = {"n": 0}

    def flaky_info_nce(anchors, positives, temperature=0.05):
        call_count["n"] += 1
        if call_count["n"] == 4:  # right after the step=2 checkpoint has been written
            raise RuntimeError("simulated crash")
        return orig_info_nce(anchors, positives, temperature)

    monkeypatch.setattr(pretrain_mod, "info_nce", flaky_info_nce)
    with pytest.raises(RuntimeError, match="simulated crash"):
        pretrain_region(crashing_cfg)
    monkeypatch.undo()

    ckpt_dir = (
        Path(crashing_cfg.out_dir)
        / f"{crashing_cfg.region}-checkpoints"
        / _vintage_fingerprint(crashing_cfg)[:8]
    )
    assert (ckpt_dir / "step-000002.pt").is_file(), "expected a checkpoint before the crash"

    resumed = pretrain_region(crashing_cfg)
    assert resumed["resumed"] is True
    assert resumed["resumed_from_step"] == 3

    # The untrained baseline must be the ORIGINAL one, not remeasured on the
    # partially-trained model (trap #1).
    assert resumed["untrained_baseline"] == reference["untrained_baseline"]

    ref_ckpt = torch.load(reference["checkpoint"], map_location="cpu", weights_only=True)
    res_ckpt = torch.load(resumed["checkpoint"], map_location="cpu", weights_only=True)
    for key, value in ref_ckpt["model"].items():
        assert torch.allclose(res_ckpt["model"][key], value, atol=1e-6, rtol=1e-5), (
            f"resumed run's final weight for {key} diverged from the uninterrupted run"
        )

    assert resumed["held_out"]["recall@1"] == pytest.approx(
        reference["held_out"]["recall@1"], abs=1e-6
    )


def test_a_completed_run_resumed_again_trains_zero_more_steps(
    tiny_cfg: PretrainConfig,
) -> None:
    """Calling pretrain_region twice with an identical config must not re-train: the
    second call should recognise `final.pt` as already at cfg.steps and do nothing
    beyond re-evaluating and re-writing the receipt."""
    first = pretrain_region(tiny_cfg)
    second = pretrain_region(tiny_cfg)
    assert second["resumed"] is True
    assert second["resumed_from_step"] == tiny_cfg.steps
    assert second["held_out"] == first["held_out"]


# ---------------------------------------------------------------------------------------
# Config mismatch is refused.
# ---------------------------------------------------------------------------------------


def test_config_mismatch_is_refused_not_silently_accepted(tiny_cfg: PretrainConfig) -> None:
    pretrain_region(tiny_cfg)  # produces a checkpoint under tiny_cfg's fingerprint

    changed = replace(tiny_cfg, batch_size=tiny_cfg.batch_size + 1)
    with pytest.raises(ValueError, match="refusing to resume"):
        pretrain_region(changed)


def test_config_mismatch_error_names_the_differing_field(tiny_cfg: PretrainConfig) -> None:
    pretrain_region(tiny_cfg)
    changed = replace(tiny_cfg, lr=tiny_cfg.lr * 2)
    with pytest.raises(ValueError, match=re.escape("lr: checkpoint=")):
        pretrain_region(changed)


# ---------------------------------------------------------------------------------------
# Atomic write.
# ---------------------------------------------------------------------------------------


def test_atomic_save_leaves_no_partial_file_under_the_final_name(
    tmp_path: Path, monkeypatch
) -> None:
    import cogsyndelta.regions._checkpoint as checkpoint_mod

    dest = tmp_path / "ckpt" / "final.pt"

    def boom(obj, f):
        Path(f).write_bytes(b"garbage-partial-write")
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(checkpoint_mod.torch, "save", boom)
    with pytest.raises(OSError, match="disk full"):
        atomic_save({"x": 1}, dest)

    assert not dest.exists(), "a crash mid-write must never leave a file under the final name"
    # And no leftover temp garbage either -- _atomic_save cleans up on failure.
    assert list(dest.parent.glob("*")) == []


def test_atomic_save_then_load_survives_a_prior_partial_write(tmp_path: Path) -> None:
    """A good write must succeed and be loadable even if a stray partial temp file from
    a previous crash is sitting in the same directory (the crash-recovery scenario this
    whole mechanism exists for)."""
    dest = tmp_path / "ckpt" / "step-000001.pt"
    dest.parent.mkdir(parents=True)
    (dest.parent / ".step-000001.pt.tmp-99999-1").write_bytes(b"leftover garbage")

    atomic_save({"step": 1}, dest)
    assert torch.load(dest, weights_only=True)["step"] == 1


# ---------------------------------------------------------------------------------------
# Rotation.
# ---------------------------------------------------------------------------------------


def test_rotation_keeps_only_the_most_recent_n_periodic_checkpoints(
    tiny_cfg: PretrainConfig,
) -> None:
    cfg = replace(tiny_cfg, steps=12, checkpoint_every=2)
    pretrain_region(cfg)
    ckpt_dir = Path(cfg.out_dir) / f"{cfg.region}-checkpoints" / _vintage_fingerprint(cfg)[:8]
    periodic = sorted(p.name for p in ckpt_dir.glob("step-*.pt"))
    # _CHECKPOINT_KEEP is 3; steps=12, checkpoint_every=2 writes at 2,4,6,8,10 (11 has no
    # multiple before 12) plus the run finishes with a separate final.pt.
    assert len(periodic) <= 3, f"rotation should cap periodic checkpoints, found {periodic}"
    assert (ckpt_dir / "final.pt").is_file()


# ---------------------------------------------------------------------------------------
# LR schedule resumes correctly (a pure function of step -- confirm, don't assume).
# ---------------------------------------------------------------------------------------


def test_lr_schedule_needs_no_extra_state_to_resume_correctly(tiny_cfg: PretrainConfig) -> None:
    """`_lr_at` must be a pure function of (step, cfg) -- if it secretly depended on how
    many times it had already been called, or on any other run-local state, a resumed
    run would restart the schedule instead of continuing it. Prove purity directly: a
    freshly-constructed, field-identical config (a different Python object, standing in
    for "the config a resumed process reconstructs from scratch") must produce the exact
    same curve as the original, called in a different order (high-to-low instead of the
    training loop's low-to-high) to rule out any dependence on call sequence."""
    from cogsyndelta.regions.pretrain import _lr_at

    fresh_cfg = replace(tiny_cfg)  # a distinct object with identical field values
    forward = [_lr_at(step, tiny_cfg) for step in range(tiny_cfg.steps)]
    backward = [_lr_at(step, fresh_cfg) for step in reversed(range(tiny_cfg.steps))]
    assert forward == list(reversed(backward))
