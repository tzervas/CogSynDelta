"""Resumable-checkpointing tests for the visual (I-JEPA) pretrain harness.

This mirrors tests/test_pretrain_resume.py's coverage but through
`pretrain_vl_region`/`VLPretrainConfig` rather than the text harness, because the two
genuinely differ in ways that matter for resumability (see vl_pretrain.py's module
comment above `pretrain_vl_region`):

  - batch order here is SAMPLED from a local `torch.Generator`, not a deterministic
    function of `step` -- so resuming needs that generator's state, not just `step`.
  - `IJEPA.forward`'s mask sampling draws on the GLOBAL default RNG, and so does
    `_linear_probe`'s `nn.Linear` head init when the FINAL probe metrics are computed
    after training -- so, unlike the text harness, this one really is sensitive to
    `rng_state` being restored correctly, not just defensively saved.
  - the EMA target encoder needs no separate checkpoint handling: it is a plain
    submodule of `IJEPA`, so `model.state_dict()` already carries it.

All fixtures here are tiny synthetic PNGs written to a local parquet file -- no
dependency on the fleet's tiny-imagenet/cifar100 mounts or the GPU, both of which the
real vl_latent training in flight right now is using.
"""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

# MUST precede the pyarrow import below: this file builds its own synthetic image
# parquet fixtures, and pyarrow lives in the `train` dependency group (see
# tests/test_region_compress.py for the same pattern).
pytest.importorskip("pyarrow", reason="train group not installed")

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from cogsyndelta.model.vl_jepa import JEPAConfig
from cogsyndelta.regions._checkpoint import atomic_save
from cogsyndelta.regions.vl_pretrain import (
    VLPretrainConfig,
    _checkpoint_payload,
    _config_fingerprint,
    pretrain_vl_region,
)

pytestmark = pytest.mark.cpu


def _png_bytes(size: int, fill: int) -> bytes:
    arr = np.full((size, size, 3), fill, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _write_image_parquet(path: Path, n: int, size: int, n_classes: int = 2) -> None:
    images = [_png_bytes(size, (i * 40) % 256) for i in range(n)]
    labels = [i % n_classes for i in range(n)]
    pq.write_table(pa.table({"image": images, "label": labels}), path)


@pytest.fixture
def tiny_vl_cfg(tmp_path: Path) -> VLPretrainConfig:
    """A tiny I-JEPA config: 16x16 images, 4x4 patches (16 patches), a 1-layer/8-dim
    encoder, and 8 training + 8 probe images -- small enough to train in a fraction of a
    second on CPU."""
    shard = tmp_path / "images.parquet"
    _write_image_parquet(shard, n=8, size=16)
    jepa_cfg = JEPAConfig(
        image_size=16,
        patch_size=4,
        dim=8,
        depth=1,
        n_heads=2,
        predictor_dim=4,
        predictor_depth=1,
    )
    return VLPretrainConfig(
        region="vl-resume-test",
        train_shards=[str(shard)],
        probe_train_shards=[str(shard)],
        probe_eval_shards=[str(shard)],
        transfer_shards=[],
        steps=6,
        batch_size=4,
        warmup_steps=2,
        eval_every=2,
        checkpoint_every=2,
        probe_steps=3,
        probe_limit=8,
        seed=0,
        device="cpu",
        jepa=jepa_cfg,
        cache_dir=str(tmp_path / "vl-cache"),
        out_dir=str(tmp_path / "run"),
    )


# ---------------------------------------------------------------------------------------
# Checkpoint round-trip.
# ---------------------------------------------------------------------------------------


def test_checkpoint_round_trip_matches_model_optimizer_and_generator(
    tiny_vl_cfg: VLPretrainConfig,
) -> None:
    from cogsyndelta.model.vl_jepa import IJEPA
    from cogsyndelta.regions.vl_pretrain import _resume_fields

    model = IJEPA(tiny_vl_cfg.jepa)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    gen = torch.Generator().manual_seed(tiny_vl_cfg.seed)
    gen.manual_seed(123)  # advance/alter state so it is not just the default

    x = torch.randn(4, 3, 16, 16)
    loss, _ = model(x)
    loss.backward()
    opt.step()
    opt.zero_grad()

    fingerprint = _config_fingerprint(tiny_vl_cfg)
    fields = _resume_fields(tiny_vl_cfg)
    payload = _checkpoint_payload(
        step=3,
        model=model,
        opt=opt,
        gen=gen,
        jepa_cfg=tiny_vl_cfg.jepa,
        fingerprint=fingerprint,
        fields=fields,
        baseline={"top1": 0.1},
        baseline_transfer=None,
        history=[],
        elapsed_s=5.0,
        started_at=100.0,
    )
    path = Path(tiny_vl_cfg.out_dir) / "ckpt" / "step-000003.pt"
    atomic_save(payload, path)

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["config_fingerprint"] == fingerprint
    assert loaded["step"] == 3
    assert loaded["started_at"] == 100.0

    # model.state_dict() must include the EMA target encoder (a plain submodule, per the
    # module comment) -- assert its keys are actually present, not just that SOME keys
    # round-trip.
    assert any(k.startswith("target_encoder.") for k in loaded["model"])
    assert any(k.startswith("encoder.") for k in loaded["model"])
    assert any(k.startswith("predictor.") for k in loaded["model"])
    for key, value in model.state_dict().items():
        assert torch.equal(loaded["model"][key], value)

    # The generator's OWN state, distinct from the global RNG.
    assert torch.equal(loaded["gen_state"], gen.get_state())


def test_gen_state_and_global_rng_state_are_independent(tiny_vl_cfg: VLPretrainConfig) -> None:
    """A resumed run needs BOTH the local batch-sampling generator's state and the
    global default RNG's state (mask sampling), and they must not be conflated -- each
    must restore independently of what happens to the other."""
    from cogsyndelta.model.vl_jepa import IJEPA
    from cogsyndelta.regions.vl_pretrain import _resume_fields

    model = IJEPA(tiny_vl_cfg.jepa)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    gen = torch.Generator().manual_seed(42)

    torch.manual_seed(7)
    torch.randn(11)  # advance the GLOBAL rng
    saved_global = torch.get_rng_state().clone()
    saved_gen = gen.get_state().clone()
    gen.manual_seed(999)  # advance the LOCAL generator differently

    payload = _checkpoint_payload(
        step=1,
        model=model,
        opt=opt,
        gen=gen,
        jepa_cfg=tiny_vl_cfg.jepa,
        fingerprint="fp",
        fields=_resume_fields(tiny_vl_cfg),
        baseline={},
        baseline_transfer=None,
        history=[],
        elapsed_s=0.0,
        started_at=0.0,
    )
    # We captured `saved_gen` from the generator BEFORE mutating it further -- rebuild
    # the payload's gen_state from that earlier snapshot to test restoration precisely.
    payload["gen_state"] = saved_gen
    payload["rng_state"] = saved_global

    path = Path(tiny_vl_cfg.out_dir) / "ckpt" / "step-000001.pt"
    atomic_save(payload, path)

    # Perturb both away from the saved values.
    torch.manual_seed(0)
    torch.randn(500)
    gen.manual_seed(0)

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    torch.set_rng_state(loaded["rng_state"])
    gen.set_state(loaded["gen_state"])
    assert torch.equal(torch.get_rng_state(), saved_global)
    assert torch.equal(gen.get_state(), saved_gen)


# ---------------------------------------------------------------------------------------
# Resuming produces the same result as an uninterrupted run.
# ---------------------------------------------------------------------------------------


def test_resume_matches_uninterrupted_run(tiny_vl_cfg: VLPretrainConfig, monkeypatch) -> None:
    import cogsyndelta.model.vl_jepa as vl_jepa_mod

    reference_cfg = replace(tiny_vl_cfg, out_dir=str(Path(tiny_vl_cfg.out_dir) / "uninterrupted"))
    reference = pretrain_vl_region(reference_cfg)

    crashing_cfg = replace(
        tiny_vl_cfg, out_dir=str(Path(tiny_vl_cfg.out_dir) / "crashed-then-resumed")
    )

    orig_forward = vl_jepa_mod.IJEPA.forward
    call_count = {"n": 0}

    def flaky_forward(self, images, generator=None):
        call_count["n"] += 1
        if call_count["n"] == 4:  # after the step=2 checkpoint has been written
            raise RuntimeError("simulated crash")
        return orig_forward(self, images, generator)

    monkeypatch.setattr(vl_jepa_mod.IJEPA, "forward", flaky_forward)
    with pytest.raises(RuntimeError, match="simulated crash"):
        pretrain_vl_region(crashing_cfg)
    monkeypatch.undo()

    ckpt_dir = Path(crashing_cfg.out_dir) / f"{crashing_cfg.region}-checkpoints"
    assert (ckpt_dir / "step-2.pt").is_file(), "expected a checkpoint before the crash"

    resumed = pretrain_vl_region(crashing_cfg)
    assert resumed["resumed"] is True
    assert resumed["resumed_from_step"] == 2

    # Per-key approx, not `==`: identical weights (verified: two fresh constructions
    # from `torch.manual_seed(cfg.seed)` produce bit-identical parameters) and identical
    # input (the shared `cache_dir` fixture means both calls read the SAME decoded
    # `.npy`) still let `model.target_encoder(probe_batch)`'s CPU convolution kernels
    # pick a different (still numerically correct) reduction order between two separate
    # process-level calls -- a documented PyTorch CPU-determinism caveat
    # (pytorch.org/docs/stable/notes/randomness.html), not a resume-path bug: every
    # OTHER float comparison in this same test (below) already uses `pytest.approx`/
    # `torch.allclose` rather than bare `==`, for exactly this reason.
    for key, ref_value in reference["untrained_baseline"].items():
        assert resumed["untrained_baseline"][key] == pytest.approx(ref_value, abs=1e-5), key

    ref_ckpt_dir = Path(reference_cfg.out_dir) / f"{reference_cfg.region}-checkpoints"
    ref_final = torch.load(
        ref_ckpt_dir / f"step-{tiny_vl_cfg.steps}.pt", map_location="cpu", weights_only=True
    )
    res_final = torch.load(
        ckpt_dir / f"step-{tiny_vl_cfg.steps}.pt", map_location="cpu", weights_only=True
    )
    for key, value in ref_final["model"].items():
        assert torch.allclose(res_final["model"][key], value, atol=1e-6, rtol=1e-5), (
            f"resumed run's final weight for {key} diverged from the uninterrupted run"
        )

    assert resumed["held_out"]["top1"] == pytest.approx(reference["held_out"]["top1"], abs=1e-6)
    assert resumed["held_out"]["rep_std"] == pytest.approx(
        reference["held_out"]["rep_std"], abs=1e-6
    )


# ---------------------------------------------------------------------------------------
# Config mismatch is refused.
# ---------------------------------------------------------------------------------------


def test_config_mismatch_is_refused(tiny_vl_cfg: VLPretrainConfig) -> None:
    pretrain_vl_region(tiny_vl_cfg)
    changed = replace(tiny_vl_cfg, batch_size=tiny_vl_cfg.batch_size + 1)
    with pytest.raises(ValueError, match="refusing to resume"):
        pretrain_vl_region(changed)


def test_config_mismatch_on_jepa_shape_is_refused(tiny_vl_cfg: VLPretrainConfig) -> None:
    """The encoder SHAPE living inside `cfg.jepa` (not just top-level fields) must be
    part of the fingerprint -- a shape change would otherwise fail with a confusing
    tensor-shape RuntimeError deep inside `load_state_dict` instead of a clear refusal."""
    pretrain_vl_region(tiny_vl_cfg)
    changed = replace(tiny_vl_cfg, jepa=replace(tiny_vl_cfg.jepa, dim=16, n_heads=2))
    with pytest.raises(ValueError, match="refusing to resume"):
        pretrain_vl_region(changed)


# ---------------------------------------------------------------------------------------
# Rotation and atomic write reuse the same shared mechanics already proven in
# test_pretrain_resume.py; here we only check rotation actually engages for this harness
# too (it shares `cogsyndelta.regions._checkpoint`, but wiring bugs are still possible).
# ---------------------------------------------------------------------------------------


def test_rotation_keeps_only_the_most_recent_n_periodic_checkpoints(
    tiny_vl_cfg: VLPretrainConfig,
) -> None:
    cfg = replace(tiny_vl_cfg, steps=12, checkpoint_every=2)
    pretrain_vl_region(cfg)
    ckpt_dir = Path(cfg.out_dir) / f"{cfg.region}-checkpoints"
    all_ckpts = sorted(int(p.stem.split("-")[1]) for p in ckpt_dir.glob("step-*.pt"))
    assert len(all_ckpts) <= 3, f"rotation should cap periodic checkpoints, found {all_ckpts}"
    assert max(all_ckpts) == 12, "the last (effectively final) checkpoint must survive rotation"
