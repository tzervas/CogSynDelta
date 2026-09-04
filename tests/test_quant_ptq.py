"""Tests for sensitivity-driven post-training quantization.

The claims worth testing here are the ones that would otherwise be flattering fiction:
that a narrower bit-width produces a genuinely smaller buffer, that codes survive a
round-trip exactly, and that the allocator spends bits where the task says they matter
rather than where tensor size suggests they might.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from cogsyndelta.quant.packing import pack_codes, packed_bytes, unpack_codes
from cogsyndelta.quant.ptq import (
    LADDER,
    apply_plan,
    build_plan,
    dequantize_tensor,
    measure_sensitivity,
    quantizable,
    quantize_tensor,
)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8, 12, 16])
def test_pack_roundtrip_is_exact(bits: int) -> None:
    """Codes must survive packing unchanged at every supported width."""
    rng = np.random.default_rng(bits)
    codes = rng.integers(0, 1 << bits, size=3000).astype(np.uint16)
    buf = pack_codes(codes, bits)
    assert np.array_equal(unpack_codes(buf, bits, codes.size), codes)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6])
def test_sub_byte_widths_actually_shrink(bits: int) -> None:
    """The whole premise of selective quantization: fewer bits must mean fewer bytes.

    Storing 4-bit codes in uint8 would pass a round-trip test while saving nothing. This
    asserts the buffer is genuinely proportional to the width.
    """
    n = 8000
    codes = np.zeros(n, dtype=np.uint16)
    buf = pack_codes(codes, bits)
    assert buf.nbytes == packed_bytes(n, bits)
    assert buf.nbytes < n, "sub-8-bit must be smaller than one byte per value"


def test_pack_rejects_codes_that_do_not_fit() -> None:
    """Silent truncation would corrupt weights into plausible-looking numbers."""
    with pytest.raises(ValueError, match="does not fit"):
        pack_codes(np.array([16], dtype=np.uint16), 4)


def test_more_bits_never_reconstructs_worse() -> None:
    """Fidelity must be monotone in width, or the ladder is meaningless."""
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    errors = []
    for bits in LADDER:
        err = (w - dequantize_tensor(quantize_tensor(w, bits))).pow(2).mean().item()
        errors.append(err)
    assert errors == sorted(errors, reverse=True)


def test_quantizable_excludes_norms_and_biases() -> None:
    """1-D parameters are negligible in size and dominant in sensitivity."""
    model = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256))
    names = quantizable(model)
    assert all(dict(model.named_parameters())[n].dim() >= 2 for n in names)
    assert not any("bias" in n for n in names)


def test_sensitivity_restores_weights_exactly() -> None:
    """A leaked perturbation would silently poison every subsequent measurement."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 64))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    measure_sensitivity(model, lambda _m: 1.0, quantizable(model), 3, 1.0)
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name]), f"{name} was not restored"


def test_sensitivity_restores_even_when_eval_raises() -> None:
    """The restore is in a finally block precisely so a failing eval cannot leak state."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(128, 128))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    def boom(_m: nn.Module) -> float:
        raise RuntimeError("eval failed")

    with pytest.raises(RuntimeError):
        measure_sensitivity(model, boom, quantizable(model), 3, 1.0)
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name])


def test_plan_promotes_the_tensor_the_task_cares_about() -> None:
    """Bits must follow measured task sensitivity, not tensor size.

    The eval here depends only on the SECOND layer, which is the smaller of the two. A
    size-based heuristic would protect the larger first layer and destroy the metric; a
    sensitivity-driven allocator promotes the one that actually matters.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 8))
    target = model[1].weight.detach().clone()

    def eval_fn(m: nn.Module) -> float:
        # Higher when layer 1's weights are closer to their fp32 values.
        return -float((m[1].weight.detach() - target).pow(2).mean().item())

    plan = build_plan(model, eval_fn, baseline=0.0, tolerance=1e-6, aggressive_bits=2)
    assert plan.bits["1.weight"] > 2, "the sensitive tensor should have been promoted"
    assert plan.bits["1.weight"] >= plan.bits["0.weight"]


def test_apply_plan_reports_measured_bytes() -> None:
    """Reported size must come from packed buffers, not from a nominal width."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(256, 256))
    names = quantizable(model)
    fp32 = sum(p.numel() * 4 for p in model.parameters())

    from cogsyndelta.quant.ptq import QuantPlan

    small = QuantPlan(bits=dict.fromkeys(names, 2))
    large = QuantPlan(bits=dict.fromkeys(names, 8))
    _, small_bytes = apply_plan(model, small)
    _, large_bytes = apply_plan(model, large)
    assert small_bytes < large_bytes < fp32


# ------------------------------------------------------------ persisted artifact


def test_pack_state_dict_roundtrips_within_quantization_error() -> None:
    """Reconstructing from the packed artifact must match what apply_plan's in-memory
    dequantization produces for the same plan -- the artifact is not a lossy summary
    of the plan, it IS the plan, just packed."""
    from cogsyndelta.quant.ptq import pack_state_dict, unpack_state_dict

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 8))
    names = quantizable(model)
    from cogsyndelta.quant.ptq import QuantPlan

    plan = QuantPlan(bits=dict.fromkeys(names, 4))
    plan.fp32 = [n for n, _ in model.named_parameters() if n not in plan.bits]

    packed = pack_state_dict(model, plan)
    restored = unpack_state_dict(packed)

    live, _ = apply_plan(model, plan)
    for name, p in live.named_parameters():
        assert torch.allclose(restored[name], p.detach(), atol=1e-5)


def test_save_packed_artifact_writes_hashable_file(tmp_path: Path) -> None:
    from cogsyndelta.quant.ptq import QuantPlan, save_packed_artifact
    from cogsyndelta.regions._checkpoint import sha256_file

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(512, 512))
    names = quantizable(model)
    plan = QuantPlan(bits=dict.fromkeys(names, 3))
    plan.fp32 = [n for n, _ in model.named_parameters() if n not in plan.bits]

    out = tmp_path / "final.ptq.pt"
    got_sha = save_packed_artifact(model, plan, out)

    assert out.is_file()
    assert got_sha == sha256_file(out)
    # Smaller than the equivalent fp32 dump -- the whole point of packing.
    fp32_bytes = sum(p.numel() * 4 for p in model.parameters())
    assert out.stat().st_size < fp32_bytes


# --------------------------------------------------- artifact shape, device, buffers


class _WithPersistentBuffer(nn.Module):
    """A module whose state_dict carries something named_parameters() never yields."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(64, 64)
        self.register_buffer("running_mean", torch.zeros(64))  # persistent by default


class _WithDerivedBuffer(nn.Module):
    """The pattern regions/text_encoder.py uses for a recomputable buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(64, 64)
        self.register_buffer("pos", torch.zeros(64), persistent=False)


def test_pack_state_dict_refuses_a_persistent_buffer() -> None:
    """A persistent buffer would be dropped in silence and the artifact would still
    claim to be a complete model, so packing must refuse rather than round it down."""
    from cogsyndelta.quant.ptq import QuantPlan, pack_state_dict

    model = _WithPersistentBuffer()
    plan = QuantPlan(bits=dict.fromkeys(quantizable(model), 4))
    with pytest.raises(ValueError, match="running_mean"):
        pack_state_dict(model, plan)


def test_pack_state_dict_allows_a_non_persistent_buffer() -> None:
    """persistent=False is exactly the escape hatch the refusal above points at: the
    buffer is absent from state_dict() by design, so nothing is being dropped."""
    from cogsyndelta.quant.ptq import QuantPlan, pack_state_dict

    model = _WithDerivedBuffer()
    plan = QuantPlan(bits=dict.fromkeys(quantizable(model), 4))
    packed = pack_state_dict(model, plan)
    assert "pos" not in packed["fp32"]
    assert set(packed["fp32"]) | set(packed["bits"]) == {n for n, _ in model.named_parameters()}


def test_load_packed_artifact_roundtrips_and_measures(tmp_path: Path) -> None:
    from cogsyndelta.quant.ptq import (
        QuantPlan,
        load_packed_artifact,
        packed_stored_bytes,
        packed_width_histogram,
        save_packed_artifact,
        unpack_state_dict,
    )

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 8))
    names = quantizable(model)
    plan = QuantPlan(bits=dict.fromkeys(names, 4))
    plan.fp32 = [n for n, _ in model.named_parameters() if n not in plan.bits]

    out = tmp_path / "final.ptq.pt"
    save_packed_artifact(model, plan, out)
    packed = load_packed_artifact(out)

    restored = unpack_state_dict(packed)
    live, measured = apply_plan(model, plan)
    for name, p in live.named_parameters():
        assert torch.allclose(restored[name], p.detach(), atol=1e-5)

    # The whole point of packed_stored_bytes: it is apply_plan's number, recomputed
    # from the file, so a receipt's stored_bytes can be checked instead of copied.
    assert packed_stored_bytes(packed) == measured
    assert packed_width_histogram(packed) == {"4": len(names)}


def test_load_packed_artifact_rejects_a_foreign_file(tmp_path: Path) -> None:
    from cogsyndelta.quant.ptq import load_packed_artifact

    plain = tmp_path / "not-ours.pt"
    torch.save({"model": {"w": torch.zeros(2)}}, plain)
    with pytest.raises(ValueError, match="missing top-level key"):
        load_packed_artifact(plain)


def test_load_packed_artifact_rejects_disagreeing_sections(tmp_path: Path) -> None:
    """`bits` naming a tensor the code sections do not carry is a truncated or
    hand-edited artifact, not a smaller one."""
    from cogsyndelta.quant.ptq import QuantPlan, load_packed_artifact, pack_state_dict

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64))
    plan = QuantPlan(bits=dict.fromkeys(quantizable(model), 4))
    packed = pack_state_dict(model, plan)
    packed["bits"]["0.nonexistent"] = 4
    bad = tmp_path / "bad.ptq.pt"
    torch.save(packed, bad)
    with pytest.raises(ValueError, match="sections disagree"):
        load_packed_artifact(bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_packed_artifact_from_a_cuda_module_has_no_cuda_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """torch.save records each tensor's device. Packing a CUDA model must still write a
    file a CPU-only host can load with no map_location -- and, by the same token, the
    same bytes a CPU host would have written."""
    from cogsyndelta.quant.ptq import QuantPlan, save_packed_artifact

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 8)).cuda()
    plan = QuantPlan(bits=dict.fromkeys(quantizable(model), 4))
    out = tmp_path / "cuda.ptq.pt"
    save_packed_artifact(model, plan, out)

    # No map_location, and CUDA made to look absent: this is exactly what a CPU-only
    # consumer does, and it raised before pack_state_dict moved tensors to the CPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loaded = torch.load(out, weights_only=True)
    assert all(t.device.type == "cpu" for t in loaded["fp32"].values())

    # And the bytes themselves carry no cuda location tag.
    with zipfile.ZipFile(out) as zf:
        pickles = [n for n in zf.namelist() if n.endswith("data.pkl")]
        assert pickles
        assert all(b"cuda" not in zf.read(n) for n in pickles)


def test_cpu_and_gpu_packing_produce_identical_bytes(tmp_path: Path) -> None:
    """The sha256 a receipt records must be a fact about the weights, not about which
    machine ran the quantizer. Skips the GPU half when there is no GPU, and still pins
    that packing the same weights twice is byte-identical.

    Both writes use the SAME basename in different directories, deliberately.
    ``torch.save`` names the zip archive inside the file after the file itself, so two
    artifacts holding bit-identical tensors under different names differ in bytes for a
    reason that has nothing to do with devices. The name is not a free variable here:
    `scripts/csd-quantize.py` derives it from the checkpoint stem and
    `scripts/csd-publish-checkpoint.py` re-derives the same one, so holding it fixed is
    what the real pipeline does and the device is the only thing left varying.
    """
    from cogsyndelta.quant.ptq import QuantPlan, save_packed_artifact
    from cogsyndelta.regions._checkpoint import sha256_file

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 8))
    plan = QuantPlan(bits=dict.fromkeys(quantizable(model), 4))

    cpu_out = tmp_path / "cpu" / "final.ptq.pt"
    cpu_sha = save_packed_artifact(model, plan, cpu_out)
    assert cpu_sha == sha256_file(cpu_out)

    other = tmp_path / "other" / "final.ptq.pt"
    if torch.cuda.is_available():
        other_sha = save_packed_artifact(model.cuda(), plan, other)
    else:
        other_sha = save_packed_artifact(model, plan, other)
    assert other_sha == cpu_sha
