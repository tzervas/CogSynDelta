"""Contract surface tests."""

from __future__ import annotations

import torch

from cogsyndelta.contracts.compactor import (
    BasisResidualCompactor,
    CalibratedQuantCompactor,
    measured_fidelity,
)
from cogsyndelta.contracts.device import DeviceContext
from cogsyndelta.contracts.region import CognitiveRegion
from cogsyndelta.poc.vae import LatentVAE


def test_device_cpu_resolve() -> None:
    ctx = DeviceContext.resolve("cpu")
    assert ctx.name == "cpu"
    assert not ctx.is_cuda


def test_device_auto_returns_valid() -> None:
    ctx = DeviceContext.resolve("auto")
    assert ctx.name == "cpu" or ctx.name.startswith("cuda")


def test_basis_residual_near_perfect_fidelity() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 512)
    c = BasisResidualCompactor(dim=512, basis_rank=64)
    blob = c.compact(x)
    recon = c.reconstruct(blob)
    fid = measured_fidelity(x, recon)
    assert fid >= 0.99, f"got {fid}"
    assert c.measured_ratio(x, blob) > 0.5


def test_calibrated_quant_roundtrip_bytes() -> None:
    torch.manual_seed(1)
    x = torch.randn(4, 256)
    c = CalibratedQuantCompactor(bits=8)
    blob = c.compact(x)
    assert blob.stored_bytes > 0
    recon = c.reconstruct(blob)
    fid = measured_fidelity(x, recon)
    assert fid >= 0.85, f"got {fid}"


def test_latent_vae_is_cognitive_region() -> None:
    model = LatentVAE(input_dim=8, hidden_dim=16, latent_dim=4)
    assert isinstance(model, CognitiveRegion)
    x = torch.randn(2, 8)
    y = model.activate(x)
    assert y.shape == x.shape
    recon, mu, logvar = model.forward(x)
    assert isinstance(recon, torch.Tensor)
    assert mu.shape[-1] == 4
    assert logvar.shape[-1] == 4


def test_calibrated_quant_error_decreases_with_bits() -> None:
    """More bits must never mean worse reconstruction.

    Regression 1: codes were clamped to (1 << bits) - 1 then cast to uint8
    unconditionally, so 12 or 16 bits truncated mod 256 with no error and no warning --
    the blob still deserialized and reconstructed to plausible numbers. Measured then:
    bits=8 -> 0.99999, bits=12 -> 0.050, bits=16 -> 0.021.

    Regression 2: vmin/scale were stored as float16, whose ULP (~3.0 * 2**-10) dominates
    the quantisation step above ~12 bits. 16-bit RMSE came out WORSE than 12-bit.

    Measured in float64 RMSE rather than float32 cosine ON PURPOSE. float32 cosine
    saturates at 1.000000119 for 12, 14 and 16 bits -- it cannot resolve differences at
    that fidelity, and asserting monotonicity on a saturated metric makes the test
    seed-dependent (it failed on 10 of 40 seeds) while telling you nothing. float64 RMSE
    is strictly decreasing on 40 of 40.
    """
    torch.manual_seed(0)
    x = torch.randn(64, 128)

    errors = []
    for bits in (2, 4, 8, 12, 16):
        compactor = CalibratedQuantCompactor(bits=bits)
        recon = compactor.reconstruct(compactor.compact(x))
        rmse = (x.double() - recon.double()).pow(2).mean().sqrt().item()
        fidelity = torch.nn.functional.cosine_similarity(x.flatten(), recon.flatten(), dim=0).item()
        # Catches the truncation regression, which was catastrophic rather than subtle.
        assert fidelity > 0.9, f"bits={bits} fidelity {fidelity:.4f}; likely dtype truncation"
        errors.append((bits, rmse))

    rmses = [e for _, e in errors]
    assert rmses == sorted(rmses, reverse=True), f"error not decreasing with bits: {errors}"
