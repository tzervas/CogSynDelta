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
