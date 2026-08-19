"""CUDA train / compress / route path.

Skips the whole module unless ``torch.cuda.is_available()``. These tests
assert the same pass criteria as the CPU PoC suite on DeviceContext
``cuda`` — they do not claim speedups or 10× compression.
"""

from __future__ import annotations

import pytest
import torch

from cogsyndelta.contracts.config import CompressionConfig, RouteConfig, TrainConfig
from cogsyndelta.contracts.device import DeviceContext
from cogsyndelta.contracts.metrics import MetricsStatus
from cogsyndelta.poc.compress import run_compression_bench
from cogsyndelta.poc.route import run_route
from cogsyndelta.poc.train import train_latent_vae

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


def test_device_cuda_resolve() -> None:
    """Resolve explicit cuda prefer onto a live CUDA device."""
    ctx = DeviceContext.resolve("cuda")
    assert ctx.is_cuda
    assert ctx.name.startswith("cuda")
    assert ctx.device.type == "cuda"


def test_train_loss_decreases_cuda() -> None:
    """LatentVAE ELBO last step is below first step on CUDA."""
    cfg = TrainConfig(steps=40, batch_size=32, hidden_dim=64, latent_dim=8, learning_rate=1e-2)
    ctx = DeviceContext.resolve("cuda")
    result = train_latent_vae(cfg, ctx, seed=123)
    assert result["device"].startswith("cuda")
    assert result["last_loss"] < result["first_loss"], (
        f"first={result['first_loss']:.4f} last={result['last_loss']:.4f}"
    )


def test_compression_bench_cuda() -> None:
    """Basis residual stays high-fidelity on a CUDA tensor."""
    cfg = CompressionConfig(embed_dim=128, basis_rank=32, quant_bits=8, min_fidelity=0.85)
    ctx = DeviceContext.resolve("cuda")
    records = run_compression_bench(cfg, ctx, batch=8, seed=7)
    by_name = {r.name: r for r in records}
    assert by_name["basis_residual"].status == MetricsStatus.PASS
    assert by_name["basis_residual"].device.startswith("cuda")
    assert by_name["basis_residual"].measured["fidelity"] >= 0.99
    quant = by_name["calibrated_quant_8bit"]
    assert quant.status == MetricsStatus.PASS
    assert quant.measured["fidelity"] >= 0.85


def test_route_bench_cuda() -> None:
    """Softmax route + 8-bit compact of the mixed stream on CUDA."""
    cfg = RouteConfig(stream_dim=32, hidden_dim=64, latent_dim=8, top_k=1, batch_size=8)
    ctx = DeviceContext.resolve("cuda")
    out = run_route(cfg, ctx, seed=11)
    assert out["route"]["device"].startswith("cuda")
    assert out["route"]["regions"] == ["residual_mlp", "stream_vae"]
    assert out["route"]["gate"] == "softmax_topk"
    by_name = {r.name: r for r in out["records"]}
    assert by_name["softmax_router_weights"].status == MetricsStatus.PASS
    assert by_name["routed_stream_quant8"].status == MetricsStatus.PASS
    assert by_name["routed_stream_quant8"].measured["fidelity"] >= 0.90
    load = out["route"]["load"]
    assert set(load) == {"residual_mlp", "stream_vae"}
    assert all(v > 0.0 for v in load.values())
    assert abs(sum(load.values()) - 1.0) < 1e-5
