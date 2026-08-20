"""PoC-2 route bench — two regions, softmax gate, optional compact."""

from __future__ import annotations

from typing import Any

import torch

from cogsyndelta.contracts.compactor import CalibratedQuantCompactor, measured_fidelity
from cogsyndelta.contracts.config import RouteConfig
from cogsyndelta.contracts.device import DeviceContext
from cogsyndelta.contracts.metrics import MetricsRecord, MetricsStatus
from cogsyndelta.contracts.registry import RegionRegistry
from cogsyndelta.poc.regions import ResidualMLPRegion
from cogsyndelta.poc.router import SoftmaxRouter
from cogsyndelta.poc.vae import LatentVAE


def default_two_region_mind(
    stream_dim: int,
    hidden_dim: int,
    latent_dim: int,
) -> RegionRegistry:
    """Residual MLP + stream-dim LatentVAE. Same [B, D] surface."""
    registry = RegionRegistry()
    registry.register(ResidualMLPRegion(dim=stream_dim, hidden_dim=hidden_dim))
    registry.register(
        LatentVAE(
            input_dim=stream_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            name="stream_vae",
        )
    )
    return registry


def run_route(
    cfg: RouteConfig,
    device_ctx: DeviceContext,
    seed: int = 42,
) -> dict[str, Any]:
    """Route a synthetic batch; optionally compact the mixed stream."""
    torch.manual_seed(seed)
    registry = default_two_region_mind(cfg.stream_dim, cfg.hidden_dim, cfg.latent_dim)
    registry.to(device_ctx.device)
    router = SoftmaxRouter(dim=cfg.stream_dim, n_regions=len(registry), top_k=cfg.top_k)
    router = device_ctx.module(router)
    router.eval()
    stream = torch.randn(cfg.batch_size, cfg.stream_dim, device=device_ctx.device)
    with torch.no_grad():
        result = router.route(stream, registry.regions())

    mean_weights = {
        name: round(float(result.weights[:, i].mean().item()), 6)
        for i, name in enumerate(result.region_names)
    }
    payload: dict[str, Any] = {
        "device": device_ctx.name,
        "regions": list(registry.names()),
        "top_k": cfg.top_k,
        "batch": cfg.batch_size,
        "stream_dim": cfg.stream_dim,
        "load": {k: round(v, 6) for k, v in result.load.items()},
        "mean_weights": mean_weights,
        "output_norm": round(float(result.output.norm(dim=-1).mean().item()), 6),
        "gate": "softmax_topk",
        "notes": "MoE softmax gate, not mHC",
    }

    records: list[MetricsRecord] = [
        MetricsRecord(
            name="softmax_router_weights",
            claim="per-token softmax weights sum to 1",
            measured={
                "weight_sum_mean": round(float(result.weights.sum(dim=-1).mean().item()), 6),
                "load": payload["load"],
            },
            status=MetricsStatus.PASS
            if torch.allclose(
                result.weights.sum(dim=-1),
                torch.ones(result.weights.size(0), device=result.weights.device),
                atol=1e-5,
            )
            else MetricsStatus.GAP,
            device=device_ctx.name,
        )
    ]

    if cfg.compact:
        x = result.output.detach()
        compactor = CalibratedQuantCompactor(bits=8)
        blob = compactor.compact(x)
        recon = compactor.reconstruct(blob)
        fid = measured_fidelity(x.cpu(), recon)
        ratio = compactor.measured_ratio(x, blob)
        records.append(
            MetricsRecord(
                name="routed_stream_quant8",
                claim="fidelity>=0.90 after compact of routed stream",
                measured={
                    "fidelity": round(fid, 6),
                    "ratio": round(ratio, 4),
                    "stored_bytes": blob.stored_bytes,
                    "original_bytes": blob.original_bytes,
                },
                status=MetricsStatus.PASS if fid >= 0.90 else MetricsStatus.GAP,
                device=device_ctx.name,
                notes="compact after mix, not a new compressor",
            )
        )
        payload["compact"] = records[-1].measured

    return {"route": payload, "records": records}
