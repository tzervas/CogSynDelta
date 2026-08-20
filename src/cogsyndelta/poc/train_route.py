"""PoC-3: train the softmax gate with Switch aux load-balance.

Routed reconstruction uses ``CognitiveRegion.activate`` only.
``LatentVAE.forward`` stays the ELBO tuple and is not called here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from cogsyndelta.contracts.config import RouteTrainConfig
from cogsyndelta.contracts.device import DeviceContext
from cogsyndelta.contracts.region import CognitiveRegion
from cogsyndelta.poc.route import default_two_region_mind
from cogsyndelta.poc.router import SoftmaxRouter, switch_aux_load_balance


def _module_regions(regions: Sequence[CognitiveRegion]) -> list[nn.Module]:
    return [region for region in regions if isinstance(region, nn.Module)]


def _trainable_params(
    router: SoftmaxRouter,
    regions: Sequence[CognitiveRegion],
    train_regions: bool,
) -> list[nn.Parameter]:
    params: list[nn.Parameter] = list(router.parameters())
    if train_regions:
        for module in _module_regions(regions):
            params.extend(module.parameters())
    return params


def _set_train(
    router: SoftmaxRouter,
    regions: Sequence[CognitiveRegion],
    training: bool,
) -> None:
    router.train(training)
    for module in _module_regions(regions):
        module.train(training)


def evaluate_route_load(
    router: SoftmaxRouter,
    regions: Sequence[CognitiveRegion],
    device_ctx: DeviceContext,
    batch_size: int,
    stream_dim: int,
    seed: int,
) -> dict[str, float]:
    """Measure top-k load fractions on a fresh synthetic batch.

    Why: DoD requires load on a batch of 8, independent of train batch size.

    Args:
        router: Trained softmax gate.
        regions: Registered regions in gate order.
        device_ctx: Resolved CPU/CUDA context.
        batch_size: Tokens in the eval batch.
        stream_dim: Shared stream width.
        seed: RNG seed for the eval batch.

    Returns:
        Region name → fraction of tokens whose top-k includes that region.
    """
    _set_train(router, regions, training=False)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    stream = torch.rand(batch_size, stream_dim, generator=generator).to(device_ctx.device)
    with torch.no_grad():
        result = router.route(stream, regions)
    return {name: round(frac, 6) for name, frac in result.load.items()}


def train_softmax_router(
    cfg: RouteTrainConfig,
    device_ctx: DeviceContext,
    seed: int = 42,
) -> dict[str, Any]:
    """Train the two-region softmax gate; routed MSE must fall.

    Each step mixes ``activate`` outputs with softmax top-k and adds
    Switch-Transformer ``N * sum(f_i * P_i)``. Region parameters are
    trained unless ``cfg.train_regions`` is false. The synthetic stream is
    drawn once per seed so first vs last reconstruction is the same tokens.

    Why: PoC-2 only evaluated a frozen gate. Training recon without aux
    LB collapses load to 1.0/0.0; aux keeps both regions in the mix.

    Args:
        cfg: Stream size, steps, learning rate, aux coefficient.
        device_ctx: Resolved CPU/CUDA context.
        seed: RNG seed for init and synthetic batches.

    Returns:
        Loss history, last eval load (batch of ``cfg.eval_batch_size``),
        device name, and the aux formula string.

    Raises:
        ValueError: If ``cfg.steps`` is less than 1.
    """
    if cfg.steps < 1:
        raise ValueError("steps must be >= 1")

    torch.manual_seed(seed)
    registry = default_two_region_mind(cfg.stream_dim, cfg.hidden_dim, cfg.latent_dim)
    registry.to(device_ctx.device)
    regions = registry.regions()
    router = SoftmaxRouter(dim=cfg.stream_dim, n_regions=len(registry), top_k=cfg.top_k)
    router = device_ctx.module(router)

    params = _trainable_params(router, regions, cfg.train_regions)
    opt = torch.optim.Adam(params, lr=cfg.learning_rate)

    # Same tokens every step so first vs last reconstruction is comparable.
    stream = torch.rand(cfg.batch_size, cfg.stream_dim, device=device_ctx.device)

    recon_losses: list[float] = []
    aux_lbs: list[float] = []
    last_load: dict[str, float] = {}
    _set_train(router, regions, training=True)

    for _ in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        result = router.route(stream, regions)
        recon = F.mse_loss(result.output, stream)
        aux = switch_aux_load_balance(result.weights, result.topk_index)
        total = recon + cfg.aux_coef * aux
        total.backward()
        opt.step()
        recon_losses.append(float(recon.item()))
        aux_lbs.append(float(aux.item()))
        last_load = {name: round(frac, 6) for name, frac in result.load.items()}

    eval_load = evaluate_route_load(
        router,
        regions,
        device_ctx,
        batch_size=cfg.eval_batch_size,
        stream_dim=cfg.stream_dim,
        seed=seed + 1,
    )

    return {
        "losses": recon_losses,
        "aux_lbs": aux_lbs,
        "first_loss": recon_losses[0],
        "last_loss": recon_losses[-1],
        "first_aux_lb": aux_lbs[0],
        "last_aux_lb": aux_lbs[-1],
        "load": eval_load,
        "train_load": last_load,
        "device": device_ctx.name,
        "steps": cfg.steps,
        "aux_formula": "N * sum(f_i * P_i)",
        "aux_coef": cfg.aux_coef,
        "train_regions": cfg.train_regions,
        "regions": list(registry.names()),
        "gate": "softmax_topk",
        "notes": "MoE softmax gate + Switch aux LB, not mHC",
    }
