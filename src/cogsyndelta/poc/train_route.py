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
from cogsyndelta.data.stream import StreamSource
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
    source: StreamSource,
    batch_size: int,
) -> dict[str, float]:
    """Measure top-k load fractions on a fresh batch drawn from ``source``.

    Why: DoD requires load on an eval batch, independent of train batch size and sized
    for a statistical margin against sampling noise (see the 2026-09-04 note below, and
    ``tests/test_poc_route_train.py::_cpu_train_cfg``).

    The batch used to be ``torch.rand``, which made this the least informative
    measurement in the PoC: a gate scores a linear projection of its input, and on
    isotropic noise the split it produces is a property of the gate's initialisation, not
    of anything about the inputs. Real embeddings share a dominant direction, so a
    non-collapsed split here is evidence that the gate discriminates within that cone
    rather than evidence that noise is unstructured.

    MEASURED, and it is a caveat on the PoC-3 "load not collapsed" claim: the split is
    stream-dependent. 24 steps, seed 42, batch 8, stream_dim 16 -- torch.rand gives
    0.125/0.875, corpus:code gives 0.125/0.875, and corpus:tinystories collapses outright
    to 1.0/0.0 while its own training batch sat at 0.125/0.875. A linear gate over a batch
    of 8 drawn from a strongly anisotropic pool is not stable, and isotropic noise hid
    that by spreading tokens over the sphere so that any linear gate had to split them.
    The claim holds for the corpus this PoC is measured on; it is not a property of the
    aux term alone.

    2026-09-04 (test/poc-route-load-determinism): that "batch of 8" figure above is
    exactly the problem. At an ~0.125 measured minority-routing rate, an 8-token eval
    batch is a ``Binomial(n=8, p=0.125)`` draw with ``P(zero minority hits) = 0.875**8
    ~= 34%`` -- a false "collapsed load" from pure sampling noise roughly one run in
    three, no code or hardware fault required. ``scripts/ci_local.sh`` (the pre-push
    gate) syncs CUDA torch and lets ``trained_corpus`` resolve to the GPU; a plain
    interactive ``pytest`` run may stay on CPU. Neither torch nor cuDNN promises
    bit-identical reductions across that difference even with a fixed seed, and 24
    Adam steps are enough to turn a sub-ULP rounding difference into a different
    trained gate. Either path only has to flip one of the 8 eval tokens to hit the 34%
    tail. The test suite fixed this by widening ``eval_batch_size`` to 64
    (``P(zero minority hits) ~= 0.0002``) rather than chasing cross-device bit-parity,
    which is not a real fix available here. This function's contract did not change --
    ``batch_size`` is still just "however many tokens the caller wants evaluated."

    Args:
        router: Trained softmax gate.
        regions: Registered regions in gate order.
        source: Stream source; a fresh draw, independent of the training batch.
        batch_size: Tokens in the eval batch.

    Returns:
        Region name → fraction of tokens whose top-k includes that region.
    """
    _set_train(router, regions, training=False)
    stream = source.sample(batch_size)
    with torch.no_grad():
        result = router.route(stream, regions)
    return {name: round(frac, 6) for name, frac in result.load.items()}


def train_softmax_router(
    cfg: RouteTrainConfig,
    device_ctx: DeviceContext,
    seed: int = 42,
    *,
    stream_source: StreamSource,
) -> dict[str, Any]:
    """Train the two-region softmax gate; routed MSE must fall.

    Each step mixes ``activate`` outputs with softmax top-k and adds
    Switch-Transformer ``N * sum(f_i * P_i)``. Region parameters are
    trained unless ``cfg.train_regions`` is false. The batch is drawn once from
    ``stream_source`` so first vs last reconstruction is the same tokens.

    Why: PoC-2 only evaluated a frozen gate. Training recon without aux
    LB collapses load to 1.0/0.0; aux keeps both regions in the mix.

    ``stream_source`` is required and has no default. It was ``torch.rand`` inline, which
    meant the routed reconstruction target was noise: the regions were being asked to
    reproduce a signal with no structure to exploit, and the gate was splitting inputs
    that differ only by chance.

    Args:
        cfg: Stream size, steps, learning rate, aux coefficient.
        device_ctx: Resolved CPU/CUDA context.
        seed: RNG seed for model init.
        stream_source: Where the ``[B, stream_dim]`` batch comes from.

    Returns:
        Loss history, last eval load (batch of ``cfg.eval_batch_size``),
        device name, the aux formula string, and the stream's provenance.

    Raises:
        ValueError: If ``cfg.steps`` is less than 1, or the stream is the wrong width.
    """
    if cfg.steps < 1:
        raise ValueError("steps must be >= 1")
    if stream_source.dim != cfg.stream_dim:
        raise ValueError(f"stream dim {stream_source.dim} != cfg.stream_dim {cfg.stream_dim}")

    torch.manual_seed(seed)
    registry = default_two_region_mind(cfg.stream_dim, cfg.hidden_dim, cfg.latent_dim)
    registry.to(device_ctx.device)
    regions = registry.regions()
    router = SoftmaxRouter(dim=cfg.stream_dim, n_regions=len(registry), top_k=cfg.top_k)
    router = device_ctx.module(router)

    params = _trainable_params(router, regions, cfg.train_regions)
    opt = torch.optim.Adam(params, lr=cfg.learning_rate)

    # Same tokens every step so first vs last reconstruction is comparable.
    stream = stream_source.sample(cfg.batch_size).to(device_ctx.device)

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
        stream_source,
        batch_size=cfg.eval_batch_size,
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
        "stream": stream_source.provenance,
        "aux_formula": "N * sum(f_i * P_i)",
        "aux_coef": cfg.aux_coef,
        "train_regions": cfg.train_regions,
        "regions": list(registry.names()),
        "gate": "softmax_topk",
        "notes": "MoE softmax gate + Switch aux LB, not mHC",
    }
