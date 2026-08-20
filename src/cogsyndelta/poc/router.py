"""Softmax top-k router — MoE gating, not mHC.

Scores the shared stream, keeps top-k regions per token, mixes their
``activate`` outputs. This is a conventional MoE gate. Moderated
hyper-connections are a later interconnect, not this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cogsyndelta.contracts.region import CognitiveRegion


@dataclass
class RouteResult:
    """One routed batch."""

    output: Tensor
    weights: Tensor
    topk_index: Tensor
    topk_weights: Tensor
    region_names: tuple[str, ...]
    load: dict[str, float]


class SoftmaxRouter(nn.Module):
    """Token-wise softmax top-k gate over registered regions."""

    def __init__(self, dim: int, n_regions: int, top_k: int = 1) -> None:
        """Linear gate over ``n_regions`` with softmax top-k mixing.

        Args:
            dim: Stream width.
            n_regions: Number of experts/regions.
            top_k: How many regions receive each token.

        Raises:
            ValueError: If ``n_regions`` or ``top_k`` is < 1.
        """
        super().__init__()
        if n_regions < 1:
            raise ValueError("n_regions must be >= 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.dim = dim
        self.n_regions = n_regions
        self.top_k = top_k
        self.gate = nn.Linear(dim, n_regions)

    def route(self, stream: Tensor, regions: Sequence[CognitiveRegion]) -> RouteResult:
        """Mix region activations. ``stream`` is ``[B, D]``."""
        if len(regions) != self.n_regions:
            raise ValueError(f"expected {self.n_regions} regions, got {len(regions)}")
        if stream.ndim != 2 or stream.size(-1) != self.dim:
            raise ValueError(f"stream must be [B, {self.dim}], got {tuple(stream.shape)}")
        names = tuple(r.name for r in regions)
        logits = self.gate(stream)
        full_weights = F.softmax(logits, dim=-1)
        k = min(self.top_k, self.n_regions)
        topk_w, topk_idx = torch.topk(full_weights, k=k, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        mixed = torch.zeros_like(stream)
        load: dict[str, float] = {}
        for i, region in enumerate(regions):
            hit = topk_idx == i
            selected = hit.any(dim=-1)
            weight_i = (topk_w * hit.to(dtype=topk_w.dtype)).sum(dim=-1)
            load[region.name] = float(selected.float().mean().item())
            if not selected.any():
                continue
            activated = region.activate(stream[selected])
            mixed[selected] = mixed[selected] + weight_i[selected].unsqueeze(-1) * activated

        return RouteResult(
            output=mixed,
            weights=full_weights,
            topk_index=topk_idx,
            topk_weights=topk_w,
            region_names=names,
            load=load,
        )


def switch_aux_load_balance(weights: Tensor, topk_index: Tensor) -> Tensor:
    """Switch-Transformer load-balance: ``N * sum_i(f_i * P_i)``.

    For a batch of ``T`` tokens and ``N`` regions:

    - ``P_i`` is the mean softmax probability on region ``i``
    - ``f_i`` is the fraction of tokens whose top-k includes ``i``

    Uniform routing yields ``1.0``. Collapse onto one region yields ``N``.

    Why: Softmax top-k without an auxiliary term sends every token to the
    currently-better region. Switch (Fedus et al., 2021) balances token
    fraction against probability mass so both regions stay in use.

    Args:
        weights: Softmax router probabilities ``[T, N]``.
        topk_index: Hard dispatch indices ``[T, K]`` from ``torch.topk``.

    Returns:
        Scalar tensor ``N * sum(f_i * P_i)``.

    Raises:
        ValueError: If ranks or batch sizes do not match.
    """
    if weights.ndim != 2:
        raise ValueError(f"weights must be [T, N], got {tuple(weights.shape)}")
    if topk_index.ndim != 2:
        raise ValueError(f"topk_index must be [T, K], got {tuple(topk_index.shape)}")
    if topk_index.size(0) != weights.size(0):
        raise ValueError(
            f"batch mismatch: weights {tuple(weights.shape)} vs "
            f"topk_index {tuple(topk_index.shape)}"
        )
    n_regions = weights.size(-1)
    dispatch = torch.zeros(
        weights.size(0),
        n_regions,
        device=weights.device,
        dtype=weights.dtype,
    )
    ones = torch.ones_like(topk_index, dtype=weights.dtype)
    dispatch.scatter_add_(1, topk_index, ones)
    dispatch = dispatch.clamp_max(1.0)
    token_frac = dispatch.mean(dim=0)
    prob_frac = weights.mean(dim=0)
    return n_regions * (token_frac * prob_frac).sum()
