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
            raise ValueError(
                f"expected {self.n_regions} regions, got {len(regions)}"
            )
        if stream.ndim != 2 or stream.size(-1) != self.dim:
            raise ValueError(
                f"stream must be [B, {self.dim}], got {tuple(stream.shape)}"
            )
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
