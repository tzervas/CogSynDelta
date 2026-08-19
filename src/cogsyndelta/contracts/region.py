"""Cognitive region protocol — MoE-adjacent, one mind not many agents.

CSD is closer to Mixture-of-Experts than to a multi-agent swarm:
- A region is an expert-like specialist inside one agent/mind
- Routing between regions (later: mHC / moderated interconnect) is the gate
- Shared compressed memory is the substrate all regions read/write
- Explore → cull → meta-optimize is cognitive control over the whole mind

The protocol surface is ``activate(stream) -> [B, D]``. Train-time
``forward`` on a region (e.g. LatentVAE ELBO tuple) is a separate API.
Do not introduce AgentFleet / SWE-agent modules on this surface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class CognitiveRegion(Protocol):
    """One specialist module of a single mind.

    Implementations are neural regions (VAE encoder, residual MLP, later
    vision / planner heads), not autonomous agents with tools and identity.
    """

    name: str

    def activate(self, stream: torch.Tensor) -> torch.Tensor:
        """Map shared stream ``[B, D]`` to same-shaped ``[B, D]``.

        This is the routing surface. It is not the train-loss forward.
        """
        ...
