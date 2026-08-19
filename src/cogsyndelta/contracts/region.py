"""Cognitive region protocol — MoE-adjacent, one mind not many agents.

CSD is closer to Mixture-of-Experts than to a multi-agent swarm:
- A region is an expert-like specialist inside one agent/mind
- Routing between regions (later: mHC / moderated interconnect) is the gate
- Shared compressed memory is the substrate all regions read/write
- Explore → cull → meta-optimize is cognitive control over the whole mind

PoC-1 ships one trainable region (LatentVAE) + shared memory compactors.
Do not introduce AgentFleet / SWE-agent modules on this surface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class CognitiveRegion(Protocol):
    """One specialist module of a single mind.

    Implementations are neural regions (VAE encoder, vision encoder, planner head),
    not autonomous agents with their own tools and identity.
    """

    name: str

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Region forward pass on a shared-device tensor."""
        ...
