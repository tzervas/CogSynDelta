"""RegionRegistry — named cognitive regions of one mind.

PoC-2: register / lookup only. Not a fleet, not an agent marketplace.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

from cogsyndelta.contracts.region import CognitiveRegion


class RegionRegistry:
    """Ordered name → region map. Duplicate names are rejected."""

    def __init__(self) -> None:
        """Create an empty name → region map."""
        self._regions: dict[str, CognitiveRegion] = {}

    def register(self, region: CognitiveRegion) -> None:
        """Add a region. Raises if name is empty or already taken."""
        if not isinstance(region, CognitiveRegion):
            raise TypeError(f"region must satisfy CognitiveRegion, got {type(region)!r}")
        name = region.name
        if not name:
            raise ValueError("region.name must be a non-empty str")
        if name in self._regions:
            raise ValueError(f"duplicate region name: {name!r}")
        self._regions[name] = region

    def get(self, name: str) -> CognitiveRegion:
        """Return the region registered under ``name``."""
        try:
            return self._regions[name]
        except KeyError:
            raise KeyError(f"unknown region: {name!r}") from None

    def names(self) -> tuple[str, ...]:
        """Registration order."""
        return tuple(self._regions)

    def regions(self) -> tuple[CognitiveRegion, ...]:
        """Registration order."""
        return tuple(self._regions.values())

    def __len__(self) -> int:
        """Number of registered regions."""
        return len(self._regions)

    def __iter__(self) -> Iterator[CognitiveRegion]:
        """Iterate regions in registration order."""
        return iter(self._regions.values())

    def to(self, device: torch.device) -> RegionRegistry:
        """Move any ``nn.Module`` regions onto ``device``. Returns self."""
        for region in self._regions.values():
            if isinstance(region, nn.Module):
                region.to(device)
        return self
