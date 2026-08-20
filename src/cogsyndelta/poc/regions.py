"""Second stream region for PoC-2 — residual MLP, same dim in/out.

Paired with LatentVAE so softmax routing is a real choice, not a stub.
"""

from __future__ import annotations

from torch import Tensor, nn


class ResidualMLPRegion(nn.Module):
    """Residual two-layer MLP on the shared stream. One cognitive region."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        name: str = "residual_mlp",
    ) -> None:
        """Build a residual MLP with the given stream width.

        Args:
            dim: Stream dimension (in and out).
            hidden_dim: Hidden width of the two-layer MLP.
            name: Registry name for this region.
        """
        super().__init__()
        self.name = name
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def activate(self, stream: Tensor) -> Tensor:
        """``[B, D] → [B, D]`` residual update."""
        return stream + self.net(stream)
