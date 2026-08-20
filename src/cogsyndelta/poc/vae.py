"""Honest Latent VAE for PoC — one expert-shaped region, not a multi-agent swarm.

CogSynDelta is MoE-adjacent: regions of one mind, not complete agents.
This module is a single trainable region (encoder/decoder expert) used to
prove the shared substrate (device + loss that falls) before multi-region routing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LatentVAE(nn.Module):
    """Minimal VAE region: encode, reparameterize, decode; ELBO loss."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        sigma_scale: float = 1.0,
    ) -> None:
        """Build a two-layer encoder/decoder VAE.

        Args:
            input_dim: Flattened observation size.
            hidden_dim: Hidden width.
            latent_dim: Bottleneck size.
            sigma_scale: Multiplier on the reparameterized std.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sigma_scale = sigma_scale
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_dec1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_dec2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map ``x`` to ``(mu, logvar)`` of the approximate posterior.

        Args:
            x: Flattened batch ``[B, input_dim]``.

        Returns:
            Mean and log-variance tensors ``[B, latent_dim]``.
        """
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample ``z = mu + sigma_scale * std * eps``.

        Args:
            mu: Posterior mean.
            logvar: Posterior log-variance.

        Returns:
            Latent sample with the same shape as ``mu``.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + self.sigma_scale * std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent ``z`` back to the input space via a sigmoid head.

        Args:
            z: Latent batch ``[B, latent_dim]``.

        Returns:
            Reconstruction ``[B, input_dim]``.
        """
        h = F.relu(self.fc_dec1(z))
        return torch.sigmoid(self.fc_dec2(h))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ELBO train forward: ``(recon, mu, logvar)``.

        Args:
            x: Flattened batch ``[B, input_dim]``.

        Returns:
            Reconstruction and the encoder statistics.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def elbo_loss(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        recon_weight: float = 1.0,
        kl_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Sum reconstruction MSE and KL, averaged over the batch.

        Args:
            recon: Decoder output.
            x: Target batch.
            mu: Encoder mean.
            logvar: Encoder log-variance.
            recon_weight: Scale on reconstruction term.
            kl_weight: Scale on KL term.

        Returns:
            Scalar loss and a dict of detached-friendly components.
        """
        recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        total = recon_weight * recon_loss + kl_weight * kl
        return total, {"total": total, "reconstruction": recon_loss, "kl": kl}
