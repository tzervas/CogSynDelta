"""PoC train loop for one cognitive region — loss must fall under test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cogsyndelta.contracts.config import TrainConfig
from cogsyndelta.contracts.device import DeviceContext
from cogsyndelta.poc.vae import LatentVAE


def train_latent_vae(
    cfg: TrainConfig,
    device_ctx: DeviceContext,
    seed: int = 42,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train one LatentVAE region on synthetic batches; return loss history."""
    torch.manual_seed(seed)
    model = LatentVAE(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        sigma_scale=cfg.sigma_scale,
    )
    model = device_ctx.module(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    losses: list[float] = []
    model.train()
    for _ in range(cfg.steps):
        x = torch.rand(cfg.batch_size, cfg.input_dim, device=device_ctx.device)
        opt.zero_grad(set_to_none=True)
        recon, mu, logvar = model(x)
        loss, _ = model.elbo_loss(recon, x, mu, logvar)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "config": cfg.model_dump(),
                "losses": losses,
                "device": device_ctx.name,
            },
            path,
        )

    return {
        "losses": losses,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "device": device_ctx.name,
        "steps": cfg.steps,
    }
