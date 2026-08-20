"""PoC configuration surface (Pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DevicePrefer = Literal["cpu", "cuda", "auto"]


class TrainConfig(BaseModel):
    """Minimal VAE train settings for PoC."""

    input_dim: int = 784
    hidden_dim: int = 128
    latent_dim: int = 16
    batch_size: int = 64
    steps: int = 50
    learning_rate: float = 1e-3
    sigma_scale: float = 1.0


class CompressionConfig(BaseModel):
    """Compression targets — claims must be backed by measured metrics."""

    min_fidelity: float = Field(default=0.90, ge=0.0, le=1.0)
    target_ratio: float = Field(default=2.0, gt=1.0)
    quant_bits: int = Field(default=8, ge=2, le=16)
    embed_dim: int = 512
    basis_rank: int = 64


class RouteConfig(BaseModel):
    """Two-region softmax route settings. Gate is MoE, not mHC."""

    stream_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 16
    top_k: int = Field(default=1, ge=1)
    batch_size: int = 16
    compact: bool = True


class RouteTrainConfig(BaseModel):
    """Train the softmax gate (and optionally regions) with Switch aux LB."""

    stream_dim: int = 16
    hidden_dim: int = 32
    latent_dim: int = 4
    top_k: int = Field(default=1, ge=1)
    batch_size: int = 8
    eval_batch_size: int = Field(default=8, ge=1)
    steps: int = 40
    learning_rate: float = 1e-2
    aux_coef: float = Field(default=1.0, ge=0.0)
    train_regions: bool = True


class PocConfig(BaseModel):
    """Top-level PoC config."""

    device: DevicePrefer = "cpu"
    train: TrainConfig = Field(default_factory=TrainConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    route: RouteConfig = Field(default_factory=RouteConfig)
    route_train: RouteTrainConfig = Field(default_factory=RouteTrainConfig)
    seed: int = 42
