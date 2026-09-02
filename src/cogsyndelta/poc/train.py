"""PoC train loop for one cognitive region — loss must fall under test."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import torch

from cogsyndelta.contracts.config import TrainConfig
from cogsyndelta.contracts.device import DeviceContext
from cogsyndelta.poc.vae import LatentVAE

_REPO = Path(__file__).resolve().parents[3]
_WIKITEXT_EXCERPT = _REPO / "tests" / "fixtures" / "wikitext-2-raw-v1.train.excerpt.txt"
_WIKITEXT_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"


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


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _token_bucket(token: str, input_dim: int) -> tuple[int, float]:
    """Map a token to (bucket, sign) deterministically.

    hashlib rather than hash(): str hashing is salted per process unless PYTHONHASHSEED
    is pinned, which would make the same corpus encode differently between runs and make
    checkpoints unreproducible.
    """
    digest = hashlib.blake2b(token.encode("utf-8", errors="replace"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % input_dim, 1.0 if (value >> 63) & 1 else -1.0


def encode_public_line(text: str, input_dim: int) -> torch.Tensor:
    """Map one train-split line to a unit-norm ``[input_dim]`` bag-of-tokens vector.

    Uses the hashing trick: each token is hashed to a bucket and accumulated with a
    signed weight. Lines that share vocabulary share coordinates, so cosine similarity
    between related lines is meaningfully greater than between unrelated ones.

    Why not an embedding model: CI has no encoder available and must not pause the 3090's
    LocalAI. This is deterministic, CPU-only, dependency-free, and — unlike the previous
    implementation — actually carries lexical signal.

    Why not hash the whole line (what this replaced): a cryptographic hash of the full
    string is *designed* to decorrelate on any input change, so near-identical sentences
    produced orthogonal vectors and the VAE was fitting 66 fixed pseudorandom points.

    Signed accumulation is deliberate: unsigned counts make hash collisions inflate
    magnitude systematically, whereas signed collisions cancel in expectation.

    Args:
        text: One WikiText-2-raw train line.
        input_dim: LatentVAE observation width.

    Returns:
        Float tensor ``[input_dim]``, L2-normalised. An empty or token-free line yields
        a zero vector rather than raising, since blank lines are normal in WikiText.
    """
    vec = torch.zeros(input_dim, dtype=torch.float32)
    for token in _TOKEN_RE.findall(text.lower()):
        bucket, sign = _token_bucket(token, input_dim)
        vec[bucket] += sign
    norm = torch.linalg.vector_norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def load_public_split_lines(
    dataset: str,
    config: str,
    split: str,
    *,
    revision: str = _WIKITEXT_REV,
) -> list[str]:
    """Load non-empty lines for a pinned public train split.

    Prefers ``CSD_WIKITEXT_TRAIN``, then the in-repo excerpt of
    WikiText-2-raw train (cc-by-sa). Does not add ``datasets`` as a
    required dep.

    Args:
        dataset: Hub id (must be ``Salesforce/wikitext`` this tick).
        config: Hub config (must be ``wikitext-2-raw-v1``).
        split: Must be ``train``.
        revision: Pinned Hub commit.

    Returns:
        Non-empty text lines.

    Raises:
        ValueError: Wrong dataset, config, or split.
        FileNotFoundError: Excerpt missing and no env override.
    """
    if dataset != "Salesforce/wikitext" or config != "wikitext-2-raw-v1":
        raise ValueError(f"unsupported public split {dataset} {config}")
    if split != "train":
        raise ValueError("region-pretrain uses the train split only")
    override = os.environ.get("CSD_WIKITEXT_TRAIN", "").strip()
    path = Path(override) if override else _WIKITEXT_EXCERPT
    if not path.is_file():
        raise FileNotFoundError(f"wikitext train lines missing at {path} (rev {revision})")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("wikitext train file had no lines")
    return lines


def train_latent_vae_on_public_split(
    dataset: str = "Salesforce/wikitext",
    config: str = "wikitext-2-raw-v1",
    split: str = "train",
    steps: int = 20,
    batch_size: int = 8,
    input_dim: int = 768,
    hidden_dim: int = 128,
    latent_dim: int = 16,
    learning_rate: float = 1e-2,
    seed: int = 42,
) -> dict[str, Any]:
    """Train LatentVAE on encoded WikiText-2-raw **train** lines (CPU).

    Why: ``train_latent_vae`` still uses ``torch.rand``. Region-pretrain
    must consume a public train split without G-TRAIN / 14B / LocalAI.

    Args:
        dataset: Hub dataset id.
        config: Hub config name.
        split: Dataset split (train only).
        steps: Optimizer steps (≥20 in the contract test).
        batch_size: Batch width.
        input_dim: Encoder width (768 in the contract test).
        hidden_dim: VAE hidden width.
        latent_dim: Bottleneck width.
        learning_rate: Adam step size.
        seed: Torch seed.

    Returns:
        Loss history plus ``split`` / ``dataset`` / ``config`` keys.
    """
    lines = load_public_split_lines(dataset, config, split)
    torch.manual_seed(seed)
    ctx = DeviceContext.resolve("cpu")
    cfg = TrainConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        batch_size=batch_size,
        steps=steps,
        learning_rate=learning_rate,
    )
    encoded = torch.stack([encode_public_line(line, input_dim) for line in lines])
    n = encoded.size(0)
    model = LatentVAE(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        sigma_scale=cfg.sigma_scale,
    )
    model = ctx.module(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    losses: list[float] = []
    model.train()
    for _step in range(cfg.steps):
        idx = torch.randint(0, n, (cfg.batch_size,))
        x = encoded[idx].to(ctx.device)
        opt.zero_grad(set_to_none=True)
        recon, mu, logvar = model(x)
        loss, _ = model.elbo_loss(recon, x, mu, logvar)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return {
        "losses": losses,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "device": ctx.name,
        "steps": cfg.steps,
        "split": split,
        "dataset": dataset,
        "config": config,
    }
