"""I-JEPA: latent visual reasoning by predicting representations, not pixels.

WHY THIS EXISTS
`core/vl_jepa_extension.py` is named for JEPA but contains no JEPA. Searching it for
`ema`, `momentum`, `mask`, `target_encoder` or `detach`-on-targets returns nothing: there
is no target encoder, no masking, no predictor loss, and therefore no training path at
all. It has a real ViT and some surrounding modules, and stops there.

WHAT JEPA ACTUALLY IS, and why each piece is load-bearing
Predict the *representation* of a masked region from the representation of the visible
context. Never reconstruct pixels. That is the whole point: pixel reconstruction spends
capacity on texture and noise that carry no semantic content, while predicting in latent
space forces the model to encode what is predictable *about meaning*. For a project whose
thesis is capability per parameter, that distinction is the entire argument for using
JEPA rather than an autoencoder.

Three mechanisms prevent representation collapse, and removing any one of them causes it:

1. **EMA target encoder.** Targets come from a slowly-updated copy of the context
   encoder, never from the encoder itself. A shared encoder can trivially satisfy the
   objective by emitting a constant.
2. **Stop-gradient on targets.** Gradients must not flow into the target encoder. If they
   do, the model optimises the target as well as the prediction and collapse is the
   cheapest solution.
3. **An asymmetric predictor.** A narrower predictor between context and target stops the
   encoder learning identity.

A collapsed model still produces a beautifully falling loss, which is why there is an
explicit collapse test rather than trust in the loss curve.

WHERE THIS FITS CSD
The encoder output is a `[B, D]` latent. That is deliberately the same surface a
`CognitiveRegion` consumes, so a visual region is a peer of a text region on the shared
stream rather than a special case bolted alongside it. Visual data never becomes tokens.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class JEPAConfig:
    """Shape of the I-JEPA stack."""

    image_size: int = 64
    patch_size: int = 8
    in_channels: int = 3
    dim: int = 384
    depth: int = 6
    n_heads: int = 6
    predictor_dim: int = 192
    predictor_depth: int = 3
    ema_base: float = 0.996
    ema_final: float = 1.0
    context_keep: float = 0.4
    n_target_blocks: int = 4
    target_scale: tuple[float, float] = (0.15, 0.2)

    def __post_init__(self) -> None:
        """Reject shapes that cannot tile or divide evenly."""
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size {self.image_size} must be divisible by patch_size {self.patch_size}"
            )
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim {self.dim} must be divisible by n_heads {self.n_heads}")

    @property
    def grid(self) -> int:
        """Patches per side."""
        return self.image_size // self.patch_size

    @property
    def n_patches(self) -> int:
        """Total patches per image."""
        return self.grid * self.grid


class PatchEmbed(nn.Module):
    """Split an image into patches and linearly embed them."""

    def __init__(self, cfg: JEPAConfig) -> None:
        """Build the patch projection."""
        super().__init__()
        self.proj = nn.Conv2d(
            cfg.in_channels, cfg.dim, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[B, C, H, W]`` to ``[B, N, D]``."""
        return self.proj(x).flatten(2).transpose(1, 2)


class ViTBlock(nn.Module):
    """Pre-norm transformer block with bidirectional attention.

    Bidirectional, not causal: an image has no reading order, and masking a future
    position would be meaningless here. This is the one place the vision stack must NOT
    reuse the causal LM's attention.
    """

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0) -> None:
        """Build attention and MLP sublayers."""
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over all positions, then apply the MLP, both on residual paths."""
        b, n, _ = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x + self.proj(attended.transpose(1, 2).contiguous().view(b, n, -1))
        return x + self.mlp(self.norm2(x))


def sincos_pos_embed(n_patches: int, dim: int) -> torch.Tensor:
    """Fixed 1-D sin/cos positional embedding, ``[1, N, D]``.

    Fixed rather than learned so that positions cost no parameters and the encoder can be
    evaluated at a different patch count without retraining a table.
    """
    pos = torch.arange(n_patches).float().unsqueeze(1)
    idx = torch.arange(dim // 2).float()
    freq = torch.exp(-math.log(10_000.0) * idx / (dim // 2))
    angles = pos * freq.unsqueeze(0)
    return torch.cat([angles.sin(), angles.cos()], dim=1).unsqueeze(0)


class ViTEncoder(nn.Module):
    """Vision transformer over patch embeddings."""

    # register_buffer types the attribute as Tensor | Module; declaring it narrows
    # the type so .expand/indexing type-check.
    pos_embed: torch.Tensor

    def __init__(self, cfg: JEPAConfig) -> None:
        """Build patch embedding, blocks and final norm."""
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.register_buffer(
            "pos_embed", sincos_pos_embed(cfg.n_patches, cfg.dim), persistent=False
        )
        self.blocks = nn.ModuleList(ViTBlock(cfg.dim, cfg.n_heads) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.dim)

    def forward(self, x: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        """Encode an image, optionally over a subset of patches.

        Args:
            x: ``[B, C, H, W]``.
            keep: Optional ``[B, K]`` long tensor of patch indices to retain. Masked
                patches are DROPPED rather than replaced with a mask token, so the
                encoder never sees a placeholder it could learn to exploit.

        Returns:
            ``[B, N or K, D]`` patch representations.
        """
        h = self.patch_embed(x) + self.pos_embed
        if keep is not None:
            h = torch.gather(h, 1, keep.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        for block in self.blocks:
            h = block(h)
        return self.norm(h)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Pool to a single ``[B, D]`` latent — the CSD shared-stream surface."""
        return self.forward(x).mean(dim=1)


class JEPAPredictor(nn.Module):
    """Predict target-patch representations from context representations.

    Deliberately narrower than the encoder (``predictor_dim < dim``). The asymmetry is
    load-bearing: an equally wide predictor lets the encoder settle on identity, which
    is one of the three routes to collapse.
    """

    pos_embed: torch.Tensor

    def __init__(self, cfg: JEPAConfig) -> None:
        """Build the projection in, the blocks, and the projection back out."""
        super().__init__()
        self.proj_in = nn.Linear(cfg.dim, cfg.predictor_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer(
            "pos_embed",
            sincos_pos_embed(cfg.n_patches, cfg.predictor_dim),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            ViTBlock(cfg.predictor_dim, max(cfg.n_heads // 2, 1))
            for _ in range(cfg.predictor_depth)
        )
        self.norm = nn.LayerNorm(cfg.predictor_dim)
        self.proj_out = nn.Linear(cfg.predictor_dim, cfg.dim, bias=False)

    def forward(
        self, context: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor
    ) -> torch.Tensor:
        """Predict representations at ``tgt_idx`` given context at ``ctx_idx``.

        Args:
            context: ``[B, K, D]`` encoder output for the visible patches.
            ctx_idx: ``[B, K]`` indices of those patches.
            tgt_idx: ``[B, M]`` indices to predict.

        Returns:
            ``[B, M, D]`` predicted representations.
        """
        b, _, _ = context.shape
        m = tgt_idx.size(1)

        h = self.proj_in(context)
        h = h + torch.gather(
            self.pos_embed.expand(b, -1, -1),
            1,
            ctx_idx.unsqueeze(-1).expand(-1, -1, h.size(-1)),
        )

        # Queries carry only position, so the predictor must infer content from context.
        queries = self.mask_token.expand(b, m, -1) + torch.gather(
            self.pos_embed.expand(b, -1, -1),
            1,
            tgt_idx.unsqueeze(-1).expand(-1, -1, self.mask_token.size(-1)),
        )

        h = torch.cat([h, queries], dim=1)
        for block in self.blocks:
            h = block(h)
        return self.proj_out(self.norm(h[:, -m:]))


def sample_masks(
    cfg: JEPAConfig, batch: int, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample context and target patch indices.

    Returns disjoint sets: targets are removed from the context, so the model cannot
    satisfy the objective by copying a patch it can already see. That overlap is an easy
    bug to introduce and it makes the loss collapse toward zero while teaching nothing.

    Args:
        cfg: Config supplying grid size and ratios.
        batch: Batch size.
        generator: Optional RNG for reproducibility.

    Returns:
        ``(ctx_idx [B, K], tgt_idx [B, M])`` long tensors.
    """
    n = cfg.n_patches
    n_target = max(1, int(n * cfg.target_scale[1] * cfg.n_target_blocks / 4))
    n_context = max(1, int(n * cfg.context_keep))

    ctx_list, tgt_list = [], []
    for _ in range(batch):
        perm = torch.randperm(n, generator=generator)
        tgt = perm[:n_target]
        remaining = perm[n_target:]
        ctx = remaining[:n_context]
        ctx_list.append(ctx)
        tgt_list.append(tgt)
    return torch.stack(ctx_list), torch.stack(tgt_list)


class IJEPA(nn.Module):
    """Image-JEPA: context encoder, EMA target encoder, and a predictor."""

    def __init__(self, cfg: JEPAConfig) -> None:
        """Build the context encoder, its EMA copy, and the predictor."""
        super().__init__()
        self.cfg = cfg
        self.encoder = ViTEncoder(cfg)
        # A deep copy, not a reference: the target must lag the context encoder.
        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.predictor = JEPAPredictor(cfg)

    @torch.no_grad()
    def update_target(self, momentum: float | None = None) -> None:
        """EMA-update the target encoder.

        Under no_grad and in-place: the target encoder must never receive gradients, and
        must never appear in the autograd graph.
        """
        m = self.cfg.ema_base if momentum is None else momentum
        for tgt, src in zip(
            self.target_encoder.parameters(), self.encoder.parameters(), strict=True
        ):
            tgt.mul_(m).add_(src.detach(), alpha=1.0 - m)
        # Distinct names from the parameter loop above: parameters() yields Parameter
        # and buffers() yields Tensor, so reusing tgt/src conflates two types.
        for tgt_buf, src_buf in zip(
            self.target_encoder.buffers(), self.encoder.buffers(), strict=True
        ):
            tgt_buf.copy_(src_buf)

    def forward(
        self, images: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One JEPA step: predict masked-region representations from visible context.

        Args:
            images: ``[B, C, H, W]``.
            generator: Optional RNG for mask sampling.

        Returns:
            ``(loss, stats)``. ``stats`` carries a representation-variance figure, which
            is the collapse signal -- a collapsed model has near-zero variance while its
            loss looks excellent.
        """
        b = images.size(0)
        ctx_idx, tgt_idx = sample_masks(self.cfg, b, generator)
        ctx_idx = ctx_idx.to(images.device)
        tgt_idx = tgt_idx.to(images.device)

        context = self.encoder(images, keep=ctx_idx)

        with torch.no_grad():
            full_target = self.target_encoder(images)
            targets = torch.gather(
                full_target, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, full_target.size(-1))
            )
            # Normalising targets stops the model minimising loss by shrinking their
            # magnitude rather than predicting them better.
            targets = F.layer_norm(targets, (targets.size(-1),))

        predicted = self.predictor(context, ctx_idx, tgt_idx)
        loss = F.smooth_l1_loss(predicted, targets)

        with torch.no_grad():
            # Std across the batch, averaged over features. Near zero means every image
            # maps to the same representation: collapse.
            rep_std = full_target.mean(dim=1).std(dim=0).mean().item()

        return loss, {"loss": loss.item(), "rep_std": rep_std}

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Produce the ``[B, D]`` latent for the CSD shared stream.

        Uses the target (EMA) encoder, which is the standard choice for downstream use:
        it is the smoothed, more stable of the two.
        """
        return self.target_encoder.embed(images)
