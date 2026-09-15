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

from cogsyndelta.vl.composite import FRAME_SIZE, N_PATCHES, PATCH_SIZE

# W7v (docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md row W7v, ~:2533; g8 sector
# `S02.md`): the encoder's geometry is no longer invented here -- it is IMPORTED from
# `composite.py`, the single source of truth W3r fixed. The default `JEPAConfig` below
# therefore renders a 16x16 grid (256 patches) at 128px, matching every composite frame
# this project renders, rather than the 64px/64-patch geometry a composite cannot fit
# through. An explicit `image_size=64, patch_size=8` config still works (see
# `sincos2d_pos_embed`: it is RECOMPUTED for whatever grid a config asks for, never
# resampled from a table built at a different grid), which is what lets an old
# checkpoint's tests keep exercising the native 64px path deliberately.


@dataclass
class JEPAConfig:
    """Shape of the I-JEPA stack.

    `image_size`/`patch_size` default to W3r's composite geometry (`composite.py`
    `FRAME_SIZE`/`PATCH_SIZE`) rather than a value invented in this module, so the
    default encoder can ingest a rendered composite frame without a shape error. Positions
    are always a 2-D sinusoid recomputed for `self.grid` at construction time (see
    `pos_kind`) -- never an interpolation of a table built for a different grid, which is
    the category error the W7v row's sector study (`g8-visual/S02.md`) rules out. Passing
    an explicit smaller `image_size`/`patch_size` (e.g. the historical 64/8) still works;
    only the DEFAULT changed.
    """

    image_size: int = FRAME_SIZE
    patch_size: int = PATCH_SIZE
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
    pos_kind: str = "sincos2d"
    """Positional-embedding family. Only `"sincos2d"` is implemented: a closed-form 2-D
    sinusoid recomputed from `self.grid` at construction (`sincos2d_pos_embed`). Named as
    a field, not hardcoded, because the W7v sector study (`g8-visual/S02.md` §5.3) names
    2-D RoPE (`pos_kind="rope2d_axial"`) as the right NEXT scheme for extrapolating past
    128px -- reserved, not built, in this increment."""

    def __post_init__(self) -> None:
        """Reject shapes that cannot tile, divide evenly, or support 2-D sincos."""
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size {self.image_size} must be divisible by patch_size {self.patch_size}"
            )
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim {self.dim} must be divisible by n_heads {self.n_heads}")
        if self.pos_kind != "sincos2d":
            raise ValueError(
                f"unsupported pos_kind {self.pos_kind!r}; only 'sincos2d' is implemented "
                f"(see the class docstring)"
            )
        if self.dim % 4 != 0:
            raise ValueError(f"2-D sincos needs dim % 4 == 0, got dim={self.dim}")
        if self.predictor_dim % 4 != 0:
            raise ValueError(
                f"2-D sincos needs predictor_dim % 4 == 0, got predictor_dim={self.predictor_dim}"
            )
        if (
            self.image_size == FRAME_SIZE
            and self.patch_size == PATCH_SIZE
            and self.n_patches != N_PATCHES
        ):
            # Defends the import above: if composite.py's own FRAME_SIZE/PATCH_SIZE/
            # N_PATCHES ever drifted out of arithmetic agreement with each other, this
            # config would silently render the wrong grid at its own stated default.
            raise ValueError(
                f"composite.py invariant broken: FRAME_SIZE={FRAME_SIZE} / "
                f"PATCH_SIZE={PATCH_SIZE} implies {self.n_patches} patches, but "
                f"N_PATCHES={N_PATCHES}"
            )

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

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Attend over all positions, then apply the MLP, both on residual paths.

        Args:
            x: ``[B, N, D]``.
            key_padding_mask: Optional ``[B, N]``, 1 for real positions and 0 for padding.
                Images have no padding and pass None; text does, and must not.

        Returns:
            ``[B, N, D]``.
        """
        b, n, _ = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None:
            keep = key_padding_mask.bool()
            # A row that is entirely padding would softmax over all -inf and produce NaN.
            # Keeping position 0 attendable costs nothing -- the pool masks that row out
            # anyway -- and turns a silent NaN cascade into a no-op.
            empty = ~keep.any(dim=1)
            if empty.any():
                keep = keep.clone()
                keep[empty, 0] = True
            attn_mask = keep[:, None, None, :]
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
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


def sincos2d_pos_embed(grid: int, dim: int) -> torch.Tensor:
    """Fixed 2-D sin/cos positional embedding for a ``grid x grid`` patch layout.

    ``[1, grid*grid, dim]``, RECOMPUTED from ``grid`` every time this is called --
    never resampled or interpolated from a table built at a different grid. That
    distinction is the whole point of this function's existence next to
    :func:`sincos_pos_embed`: an image's positions have two axes and a 1-D raster over
    patch index wraps a row's last column into the next row's first (patch 7 and patch 8
    on an 8x8 grid are adjacent in a 1-D table and on opposite sides of the image), so
    interpolating that table when the grid changes is a category error, not a resize
    (`g8-visual/S02.md` §2 item 1, §5.2; taxonomy row W7v). :func:`sincos_pos_embed`
    itself is correct AS IS for `TextEncoder` (a genuinely 1-D sequence) and is untouched
    by this function's existence.

    Steals `facebookresearch/ijepa`'s ``get_2d_sincos_pos_embed`` closed form (half the
    channels encode the row, half encode the column) and drops its class-token `-1`
    offset, which CSD's CLS-free encoder has no use for (`S02.md` §3.3, survive #27).

    Args:
        grid: Patches per side (`JEPAConfig.grid`).
        dim: Embedding width. Must be divisible by 4: each of the two axes gets `dim/2`
            channels, and each axis's 1-D sinusoid itself needs an even split for
            sin/cos, matching `JEPAConfig.__post_init__`'s `dim % 4 == 0` guard.

    Returns:
        ``[1, grid*grid, dim]``, patch order matching :class:`PatchEmbed`'s
        ``flatten(2)`` raster order: patch ``i`` is row ``i // grid``, column ``i % grid``.
    """
    half = dim // 2
    row = torch.arange(grid).float()
    col = torch.arange(grid).float()
    grid_row, grid_col = torch.meshgrid(row, col, indexing="ij")
    # PatchEmbed's Conv2d output [B, D, H, W] is flatten(2)'d to [B, D, H*W] then
    # transposed to [B, H*W, D] -- row-major, so patch i is (row=i//grid, col=i%grid).
    # Matching that order here is what keeps a position's meaning aligned with the
    # patch actually sitting there.
    pos_row = grid_row.reshape(-1)
    pos_col = grid_col.reshape(-1)

    idx = torch.arange(half // 2).float()
    freq = torch.exp(-math.log(10_000.0) * idx / (half // 2))
    row_angles = pos_row.unsqueeze(1) * freq.unsqueeze(0)
    col_angles = pos_col.unsqueeze(1) * freq.unsqueeze(0)
    emb_row = torch.cat([row_angles.sin(), row_angles.cos()], dim=1)
    emb_col = torch.cat([col_angles.sin(), col_angles.cos()], dim=1)
    return torch.cat([emb_row, emb_col], dim=1).unsqueeze(0)


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
        # 2-D sincos, recomputed for cfg.grid -- see sincos2d_pos_embed's docstring for
        # why this is never an interpolation of a table built at a different grid.
        self.register_buffer("pos_embed", sincos2d_pos_embed(cfg.grid, cfg.dim), persistent=False)
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
        h, mask = self.tokens(x)
        return self.pool(h, mask)

    def tokens(
        self, x: torch.Tensor, keep: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Region-native representation BEFORE pooling (DEC-14/DEC-15).

        Images carry no padding, unlike text, so unlike :class:`TextEncoder` there is no
        masked-out content here -- every patch position `tokens()` returns is real. The
        mask return value exists anyway because :meth:`pool` and the Faculty contract
        (§2.2) take one uniformly across regions; here it is always all-ones.

        Args:
            x: ``[B, C, H, W]``.
            keep: Optional ``[B, K]`` long tensor of patch indices to retain -- see
                :meth:`forward`.

        Returns:
            ``(h [B, N or K, D], mask [B, N or K])``, ``mask`` all-ones.
        """
        h = self.forward(x, keep=keep)
        mask = torch.ones(h.shape[0], h.shape[1], dtype=h.dtype, device=h.device)
        return h, mask

    def pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``[B, T, D] -> [B, D]``: masked mean over patch tokens (DEC-14/DEC-15).

        With an all-ones mask (the only mask :meth:`tokens` ever produces) this is
        ``h.mean(dim=1)`` written as a masked mean instead, so it is the same expression
        :meth:`TextEncoder.pool` uses and the two regions share one contract. There is no
        ``proj`` here: unlike the text regions, `ViTEncoder`'s output width already is the
        shared-stream width the catalogue declares for `visual` (DEC-15), so pooling is
        the whole of it.

        Args:
            h: ``[B, T, D]`` patch representations, as returned by :meth:`tokens`.
            mask: ``[B, T]``, real-position indicator (all-ones for `ViTEncoder`).

        Returns:
            ``[B, D]``.
        """
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)


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
            sincos2d_pos_embed(cfg.grid, cfg.predictor_dim),
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


def check_checkpoint_grid_compatible(checkpoint_config: dict[str, object], cfg: JEPAConfig) -> None:
    """Refuse to attach a checkpoint's weights to a differently-gridded config.

    THE SILENT FAILURE THIS CLOSES: `ViTEncoder.pos_embed` is a non-persistent buffer
    (`register_buffer(..., persistent=False)`), so it is never part of `state_dict()` --
    and `PatchEmbed`'s `Conv2d` and every `ViTBlock`'s parameters do not depend on
    `image_size` at all, only on `dim`/`n_heads`/`patch_size`. That means a bare
    `model.load_state_dict(checkpoint["model"])` SUCCEEDS SILENTLY even when the
    checkpoint was trained at a different resolution: every persisted tensor lines up
    shape-for-shape while the freshly-constructed `pos_embed` means something entirely
    different from the one the checkpoint's weights were trained against (2-D sincos
    recomputed for the NEW grid, not the checkpoint's). This is the "silent resize
    masquerade" the W7v sector study (`g8-visual/S02.md` §9, "Pitfalls") names -- the
    fix here is an explicit pre-check, not a `_decode_split` pixel resize, which would
    paper over the mismatch rather than refuse it.

    Call this before `load_state_dict`, not instead of it.

    Args:
        checkpoint_config: The `JEPAConfig` fields recorded on the checkpoint (e.g. a
            `csd-vl-pretrain-checkpoint/v1` payload's ``"config"`` key --
            `dataclasses.asdict(jepa_cfg)`).
        cfg: The config the encoder is about to be (or already was) constructed with.

    Raises:
        ValueError: `checkpoint_config` is missing `image_size`/`patch_size`, or its
            implied grid does not match `cfg.grid` -- naming both grids, both patch
            counts, and both `(image_size, patch_size)` pairs.
    """
    ckpt_image_size = checkpoint_config.get("image_size")
    ckpt_patch_size = checkpoint_config.get("patch_size")
    if not isinstance(ckpt_image_size, int) or not isinstance(ckpt_patch_size, int):
        raise ValueError(
            "checkpoint config is missing an integer image_size/patch_size -- cannot "
            f"verify its patch grid matches the current config's {cfg.grid}x{cfg.grid} "
            f"grid ({cfg.n_patches} patches, image_size={cfg.image_size}, "
            f"patch_size={cfg.patch_size}). checkpoint_config: {checkpoint_config!r}"
        )
    if ckpt_patch_size == 0 or ckpt_image_size % ckpt_patch_size != 0:
        raise ValueError(
            f"checkpoint config's image_size={ckpt_image_size} is not divisible by its "
            f"patch_size={ckpt_patch_size} -- cannot compute its grid."
        )
    ckpt_grid = ckpt_image_size // ckpt_patch_size
    if ckpt_grid != cfg.grid:
        ckpt_n_patches = ckpt_grid * ckpt_grid
        raise ValueError(
            f"checkpoint grid {ckpt_grid}x{ckpt_grid} ({ckpt_n_patches} patches, from "
            f"image_size={ckpt_image_size}, patch_size={ckpt_patch_size}) does not "
            f"match config grid {cfg.grid}x{cfg.grid} ({cfg.n_patches} patches, from "
            f"image_size={cfg.image_size}, patch_size={cfg.patch_size}). pos_embed is a "
            f"non-persistent buffer, so load_state_dict alone would NOT have caught "
            f"this and would silently attach mismatched-resolution weights under a "
            f"freshly (and wrongly) positioned pos_embed. Refusing to load."
        )


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

    @torch.no_grad()
    def tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`visual`'s pre-pool token surface for the CSD shared stream (DEC-14/DEC-15).

        Delegates to the **target (EMA) encoder**, not the context encoder: DEC-34
        establishes that the target encoder is the deployed half -- ``encode`` above
        already reads through it exclusively -- so the token surface a controller would
        read has to come from the same network ``encode`` and every existing receipt
        does, or ``pool(tokens(x))`` and ``encode(x)`` would silently be about two
        different models.

        Returns:
            ``(h [B, N, D], mask [B, N])`` -- ``[B, 256, 384]`` at this module's default
            (128px) config, mask all-ones (images carry no padding). ``N`` tracks
            ``cfg.n_patches``, not a fixed 64: this docstring hard-coded the pre-W7v
            64-patch shape until the encoder's default grid changed
            (`g8-visual/S02.md` §3 survive #12).
        """
        return self.target_encoder.tokens(images)

    @torch.no_grad()
    def pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``[B, T, D] -> [B, D]``, via the target encoder's pooling (DEC-34).

        ``pool(tokens(x))`` reproduces ``encode(x)`` exactly: both are the target
        encoder's masked mean over its own patch tokens, computed through the same
        module rather than two independently-written expressions that merely agree.
        """
        return self.target_encoder.pool(h, mask)
