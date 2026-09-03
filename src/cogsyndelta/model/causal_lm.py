"""A small causal language model: the backbone regions plug into.

WHY THIS EXISTS
The repo had no language model. No token embedding, no causal mask, no next-token loss,
no perplexity — the only "text" path hashed 66 fixture lines. So "train a brain region on
code / retrieval / math" had nothing to be a region *of*, and no capability metric could
exist. This is the smallest honest thing that fixes that.

ARCHITECTURE, and why each choice
Standard decoder-only stack, deliberately unoriginal. The research bet in this project is
the region composition and the memory system; the backbone should be a known-good
baseline so that when a number moves, the cause is the thing under test and not an
untested transformer variant.

- **Pre-norm** blocks. Post-norm needs warmup tuning to stay stable at depth; pre-norm
  trains reliably from step 0, which matters when runs are short and frequent.
- **RMSNorm** rather than LayerNorm. Same stability, no mean subtraction and no bias, so
  slightly fewer parameters and slightly less work per token.
- **RoPE** rather than learned positional embeddings. Positions enter through the
  rotation rather than as a trained table, so nothing has to be re-learned to evaluate at
  a different context length, and no parameters are spent on position.
- **SwiGLU** feed-forward. Better quality per parameter than GELU-MLP at equal width,
  which is the entire objective here.
- **Weight tying** between the token embedding and the LM head. At this scale the
  embedding is a large fraction of the parameter count — untied, a 50257-vocab model
  spends most of its budget on two copies of the same table.
- **F.scaled_dot_product_attention(is_causal=True)**. Uses the fused kernel and gets the
  mask right; a hand-rolled mask is a classic place to silently leak future tokens, and
  the resulting loss curve looks *better*, not worse.

WHAT THIS IS NOT
No MoE, no regions, no memory yet — those compose on top, and the FFN slot is the
intended attachment point. Getting a plain baseline to fall to a believable perplexity
first is what makes any later claim about regions measurable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class CausalLMConfig:
    """Shape of a small decoder-only LM.

    Defaults give 29,920,512 parameters total and 10,621,824 excluding the tied
    embedding (measured, not estimated). Both numbers are quoted because at this scale
    the 50257x384 embedding is most of the model, so a single headline figure misleads.
    """

    vocab_size: int = 50257
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    seq_len: int = 512
    ffn_mult: float = 8 / 3  # SwiGLU convention: ~2/3 of 4x, since it uses three matrices
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        """Reject a config whose dim is not divisible by n_heads."""
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim {self.dim} must be divisible by n_heads {self.n_heads}")

    @property
    def head_dim(self) -> int:
        """Width of a single attention head."""
        return self.dim // self.n_heads


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Build an RMSNorm over ``dim`` features."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``x`` by its root-mean-square and scale."""
        # Compute in fp32: under autocast the reciprocal-sqrt of a mean of squares is a
        # common place for bf16 to lose enough precision to destabilise training.
        dtype = x.dtype
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed.to(dtype)) * self.weight


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tables of shape ``[seq_len, head_dim // 2]``."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    angles = torch.outer(pos, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``[B, H, T, Dh]`` by the cached angles.

    Splits the head dimension into even/odd halves and applies a 2-D rotation to each
    pair, which is the standard interleaved formulation.
    """
    t = x.shape[-2]
    cos_t = cos[:t].unsqueeze(0).unsqueeze(0)
    sin_t = sin[:t].unsqueeze(0).unsqueeze(0)
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    rot_even = x_even * cos_t - x_odd * sin_t
    rot_odd = x_even * sin_t + x_odd * cos_t
    return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE."""

    def __init__(self, cfg: CausalLMConfig) -> None:
        """Build causal self-attention from a model config."""
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Attend over ``x`` ``[B, T, D]`` with RoPE-rotated queries and keys."""
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # is_causal=True rather than a hand-built mask: the fused kernel is faster and
        # cannot be got subtly wrong. A mask that leaks even one future position makes
        # the loss curve look better, so the bug hides in the direction nobody checks.
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).contiguous().view(b, t, -1))


class SwiGLU(nn.Module):
    """Gated feed-forward block. This is the slot regions will later occupy."""

    def __init__(self, dim: int, hidden: int) -> None:
        """Build a SwiGLU block projecting ``dim`` -> ``hidden`` -> ``dim``."""
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated feed-forward transform."""
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, cfg: CausalLMConfig) -> None:
        """Build one pre-norm block: attention, then gated feed-forward."""
        super().__init__()
        hidden = int(cfg.dim * cfg.ffn_mult)
        hidden = 64 * ((hidden + 63) // 64)  # round up for kernel-friendly shapes
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg.dim, hidden)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Run attention then feed-forward, each on its own residual path."""
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class CausalLM(nn.Module):
    """Small decoder-only language model."""

    def __init__(self, cfg: CausalLMConfig) -> None:
        """Build the model, tie the head to the embedding, and cache RoPE tables."""
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight

        cos, sin = build_rope_cache(cfg.seq_len, cfg.head_dim, cfg.rope_theta, torch.device("cpu"))
        # Buffers so .to(device) moves them and they are not counted as parameters.
        # persistent=False means they are NOT in state_dict -- correct here, since
        # __init__ rebuilds them deterministically from the config.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale residual-path output projections by 1/sqrt(2 * n_layers). Without it the
        # residual stream variance grows with depth and deep models start unstable.
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Normal(0, 0.02) init for Linear and Embedding; zero bias."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model.

        Args:
            idx: ``[B, T]`` input token ids.
            targets: Optional ``[B, T]`` next-token targets.

        Returns:
            ``(logits, loss)``; ``loss`` is None when targets are not given.
        """
        _, t = idx.shape
        if t > self.cfg.seq_len:
            raise ValueError(f"sequence length {t} exceeds configured {self.cfg.seq_len}")

        x = self.embed(idx)
        cos, sin = self.rope_cos, self.rope_sin
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        return logits, loss

    def num_parameters(self, *, embedding: bool = True) -> int:
        """Parameter count.

        Args:
            embedding: When False, exclude the token embedding. Reporting both matters at
                this scale — a tied 50257x384 table is ~19M, which can be most of the
                model, so a single headline number is easy to misread.
        """
        total = sum(p.numel() for p in self.parameters())
        if not embedding:
            total -= self.embed.weight.numel()
        return total

    @torch.no_grad()
    def estimate_perplexity(self, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Token-weighted perplexity over held-out batches.

        Weighted by token count rather than averaging per-batch losses, so a short final
        batch cannot skew the result.
        """
        was_training = self.training
        self.eval()
        total_nll, total_tokens = 0.0, 0
        for inputs, targets in batches:
            _, loss = self(inputs, targets)
            assert loss is not None
            n = targets.numel()
            total_nll += loss.item() * n
            total_tokens += n
        if was_training:
            self.train()
        return math.exp(total_nll / max(total_tokens, 1))
