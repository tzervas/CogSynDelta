"""Text encoder: token ids to a point on the shared ``[B, D]`` stream.

WHY THIS EXISTS
The region corpora are pair-shaped -- (docstring, code), (premise, hypothesis),
(query, passage). Training a region on them needs something that turns text into a single
vector so two texts can be compared. Nothing in the repo did that: the causal LM produces
per-token logits, and the only prior "text encoder" was a Blake2 hash.

This is deliberately NOT a language model. It never predicts a next token. Its whole job
is to place a piece of text somewhere on the shared stream such that related texts land
near each other -- which is exactly what a `code`, `compress`, or `retrieve` region needs,
and what CSD-BRAIN-REGIONS.md specifies when it says the `code` region does
"docstring <-> function (search + gen), **not** next-token over all GitHub".

DESIGN
Bidirectional attention, because an encoder should see the whole input; causal masking
here would cripple it for no reason. Mean pooling over non-padding positions rather than a
CLS token: CLS needs training signal to become meaningful and is worse at small scale,
while mean pooling works immediately and costs no parameters.

Padding is masked out of BOTH attention and the pool. Mean-pooling over padding is a
classic silent bug -- it makes every short text drift toward the same vector, which looks
like the model "learning similarity" and is actually just averaging in a constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# A generic pre-norm bidirectional transformer block. It lives in the vision module
# because that is where it was first needed; it is not vision-specific.
from cogsyndelta.model.vl_jepa import ViTBlock, sincos_pos_embed


@dataclass
class TextEncoderConfig:
    """Shape of the text encoder."""

    vocab_size: int = 50257
    dim: int = 384
    depth: int = 4
    n_heads: int = 6
    max_len: int = 256
    out_dim: int | None = None
    """Projection width. None keeps ``dim``; set it to match the mind's stream_dim."""

    def __post_init__(self) -> None:
        """Reject a config whose dim is not divisible by n_heads."""
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim {self.dim} must be divisible by n_heads {self.n_heads}")


class TextEncoder(nn.Module):
    """Encode token ids to a single ``[B, D]`` vector on the shared stream."""

    # register_buffer types the attribute as Tensor | Module, so indexing it fails type
    # checking. Declaring it narrows the type without changing behaviour.
    pos_embed: torch.Tensor

    def __init__(self, cfg: TextEncoderConfig, name: str = "text_encoder") -> None:
        """Build embedding, blocks, norm and the optional output projection."""
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.register_buffer("pos_embed", sincos_pos_embed(cfg.max_len, cfg.dim), persistent=False)
        self.blocks = nn.ModuleList(ViTBlock(cfg.dim, cfg.n_heads) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.dim)
        out = cfg.out_dim or cfg.dim
        self.proj = nn.Linear(cfg.dim, out, bias=False) if out != cfg.dim else nn.Identity()
        self.out_dim = out
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Normal(0, 0.02) for Linear and Embedding; zero bias."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode to ``[B, out_dim]``.

        Args:
            input_ids: ``[B, T]`` token ids.
            attention_mask: ``[B, T]`` with 1 for real tokens, 0 for padding. When None
                every position is treated as real.

        Returns:
            ``[B, out_dim]`` unnormalised embeddings. Normalisation is the caller's
            choice -- a contrastive loss wants unit vectors, a regression head may not.
        """
        b, t = input_ids.shape
        if t > self.cfg.max_len:
            raise ValueError(f"sequence length {t} exceeds max_len {self.cfg.max_len}")

        h = self.embed(input_ids) + self.pos_embed[:, :t]
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)

        if attention_mask is None:
            pooled = h.mean(dim=1)
        else:
            # Mask BEFORE summing. Averaging over padding pulls every short text toward
            # the same vector, which reads as "the model learned similarity" and is in
            # fact averaging in a constant.
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)

        return self.proj(pooled)

    def encode(self, inputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """StreamEncoder role: map text to ``[B, D]``.

        Args:
            inputs: Either ``input_ids`` or an ``(input_ids, attention_mask)`` pair.

        Returns:
            ``[B, out_dim]``.
        """
        if isinstance(inputs, tuple):
            return self.forward(*inputs)
        return self.forward(inputs)


def info_nce(
    anchors: torch.Tensor, positives: torch.Tensor, temperature: float = 0.05
) -> tuple[torch.Tensor, dict[str, float]]:
    """Symmetric InfoNCE over in-batch negatives.

    Each anchor's positive is its pair; every other item in the batch is a negative. This
    is what makes a pair corpus trainable without mining hard negatives, and why batch
    size matters more here than in a next-token objective -- the number of negatives IS
    the batch size.

    Symmetric (both directions averaged) because retrieval is used both ways: find the
    function for a docstring, and the docstring for a function. Training one direction
    only produces an encoder that is good at one and mediocre at the other.

    Args:
        anchors: ``[B, D]``.
        positives: ``[B, D]``, aligned with ``anchors``.
        temperature: Softmax temperature. Fixed rather than learned -- a learned
            temperature can shrink toward zero and drive the loss down without improving
            the ranking, which is a quiet way to fake progress.

    Returns:
        ``(loss, stats)``. Stats carry in-batch retrieval accuracy, which is the honest
        signal: a collapsed encoder has low loss and chance-level accuracy.
    """
    if anchors.shape != positives.shape:
        raise ValueError(f"shape mismatch: {tuple(anchors.shape)} vs {tuple(positives.shape)}")
    if anchors.size(0) < 2:
        raise ValueError("InfoNCE needs at least 2 examples: with one there are no negatives")

    a = F.normalize(anchors, dim=-1)
    p = F.normalize(positives, dim=-1)
    logits = (a @ p.T) / temperature
    labels = torch.arange(a.size(0), device=a.device)

    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    with torch.no_grad():
        acc = (logits.argmax(dim=-1) == labels).float().mean().item()
        # Chance is 1/B. Accuracy at chance with a falling loss means collapse.
        stats = {
            "loss": loss.item(),
            "in_batch_acc": acc,
            "chance": 1.0 / a.size(0),
            "emb_std": a.std(dim=0).mean().item(),
        }
    return loss, stats
