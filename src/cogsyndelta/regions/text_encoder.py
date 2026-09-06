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

    def tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Region-native representation BEFORE pooling and BEFORE ``proj`` (DEC-14/DEC-15).

        This is everything ``forward`` used to do up to (and including) the final norm --
        unchanged, weight for weight -- just no longer thrown away one line later. The
        pooling line moved to :meth:`pool`; nothing about the blocks or the attention mask
        handling below changed.

        Args:
            input_ids: ``[B, T]`` token ids.
            attention_mask: ``[B, T]`` with 1 for real tokens, 0 for padding. When None
                every position is treated as real.

        Returns:
            ``(h [B, T, dim], mask [B, T])``. ``mask`` is ``attention_mask`` when given,
            else an all-ones mask of the same shape -- :meth:`pool` always receives an
            explicit mask, so "no padding" and "masked mean over an all-real batch" are
            the same code path rather than two.
        """
        b, t = input_ids.shape
        if t > self.cfg.max_len:
            raise ValueError(f"sequence length {t} exceeds max_len {self.cfg.max_len}")

        h = self.embed(input_ids) + self.pos_embed[:, :t]
        for block in self.blocks:
            # The mask goes into ATTENTION, not just the pool. Without it every real token
            # attends to padding, so the same sentence encodes differently depending on how
            # much padding its batch happened to carry -- measured at cosine 0.958 between
            # identical inputs padded to 6 and 66 positions. _tokenize pads to the longest
            # item in the batch, so that made every result batch-composition dependent.
            h = block(h, attention_mask)
        h = self.norm(h)

        mask = (
            torch.ones(b, t, dtype=h.dtype, device=h.device)
            if attention_mask is None
            else attention_mask
        )
        return h, mask

    def pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``[B, T, dim] -> [B, out_dim]``: masked mean, then ``proj``.

        The region's standalone answer (DEC-14/DEC-15), kept so every existing receipt
        stays reproducible and comparable. Same masked-mean expression ``forward`` used to
        contain -- moved here, not rewritten.

        Args:
            h: ``[B, T, dim]`` token representations, as returned by :meth:`tokens`.
            mask: ``[B, T]`` with 1 for real positions, 0 for padding.

        Returns:
            ``[B, out_dim]`` unnormalised embeddings. Normalisation is the caller's
            choice -- a contrastive loss wants unit vectors, a regression head may not.
        """
        # Mask BEFORE summing. Averaging over padding pulls every short text toward
        # the same vector, which reads as "the model learned similarity" and is in
        # fact averaging in a constant.
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)
        return self.proj(pooled)

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
        h, mask = self.tokens(input_ids, attention_mask)
        return self.pool(h, mask)

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
    anchors: torch.Tensor,
    positives: torch.Tensor,
    temperature: float = 0.05,
    extra_negatives: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Symmetric InfoNCE over in-batch negatives, plus any negatives from beyond the batch.

    Each anchor's positive is its pair; every other item in the batch is a negative. This
    is what makes a pair corpus trainable without mining hard negatives, and why batch
    size matters more here than in a next-token objective -- the number of negatives IS
    the batch size.

    Symmetric (both directions averaged) because retrieval is used both ways: find the
    function for a docstring, and the docstring for a function. Training one direction
    only produces an encoder that is good at one and mediocre at the other.

    The loss is computed in fp32 whatever dtype the encoders produced. That is not
    defensive habit; it is required once the forward runs under bf16 autocast, and the
    comment on the cast below says exactly why.

    NEGATIVES BEYOND THE BATCH, AND WHY THE SYMMETRIC TERM DOES NOT SEE THEM
    `extra_negatives` appends `K` extra COLUMNS to the logits, so the anchor-to-positive
    direction ranks its gold against `B - 1 + K` distractors instead of `B - 1`. The
    transposed direction cannot take them: `logits.T` would be `[B + K, B]`, whose extra
    rows are negatives with no anchor of their own and no label to point at. So the
    symmetric term stays restricted to the `[B, B]` in-batch block --
    `0.5*(CE(logits, labels) + CE(logits[:, :B].T, labels))` -- which at `K = 0` is
    `logits[:, :B] is logits`, i.e. exactly the line this function has always computed.
    That identity is the point rather than a convenience: the in-batch-only control arm
    of a negatives experiment must be a special case of the treatment code, not a second
    implementation of it, or the arms differ in two things. `tests/test_negatives.py`
    asserts the bit-identity rather than arguing for it.
    (Pre-registration: PREREG-RETRIEVAL-NEGATIVES-2026-09-06 rev 3, sections 3 and 2.)

    Args:
        anchors: ``[B, D]``.
        positives: ``[B, D]``, aligned with ``anchors``.
        temperature: Softmax temperature. Fixed rather than learned -- a learned
            temperature can shrink toward zero and drive the loss down without improving
            the ranking, which is a quiet way to fake progress.
        extra_negatives: Optional ``[K, D]`` of encoded texts that are negatives for
            every anchor -- a cross-batch bank (:class:`NegativeBank`) or mined hard
            negatives. DETACHED here whatever the caller passes, so they enter the
            denominator without contributing a gradient of their own; an empty tensor is
            treated as absent, which is what makes a warm-up step with an empty bank
            bit-identical to the control.

    Returns:
        ``(loss, stats)``. Stats carry in-batch retrieval accuracy, which is the honest
        signal: a collapsed encoder has low loss and chance-level accuracy, plus the
        whole-row `full_acc` and its own `full_chance` denominator -- reporting an
        accuracy over `B + K` columns against `1/B` would read as a collapse that is
        really just a bigger denominator.

    Raises:
        ValueError: If the two towers disagree in shape, if there is only one example
            (no negatives to speak of), or if `extra_negatives` is not ``[K, D]`` with
            the same ``D``.
    """
    if anchors.shape != positives.shape:
        raise ValueError(f"shape mismatch: {tuple(anchors.shape)} vs {tuple(positives.shape)}")
    if anchors.size(0) < 2:
        raise ValueError("InfoNCE needs at least 2 examples: with one there are no negatives")
    if extra_negatives is not None and extra_negatives.numel():
        if extra_negatives.dim() != 2 or extra_negatives.size(1) != anchors.size(1):
            raise ValueError(
                f"extra_negatives must be [K, {anchors.size(1)}], got "
                f"{tuple(extra_negatives.shape)}"
            )

    # fp32 FOR THE LOSS, ALWAYS -- including under a bf16 autocast around the encoders.
    # bf16 carries 8 mantissa bits, and these logits are divided by temperature=0.05,
    # i.e. multiplied by 20, so a unit-cosine logit lands near +-20 where the bf16 quantum
    # is ~0.125. A softmax over a batch of 1,280 classes on logits that coarse is a real
    # perturbation of the ranking, and it would surface as a quiet recall loss that looks
    # exactly like seed variance. `F.cross_entropy` is on autocast's fp32 promote-list and
    # upcasts itself; the matmul feeding it does NOT, which is the half that has to be
    # forced here. It also keeps `emb_std` -- the collapse signal -- an fp32 number.
    #
    # The cost is nil twice over: `.float()` on an fp32 tensor returns it unchanged, so
    # the fp32 path is bit-identical to before this line existed, and at batch 1,280 the
    # matmul is 0.4 GFLOP against a step that already does hundreds.
    a = F.normalize(anchors.float(), dim=-1)
    p = F.normalize(positives.float(), dim=-1)
    logits = (a @ p.T) / temperature
    labels = torch.arange(a.size(0), device=a.device)
    if extra_negatives is not None and extra_negatives.numel():
        # `.detach()` here as well as at the call site: these columns are a denominator,
        # never a second training signal. A bank vector was encoded by an older copy of
        # the weights and a mined negative is deliberately no-grad, so a gradient
        # reaching either would be training on a stale or unintended path.
        n = F.normalize(extra_negatives.detach().float(), dim=-1)
        logits = torch.cat([logits, (a @ n.T) / temperature], dim=1)

    # `logits[:, :B]` is the whole tensor when there are no extra columns -- a narrow
    # over the full width returns the same storage, sizes and strides -- so this line is
    # the pre-existing `0.5*(CE(logits, labels) + CE(logits.T, labels))` bit for bit at
    # K = 0, and the control arm needs no separate code path.
    in_batch = logits[:, : a.size(0)]
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(in_batch.T, labels))

    with torch.no_grad():
        # Over the [B, B] BLOCK, not the whole row: once extra columns exist, an argmax
        # over the row answers a different question ("did the gold beat B-1+K
        # distractors") and would silently redefine a statistic that appears in every
        # receipt this project has written. That question is `full_acc`, reported
        # alongside with its own denominator.
        acc = (in_batch.argmax(dim=-1) == labels).float().mean().item()
        # Chance is 1/B. Accuracy at chance with a falling loss means collapse.
        stats = {
            "loss": loss.item(),
            "in_batch_acc": acc,
            "chance": 1.0 / a.size(0),
            "full_acc": (logits.argmax(dim=-1) == labels).float().mean().item(),
            "full_chance": 1.0 / logits.size(1),
            "negatives_per_query": float(logits.size(1) - 1),
            "emb_std": a.std(dim=0).mean().item(),
        }
    return loss, stats


class NegativeBank:
    """A FIFO bank of the last ``capacity`` encoded positives, kept detached.

    WHY A BANK AT ALL
    In-batch InfoNCE gives each anchor `B - 1` negatives and nothing else, so on a
    57,638-passage retrieval pool no gradient ever separates the gold from the 56,000-odd
    passages the batch never showed. A bank re-uses positives already encoded on previous
    steps as extra denominator columns: at `capacity = 16384` and batch 1,280 that is
    17,663 negatives per query instead of 1,279, for the cost of one `16384 x 256` fp32
    buffer (16.0 MiB) and a wider logits matmul.

    NO MOMENTUM ENCODER, DELIBERATELY
    MoCo pairs a bank with a slowly-updated key encoder because its vectors go stale.
    This is the plain version: vectors are whatever the encoder produced when they were
    pushed, and staleness is a property of the arm being measured, not something a second
    encoder is introduced to hide. Adding one would be a second variable in a round whose
    whole design is one variable (PREREG-RETRIEVAL-NEGATIVES-2026-09-06 rev 3, Table 2).

    Attributes:
        capacity: `K`, the number of vectors retained. The oldest is evicted first.
    """

    def __init__(self, capacity: int, device: torch.device | str | None = None) -> None:
        """Create an empty bank.

        The buffer is allocated on the first push rather than here, so the embedding
        width comes from the encoder that actually ran instead of from a config field
        that could disagree with it.

        Args:
            capacity: `K`, maximum vectors retained. Must be positive.
            device: Where the buffer lives. Defaults to the device of the first push.

        Raises:
            ValueError: If ``capacity`` is not positive -- "a bank of zero" is spelled by
                not constructing one, so that the control arm has no bank object at all.
        """
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self._device = torch.device(device) if device is not None else None
        self._buffer: torch.Tensor | None = None
        self._cursor = 0
        self._filled = 0

    def __len__(self) -> int:
        """Vectors currently held, at most ``capacity``."""
        return self._filled

    def push(self, vectors: torch.Tensor) -> None:
        """Append encoded positives, evicting the oldest once full.

        Args:
            vectors: ``[N, D]``. Detached and cast to fp32 before storage -- fp32 because
                the bank outlives the autocast region that produced these vectors, and a
                bf16 buffer would quantise a stored logit source to ~0.125 near the
                temperature-scaled range (see `info_nce`'s own fp32 comment).

        Raises:
            ValueError: If ``vectors`` is not ``[N, D]``, or its width disagrees with
                what the bank already holds.
        """
        if vectors.dim() != 2:
            raise ValueError(f"expected [N, D], got {tuple(vectors.shape)}")
        rows = vectors.detach().float()
        if self._buffer is None:
            device = self._device if self._device is not None else rows.device
            self._buffer = torch.zeros(self.capacity, rows.size(1), dtype=torch.float32)
            self._buffer = self._buffer.to(device)
        elif rows.size(1) != self._buffer.size(1):
            raise ValueError(f"bank holds width {self._buffer.size(1)}, got {rows.size(1)}")
        rows = rows.to(self._buffer.device)
        # More rows than the bank holds: only the last `capacity` of them survive, which
        # is what a FIFO of this size would contain after pushing them one at a time.
        if rows.size(0) > self.capacity:
            rows = rows[-self.capacity :]
        first = self.capacity - self._cursor
        head = rows[:first]
        self._buffer[self._cursor : self._cursor + head.size(0)] = head
        tail = rows[first:]
        if tail.numel():
            self._buffer[: tail.size(0)] = tail
        self._cursor = (self._cursor + rows.size(0)) % self.capacity
        self._filled = min(self.capacity, self._filled + rows.size(0))

    def negatives(self) -> torch.Tensor:
        """The vectors held, as ``[len(self), D]``.

        Returns:
            A detached view of the filled part of the ring, in no particular order --
            softmax columns are exchangeable, so only WHICH vectors are present matters,
            never their order. Empty (``[0, 0]``) before the first push, which
            `info_nce` treats as "no extra negatives" and therefore as the control.
        """
        if self._buffer is None or self._filled == 0:
            return torch.zeros(0, 0)
        return self._buffer[: self._filled].detach()
