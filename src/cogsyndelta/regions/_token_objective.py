"""§4.0's token-aware retrain objective: `L_token` (masked-token prediction, attached at
the FINAL block per W1d) and `L_decorr` (a covariance-decorrelation penalty on the same
surface). Both are opt-in from `regions/pretrain.py`'s `PretrainConfig`
(`token_loss_weight`/`decorr_weight`, default 0.0) and both read from
`TextEncoder.tokens()` -- the pre-pool, post-norm token matrix W0 added for exactly this
purpose -- rather than from `forward()`'s pooled output, which is the surface InfoNCE
alone can never put a gradient on directly (§4.0: "a gradient AT EVERY POSITION, which
InfoNCE structurally cannot supply").

LIVES HERE, NOT IN `pretrain.py`, ON PURPOSE (N2 import hygiene, same shape as
`_checkpoint.py`)
Neither function below touches `tokenizers` -- `_mlm_token_loss` takes already-tokenised
`input_ids`/`attention_mask` tensors, never a `Tokenizer`. `pretrain.py` imports
`tokenizers` at module level for the rest of its job (building splits, encoding corpora),
so importing anything from it -- even these two tokenizer-free functions -- drags that
dependency in. Splitting them out means `tests/test_token_loss_masked_equivalence.py`'s
pure-tensor cases (construct `input_ids` with `torch.randint`, never touch a real
tokenizer) can import directly from here and run in `scripts/ci_local.sh`'s isolated venv,
which deliberately excludes the `train` dependency group (see that script's header and
`tests/test_import_hygiene.py`). `pretrain.py` re-exports these three names unchanged, so
every existing call site (`from cogsyndelta.regions.pretrain import _mlm_token_loss`, the
training loop itself) keeps working with no change of its own.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn


def _build_mlm_head(
    dim: int, vocab_size: int, device: torch.device
) -> tuple[nn.Linear, nn.Parameter]:
    """A discardable MLM head plus a learned `[MASK]` replacement vector.

    Neither is part of `TextEncoder`'s own module tree or state dict -- §4.0 describes
    masked-token prediction as "a BERT-style MLM head ... discarded after training", and
    every existing checkpoint reader (`csd-quantize.py`, `csd-benchmark.py`,
    `regions/retrieve.py`) loads `TextEncoder.state_dict()` and nothing else. A `[MASK]`
    EMBEDDING rather than a reserved vocabulary id: the GPT-2 BPE table this project uses
    has no spare id set aside for one, and adding a row would change `vocab_size` (and
    therefore every existing checkpoint's embedding shape) for a feature most regions
    never turn on.

    Returns:
        `(mlm_head, mask_embedding)`, both already moved to `device` and initialised
        Normal(0, 0.02) zero-bias -- the same convention `TextEncoder._init_weights` uses,
        so the auxiliary head starts in the same regime as the trunk it is attached to.
    """
    head = nn.Linear(dim, vocab_size).to(device)
    nn.init.normal_(head.weight, mean=0.0, std=0.02)
    nn.init.zeros_(head.bias)
    mask_embedding = nn.Parameter(torch.zeros(dim, device=device))
    nn.init.normal_(mask_embedding, mean=0.0, std=0.02)
    return head, mask_embedding


def _ce_chunk_sum(
    mlm_head: nn.Linear, h_chunk: torch.Tensor, targets_chunk: torch.Tensor
) -> torch.Tensor:
    """One chunk's contribution to the masked-token cross-entropy, as a SUM (not a
    mean) over its rows -- the caller divides the accumulated sum by the TOTAL `n_masked`
    across every chunk, so this returning a sum rather than a per-chunk mean is what
    keeps the final reduction identical to a single unchunked `F.cross_entropy(...,
    reduction="mean")` over all masked positions at once (see `_mlm_token_loss`).

    This is the function `torch.utils.checkpoint.checkpoint` recomputes during backward:
    only `h_chunk`'s already-small `[chunk, dim]` slice is kept for that recompute --
    the `[chunk, vocab_size]` logits this materialises are freed again as soon as this
    call returns, both on the initial (no-grad) forward and on each backward recompute --
    so peak memory from the vocabulary projection is `O(chunk x vocab_size)`, never
    `O(n_masked x vocab_size)`, regardless of how many total masked positions there are.
    """
    logits = mlm_head(h_chunk)  # [chunk, vocab_size]
    return F.cross_entropy(logits.float(), targets_chunk, reduction="sum")


def _chunked_masked_ce_sum(
    mlm_head: nn.Linear, masked_h: torch.Tensor, targets: torch.Tensor, chunk: int
) -> torch.Tensor:
    """Sum of per-position cross-entropy over every row of `masked_h`, computed CHUNK
    rows at a time under `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` so
    only one chunk's `[chunk, vocab_size]` logits (and `log_softmax` backward buffers) are
    ever live at once, rather than the full `[n_masked, vocab_size]` at once.

    `use_reentrant=False`: the non-reentrant checkpoint implementation builds the real
    autograd graph on each backward recompute (rather than reentering `.backward()`
    inside a custom `Function`), which is what makes the gradient this produces, w.r.t.
    both `mlm_head`'s parameters and `masked_h` (and, through it, the trunk), EXACT --
    bit-for-bit the same computation an unchunked backward would do, only spread over
    `ceil(n_masked / chunk)` smaller recomputations instead of one large one.

    Args:
        mlm_head: The MLM head, from :func:`_build_mlm_head`.
        masked_h: `[n_masked, dim]` -- already gathered to the masked positions, so this
            function's own memory footprint (as opposed to what it computes) never
            depends on `vocab_size` at all.
        targets: `[n_masked]`, the original token ids at those positions.
        chunk: Rows per chunk. Must be `> 0` and `< n_masked` -- the caller
            (`_mlm_token_loss`) takes the unchunked fast path itself whenever chunking
            would not help (`chunk <= 0` or `n_masked <= chunk`).

    Returns:
        A scalar: the SUM (not mean) of cross-entropy over every row of `masked_h`.
    """
    n = masked_h.size(0)
    total = masked_h.new_zeros(())
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        total = total + torch.utils.checkpoint.checkpoint(
            _ce_chunk_sum,
            mlm_head,
            masked_h[start:end],
            targets[start:end],
            use_reentrant=False,
        )
    return total


def _mlm_token_loss(
    model,
    mlm_head: nn.Linear,
    mask_embedding: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_prob: float,
    chunk: int = 0,
) -> tuple[torch.Tensor, int]:
    """§4.0's masked-token prediction, attached at the FINAL block.

    Replicates `TextEncoder.tokens()` (embed -> blocks -> norm) with one difference: at a
    sampled subset of REAL (non-padding) positions, the token embedding is replaced by
    `mask_embedding` before the blocks run, and the head predicts the ORIGINAL id at
    those positions from the final-block representation. Reads `model.embed`,
    `model.pos_embed` and `model.blocks` directly rather than adding a masking parameter
    to `TextEncoder.tokens()` itself -- the same sanctioned access pattern
    `docs/design/evidence/w1-token-rank-2026-09-02/measure_w1.py` already uses to capture
    the identical surface read-only, so every OTHER caller of `tokens()`/`forward()` (every
    region that never sets `token_loss_weight`) sees no change to `TextEncoder` at all.

    THE MASK DRAW ITSELF IS NEVER CHUNKED -- `torch.rand(b, t, ...)` runs exactly once,
    unconditionally, before `chunk` is even consulted, so a seeded caller gets the SAME
    `mlm_mask` (and therefore the same masked positions and the same `n_masked`)
    regardless of `chunk`'s value; only how the vocabulary projection over those already-
    chosen positions is computed changes.

    CHUNKING (`chunk > 0`): a batch-1280 probe (`docs/design/evidence/
    w4-masked-token-loss-2026-09-03/`) OOM'd inside this function's SECOND per-step call
    -- `logits = mlm_head(h[mlm_mask])` materialises `[n_masked, vocab_size]` in fp32 (at
    that batch, ~18,400 x 50,257 -- roughly 3.7 GiB, doubled again by `log_softmax`'s own
    backward buffers, twice per step for the anchor/positive sides) and CUDA autograd
    keeps it all live until `.backward()` walks back through it. `chunk > 0` processes
    `masked_h` (the gather already applied -- see below) `chunk` rows at a time through
    :func:`_chunked_masked_ce_sum`, each chunk checkpointed
    (`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`) so only one chunk's
    logits are ever resident; the returned loss is mathematically IDENTICAL to the
    unchunked `F.cross_entropy(..., reduction="mean")` this function has always computed
    (see that helper's own docstring) -- `chunk` is a memory knob, not a numerical one.
    `chunk <= 0`, or `n_masked <= chunk` (chunking would not reduce the peak this call
    makes anyway), takes the original, byte-identical unchunked path.

    Args:
        model: A `TextEncoder`.
        mlm_head: `nn.Linear(dim, vocab_size)`, from :func:`_build_mlm_head`.
        mask_embedding: `[dim]`, from :func:`_build_mlm_head`.
        input_ids: `[B, T]`.
        attention_mask: `[B, T]`, 1 for real tokens.
        mask_prob: Fraction of real positions to mask, in expectation.
        chunk: Rows of `masked_h` to project to the vocabulary at once. `0` (the
            default) disables chunking -- the original single-projection shape.

    Returns:
        `(loss, n_masked)`. `loss` is `0.0` (a zero tensor, no grad) when the sampled mask
        selects nothing -- possible on a very short batch -- so a caller can add it to the
        total loss unconditionally without special-casing an empty selection. `n_masked`
        is reported for the receipt/history, not used in the loss itself.
    """
    b, t = input_ids.shape
    real = attention_mask.bool()
    draw = torch.rand(b, t, device=input_ids.device)
    mlm_mask = (draw < mask_prob) & real
    n_masked = int(mlm_mask.sum().item())
    if n_masked == 0:
        return input_ids.new_zeros((), dtype=torch.float32), 0

    h = model.embed(input_ids)
    h = torch.where(mlm_mask.unsqueeze(-1), mask_embedding.to(h.dtype), h)
    h = h + model.pos_embed[:, :t]
    for block in model.blocks:
        h = block(h, attention_mask)
    h = model.norm(h)  # [B, T, dim] -- the FINAL block, pre-pool, post-norm

    masked_h = h[mlm_mask]  # [n_masked, dim]
    targets = input_ids[mlm_mask]

    if chunk <= 0 or n_masked <= chunk:
        logits = mlm_head(masked_h)  # [n_masked, vocab_size]
        loss = F.cross_entropy(logits.float(), targets)
        return loss, n_masked

    loss_sum = _chunked_masked_ce_sum(mlm_head, masked_h, targets, chunk)
    loss = loss_sum / n_masked
    return loss, n_masked


def _token_decorrelation_loss(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """§4.0's `L_decorr`: a VICReg/Barlow-Twins-shaped off-diagonal penalty on the
    token-position feature covariance, driving cross-position redundancy down.

    Flattens every REAL (non-padding) token position across the whole batch into one
    `[N, dim]` matrix -- the same "token-global" surface the W4/W7 rank gate measures
    (see `cogsyndelta.eval.benchmark.participation_ratio`) -- centers it, and penalizes
    the squared off-diagonal mass of its `dim x dim` feature covariance, normalised by
    `dim` so the penalty's scale does not grow with the encoder width.

    Args:
        h: `[B, T, dim]`, the FINAL block's pre-pool token representations.
        mask: `[B, T]`, 1 for real positions.

    Returns:
        A scalar loss, `0.0` (no grad) when fewer than 2 real positions survive.
    """
    flat = h[mask.bool()].float()
    if flat.size(0) < 2:
        return h.new_zeros(())
    flat = flat - flat.mean(dim=0, keepdim=True)
    n = flat.size(0)
    dim = flat.size(1)
    cov = (flat.T @ flat) / max(1, n - 1)
    off_diag_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag_sq / dim
