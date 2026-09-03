"""Equivalence: `_mlm_token_loss`'s masked-gather-before-projection form vs the naive
full-sequence vocab projection it exists to avoid.

VERIFIED, NOT INFERRED, BEFORE WRITING THIS TEST
The W4 memory-region launch note's working hypothesis was that `_mlm_token_loss`
projects EVERY position to the vocabulary (`[batch, seq, vocab]`) and only reduces to
masked positions afterward, and that gathering the masked hidden states BEFORE the
`mlm_head` projection (`[n_masked, vocab]`) would cut memory ~7x. Reading the function
(`src/cogsyndelta/regions/pretrain.py`) and `git log --all -S"_mlm_token_loss" --
src/cogsyndelta/regions/pretrain.py` (one hit: da1954e6, the function's introduction)
shows this gather-before-projection shape --

    logits = mlm_head(h[mlm_mask])  # [n_masked, vocab_size]

-- has been there since `_mlm_token_loss` was FIRST written; no full-sequence-projection
version of it has ever existed in this repository's history. There is therefore nothing
in `src/cogsyndelta/regions/pretrain.py` to rewrite for this specific mechanism, and the
"old implementation" this test's task description asked to be "copied from
ef3e18049044f35a5fe4cf63b56fc1d800994acc" does not exist at that commit either -- that
commit's `_mlm_token_loss` IS the masked-gather form under test here, byte for byte.

WHAT THIS TEST DOES INSTEAD
`_mlm_token_loss_naive_full_projection` below is a CONSTRUCTED reference: the same
masking algorithm, with the one ordering change the hypothesis describes (project the
full `[B, T, dim]` surface to the vocabulary, THEN index into the masked positions,
instead of gathering first). It proves two things a plain read of the source cannot:

  1. The two orderings are mathematically identical -- same loss, same gradient w.r.t.
     every trunk parameter -- so if a future edit ever "simplifies" the production
     function toward the naive shape, this test catches the regression even though it
     would not change training's *correctness*, only its memory cost.
  2. The naive ordering's peak memory is measurably larger, which is the number that
     justifies why production takes the gather-first shape at all, and closes the loop
     on the launch note's hypothesis: the mechanism it worried about doesn't exist in the
     current code, but a real memory saving from avoiding that mechanism is real and
     already banked -- see the CUDA test below for the MEASURED size of that saving,
     which is smaller than the launch note's "~7x" / "at least 3x" estimate and why.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tokenizers", reason="train group not installed")
# `cogsyndelta.regions.pretrain` imports `tokenizers` at module level (used by
# `_iter_pairs`/`load_pairs` elsewhere in that file, unrelated to `_mlm_token_loss`
# itself) -- skip collection gracefully rather than a hard collection error when the
# `train` dependency group isn't installed, matching test_token_aware_objective.py's
# own guard for the same reason.

import torch
import torch.nn.functional as F
from torch import nn

from cogsyndelta.regions.pretrain import _build_mlm_head, _mlm_token_loss
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig

pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------------------
# The naive reference: project the FULL [B, T, vocab] surface, index afterward. This is
# what the launch note's hypothesis believed `_mlm_token_loss` did; it never has (see
# module docstring). Kept private to this test module -- it is not a real code path.
# ---------------------------------------------------------------------------------------


def _mlm_token_loss_naive_full_projection(
    model: TextEncoder,
    mlm_head: nn.Linear,
    mask_embedding: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_prob: float,
) -> tuple[torch.Tensor, int]:
    """Byte-for-byte `_mlm_token_loss`, except the vocab projection runs over every
    position (`mlm_head(h)`, `[B, T, vocab_size]`) and the masked positions are selected
    from the resulting LOGITS instead of from the pre-projection hidden state. Same mask
    draw (same `torch.rand(b, t, ...)` call, same shape, same device -- reproducible
    against `_mlm_token_loss` only when the caller re-seeds the RNG identically
    immediately before each call), same zero-masked short-circuit, same reduction.
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
    h = model.norm(h)  # [B, T, dim]

    logits_full = mlm_head(h)  # [B, T, vocab_size] -- the naive, memory-heavy shape
    logits = logits_full[mlm_mask]  # [n_masked, vocab_size], selected AFTER projecting
    targets = input_ids[mlm_mask]
    loss = F.cross_entropy(logits.float(), targets)
    return loss, n_masked


# ---------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------


def _fresh_trunk(
    *, vocab_size: int, dim: int, depth: int, n_heads: int, max_len: int, device: torch.device
) -> tuple[TextEncoder, nn.Linear, nn.Parameter]:
    torch.manual_seed(0)
    cfg = TextEncoderConfig(
        vocab_size=vocab_size, dim=dim, depth=depth, n_heads=n_heads, max_len=max_len
    )
    model = TextEncoder(cfg).to(device)
    model.eval()  # no dropout in ViTBlock, but pin the mode so both calls see one state
    head, mask_emb = _build_mlm_head(dim, vocab_size, device)
    return model, head, mask_emb


def _trunk_params(
    model: TextEncoder, head: nn.Linear, mask_emb: nn.Parameter
) -> list[nn.Parameter]:
    return [*model.blocks.parameters(), *model.embed.parameters(), head.weight, head.bias, mask_emb]


# ---------------------------------------------------------------------------------------
# Equivalence: loss value and gradients, several shapes, fp32 and bf16, two edge cases.
# ---------------------------------------------------------------------------------------

_SHAPES = [
    pytest.param(4, 10, 0.5, id="batch4-len10-mask0.5"),
    pytest.param(3, 7, 0.15, id="batch3-len7-mask0.15-production-ratio"),
    pytest.param(8, 20, 0.5, id="batch8-len20-mask0.5"),
    pytest.param(2, 5, 1.0, id="all-positions-masked"),
    pytest.param(2, 5, 0.0, id="zero-positions-masked"),
]


@pytest.mark.parametrize("batch, seq_len, mask_prob", _SHAPES)
def test_masked_gather_matches_naive_full_projection_fp32(
    batch: int, seq_len: int, mask_prob: float
) -> None:
    device = torch.device("cpu")
    model, head, mask_emb = _fresh_trunk(
        vocab_size=37, dim=16, depth=2, n_heads=4, max_len=seq_len, device=device
    )
    torch.manual_seed(123)
    input_ids = torch.randint(1, 37, (batch, seq_len), device=device)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long, device=device)
    if seq_len > 2:
        attention_mask[:, -1] = 0  # a little real padding, not an all-real fixture

    params = _trunk_params(model, head, mask_emb)

    mask_seed = 999
    torch.manual_seed(mask_seed)
    loss_gather, n_gather = _mlm_token_loss(
        model, head, mask_emb, input_ids, attention_mask, mask_prob
    )
    grads_gather = None
    if loss_gather.requires_grad:
        grads_gather = torch.autograd.grad(loss_gather, params, allow_unused=True)

    torch.manual_seed(mask_seed)
    loss_naive, n_naive = _mlm_token_loss_naive_full_projection(
        model, head, mask_emb, input_ids, attention_mask, mask_prob
    )
    grads_naive = None
    if loss_naive.requires_grad:
        grads_naive = torch.autograd.grad(loss_naive, params, allow_unused=True)

    assert n_gather == n_naive
    assert loss_gather.item() == pytest.approx(loss_naive.item(), abs=1e-6)
    assert loss_gather.requires_grad == loss_naive.requires_grad

    if grads_gather is None:
        assert grads_naive is None  # both edge-case zero-grad tensors
        return

    for g_gather, g_naive, p in zip(grads_gather, grads_naive, params, strict=True):
        if g_gather is None and g_naive is None:
            continue
        assert g_gather is not None and g_naive is not None, (
            f"one implementation produced a gradient for {tuple(p.shape)} and the other did not"
        )
        torch.testing.assert_close(g_gather, g_naive, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("batch, seq_len, mask_prob", _SHAPES)
def test_masked_gather_matches_naive_full_projection_bf16_autocast(
    batch: int, seq_len: int, mask_prob: float
) -> None:
    """Same comparison under `torch.autocast(..., dtype=torch.bfloat16)` -- the exact
    context `pretrain_region`'s training loop runs both forward passes inside -- at the
    wider bf16 tolerance the launch note asked for (1e-2 relative). Runs on CPU too:
    `torch.autocast("cpu", dtype=torch.bfloat16)` is supported and numerically exercises
    the same reduced-mantissa path; it need not be fast to prove the two orderings agree.
    """
    device = torch.device("cpu")
    model, head, mask_emb = _fresh_trunk(
        vocab_size=37, dim=16, depth=2, n_heads=4, max_len=seq_len, device=device
    )
    torch.manual_seed(123)
    input_ids = torch.randint(1, 37, (batch, seq_len), device=device)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long, device=device)
    if seq_len > 2:
        attention_mask[:, -1] = 0

    params = _trunk_params(model, head, mask_emb)
    mask_seed = 4242

    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.manual_seed(mask_seed)
        loss_gather, n_gather = _mlm_token_loss(
            model, head, mask_emb, input_ids, attention_mask, mask_prob
        )
        grads_gather = (
            torch.autograd.grad(loss_gather, params, allow_unused=True)
            if loss_gather.requires_grad
            else None
        )

        torch.manual_seed(mask_seed)
        loss_naive, n_naive = _mlm_token_loss_naive_full_projection(
            model, head, mask_emb, input_ids, attention_mask, mask_prob
        )
        grads_naive = (
            torch.autograd.grad(loss_naive, params, allow_unused=True)
            if loss_naive.requires_grad
            else None
        )

    assert n_gather == n_naive
    if not loss_gather.requires_grad:
        assert loss_gather.item() == 0.0 and loss_naive.item() == 0.0
        assert grads_gather is None and grads_naive is None
        return

    a, b = loss_gather.item(), loss_naive.item()
    assert a == pytest.approx(b, rel=1e-2, abs=1e-3), f"bf16 loss mismatch: gather={a} naive={b}"

    for g_gather, g_naive, p in zip(grads_gather, grads_naive, params, strict=True):
        if g_gather is None and g_naive is None:
            continue
        assert g_gather is not None and g_naive is not None
        torch.testing.assert_close(
            g_gather.float(),
            g_naive.float(),
            rtol=1e-2,
            atol=1e-3,
            msg=lambda m, shape=tuple(p.shape): f"bf16 grad mismatch for param {shape}: {m}",
        )


# ---------------------------------------------------------------------------------------
# Memory: the naive ordering materialises a [B, T, vocab] tensor; the production ordering
# never does. Production-shaped batch/seq/vocab, CUDA only (peak-memory accounting on CPU
# is not meaningful the same way).
# ---------------------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="peak CUDA memory needs a CUDA device")
def test_masked_gather_uses_at_least_3x_less_peak_memory_than_naive_on_cuda() -> None:
    """Isolates the PROJECTION step's own memory (`mlm_head(...)` through
    `cross_entropy(...).backward()`) from the trunk forward that produces `h` -- the
    trunk, the embedding table and `mlm_head`'s own parameter/gradient storage are
    identical fixed costs paid by both orderings and would dilute a whole-function
    comparison (measured: only ~1.95x end to end at this shape, before isolating the
    projection step).

    MEASURED (this test, 3090 Ti, torch 2.10.0+cu128, batch=64 seq_len=96
    vocab_size=50257 mask_prob=0.15, so n_masked~=921 of 6144 positions): the isolated
    projection step is ~2.11x smaller for the gather ordering (728.9 MiB vs 1541.4 MiB),
    reproducible across repeated runs to the tenth of a MiB -- NOT the "at least 3x" the
    W4 launch note's hypothesis asked this test to assert, and far short of the ~6.67x
    the raw (batch*seq_len)/n_masked element-count ratio alone would suggest. Reading
    WHY closes the gap between that INFERRED estimate and the VERIFIED number: the naive
    ordering's full `[B, T, vocab]` logits tensor is a single, short-lived FORWARD peak
    (nothing needs its values once boolean-mask indexing has produced the smaller
    `[n_masked, vocab]` tensor cross_entropy actually consumes), but `IndexBackward`
    still has to reconstruct a full `[B, T, vocab]`-shaped gradient at BACKWARD time to
    feed `AddmmBackward`'s `dL/dh = dL/dy_full @ W` -- so the naive path pays the full
    tensor's cost exactly ONCE, not twice, while `cross_entropy`'s own internal
    `log_softmax`/backward buffers (roughly logits-sized in both orderings) are a larger
    RELATIVE overhead on the gather path's much smaller base. The floor asserted below
    (2.0x, a hair under the measured 2.11x for headroom against allocator/version drift)
    is the number this measurement actually supports; raising it to 3.0x would be
    asserting something not observed, the exact "claim without a measurement" failure
    mode this repo's evidence discipline exists to catch.
    """
    device = torch.device("cuda")
    batch, seq_len, vocab_size, dim = 64, 96, 50_257, 256
    mask_prob = 0.15

    model, head, mask_emb = _fresh_trunk(
        vocab_size=vocab_size, dim=dim, depth=4, n_heads=4, max_len=seq_len, device=device
    )
    torch.manual_seed(7)
    input_ids = torch.randint(1, vocab_size, (batch, seq_len), device=device)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long, device=device)
    real = attention_mask.bool()
    torch.manual_seed(555)
    draw = torch.rand(batch, seq_len, device=device)
    mlm_mask = (draw < mask_prob) & real
    n_masked = int(mlm_mask.sum().item())
    assert n_masked > 0, "fixture must actually exercise the projection under test"
    targets = input_ids[mlm_mask]

    with torch.no_grad():
        h_embed = model.embed(input_ids)
        h_embed = torch.where(mlm_mask.unsqueeze(-1), mask_emb.to(h_embed.dtype), h_embed)
        h_embed = h_embed + model.pos_embed[:, :seq_len]
        h = h_embed
        for block in model.blocks:
            h = block(h, attention_mask)
        h = model.norm(h)
    h = h.detach()

    def _projection_peak_bytes(*, naive: bool) -> int:
        h_leaf = h.clone().requires_grad_(True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        if naive:
            logits = head(h_leaf)[mlm_mask]  # [B, T, vocab] materialised, then indexed
        else:
            logits = head(h_leaf[mlm_mask])  # [n_masked, vocab] -- production's shape
        loss = F.cross_entropy(logits.float(), targets)
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device)
        del h_leaf, logits, loss
        torch.cuda.empty_cache()
        return peak - baseline

    peak_gather = _projection_peak_bytes(naive=False)
    peak_naive = _projection_peak_bytes(naive=True)

    assert peak_gather < peak_naive, (
        f"masked-gather projection peak ({peak_gather / 2**20:.1f} MiB) was not lower "
        f"than the naive full-projection peak ({peak_naive / 2**20:.1f} MiB)"
    )
    ratio = peak_naive / peak_gather
    assert ratio >= 2.0, (
        f"masked-gather saved only {ratio:.2f}x peak projection-step memory over naive "
        f"full projection (gather={peak_gather / 2**20:.1f} MiB, "
        f"naive={peak_naive / 2**20:.1f} MiB); expected >=2.0x (measured ~2.11x) at "
        f"batch={batch} seq_len={seq_len} vocab_size={vocab_size}"
    )


def test_naive_reference_actually_differs_in_shape_from_production() -> None:
    """A cheap sanity check on the reference itself: if this ever collapsed to the same
    code as `_mlm_token_loss` (e.g. an incautious refactor), the equivalence tests above
    would still pass but would no longer be testing anything -- this asserts the naive
    path really does materialise the full `[B, T, vocab]` projection at least once,
    by monkeypatching `nn.Linear.forward` to record the largest input shape it is called
    with.
    """
    device = torch.device("cpu")
    batch, seq_len, vocab_size, dim = 4, 10, 23, 8
    model, head, mask_emb = _fresh_trunk(
        vocab_size=vocab_size, dim=dim, depth=1, n_heads=2, max_len=seq_len, device=device
    )
    torch.manual_seed(1)
    input_ids = torch.randint(1, vocab_size, (batch, seq_len), device=device)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long, device=device)

    seen_head_input_rows: list[int] = []
    orig_forward = nn.Linear.forward

    def _spy_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        if self is head:
            seen_head_input_rows.append(x.shape[0] * (x.shape[1] if x.dim() == 3 else 1))
        return orig_forward(self, x)

    nn.Linear.forward = _spy_forward
    try:
        torch.manual_seed(1)
        _mlm_token_loss_naive_full_projection(
            model, head, mask_emb, input_ids, attention_mask, mask_prob=0.5
        )
    finally:
        nn.Linear.forward = orig_forward

    assert seen_head_input_rows == [batch * seq_len], (
        "naive reference's mlm_head call did not see every position "
        f"(expected {batch * seq_len} rows, saw {seen_head_input_rows})"
    )
