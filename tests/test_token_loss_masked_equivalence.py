"""Equivalence: `_mlm_token_loss`'s masked-gather-before-projection form vs the naive
full-sequence vocab projection it exists to avoid, AND its chunked-projection form
(`chunk > 0`) vs the original unchunked one.

VERIFIED, NOT INFERRED, BEFORE WRITING THE GATHER-VS-NAIVE TESTS
The W4 memory-region launch note's working hypothesis was that `_mlm_token_loss`
projects EVERY position to the vocabulary (`[batch, seq, vocab]`) and only reduces to
masked positions afterward, and that gathering the masked hidden states BEFORE the
`mlm_head` projection (`[n_masked, vocab]`) would cut memory ~7x. Reading the function
(`src/cogsyndelta/regions/_token_objective.py`, moved there from
`src/cogsyndelta/regions/pretrain.py` -- see below) and `git log --all
-S"_mlm_token_loss" -- src/cogsyndelta/regions/pretrain.py` (one hit: da1954e6, the
function's introduction) shows this gather-before-projection shape --

    logits = mlm_head(masked_h)  # [n_masked, vocab_size]

-- has been there since `_mlm_token_loss` was FIRST written; no full-sequence-projection
version of it has ever existed in this repository's history. `_mlm_token_loss_naive_full_
projection` below is a CONSTRUCTED reference (the shape the hypothesis worried about),
not a literal prior implementation -- see the two tests built on it for what it proves.

WHY THIS MODULE NO LONGER GUARDS ITS IMPORTS WITH `pytest.importorskip("tokenizers")`
Every function this file tests -- `_build_mlm_head`, `_mlm_token_loss`,
`_chunked_masked_ce_sum` -- now lives in `cogsyndelta.regions._token_objective`, which
imports only `torch` (see that module's own docstring for why it was split out of
`pretrain.py`: N2 import hygiene, the same shape as `_checkpoint.py` /
`tests/test_import_hygiene.py`). `TextEncoder`/`TextEncoderConfig`
(`cogsyndelta.regions.text_encoder`) are equally tokenizer-free (`torch` +
`cogsyndelta.model.vl_jepa`, itself `torch`-only). Every input here is built directly
with `torch.randint`/`torch.ones` -- never a real `Tokenizer` -- so nothing in this file
has ever needed `tokenizers` to be installed; the guard on the OLD version of this file
was collateral from importing through `cogsyndelta.regions.pretrain` (which genuinely
does need it, for `Tokenizer`-typed corpus-building code unrelated to the loss function
under test here) rather than from any real dependency of what these tests exercise. This
is why the whole module used to be skipped in `scripts/ci_local.sh`'s isolated venv
(`uv sync --group dev`, which excludes `tokenizers`) even though nothing in it touched a
tokenizer -- importing straight from `_token_objective`/`text_encoder` fixes that for
every test in this file, not a subset, so there is no remaining case that needs a skip.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from cogsyndelta.regions._token_objective import (
    _build_mlm_head,
    _chunked_masked_ce_sum,
    _mlm_token_loss,
)
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
    """Byte-for-byte `_mlm_token_loss` at `chunk=0`, except the vocab projection runs
    over every position (`mlm_head(h)`, `[B, T, vocab_size]`) and the masked positions
    are selected from the resulting LOGITS instead of from the pre-projection hidden
    state. Same mask draw (same `torch.rand(b, t, ...)` call, same shape, same device --
    reproducible against `_mlm_token_loss` only when the caller re-seeds the RNG
    identically immediately before each call), same zero-masked short-circuit, same
    reduction.
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


def _assert_losses_and_grads_match(
    loss_a: torch.Tensor,
    grads_a: tuple[torch.Tensor | None, ...] | None,
    loss_b: torch.Tensor,
    grads_b: tuple[torch.Tensor | None, ...] | None,
    params: list[nn.Parameter],
    *,
    rtol: float,
    atol: float,
) -> None:
    """Shared assertion body for every fp32/bf16 equivalence test below (gather-vs-naive
    and chunked-vs-unchunked alike): same loss value, same `requires_grad`-ness, and
    every gradient (trunk blocks, embedding table, MLM head weight/bias, mask embedding)
    equal within tolerance -- or both `None` together on the zero-masked edge case."""
    assert loss_a.requires_grad == loss_b.requires_grad
    if not loss_a.requires_grad:
        assert grads_a is None and grads_b is None
        return
    assert grads_a is not None and grads_b is not None
    for ga, gb, p in zip(grads_a, grads_b, params, strict=True):
        if ga is None and gb is None:
            continue
        assert ga is not None and gb is not None, (
            f"one implementation produced a gradient for {tuple(p.shape)} and the other did not"
        )
        torch.testing.assert_close(ga.float(), gb.float(), rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------------------
# Equivalence: loss value and gradients, several shapes, fp32 and bf16, two edge cases.
# gather-before-projection (chunk=0) vs the naive full-[B,T,vocab] reference.
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
    assert loss_gather.item() == pytest.approx(loss_naive.item(), abs=1e-6)
    _assert_losses_and_grads_match(
        loss_gather, grads_gather, loss_naive, grads_naive, params, rtol=1e-6, atol=1e-6
    )


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
    _assert_losses_and_grads_match(
        loss_gather, grads_gather, loss_naive, grads_naive, params, rtol=1e-2, atol=1e-3
    )


# ---------------------------------------------------------------------------------------
# Equivalence: chunked (`chunk > 0`, `torch.utils.checkpoint.checkpoint` per chunk) vs
# unchunked (`chunk=0`) -- the memory optimisation this file's task added. Same shapes-
# and-tolerances shape as the gather-vs-naive tests above, plus shapes chosen to hit the
# boundary conditions by NAME: n_masked < chunk (the shortcut path), n_masked not a
# multiple of chunk (a ragged last chunk), and a chunk that divides n_masked evenly.
#
# mask_prob=1.0 makes `n_masked` DETERMINISTIC (every real position gets masked --
# `torch.rand(...)` draws from `[0, 1)`, so `draw < 1.0` is always true regardless of
# seed) with no padding, so `n_masked` is exactly the fixture's real-token count and the
# boundary-condition cases below hit precisely what their ids claim rather than hoping a
# random draw lands there.
# ---------------------------------------------------------------------------------------

_CHUNK_CASES = [
    # Realistic mask_prob draws -- n_masked is RNG-determined, not hand-picked; proves
    # chunking survives a real (non-deterministic-count) masked selection, same shapes
    # the gather-vs-naive tests above use.
    pytest.param(4, 10, 0.5, 3, id="batch4-len10-mask0.5-chunk3"),
    pytest.param(3, 7, 0.15, 2, id="batch3-len7-mask0.15-production-ratio-chunk2"),
    pytest.param(8, 20, 0.5, 16, id="batch8-len20-mask0.5-chunk16"),
    # Deterministic n_masked (mask_prob=1.0, no padding -- see module docstring above
    # this list), hand-picked to hit exact boundary conditions by name.
    pytest.param(1, 8, 1.0, 4, id="n_masked-8-exact-multiple-of-chunk-4"),
    pytest.param(1, 7, 1.0, 3, id="n_masked-7-not-a-multiple-of-chunk-3"),
    pytest.param(1, 5, 1.0, 100, id="n_masked-5-less-than-chunk-100"),
    # Edge cases the task named explicitly.
    pytest.param(2, 5, 1.0, 2, id="all-positions-masked-chunked"),
    pytest.param(2, 5, 0.0, 4, id="zero-positions-masked-chunked"),
]


def _chunk_fixture(
    batch: int, seq_len: int, mask_prob: float, device: torch.device
) -> tuple[TextEncoder, nn.Linear, nn.Parameter, torch.Tensor, torch.Tensor]:
    model, head, mask_emb = _fresh_trunk(
        vocab_size=37, dim=16, depth=2, n_heads=4, max_len=seq_len, device=device
    )
    torch.manual_seed(123)
    input_ids = torch.randint(1, 37, (batch, seq_len), device=device)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long, device=device)
    # Only the RNG-determined-n_masked cases get a little real padding, matching the
    # gather-vs-naive fixtures above; the mask_prob=1.0 deterministic-count cases stay
    # all-real so n_masked is exactly seq_len (padding would only shrink it, still
    # deterministic, but the case ids above promise an exact count).
    if seq_len > 2 and mask_prob < 1.0:
        attention_mask[:, -1] = 0
    return model, head, mask_emb, input_ids, attention_mask


@pytest.mark.parametrize("batch, seq_len, mask_prob, chunk", _CHUNK_CASES)
def test_chunked_matches_unchunked_fp32(
    batch: int, seq_len: int, mask_prob: float, chunk: int
) -> None:
    device = torch.device("cpu")
    model, head, mask_emb, input_ids, attention_mask = _chunk_fixture(
        batch, seq_len, mask_prob, device
    )
    params = _trunk_params(model, head, mask_emb)
    mask_seed = 999

    torch.manual_seed(mask_seed)
    loss_unchunked, n_unchunked = _mlm_token_loss(
        model, head, mask_emb, input_ids, attention_mask, mask_prob, chunk=0
    )
    grads_unchunked = (
        torch.autograd.grad(loss_unchunked, params, allow_unused=True)
        if loss_unchunked.requires_grad
        else None
    )

    torch.manual_seed(mask_seed)
    loss_chunked, n_chunked = _mlm_token_loss(
        model, head, mask_emb, input_ids, attention_mask, mask_prob, chunk=chunk
    )
    grads_chunked = (
        torch.autograd.grad(loss_chunked, params, allow_unused=True)
        if loss_chunked.requires_grad
        else None
    )

    assert n_unchunked == n_chunked
    if mask_prob == 1.0:
        assert n_unchunked == batch * seq_len, "mask_prob=1.0 fixture must mask every real position"
    elif mask_prob == 0.0:
        assert n_unchunked == 0, "mask_prob=0.0 fixture must mask nothing"
    assert loss_unchunked.item() == pytest.approx(loss_chunked.item(), abs=1e-6)
    _assert_losses_and_grads_match(
        loss_unchunked, grads_unchunked, loss_chunked, grads_chunked, params, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("batch, seq_len, mask_prob, chunk", _CHUNK_CASES)
def test_chunked_matches_unchunked_bf16_autocast(
    batch: int, seq_len: int, mask_prob: float, chunk: int
) -> None:
    """Same comparison as `test_chunked_matches_unchunked_fp32` under
    `torch.autocast(..., dtype=torch.bfloat16)` -- `torch.utils.checkpoint.checkpoint`'s
    non-reentrant recompute must reproduce the SAME autocast state on every chunk's
    backward recompute as the unchunked path saw on its one forward, or this would drift
    outside bf16 tolerance."""
    device = torch.device("cpu")
    model, head, mask_emb, input_ids, attention_mask = _chunk_fixture(
        batch, seq_len, mask_prob, device
    )
    params = _trunk_params(model, head, mask_emb)
    mask_seed = 4242

    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.manual_seed(mask_seed)
        loss_unchunked, n_unchunked = _mlm_token_loss(
            model, head, mask_emb, input_ids, attention_mask, mask_prob, chunk=0
        )
        grads_unchunked = (
            torch.autograd.grad(loss_unchunked, params, allow_unused=True)
            if loss_unchunked.requires_grad
            else None
        )

        torch.manual_seed(mask_seed)
        loss_chunked, n_chunked = _mlm_token_loss(
            model, head, mask_emb, input_ids, attention_mask, mask_prob, chunk=chunk
        )
        grads_chunked = (
            torch.autograd.grad(loss_chunked, params, allow_unused=True)
            if loss_chunked.requires_grad
            else None
        )

    assert n_unchunked == n_chunked
    if not loss_unchunked.requires_grad:
        assert loss_unchunked.item() == 0.0 and loss_chunked.item() == 0.0
        assert grads_unchunked is None and grads_chunked is None
        return

    a, b = loss_unchunked.item(), loss_chunked.item()
    assert a == pytest.approx(b, rel=1e-2, abs=1e-3), (
        f"bf16 loss mismatch: unchunked={a} chunked={b}"
    )
    _assert_losses_and_grads_match(
        loss_unchunked, grads_unchunked, loss_chunked, grads_chunked, params, rtol=1e-2, atol=1e-3
    )


def test_chunked_masked_ce_sum_matches_unchunked_cross_entropy_directly() -> None:
    """Tests `_chunked_masked_ce_sum` in isolation, one level below `_mlm_token_loss`:
    no trunk, no `TextEncoder` at all -- just a synthetic `[n, dim]` hidden-state matrix
    and an `nn.Linear` head, so this pins the helper's own contract (sum, not mean; exact
    match to `F.cross_entropy(..., reduction='sum')`) independent of anything about the
    masking or the trunk forward. `n=13`, `chunk=5` -- three chunks of 5, 5, 3 -- is the
    ragged-last-chunk case at the smallest scale that actually exercises it.
    """
    torch.manual_seed(0)
    n, dim, vocab = 13, 6, 11
    head = nn.Linear(dim, vocab)
    h = torch.randn(n, dim, requires_grad=True)
    targets = torch.randint(0, vocab, (n,))

    h_ref = h.detach().clone().requires_grad_(True)
    ref_loss_sum = F.cross_entropy(head(h_ref).float(), targets, reduction="sum")
    ref_grads = torch.autograd.grad(ref_loss_sum, [h_ref, head.weight, head.bias])

    chunk_loss_sum = _chunked_masked_ce_sum(head, h, targets, chunk=5)
    chunk_grads = torch.autograd.grad(chunk_loss_sum, [h, head.weight, head.bias])

    assert chunk_loss_sum.item() == pytest.approx(ref_loss_sum.item(), abs=1e-6)
    for g_ref, g_chunk in zip(ref_grads, chunk_grads, strict=True):
        torch.testing.assert_close(g_ref, g_chunk, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------------------
# Memory: the naive ordering materialises a [B, T, vocab] tensor; the production ordering
# never does. Chunking cuts the production ordering's own peak further still. Production-
# shaped batch/seq/vocab, CUDA only (peak-memory accounting on CPU is not meaningful the
# same way).
# ---------------------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="peak CUDA memory needs a CUDA device")
def test_masked_gather_uses_at_least_2x_less_peak_memory_than_naive_on_cuda() -> None:
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
    W4 launch note's hypothesis asked this test to assert (hence this test's name: it was
    previously named for the 3x it never measured, not the 2x it actually asserts and
    observes), and far short of the ~6.67x the raw (batch*seq_len)/n_masked element-count
    ratio alone would suggest. Reading WHY closes the gap between that INFERRED estimate
    and the VERIFIED number: the naive ordering's full `[B, T, vocab]` logits tensor is a
    single, short-lived FORWARD peak (nothing needs its values once boolean-mask indexing
    has produced the smaller `[n_masked, vocab]` tensor cross_entropy actually consumes),
    but `IndexBackward` still has to reconstruct a full `[B, T, vocab]`-shaped gradient at
    BACKWARD time to feed `AddmmBackward`'s `dL/dh = dL/dy_full @ W` -- so the naive path
    pays the full tensor's cost exactly ONCE, not twice, while `cross_entropy`'s own
    internal `log_softmax`/backward buffers (roughly logits-sized in both orderings) are
    a larger RELATIVE overhead on the gather path's much smaller base. The floor asserted
    below (2.0x, a hair under the measured 2.11x for headroom against allocator/version
    drift) is the number this measurement actually supports; raising it to 3.0x would be
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="peak CUDA memory needs a CUDA device")
def test_chunking_lowers_peak_memory_at_production_shape_on_cuda() -> None:
    """End-to-end `_mlm_token_loss` (trunk forward through backward, not just the
    isolated projection step above) at the production `memory`-region shape
    (batch=64, seq_len=96, vocab_size=50257, dim=256, depth=4, mask_prob=0.15) --
    chunk=512 vs chunk=0 (unchunked).

    MEASURED (this test, 3090 Ti, torch 2.10.0+cu128, same shape, n_masked=946 -- the
    RNG draw here differs from the isolated-projection test above, which fixes
    `input_ids`/the mask draw on the CUDA device directly rather than through
    `_mlm_token_loss`'s own `torch.rand` call, so the two tests' `n_masked` are close but
    not identical): chunk=512 peaks at 737.5 MiB against the unchunked path's 946.9 MiB
    -- a 1.28x reduction end to end (smaller than the isolated-projection-step ratio
    above because this measurement includes the trunk forward/backward, the embedding
    table, and `mlm_head`'s own parameter/gradient storage, all fixed costs unaffected by
    chunking, exactly the dilution the isolated test above exists to avoid). The floor
    asserted below (1.15x, under the measured 1.28x) is the number this measurement
    actually supports for headroom against allocator/version drift -- not a target
    reverse-engineered from a desired batch size; the batch=1280 re-probe in
    `docs/design/evidence/w4-masked-token-loss-2026-09-03/` is the number that answers
    "does the pre-registered batch fit now".
    """
    device = torch.device("cuda")
    batch, seq_len, vocab_size, dim = 64, 96, 50_257, 256
    mask_prob = 0.15
    chunk = 512

    def _peak_bytes(*, chunk_value: int) -> int:
        model, head, mask_emb = _fresh_trunk(
            vocab_size=vocab_size, dim=dim, depth=4, n_heads=4, max_len=seq_len, device=device
        )
        torch.manual_seed(7)
        input_ids = torch.randint(1, vocab_size, (batch, seq_len), device=device)
        attention_mask = torch.ones(batch, seq_len, dtype=torch.long, device=device)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.manual_seed(555)
        loss, n_masked = _mlm_token_loss(
            model, head, mask_emb, input_ids, attention_mask, mask_prob, chunk=chunk_value
        )
        assert n_masked > 0, "fixture must actually exercise the projection under test"
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device)
        del model, head, mask_emb, input_ids, attention_mask, loss
        torch.cuda.empty_cache()
        return peak - baseline

    peak_unchunked = _peak_bytes(chunk_value=0)
    peak_chunked = _peak_bytes(chunk_value=chunk)

    assert peak_chunked < peak_unchunked, (
        f"chunk={chunk} peak ({peak_chunked / 2**20:.1f} MiB) was not lower than "
        f"unchunked peak ({peak_unchunked / 2**20:.1f} MiB)"
    )
    ratio = peak_unchunked / peak_chunked
    assert ratio >= 1.15, (
        f"chunk={chunk} saved only {ratio:.2f}x end-to-end peak memory over unchunked "
        f"(chunked={peak_chunked / 2**20:.1f} MiB, unchunked={peak_unchunked / 2**20:.1f} "
        f"MiB); expected >=1.15x (measured ~1.32x) at batch={batch} seq_len={seq_len} "
        f"vocab_size={vocab_size}"
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
