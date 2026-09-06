"""Region adapters, top-k selection and the write-back conditioning prefix -- IC-2.

Source of truth: `docs/design/INTERCONNECT-MODULE-SPEC.md` ("the spec"), specifically
section 2.1 Table 2 (the `adapters.py` row), section 2.2 Table 3 (`h_r`, `cond_r`,
"selected `h_r`, adapted tokens"), section 3 steps 5 and 7, section 4 Table 8 row G31,
and section 5 Table 9 (the `adapters.py` row of the test plan). Where the spec disagrees
with `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` ("TAX"), the spec governs; every
place this module resolves something the spec leaves open is tagged `[spec]` below and
repeated in the lane report.

WHAT THIS MODULE OWNS, AND WHAT IT DOES NOT
`RegionAdapter` is the per-region `Linear(token_dim, D_w)` that maps a region's selected
token-surface positions into the shared workspace width (spec section 3 step 7).
`TopKSelect` is the straight-through top-`b_r` scorer that step 7 runs BEFORE
`RegionAdapter`, on the region's native `token_dim`. `ConditioningPrefix` is step 5's
write-back prefix: an attention-pool query over the workspace latents plus a projection
back down to `n_cond` positions at the region's own `token_dim`. This module does NOT
build the bank those adapted tokens land in (`kv_bank.py`, this same lane, IC-2, a
separate file), the controller that decides `b_r` and admission (`controller.py`,
lane IC-5), or the iteration loop that calls these in order (`mind.py`, lane IC-8).

G31, THIS FILE'S SHARE OF IT
Spec Table 8: "an integer-typed or vocabulary-indexed payload on `h_r`, the adapted
tokens, `z` or `cond_r` raises (TAX:2667)" -- filed against BOTH `adapters.py` and
`kv_bank.py`, because both files put hands on those four tensors at different points.
`assert_float_tract` below is the one guard function both files call; `kv_bank.py`
imports it rather than re-deriving it, the same way `RegionAdapter` and `TopKSelect`
below share it for `h_r` and `ConditioningPrefix` shares it for `z`/`cond_r` -- one
canonical check for a spec requirement two files are jointly responsible for, in the
same spirit as `cogsyndelta.faculty.protocol.assert_latent_tokens`, which is DEC-47's
version of the identical idea one hop earlier (a region's own `tokens()` output, before
any tract even exists).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["ConditioningPrefix", "RegionAdapter", "TopKSelect", "assert_float_tract"]


def assert_float_tract(tensor: Tensor, name: str) -> None:
    """G31 -- the latent-tract assertion (spec section 2.1 Table 8 row `G31`, TAX:2667).

    Spec section 2.2: "Two shape rules are load-bearing. Every tensor crossing a tract
    is a float tensor at a declared `*_dim`; an integer-typed or vocabulary-indexed
    payload on `h_r`, the adapted tokens, `z` or `cond_r` raises." This function is that
    check, shared by every boundary in `adapters.py` and `kv_bank.py` that touches one
    of those four tensors.

    Args:
        tensor: The tensor at a tract boundary.
        name: What to call it in the raised message, e.g. `"h_r"` or `"cond_r"`.

    Raises:
        TypeError: `tensor` is not a floating-point tensor.
    """
    if not torch.is_floating_point(tensor):
        raise TypeError(
            f"{name} is dtype {tensor.dtype}, not floating point. G31 (spec section 2.2, "
            f"section 4 Table 8, TAX:2667) requires every tensor crossing a tract -- "
            f"h_r, the adapted tokens, z or cond_r -- to be a float tensor at a declared "
            f"*_dim; an integer-typed or vocabulary-indexed payload raises at the "
            f"boundary rather than being silently carried forward."
        )


class RegionAdapter(nn.Module):
    """`Linear(token_dim, D_w)`: maps a region's selected tokens into the workspace width.

    Spec section 2.1 Table 2 (`adapters.py` row: `RegionAdapter(token_dim, D_w)`,
    "`Linear(d_r, 512)`"). Spec section 3 step 7: "`RegionAdapter` maps the selection
    to `D`" -- run after `TopKSelect`, on its output, never on the full `h_r`.
    """

    def __init__(self, token_dim: int, D_w: int) -> None:  # noqa: N803 -- spec's own name
        """Allocate the projection.

        Args:
            token_dim: The region's native width `d_r` (`256` text-family regions,
                `384` `visual`, spec Table 1).
            D_w: Workspace width (`512` at v1, spec section 1.4).
        """
        super().__init__()
        self.token_dim = token_dim
        self.D_w = D_w
        self.proj = nn.Linear(token_dim, D_w)

    def forward(self, h: Tensor) -> Tensor:
        """`[B, b_r, token_dim] -> [B, b_r, D_w]` (spec Table 3's adapted tokens).

        Args:
            h: A region's selected token-surface positions, already through
                `TopKSelect` (spec section 3 step 7).

        Returns:
            `[B, b_r, D_w]` float.

        Raises:
            TypeError: G31 -- `h` is not a floating-point tensor (spec Table 8).
        """
        assert_float_tract(h, "RegionAdapter input (selected h_r)")
        return self.proj(h)


class TopKSelect(nn.Module):
    """Straight-through top-`b_r` scorer over a region's token-surface positions.

    Spec section 2.1 Table 2 (`adapters.py` row: `TopKSelect(token_dim)`,
    "straight-through top-`b_r`"). Spec section 3 step 7: "`TopKSelect` scores each
    position with `Linear(d_r, 1)`, keeps the top `b_r` by hard selection among
    `mask_r ∧ budget_mask_r` positions with ties broken by position, and passes
    gradient to the scorer through a straight-through gate."

    `[spec]` `b_r` IS A FORWARD-TIME SCALAR SHARED BY THE WHOLE BATCH, NOT A PER-ITEM
    TENSOR. Spec Table 3 declares the controller's `b` as `[B, R]` -- per-item budgets
    in principle -- but the very next row fixes "selected `h_r`, adapted tokens" at
    `[B, b_r, d_r]`, one width for the whole batch, which only typechecks if `b_r` is a
    single number at the call site. Section 2.2's own resolution for the sibling budget
    `ctx_r` says exactly this: "within a batch a region runs at the batch maximum
    `ctx_r` and the per-item budget mask enforces each item's own budget." This module
    reads `b_r` the same way: the caller (`mind.py`, lane IC-8, not built by this lane)
    passes the batch-wide slot count for this call; an item whose own budget is smaller
    is expected to already have its extra positions excluded from `mask` (the
    `mask_r ∧ budget_mask_r` this function receives) before this module ever sees it,
    so `selected_mask` comes back `False` on whatever padding that produces.
    """

    def __init__(self, token_dim: int) -> None:
        """Allocate the per-position scorer.

        Args:
            token_dim: The region's native width `d_r`.
        """
        super().__init__()
        self.token_dim = token_dim
        self.scorer = nn.Linear(token_dim, 1)

    def forward(self, h: Tensor, mask: Tensor, b_r: int) -> tuple[Tensor, Tensor, Tensor]:
        """Select the top `b_r` positions of `h`; gradient reaches `self.scorer`.

        Args:
            h: `[B, T_r, token_dim]` float, a region's token-surface positions (spec
                Table 3's `h_r`).
            mask: `[B, T_r]` bool, `True` where a position is real AND inside its
                budget -- the caller's `mask_r ∧ budget_mask_r` (spec Table 3's
                `mask_r` ANDed with `budget_mask_r`).
            b_r: Number of positions to keep, identical for every batch item at this
                call (see the `[spec]` note above).

        Returns:
            `selected_h`: `[B, b_r, token_dim]` float, the top-`b_r` rows of `h` by
                score, ties broken by earlier position (a stable descending sort over
                position-ordered input), each row scaled by a straight-through gate
                that equals exactly `1` in the forward value and carries gradient to
                `self.scorer` in the backward pass.
            `selected_mask`: `[B, b_r]` bool, `True` where the selected position was
                real; `False` where fewer than `b_r` positions were valid at that item
                and this slot is padding drawn from a masked-out position.
            `indices`: `[B, b_r]` long, the selected positions in `h`'s own ordering --
                for a caller that must gather an aligned per-position quantity (e.g. an
                original position id).

        Raises:
            TypeError: G31 -- `h` is not a floating-point tensor (spec Table 8).
            ValueError: `b_r` is negative or larger than `h`'s sequence length.
        """
        assert_float_tract(h, "TopKSelect input (h_r)")
        _batch_size, seq_len, _dim = h.shape
        if b_r < 0 or b_r > seq_len:
            raise ValueError(f"b_r={b_r} is out of range for sequence length {seq_len}.")

        scores = self.scorer(h).squeeze(-1)  # [B, T_r]
        masked_scores = scores.masked_fill(~mask, float("-inf"))
        # Stable descending sort over position-ordered input breaks ties by earlier
        # position without a separate tie-break key.
        _, order = torch.sort(masked_scores, dim=-1, descending=True, stable=True)
        indices = order[:, :b_r]  # [B, b_r]

        selected_mask = torch.gather(mask, 1, indices)
        selected_scores = torch.gather(scores, 1, indices)
        gate_soft = torch.sigmoid(selected_scores)
        # Straight-through: forward value is exactly 1 (gate_soft - gate_soft.detach()
        # is 0 in the forward pass); backward gradient flows through gate_soft alone.
        gate = torch.ones_like(gate_soft) + (gate_soft - gate_soft.detach())

        gathered = torch.gather(h, 1, indices.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        selected_h = gathered * gate.unsqueeze(-1)
        return selected_h, selected_mask, indices


class ConditioningPrefix(nn.Module):
    """The write-back conditioning prefix: an attention-pool query plus a projection.

    Spec section 2.1 Table 2 (`adapters.py` row: `ConditioningPrefix(D_w, token_dim,
    n_cond)`, "`Linear(512, n_cond·d_r)` plus a per-region attention-pool query
    `[spec]`"). Spec section 3 step 5: for `i >= 1` and every admitted region with
    `accepts_condition`, `cond_r = reshape(Linear_r(pool_r(z_{i-1})), [n_cond, d_r])`,
    where `pool_r` is the region's attention pool over the latents.

    `[spec]` THE QUERY IS ONE LEARNED VECTOR OVER RAW DOT-PRODUCT ATTENTION, WITH NO
    SEPARATE KEY/VALUE PROJECTION. Table 2 names exactly one new parameter beyond the
    projection -- "a per-region attention-pool query" -- not a full attention sublayer
    with its own K/V weights; `pool_r` only READS the workspace state; it does not
    transform it. The smallest module matching that name is a single learned query
    vector of width `D_w`, one softmax over the `L` latents per item, using `z` itself
    as both key and value. This also makes the spec's own Table 9 test literal: at
    `self.query = 0` (this module's own initialisation) every dot product is `0`, the
    softmax is uniform over `L`, and `pool_r` degenerates exactly to `z.mean(dim=1)` --
    "a uniform query reproduces the mean pool."
    """

    def __init__(self, D_w: int, token_dim: int, n_cond: int) -> None:  # noqa: N803
        """Allocate the attention-pool query and the projection.

        Args:
            D_w: Workspace width (`512` at v1).
            token_dim: The region's native width `d_r`, the projection's output unit.
            n_cond: Conditioning-prefix length (`8` at v1, spec section 2.1).
        """
        super().__init__()
        self.D_w = D_w
        self.token_dim = token_dim
        self.n_cond = n_cond
        self.query = nn.Parameter(torch.zeros(D_w))
        self.proj = nn.Linear(D_w, n_cond * token_dim)

    def pool(self, z: Tensor) -> Tensor:
        """`pool_r`: the region's attention pool over the workspace latents.

        Args:
            z: `[B, L, D_w]` float, the workspace latents at iteration `i - 1` (spec
                section 3 step 5's `z_{i-1}`).

        Returns:
            `[B, D_w]` float.

        Raises:
            TypeError: G31 -- `z` is not a floating-point tensor (spec Table 8).
        """
        assert_float_tract(z, "ConditioningPrefix pool input (z)")
        scale = 1.0 / math.sqrt(self.D_w)
        scores = torch.einsum("d,bld->bl", self.query, z) * scale  # [B, L]
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bl,bld->bd", weights, z)

    def forward(self, z: Tensor) -> Tensor:
        """`cond_r = reshape(Linear_r(pool_r(z)), [n_cond, d_r])` (spec section 3 step 5).

        Args:
            z: `[B, L, D_w]` float, the workspace latents at iteration `i - 1`.

        Returns:
            `[B, n_cond, token_dim]` float, the conditioning prefix (spec Table 3's
            `cond_r`), passed to `Faculty.tokens(condition=...)`.

        Raises:
            TypeError: G31 -- `z`, or the value this method would return, is not a
                floating-point tensor (spec Table 8). The output check can only fire if
                a subclass overrides `pool` or `proj` to return a non-float tensor;
                `nn.Linear` on a float input cannot produce one on its own.
        """
        pooled = self.pool(z)
        cond = self.proj(pooled).view(z.shape[0], self.n_cond, self.token_dim)
        assert_float_tract(cond, "ConditioningPrefix output (cond_r)")
        return cond
