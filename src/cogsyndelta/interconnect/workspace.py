"""Latent workspace blocks -- lane IC-3 of the interconnect.

Source of truth: `docs/design/INTERCONNECT-MODULE-SPEC.md` ("the spec"), specifically
section 2.1 Table 2 (the `workspace.py` row), section 3 steps 9-10 (the workspace block
and the attention-mass export), and section 5 Table 9 (the `workspace.py` row of the test
plan). Where the spec disagrees with `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md`
("TAX"), the spec governs; every place this module resolves something the spec leaves
open is tagged `[lane]` below and repeated in the lane report.

WHAT THIS MODULE OWNS, AND WHAT IT DOES NOT
`LatentBank` holds the `L` learned workspace latents (spec Table 3, `z`). `WorkspaceBlock`
is one untied cross-attention / self-attention / MLP block over pre-LN residual paths
(spec section 3 step 9). `Workspace` stacks `n_iter` untied `WorkspaceBlock`s and computes
the attention-mass export `a[b, i, r]` (spec section 3 step 10). This module does NOT
build the bank (`kv_bank.py`, lane IC-2), the controller's admission/halt decision
(`controller.py`, lane IC-5), or the iteration loop's conditioning/encode/select steps
(spec section 3 steps 5-8, owned by `mind.py`, lane IC-8). `Workspace.forward` is called
once per request and internally runs the whole `n_iter`-block stack; the caller supplies
the already-assembled per-iteration bank tensors (steps 5-8 already applied) for every
iteration up front.

`[lane]` SHAPE DEVIATION FROM TABLE 3, RESOLVED. Table 3 gives `bank_k`, `bank_v`,
`key_mask` and `slot_region` as `[B, 256, D]` / `[B, 256, D]` / `[B, 256]` / `[B, 256]` --
the shape `KVBank` produces for ONE iteration. Table 2 assigns `workspace.py` (not
`mind.py`) ownership of "the `n_iter` untied blocks", i.e. the iteration loop itself lives
here. Since admission and conditioning can change the bank contents at every iteration
(spec section 3 steps 5-8), `Workspace.forward` needs one bank per iteration, not one
bank reused `n_iter` times. The smallest honest resolution: `bank_k`/`bank_v`/`key_mask`/
`slot_region` are accepted here as Table 3's per-iteration tensor with an added axis 1 of
size `n_iter` -- `[B, n_iter, T, D_w]` / `[B, n_iter, T, D_w]` / `[B, n_iter, T]` /
`[B, n_iter, T]` -- so iteration `i`'s slice is exactly Table 3's shape. `T` is generic
(not hardcoded to 256) so CPU unit tests can use a toy bank size, matching the spec's own
integration-test configuration (section 5, `B_read = 16`).

`[lane]` `n_regions` IS A FORWARD ARGUMENT, NOT A CONSTRUCTOR ARGUMENT. Table 2's
constructor signature for `Workspace` is `Workspace(D_w, L, n_iter, heads, mlp_ratio)` --
no `R`. `Workspace` holds no per-region parameters (unlike `kv_bank.py`'s type
embeddings), so nothing about its weights depends on `R`; only the shape of the exported
`a [B, n_iter, R]` does, and `R` differs between phases (`R = 4` before E2 admits the
store, `R = 5` after, spec section 2.3). Taking `n_regions` at call time rather than
construction time avoids rebuilding the module across that transition and matches the
constructor argument list to the letter.

`[lane]` `LN(bank)` APPLIES ONE SHARED LAYERNORM TO BOTH `bank_k` AND `bank_v`. TAX's
diagram writes `CrossAttn(q=LN(z), kv=LN(KV_i))` for a single tensor `KV_i`; the spec
(section 3 step 9) writes the block as `CrossAttn(LN(z), LN(bank))`, also one `bank`.
Table 3 makes `bank_k` and `bank_v` two tensors that are "identical on region slots and
differ only on store slots" -- so `LN(bank)` cannot mean literally one LayerNorm call on
one tensor once a store is admitted. The smallest honest reading: one `nn.LayerNorm`
module, applied once to `bank_k` and once to `bank_v` (same weights, two calls), each
result then going through its own per-block `W_k`/`W_v`-shaped linear projection into the
attention's key/value space. This keeps "`LN(bank)`" as one normalisation rule while
respecting that `bank_k` and `bank_v` are two tensors from `kv_bank.py`'s Table 3 contract.

`[lane]` `WorkspaceBlock` ALWAYS RUNS THE EXPLICIT-SOFTMAX PATH INSIDE `Workspace`. The
spec (section 3 step 10) says "the block runs SDPA unless `return_attention` is set,"
naming SDPA as `WorkspaceBlock`'s own default. `Workspace`'s entire purpose (Table 2)
is producing `a`, which needs the materialised softmax (TAX: "`a_r` is read off the
softmax, never predicted by a head with its own loss"), so `Workspace.forward` always
calls its blocks with `return_attention=True`. The SDPA-only default lives on
`WorkspaceBlock` itself as the fast path a caller who does not need `a` may choose, and
this module's tests exercise both paths directly on `WorkspaceBlock` to prove they agree
to 1e-4 (spec Table 9), independent of which path `Workspace` happens to use internally.

Every tract tensor here is required to be a floating-point tensor; unlike
`cogsyndelta.faculty.protocol.assert_latent_tokens` this module does not re-check that at
its boundary (`workspace.py`'s slice of guard duty is none -- G31, the latent-tract
assertion, lives in `adapters.py` and `kv_bank.py` per spec Table 8), so this file
contains no guard code, as the lane brief requires.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["LatentBank", "Workspace", "WorkspaceBlock"]


class LatentBank(nn.Module):
    """The `L` learned workspace latents (spec Table 3, `z`; spec section 2.1 Table 2)."""

    def __init__(self, L: int, D_w: int) -> None:  # noqa: N803 -- spec's own names, verbatim
        """Allocate the `[L, D_w]` latent parameter.

        Args:
            L: Number of latents (`64` at v1, spec section 1.4).
            D_w: Workspace width (`512` at v1, spec section 1.4).
        """
        super().__init__()
        self.L = L
        self.D_w = D_w
        self.latents = nn.Parameter(torch.randn(L, D_w) * 0.02)

    def forward(self, batch_size: int) -> Tensor:
        """Broadcast the learned latents to `z_0`.

        Args:
            batch_size: `B`.

        Returns:
            `[B, L, D_w]` float, spec section 3's `z` before iteration 0. Every batch item
            starts from the identical learned latents (spec section 1.4, TAX:1219-1246);
            per-item state only appears after the first block runs.
        """
        return self.latents.unsqueeze(0).expand(batch_size, -1, -1)


class WorkspaceBlock(nn.Module):
    """One untied cross-attn / self-attn / MLP block over pre-LN residual paths.

    Implements spec section 3 step 9: `z += CrossAttn(LN(z), LN(bank))`,
    `z += SelfAttn(LN(z))`, `z += MLP(LN(z))`. Each `Workspace` instantiates `n_iter` of
    these with independent weights (spec section 2.1 Table 2, "untied"), so iteration
    `i`'s block never shares a parameter with iteration `j`'s.
    """

    def __init__(self, D_w: int, heads: int, mlp_ratio: float) -> None:  # noqa: N803
        """Build the block's cross-attention, self-attention and MLP sublayers.

        Args:
            D_w: Workspace width.
            heads: Attention head count (`8` at v1, spec section 1.4).
            mlp_ratio: MLP hidden-width multiplier (`4` at v1, spec section 1.4).

        Raises:
            ValueError: `D_w` is not divisible by `heads`.
        """
        super().__init__()
        if D_w % heads != 0:
            raise ValueError(f"D_w={D_w} is not divisible by heads={heads}.")
        self.D_w = D_w
        self.heads = heads
        self.head_dim = D_w // heads

        self.norm_q_cross = nn.LayerNorm(D_w)
        self.norm_kv_cross = nn.LayerNorm(D_w)
        self.cross_q = nn.Linear(D_w, D_w, bias=False)
        self.cross_k = nn.Linear(D_w, D_w, bias=False)
        self.cross_v = nn.Linear(D_w, D_w, bias=False)
        self.cross_out = nn.Linear(D_w, D_w, bias=False)

        self.norm_self = nn.LayerNorm(D_w)
        self.self_qkv = nn.Linear(D_w, 3 * D_w, bias=False)
        self.self_out = nn.Linear(D_w, D_w, bias=False)

        self.norm_mlp = nn.LayerNorm(D_w)
        hidden = int(D_w * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(D_w, hidden), nn.GELU(), nn.Linear(hidden, D_w))

    def _split_heads(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        b, h, n, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, h * d)

    def _cross_attend(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_mask: Tensor | None,
        return_attention: bool,
    ) -> tuple[Tensor, Tensor | None]:
        """Cross-attention over the bank, as SDPA or explicit softmax (spec section 3 step 10).

        Args:
            q: `[B, H, L, head_dim]`.
            k: `[B, H, T, head_dim]`.
            v: `[B, H, T, head_dim]`.
            key_mask: `[B, T]` bool, True where attendable, or None when every slot is
                attendable. A batch item whose row is entirely False would soften to a
                uniform-over-`-inf` row and produce NaN; that row is patched to keep slot
                0 attendable, following the same guard `ViTBlock` uses for an
                all-padding row (`src/cogsyndelta/model/vl_jepa.py`) -- the spec's own
                guarantees (the store's `lo_store = 8` floor, section 3 step 8) mean this
                never fires on a real schedule, only on adversarial unit-test inputs.
            return_attention: True runs the explicit-softmax path and returns the raw
                weights; False runs `F.scaled_dot_product_attention` and returns None.

        Returns:
            `(attended [B, H, L, head_dim], weights [B, H, L, T] or None)`.
        """
        attn_mask = None
        if key_mask is not None:
            keep = key_mask.bool()
            empty = ~keep.any(dim=1)
            if empty.any():
                keep = keep.clone()
                keep[empty, 0] = True
            attn_mask = keep[:, None, None, :]
        if return_attention:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            attended = torch.matmul(weights, v)
            return attended, weights
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        return attended, None

    def forward(
        self,
        z: Tensor,
        bank_k: Tensor,
        bank_v: Tensor,
        key_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Run this iteration's block (spec section 3 step 9).

        Args:
            z: `[B, L, D_w]`, the workspace latents before this block.
            bank_k: `[B, T, D_w]`, this iteration's assembled bank keys (spec Table 3;
                spec section 3 step 8).
            bank_v: `[B, T, D_w]`, this iteration's assembled bank values.
            key_mask: `[B, T]` bool, True where attendable; None means every slot is
                attendable.
            return_attention: When True, run the explicit-softmax cross-attention path
                and return the raw per-head weights `[B, H, L, T]` (spec Table 3's
                "cross-attention weights, `a`" row, the un-exported intermediate). When
                False (the default), run SDPA and return None in their place. The two
                paths must agree on the returned `z` to `1e-4` (spec section 5 Table 9).

        Returns:
            `(z, weights)`: updated latents `[B, L, D_w]`; `weights` is
            `[B, H, L, T]` when `return_attention` else `None`.
        """
        q = self._split_heads(self.cross_q(self.norm_q_cross(z)))
        k = self._split_heads(self.cross_k(self.norm_kv_cross(bank_k)))
        v = self._split_heads(self.cross_v(self.norm_kv_cross(bank_v)))
        attended, weights = self._cross_attend(q, k, v, key_mask, return_attention)
        z = z + self.cross_out(self._merge_heads(attended))

        h = self.norm_self(z)
        sq, sk, sv = self.self_qkv(h).chunk(3, dim=-1)
        sq, sk, sv = self._split_heads(sq), self._split_heads(sk), self._split_heads(sv)
        self_attended = F.scaled_dot_product_attention(sq, sk, sv, is_causal=False)
        z = z + self.self_out(self._merge_heads(self_attended))

        return z + self.mlp(self.norm_mlp(z)), weights


class Workspace(nn.Module):
    """The `n_iter` untied `WorkspaceBlock`s plus the attention-mass export.

    Spec section 2.1 Table 2 (`workspace.py` row); spec section 3 steps 9-10.
    """

    def __init__(self, D_w: int, L: int, n_iter: int, heads: int, mlp_ratio: float) -> None:  # noqa: N803
        """Build the latent bank and `n_iter` untied `WorkspaceBlock`s.

        Args:
            D_w: Workspace width (`512` at v1).
            L: Latent count (`64` at v1).
            n_iter: Number of workspace blocks, and the upper bound on iterations
                (spec section 2.1: "One name for the iteration count").
            heads: Attention head count per block.
            mlp_ratio: MLP hidden-width multiplier per block.
        """
        super().__init__()
        self.D_w = D_w
        self.L = L
        self.n_iter = n_iter
        self.heads = heads
        self.latent_bank = LatentBank(L, D_w)
        self.blocks = nn.ModuleList([WorkspaceBlock(D_w, heads, mlp_ratio) for _ in range(n_iter)])

    def forward(
        self,
        bank_k: Tensor,
        bank_v: Tensor,
        key_mask: Tensor,
        slot_region: Tensor,
        n_regions: int,
        active: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Run the `n_iter`-block stack and export the per-region attention mass.

        Iteration `i` runs `self.blocks[i]` (spec section 2.1: "Iteration `i` runs block
        `i` with untied weights, so halting at iteration 2 skips blocks 3 and 4"). Here
        "skips" means the block still computes on the full batch (there is no per-item
        early exit inside a single forward pass over a batched tensor), but its effect on
        `z` and its contribution to `a` are masked out for any batch item inactive at
        that iteration, per `active` -- see the Returns section. This is spec section 3
        step 10's "for `i >= halt_at` it is zero and `active[b, i]` is false" applied per
        item rather than per batch.

        Args:
            bank_k: `[B, n_iter, T, D_w]` float, this request's per-iteration assembled
                bank keys (spec Table 3's `[B, T, D_w]` stacked over iterations -- see
                the module docstring's `[lane]` shape note).
            bank_v: `[B, n_iter, T, D_w]` float, likewise for values.
            key_mask: `[B, n_iter, T]` bool, True where attendable.
            slot_region: `[B, n_iter, T]` int in `{-1, .., n_regions - 1}`; `-1` marks an
                empty or otherwise non-attributable slot and is excluded from `a`.
            n_regions: `R`, the participant count this request's bank was built against
                (`4` or `5`, spec section 2.3) -- see the module docstring's `[lane]`
                note on why this is a forward argument.
            active: `[B, n_iter]` bool, True where iteration `i` is executed for that
                batch item (spec Table 3's `active`). Defaults to all-True, the phase-A
                dense schedule (`halt_at = max_iters = n_iter`, spec section 3 step 2).

        Returns:
            `z`: `[B, L, D_w]` float, the final latents after the last active iteration
                per item (spec Table 3).
            `a`: `[B, n_iter, n_regions]` float, the connection-strength export (spec
                section 3 step 10). `a[b, i, :]` sums to 1 over the regions with
                attendable slots at iteration `i` when `active[b, i]`, and is all zero
                when not.
            `z_history`: length-`n_iter` list of `[B, L, D_w]` tensors, `z` after each
                block runs (spec section 5 Table 9, "`z` keeps its shape across
                iterations" -- exposed per-iteration so that invariant is checkable at
                every step, not only on the final output).
        """
        batch_size = bank_k.shape[0]
        device = bank_k.device
        dtype = bank_k.dtype
        z = self.latent_bank(batch_size)

        if active is None:
            active = torch.ones(batch_size, self.n_iter, dtype=torch.bool, device=device)

        region_ids = torch.clamp(slot_region, min=0)
        onehot = F.one_hot(region_ids, num_classes=n_regions).to(dtype)
        onehot = onehot * (slot_region >= 0).unsqueeze(-1).to(dtype)

        a = torch.zeros(batch_size, self.n_iter, n_regions, dtype=dtype, device=device)
        z_history: list[Tensor] = []

        for i, block in enumerate(self.blocks):
            z_new, weights = block(
                z, bank_k[:, i], bank_v[:, i], key_mask[:, i], return_attention=True
            )
            mass_per_key = weights.sum(dim=(1, 2))
            region_mass = torch.einsum("bt,btr->br", mass_per_key, onehot[:, i]) / (
                self.heads * self.L
            )
            active_i = active[:, i]
            a[:, i] = region_mass * active_i.unsqueeze(-1).to(dtype)
            z = torch.where(active_i[:, None, None], z_new, z)
            z_history.append(z)

        return z, a, z_history
