"""The fixed 256-slot KV bank and the episodic store's key/value projections -- IC-2.

Source of truth: `docs/design/INTERCONNECT-MODULE-SPEC.md` ("the spec"), specifically
section 2.1 Table 2 (the `kv_bank.py` row), section 2.2 Table 3 (`bank_k`, `bank_v`,
`key_mask`, `slot_region`), section 3 step 8, section 4 Table 8 row G31, and section 5
Table 9 (the `kv_bank.py` row). Where the spec disagrees with
`docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` ("TAX"), the spec governs; every place
this module resolves something the spec leaves open is tagged `[spec]` below and
repeated in the lane report.

WHAT THIS MODULE OWNS, AND WHAT IT DOES NOT
`KVBank` packs one iteration's admitted, adapted tokens plus the episodic store's
read-out into the fixed `[B, 256, D_w]` bank the workspace cross-attends over (spec
section 3 step 8). `StoreProjection` is the store's own `W_k`/`W_v`: the store returns
latents at `pooled_dim = D_w = 512` already (spec Table 1), but they are not a region's
adapted tokens -- they never passed through a `RegionAdapter` -- so the bank gives them
their own pair of projections instead. This module does NOT run `RegionAdapter` or
`TopKSelect` (`adapters.py`, this same lane, IC-2, a separate file), decide which
regions are admitted or what `b_r` is (`controller.py`, lane IC-5), read the episodic
store (`episodic_store.py`, lane IC-6), or run the workspace blocks that consume this
bank's output (`workspace.py`, lane IC-3).

G31, THIS FILE'S SHARE OF IT
Spec Table 8 files G31 against both `adapters.py` and `kv_bank.py`. This file imports
`assert_float_tract` from `adapters.py` rather than re-deriving it -- one canonical
check, two boundaries: the adapted region tokens and the store's read-out latents,
both checked in `KVBank.forward` before they touch `bank_k`/`bank_v`, and again inside
`StoreProjection.forward` for a caller that reaches it directly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from cogsyndelta.interconnect.adapters import assert_float_tract

__all__ = ["STORE_PARTICIPANT", "KVBank", "StoreProjection"]

STORE_PARTICIPANT = "episodic_store"
"""The store's participant name, spec Table 1 -- the one entry `KVBank` treats specially."""


def _sincos_positions(n: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Fixed 1-D sin/cos positions, `[n, dim]` (spec section 2.1 Table 2, "sincos positions").

    `[spec]` DUPLICATED HERE RATHER THAN IMPORTED FROM `cogsyndelta.model.vl_jepa`. That
    module already has an identical closed form (`sincos_pos_embed`), but it is
    `visual`'s region-INTERNAL positional embedding, a different tract from the
    workspace's own bank position -- reaching into a region's model file from the
    interconnect would run the wrong direction of the boundary Table 2 draws (regions
    arrive frozen through the W0 protocol; the interconnect does not reach back into a
    region's implementation to borrow a helper). The closed form is six lines; it is
    duplicated here rather than shared through a new common module neither spec asks
    for.

    Args:
        n: Number of positions.
        dim: Embedding width; must be even.
        device: Target device.
        dtype: Target floating-point dtype.

    Returns:
        `[n, dim]` float, position `t`'s row is `[sin(t·f_0), .., cos(t·f_0), ..]` over
        `dim // 2` geometrically spaced frequencies, the standard closed form.

    Raises:
        ValueError: `dim` is odd.
    """
    if dim % 2 != 0:
        raise ValueError(f"sincos positions need an even dim, got dim={dim}.")
    if n == 0:
        return torch.zeros(0, dim, device=device, dtype=dtype)
    pos = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    idx = torch.arange(dim // 2, device=device, dtype=dtype)
    freq = torch.exp(-math.log(10_000.0) * idx / (dim // 2))
    angles = pos * freq.unsqueeze(0)
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class StoreProjection(nn.Module):
    """`W_k` and `W_v`: the episodic store's key/value projections into `D_w`.

    Spec section 2.1 Table 2 (`kv_bank.py` row: `StoreProjection(D_w)`). Spec section 3
    step 8: "Store slots take `W_k z` into `bank_k` and `W_v z` into `bank_v` over the
    latents `EpisodicStore.read` returned." Spec section 5 Table 9: "`W_k` and `W_v`
    carry no bias."
    """

    def __init__(self, D_w: int) -> None:  # noqa: N803 -- spec's own name
        """Allocate the two bias-free projections.

        Args:
            D_w: Workspace width (`512` at v1), also the store's `pooled_dim` (spec
                Table 1), so no separate width adapter sits in front of this class.
        """
        super().__init__()
        self.D_w = D_w
        self.w_k = nn.Linear(D_w, D_w, bias=False)
        self.w_v = nn.Linear(D_w, D_w, bias=False)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """`[B, b_store, D_w] -> ([B, b_store, D_w], [B, b_store, D_w])`.

        Args:
            z: The store's read-out latents (spec section 3 step 8).

        Returns:
            `(W_k @ z, W_v @ z)`, each `[B, b_store, D_w]` float.

        Raises:
            TypeError: G31 -- `z` is not a floating-point tensor (spec Table 8).
        """
        assert_float_tract(z, "StoreProjection input (store latents)")
        return self.w_k(z), self.w_v(z)


class KVBank(nn.Module):
    """Packs one iteration's admitted, adapted tokens into the fixed 256-slot bank.

    Spec section 2.1 Table 2 (`kv_bank.py` row): "the packed bank, `slot_region`, type
    embeddings, sincos positions, `W_k` and `W_v`." Spec section 3 step 8: "Pack the
    adapted tokens into `[B, 256, D]` in fixed participant order, write `slot_region`,
    add `type_emb[r]` and `sincos(pos)` to both streams, and clear `key_mask` for empty
    slots and for slots whose region is not admitted at `i`." Spec section 2.2: "The
    bank has a fixed width of `B_read = 256` slots with `slot_region` naming each
    slot's owner, so admission, I2′ severance and `a` are masks and scatter-adds over
    one static shape `[spec]`."

    `[spec]` SLOT LAYOUT IS FIXED FOR THE WHOLE REQUEST, NOT RECOMPUTED PER ITERATION.
    `b` (spec Table 3) is a controller output from section 3 step 2, computed once
    before any region runs; steps 5-11 repeat per iteration but never recompute it. So
    each participant's slot COUNT is fixed for the request, and only the CONTENTS (and
    `key_mask`) of an unadmitted region's range change between iterations -- an
    unadmitted region's slots are zero-filled and masked rather than omitted, keeping
    the bank's shape identical at every iteration (a requirement `workspace.py`, lane
    IC-3, already documents from its own side as its own `[lane]` shape note: it stacks
    one bank per iteration and needs every one of them the same `T`). `KVBank.forward`
    therefore takes every participant's slot count (`slot_budget`, fixed across the
    calls making up one request) separately from the tokens actually produced THIS
    iteration for the admitted subset (`adapted_tokens`); a participant absent from
    `adapted_tokens` is exactly "not admitted at `i`" (spec section 3 step 8).

    `[spec]` STORE SLOTS GET NO SINCOS TERM; REGION SLOTS DO. Spec section 3 step 8 is
    explicit that region slots get `type_emb[r]` and `sincos(pos)` "to both streams",
    while store slots "receive `type_emb[store]` and no positional term `[spec]`" (the
    taxonomy's own tag, TAX:1253) -- residency score, not position, orders the store's
    slots, so a position embedding there would assert an ordering the design declines
    to claim.
    """

    def __init__(self, participants: Sequence[str], D_w: int, B_read: int) -> None:  # noqa: N803
        """Allocate one type embedding per participant and, if present, the store's projections.

        Args:
            participants: Fixed participant order (spec Table 1: `language`, `memory`,
                `reasoning`, `visual`, and `episodic_store` when `R = 5`). This order
                fixes both the tie-break of spec section 3 step 2 and this bank's own
                slot layout (participant `i`'s range is contiguous within `[0, B_read)`
                in this order, spec section 3 step 8's "fixed participant order").
            D_w: Workspace width (`512` at v1).
            B_read: Total bank slots (`256` at v1, spec section 1.4).

        Raises:
            ValueError: `participants` is empty or names one twice.
        """
        super().__init__()
        if not participants:
            raise ValueError("KVBank needs at least one participant.")
        if len(set(participants)) != len(participants):
            raise ValueError(f"participants must be unique, got {list(participants)!r}.")
        self.participants = list(participants)
        self.D_w = D_w
        self.B_read = B_read
        self.type_emb = nn.Parameter(torch.randn(len(self.participants), D_w) * 0.02)
        self.store_index = (
            self.participants.index(STORE_PARTICIPANT)
            if STORE_PARTICIPANT in self.participants
            else None
        )
        # Spec section 2.3 Table 4, the `R = 4` row: "`W_k`/`W_v` and the store summary
        # STAY INSTANTIATED and receive no gradient in W5." So `StoreProjection` is
        # built unconditionally -- the pre-E2 `R = 4` configuration keeps the 524,288
        # parameters the table counts for it, and a checkpoint saved at `R = 4` loads
        # into an `R = 5` module and back without a key mismatch, which is the point of
        # counting them in the first place. What changes at `R = 4` is that they are
        # frozen and unreachable: no store participant means no store slots, so
        # `forward` below never calls this module.
        self.store_projection = StoreProjection(D_w)
        if self.store_index is None:
            self.store_projection.requires_grad_(False)

    def forward(
        self,
        slot_budget: Mapping[str, int],
        adapted_tokens: Mapping[str, tuple[Tensor, Tensor]],
        store_latents: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Assemble one iteration's `[B, B_read, D_w]` bank (spec section 3 step 8).

        Args:
            slot_budget: `{participant: b_r}` for every participant in
                `self.participants` -- the request-wide slot count (spec Table 3's
                `b`); `sum(slot_budget.values())` must equal `self.B_read` exactly
                (spec section 3 step 2's `Σ b_r = B_read`).
            adapted_tokens: `{participant: (tokens, mask)}` for the participants
                admitted (encoded) THIS iteration only. `tokens` is
                `[B, slot_budget[participant], D_w]` float -- spec Table 3's adapted
                tokens, already through `adapters.RegionAdapter` -- and `mask` is
                `[B, slot_budget[participant]]` bool. A participant absent from this
                mapping is "not admitted at `i`" (spec section 3 step 8): its slots are
                zero-filled with `key_mask` cleared. `episodic_store` is never a key
                here; its latents go through `store_latents` instead.
            store_latents: `(z, mask)` for `episodic_store`'s read-out this request,
                `z [B, slot_budget[episodic_store], D_w]` float and
                `mask [B, slot_budget[episodic_store]]` bool, required exactly when
                `episodic_store in self.participants` (`R = 5`) and forbidden
                otherwise (`R = 4`). An empty partition (spec section 3 step 8's
                `lo_store = 8` floor still reserves the slots) is `mask` all `False`,
                not `store_latents=None` -- `None` means "no store participant at all".

        Returns:
            `bank_k`, `bank_v`: `[B, B_read, D_w]` float (spec Table 3); identical on
                region slots, independently projected through `StoreProjection` on
                store slots.
            `key_mask`: `[B, B_read]` bool, `True` where attendable.
            `slot_region`: `[B, B_read]` long, the owning participant's index into
                `self.participants` for every slot (spec Table 3: "int in
                `{-1 .. R-1}`" -- this bank never emits `-1` itself, since every slot
                belongs to some participant's fixed range; `-1` is reserved for a
                caller-side severance mask, spec section 3's I2′ note, applied after
                this function returns).

        Raises:
            ValueError: `slot_budget`'s keys are not exactly `self.participants`, its
                values do not sum to `self.B_read`, `store_latents` is given without an
                `episodic_store` participant (or required but missing), or a supplied
                tensor's slot count disagrees with `slot_budget`.
            TypeError: G31 -- a tensor in `adapted_tokens` or `store_latents` is not
                floating point (spec Table 8).
        """
        if set(slot_budget) != set(self.participants):
            raise ValueError(
                f"slot_budget must cover exactly {self.participants!r}, "
                f"got {sorted(slot_budget)!r}."
            )
        total = sum(slot_budget.values())
        if total != self.B_read:
            raise ValueError(f"slot_budget sums to {total}, expected B_read={self.B_read}.")
        if self.store_index is None and store_latents is not None:
            raise ValueError(
                "store_latents was given but this KVBank has no 'episodic_store' participant."
            )
        if self.store_index is not None and store_latents is None:
            raise ValueError(
                "this KVBank has an 'episodic_store' participant; store_latents is required."
            )

        probe = next(iter(adapted_tokens.values()), None)
        probe_tensor = probe[0] if probe is not None else None
        if probe_tensor is None and store_latents is not None:
            probe_tensor = store_latents[0]
        if probe_tensor is None:
            raise ValueError("forward() needs at least one admitted region or store_latents.")
        batch_size = probe_tensor.shape[0]
        device = probe_tensor.device
        dtype = probe_tensor.dtype

        bank_k = torch.zeros(batch_size, self.B_read, self.D_w, device=device, dtype=dtype)
        bank_v = torch.zeros(batch_size, self.B_read, self.D_w, device=device, dtype=dtype)
        key_mask = torch.zeros(batch_size, self.B_read, dtype=torch.bool, device=device)
        slot_region = torch.full((batch_size, self.B_read), -1, dtype=torch.long, device=device)

        offset = 0
        for r_idx, name in enumerate(self.participants):
            b_r = slot_budget[name]
            sl = slice(offset, offset + b_r)
            slot_region[:, sl] = r_idx

            if name == STORE_PARTICIPANT:
                z, mask = store_latents  # type: ignore[misc]  # validated non-None above
                assert_float_tract(z, f"KVBank store latents ({name})")
                if z.shape[1] != b_r:
                    raise ValueError(
                        f"store_latents has {z.shape[1]} positions, slot_budget[{name!r}]={b_r}."
                    )
                k, v = self.store_projection(z)
                bank_k[:, sl] = k + self.type_emb[r_idx]
                bank_v[:, sl] = v + self.type_emb[r_idx]
                key_mask[:, sl] = mask
            elif name in adapted_tokens:
                tokens, mask = adapted_tokens[name]
                assert_float_tract(tokens, f"KVBank adapted tokens ({name})")
                if tokens.shape[1] != b_r:
                    raise ValueError(
                        f"adapted_tokens[{name!r}] has {tokens.shape[1]} positions, "
                        f"slot_budget[{name!r}]={b_r}."
                    )
                pos = _sincos_positions(b_r, self.D_w, device, dtype)
                stream = tokens + self.type_emb[r_idx] + pos
                bank_k[:, sl] = stream
                bank_v[:, sl] = stream
                key_mask[:, sl] = mask
            # else: not admitted at this iteration -- slots stay zero, key_mask False.

            offset += b_r

        return bank_k, bank_v, key_mask, slot_region
