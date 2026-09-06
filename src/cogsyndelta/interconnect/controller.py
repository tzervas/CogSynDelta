"""The thalamic controller -- lane IC-5, `docs/design/INTERCONNECT-MODULE-SPEC.md`
("the spec" below) section 2.1 Table 2's `controller.py` row.

WHAT THIS MODULE IS
The spec's diagram (taxonomy TAX:1219-1246, quoted in the spec's section 1) puts one
component before every region runs: "raw input -> cheap summary -> 2 blocks @256 ->
ctx, b, A, halt". This file is that component. It owns four things per spec section
2.1 Table 2: the cheap per-participant summary (spec section 3 step 1), two transformer
blocks over that summary, four linear heads reading the blocks' output, and the two
"floored simplex" budget distributions plus the box-integerised read-token allocation,
the forced admission matrix, and the halt iteration (spec section 3 steps 2-3). It does
not run any region, assemble the KV bank, or execute a workspace block -- those are
`adapters.py`/`kv_bank.py` (lane IC-2) and `workspace.py` (lane IC-3); this file's
output (`ctx`, `b`, `A`, `halt_at`) is what those files, and `mind.py` (lane IC-8),
consume to decide what runs.

SPEC SECTIONS THIS FILE IMPLEMENTS
Section 2.1 Table 2 (this file's row); section 2.3 Table 4's "thalamic controller" row
and Table 5 (the summary and heads, undimensioned by the taxonomy, `[spec]`); section 3
steps 1-3 (summarise, budgets, admission and halt); section 5 Table 9 (this file's test
row, including the `box_integerise` fixtures reproduced in the tests below).

SPEC SILENCES THIS FILE RESOLVES (recorded per the lane's operating instructions)
  - Table 2 lists exactly three names for this file -- `ThalamicController`,
    `FlooredSimplex`, `box_integerise` -- but `schedule.py` (lane IC-1) already
    establishes the precedent this file follows: its own Table 2 row lists four
    dataclasses plus `ScheduleValidator`, yet that file also defines
    `ScheduleViolationError` and `ScheduleValidatorConfigError`, undeclared by Table 2,
    because `ScheduleValidator`'s constructor needed a typed shape for `participants`
    nothing upstream defined. The same gap exists here: `ThalamicController`'s
    constructor argument `participants` (Table 2) needs a typed per-participant shape,
    and `forward` must return several named tensors with no result type named anywhere.
    `ControllerParticipant`, `ControllerOutput` and `ControllerConfigError` below are
    this file's minimal answers, exactly as IC-1 resolved the analogous gap.
  - `participants` must be given in Table 1's participant order (`language`, `memory`,
    `reasoning`, `visual`, `episodic_store`). Nothing in the spec's prose gives
    `box_integerise`'s "ties ... going to the earlier participant in Table 1 order"
    (section 3 step 2) another source of ordering, and a Python `dict` already
    preserves insertion order, so this file reads participant order directly off the
    `Mapping` the caller constructs rather than inventing a second ordering channel.
  - The controller's own per-block MLP ratio and the halt threshold's `epsilon` are not
    among Table 2's constructor arguments and the spec's `InterconnectConfig` (section
    2.1) does not name a controller-specific MLP ratio, only the shared workspace
    `mlp_ratio: 4`. The parameter arithmetic settles the ratio unambiguously: Table 4's
    "thalamic controller ... 1,683,978" reproduces to the digit only at `mlp_ratio=4`,
    biased linears throughout, and each block's two internal LayerNorms held apart from
    the per-block 788,736 figure (see `_ControllerBlock`'s docstring for the worked
    arithmetic) -- so `mlp_ratio=4` is hardcoded, not exposed, matching the workspace's
    own default. `epsilon` (section 3 step 3, "reaches `1 - epsilon`") has no numeric
    value anywhere in the spec or taxonomy; `halt_epsilon` is added as a trailing
    constructor keyword with a small default (`1e-3`) so the literal Table 2 call
    `ThalamicController(participants, d_ctrl, depth_ctrl, heads_ctrl, n_iter, B_read,
    B_kv, eta)` still works unchanged.
  - `summarise`'s per-participant "cheap" features (Table 5's "mean of language's frozen
    embedding rows", "mean of the frozen patch embedding", "occupancy statistics") are
    computed from a frozen region's own weights or the store's own bookkeeping -- this
    file owns no region and no store (Table 2's ownership row for this file names
    neither), and `ThalamicController`'s constructor (Table 2) takes no `Faculty` or
    `EpisodicStore` argument to compute them from. `summarise` therefore takes those
    features already reduced to a fixed-width vector per participant
    (`raw_summary: Mapping[str, Tensor]`, one `[B, summary_dim_r]` entry per
    participant), and this file owns only the trainable half of each Table 5 summary
    row: the projection to `d_ctrl` (`Linear(384, 256)` for a participant whose raw
    feature is already width 256; `Linear(4, 256)` for the store's four occupancy
    numbers) plus the slot embeddings. Which upstream code builds the raw features is
    `mind.py`'s job (lane IC-8, which does hold the faculties and the store).
  - Table 3's admission row calls `A`'s construction "a straight-through sigmoid";
    section 3 step 3 forces admission at the argmax iteration when the threshold
    admits nowhere. `A` is typed `bool` at every tract boundary (Table 3), so this file
    returns the forced, thresholded `bool` tensor -- the value a straight-through
    estimator would produce on its forward pass -- and separately returns `A_logits`
    (`ControllerOutput`) so a later lane building the actual backward pass (phase C,
    spec section 4 Table 6, "the controller through its straight-through estimators")
    has the pre-threshold logits to differentiate through; wiring that estimator into a
    training loop is not this lane's job (Table 2 does not list a loss or an optimiser
    among this file's owners).

WHAT THIS MODULE DOES NOT DO
It does not validate a `Schedule` (that is `ScheduleValidator`, lane IC-1) or refuse
one -- everything this file emits is already inside its declared bounds by
construction (clipping, box allocation, forced admission, clamped halt), so there is no
per-call rejection path here, only the construction-time refusals `ScheduleValidator`
also has (spec section 3 step 2's "construction refuses ..." sentences). It does not
run a workspace block, encode a region, or touch the KV bank. This lane contains no
G-numbered guard.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional

__all__ = [
    "ControllerConfigError",
    "ControllerOutput",
    "ControllerParticipant",
    "FlooredSimplex",
    "ThalamicController",
    "box_integerise",
]


class ControllerConfigError(ValueError):
    """A `ThalamicController` (or `ControllerParticipant`) was built with a bad configuration.

    Mirrors `schedule.ScheduleValidatorConfigError` (lane IC-1): this fires on
    construction-time arguments, before any forward pass, catching exactly the
    infeasibilities spec section 3 step 2 names by name -- a `ctx_min` no floor share
    of `B_kv` can reach, and a `b`-box whose floors already exceed `B_read` or whose
    ceilings already fall short of it -- plus the plain shape/range checks
    (`d_ctrl % heads_ctrl`, `eta` outside `[0, 1]`, an empty `participants` mapping)
    the spec assumes are already true of any real configuration.
    """


@dataclass(frozen=True)
class ControllerParticipant:
    """Per-participant fields `ThalamicController` needs, from spec Table 1 / Table 4a.

    Not a restatement of `faculty.protocol.Faculty` or of `schedule.ParticipantBudget`
    (a different lane's file, scoped to `ScheduleValidator`'s own bounds): only the
    fields this file's two simplex formulas and the summary projection read. See the
    module docstring's "spec silences" section for why this class exists.

    Attributes:
        summary_dim: Width of this participant's raw per-item summary feature, the
            `raw_summary[name]` entry `summarise` receives -- 256 for a participant
            whose cheap feature is already `d_ctrl`-wide (Table 5's "text summary",
            "borrowed and frozen ... 0" params), 384 for `visual` (Table 5's
            `Linear(384, 256)`), 4 for `episodic_store` (Table 5's `Linear(4, 256)`
            over occupancy statistics).
        ctx_min: Table 4a's floor, or `None` for a participant with no `ctx` axis (the
            store: "the store has no `ctx`, its context is the scope partition", spec
            section 2.2 Table 3).
        ctx_max: Table 4a's ceiling; `None` exactly when `ctx_min` is `None`.
        kv_bytes_per_token: `c_r` (spec Table 1); `None` exactly when `ctx_min` is `None`.
        token_budget_min: Table 1's `token_budget` minimum, `lo_r`'s other operand.
        token_budget_max: Table 1's `token_budget` maximum, `hi_r`.
    """

    summary_dim: int
    ctx_min: int | None
    ctx_max: int | None
    kv_bytes_per_token: int | None
    token_budget_min: int
    token_budget_max: int

    def __post_init__(self) -> None:
        """Enforce the same-nullity and ordering constraints spec Table 1/4a implies."""
        if self.summary_dim < 1:
            raise ControllerConfigError(
                f"ControllerParticipant: summary_dim must be >= 1, got {self.summary_dim}."
            )
        ctx_fields = (self.ctx_min, self.ctx_max, self.kv_bytes_per_token)
        if any(f is None for f in ctx_fields) and not all(f is None for f in ctx_fields):
            raise ControllerConfigError(
                "ControllerParticipant: ctx_min, ctx_max and kv_bytes_per_token must be "
                f"all None (no ctx axis) or all set; got {ctx_fields}."
            )
        if self.ctx_min is not None and self.ctx_max is not None and self.ctx_min > self.ctx_max:
            raise ControllerConfigError(
                f"ControllerParticipant: ctx_min={self.ctx_min} > ctx_max={self.ctx_max}."
            )
        if self.kv_bytes_per_token is not None and self.kv_bytes_per_token < 1:
            raise ControllerConfigError(
                "ControllerParticipant: kv_bytes_per_token must be >= 1, got "
                f"{self.kv_bytes_per_token}."
            )
        if self.token_budget_min > self.token_budget_max:
            raise ControllerConfigError(
                "ControllerParticipant: token_budget_min="
                f"{self.token_budget_min} > token_budget_max={self.token_budget_max}."
            )


class ControllerOutput(NamedTuple):
    """Everything `ThalamicController.forward` produces, spec section 2.2 Table 3.

    Not one of Table 2's three named classes (see the module docstring's "spec
    silences" section) -- a plain result container, chosen as a `NamedTuple` so it
    behaves like the tuple a minimal implementation would return while giving every
    field a name later lanes (`mind.py`, `losses.py`) can read by attribute.

    Attributes:
        ctx: `[B, R_ctx]` int64, `ctx_r` for the participants with a `ctx` axis, in
            `participants` order restricted to those participants.
        b: `[B, R]` int64, `b_r` for every participant, `Σ_r b_r = B_read` exactly.
        A: `[B, I, R]` bool, the admission matrix; `Σ_i A[i, r] >= 1` for every `r`.
        halt_at: `[B]` int64 in `[1, n_iter]`, the realised iteration bound.
        active: `[B, I]` bool, `True` for an iteration this `halt_at` executes.
        halt_logits: `[B, I]` float, the `halt_head` output before the cumulative-sum
            halting rule; the phase-B distillation target's own logits are `s_b`, not
            this field (spec section 4, "Loss definitions").
        ctx_logits: `[B, R_ctx]` float, the `ctx_head` logits feeding `beta_ctx`'s
            `FlooredSimplex`, restricted to the participants with a `ctx` axis.
        b_logits: `[B, R]` float, the `b_head` logits feeding `beta_b`'s
            `FlooredSimplex` -- this is `s` in spec section 4's `L_B` distillation term.
        A_logits: `[B, R, I]` float, the `A_head` logits before thresholding and forced
            admission; the gradient path a later lane's straight-through estimator uses.
    """

    ctx: Tensor
    b: Tensor
    A: Tensor
    halt_at: Tensor
    active: Tensor
    halt_logits: Tensor
    ctx_logits: Tensor
    b_logits: Tensor
    A_logits: Tensor


class FlooredSimplex(nn.Module):
    """`eta/N + (1 - eta) * softmax(logits)`, spec section 3 step 2.

    A probability simplex over the last dimension with a hard per-category floor: as
    `logits -> -inf` for one category, its returned probability tends to `eta/N`, never
    lower, while the whole row still sums to exactly 1 (`eta + (1 - eta) * 1 = 1`).
    Stateless and parameter-free -- registered as an `nn.Module` only so it participates
    in `ThalamicController`'s module tree the way the spec's own diagram nests it
    ("2 blocks @256 -> ctx, b, A, halt"), not because it owns weights.

    Used twice by `ThalamicController.forward`, at two different `N`: `beta_ctx` over
    `R_ctx` (the encoding participants only) and `beta_b` over `R` (every participant,
    store included) -- one class, two call sites, per spec section 3 step 2's shared
    formula shape.
    """

    def forward(self, logits: Tensor, eta: float) -> Tensor:
        """Return the floored simplex over `logits`'s last dimension.

        Args:
            logits: `[..., N]` float, any real values.
            eta: The collapse floor `eta` (spec section 1: `floor_eta`), in `[0, 1]`.

        Returns:
            `[..., N]` float, each row summing to 1 with every entry `>= eta / N`.
        """
        n = logits.shape[-1]
        return eta / n + (1.0 - eta) * torch.softmax(logits, dim=-1)


def box_integerise(target: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    """Largest-remainder allocation with capping, spec section 3 step 2.

    Implements the exact procedure spec section 3 step 2 describes for `b`: "start
    every region at `lo_r`; share the pool `B_read - sum(lo)` among the uncapped
    regions in proportion to `beta_b`, as real values; cap any region whose share would
    carry it past `hi_r` at `hi_r` and return its unused share to the pool; repeat
    until no region is newly capped or every region is at `hi_r` ...; then integerise
    the uncapped shares by largest remainder over the remaining integer pool, ties on
    equal remainders going to the earlier participant in Table 1 order."

    `target` plays the role of `B_read * beta_b` (or, for a direct `box_integerise`
    call, whatever real-valued vector should be integerised under the same box and
    total) -- the redistribution proportions used while capping are `target`'s own
    relative weights among the still-uncapped entries, which are identical to
    `beta_b`'s relative weights among that subset because `target = B_read * beta_b`
    scales every entry by the same constant. This function does not itself check
    `sum(lo) <= sum(target) <= sum(hi)` (spec section 3 step 2's "construction
    refuses ..." sentence) -- that feasibility check is `ThalamicController.__init__`'s
    job, exactly as `ScheduleValidator.__init__` (lane IC-1) checks the analogous bound
    before any `Schedule` reaches its validator; a caller that violates it here gets an
    allocation that no longer sums to `round(target.sum())`, not a raised error.

    Args:
        target: `[..., R]` float, the real-valued allocation to integerise; the last
            dimension is the participant axis, in Table 1 order (ties are broken by
            that order). Any number of leading (batch) dimensions.
        lo: `[R]` or `[..., R]` float or int, broadcastable to `target`'s shape --
            each participant's floor, `lo_r`.
        hi: As `lo`, each participant's ceiling, `hi_r`.

    Returns:
        `[..., R]` int64, each row summing to `round(target.sum(-1))` exactly, every
        entry inside `[lo_r, hi_r]`.
    """
    target = target.to(torch.float64)
    lo = lo.to(torch.float64).expand_as(target)
    hi = hi.to(torch.float64).expand_as(target)

    flat_target = target.reshape(-1, target.shape[-1])
    flat_lo = lo.reshape(-1, lo.shape[-1])
    flat_hi = hi.reshape(-1, hi.shape[-1])

    rows = [
        _box_integerise_row(flat_target[i], flat_lo[i], flat_hi[i])
        for i in range(flat_target.shape[0])
    ]
    result = torch.stack(rows, dim=0).reshape(target.shape)
    return result.to(torch.int64)


def _box_integerise_row(target: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    """One batch item's worth of `box_integerise`, spec section 3 step 2. Pure Python
    over `R` scalars -- `R` is five participants at v1, so a per-row loop costs nothing
    next to a single forward pass and stays a direct transcription of the spec's prose.
    """
    r = target.shape[0]
    total = float(target.sum().item())
    beta = [float(target[i].item()) / total if total > 0 else 1.0 / r for i in range(r)]
    lo_f = [float(lo[i].item()) for i in range(r)]
    hi_f = [float(hi[i].item()) for i in range(r)]

    share = list(lo_f)
    capped = [False] * r
    pool = total - sum(lo_f)

    for _round in range(r):
        uncapped = [i for i in range(r) if not capped[i]]
        if not uncapped:
            break
        beta_sum = sum(beta[i] for i in uncapped)
        if beta_sum <= 0.0:
            add = {i: pool / len(uncapped) for i in uncapped}
        else:
            add = {i: pool * beta[i] / beta_sum for i in uncapped}
        newly_capped = [i for i in uncapped if lo_f[i] + add[i] > hi_f[i] + 1e-9]
        if not newly_capped:
            for i in uncapped:
                share[i] = lo_f[i] + add[i]
            break
        for i in newly_capped:
            share[i] = hi_f[i]
            capped[i] = True
        pool = (
            total
            - sum(share[i] for i in range(r) if capped[i])
            - sum(lo_f[i] for i in range(r) if not capped[i])
        )

    floors = [int(s) for s in share]  # floor for s >= 0, true for every box_integerise use here
    remainders = [share[i] - floors[i] for i in range(r)]
    remainder_pool = round(total) - sum(floors)
    order = sorted(range(r), key=lambda i: (-remainders[i], i))
    result = list(floors)
    for i in order[:remainder_pool]:
        result[i] += 1
    return torch.tensor(result, dtype=torch.float64)


def _admit(A_logits: Tensor) -> Tensor:  # noqa: N803 -- matches the spec's own `A` naming
    """Forced, thresholded admission, spec section 3 step 3.

    `A = 1[sigma(A_logits) > 0.5]`, then, for any participant `r` this thresholding
    admits at no iteration, forced to `True` at `r`'s own argmax iteration ("forced so
    every declared region is admitted at least once at its argmax iteration") -- so
    `sum_i result[b, r, i] >= 1` always holds, for any `A_logits`, including a row
    built to starve one participant (every one of its `I` logits very negative: the
    threshold admits nowhere, and the forced argmax picks the least-negative of them).
    Factored out of `ThalamicController.forward` so this guarantee is directly testable
    against adversarial logits without a full forward pass.

    Args:
        A_logits: `[B, R, I]` float, the `A_head` output.

    Returns:
        `[B, R, I]` bool.
    """
    soft = torch.sigmoid(A_logits)
    hard = (soft > 0.5).to(soft.dtype)
    no_admit = hard.sum(dim=-1) == 0
    argmax_idx = A_logits.argmax(dim=-1)
    forced = functional.one_hot(argmax_idx, num_classes=A_logits.shape[-1]).to(hard.dtype)
    hard = torch.where(no_admit.unsqueeze(-1), forced, hard)
    return hard.bool()


def _halt(halt_logits: Tensor, n_iter: int, epsilon: float) -> tuple[Tensor, Tensor]:
    """The realised `halt_at` and `active` mask, spec section 3 step 3 / step 11.

    "`halt_at` is the first `i` at which the cumulative halt probability reaches
    `1 - epsilon`, else `max_iters`" -- `max_iters` here is `n_iter`, this
    controller's own ceiling (see the module docstring's "spec silences" section).
    Factored out of `ThalamicController.forward` so the "an input exists whose
    `halt_at < n_iter`" case (spec section 5 Table 9) is directly constructible: make
    `halt_logits[:, 0]` large and positive so `sigmoid` is near 1 and the cumulative
    sum clears `1 - epsilon` at the first iteration.

    Args:
        halt_logits: `[B, I]` float, the `halt_head` output on the CLS slot.
        n_iter: The iteration ceiling; `halt_at` never exceeds it.
        epsilon: The halting threshold's `epsilon` (spec section 3 step 3).

    Returns:
        `(halt_at [B] int64 in [1, n_iter], active [B, I] bool)`.
    """
    p = torch.sigmoid(halt_logits.to(torch.float64))
    cum = torch.cumsum(p, dim=-1)
    reached = cum >= (1.0 - epsilon)
    any_reached = reached.any(dim=-1)
    first_idx = torch.argmax(reached.to(torch.int64), dim=-1)
    fallback = torch.full_like(first_idx, n_iter - 1)
    halt_at = torch.where(any_reached, first_idx, fallback) + 1
    iters = torch.arange(n_iter, device=halt_at.device)
    active = iters.unsqueeze(0) < halt_at.unsqueeze(-1)
    return halt_at, active


class _ControllerBlock(nn.Module):
    """One of the controller's two blocks, spec section 2.3 Table 4's "thalamic
    controller" row: `2 blocks @256 (2 x 788,736) + norms 2,048 + summary and heads
    104,458`.

    Pre-norm self-attention over all `R + 1` slots, then an MLP, both residual --
    the same shape the spec's own workspace block uses (section 3 step 9), applied to
    the summary sequence instead of the latents. `788,736` is `attn + mlp`, WITHOUT
    this block's own two `LayerNorm`s, which Table 4's arithmetic pulls out into the
    separate `norms 2,048` term (both blocks together: `2 blocks x 2 norms/block x
    (2 x 256) = 2,048`); the two figures are worked below because nothing upstream
    states the split explicitly `[spec]`:

        attn (`nn.MultiheadAttention`, biased Q/K/V/O, no head-count dependence):
            4 * (256*256 + 256) = 263,168
        mlp (`Linear(256, 1024) -> GELU -> Linear(1024, 256)`, `mlp_ratio=4`, biased):
            (256*1024 + 1024) + (1024*256 + 256) = 525,568
        263,168 + 525,568 = 788,736                                    -- matches Table 4
        this block's own norm1 + norm2: 2 * (2*256) = 1,024 (not in the 788,736 above)

    `ThalamicController` also holds one more `LayerNorm(d_ctrl)` after both blocks
    (Table 5's separate "final norm | 512" row) -- a third kind of norm, applied once,
    not owned by this class.
    """

    def __init__(self, d_ctrl: int, heads_ctrl: int, mlp_ratio: int) -> None:
        """Build one pre-norm self-attention + MLP block at width `d_ctrl`.

        Args:
            d_ctrl: Controller hidden width (spec section 2.1's `controller_dim`).
            heads_ctrl: Attention head count (spec section 2.1's `controller_heads`).
            mlp_ratio: MLP hidden-width multiplier; hardcoded to 4 by
                `ThalamicController` (see the module docstring's "spec silences").
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(d_ctrl)
        self.attn = nn.MultiheadAttention(d_ctrl, heads_ctrl, bias=True, batch_first=True)
        self.norm2 = nn.LayerNorm(d_ctrl)
        hidden = d_ctrl * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_ctrl, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_ctrl),
        )

    def forward(self, x: Tensor) -> Tensor:
        """`x + SelfAttn(LN(x))`, then `+ MLP(LN(.))`; `x` is `[B, R+1, d_ctrl]`."""
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ThalamicController(nn.Module):
    """The controller block of spec section 2.1 Table 2 / section 2.3 Table 5.

    Owns "the cheap summary, two blocks, four heads, both simplexes, admission, halt"
    (Table 2's row for this file): `summarise` builds the pre-block summary `s`
    (spec section 3 step 1's first half); `forward` runs the two `_ControllerBlock`s,
    the four linear heads, both `FlooredSimplex` budget distributions, `box_integerise`
    for `b`, and the forced-admission / cumulative-halt rules (spec section 3 steps
    1-3 in full).

    `participants` must be given in Table 1's participant order (`language`, `memory`,
    `reasoning`, `visual`, `episodic_store` at v1) -- `box_integerise`'s tie-break and
    every `[B, R, ...]`-shaped tensor this class produces use that order directly, via
    the `Mapping`'s own iteration order.
    """

    # Class-level type annotations for the registered buffers `__init__` fills in below:
    # `nn.Module.__getattr__`'s stub returns `Tensor | Module` for any attribute mypy
    # cannot otherwise resolve, which the arithmetic in `forward` and `box_integerise`
    # (both `Tensor`-only) then rejects; annotating each buffer's name here (a plain
    # declaration, not an assignment -- the real value is `register_buffer`'s job) is
    # the standard fix so static access resolves to `Tensor`.
    _lo: Tensor
    _hi: Tensor
    _ctx_idx: Tensor
    _ctx_min: Tensor
    _ctx_max: Tensor
    _c: Tensor

    def __init__(
        self,
        participants: Mapping[str, ControllerParticipant],
        d_ctrl: int = 256,
        depth_ctrl: int = 2,
        heads_ctrl: int = 4,
        n_iter: int = 4,
        B_read: int = 256,  # noqa: N803 -- spec's own constructor arg name (Table 2)
        B_kv: int = 3_221_225_472,  # noqa: N803 -- spec's own constructor arg name (Table 2)
        eta: float = 0.15,
        *,
        halt_epsilon: float = 1e-3,
    ) -> None:
        """Build a controller over a fixed participant set and budget configuration.

        Args:
            participants: Every participant this controller will ever see, in Table 1
                order, by name.
            d_ctrl: Controller hidden width (spec section 2.1's `controller_dim`,
                default 256).
            depth_ctrl: Number of `_ControllerBlock`s (spec section 2.1's
                `controller_depth`, default 2, "two controller blocks").
            heads_ctrl: Attention heads per block (spec section 2.1's
                `controller_heads`; taxonomy TAX:1279 fixes this design at 4).
            n_iter: The workspace's iteration count, the ceiling on `halt_at`
                (spec section 2.1, "One name for the iteration count").
            B_read: The workspace bank's total read-token slots (spec section 1: 256).
            B_kv: The total KV/activation byte budget (spec section 1:
                `budget_total_kv_bytes`).
            eta: The collapse floor `eta` (spec section 1: `floor_eta`, default 0.15).
            halt_epsilon: The halting rule's `epsilon` (spec section 3 step 3, "reaches
                `1 - epsilon`"); not in Table 2's argument list (see the module
                docstring's "spec silences") and given a small default so the literal
                Table 2 call signature is unaffected.

        Raises:
            ControllerConfigError: `participants` is empty; `d_ctrl`, `depth_ctrl`,
                `heads_ctrl`, `n_iter`, `B_read` or `B_kv` is not positive; `d_ctrl` is
                not divisible by `heads_ctrl`; `eta` is outside `[0, 1]`; some
                participant's `ctx_min` cannot be reached within its floor share of
                `B_kv` (spec section 3 step 2); or the `b`-box is infeasible
                (`sum(lo) > B_read` or `sum(hi) < B_read`).
        """
        super().__init__()
        if not participants:
            raise ControllerConfigError("ThalamicController needs at least one participant.")
        if d_ctrl < 1:
            raise ControllerConfigError(f"d_ctrl must be >= 1, got {d_ctrl}.")
        if depth_ctrl < 1:
            raise ControllerConfigError(f"depth_ctrl must be >= 1, got {depth_ctrl}.")
        if heads_ctrl < 1:
            raise ControllerConfigError(f"heads_ctrl must be >= 1, got {heads_ctrl}.")
        if d_ctrl % heads_ctrl != 0:
            raise ControllerConfigError(
                f"d_ctrl={d_ctrl} must be divisible by heads_ctrl={heads_ctrl}."
            )
        if n_iter < 1:
            raise ControllerConfigError(f"n_iter must be >= 1, got {n_iter}.")
        if B_read < 1:
            raise ControllerConfigError(f"B_read must be >= 1, got {B_read}.")
        if B_kv < 1:
            raise ControllerConfigError(f"B_kv must be >= 1, got {B_kv}.")
        if not (0.0 <= eta <= 1.0):
            raise ControllerConfigError(f"eta must be in [0, 1], got {eta}.")

        self.participant_names: tuple[str, ...] = tuple(participants)
        self.participants: dict[str, ControllerParticipant] = dict(participants)
        self.d_ctrl = d_ctrl
        self.n_iter = n_iter
        self.B_read = B_read
        self.B_kv = B_kv
        self.eta = eta
        self.halt_epsilon = halt_epsilon

        r_total = len(self.participant_names)
        ctx_names = [n for n in self.participant_names if self.participants[n].ctx_min is not None]
        r_ctx = len(ctx_names) or 1
        self.R = r_total
        self.R_ctx = len(ctx_names)

        for name in ctx_names:
            p = self.participants[name]
            floor_bytes = (eta / r_ctx) * B_kv
            needed_bytes = (p.kv_bytes_per_token or 0) * (p.ctx_min or 0)
            if floor_bytes < needed_bytes:
                raise ControllerConfigError(
                    f"ThalamicController: participant {name!r} cannot reach its own "
                    f"ctx_min ({p.ctx_min}) within its floor share of B_kv: "
                    f"eta/R_ctx*B_kv={floor_bytes:.1f} < c_r*ctx_min={needed_bytes} "
                    "(INTERCONNECT-MODULE-SPEC.md section 3 step 2)."
                )

        lo = {
            name: max(p.token_budget_min, math.ceil(eta / r_total * B_read))
            for name, p in self.participants.items()
        }
        hi = {name: p.token_budget_max for name, p in self.participants.items()}
        if sum(lo.values()) > B_read:
            raise ControllerConfigError(
                f"ThalamicController: sum of token-budget floors {sum(lo.values())} "
                f"exceeds B_read={B_read} (INTERCONNECT-MODULE-SPEC.md section 3 step 2)."
            )
        if sum(hi.values()) < B_read:
            raise ControllerConfigError(
                f"ThalamicController: sum of token-budget ceilings {sum(hi.values())} "
                f"falls short of B_read={B_read} (INTERCONNECT-MODULE-SPEC.md section 3 "
                "step 2)."
            )

        self.register_buffer(
            "_lo", torch.tensor([lo[n] for n in self.participant_names], dtype=torch.float64)
        )
        self.register_buffer(
            "_hi", torch.tensor([hi[n] for n in self.participant_names], dtype=torch.float64)
        )
        ctx_idx = [self.participant_names.index(n) for n in ctx_names]
        self.register_buffer("_ctx_idx", torch.tensor(ctx_idx, dtype=torch.long))
        self.register_buffer(
            "_ctx_min",
            torch.tensor([self.participants[n].ctx_min for n in ctx_names], dtype=torch.float64),
        )
        self.register_buffer(
            "_ctx_max",
            torch.tensor([self.participants[n].ctx_max for n in ctx_names], dtype=torch.float64),
        )
        self.register_buffer(
            "_c",
            torch.tensor(
                [self.participants[n].kv_bytes_per_token for n in ctx_names], dtype=torch.float64
            ),
        )

        self.summary_proj = nn.ModuleDict(
            {
                name: (
                    nn.Identity()
                    if self.participants[name].summary_dim == d_ctrl
                    else nn.Linear(self.participants[name].summary_dim, d_ctrl)
                )
                for name in self.participant_names
            }
        )
        self.slot_embed = nn.Parameter(torch.zeros(r_total + 1, d_ctrl))
        nn.init.normal_(self.slot_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [_ControllerBlock(d_ctrl, heads_ctrl, mlp_ratio=4) for _ in range(depth_ctrl)]
        )
        self.final_norm = nn.LayerNorm(d_ctrl)

        self.ctx_head = nn.Linear(d_ctrl, 1)
        self.b_head = nn.Linear(d_ctrl, 1)
        self.A_head = nn.Linear(d_ctrl, n_iter)
        self.halt_head = nn.Linear(d_ctrl, n_iter)

        self.simplex = FlooredSimplex()

    def summarise(self, raw_summary: Mapping[str, Tensor]) -> Tensor:
        """Build `s [B, R+1, d_ctrl]`, spec section 3 step 1 / section 2.2 Table 3's
        "controller summary `s`" row: "`controller.summarise` -> controller blocks".

        Slot 0 is the CLS slot (no raw feature of its own -- its content is entirely
        its learned slot embedding, spec Table 5's "slot embeddings" row); slots
        `1..R` are the participants, in `participants` order, each projected from its
        own raw feature width to `d_ctrl` (identity when they already match, e.g. the
        text participants' borrowed, frozen, zero-parameter row; a learned
        `Linear(summary_dim, d_ctrl)` otherwise, e.g. `visual`'s `Linear(384, 256)` and
        the store's `Linear(4, 256)`). This method does not run the controller blocks
        (spec section 3 step 1's second half, "add the slot embeddings, run the two
        controller blocks") -- see `forward`.

        Args:
            raw_summary: One `[B, summary_dim_r]` entry per participant name in
                `self.participant_names`; produced upstream by `mind.py` from each
                frozen region's own weights or the store's own bookkeeping (see the
                module docstring's "spec silences" section).

        Returns:
            `s`, `[B, R+1, d_ctrl]` float, slot embeddings already added.

        Raises:
            ControllerConfigError: `raw_summary`'s keys are not exactly
                `self.participant_names`.
        """
        if set(raw_summary) != set(self.participant_names):
            raise ControllerConfigError(
                f"summarise: expected one entry per participant {self.participant_names}, "
                f"got {sorted(raw_summary)}."
            )
        sample = raw_summary[self.participant_names[0]]
        batch, device, dtype = sample.shape[0], sample.device, sample.dtype
        cls = torch.zeros(batch, 1, self.d_ctrl, device=device, dtype=dtype)
        parts = [cls]
        for name in self.participant_names:
            proj = self.summary_proj[name](raw_summary[name])
            parts.append(proj.unsqueeze(1))
        s = torch.cat(parts, dim=1)
        return s + self.slot_embed.unsqueeze(0).to(dtype=dtype)

    def forward(self, raw_summary: Mapping[str, Tensor]) -> ControllerOutput:
        """Run the full controller, spec section 3 steps 1-3.

        Builds `s` (`summarise`), runs the `depth_ctrl` blocks and the final norm,
        reads the four heads, forms `beta_ctx`/`beta_b` (`FlooredSimplex`), derives
        `ctx` (clip) and `b` (`box_integerise`), and forms the forced admission matrix
        `A` and the realised `halt_at` from the cumulative halt rule.

        Args:
            raw_summary: As `summarise`.

        Returns:
            A `ControllerOutput`; see its own docstring for every field's shape and
            spec section 2.2 Table 3 for the boundary contract each one satisfies:
            `ctx_min <= ctx_r <= ctx_max_r`; `sum(b) == B_read` with every `b_r` inside
            its box; `sum_i A[i, r] >= 1` for every `r`; `1 <= halt_at <= n_iter`.
        """
        s = self.summarise(raw_summary)
        for block in self.blocks:
            s = block(s)
        s = self.final_norm(s)
        cls, participant_s = s[:, 0], s[:, 1:]

        ctx_logits_full = self.ctx_head(participant_s).squeeze(-1)
        b_logits = self.b_head(participant_s).squeeze(-1)
        A_logits = self.A_head(participant_s)
        halt_logits = self.halt_head(cls)

        ctx_logits = ctx_logits_full.index_select(1, self._ctx_idx)
        if self.R_ctx > 0:
            beta_ctx = self.simplex(ctx_logits.to(torch.float64), self.eta)
            ctx_raw = torch.floor(beta_ctx * self.B_kv / self._c)
            ctx = torch.maximum(torch.minimum(ctx_raw, self._ctx_max), self._ctx_min)
            ctx = ctx.to(torch.int64)
        else:
            ctx = torch.zeros(s.shape[0], 0, dtype=torch.int64, device=s.device)

        beta_b = self.simplex(b_logits.to(torch.float64), self.eta)
        target = beta_b * self.B_read
        b = box_integerise(target, self._lo, self._hi)

        A = _admit(A_logits).transpose(1, 2)
        halt_at, active = _halt(halt_logits, self.n_iter, self.halt_epsilon)

        return ControllerOutput(
            ctx=ctx,
            b=b,
            A=A,
            halt_at=halt_at,
            active=active,
            halt_logits=halt_logits,
            ctx_logits=ctx_logits,
            b_logits=b_logits,
            A_logits=A_logits,
        )
