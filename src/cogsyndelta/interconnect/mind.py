"""`WhiteMatter`: the interconnect composition root -- lane IC-8.

Source of truth: `docs/design/INTERCONNECT-MODULE-SPEC.md` ("the spec"), section 2.1
Table 2's `mind.py` row ("the forward pass of section 3, the per-request `h_r` cache,
the frozen-schedule injection path"), section 3 in full (the fourteen-step forward
pass), and section 7 Table 10 (IC-8, wave 3, depends on IC-1..7, IC-10, W0). Where the
spec disagrees with `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` ("TAX"), the spec
governs. This module wires together the other nine files' classes; it does not define
new tensor math beyond what section 3 assigns to "the loop that calls these in order"
(as `adapters.py`'s docstring puts it) and the two gaps recorded below that no wave-1/2
lane's file owns.

WHAT THIS FILE OWNS, AND WHAT IT DOES NOT
`InterconnectConfig` is the constructor's hyperparameter bundle (section 2.1's
`InterconnectConfig` defaults, plus the per-participant Table 1/4a numbers no
`Faculty` attribute carries -- see `ParticipantSpec` below). `WhiteMatter` is the
`nn.Module` built from `(config, faculties, store)` exactly as Table 2's row names
its constructor. It does not itself define `G30`/`G31`/`G32` (those are
`schedule.py`/`adapters.py`+`kv_bank.py`/`episodic_store.py`'s own guard code); it
calls the already-built `ScheduleValidator` (G30) at schedule-emission time and relies
on `adapters.assert_float_tract` (G31, invoked inside `RegionAdapter`/`TopKSelect`/
`ConditioningPrefix`/`KVBank`/`StoreProjection` already) and `EpisodicStore`'s own G32
checks (invoked inside `InMemoryStoreStub.read`/`write` already) firing at their own
call sites. It does not define G27-G29/G33-G36 (`gates.py`, lane IC-10, called by the
compose-stage script over a receipt this module's output feeds, not called from inside
a forward pass).

FOUR CROSS-LANE TENSIONS THIS FILE RESOLVES (recorded here and repeated in the lane
report, per the task's instruction to record where the spec is silent or lanes
disagree)

1. **EXECUTION-TIME BUDGETS ARE REQUEST-WIDE, NOT PER-ITEM.** `Schedule` (`schedule.py`,
   IC-1) is explicitly "one JSON object per item (no batch dimension)". `KVBank`
   (`kv_bank.py`, IC-2) needs one static slot layout for the whole call ("slot layout
   is fixed for the whole request, not recomputed per iteration" -- its own docstring).
   `ThalamicController` (`controller.py`, IC-5), however, produces genuinely per-item
   `ctx [B, R_ctx]`, `b [B, R]`, `A [B, I, R]` and `halt_at [B]` -- and `box_integerise`
   guarantees `Σ_r b_r = B_read` only PER ROW, so taking each region's per-item value
   independently (e.g. a per-region batch max, the resolution section 2.2 already gives
   for `ctx`) does not preserve `Σ_r b_r = B_read` across regions collectively; there is
   no batch-max rule that both keeps `KVBank`'s static layout and satisfies that
   cross-participant sum. Spec section 3 step 2 resolves the one case that actually
   drives every phase this module ships against: *"Phase A and the fallback `Schedule`
   use the dense allocation ... `A ≡ 1`, `ctx = ctx_max`, `halt_at = max_iters =
   n_iter`, and `b = box_integerise(uniform)`"* -- Table 6 confirms phase A runs with
   "controller (bypassed)". This file therefore treats `ctx`, `b`, `admitted` and
   `halt_at` as REQUEST-WIDE values for every actual forward pass (whether the dense
   fallback this file builds when `schedule=None`, or a `Schedule` the caller passes
   in), matching `Schedule`'s own one-object shape exactly rather than approximating a
   batch of them. `ThalamicController.forward` still runs on every call (so its
   per-item logits exist for phase B's `L_B` distillation target, `DistilLoss` in
   `losses.py`) but its output does not drive admission, budgets or the iteration
   bound here -- consuming it to train phase C/D's straight-through schedule is a
   later phase's own driver, not this module's forward pass (Table 6: phase A is the
   only row this lane's CPU smoke and integration tests exercise).

2. **THE MISSING FINAL NORM.** Spec section 2.3 Table 4's "latent bank, final norm,
   type embeddings" row counts `64·512 + 1,024 + R·512` -- the `1,024` is one
   `LayerNorm(512)`'s affine pair (`2·512`). `kv_bank.py` (IC-2) owns the type
   embeddings; `workspace.py` (IC-3) owns the latent bank; NEITHER file instantiates a
   final `LayerNorm`, and `Workspace.forward` returns `z` un-normed. This is a real gap
   against Table 4's own parameter budget, not a design choice either lane recorded --
   flagged as a defect in the lane report. Since no wave-1/2 lane owns it and the
   composition root is where every component meets, `WhiteMatter` instantiates the
   missing `nn.LayerNorm(D_w)` itself and applies it to `z_N` before read-out and
   before the store write (spec section 3 steps 12-13 both read "`z_N`", the final
   normed latents by Table 4's own count), restoring the counted parameter rather than
   silently dropping it.

3. **THE CONDITIONING-PREFIX / WORKSPACE-BATCHING ORDER.** `Workspace.forward`
   (`workspace.py`, IC-3) takes every iteration's bank tensors pre-assembled UP FRONT
   ("the caller supplies the already-assembled per-iteration bank tensors ... for every
   iteration up front" -- its own docstring's `[lane]` note). But spec section 3 step 5
   builds iteration `i`'s conditioning prefix from `z_{i-1}`, the workspace state AFTER
   `i-1` iterations already ran -- which does not exist until the loop is already
   underway. `Workspace.forward`'s all-up-front contract cannot express this: iteration
   `i`'s bank cannot be assembled before iteration `i-1`'s block has produced `z_{i-1}`
   whenever write-back is on. This file therefore does not call `Workspace.forward`;
   it drives `self.workspace.latent_bank` and `self.workspace.blocks[i]` directly, one
   iteration at a time, reproducing `Workspace.forward`'s own attention-mass-export
   arithmetic inline (the same `onehot`/`einsum` shape, so `a`'s contract is identical)
   so that each iteration's `ConditioningPrefix` call sees the REAL, just-computed
   `z_{i-1}` rather than a value assembled before the loop started. Flagged as a defect
   against `workspace.py`'s `Workspace.forward` entry point in the lane report --
   `WorkspaceBlock` itself (the untied per-iteration block IC-3 also exports) is used
   here exactly as built, with no modification.

4. **G30'S DENSE-FALLBACK EFFECT ON A REFUSED CALLER SCHEDULE IS DEFERRED (IC-R1
   SKEPTIC F5).** Table 8's G30 row gives two effects for a `Schedule` that fails a B1
   bound: "the `Schedule` is refused before execution AND the dense fallback runs."
   `forward` (below) implements only the first half on the caller-supplied-`Schedule`
   arm -- `self.schedule_validator.validate(schedule)` lets `ScheduleViolationError`
   propagate out of the call, with no substitute-with-dense-and-retry branch after it.
   The `schedule=None` arm already IS the dense fallback (`_dense_schedule`) for the
   one case spec section 3 step 2 names as load-bearing (phase A, tension 1 above);
   this module never constructs a bad frozen `Schedule` itself, so the missing branch
   is unreached by every phase this lane's tests exercise (Table 6: phase A only).
   Implementing the substitute-and-retry path is deferred to whichever phase first
   drives `WhiteMatter` with a real (non-dense, non-`None`) caller-supplied `Schedule`
   that can actually fail G30 in production -- spec-vs-code, not spec-vs-taxonomy.

WHAT `inputs` LOOKS LIKE
A single `Mapping[str, Any]` for the whole request: one entry per faculty participant
name, holding whatever that `Faculty.tokens()` implementation expects as its native
`inputs` argument (Table 1: text ids for `language`/`memory`/`reasoning`, an image
tensor for `visual`); plus, optionally, `"candidates"` (`[B, k-1, D_w]`, Q7's
pre-embedded content candidates, out of this lane's scope per `readout.py`'s own
docstring), `"scope"` (one `episodic_store.Scope`, broadcast to every batch item) or
`"scopes"` (`Sequence[Scope | None]`, one per item), and `"domain"` /
`"logical_key"` for the store's read/write partition (spec section 3 steps 8 and 13).
`episodic_store` is never a key of `inputs` itself -- the store is a constructor
argument, not a per-request input, and its scope/domain travel through `inputs` only
because *deriving* a scope is server-side (`derive_scope`, `episodic_store.py`), never
this module's job.

THE PER-REQUEST `h_r` CACHE (spec section 3 step 6: "reading the per-request cache
when `(ctx_r, cond_r)` is unchanged; the cache holds at most `R × I` entries and dies
with the request"). `[lane]` `ctx_r` is already request-wide (tension 1 above) so it
never changes across iterations within one call, which collapses the cache key to
"was `cond_r` `None` this iteration". With `write_back = False`, `cond_r` is `None` at
every iteration (spec section 3 step 5), so a region admitted at more than one
iteration is encoded once and served from cache thereafter -- exactly "only depth-0
emissions are cacheable once write-back is on" (spec section 4) generalises to "every
emission is cacheable" when write-back is off. With `write_back = True`, only the
first (`cond_r = None`) emission for a region is cached; any later, conditioned
emission recomputes and is never itself cached (a fresh `cond_r` each time it can
occur), matching the spec's own phase-2 caveat.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cogsyndelta.faculty.protocol import Faculty, assert_latent_tokens
from cogsyndelta.interconnect.adapters import ConditioningPrefix, RegionAdapter, TopKSelect
from cogsyndelta.interconnect.controller import (
    ControllerOutput,
    ControllerParticipant,
    ThalamicController,
    box_integerise,
)
from cogsyndelta.interconnect.episodic_store import EpisodicStore, Scope
from cogsyndelta.interconnect.kv_bank import STORE_PARTICIPANT, KVBank
from cogsyndelta.interconnect.readout import FrontalReadout, RankHead, UnifyProbes
from cogsyndelta.interconnect.schedule import (
    OutputSpec,
    ParticipantBudget,
    Schedule,
    ScheduleValidator,
    StepBudget,
    read_token_floor,
)
from cogsyndelta.interconnect.workspace import Workspace

__all__ = [
    "InterconnectConfig",
    "InterconnectConfigError",
    "ParticipantSpec",
    "WhiteMatter",
    "WhiteMatterOutput",
]


class InterconnectConfigError(ValueError):
    """A `WhiteMatter` was built with a bad `InterconnectConfig`.

    Mirrors `schedule.ScheduleValidatorConfigError` and `controller.ControllerConfigError`
    (IC-1, IC-5): this fires on construction-time arguments, before any forward pass.
    Currently the single case spec section 2.1 names: a caller-supplied `flops_ceiling`
    below the dense schedule's own `region_token_flops` (IC-R1 skeptic F3 -- see
    `WhiteMatter.__init__`).
    """


@dataclass(frozen=True)
class ParticipantSpec:
    """Per-participant Table 1 / Table 4a numbers no `Faculty` attribute carries.

    `Faculty` (`faculty/protocol.py`, W0) declares `token_dim`, `pooled_dim`,
    `kv_bytes_per_token` and `accepts_condition` -- everything a REGION knows about
    itself. It does not, and should not, know `ctx_min`/`ctx_max` (a workspace-side
    budget bound, Table 4a) or `token_budget_min`/`max` (Table 1's `b_r` box) or `phi`
    (its own measured MACs/position, an offline measurement, not a runtime attribute).
    `[lane]`: Table 2 fixes `WhiteMatter(config, faculties, store)` as a three-argument
    constructor; since these bounds are declarative per-participant data rather than a
    fourth live component, they travel on `InterconnectConfig.participants` instead of
    a fourth argument, keeping the literal Table 2 signature.
    """

    ctx_min: int | None
    """Table 4a's floor, or `None` for a participant with no `ctx` axis (the store)."""
    ctx_max: int | None
    """Table 4a's ceiling; `None` exactly when `ctx_min` is `None`."""
    token_budget_min: int
    """Table 1's `token_budget` minimum, `lo_r`'s other operand."""
    token_budget_max: int
    """Table 1's `token_budget` maximum, `hi_r`."""
    phi: float
    """`phi_r`, measured MACs per encoded position (spec section 4, `FLOPs` definition)."""


@dataclass(frozen=True)
class InterconnectConfig:
    """`WhiteMatter`'s hyperparameters -- spec section 2.1's `InterconnectConfig` defaults.

    `workspace_dim` (`D_w`), `latents` (`L`), `n_iter`, `heads`, `mlp_ratio`,
    `budget_total_read_tokens` (`B_read`), `budget_total_kv_bytes` (`B_kv`) and
    `floor_eta` (`eta`) are the "§1.4 interconnect block" defaults named in the spec's
    prose; `n_cond`, `controller_dim`, `controller_depth`, `controller_heads`,
    `flops_ceiling`, `write_back`, `k_candidates` and `unify_probes_trainable` are the
    named additions. `participants`, `rank_temperature`, `halt_epsilon`, `precision`,
    `allowed_modalities` and `resident_heads` are this file's own additions where the
    spec assigns a value to a constructor this module builds but does not itself name
    an argument for (rank temperature: "a recorded temperature", section 3 step 12;
    `halt_epsilon`: unnamed anywhere, matching `controller.py`'s own default; the
    validator's modality sets: section 2.1's `ScheduleValidator` constructor args).
    """

    participants: Mapping[str, ParticipantSpec]
    workspace_dim: int = 512
    latents: int = 64
    n_iter: int = 4
    heads: int = 8
    mlp_ratio: int = 4
    budget_total_read_tokens: int = 256
    budget_total_kv_bytes: int = 3_221_225_472
    floor_eta: float = 0.15
    n_cond: int = 8
    controller_dim: int = 256
    controller_depth: int = 2
    controller_heads: int = 4
    flops_ceiling: float | None = None
    write_back: bool = True
    k_candidates: int = 32
    unify_probes_trainable: bool = True
    rank_temperature: float = 1.0
    halt_epsilon: float = 1e-3
    precision: str = "fp32"
    allowed_modalities: tuple[str, ...] = ("text",)
    resident_heads: tuple[str, ...] = ("text",)
    wall_ms_budget: float = 5_000.0


class WhiteMatterOutput(NamedTuple):
    """Everything one `WhiteMatter.forward` call produces.

    Not a Table 2 type (no class name is assigned to a forward-pass result anywhere in
    the spec) -- `ControllerOutput` (`controller.py`) and `Schedule` (`schedule.py`)
    both set this precedent: a plain result container where nothing upstream names one.
    """

    f: Tensor
    """`[B, D_w]` float, spec section 3 step 12's read-out vector."""
    scores: Tensor | None
    """`[B, k]` float, `RankHead`'s cosine scores, or `None` when `inputs` carried no
    `"candidates"` key (a forward pass need not always rank)."""
    z: Tensor
    """`[B, L, D_w]` float, the final normed workspace latents `z_N` (tension 2 above)."""
    a: Tensor
    """`[B, halt_at, R]` float, the per-iteration connection-strength export (spec
    section 3 step 10); only the executed iterations are present -- an iteration at or
    beyond `halt_at` is never computed at all (tension 1), so there is nothing to zero."""
    z_history: tuple[Tensor, ...]
    """Length-`halt_at` tuple of `[B, L, D_w]`, `z` after each executed block."""
    schedule: Schedule
    """The `Schedule` this call executed: the caller's own `schedule` argument,
    returned unchanged (spec section 3's frozen-schedule arm: `forward(inputs,
    schedule=s0)` re-emits `s0` byte-identically), or this file's own dense-allocation
    `Schedule` when `schedule=None` was passed (spec section 3 step 2's fallback)."""
    controller: ControllerOutput | None
    """`ThalamicController`'s per-item output over this request's raw summaries, for
    phase B's `L_B` distillation target -- see the module docstring's tension 1. Only
    computed (non-`None`) when `schedule=None`; a frozen-schedule call has no
    controller-summary inputs of its own to run the controller against."""
    probe_outputs: dict[str, Tensor]
    """`{region: [B, pooled_dim_r]}`, `UnifyProbes.forward(f)`'s output (`L_unify`'s
    other operand, `pool_r(h_r)`, is the caller's to compute from `Faculty.pool`)."""
    write_receipt: list[Any] | None
    """One `WriteReceipt` per batch item with a non-`None` scope (spec section 3 step
    13), or `None` when this `WhiteMatter` has no `episodic_store` participant."""
    edges: dict[str, Any] | None
    """Spec section 3 step 14 / Table 7's write-back topology fields, present only when
    `config.write_back` is `True` (Table 8 row G27's effect: "`edges` ... omitted"
    otherwise) -- see `_derive_edges`."""
    ctx: dict[str, int]
    """Request-wide `ctx_r` per encoding participant (tension 1)."""
    b: dict[str, int]
    """Request-wide `b_r` per participant, `Σ_r b_r = B_read` exactly (tension 1)."""
    halt_at: int
    """Request-wide realised iteration bound (tension 1)."""


class WhiteMatter(nn.Module):
    """The interconnect's composition root -- spec section 2.1 Table 2's `mind.py` row.

    `WhiteMatter(config, faculties, store)`: `faculties` is `{name: Faculty}` for every
    NON-store participant named in `config.participants` (every entry Table 1 calls a
    region, i.e. everything but `episodic_store`); `store` is the `EpisodicStore` for
    the `episodic_store` participant, or `None` when `config.participants` does not
    name one (the `R = 4`, pre-E2 configuration, spec section 2.3).
    """

    def __init__(
        self,
        config: InterconnectConfig,
        faculties: dict[str, Faculty],
        store: EpisodicStore | None,
    ) -> None:
        """Build every wave-1/2 component over a fixed participant set.

        Args:
            config: Module-wide and per-participant hyperparameters.
            faculties: `{name: Faculty}` for every participant except `episodic_store`.
            store: The episodic store, or `None` when `episodic_store` is not a
                declared participant.

        Raises:
            ValueError: `faculties`'s keys disagree with `config.participants`'s
                non-store names, or `store`'s presence disagrees with whether
                `episodic_store` is a declared participant.
        """
        super().__init__()
        self.config = config
        self.faculties: dict[str, Faculty] = dict(faculties)
        self.store = store

        participant_names = tuple(config.participants)
        region_names = tuple(n for n in participant_names if n != STORE_PARTICIPANT)
        if set(faculties) != set(region_names):
            raise ValueError(
                f"WhiteMatter: faculties keys {sorted(faculties)} must exactly match "
                f"config.participants' non-store names {sorted(region_names)}."
            )
        has_store = STORE_PARTICIPANT in participant_names
        if has_store and store is None:
            raise ValueError(
                "WhiteMatter: config.participants names 'episodic_store' but store=None."
            )
        if not has_store and store is not None:
            raise ValueError(
                "WhiteMatter: store was given but config.participants does not name "
                "'episodic_store'."
            )
        self.participant_names = participant_names
        self.region_names = region_names
        self.has_store = has_store

        D_w = config.workspace_dim

        self.region_adapters = nn.ModuleDict(
            {name: RegionAdapter(faculties[name].token_dim, D_w) for name in region_names}
        )
        self.selectors = nn.ModuleDict(
            {name: TopKSelect(faculties[name].token_dim) for name in region_names}
        )
        self.conditioning = nn.ModuleDict(
            {
                name: ConditioningPrefix(D_w, faculties[name].token_dim, config.n_cond)
                for name in region_names
                if faculties[name].accepts_condition
            }
        )

        self.kv_bank = KVBank(participant_names, D_w, config.budget_total_read_tokens)
        self.workspace = Workspace(
            D_w, config.latents, config.n_iter, config.heads, config.mlp_ratio
        )
        # Tension 2: the parameter Table 4 counts and no wave-1/2 lane instantiates.
        self.final_norm = nn.LayerNorm(D_w)
        self.frontal_readout = FrontalReadout(D_w, config.mlp_ratio)
        self.rank_head = RankHead(D_w, config.k_candidates, config.rank_temperature)
        pooled_dims = {name: faculties[name].pooled_dim for name in region_names}
        self.unify_probes = UnifyProbes(D_w, pooled_dims)
        if not config.unify_probes_trainable:
            for p in self.unify_probes.parameters():
                p.requires_grad_(False)

        controller_participants: dict[str, ControllerParticipant] = {}
        for name in participant_names:
            spec = config.participants[name]
            summary_dim = 4 if name == STORE_PARTICIPANT else faculties[name].pooled_dim
            controller_participants[name] = ControllerParticipant(
                summary_dim=summary_dim,
                ctx_min=spec.ctx_min,
                ctx_max=spec.ctx_max,
                kv_bytes_per_token=(
                    None if name == STORE_PARTICIPANT else faculties[name].kv_bytes_per_token
                ),
                token_budget_min=spec.token_budget_min,
                token_budget_max=spec.token_budget_max,
            )
        self.controller = ThalamicController(
            controller_participants,
            config.controller_dim,
            config.controller_depth,
            config.controller_heads,
            config.n_iter,
            config.budget_total_read_tokens,
            config.budget_total_kv_bytes,
            config.floor_eta,
            halt_epsilon=config.halt_epsilon,
        )

        validator_participants = {
            name: ParticipantBudget(
                ctx_min=config.participants[name].ctx_min,
                ctx_max=config.participants[name].ctx_max,
                kv_bytes_per_token=(
                    None if name == STORE_PARTICIPANT else faculties[name].kv_bytes_per_token
                ),
                token_budget_min=config.participants[name].token_budget_min,
                token_budget_max=config.participants[name].token_budget_max,
                accepts_condition=(
                    False if name == STORE_PARTICIPANT else faculties[name].accepts_condition
                ),
            )
            for name in participant_names
        }
        # Spec section 2.1: "the dense schedule's own region_token_flops, Sum_i Sum_r
        # phi_r * ctx_max_r over n_iter iterations, the worst case B1 can reach" --
        # computed unconditionally (not only for the None-default arm below) because
        # "construction refuses a smaller value" is checked against it too.
        dense_region_token_flops = config.n_iter * sum(
            config.participants[name].phi * (config.participants[name].ctx_max or 0)
            for name in participant_names
        )
        dense_region_token_flops = max(dense_region_token_flops, 1e-6)
        flops_ceiling = config.flops_ceiling
        if flops_ceiling is None:
            flops_ceiling = dense_region_token_flops
        elif flops_ceiling < dense_region_token_flops:
            # IC-R1 skeptic F3: spec section 2.1's last sentence -- "construction
            # refuses a smaller value, so G30's dense fallback can never be refused by
            # its own validator." A ceiling under the dense schedule's own FLOPs means
            # the dense fallback this file builds when schedule=None (tension 1 above)
            # would immediately trip G30 against itself; refused here instead of
            # letting that surface as a ScheduleViolationError from inside forward().
            raise InterconnectConfigError(
                f"InterconnectConfig.flops_ceiling={flops_ceiling} is below the dense "
                f"schedule's own region_token_flops={dense_region_token_flops}; the "
                "dense fallback this module builds when schedule=None could never "
                "validate against its own ceiling"
            )
        self._flops_ceiling = float(flops_ceiling)
        self.schedule_validator = ScheduleValidator(
            validator_participants,
            config.budget_total_read_tokens,
            config.budget_total_kv_bytes,
            config.n_iter,
            config.floor_eta,
            self._flops_ceiling,
            config.allowed_modalities,
            config.resident_heads,
        )

    # ------------------------------------------------------------------
    # Schedule construction / unpacking
    # ------------------------------------------------------------------

    def _dense_schedule(self) -> Schedule:
        """Spec section 3 step 2's dense allocation: `A ≡ 1`, `ctx = ctx_max`,
        `halt_at = n_iter`, `b = box_integerise(uniform)` -- the schedule phase A and
        every fallback run against (Table 6).
        """
        cfg = self.config
        n = len(self.participant_names)
        uniform_target = torch.full((n,), cfg.budget_total_read_tokens / n, dtype=torch.float64)
        # Spec section 3 step 2's lo_r, through the module-wide definition the
        # validator and the controller also use -- NOT an inlined variant. An earlier
        # inlined `max(token_budget_min, 0)` here dropped the `ceil(eta/R * B_read)`
        # term and made this very schedule fail `self.schedule_validator.validate`
        # below (`read_token_floor`'s docstring records the measured refusal).
        lo = torch.tensor(
            [
                read_token_floor(
                    cfg.participants[name].token_budget_min,
                    cfg.floor_eta,
                    n,
                    cfg.budget_total_read_tokens,
                )
                for name in self.participant_names
            ],
            dtype=torch.float64,
        )
        hi = torch.tensor(
            [cfg.participants[name].token_budget_max for name in self.participant_names],
            dtype=torch.float64,
        )
        b_vec = box_integerise(uniform_target, lo, hi).tolist()

        context_tokens: dict[str, int] = {}
        read_tokens: dict[str, int] = {}
        admitted: dict[str, Sequence[bool]] = {}
        condition: dict[str, bool] = {}
        precision: dict[str, str] = {}
        priority: dict[str, int] = {}
        total_kv_bytes = 0
        for idx, name in enumerate(self.participant_names):
            spec = cfg.participants[name]
            is_store = name == STORE_PARTICIPANT
            ctx_r = 0 if is_store else int(spec.ctx_max or 0)
            context_tokens[name] = ctx_r
            read_tokens[name] = int(b_vec[idx])
            admitted[name] = tuple([True] * cfg.n_iter)
            condition[name] = (
                False
                if is_store
                else bool(cfg.write_back and self.faculties[name].accepts_condition)
            )
            precision[name] = cfg.precision
            priority[name] = idx
            if not is_store:
                total_kv_bytes += self.faculties[name].kv_bytes_per_token * ctx_r

        region_token_flops = cfg.n_iter * sum(
            cfg.participants[name].phi * context_tokens[name] for name in self.participant_names
        )
        step_budget = StepBudget(
            max_iters=cfg.n_iter,
            kv_bytes=total_kv_bytes,
            read_tokens=cfg.budget_total_read_tokens,
            wall_ms=cfg.wall_ms_budget,
            flops_ceiling=self._flops_ceiling,
        )
        output = OutputSpec(modalities=tuple(cfg.resident_heads))
        schedule = Schedule.assemble(
            context_tokens=context_tokens,
            read_tokens=read_tokens,
            admitted=admitted,
            condition=condition,
            precision=precision,
            priority=priority,
            region_token_flops=region_token_flops,
            step_budget=step_budget,
            output=output,
            halt_at=cfg.n_iter,
        )
        return self.schedule_validator.validate(schedule)

    @staticmethod
    def _unpack_schedule(
        schedule: Schedule,
    ) -> tuple[dict[str, int], dict[str, int], dict[str, tuple[bool, ...]], int]:
        """Pull the request-wide execution values out of a validated `Schedule`."""
        ctx: dict[str, int] = {}
        b: dict[str, int] = {}
        admitted: dict[str, tuple[bool, ...]] = {}
        for node in schedule.nodes:
            ctx[node.region] = node.context_tokens
            b[node.region] = node.read_tokens
            admitted[node.region] = tuple(node.admitted)
        return ctx, b, admitted, schedule.halt_at

    # ------------------------------------------------------------------
    # Controller summary (phase-B target exposure only -- tension 1)
    # ------------------------------------------------------------------

    def _raw_summary(
        self, inputs: Mapping[str, Any], batch_size: int, device: torch.device
    ) -> dict[str, Tensor]:
        """Build the controller's cheap per-participant summary (spec section 3 step 1,
        Table 5). `[lane]`: Table 5's per-region features (a bag-of-tokens mean over
        `language`'s embedding table, a mean over `visual`'s patch embedding) reach
        directly into a region's own weights -- something the frozen, black-box
        `Faculty` protocol (`tokens()`/`pool()` only) does not expose, and reaching
        past that boundary would violate the frozen-region contract Table 2 draws
        ("the interconnect does not reach back into a region's implementation").
        The smallest honest generic substitute available through the declared
        interface: run each region ONCE at its own `ctx_min` (the smallest, cheapest
        budget Table 4a declares) and take `Faculty.pool()`'s output directly as the
        raw summary feature -- `pooled_dim` already equals Table 5's declared summary
        widths (256 for the three `pooled_dim=256` regions, 384 for `visual`), so no
        width mismatch is introduced. This does technically "run" each region, unlike
        spec section 3 step 1's "without running any region" -- recorded as a
        deviation, not silently absorbed, because no other data source for this
        method's contract exists behind the declared `Faculty` surface.
        """
        raw: dict[str, Tensor] = {}
        for name in self.region_names:
            fac = self.faculties[name]
            ctx_r = self.config.participants[name].ctx_min or 1
            h, mask = fac.tokens(inputs[name], context_tokens=ctx_r)
            assert_latent_tokens(h)
            raw[name] = fac.pool(h, mask)
        if self.has_store:
            # [lane]: Table 5's store summary is "occupancy statistics" (resident
            # count, bytes, oldest/newest age) via a Linear(4, 256). The E0
            # `EpisodicStore` interface (episodic_store.py) exposes none of these
            # generically -- `capacity_bytes` always raises `ContractGap` in the
            # stub, and nothing on the Protocol returns age/byte statistics. The
            # smallest honest proxy available through the load-bearing `read` call
            # alone: resident count from the returned mask, zero for the three
            # figures the interface cannot supply.
            scopes = self._scopes_for(inputs, batch_size)
            counts = []
            for scope in scopes:
                _lat, mask = self.store.read(scope, domain=None, global_query=True, b_store=1)  # type: ignore[union-attr]
                counts.append(float(mask.sum().item()))
            resident = torch.tensor(counts, device=device).unsqueeze(-1)
            zeros = torch.zeros(batch_size, 3, device=device)
            raw[STORE_PARTICIPANT] = torch.cat([resident, zeros], dim=-1)
        return raw

    # ------------------------------------------------------------------
    # Store scope plumbing
    # ------------------------------------------------------------------

    @staticmethod
    def _scopes_for(inputs: Mapping[str, Any], batch_size: int) -> list[Scope | None]:
        if "scopes" in inputs:
            scopes = list(inputs["scopes"])
            if len(scopes) != batch_size:
                raise ValueError(
                    f"inputs['scopes'] has {len(scopes)} entries, expected {batch_size}."
                )
            return scopes
        return [inputs.get("scope")] * batch_size

    def _read_store(
        self, inputs: Mapping[str, Any], batch_size: int, b_store: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Spec section 3 step 8's store read, one item's scope at a time (the
        `EpisodicStore.read` contract is single-scope; see `episodic_store.py`'s own
        ambiguity note 2). Returns zero-width-safe `[B, b_store, D_w]`/`[B, b_store]`.
        """
        assert self.store is not None
        domain = inputs.get("domain")
        scopes = self._scopes_for(inputs, batch_size)
        latents_list = []
        mask_list = []
        for scope in scopes:
            lat, mask = self.store.read(
                scope, domain=domain, global_query=(domain is None), b_store=b_store
            )
            latents_list.append(lat)
            mask_list.append(mask)
        latents = torch.stack(latents_list, dim=0).to(device)
        mask = torch.stack(mask_list, dim=0).to(device)
        if latents.shape[-1] != self.config.workspace_dim:
            # Nothing has ever been written to this store: the stub infers D from the
            # first write and returns width 0 until then (episodic_store.py ambiguity
            # note 4). An all-empty bank of the right width is the honest "no store
            # data yet" answer, not a shape error.
            latents = latents.new_zeros(batch_size, b_store, self.config.workspace_dim)
            mask = torch.zeros(batch_size, b_store, dtype=torch.bool, device=device)
        return latents, mask

    def _write_store(self, inputs: Mapping[str, Any], z_final: Tensor) -> list[Any] | None:
        """Spec section 3 step 13: one record per turn, the mean of the final-normed
        latents, under `(scope, domain, logical_key)`.
        """
        if self.store is None:
            return None
        batch_size = z_final.shape[0]
        scopes = self._scopes_for(inputs, batch_size)
        domain = inputs.get("domain", "general")
        logical_key = inputs.get("logical_key", "turn")
        record = z_final.mean(dim=1)
        receipts = []
        for i, scope in enumerate(scopes):
            if scope is None:
                continue  # G32: an unknown principal cannot write (episodic_store.py).
            receipts.append(self.store.write(scope, domain, logical_key, record[i].detach()))
        return receipts

    # ------------------------------------------------------------------
    # Write-back topology (spec section 3 step 14 / Table 7, Table 8 G27/G28)
    # ------------------------------------------------------------------

    def _derive_edges(self, admitted: dict[str, tuple[bool, ...]]) -> dict[str, Any]:
        """Spec section 3's "Admission matrix semantics": `depth(r) = min{i :
        admitted[r][i]}`; regions first admitted together at the same `i >= 1` with
        `accepts_condition` form a lockstep group. Computed only when
        `config.write_back` is `True` (Table 8 G27's effect otherwise: `edges`
        omitted, `topology: "not demonstrated"` -- callers see that by `edges is None`
        on `WhiteMatterOutput`, since this method is only invoked under that guard).
        """
        depth = {
            name: next((i for i, a in enumerate(rows) if a), len(rows))
            for name, rows in admitted.items()
        }
        lockstep: dict[int, list[str]] = {}
        for name, d in depth.items():
            if d == 0:
                continue
            if name in self.region_names and self.faculties[name].accepts_condition:
                lockstep.setdefault(d, []).append(name)
        groups = [names for names in lockstep.values() if len(names) > 1]
        return {"depth": depth, "lockstep_groups": groups}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, inputs: Mapping[str, Any], schedule: Schedule | None = None
    ) -> WhiteMatterOutput:
        """Run the fourteen-step forward pass (spec section 3).

        Args:
            inputs: See the module docstring's "What `inputs` looks like" section.
            schedule: `None` runs steps 1-4 (controller for phase-B exposure only, per
                tension 1; the dense allocation drives execution) and validates the
                result (G30). A validated `Schedule` skips straight to steps 5-14,
                executed exactly as given -- the frozen-schedule arm (spec section 3's
                preamble).

        Returns:
            A `WhiteMatterOutput`.

        Raises:
            schedule.ScheduleViolationError: G30 -- `schedule` (whichever produced it)
                violates a bound.
        """
        cfg = self.config
        sample_region = self.region_names[0]
        sample_input = inputs[sample_region]
        device = sample_input.device if isinstance(sample_input, Tensor) else torch.device("cpu")

        controller_out: ControllerOutput | None = None
        if schedule is None:
            batch_size = self._batch_size(inputs)
            raw_summary = self._raw_summary(inputs, batch_size, device)
            controller_out = self.controller(raw_summary)
            schedule = self._dense_schedule()
        else:
            schedule = self.schedule_validator.validate(schedule)

        ctx, b, admitted, halt_at = self._unpack_schedule(schedule)
        batch_size = self._batch_size(inputs)
        n_regions = len(self.participant_names)

        store_latents: tuple[Tensor, Tensor] | None = None
        if self.has_store:
            store_latents = self._read_store(inputs, batch_size, b[STORE_PARTICIPANT], device)

        z = self.workspace.latent_bank(batch_size)
        a_list: list[Tensor] = []
        z_history: list[Tensor] = []
        h_cache: dict[str, tuple[Tensor, Tensor]] = {}

        for i in range(halt_at):
            cond: dict[str, Tensor] = {}
            if cfg.write_back and i >= 1:
                for name, prefix in self.conditioning.items():
                    if admitted[name][i]:
                        cond[name] = prefix(z)

            adapted_tokens: dict[str, tuple[Tensor, Tensor]] = {}
            for name in self.region_names:
                if not admitted[name][i]:
                    continue  # Spec step 6: "not admitted... neither encoded nor read."
                fac = self.faculties[name]
                use_cache = name not in cond
                if use_cache and name in h_cache:
                    h, mask = h_cache[name]
                else:
                    h, mask = fac.tokens(
                        inputs[name], context_tokens=ctx[name], condition=cond.get(name)
                    )
                    assert_latent_tokens(h)
                    if use_cache:
                        h_cache[name] = (h, mask)
                selected_h, selected_mask, _idx = self.selectors[name](h, mask, b[name])
                adapted_tokens[name] = (self.region_adapters[name](selected_h), selected_mask)

            bank_k, bank_v, key_mask, slot_region = self.kv_bank(
                dict(b), adapted_tokens, store_latents
            )

            z_new, weights = self.workspace.blocks[i](
                z, bank_k, bank_v, key_mask, return_attention=True
            )
            region_ids = torch.clamp(slot_region, min=0)
            onehot = F.one_hot(region_ids, num_classes=n_regions).to(z.dtype)
            onehot = onehot * (slot_region >= 0).unsqueeze(-1).to(z.dtype)
            mass_per_key = weights.sum(dim=(1, 2))
            region_mass = torch.einsum("bt,btr->br", mass_per_key, onehot) / (
                cfg.heads * cfg.latents
            )
            a_list.append(region_mass)
            z = z_new
            z_history.append(z)

        z = self.final_norm(z)  # Tension 2: z_N, the final normed latents.
        a = torch.stack(a_list, dim=1) if a_list else z.new_zeros(batch_size, 0, n_regions)

        f = self.frontal_readout(z)
        scores = None
        if "candidates" in inputs:
            scores = self.rank_head(f, inputs["candidates"])
        probe_outputs = self.unify_probes(f)
        write_receipt = self._write_store(inputs, z)
        edges = self._derive_edges(admitted) if cfg.write_back else None

        return WhiteMatterOutput(
            f=f,
            scores=scores,
            z=z,
            a=a,
            z_history=tuple(z_history),
            schedule=schedule,
            controller=controller_out,
            probe_outputs=probe_outputs,
            write_receipt=write_receipt,
            edges=edges,
            ctx=ctx,
            b=b,
            halt_at=halt_at,
        )

    @staticmethod
    def _batch_size(inputs: Mapping[str, Any]) -> int:
        for value in inputs.values():
            if isinstance(value, Tensor):
                return value.shape[0]
        raise ValueError("WhiteMatter.forward: inputs contains no tensor to infer batch size from.")
