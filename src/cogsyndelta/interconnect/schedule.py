"""The `Schedule` data model and its validator -- lane IC-1, guard G30.

WHAT THIS MODULE IS
`docs/design/INTERCONNECT-MODULE-SPEC.md` (the spec below) section 3 step 4 calls the
`Schedule` "the pre-execution" object: everything the controller decided about a request
before any region runs, packaged so it can be inspected, refused, logged, or replayed
byte-identically (the frozen-schedule arm, spec section 3's preamble). This module owns
two things per spec section 2.1 Table 2: **emission** -- packaging already-computed
per-participant values into the frozen dataclasses below (`Schedule.assemble`) -- and
**validation** -- refusing a `Schedule` that violates any bound the spec's section 4
Table 8 calls G30, before the module lets it run.

WHY A SEPARATE VALIDATOR RATHER THAN VALIDATING TYPES
A frozen dataclass constructed directly in Python cannot carry an unexpected field, so
`Schedule(nodes=..., trace_id="x")` already raises `TypeError` for free. The guard this
module owns is for the boundary the dataclasses cannot police by construction: a
`Schedule` arriving as parsed JSON (`Schedule.from_dict`) can carry any key at all, and a
`Schedule` built correctly in Python can still carry field VALUES that violate a §9.9 B1
bound -- a `read_tokens` over its participant's ceiling, a `depth` that disagrees with
its own `admitted` array, a `region_token_flops` over the ceiling. `ScheduleValidator`
is the single place both kinds of violation are caught, both raising the same
`ScheduleViolationError`, so "the Schedule validator refused it" means one thing regardless
of which check fired.

WHAT THIS MODULE DOES NOT DO
It does not compute `ctx`, `b`, `A` or `halt_at` -- that is section 3 steps 1-3, owned by
`controller.py` (lane IC-5, a later wave; `box_integerise` and the two simplexes are its
functions, not this module's). It does not run any region, assemble the KV bank, or
build the dense fallback's own numbers -- `mind.py` (lane IC-8) calls
`Schedule.assemble` with those numbers already computed and calls `ScheduleValidator`
to refuse or accept the result before deciding whether to run it or fall back to the
dense schedule G30 names. `ScheduleValidator`'s own construction-time refusals (this
module's `ScheduleValidatorConfigError`) are the guarantee, stated in spec section 3
step 2, that the dense fallback can never itself be refused by a correctly configured
validator -- but this module trusts its caller to have actually built the dense
fallback correctly; it only refuses to be misconfigured in a way that would make that
promise false.

SPEC SECTIONS THIS FILE IMPLEMENTS
Section 2.1 Table 2 (this file's row); section 2.2 Table 3 (the `Schedule` row and its
bounds); section 3 step 4 (emit and validate); section 4 Table 8 row G30 and Table 8a
(field-by-field disposition); section 5 Table 9 (this file's test row). Every reference
below of the form "spec §N" means a section of `INTERCONNECT-MODULE-SPEC.md`, not the
taxonomy it derives from.

SPEC SILENCES THIS FILE RESOLVES (recorded per the lane's operating instructions)
  - Table 2 lists four dataclasses for this file (`Schedule`, `ScheduleNode`,
    `StepBudget`, `OutputSpec`) plus `ScheduleValidator`, but `ScheduleValidator`'s own
    constructor argument `participants` needs a typed shape and nothing upstream of
    this lane defines one yet (`kv_bank.py`/`adapters.py` are lane IC-2, a later wave).
    `ParticipantBudget` below is the smallest such shape: only the five fields this
    file's bounds actually read from spec Table 1 and Table 4a, not a restatement of
    the full `Faculty` protocol (`faculty/protocol.py`, which this file does not need).
  - "A store namespace named" is one of G30's firing conditions (Table 8) but the spec
    never names the JSON key a namespace would arrive under. This file treats
    `"scope"`, `"namespace"` and `"store_namespace"` as forbidden alongside `trace_id`
    (spec Table 8a's own forbidden field), matching Q1's vocabulary
    ("What is `scope` in `(scope, domain, logical_key)`") -- the actual store-scope
    guard (G32) lives in `episodic_store.py`, a different lane; this file only refuses
    a `Schedule` that tries to carry one.
  - `ScheduleNode.condition` records whether this node's conditioning prefix
    (DEC-17, spec section 3 step 5) is applied at this schedule's iterations that admit
    it; the spec's Table 3 does not give the `Schedule`-level field a name distinct
    from the tensor `cond_r`, since `cond_r` itself is never JSON. The validator checks
    it against the participant's declared `accepts_condition` (a region that does not
    accept conditioning cannot be scheduled with `condition=True`).
  - `Schedule.halt_at` is a top-level field distinct from `step_budget.max_iters`:
    `max_iters` is the request's ceiling (spec §2.1's "`n_iter` is ... the upper bound
    ... a request's `step_budget.max_iters <= n_iter` is the bound the validator
    checks"), and `halt_at` is the realised value spec section 3 step 14 says the
    schedule finalises, bound by `halt_at <= max_iters` (Table 3).
  - `resident_heads` and `allowed_modalities` are taken as plain `Collection[str]`
    membership sets (a modality name is in the set or it is not) rather than a richer
    structure, since the spec never says a resident head or an allowed modality carries
    any per-modality data this validator needs beyond presence.
  - Table 8a keeps `resident` (always `true` at v1) and `codec` (always `"none"` at v1)
    as declared-but-inert fields. This validator reads "always" as a bound, not a
    description of current behaviour: a `Schedule` claiming `resident=False` or a
    `codec` other than `"none"` describes a capability nothing in the module
    implements, so it is refused rather than silently accepted and ignored -- the same
    fail-closed reading Table 8a already applies to `output.stream` and the two
    latency fields, which this file also refuses when non-default.
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

__all__ = [
    "OutputSpec",
    "ParticipantBudget",
    "Schedule",
    "ScheduleNode",
    "ScheduleValidator",
    "ScheduleValidatorConfigError",
    "ScheduleViolationError",
    "StepBudget",
    "read_token_floor",
]


def read_token_floor(token_budget_min: int, eta: float, r_total: int, b_read: int) -> int:
    """Spec §3 step 2's `lo_r = max(token_budget.min_r, ceil(eta/R · B_read))`.

    The single definition of that floor for the whole module. Three places enforce it --
    `ScheduleValidator` (below), `ThalamicController.__init__` (`controller.py`) and
    `WhiteMatter._dense_schedule` (`mind.py`) -- and they have to agree exactly, because
    the third BUILDS the schedule the first REFUSES. IC-R2 skeptic, measured:
    `_dense_schedule` had inlined `max(token_budget_min, 0)`, dropping the
    `ceil(eta/R · B_read)` term, so the dense fallback the module builds when
    `schedule=None` was refused by its own validator ("G30: memory: read_tokens=7
    outside [8, 96]") on any config where that term binds. A shared function makes the
    divergence impossible to reintroduce silently.

    Args:
        token_budget_min: The participant's own `token_budget.min_r` (spec Table 1).
        eta: The collapse floor `eta` (spec §1's `floor_eta`).
        r_total: `R`, the participant count the floor share is split across.
        b_read: `B_read`, the bank's total read-token slots.

    Returns:
        `lo_r`, the smallest `read_tokens` this participant may be given.
    """
    return max(token_budget_min, math.ceil(eta / r_total * b_read))


#: Table 8a's closed precision vocabulary: "fixed to the checkpoint dtype at v1; any
#: other value is refused."
PRECISIONS: frozenset[str] = frozenset({"fp32", "bf16", "int8", "int4"})

#: Table 8a's only v1 codec value.
CODEC_V1 = "none"

#: Keys no `Schedule` may carry, at any nesting level: `trace_id` is spec Table 8a's own
#: named exclusion ("superseded: server-minted, never a model field"); the other three
#: are this file's resolution of "a store namespace named" (G30, Table 8) -- see the
#: module docstring's "spec silences" section.
FORBIDDEN_KEYS: frozenset[str] = frozenset({"trace_id", "scope", "namespace", "store_namespace"})


class ScheduleViolationError(RuntimeError):
    """G30 fail-closed: a `Schedule` violates a bound, or carries a field it may never carry.

    "Violates a bound" means a spec §9.9 B1 bound; "a field it may never carry" means
    `trace_id` or a store namespace/scope key. Raised by both
    `Schedule.from_dict` (structural: forbidden or unknown fields, missing fields) and
    `ScheduleValidator.validate` (semantic: budgets, admitted/depth consistency,
    `flops_ceiling`, modality residency) -- one exception type for the one guard, so a
    caller does not need to know which half of the check fired to react correctly:
    catch `ScheduleViolationError`, run the dense fallback (spec §3 step 4, §4 Table 8).
    """


class ScheduleValidatorConfigError(ValueError):
    """A `ScheduleValidator` was constructed with an infeasible configuration.

    Distinct from `ScheduleViolationError`: this fires on the validator's OWN construction
    arguments, before any `Schedule` exists to validate, catching exactly the two
    infeasibilities spec §3 step 2 names by name -- a `ctx_min` no floor share of
    `B_kv` can reach, and a token-budget box whose floors already exceed `B_read` or
    whose ceilings already fall short of it. Both would make every possible `Schedule`
    unsatisfiable, which is a configuration bug, not a per-request refusal.
    """


@dataclass(frozen=True)
class StepBudget:
    """Per-request execution ceilings a `Schedule` declares, spec §2.1 Table 2 / §3 step 4.

    Attributes:
        max_iters: Upper bound on iterations this request may run, `<= n_iter`
            (spec §2.1: "a request's `step_budget.max_iters <= n_iter` is the bound the
            validator checks").
        kv_bytes: Total `Σ_r c_r · ctx_r` this schedule declares it will spend, checked
            against the sum the validator computes from the nodes and against `B_kv`.
        read_tokens: Total `Σ_r b_r` this schedule declares, checked against `B_read`.
        wall_ms: The runtime deadline for the whole turn (Table 8a: "keep and enforce").
        flops_ceiling: This schedule's own FLOPs ceiling, `<=` the validator's
            configured `flops_ceiling` (spec §2.1's `InterconnectConfig.flops_ceiling`).
    """

    max_iters: int
    kv_bytes: int
    read_tokens: int
    wall_ms: float
    flops_ceiling: float


@dataclass(frozen=True)
class OutputSpec:
    """What the schedule declares it will emit, spec §3 step 4 / §4 Table 8a.

    Attributes:
        modalities: Every modality the read-out may produce this turn; checked against
            the validator's `allowed_modalities` and `resident_heads`.
        stream: Table 8a: `"stream" is "false" at v1`; kept as a declared field, DEC-44.
        first_token_ms: Table 8a: "recorded `null`", the field's enforcement deferred
            with the speech head.
        speech_frame_ms: As `first_token_ms`, DEC-48.
    """

    modalities: tuple[str, ...]
    stream: bool = False
    first_token_ms: float | None = None
    speech_frame_ms: float | None = None


@dataclass(frozen=True)
class ScheduleNode:
    """One participant's row in a `Schedule`, spec §2.2 Table 3 / §3 step 4.

    Attributes:
        region: Participant name, e.g. `"language"`, `"episodic_store"`.
        active: Whether this region is admitted at any iteration,
            `any(admitted)` (spec §3's admission-matrix semantics, "`Σ_i A[i, r] >= 1`").
        depth: `min{i : admitted[i]}`, the region's first admitting iteration
            (spec §3, "Admission matrix semantics").
        admitted: Per-iteration admission, length `n_iter`; `A[:, r]` for this region.
        context_tokens: `ctx_r`, positions this region encodes this turn.
        read_tokens: `b_r`, positions this region contributes to the workspace bank.
        condition: Whether this node's conditioning prefix (DEC-17, spec §3 step 5) is
            applied; only meaningful when the participant's `accepts_condition` is true.
        priority: Table 8a: "the rank of `β_b` in descending order, ties by participant
            order"; recorded for DEC-70's arbiter, not consumed by this module.
        precision: One of `PRECISIONS`; Table 8a: "fixed to the checkpoint dtype at v1".
        resident: Table 8a: "always `true` at v1, because `WeightStore.ensure_resident`
            is a no-op".
        codec: Table 8a: always `"none"` at v1; the tract codec is a placeholder.
    """

    region: str
    active: bool
    depth: int
    admitted: tuple[bool, ...]
    context_tokens: int
    read_tokens: int
    condition: bool
    priority: int
    precision: str
    resident: bool
    codec: str


@dataclass(frozen=True)
class Schedule:
    """The pre-execution schedule for one item, spec §2.2 Table 3 / §3 step 4.

    One `Schedule` covers one item (no batch dimension: spec §2.2, "one JSON object per
    item"). Construct it with `Schedule.assemble` from already-computed per-participant
    values (the "emit" half of this module's job), refuse or accept it with
    `ScheduleValidator.validate` (the "validate" half), and serialise it with
    `to_dict`/`to_json`/`from_dict`/`from_json` for the frozen-schedule replay arm
    (spec §3's preamble, `WhiteMatter.forward(inputs, schedule=...)`).

    Attributes:
        nodes: One `ScheduleNode` per participant.
        step_budget: The request's declared ceilings.
        region_token_flops: `Σ_i Σ_r A[i, r] · φ_r · ctx_r`, this schedule's predicted
            total (spec §4, "Loss definitions"); computed by the caller, not this
            module -- `φ_r` is a region's measured MACs per position, not data this
            file has access to.
        output: What the schedule declares it will emit.
        halt_at: The realised iteration bound, `<= step_budget.max_iters` (spec §3
            step 14, "`halt_at` is the realised value").
    """

    nodes: tuple[ScheduleNode, ...]
    step_budget: StepBudget
    region_token_flops: float
    output: OutputSpec
    halt_at: int

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for JSON serialisation; the inverse of `from_dict`."""
        return asdict(self)

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialise to a JSON string (spec §3's frozen-schedule replay arm)."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> Schedule:
        """Parse a `Schedule` from a plain dict, refusing anything malformed (G30).

        Structural refusals only: a forbidden field at any level (`trace_id`, a store
        namespace/scope key), an unrecognised field, or a missing required field. Bound
        violations on an otherwise well-formed `Schedule` are `ScheduleValidator`'s job.

        Args:
            data: A mapping shaped like `to_dict`'s output -- typically
                `json.loads(...)`, but any mapping works.

        Returns:
            The parsed `Schedule`.

        Raises:
            ScheduleViolationError: `data`, or one of its nested objects, carries a
                forbidden or unrecognised key, or is missing a required key.
        """
        _check_keys(data, _SCHEDULE_FIELDS, where="schedule")
        try:
            nodes = tuple(_node_from_dict(n) for n in data["nodes"])
            step_budget = _step_budget_from_dict(data["step_budget"])
            output = _output_from_dict(data["output"])
            return Schedule(
                nodes=nodes,
                step_budget=step_budget,
                region_token_flops=float(data["region_token_flops"]),
                output=output,
                halt_at=int(data["halt_at"]),
            )
        except KeyError as exc:
            raise ScheduleViolationError(f"G30: schedule is missing required field {exc}.") from exc

    @staticmethod
    def from_json(raw: str) -> Schedule:
        """Parse a `Schedule` from a JSON string; see `from_dict`."""
        return Schedule.from_dict(json.loads(raw))

    @staticmethod
    def assemble(
        *,
        context_tokens: Mapping[str, int],
        read_tokens: Mapping[str, int],
        admitted: Mapping[str, Sequence[bool]],
        condition: Mapping[str, bool],
        precision: Mapping[str, str],
        priority: Mapping[str, int],
        region_token_flops: float,
        step_budget: StepBudget,
        output: OutputSpec,
        halt_at: int,
        resident: Mapping[str, bool] | None = None,
        codec: Mapping[str, str] | None = None,
    ) -> Schedule:
        """Emit a `Schedule` from already-computed per-participant values (spec §3 step 4).

        This is the "emit" half of the module's job: the ctx/b/A/halt values
        themselves come from `controller.py` (a later lane); this method only packages
        them, deriving `active` and `depth` from `admitted` rather than accepting them
        as separate inputs, so those two fields cannot independently disagree with the
        array that defines them. `resident` and `codec` default to Table 8a's only v1
        values (`True`, `"none"`) when omitted.

        Args:
            context_tokens: `ctx_r` per participant name.
            read_tokens: `b_r` per participant name.
            admitted: `A[:, r]` per participant name, length `n_iter` each.
            condition: Whether each participant's conditioning prefix is applied.
            precision: Each participant's precision, one of `PRECISIONS`.
            priority: Each participant's declared rank (spec Table 8a).
            region_token_flops: The schedule's predicted total (see the class
                docstring); computed by the caller.
            step_budget: The request's declared ceilings.
            output: What the schedule declares it will emit.
            halt_at: The realised iteration bound.
            resident: Per-participant residency; defaults to `True` for every
                participant when omitted (Table 8a).
            codec: Per-participant codec; defaults to `"none"` for every participant
                when omitted (Table 8a).

        Returns:
            The assembled `Schedule`, not yet validated -- call
            `ScheduleValidator.validate` before executing it.

        Raises:
            ScheduleViolationError: the per-participant mappings do not all cover the same
                set of participant names.
        """
        participants = tuple(context_tokens)
        required = set(participants)
        for name, mapping in (
            ("read_tokens", read_tokens),
            ("admitted", admitted),
            ("condition", condition),
            ("precision", precision),
            ("priority", priority),
        ):
            if set(mapping) != required:
                raise ScheduleViolationError(
                    f"G30: assemble() requires one {name!r} entry per participant in "
                    f"context_tokens; got {sorted(mapping)} for {sorted(required)}."
                )
        nodes = []
        for region in participants:
            admitted_row = tuple(bool(a) for a in admitted[region])
            is_active = any(admitted_row)
            depth = next((i for i, a in enumerate(admitted_row) if a), len(admitted_row))
            nodes.append(
                ScheduleNode(
                    region=region,
                    active=is_active,
                    depth=depth,
                    admitted=admitted_row,
                    context_tokens=int(context_tokens[region]),
                    read_tokens=int(read_tokens[region]),
                    condition=bool(condition[region]),
                    priority=int(priority[region]),
                    precision=str(precision[region]),
                    resident=True if resident is None else bool(resident[region]),
                    codec=CODEC_V1 if codec is None else str(codec[region]),
                )
            )
        return Schedule(
            nodes=tuple(nodes),
            step_budget=step_budget,
            region_token_flops=float(region_token_flops),
            output=output,
            halt_at=int(halt_at),
        )


_STEP_BUDGET_FIELDS: frozenset[str] = frozenset(f.name for f in fields(StepBudget))
_OUTPUT_FIELDS: frozenset[str] = frozenset(f.name for f in fields(OutputSpec))
_NODE_FIELDS: frozenset[str] = frozenset(f.name for f in fields(ScheduleNode))
_SCHEDULE_FIELDS: frozenset[str] = frozenset(f.name for f in fields(Schedule))


def _check_keys(data: Mapping[str, Any], allowed: frozenset[str], *, where: str) -> None:
    """Raise `ScheduleViolationError` if `data` carries a forbidden or unrecognised key.

    Args:
        data: The candidate mapping to check.
        allowed: The exact key set `data` may contain.
        where: A short label for the error message (e.g. `"schedule node"`).

    Raises:
        ScheduleViolationError: `data` has a key outside `allowed`.
    """
    extra = set(data) - allowed
    if not extra:
        return
    forbidden = extra & FORBIDDEN_KEYS
    if forbidden:
        raise ScheduleViolationError(
            f"G30: {where} carries forbidden field(s) {sorted(forbidden)} -- trace_id is "
            "server-minted and a store scope/namespace is never a model field "
            "(INTERCONNECT-MODULE-SPEC.md §3 step 4, §4 Table 8a)."
        )
    raise ScheduleViolationError(f"G30: {where} carries unrecognised field(s) {sorted(extra)}.")


def _node_from_dict(data: Mapping[str, Any]) -> ScheduleNode:
    """Parse one `ScheduleNode` from a dict; structural checks only, see `Schedule.from_dict`."""
    _check_keys(data, _NODE_FIELDS, where="schedule node")
    try:
        return ScheduleNode(
            region=str(data["region"]),
            active=bool(data["active"]),
            depth=int(data["depth"]),
            admitted=tuple(bool(a) for a in data["admitted"]),
            context_tokens=int(data["context_tokens"]),
            read_tokens=int(data["read_tokens"]),
            condition=bool(data["condition"]),
            priority=int(data["priority"]),
            precision=str(data["precision"]),
            resident=bool(data["resident"]),
            codec=str(data["codec"]),
        )
    except KeyError as exc:
        raise ScheduleViolationError(
            f"G30: schedule node is missing required field {exc}."
        ) from exc


def _step_budget_from_dict(data: Mapping[str, Any]) -> StepBudget:
    """Parse a `StepBudget` from a dict; structural checks only, see `Schedule.from_dict`."""
    _check_keys(data, _STEP_BUDGET_FIELDS, where="step_budget")
    try:
        return StepBudget(
            max_iters=int(data["max_iters"]),
            kv_bytes=int(data["kv_bytes"]),
            read_tokens=int(data["read_tokens"]),
            wall_ms=float(data["wall_ms"]),
            flops_ceiling=float(data["flops_ceiling"]),
        )
    except KeyError as exc:
        raise ScheduleViolationError(f"G30: step_budget is missing required field {exc}.") from exc


def _output_from_dict(data: Mapping[str, Any]) -> OutputSpec:
    """Parse an `OutputSpec` from a dict; structural checks only, see `Schedule.from_dict`."""
    _check_keys(data, _OUTPUT_FIELDS, where="output")
    try:
        modalities = tuple(str(m) for m in data["modalities"])
    except KeyError as exc:
        raise ScheduleViolationError(f"G30: output is missing required field {exc}.") from exc
    return OutputSpec(
        modalities=modalities,
        stream=bool(data.get("stream", False)),
        first_token_ms=data.get("first_token_ms"),
        speech_frame_ms=data.get("speech_frame_ms"),
    )


@dataclass(frozen=True)
class ParticipantBudget:
    """Per-participant bounds `ScheduleValidator` needs, from spec Table 1 / Table 4a.

    Not a restatement of `faculty.protocol.Faculty`: only the fields this file's B1
    bounds read. See the module docstring's "spec silences" section for why this class
    exists (Table 2 does not name a type for `ScheduleValidator`'s `participants` arg).

    Attributes:
        ctx_min: Table 4a's floor, or `None` for a participant with no `ctx` axis (the
            store: "the store has no `ctx`, its context is the scope partition",
            spec §2.2 Table 3).
        ctx_max: Table 4a's ceiling; `None` exactly when `ctx_min` is `None`.
        kv_bytes_per_token: `c_r`, or `None` when `ctx_min` is `None`.
        token_budget_min: Table 1's `token_budget` minimum (`lo_r`'s other operand).
        token_budget_max: Table 1's `token_budget` maximum, `hi_r`.
        accepts_condition: Table 1's `accepts_condition`.
    """

    ctx_min: int | None
    ctx_max: int | None
    kv_bytes_per_token: int | None
    token_budget_min: int
    token_budget_max: int
    accepts_condition: bool

    def __post_init__(self) -> None:
        """Enforce that `ctx_min`/`ctx_max`/`kv_bytes_per_token` are `None` together."""
        ctx_fields = (self.ctx_min, self.ctx_max, self.kv_bytes_per_token)
        if any(f is None for f in ctx_fields) and not all(f is None for f in ctx_fields):
            raise ScheduleValidatorConfigError(
                "ParticipantBudget: ctx_min, ctx_max and kv_bytes_per_token must be "
                f"all None (no ctx axis) or all set; got {ctx_fields}."
            )
        if self.ctx_min is not None and self.ctx_max is not None and self.ctx_min > self.ctx_max:
            raise ScheduleValidatorConfigError(
                f"ParticipantBudget: ctx_min={self.ctx_min} > ctx_max={self.ctx_max}."
            )
        if self.token_budget_min > self.token_budget_max:
            raise ScheduleValidatorConfigError(
                "ParticipantBudget: token_budget_min="
                f"{self.token_budget_min} > token_budget_max={self.token_budget_max}."
            )


class ScheduleValidator:
    """Enforces every spec §9.9 B1 bound on a pre-execution `Schedule` -- G30.

    Constructed once per `InterconnectConfig` (spec §2.1 Table 2's constructor
    signature); `validate` is then called per request. Construction itself refuses an
    infeasible configuration (spec §3 step 2's two named refusals) with
    `ScheduleValidatorConfigError`, distinct from the per-`Schedule`
    `ScheduleViolationError` `validate` raises.
    """

    def __init__(
        self,
        participants: Mapping[str, ParticipantBudget],
        B_read: int,  # noqa: N803 -- spec's own constructor arg name (Table 2)
        B_kv: int,  # noqa: N803 -- spec's own constructor arg name (Table 2)
        n_iter: int,
        eta: float,
        flops_ceiling: float,
        allowed_modalities: Collection[str],
        resident_heads: Collection[str],
    ) -> None:
        """Build a validator over a fixed participant set and budget configuration.

        Args:
            participants: Every participant this validator will ever see, by name.
            B_read: The workspace bank's total read-token slots (spec §1: 256).
            B_kv: The total KV/activation byte budget (spec §1: `budget_total_kv_bytes`).
            n_iter: The workspace's iteration count; also the ceiling on
                `step_budget.max_iters` (spec §2.1, "One name for the iteration count").
            eta: The collapse floor `η` (spec §1: `floor_eta`, default 0.15).
            flops_ceiling: The module's configured FLOPs ceiling; a `Schedule` whose
                `region_token_flops` exceeds this is refused.
            allowed_modalities: Every modality an `OutputSpec` may declare.
            resident_heads: Every modality that currently has a resident head.

        Raises:
            ScheduleValidatorConfigError: `participants` is empty; `B_read`, `B_kv` or
                `n_iter` is not positive; `eta` is outside `[0, 1]`; `flops_ceiling` is
                not positive; some participant's `ctx_min` cannot be reached within its
                floor share of `B_kv` (spec §3 step 2, "clipping up cannot [break the
                byte bound] because construction refuses a config where
                `η/R_ctx · B_kv < c_r · ctx_min`"); or the token-budget
                box is infeasible ("construction refuses `Σ lo > B_read` or
                `Σ hi < B_read`").
        """
        if not participants:
            raise ScheduleValidatorConfigError("ScheduleValidator needs at least one participant.")
        if n_iter < 1:
            raise ScheduleValidatorConfigError(f"n_iter must be >= 1, got {n_iter}.")
        if B_read < 1:
            raise ScheduleValidatorConfigError(f"B_read must be >= 1, got {B_read}.")
        if B_kv < 1:
            raise ScheduleValidatorConfigError(f"B_kv must be >= 1, got {B_kv}.")
        if not (0.0 <= eta <= 1.0):
            raise ScheduleValidatorConfigError(f"eta must be in [0, 1], got {eta}.")
        if flops_ceiling <= 0:
            raise ScheduleValidatorConfigError(
                f"flops_ceiling must be positive, got {flops_ceiling}."
            )

        self.participants: dict[str, ParticipantBudget] = dict(participants)
        self.B_read = B_read
        self.B_kv = B_kv
        self.n_iter = n_iter
        self.eta = eta
        self.flops_ceiling = flops_ceiling
        self.allowed_modalities = frozenset(allowed_modalities)
        self.resident_heads = frozenset(resident_heads)

        ctx_participants = {n: p for n, p in self.participants.items() if p.ctx_min is not None}
        r_ctx = len(ctx_participants) or 1
        for name, p in ctx_participants.items():
            floor_bytes = (eta / r_ctx) * B_kv
            needed_bytes = (p.kv_bytes_per_token or 0) * (p.ctx_min or 0)
            if floor_bytes < needed_bytes:
                raise ScheduleValidatorConfigError(
                    f"ScheduleValidator: participant {name!r} cannot reach its own "
                    f"ctx_min ({p.ctx_min}) within its floor share of B_kv: "
                    f"eta/R_ctx*B_kv={floor_bytes:.1f} < c_r*ctx_min={needed_bytes} "
                    "(INTERCONNECT-MODULE-SPEC.md §3 step 2)."
                )

        r_total = len(self.participants)
        lo = {
            name: read_token_floor(p.token_budget_min, eta, r_total, B_read)
            for name, p in self.participants.items()
        }
        hi = {name: p.token_budget_max for name, p in self.participants.items()}
        if sum(lo.values()) > B_read:
            raise ScheduleValidatorConfigError(
                f"ScheduleValidator: sum of token-budget floors {sum(lo.values())} "
                f"exceeds B_read={B_read} (INTERCONNECT-MODULE-SPEC.md §3 step 2)."
            )
        if sum(hi.values()) < B_read:
            raise ScheduleValidatorConfigError(
                f"ScheduleValidator: sum of token-budget ceilings {sum(hi.values())} "
                f"falls short of B_read={B_read} (INTERCONNECT-MODULE-SPEC.md §3 step 2)."
            )
        self._lo = lo
        self._hi = hi

    def validate(self, schedule: Schedule) -> Schedule:
        """Refuse `schedule` if it violates any B1 bound; otherwise return it unchanged.

        Pure and deterministic: never mutates `schedule` (it is frozen) or `self`, and
        two calls with the same `schedule` always agree, in either order, regardless of
        how many other schedules were validated in between (spec §5 Table 9, "the
        validator is pure and deterministic").

        Args:
            schedule: The `Schedule` to check, typically freshly built by
                `Schedule.assemble` or parsed by `Schedule.from_dict`.

        Returns:
            `schedule`, unchanged, when every bound holds.

        Raises:
            ScheduleViolationError: any bound this method checks does not hold. The message
                names the specific bound and the spec location it comes from.
        """
        seen: set[str] = set()
        total_kv_bytes = 0
        total_read_tokens = 0
        priorities: list[int] = []

        for node in schedule.nodes:
            if node.region not in self.participants:
                raise ScheduleViolationError(
                    f"G30: schedule names unknown participant {node.region!r}."
                )
            if node.region in seen:
                raise ScheduleViolationError(
                    f"G30: participant {node.region!r} appears more than once."
                )
            seen.add(node.region)
            p = self.participants[node.region]
            self._check_admission(node)
            total_kv_bytes += self._check_context_tokens(node, p)
            total_read_tokens += self._check_read_tokens(node)
            self._check_node_flags(node, p)
            priorities.append(node.priority)

        self._check_totals(schedule, seen, priorities, total_read_tokens, total_kv_bytes)
        self._check_step_budget(schedule, total_read_tokens, total_kv_bytes)
        self._check_output(schedule)
        return schedule

    def _check_admission(self, node: ScheduleNode) -> None:
        """One node's `admitted`/`depth`/`active` agreement (spec §3, admission-matrix
        semantics, and §3 step 3's forced admission).

        Args:
            node: The node to check.

        Raises:
            ScheduleViolationError: the admission row is the wrong length, is all-zero,
                or disagrees with the node's own `depth` or `active`.
        """
        if len(node.admitted) != self.n_iter:
            raise ScheduleViolationError(
                f"G30: {node.region}: admitted has length {len(node.admitted)}, "
                f"expected n_iter={self.n_iter}."
            )
        if not any(node.admitted):
            raise ScheduleViolationError(
                f"G30: {node.region}: admitted is all-zero; every declared "
                "participant must be admitted at least once (INTERCONNECT-MODULE-"
                "SPEC.md §3 step 3, forced admission)."
            )
        expected_depth = next(i for i, a in enumerate(node.admitted) if a)
        if node.depth != expected_depth:
            raise ScheduleViolationError(
                f"G30: {node.region}: depth={node.depth} disagrees with "
                f"min{{i: admitted[i]}}={expected_depth}."
            )
        if node.active != any(node.admitted):
            raise ScheduleViolationError(
                f"G30: {node.region}: active={node.active} disagrees with admitted."
            )

    @staticmethod
    def _check_context_tokens(node: ScheduleNode, p: ParticipantBudget) -> int:
        """One node's `context_tokens` against its participant's ctx axis (Table 4a).

        Args:
            node: The node to check.
            p: That node's participant budget.

        Returns:
            This node's KV byte contribution, `c_r · ctx_r`, or `0` when the participant
            has no ctx axis.

        Raises:
            ScheduleViolationError: `context_tokens` is outside `[ctx_min, ctx_max]`, or
                is non-zero for a participant that has no ctx axis (the store).
        """
        if p.ctx_min is None:
            if node.context_tokens != 0:
                raise ScheduleViolationError(
                    f"G30: {node.region}: has no ctx axis but context_tokens="
                    f"{node.context_tokens} != 0."
                )
            return 0
        if not (p.ctx_min <= node.context_tokens <= (p.ctx_max or p.ctx_min)):
            raise ScheduleViolationError(
                f"G30: {node.region}: context_tokens={node.context_tokens} "
                f"outside [{p.ctx_min}, {p.ctx_max}]."
            )
        return (p.kv_bytes_per_token or 0) * node.context_tokens

    def _check_read_tokens(self, node: ScheduleNode) -> int:
        """One node's `read_tokens` against its `[lo_r, hi_r]` box (spec §3 step 2).

        Args:
            node: The node to check.

        Returns:
            This node's `read_tokens`, for the caller's running total.

        Raises:
            ScheduleViolationError: `read_tokens` is outside the box.
        """
        lo, hi = self._lo[node.region], self._hi[node.region]
        if not (lo <= node.read_tokens <= hi):
            raise ScheduleViolationError(
                f"G30: {node.region}: read_tokens={node.read_tokens} outside [{lo}, {hi}]."
            )
        return node.read_tokens

    @staticmethod
    def _check_node_flags(node: ScheduleNode, p: ParticipantBudget) -> None:
        """One node's declared-value fields: `condition`, `precision`, `resident`, `codec`.

        Args:
            node: The node to check.
            p: That node's participant budget.

        Raises:
            ScheduleViolationError: `condition` is set on a participant that does not
                accept one, `precision` is outside `PRECISIONS`, or `resident`/`codec`
                carries a value v1 does not implement (Table 8a).
        """
        if node.condition and not p.accepts_condition:
            raise ScheduleViolationError(
                f"G30: {node.region}: condition=True but this participant's "
                "accepts_condition is False."
            )
        if node.precision not in PRECISIONS:
            raise ScheduleViolationError(
                f"G30: {node.region}: precision {node.precision!r} outside {sorted(PRECISIONS)}."
            )
        if not node.resident:
            raise ScheduleViolationError(
                f"G30: {node.region}: resident=False is not supported at v1 (Table 8a)."
            )
        if node.codec != CODEC_V1:
            raise ScheduleViolationError(
                f"G30: {node.region}: codec {node.codec!r} != {CODEC_V1!r}, the only "
                "v1 value (Table 8a)."
            )

    def _check_totals(
        self,
        schedule: Schedule,
        seen: Collection[str],
        priorities: Sequence[int],
        total_read_tokens: int,
        total_kv_bytes: int,
    ) -> None:
        """The schedule-wide sums the per-node loop accumulated.

        Args:
            schedule: The schedule being validated.
            seen: Every participant name the nodes named.
            priorities: Every node's declared priority, in node order.
            total_read_tokens: `Σ_r b_r`.
            total_kv_bytes: `Σ_r c_r · ctx_r`.

        Raises:
            ScheduleViolationError: a declared participant has no node, the priorities
                are not a `0..R-1` permutation, `Σ_r b_r != B_read`, or the KV total
                exceeds `B_kv`.
        """
        missing = set(self.participants) - set(seen)
        if missing:
            raise ScheduleViolationError(
                f"G30: schedule omits declared participant(s) {sorted(missing)}."
            )
        if sorted(priorities) != list(range(len(schedule.nodes))):
            raise ScheduleViolationError(
                f"G30: node priorities {list(priorities)} are not a 0..R-1 permutation."
            )
        if total_read_tokens != self.B_read:
            raise ScheduleViolationError(
                f"G30: sum of read_tokens {total_read_tokens} != B_read={self.B_read}."
            )
        if total_kv_bytes > self.B_kv:
            raise ScheduleViolationError(
                f"G30: total kv bytes {total_kv_bytes} exceeds B_kv={self.B_kv}."
            )

    def _check_step_budget(
        self, schedule: Schedule, total_read_tokens: int, total_kv_bytes: int
    ) -> None:
        """`step_budget`, `halt_at` and the two FLOPs bounds (spec §2.1, Table 3).

        Args:
            schedule: The schedule being validated.
            total_read_tokens: The total the nodes actually declare.
            total_kv_bytes: The KV total the nodes actually declare.

        Raises:
            ScheduleViolationError: `max_iters` or `halt_at` is out of range, a declared
                budget total disagrees with the computed one, `wall_ms` is not positive,
                or either FLOPs figure exceeds its ceiling.
        """
        sb = schedule.step_budget
        if not (1 <= sb.max_iters <= self.n_iter):
            raise ScheduleViolationError(
                f"G30: step_budget.max_iters={sb.max_iters} outside [1, n_iter={self.n_iter}]."
            )
        if not (1 <= schedule.halt_at <= sb.max_iters):
            raise ScheduleViolationError(
                f"G30: halt_at={schedule.halt_at} outside [1, max_iters={sb.max_iters}]."
            )
        if sb.read_tokens != total_read_tokens:
            raise ScheduleViolationError(
                f"G30: step_budget.read_tokens={sb.read_tokens} != computed total "
                f"{total_read_tokens}."
            )
        if sb.kv_bytes != total_kv_bytes:
            raise ScheduleViolationError(
                f"G30: step_budget.kv_bytes={sb.kv_bytes} != computed total {total_kv_bytes}."
            )
        if sb.wall_ms <= 0:
            raise ScheduleViolationError(
                f"G30: step_budget.wall_ms must be positive, got {sb.wall_ms}."
            )
        if sb.flops_ceiling > self.flops_ceiling:
            raise ScheduleViolationError(
                f"G30: step_budget.flops_ceiling={sb.flops_ceiling} exceeds this "
                f"validator's flops_ceiling={self.flops_ceiling}."
            )
        if schedule.region_token_flops > sb.flops_ceiling:
            raise ScheduleViolationError(
                f"G30: region_token_flops={schedule.region_token_flops} exceeds "
                f"step_budget.flops_ceiling={sb.flops_ceiling}."
            )
        if schedule.region_token_flops > self.flops_ceiling:
            raise ScheduleViolationError(
                f"G30: region_token_flops={schedule.region_token_flops} exceeds "
                f"flops_ceiling={self.flops_ceiling}."
            )

    def _check_output(self, schedule: Schedule) -> None:
        """`schedule.output` against the allowed modalities, resident heads and Table 8a's
        v1 defaults.

        Args:
            schedule: The schedule being validated.

        Raises:
            ScheduleViolationError: a modality is not allowed or has no resident head, or
                `stream`/`first_token_ms`/`speech_frame_ms` carries a non-v1 value.
        """
        for modality in schedule.output.modalities:
            if modality not in self.allowed_modalities:
                raise ScheduleViolationError(
                    f"G30: output modality {modality!r} is not in allowed_modalities."
                )
            if modality not in self.resident_heads:
                raise ScheduleViolationError(
                    f"G30: output modality {modality!r} has no resident head."
                )
        if schedule.output.stream:
            raise ScheduleViolationError(
                "G30: output.stream=True is not supported at v1 (Table 8a)."
            )
        if (
            schedule.output.first_token_ms is not None
            or schedule.output.speech_frame_ms is not None
        ):
            raise ScheduleViolationError(
                "G30: first_token_ms and speech_frame_ms must be null at v1 (Table 8a)."
            )
