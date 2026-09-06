"""DEC-63's dynamic byte capacity: section 8 gap (a)'s formula and its pre-committed floor.

THE FORMULA, verbatim from section 8 gap (a) of
`docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md`:

    capacity_bytes(host, tick) = max(0, VRAM_total(host)
                                      - KV_reserved(context_len, regions_active)
                                      - activation_reserve
                                      - safety_margin)

and its own instruction about how it is to be evaluated: *"computed as a pure function of live
inputs, per host, at scheduler-tick time and never cached -- because the 1080 Ti is preemptible
and the active-region set changes under the store, so a capacity fixed at process start is not
dynamic."* Every function here is therefore pure and every dataclass frozen; the only impure
thing in this module is `probe_host_vram`, which shells out to `nvidia-smi` and is the "live
input" the formula is a function of. Nothing memoises anything.

THE FLOOR IS PRE-COMMITTED, NOT DISCOVERED AT DEPLOYMENT. Section 9.11 found that the residual
may round to zero on the 5080 -- the 16 GiB card that is the deployment target -- and gap (a)
names the alternative: *"a fixed floor, reserved for the store before the KV budget is
computed, sized at `safety_margin`'s order (2 GiB)."* `capacity_decision` implements both
branches and always reports which one fired, in `CapacityDecision.branch`. A store with a
capacity of zero on the card it will actually run on is the failure this branch exists to
prevent, so "residual came out zero, therefore the store has no capacity" is not a reachable
outcome of this module: it is either a positive residual or the floor.

"RESERVED BEFORE THE KV BUDGET IS COMPUTED" is what makes the floor branch different from a
`max(residual, floor)`. In the floor branch the store's slice is taken off `VRAM_total` first
and the scheduler's KV budget is what remains; `CapacityDecision.kv_headroom_bytes` reports
that remainder so the caller can see what the floor cost the context, which is the product
trade section 8 says this choice is. A `max()` would instead hand the store 2 GiB that the KV
cache has already been promised, i.e. overcommit the card.

WHAT THIS MODULE DOES NOT DECIDE. `safety_margin` is section 8's one free parameter and it is
the operator's; the default here is 2 GiB, the same order as `hypha`'s display reserve, which
is what gap (a) recommends starting at. `KV_reserved` is the scheduler's number, not this
module's: `SchedulerBudgets` carries it as data, with a helper that computes it from a context
length and an active-region set for the probe's benefit. Changing either changes a constant,
not a shape.

NO TORCH. This module imports the standard library only. The 5080 half of E1's capacity probe
runs inside the GPU CI runner's container, and requiring a working torch there to read a
number out of `nvidia-smi` would make the gate skippable for a reason that has nothing to do
with the gate. `store.py` is where torch appears.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "DEFAULT_ACTIVATION_RESERVE_BYTES",
    "DEFAULT_KV_BYTES_PER_TOKEN_PER_REGION",
    "DEFAULT_SAFETY_MARGIN_BYTES",
    "DEFAULT_STORE_FLOOR_BYTES",
    "GIB",
    "MIB",
    "CapacityBranch",
    "CapacityDecision",
    "HostVram",
    "NvidiaSmiError",
    "SchedulerBudgets",
    "capacity_decision",
    "kv_reserved_bytes",
    "probe_host_vram",
]

MIB = 1024 * 1024
"""One mebibyte, in bytes. `nvidia-smi` reports VRAM in these."""

GIB = 1024 * 1024 * 1024
"""One gibibyte, in bytes. Every reserve below is expressed as a multiple of this."""

DEFAULT_SAFETY_MARGIN_BYTES = 2 * GIB
"""Section 8 gap (a)'s one free parameter, at its recommended starting value: *"the same
order as `hypha`'s display reserve (2 GiB)"*. The operator's to change; a change here is a
constant, not a redesign."""

DEFAULT_ACTIVATION_RESERVE_BYTES = 1 * GIB
"""The forward working set the formula subtracts alongside the KV cache. 1 GiB is `hypha`'s
own CUDA-scratch reserve (`facade/hypha.rs:13-26`, 1024 MiB), which is the only measured
number either source repo has for this term -- so it is borrowed rather than invented, and it
is a caller-overridable field on `SchedulerBudgets`, not a hidden constant."""

DEFAULT_STORE_FLOOR_BYTES = 2 * GIB
"""Gap (a)'s named alternative when the residual rounds to zero: a fixed floor *"sized at
`safety_margin`'s order (2 GiB)"*."""

DEFAULT_KV_BYTES_PER_TOKEN_PER_REGION = 2 * 512 * 2
"""K and V, one workspace latent each, at `D_w = 512` in fp16: `2 (k,v) * 512 * 2 bytes` =
2048 B per token per active region. This is the interconnect's own width (section 2.3), not a
serving-engine estimate, and it is a default the probe prints and a caller can override --
`KV_reserved` is the scheduler's number and this is only how the probe fills it in when it is
computing a capacity for a hypothetical tick rather than reading one off a live scheduler."""


class NvidiaSmiError(RuntimeError):
    """`nvidia-smi` is absent, failed, or returned something unparseable.

    Raised rather than returning a plausible default: a capacity computed from a guessed
    `VRAM_total` is exactly the "measured a constant" failure E1's gate (ii) refuses, and it
    would be indistinguishable from a real measurement in the receipt.
    """


class CapacityBranch(str, Enum):
    """Which of gap (a)'s two branches produced a `CapacityDecision`."""

    RESIDUAL = "residual"
    """The subtractive formula returned a positive number and it was used as-is."""

    FLOOR = "floor"
    """The residual was zero or negative, so the pre-committed fixed floor fired: a slice
    reserved for the store ahead of the KV budget (section 9.11's 5080 case)."""


@dataclass(frozen=True, slots=True)
class HostVram:
    """`VRAM_total(host)` -- one card's total memory, and enough identity to name it.

    Frozen because the formula is a pure function of it; a mutable reading could be updated
    between two calls and make two capacities differ for a reason that is not the tick.
    """

    host: str
    """Operator-facing host label, e.g. `"akula-prime"` or `"gpu5080"`."""

    device_name: str
    """`nvidia-smi`'s own name for the card, e.g. `"NVIDIA GeForce RTX 5080"`."""

    total_bytes: int
    """Total VRAM in bytes, converted from `nvidia-smi`'s MiB."""

    index: int = 0
    """CUDA index of the card on that host."""


@dataclass(frozen=True, slots=True)
class SchedulerBudgets:
    """The scheduler's side of the formula at one tick: what the KV cache and the forward pass
    have already claimed.

    `kv_reserved_bytes` is normally read off a live scheduler. `from_context` builds one from a
    context length and an active-region set, which is what E1's probe does when it is asked for
    the capacity at a hypothetical tick (and is how gate (ii)'s "the value changes when the
    active-region set changes" assertion is constructed).
    """

    kv_reserved_bytes: int
    """`KV_reserved(context_len, regions_active)` at this tick, in bytes."""

    activation_reserve_bytes: int = DEFAULT_ACTIVATION_RESERVE_BYTES
    """The forward working set, in bytes."""

    context_len: int | None = None
    """The context length this KV reserve was computed for, recorded for the receipt. `None`
    when the reserve was read off a live scheduler rather than computed."""

    regions_active: tuple[str, ...] = ()
    """The active-region set this KV reserve was computed for, recorded for the receipt."""

    @staticmethod
    def from_context(
        context_len: int,
        regions_active: tuple[str, ...],
        *,
        kv_bytes_per_token_per_region: int = DEFAULT_KV_BYTES_PER_TOKEN_PER_REGION,
        activation_reserve_bytes: int = DEFAULT_ACTIVATION_RESERVE_BYTES,
    ) -> SchedulerBudgets:
        """Compute `KV_reserved(context_len, regions_active)` and wrap it with the rest.

        Args:
            context_len: Context length in tokens the KV cache is sized for.
            regions_active: The regions holding KV at this tick. Its LENGTH is what makes the
                capacity move when the active set changes.
            kv_bytes_per_token_per_region: Bytes of K plus V per token per region.
            activation_reserve_bytes: The forward working set, in bytes.

        Returns:
            A `SchedulerBudgets` carrying the computed reserve and the inputs it came from.

        Raises:
            ValueError: `context_len` is negative or `kv_bytes_per_token_per_region` is not
                positive.
        """
        if context_len < 0:
            raise ValueError(f"context_len must be >= 0, got {context_len}")
        if kv_bytes_per_token_per_region <= 0:
            raise ValueError(
                f"kv_bytes_per_token_per_region must be > 0, got {kv_bytes_per_token_per_region}"
            )
        return SchedulerBudgets(
            kv_reserved_bytes=kv_reserved_bytes(
                context_len, regions_active, kv_bytes_per_token_per_region
            ),
            activation_reserve_bytes=activation_reserve_bytes,
            context_len=context_len,
            regions_active=tuple(regions_active),
        )


def kv_reserved_bytes(
    context_len: int,
    regions_active: tuple[str, ...],
    kv_bytes_per_token_per_region: int = DEFAULT_KV_BYTES_PER_TOKEN_PER_REGION,
) -> int:
    """`KV_reserved(context, regions_active)` -- linear in both arguments, and deliberately so.

    The formula's own signature makes the reserve a function of the context length AND the
    active-region set, which is the clause that makes capacity dynamic rather than per-host
    constant. A model that ignored `regions_active` would still produce two different numbers
    on two different cards and would still be wrong.

    Args:
        context_len: Context length in tokens.
        regions_active: The regions holding KV at this tick.
        kv_bytes_per_token_per_region: Bytes of K plus V per token per region.

    Returns:
        The KV reserve in bytes.
    """
    return context_len * len(regions_active) * kv_bytes_per_token_per_region


@dataclass(frozen=True, slots=True)
class CapacityDecision:
    """One evaluation of gap (a)'s formula: the number, the branch, and every input.

    Every subtrahend is carried so the receipt can be re-derived by hand from the record
    alone -- E1's gate (ii) requires the receipt to record *which* of residual and floor is
    active on each card, and a bare number cannot show that.
    """

    host: str
    """Host label the capacity was computed for."""

    device_name: str
    """The card, as `nvidia-smi` names it."""

    capacity_bytes: int
    """The answer: bytes the store may hold resident on this host at this tick."""

    branch: CapacityBranch
    """Which of gap (a)'s two branches produced `capacity_bytes`."""

    vram_total_bytes: int
    """`VRAM_total(host)`."""

    kv_reserved_bytes: int
    """`KV_reserved(context_len, regions_active)` at this tick."""

    activation_reserve_bytes: int
    """The forward working set."""

    safety_margin_bytes: int
    """Section 8's one free parameter."""

    residual_bytes: int
    """What the subtractive formula returned, floored at zero. Equal to `capacity_bytes` on
    the residual branch; zero on the floor branch, which is why the floor fired."""

    floor_bytes: int
    """The pre-committed floor that would fire (and, on the floor branch, did)."""

    kv_headroom_bytes: int
    """What is left for the KV cache after the store's claim. On the residual branch this is
    just `kv_reserved_bytes` -- the store took what was left over and cost the context
    nothing. On the floor branch the floor is reserved FIRST, so this is what remains of
    `VRAM_total` after the floor, the activation reserve and the safety margin: the context
    length the floor bought its recall with."""

    context_len: int | None = None
    """Context length behind `kv_reserved_bytes`, when it was computed rather than read."""

    regions_active: tuple[str, ...] = ()
    """Active-region set behind `kv_reserved_bytes`, when it was computed rather than read."""

    def to_dict(self) -> dict[str, Any]:
        """Render as JSON-safe primitives for a receipt.

        Returns:
            A dict of plain types; `branch` becomes its string value and `regions_active` a
            list, so `json.dump` needs no custom encoder.
        """
        return {
            "host": self.host,
            "device_name": self.device_name,
            "capacity_bytes": self.capacity_bytes,
            "capacity_mib": round(self.capacity_bytes / MIB, 3),
            "branch": self.branch.value,
            "vram_total_bytes": self.vram_total_bytes,
            "vram_total_mib": round(self.vram_total_bytes / MIB, 3),
            "kv_reserved_bytes": self.kv_reserved_bytes,
            "activation_reserve_bytes": self.activation_reserve_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "residual_bytes": self.residual_bytes,
            "floor_bytes": self.floor_bytes,
            "kv_headroom_bytes": self.kv_headroom_bytes,
            "context_len": self.context_len,
            "regions_active": list(self.regions_active),
        }


def capacity_decision(
    vram: HostVram,
    budgets: SchedulerBudgets,
    *,
    safety_margin_bytes: int = DEFAULT_SAFETY_MARGIN_BYTES,
    floor_bytes: int = DEFAULT_STORE_FLOOR_BYTES,
) -> CapacityDecision:
    """Evaluate gap (a)'s formula for one host at one tick, taking the floor branch if needed.

    Pure: called twice with the same arguments it returns the same answer, and it holds no
    state between calls, so a caller that wants a dynamic capacity gets one by calling it
    every tick and a caller that caches the result has opted out of DEC-63 visibly.

    Args:
        vram: The live `VRAM_total(host)` reading, from `probe_host_vram`.
        budgets: The scheduler's KV and activation claims at this tick.
        safety_margin_bytes: Section 8's free parameter.
        floor_bytes: The pre-committed floor for the zero-residual case.

    Returns:
        A `CapacityDecision` whose `branch` says which of the two produced the number.

    Raises:
        ValueError: `floor_bytes` is not positive, or the floor cannot fit on this card at all
            (`floor + activation_reserve + safety_margin > VRAM_total`) -- a card that small
            cannot host the store under this policy and saying so is better than returning a
            capacity the card cannot honour.
    """
    if floor_bytes <= 0:
        raise ValueError(f"floor_bytes must be > 0, got {floor_bytes}")

    residual = (
        vram.total_bytes
        - budgets.kv_reserved_bytes
        - budgets.activation_reserve_bytes
        - safety_margin_bytes
    )
    residual = max(0, residual)

    if residual > 0:
        return CapacityDecision(
            host=vram.host,
            device_name=vram.device_name,
            capacity_bytes=residual,
            branch=CapacityBranch.RESIDUAL,
            vram_total_bytes=vram.total_bytes,
            kv_reserved_bytes=budgets.kv_reserved_bytes,
            activation_reserve_bytes=budgets.activation_reserve_bytes,
            safety_margin_bytes=safety_margin_bytes,
            residual_bytes=residual,
            floor_bytes=floor_bytes,
            kv_headroom_bytes=budgets.kv_reserved_bytes,
            context_len=budgets.context_len,
            regions_active=budgets.regions_active,
        )

    # Floor branch: the store's slice comes off VRAM_total BEFORE the KV budget, so what the
    # KV cache gets is the remainder rather than what it asked for. That is the trade section
    # 8 describes ("trades context length for recall"), made visible in kv_headroom_bytes
    # instead of silently overcommitting the card with a max(residual, floor).
    kv_headroom = (
        vram.total_bytes - floor_bytes - budgets.activation_reserve_bytes - safety_margin_bytes
    )
    if kv_headroom < 0:
        raise ValueError(
            f"{vram.host}: the {floor_bytes}-byte store floor does not fit on "
            f"{vram.device_name} ({vram.total_bytes} bytes total) alongside an activation "
            f"reserve of {budgets.activation_reserve_bytes} and a safety margin of "
            f"{safety_margin_bytes}. This card cannot host the store under this policy."
        )
    return CapacityDecision(
        host=vram.host,
        device_name=vram.device_name,
        capacity_bytes=floor_bytes,
        branch=CapacityBranch.FLOOR,
        vram_total_bytes=vram.total_bytes,
        kv_reserved_bytes=budgets.kv_reserved_bytes,
        activation_reserve_bytes=budgets.activation_reserve_bytes,
        safety_margin_bytes=safety_margin_bytes,
        residual_bytes=0,
        floor_bytes=floor_bytes,
        kv_headroom_bytes=kv_headroom,
        context_len=budgets.context_len,
        regions_active=budgets.regions_active,
    )


def probe_host_vram(*, host: str, index: int = 0, timeout_s: float = 20.0) -> HostVram:
    """Read `VRAM_total` for one card off a live `nvidia-smi`.

    The one impure function in this module, and the "live input" gap (a) requires: a capacity
    derived from a hard-coded card size would produce two different numbers on two hosts and
    still have measured nothing.

    Args:
        host: Host label to stamp on the reading.
        index: CUDA index of the card to read.
        timeout_s: Seconds to wait for `nvidia-smi` before giving up.

    Returns:
        A `HostVram` with the card's name and total bytes.

    Raises:
        NvidiaSmiError: `nvidia-smi` is not on PATH, exited non-zero, timed out, or printed a
            line this function cannot parse into a name and a MiB count.
    """
    binary = shutil.which("nvidia-smi")
    if binary is None:
        raise NvidiaSmiError("nvidia-smi is not on PATH; cannot read VRAM_total for the formula")
    argv = [
        binary,
        f"--id={index}",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise NvidiaSmiError(f"nvidia-smi timed out after {timeout_s}s: {' '.join(argv)}") from exc
    if completed.returncode != 0:
        raise NvidiaSmiError(
            f"nvidia-smi exited {completed.returncode}: {completed.stderr.strip() or '(no stderr)'}"
        )

    line = completed.stdout.strip().splitlines()
    if not line:
        raise NvidiaSmiError(f"nvidia-smi printed nothing for index {index}")
    parts = [field.strip() for field in line[0].split(",")]
    if len(parts) != 2:
        raise NvidiaSmiError(f"cannot parse nvidia-smi row {line[0]!r} as 'name, memory.total'")
    name, total_mib_text = parts
    try:
        total_mib = int(float(total_mib_text))
    except ValueError as exc:
        raise NvidiaSmiError(f"cannot parse {total_mib_text!r} as a MiB count") from exc
    if total_mib <= 0:
        raise NvidiaSmiError(f"nvidia-smi reported {total_mib} MiB total on index {index}")
    return HostVram(host=host, device_name=name, total_bytes=total_mib * MIB, index=index)
