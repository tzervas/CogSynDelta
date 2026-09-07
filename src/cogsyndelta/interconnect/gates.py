"""Receipt-level fail-closed guards -- `docs/design/INTERCONNECT-MODULE-SPEC.md` section 4,
Table 8, rows G27, G28, G29, G33, G34, G35, G36.

WHAT THIS IS
Table 8 splits the interconnect's guards by where they fire: G30, G31 and G32 fire
INSIDE the module at run time (`schedule.py`, `adapters.py`/`kv_bank.py`,
`episodic_store.py` -- other lanes' files); the seven guards here fire OVER RECEIPTS,
after a phase has already run, called by the compose-stage script this module does not
own (spec section 1, "not the compose-stage driver script"). Every guard below is a pure
function of the numbers a receipt already carries -- it takes plain scalars, mappings and
sequences, never a `Receipt` object -- so the compose-stage script can call one after
reading a receipt back from disk without importing this module's shape onto its own.

FAIL-CLOSED, NOT FAIL-WARN
"Each guard raises rather than warns, and each has a constructed failing case in section
5" (spec section 4, the paragraph introducing Table 8). Every function below raises
`GateFailure` -- never returns a bool, never logs and continues -- when the condition
Table 8 names as "fires when" holds, and returns `None` silently otherwise. A guard that
can be satisfied by the caller ignoring its return value is not fail-closed; see
`docs/design/evidence` and the KB note `verify-guards-can-fail` for what happened to three
guards this project already shipped that were shaped that way.

WHY "POINT" IS A MODULE CONSTANT
Table 8's own prose measures two of these guards in "points": G27 ("drops > 1 point") and
the phase-A gate table's overfit gap ("gap >= 5 points", G35). Both are percentage-point
deltas on a `recall@1`-shaped metric already expressed as a fraction in `[0, 1]` in every
receipt this project writes (`regions/_receipt.py`, `eval/metrics.py`) -- "1 point" is
`0.01` of that fraction, not `1.0`. `POINT` names that conversion once so a caller never
has to guess whether a threshold in this file is already in "points" or already a raw
fraction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "POINT",
    "GateFailure",
    "check_attention_mass_floor",
    "check_frozen_set_identity",
    "check_null_gate",
    "check_overfit_gate",
    "check_phase_d_revert",
    "check_topology_agreement",
    "check_write_back_gate",
]

#: One percentage point on a `[0, 1]`-scaled metric such as `recall@1` (spec section 4,
#: Table 6's overfit gate and Table 8's G27, both stated in "points").
POINT = 0.01


class GateFailure(RuntimeError):  # noqa: N818 -- spec section 2.1 Table 2 names this class exactly
    """Raised by every guard in this module when the condition it exists to catch holds.

    Named `GateFailure` rather than a per-guard subclass because Table 2 names exactly
    this one class for `gates.py` -- callers distinguish which guard fired by matching
    the leading `G<NN>:` token in the message (the convention `splits.py`'s
    `SplitGuardError` already uses for G26), not by exception type. `noqa: N818`: ruff's
    exception-naming rule wants an `Error` suffix, which would depart from the spec's own
    class name for no behavioural gain.
    """


def check_write_back_gate(
    region: str,
    *,
    own_bin_before: float,
    own_bin_after: float,
    composed_before: float,
    composed_after: float,
) -> None:
    """G27 -- W5b write-back gate (spec section 4 Table 8; TAX:1435-1439).

    Fires when a conditioned region's own-bin score drops more than one point, OR the
    composed metric does not improve. Both halves are checked independently -- a region
    that holds its own-bin score while the composed metric regresses still fails, and the
    reverse -- because the effect this guard exists to prevent (write-back enabled with no
    net benefit) can be caused by either half alone.

    Args:
        region: Name of the conditioned region being evaluated, for the message only.
        own_bin_before: The region's own-bin score (e.g. `recall@1`, `[0, 1]`) with
            write-back off.
        own_bin_after: The same score with write-back on.
        composed_before: The composed metric with write-back off.
        composed_after: The composed metric with write-back on.

    Raises:
        GateFailure: The own-bin drop exceeds one point, or the composed metric did not
            improve. Effect (applied by the caller, not this function): write-back is
            disabled for `region`; `edges` and `lockstep_groups` are omitted from the
            `Schedule`; `topology: "not demonstrated"` is recorded.
    """
    drop = own_bin_before - own_bin_after
    composed_delta = composed_after - composed_before
    reasons = []
    # `> POINT`, not `>=` -- exactly one point must not fire (spec: "drops > 1 point").
    # The tolerance absorbs float subtraction noise at the boundary (e.g. 0.800 - 0.01
    # landing a few ulps above 0.01) without loosening the threshold itself.
    if drop > POINT and not math.isclose(drop, POINT, rel_tol=1e-9, abs_tol=1e-12):
        reasons.append(f"own-bin score dropped {drop / POINT:.2f} points (limit 1.00)")
    if composed_delta <= 0.0:
        reasons.append(
            f"composed metric did not improve ({composed_before:.4f} -> {composed_after:.4f})"
        )
    if reasons:
        raise GateFailure(
            f"G27: write-back gate fired for region {region!r}: " + "; ".join(reasons)
        )


def check_topology_agreement(agreement_fraction: float, *, item_count: int | None = None) -> None:
    """G28 -- topology agreement between the two derivations (spec section 4 Table 8;
    TAX:1469-1490).

    Fires when the admission-matrix derivation and the cross-check `Ĉ` derivation of
    section 3's "Admission matrix semantics" paragraph agree on fewer than 95% of items.

    Args:
        agreement_fraction: Fraction of items on which the two derivations agree, `[0, 1]`.
        item_count: Number of items the fraction was computed over, for the message only;
            optional because a caller may only have the fraction on hand.

    Raises:
        GateFailure: Agreement is below 95%. Effect (applied by the caller): the
            disagreement is reported as a bug; the topology fields are void, and declared
            void, under the no-write-back fallback.
    """
    if agreement_fraction < 0.95:
        scope = f" over {item_count} items" if item_count is not None else ""
        raise GateFailure(
            f"G28: topology derivations agree on {agreement_fraction:.1%}{scope}, "
            "below the 95% floor"
        )


def check_attention_mass_floor(
    region: str,
    mean_a: float,
    eta: float,
    r: int,
    *,
    printed_floor: float | None = None,
) -> None:
    """G29 -- attention-mass floor (spec section 4 Table 8; TAX:1659-1669, TAX:2661-2662).

    Fires when `region`'s mean admitted attention mass falls below `eta / r` -- the same
    check for a phase-A region and for E2's store (pass `region="store"`) -- or when a
    receipt's own printed floor does not equal `eta / r` at its own `r`. The two failure
    modes are independent: a receipt can print the right floor while a region is genuinely
    collapsed, or print a stale floor computed at a different `R` while every region is
    fine.

    Args:
        region: Name of the region (or `"store"`) being checked, for the message only.
        mean_a: Mean attention mass `a` this region received over active iterations,
            `[0, 1]`.
        eta: The floor's `eta` (`floor_eta` in `InterconnectConfig`, default 0.15).
        r: The participant count the receipt was measured at.
        printed_floor: The floor value a receipt printed alongside its own `R`, if the
            caller has one to check; `None` skips that half of the check.

    Raises:
        GateFailure: `mean_a` is below `eta / r`, or `printed_floor` disagrees with
            `eta / r`. Effect (applied by the caller): the region is named in
            `collapsed_in_phase_A` and excluded from B's targets; the store is reverted in
            E2; the receipt is refused.
    """
    floor = eta / r
    if mean_a < floor:
        raise GateFailure(
            f"G29: {region!r} mean attention mass {mean_a:.4f} is below the floor "
            f"eta/R = {eta}/{r} = {floor:.4f}"
        )
    if printed_floor is not None and not math.isclose(printed_floor, floor, rel_tol=1e-9):
        raise GateFailure(
            f"G29: receipt printed floor {printed_floor:.4f} != eta/R = {eta}/{r} = "
            f"{floor:.4f} at its own R"
        )


def check_frozen_set_identity(participants: Sequence[Mapping[str, Any]]) -> None:
    """G33 -- frozen-set identity (spec section 4 Table 8; TAX:2598-2626).

    Fires when a participant's checkpoint sha256 does not match the one the receipt it
    cites recorded, or when a participant declares `kind: "nonparametric_store"` while
    also being parametric (a region trying to claim the store's exemption from the usual
    checkpoint-identity check).

    Args:
        participants: One mapping per participant, each with `name`, `checkpoint_sha256`
            (measured from the checkpoint file on disk), `receipt_checkpoint_sha256` (the
            sha256 the cited region receipt recorded), `kind`, and `parametric` (bool).
            `kind` and `parametric` are both required -- a row that omits either is
            refused rather than silently read as "not claiming the exemption" (a row
            missing `parametric` while declaring `kind: "nonparametric_store"` must not
            get the store's identity exemption for free).

    Raises:
        GateFailure: Naming the first offending participant and which half of the check
            failed -- a sha mismatch, a missing `kind`/`parametric` field, or an honest
            `parametric: True` participant claiming `nonparametric_store`. Effect
            (applied by the caller): phase A and E2 refuse to start.
    """
    for p in participants:
        name = p["name"]
        measured = p["checkpoint_sha256"]
        cited = p["receipt_checkpoint_sha256"]
        if measured != cited:
            raise GateFailure(
                f"G33: participant {name!r} checkpoint sha256 {measured} does not match "
                f"the sha256 {cited} its receipt cites"
            )
        if "kind" not in p or "parametric" not in p:
            raise GateFailure(
                f"G33: participant {name!r} omits 'kind' or 'parametric'; both are "
                "required to check the nonparametric_store exemption, and a row that "
                "omits one does not get the exemption by default"
            )
        if p["kind"] == "nonparametric_store" and p["parametric"]:
            raise GateFailure(
                f"G33: participant {name!r} declares kind 'nonparametric_store' but is "
                "a parametric region"
            )


def check_phase_d_revert(
    d_metric: float,
    c_metric: float,
    d_flops: float,
    c_flops: float,
) -> None:
    """G34 -- phase-D revert (spec section 4 Table 8; TAX:1753-1755).

    Fires when D does not beat C on the composed metric at equal or lower FLOPs. "Beats"
    is strict on the metric (a tie is not a win) and non-strict on FLOPs (equal FLOPs at a
    strictly better metric is a legitimate win).

    Args:
        d_metric: Phase D's composed metric.
        c_metric: Phase C's composed metric.
        d_flops: Phase D's measured FLOPs.
        c_flops: Phase C's measured FLOPs.

    Raises:
        GateFailure: D does not strictly beat C's metric, or D's FLOPs exceed C's. Effect
            (applied by the caller): C ships with `scheduler: "imitative"`.
    """
    if not (d_metric > c_metric and d_flops <= c_flops):
        raise GateFailure(
            f"G34: phase D (metric={d_metric:.4f}, flops={d_flops:.0f}) does not beat "
            f"phase C (metric={c_metric:.4f}, flops={c_flops:.0f}) at equal or lower "
            "FLOPs; reverting to scheduler: imitative"
        )


def check_overfit_gate(train_metric: float, held_out_metric: float) -> None:
    """G35 -- overfit gate (spec section 4 Table 6, Table 8; TAX:1757-1761).

    Fires when the train/held-out gap on the composed metric is five points or more.

    Args:
        train_metric: The composed metric measured on the training split.
        held_out_metric: The composed metric measured on the held-out split.

    Raises:
        GateFailure: `train_metric - held_out_metric >= 5 * POINT`. Effect (applied by
            the caller): the phase-A receipt is marked FAIL.
    """
    gap = train_metric - held_out_metric
    threshold = 5 * POINT
    # `>=`, not `>` -- exactly five points must fire (spec: "gap >= 5 points"). The
    # `isclose` half catches the boundary from the OTHER side G27's tolerance guards:
    # a gap that is mathematically exactly the threshold can land a few ulps UNDER it
    # after float subtraction (e.g. 0.35 - 0.30 == 0.049999999999999996, just below
    # 0.05) and must still fire rather than silently pass on subtraction noise.
    if gap >= threshold or math.isclose(gap, threshold, rel_tol=1e-9, abs_tol=1e-12):
        raise GateFailure(
            f"G35: train/held-out gap {gap / POINT:.2f} points >= the 5.00 point ceiling "
            f"(train={train_metric:.4f}, held_out={held_out_metric:.4f})"
        )


def check_null_gate(
    general_null_recall: float,
    per_bin_null_fpr: Mapping[str, float],
) -> None:
    """G36 -- `NULL` gate (spec section 4 Table 8; TAX:1701-1702).

    Fires when `NULL` recall on the general bin is at or below 0.50, or `NULL`'s
    false-positive rate is at or above 0.05 on any other bin. Both conditions are checked;
    the first one found raises, so a caller cannot pass a general-bin failure and have a
    bin-level failure silently shadow it or vice versa -- each is independently sufficient
    to fire this guard.

    Args:
        general_null_recall: `NULL` recall on the general bin, `[0, 1]`.
        per_bin_null_fpr: `NULL` false-positive rate per non-general bin, `[0, 1]` each.

    Raises:
        GateFailure: Naming which bin and which threshold failed. Effect (applied by the
            caller): the phase-A receipt is marked FAIL per bin.
    """
    if general_null_recall <= 0.50:
        raise GateFailure(f"G36: NULL recall on the general bin {general_null_recall:.4f} <= 0.50")
    for bin_name, fpr in per_bin_null_fpr.items():
        if fpr >= 0.05:
            raise GateFailure(
                f"G36: NULL false-positive rate {fpr:.4f} on bin {bin_name!r} >= 0.05"
            )
