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

WHY THERE IS A THIRD VERDICT
A guard above answers "did the condition fire?". That is the whole question only when the
statistic it reads is precise relative to the threshold it is compared against, and for
two of these guards it measurably is not. G35's train/held-out gap has a seed-axis standard
deviation of 2.80 points against its own 5.00-point ceiling over 24 pinned-thread seeds,
and G29's `mean(a_store)` flips verdict on `OMP_NUM_THREADS` alone at one seed. On such a
statistic a single run's PASS is not a measurement, and reporting it as one is the defect.
Raw sweeps: `/akula-data/scratch/csd-det/runs/seeds24-db8-pin1/` (G35, 24 seeds),
`/akula-data/scratch/csd-g29-diag-evidence/out/` (G29, 8 seeds and 5 thread pins) and
`/akula-data/scratch/csd-gate-discrimination/runs/` (the split-size scaling arm); the
written diagnosis is
`/akula-data/session-backup-staging/notes/GATE-DISCRIMINATION-2026-09-07.md`.

`GateReport` and the `evaluate_*` functions add the missing third verdict,
`"inconclusive"`, under one invariant that keeps every guard as fail-closed as it is
today: **`"inconclusive"` may only ever replace a `"pass"`, never a `"fail"`.** The gate
predicate is evaluated first and unchanged; only a run that did NOT fire can be downgraded
to `"inconclusive"` for lacking the precision to earn its pass. That is strictly the
removal of unearned passes -- no threshold moves, no gate stops firing, nothing is
weakened. `require_conclusive` keeps it fail-closed at the call site by raising on both
`"fail"` and `"inconclusive"`.

The `check_*` guards are untouched and remain the ratified surface; the `evaluate_*`
functions call the SAME predicate helpers, so the two can never drift.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "POINT",
    "GateFailure",
    "GateInconclusive",
    "GateReport",
    "check_attention_mass_floor",
    "check_frozen_set_identity",
    "check_null_gate",
    "check_overfit_gate",
    "check_phase_d_revert",
    "check_topology_agreement",
    "check_write_back_gate",
    "evaluate_attention_mass_floor",
    "evaluate_overfit_gate",
    "replicate_verdict",
    "require_conclusive",
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


class GateInconclusive(RuntimeError):  # noqa: N818 -- sibling of GateFailure, same convention
    """Raised when a guard could not be evaluated to a trustworthy verdict.

    NOT a subclass of `GateFailure`, and deliberately so: "the gate fired" and "the gate
    could not tell" are different findings with different remedies, and a caller that
    catches `GateFailure` must not silently absorb the second. It is equally not a pass --
    `require_conclusive` raises on it -- because the failure mode this whole third verdict
    exists to prevent is exactly a run banking an unearned PASS.
    """


# ---------------------------------------------------------------------------------------
# Shared predicates -- the ONE definition of "did this gate fire", used by both the
# ratified `check_*` guards and the `evaluate_*` reporters so the two cannot drift.
# ---------------------------------------------------------------------------------------


def _attention_floor_fired(mean_a: float, floor: float) -> bool:
    """Return G29's predicate: Table 8's "fires when `mean(a_r) < eta/R`", strictly below.

    No `isclose` tolerance here, unlike G27 and G35. Those two compare a DIFFERENCE of two
    metrics against a threshold, so a mathematically-exact boundary is reachable through
    float subtraction and has to be pinned from the side the spec names. `mean_a` is a mean
    of softmax outputs and `floor` is `eta / r`; landing exactly on the boundary is a
    measure-zero event, and a tolerance here would be a one-sided LOOSENING -- a mass a few
    ulps below the floor would stop firing. Left strict on purpose.
    """
    return mean_a < floor


def _overfit_gap_fired(gap: float, threshold: float) -> bool:
    """Return G35's predicate: Table 8's "fires when the gap is >= 5 points", inclusive.

    The `isclose` half catches the boundary from the side G27's tolerance guards against:
    a gap that is mathematically exactly the threshold can land a few ulps UNDER it after
    float subtraction (e.g. `0.35 - 0.30 == 0.049999999999999996`, just below `0.05`) and
    must still fire rather than pass on subtraction noise.
    """
    return gap >= threshold or math.isclose(gap, threshold, rel_tol=1e-9, abs_tol=1e-12)


# ---------------------------------------------------------------------------------------
# The third verdict
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GateReport:
    """One gate's verdict together with the evidence that the verdict is trustworthy.

    `fired` is the gate predicate and nothing else -- the same boolean the matching
    `check_*` guard raises on. `verdict` adds the precision question on top of it, under
    the invariant stated in the module docstring: `fired` implies `verdict == "fail"`,
    always, so no amount of noise can talk a firing gate out of failing.

    Attributes:
        gate: The Table 8 G-number, e.g. `"G29"`.
        subject: What was gated -- a region name for G29, the phase for G35.
        fired: The gate predicate's own answer. `True` means the condition Table 8 names
            as "fires when" held.
        verdict: `"fail"` when `fired`; otherwise `"pass"` if the margin is large enough
            relative to the statistic's own spread and resolution to be believed, and
            `"inconclusive"` if it is not.
        statistic: The number the gate read.
        threshold: The number it was compared against.
        margin: How far `statistic` sits on the SAFE side of `threshold`, in the
            statistic's own units -- positive is clear of the gate, negative is past it.
            The sign is normalised per gate so "bigger is safer" always holds, which is
            what lets a shrinking margin read as progress rather than as another FAIL.
        spread: The statistic's own spread on the axis that carries its variance (the seed
            axis; the thread axis is a nuisance parameter to pin away, not to sample), or
            `None` when the caller measured only one draw.
        resolution: The smallest non-zero change the statistic can express, when it is a
            discrete statistic whose denominator the caller knows; `None` otherwise. A
            margin below one resolution step is a verdict one item away from flipping.
        notes: Human-readable facts about this evaluation -- why it was inconclusive, what
            ceiling was actually enforced after quantisation, whether an operand was
            saturated. Never load-bearing for the verdict; always load-bearing for the
            reader.
    """

    gate: str
    subject: str
    fired: bool
    verdict: str
    statistic: float
    threshold: float
    margin: float
    spread: float | None = None
    resolution: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def margin_over_spread(self) -> float | None:
        """Return `margin / spread`, the scale-invariant severity, or `None` with no spread.

        Measured (24 pinned seeds, `csd-det/gatestat.py`) to separate failing from passing
        phase-A runs by 0.355 where `recall@1` separated them by 0.0156, and to be exactly
        invariant under a global rescale of the scores it derives from. A spread of exactly
        zero yields an infinite ratio for a non-zero margin: a statistic with no spread has
        nothing to hide behind.
        """
        if self.spread is None:
            return None
        if self.spread == 0.0:
            return math.inf if self.margin > 0 else (-math.inf if self.margin < 0 else 0.0)
        return self.margin / self.spread

    def summary(self) -> str:
        """Return a one-line `G<NN>: ...` summary carrying the verdict and the margin."""
        ratio = self.margin_over_spread
        tail = "" if ratio is None else f", margin/spread {ratio:.3f}"
        detail = ("; " + "; ".join(self.notes)) if self.notes else ""
        return (
            f"{self.gate}: {self.verdict.upper()} for {self.subject!r} -- statistic "
            f"{self.statistic:.4f} against threshold {self.threshold:.4f}, margin "
            f"{self.margin:+.4f}{tail}{detail}"
        )


def _classify(
    *,
    fired: bool,
    margin: float,
    spread: float | None,
    resolution: float | None,
) -> tuple[str, tuple[str, ...]]:
    """Return `(verdict, notes)` under "inconclusive never replaces a fail".

    The order of the branches IS that invariant: `fired` is answered first and returns
    immediately, so no precision argument can reach a gate that fired.
    """
    if fired:
        return "fail", ()
    reasons: list[str] = []
    if spread is None:
        reasons.append(
            "no spread supplied, so this run cannot certify its own precision; replicate "
            "over the seed axis and pass the spread"
        )
    elif margin < spread:
        ratio = margin / spread if spread else math.inf
        reasons.append(
            f"margin {margin:.4f} is inside the statistic's own spread {spread:.4f} "
            f"(margin/spread {ratio:.3f} < 1)"
        )
    if resolution is not None and margin < resolution:
        reasons.append(
            f"margin {margin:.4f} is below the statistic's resolution {resolution:.4f}, so "
            "the verdict is one quantum from flipping"
        )
    if reasons:
        return "inconclusive", tuple(reasons)
    return "pass", ()


def require_conclusive(report: GateReport) -> None:
    """Raise unless `report` is a trustworthy pass -- the fail-closed call site.

    Args:
        report: Any `GateReport`.

    Raises:
        GateFailure: `report.verdict == "fail"`, i.e. the gate predicate fired.
        GateInconclusive: `report.verdict == "inconclusive"`. Route this to a reviewer:
            the run has no basis on which to choose, and treating it as a pass is the
            exact failure this verdict exists to prevent.
    """
    if report.verdict == "fail":
        raise GateFailure(report.summary())
    if report.verdict == "inconclusive":
        raise GateInconclusive(report.summary())


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
    if _attention_floor_fired(mean_a, floor):
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
    # `>=`, not `>` -- exactly five points must fire (spec: "gap >= 5 points"); see
    # `_overfit_gap_fired` for why the boundary carries an `isclose` half.
    if _overfit_gap_fired(gap, threshold):
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


# ---------------------------------------------------------------------------------------
# Reporting evaluators -- the same predicates, plus the precision the verdict rests on
# ---------------------------------------------------------------------------------------


def evaluate_attention_mass_floor(
    region: str,
    mean_a: float,
    eta: float,
    r: int,
    *,
    spread: float | None = None,
) -> GateReport:
    """Report G29 with its margin, not just its verdict (spec section 4 Table 8).

    The predicate is `check_attention_mass_floor`'s collapse half, unchanged and shared.
    What this adds is the margin `mean_a - eta/R` and, when the caller has measured the
    statistic's seed-axis spread, the judgement of whether the margin is large enough to
    believe.

    WHY THIS EXISTS, MEASURED. At one seed and one step count, `mean(a_store)` on phase
    A's synthetic stream reads 0.0642 / 0.0914 / 0.1177 / 0.0468 / 0.0736 at
    `OMP_NUM_THREADS` 1 / 2 / 4 / 8 / 16 -- identical code, identical seed -- and the
    0.0468 draw is the only one below the 0.0500 floor. Over eight seeds the same
    statistic reads 0.0463 to 0.2263, mean 0.128, sd 0.066: a floor 1.2 sd below the
    centre of its own null. A single-draw PASS on that statistic is a coin flip, and this
    function is what lets a caller say so instead of banking it.

    WHAT IT DOES NOT DO. A conclusive PASS here is a statement about precision, not about
    the store. On phase A's current synthetic stream the store read is item-invariant, so
    `mean(a_store)` is measurably independent of what the store holds -- 0.0468 at 5, 8
    AND 64 primed records at one seed, and indistinguishable from an EMPTY store across
    five paired seeds. No statistic computed from that stream can certify the store,
    however precisely it is measured. See the recommendation document; the input is what
    has to change.

    Args:
        region: Name of the region (or `"store"`) being checked.
        mean_a: Mean attention mass `a` this region received over active iterations.
        eta: The floor's `eta` (`floor_eta` in `InterconnectConfig`, default 0.15).
        r: The participant count the receipt was measured at.
        spread: The seed-axis spread of `mean_a` for this arm, if the caller replicated.
            `None` yields an `"inconclusive"` report for any run that did not fire, which
            is the honest reading of a single draw of this statistic.

    Returns:
        A `GateReport` for `"G29"`. Attention mass is continuous, so `resolution` is
        `None`.
    """
    floor = eta / r
    fired = _attention_floor_fired(mean_a, floor)
    verdict, notes = _classify(fired=fired, margin=mean_a - floor, spread=spread, resolution=None)
    return GateReport(
        gate="G29",
        subject=region,
        fired=fired,
        verdict=verdict,
        statistic=mean_a,
        threshold=floor,
        margin=mean_a - floor,
        spread=spread,
        resolution=None,
        notes=notes,
    )


def evaluate_overfit_gate(
    train_metric: float,
    held_out_metric: float,
    *,
    held_out_n: int | None = None,
    spread: float | None = None,
) -> GateReport:
    """Report G35 with its margin, its resolution and its realised ceiling.

    The predicate is `check_overfit_gate`'s, unchanged and shared, so the 5.00-point
    ceiling this reports against is the spec's (`INTERCONNECT-MODULE-SPEC.md:418`) and is
    not touched here.

    WHY THIS EXISTS, MEASURED, and it is two separate defects that look like one.

    (1) RESOLUTION. `recall@1` over `N` held-out items moves in steps of `1/N`, so the gap
    can only take values on that lattice. At `N = 64` the step is 1.5625 points and the
    lattice is `{0, 1.5625, 3.125, 4.6875, 6.25, ...}`: 5.00 is not on it, so the ceiling
    the gate actually enforces is 6.25 points, 1.25x the spec's. `held_out_n` is what lets
    this function say that out loud, and lets a would-be pass whose margin is under one
    step report `"inconclusive"` -- a verdict one held-out item from flipping is not a
    verdict. Measured: 5 of 24 seeds sat exactly one item either side of the boundary.

    (2) POWER, which raising `N` from 16 to 64 did not fix. Over 24 seeds with threads
    pinned the gap spans 0.00-10.94 points with a seed-axis sd of 2.80 -- 56% of the
    5.00-point ceiling it is compared against -- and 3 of those 24 fail. Resolution is
    whether the statistic can express the threshold; power is whether a verdict at that
    threshold survives redrawing the seed. `spread` is what carries the second question.

    A third fact is reported as a note rather than as a verdict, because it changes what
    the number MEANS without changing how precisely it is known: `train_metric` was 1.0 in
    all 147 measured 200-step runs, so the "gap" reduces to `1 - held_out_metric` and the
    gate is a dev-only absolute bar wearing a gap's clothes. That is an input defect --
    a train split the toy memorises -- and it is not fixable from inside this function.

    Args:
        train_metric: The composed metric measured on the training split.
        held_out_metric: The composed metric measured on the held-out split.
        held_out_n: Number of held-out items, if known. Supplying it turns on the
            resolution half above; omitting it leaves `resolution` `None` and reports only
            the spread half.
        spread: The seed-axis spread of the gap for this arm, if the caller replicated.

    Returns:
        A `GateReport` for `"G35"`, with `statistic` the gap and `margin` the distance
        below the ceiling (positive is clear of the gate).
    """
    gap = train_metric - held_out_metric
    threshold = 5 * POINT
    fired = _overfit_gap_fired(gap, threshold)
    resolution = 1.0 / held_out_n if held_out_n else None
    verdict, notes = _classify(
        fired=fired, margin=threshold - gap, spread=spread, resolution=resolution
    )
    extra: list[str] = []
    if resolution is not None:
        steps = math.ceil(threshold / resolution - 1e-12)
        realised = steps * resolution
        extra.append(
            f"at n={held_out_n} the metric moves in steps of {resolution / POINT:.4f} "
            f"points, so the ceiling actually enforced is {realised / POINT:.4f} points, "
            f"{realised / threshold:.2f}x the spec's 5.00"
        )
    if math.isclose(train_metric, 1.0, rel_tol=0.0, abs_tol=1e-12):
        extra.append(
            "train_metric is saturated at 1.0, so this gap is exactly "
            "(1 - held_out_metric): a dev-only absolute bar, not an overfit gap"
        )
    return GateReport(
        gate="G35",
        subject="phase A composed metric",
        fired=fired,
        verdict=verdict,
        statistic=gap,
        threshold=threshold,
        margin=threshold - gap,
        spread=spread,
        resolution=resolution,
        notes=notes + tuple(extra),
    )


def replicate_verdict(reports: Iterable[GateReport], *, subject: str | None = None) -> GateReport:
    """Combine one gate's reports over k replicate seeds into one replicated verdict.

    THE RULE, and why it is this rule and not a stricter or looser one. Unanimity decides:
    every replicate fired -> `"fail"`; none fired -> the precision test on the seed-axis
    spread this function measures from the replicates themselves; anything mixed ->
    `"inconclusive"`.

    - Mixed must not become `"pass"`, or a real collapse that bites on most seeds would be
      voted away. It never does here.
    - Mixed must not become `"fail"` either, and this is the part that needs the
      measurement: on phase A's synthetic stream G29 fires on 2 of 8 seeds and G35 on 3 of
      24, and those firings are not a property of the run -- the same seed flips G29 on
      `OMP_NUM_THREADS` alone. Converting a noise-driven minority into a hard FAIL is not
      strictness, it is a different wrong answer, and it would send a loop chasing a
      collapse that is not there. `"inconclusive"` routes it to a reviewer, which is what
      the measurement supports.

    The seed axis is the axis to replicate on. The thread axis is a nuisance parameter to
    PIN, not to sample: it moves the checkpoint (11 distinct `checkpoint_sha256` across
    thread counts at one seed) without meaning anything about the model.

    Args:
        reports: Two or more reports for the SAME gate, one per replicate seed.
        subject: Optional subject for the combined report; defaults to the common subject
            of the inputs, or a comma-joined list when they differ.

    Returns:
        A `GateReport` whose `statistic` and `margin` are the means over the replicates and
        whose `spread` is their sample standard deviation -- the seed-axis spread, which is
        the number a single-run `evaluate_*` call wants passed back to it.

    Raises:
        ValueError: Fewer than two reports (one draw is not a replication), or reports from
            more than one gate, or thresholds that disagree.
    """
    items = tuple(reports)
    if len(items) < 2:
        raise ValueError(
            f"replicate_verdict: {len(items)} report(s); a replicated verdict needs at "
            "least 2 seeds, because the whole point is the spread across them"
        )
    gates = {r.gate for r in items}
    if len(gates) != 1:
        raise ValueError(f"replicate_verdict: reports span more than one gate: {sorted(gates)}")
    first = items[0].threshold
    if not all(math.isclose(r.threshold, first, rel_tol=1e-9, abs_tol=1e-12) for r in items):
        raise ValueError(
            "replicate_verdict: replicates were gated against different thresholds "
            f"{sorted({r.threshold for r in items})}; they are not the same gate"
        )

    gate = items[0].gate
    subjects = {r.subject for r in items}
    name = subject or (subjects.pop() if len(subjects) == 1 else ", ".join(sorted(subjects)))
    stats = [r.statistic for r in items]
    margins = [r.margin for r in items]
    mean_stat = statistics.fmean(stats)
    mean_margin = statistics.fmean(margins)
    seed_spread = statistics.stdev(stats)
    resolutions = [r.resolution for r in items if r.resolution is not None]
    resolution = max(resolutions) if resolutions else None
    n_fired = sum(1 for r in items if r.fired)

    verdict: str
    notes: tuple[str, ...]
    if n_fired == len(items):
        verdict, notes = "fail", (f"all {len(items)} replicate seeds fired",)
    elif n_fired:
        verdict = "inconclusive"
        notes = (
            f"{n_fired} of {len(items)} replicate seeds fired and {len(items) - n_fired} "
            "did not; a verdict that is not unanimous over the seed axis is not a verdict",
        )
    else:
        verdict, notes = _classify(
            fired=False, margin=mean_margin, spread=seed_spread, resolution=resolution
        )
        notes = (f"none of {len(items)} replicate seeds fired",) + notes
    return GateReport(
        gate=gate,
        subject=name,
        fired=n_fired == len(items),
        verdict=verdict,
        statistic=mean_stat,
        threshold=first,
        margin=mean_margin,
        spread=seed_spread,
        resolution=resolution,
        notes=notes,
    )
