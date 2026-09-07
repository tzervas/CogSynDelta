"""Grade a pre-registered contrast, and refuse a receipt that is not the arm it claims.

WHY A GRADING-TIME CHECK AND NOT A LAUNCH-TIME ONE
PREREG-RETRIEVAL-NEGATIVES-2026-09-06 (rev 3) section 2.1 requires **both auxiliary
weights at 0.0 in every arm**, a deliberate divergence from the production configuration.
That requirement can be violated silently: `scripts/csd-train-all.py` hard-coded
`memory`'s 0.1/0.1 with no override, so an arm launched through it would have trained the
production objective, finished normally, written a plausible receipt, and produced a
number that looks like a result. Nothing would have failed.

A check at the launcher only protects the launcher somebody remembered to fix. This one
sits at the point the number is READ, so it also covers a run started by hand from a
Python prompt six weeks from now, a resumed checkpoint, or an entry point that does not
exist yet. `pretrain_region` records the weights as the loss actually used them
(`receipt["objective_weights"]["measured"]`, collected at the multiplication site rather
than from the parsed config), and this module refuses to grade a receipt whose measured
weights are not the declared ones.

GUARD NUMBER
G40. Free at `12e2d1f`: G26 is the text split guard (`cogsyndelta.splits`), G27-G36 are
INTERCONNECT-MODULE-SPEC.md Table 8's, G37 is `eval/geometry.py`'s geometry reference,
G38/G39 are `regions/_mining.py`'s mining guards, and G41 is
`cogsyndelta.splits`'s reserved-holdout guard. The next new guard is G42.

FAIL CLOSED
A receipt with no measured block, a `None` measured block (no step ran, or the weights
changed mid-run), or a measured block that disagrees with the arm is REFUSED. "The field
is missing so there is nothing to check" is the shape that lets an unmeasured run pass as
a measured one, and it is the shape `tests/test_guards_can_fail.py` exists to catch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cogsyndelta.eval.paired import RESAMPLES, align_paired_arms, paired_bootstrap

PASS_A_BAR = 0.02
"""Section 5.2's bar: the one-sided 95% paired-bootstrap lower bound on ΔSuccess@10 must
exceed this, on both seeds. Pre-registered; moving it after seeing a result is choosing
the bar that gives the answer."""

PRIMARY_METRIC = "recall@10"
"""The receipt key Success@10 is stored under. Section 5.1 is explicit that this is
Success@10 ("is the query's best relevant passage in the top 10"), NOT BEIR Recall@10,
and that the key collision is a known one -- so a grader must name which it means."""


class PreregGuardError(Exception):
    """G40 fail-closed: a receipt is not the pre-registered arm it is being graded as."""


@dataclass(frozen=True)
class PreregArm:
    """One arm as the pre-registration declares it, in the fields a receipt can be
    checked against.

    Attributes:
        name: The arm's name in the pre-registration (`C`, `T1`, `T2`).
        token_loss_weight: `ζ` the arm must have trained with.
        decorr_weight: `γ` the arm must have trained with.
        negative_set: `in_batch`, `bank` or `mined` -- the ONE variable the round
            changes, so a receipt whose negative set is not this arm's is not this arm.
            None skips the check, for an arm the pre-registration does not pin.
    """

    name: str
    token_loss_weight: float
    decorr_weight: float
    negative_set: str | None = None


E_N_ARMS: dict[str, PreregArm] = {
    "C": PreregArm(name="C", token_loss_weight=0.0, decorr_weight=0.0, negative_set="in_batch"),
    "T1": PreregArm(name="T1", token_loss_weight=0.0, decorr_weight=0.0, negative_set="bank"),
    "T2": PreregArm(name="T2", token_loss_weight=0.0, decorr_weight=0.0, negative_set="mined"),
}
"""Round E-N's three arms (Table 2 and section 2.1). Both weights 0.0 in every one."""

E_N_CONFIRMATION = PreregArm(
    name="confirmation", token_loss_weight=0.1, decorr_weight=0.1, negative_set=None
)
"""Section 2.1's conditional confirmation run: the winning arm at the PRODUCTION weights,
seed 0. It is a different declared configuration, not an exception to the check -- which
is why the arms are data rather than a hard-coded pair of zeros."""


def _measured_weights(receipt: Mapping[str, Any]) -> tuple[float, float]:
    """The weights the loss actually applied, or a refusal.

    Args:
        receipt: A `csd-pretrain-receipt/v1` receipt.

    Returns:
        `(token_loss_weight, decorr_weight)` as measured at the loss site.

    Raises:
        PreregGuardError: If the receipt carries no measured weights. A receipt written
            before this field existed cannot be graded against a pre-registration that
            requires a particular objective -- it does not record what its objective was.
    """
    block = receipt.get("objective_weights")
    if not isinstance(block, Mapping):
        raise PreregGuardError(
            "G40: receipt has no `objective_weights` block, so the weights its loss "
            "actually used are unknown; it cannot be graded against a pre-registration "
            "that pins them"
        )
    measured = block.get("measured")
    if not isinstance(measured, Mapping):
        seen = block.get("values_seen")
        raise PreregGuardError(
            f"G40: receipt records no single measured weight pair (values_seen={seen}, "
            f"steps_measured={block.get('steps_measured')}); either no step ran or the "
            f"weights changed mid-run, and neither is a gradeable arm"
        )
    try:
        return float(measured["token_loss_weight"]), float(measured["decorr_weight"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreregGuardError(f"G40: measured weights are unreadable: {measured!r}") from exc


def assert_receipt_matches_arm(receipt: Mapping[str, Any], arm: PreregArm) -> None:
    """G40: refuse a receipt whose run is not the arm it is being graded as.

    Checks the weights the loss MEASURED, never the ones the config declared: the two
    disagreeing is precisely the failure this guards, and a check that reads the declared
    pair would agree with itself while the run did something else.

    Args:
        receipt: A `csd-pretrain-receipt/v1` receipt.
        arm: The pre-registered arm it is claimed to be.

    Raises:
        PreregGuardError: On a missing/ambiguous measured block, on a weight that differs
            from the arm's declared value, or on a negative set that is not the arm's.
    """
    token_weight, decorr_weight = _measured_weights(receipt)
    if (token_weight, decorr_weight) != (arm.token_loss_weight, arm.decorr_weight):
        raise PreregGuardError(
            f"G40: arm {arm.name} is pre-registered at token_loss_weight="
            f"{arm.token_loss_weight}, decorr_weight={arm.decorr_weight}, but this run "
            f"MEASURED {token_weight}/{decorr_weight} at the loss site. That is a "
            f"different objective, so its numbers do not answer the pre-registered "
            f"question -- refusing to grade it rather than reporting a result for the "
            f"wrong experiment"
        )
    if arm.negative_set is not None:
        negatives = receipt.get("negatives")
        recorded = negatives.get("set") if isinstance(negatives, Mapping) else None
        if recorded != arm.negative_set:
            raise PreregGuardError(
                f"G40: arm {arm.name} is the {arm.negative_set!r} negative set, but this "
                f"receipt records {recorded!r}"
            )


def _per_query(receipt: Mapping[str, Any], metric: str) -> tuple[list[str], list[float]]:
    """The trained arm's per-query vector for `metric`, with its query ids.

    Args:
        receipt: A memory-region receipt carrying the `retrieval` block.
        metric: Receipt key, e.g. `recall@10`.

    Returns:
        `(query_ids, values)`.

    Raises:
        ValueError: If the receipt has no per-query emission for that metric. This is a
            shape error, not a guard: the run is not disqualified, it was evaluated by a
            path that did not emit the terms a paired test needs.
    """
    full_pool = ((receipt.get("retrieval") or {}).get("full_pool")) or {}
    ids = full_pool.get("query_ids")
    values = ((full_pool.get("per_query") or {}).get("trained") or {}).get(metric)
    if not ids or not values:
        raise ValueError(
            f"receipt carries no per-query {metric!r} for the trained arm; the paired "
            f"decision rule needs `retrieval.full_pool.per_query.trained[{metric!r}]` "
            f"and `retrieval.full_pool.query_ids`"
        )
    return list(ids), list(values)


def grade_contrast(
    *,
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    treatment_arm: PreregArm,
    control_arm: PreregArm | None = None,
    seed: int,
    metric: str = PRIMARY_METRIC,
    resamples: int = RESAMPLES,
    bar: float = PASS_A_BAR,
) -> dict[str, Any]:
    """Grade one treatment-minus-control contrast under the pre-registered rule.

    The G40 check runs on BOTH receipts before anything is computed, so an arm that did
    not run the pre-registered objective produces a refusal rather than a number.

    Args:
        control: The control arm's receipt.
        treatment: The treatment arm's receipt.
        treatment_arm: Which pre-registered arm `treatment` is.
        control_arm: Which arm `control` is; defaults to E-N's `C`.
        seed: Bootstrap seed, recorded in the returned block.
        metric: Receipt key to grade on. Defaults to Success@10.
        resamples: Bootstrap replicates; the pre-registered 10,000.
        bar: The lower bound must EXCEED this. Defaults to section 5.2's 0.02.

    Returns:
        The bootstrap receipt, the bar, and `passed` -- one seed's verdict. Section 5.2's
        rule needs both seeds to clear it, which is the caller's conjunction to make, not
        this function's.

    Raises:
        PreregGuardError: G40, if either receipt is not the arm it is graded as.
        ValueError: If a receipt carries no per-query emission for `metric`.
    """
    control_arm = control_arm or E_N_ARMS["C"]
    assert_receipt_matches_arm(control, control_arm)
    assert_receipt_matches_arm(treatment, treatment_arm)

    control_ids, control_values = _per_query(control, metric)
    treatment_ids, treatment_values = _per_query(treatment, metric)
    aligned_control, aligned_treatment = align_paired_arms(
        control_ids, control_values, treatment_ids, treatment_values
    )
    # `.tolist()` because `paired_bootstrap` takes sequences: the arrays here are float64
    # already (`align_paired_arms` says why), so this converts container, not precision.
    result = paired_bootstrap(
        aligned_control.tolist(), aligned_treatment.tolist(), seed=seed, resamples=resamples
    )
    return {
        "metric": metric,
        "control_arm": control_arm.name,
        "treatment_arm": treatment_arm.name,
        "bar": bar,
        "passed": result.lower_bound > bar,
        "bootstrap": result.as_receipt(),
    }
