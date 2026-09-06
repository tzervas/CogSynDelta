"""`cogsyndelta.eval.paired`: the paired bootstrap that decides a two-arm contrast.

These test the three things a decision rule can be wrong about without looking wrong:
whether it reproduces (a bound nobody can replay is not evidence), whether the
convention it claims is the convention it applies (a "one-sided 95% lower bound" that is
secretly a two-sided one is a different, stricter gate under the same name), and whether
it will quietly compare arms measured over different queries -- which turns "the
treatment is better" into "the treatment is better on the queries it managed".

The two analytic cases -- a constant improvement and a null contrast -- are here because
they are the only ones where the right answer is known without appealing to the
implementation being tested: resampling a constant vector can only ever give that
constant back.
"""

from __future__ import annotations

import numpy as np
import pytest

from cogsyndelta.eval.paired import (
    CONFIDENCE,
    RESAMPLES,
    PairedBootstrapResult,
    align_paired_arms,
    paired_bootstrap,
)

pytestmark = pytest.mark.cpu


def _success_at_10_arm(rate: float, n: int, seed: int) -> list[float]:
    """A 0/1 Success@10 vector at roughly `rate`, shaped like the real primary metric."""
    rng = np.random.default_rng(seed)
    return [float(v) for v in (rng.random(n) < rate).astype(np.float64)]


def _reciprocal_rank_arm(n: int, seed: int) -> list[float]:
    """A continuous-valued per-query arm, shaped like the secondary metric MRR.

    Used wherever a test asserts that two bounds DIFFER. Success@10's per-query terms are
    0/1, so a resampled mean over n queries is a multiple of 1/n and two bootstrap
    distributions can land on the identical order statistic by coincidence -- which would
    make such a test flaky for a reason that has nothing to do with the code. MRR's 1/rank
    terms are the real metric that does not have that granularity.
    """
    rng = np.random.default_rng(seed)
    return [1.0 / float(rank) for rank in rng.integers(1, 5000, size=n)]


# ---------------------------------------------------------------------------------------
# Reproducibility. The bound goes into a receipt; a receipt records a number that can be
# re-derived, or it records nothing.
# ---------------------------------------------------------------------------------------


def test_the_same_seed_gives_the_identical_bound_twice() -> None:
    control = _success_at_10_arm(0.20, 500, seed=11)
    treatment = _success_at_10_arm(0.26, 500, seed=12)
    first = paired_bootstrap(control, treatment, seed=0)
    second = paired_bootstrap(control, treatment, seed=0)
    assert first == second
    assert first.lower_bound == second.lower_bound
    assert first.point_estimate == second.point_estimate


def test_a_different_seed_moves_the_bound_but_not_the_point_estimate() -> None:
    """The point estimate is a property of the data; only the bound is resampled.

    If a seed change moved the point estimate too, the "estimate" would be a bootstrap
    artefact rather than the observed mean difference -- and the gate would be reading a
    number that depends on the RNG.
    """
    control = _reciprocal_rank_arm(500, seed=11)
    treatment = _reciprocal_rank_arm(500, seed=12)
    first = paired_bootstrap(control, treatment, seed=0)
    second = paired_bootstrap(control, treatment, seed=1)
    assert first.point_estimate == second.point_estimate
    assert first.lower_bound != second.lower_bound
    assert abs(first.lower_bound - second.lower_bound) < 0.01  # same distribution, not noise


def test_the_bound_is_reproducible_in_a_fresh_process() -> None:
    """`default_rng(seed)` is stable across processes, and this is the assertion that
    keeps it that way if the RNG is ever swapped for a global `np.random` call."""
    import subprocess
    import sys

    program = (
        "from cogsyndelta.eval.paired import paired_bootstrap;"
        "print(repr(paired_bootstrap([0.0, 1.0, 0.0, 1.0, 1.0],"
        " [1.0, 1.0, 0.0, 1.0, 1.0], seed=7, resamples=1000).lower_bound))"
    )
    out = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    in_process = paired_bootstrap(
        [0.0, 1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0, 1.0], seed=7, resamples=1000
    ).lower_bound
    assert float(out.stdout.strip()) == in_process


# ---------------------------------------------------------------------------------------
# Cases with a known answer.
# ---------------------------------------------------------------------------------------


def test_a_constant_improvement_bounds_at_that_constant() -> None:
    """Every query improves by 0.25, so every resample has mean 0.25 and so does the 5th
    percentile. There is no query-level variance left for the bootstrap to find, which
    makes this the one case where the exact bound is knowable a priori."""
    control = [0.0, 0.25, 0.5, 0.75, 1.0, 0.4, 0.6, 0.8]
    treatment = [v + 0.25 for v in control]
    result = paired_bootstrap(control, treatment, seed=3)
    assert result.point_estimate == pytest.approx(0.25)
    assert result.lower_bound == pytest.approx(0.25)


def test_identical_arms_give_a_zero_point_estimate_and_a_zero_bound() -> None:
    """The null case. A treatment that changed nothing must not clear a bar above zero,
    and the degenerate difference vector makes that exact rather than probabilistic."""
    arm = _success_at_10_arm(0.20, 500, seed=5)
    result = paired_bootstrap(arm, list(arm), seed=0)
    assert result.point_estimate == 0.0
    assert result.lower_bound == 0.0


def test_a_hurt_arm_bounds_below_zero() -> None:
    """A treatment that loses on every query must produce a negative bound, not a small
    positive one -- the sign of the bound is the whole decision."""
    control = [1.0] * 20
    treatment = [0.9] * 20
    result = paired_bootstrap(control, treatment, seed=0)
    assert result.point_estimate == pytest.approx(-0.1)
    assert result.lower_bound < 0.0


def test_the_bound_is_the_fifth_percentile_of_the_replicate_means() -> None:
    """The declared convention, re-derived independently.

    The result claims a one-sided 95% percentile bootstrap; this replays the same RNG
    stream by hand and checks the returned bound is that distribution's 5th percentile --
    not the 2.5th a two-sided interval would report under the same confidence level.
    """
    control = _reciprocal_rank_arm(200, seed=21)
    treatment = _reciprocal_rank_arm(200, seed=22)
    result = paired_bootstrap(control, treatment, seed=9, resamples=2000)

    differences = np.asarray(treatment) - np.asarray(control)
    rng = np.random.default_rng(9)
    picks = rng.integers(0, differences.size, size=(2000, differences.size), dtype=np.int64)
    replicate_means = differences[picks].mean(axis=1)
    assert result.lower_bound == float(np.percentile(replicate_means, 5.0, method="linear"))
    assert result.lower_bound != float(np.percentile(replicate_means, 2.5, method="linear"))


# ---------------------------------------------------------------------------------------
# The declared provenance, and that the knobs are actually honoured.
# ---------------------------------------------------------------------------------------


def test_the_pre_registered_defaults_are_what_runs() -> None:
    """10,000 resamples at 95%, one-sided. Pre-registered values, not tuning knobs: a
    resample count chosen after seeing a result is a resample count chosen to pass."""
    assert RESAMPLES == 10_000
    assert CONFIDENCE == 0.95
    result = paired_bootstrap([0.0, 1.0, 1.0], [1.0, 1.0, 1.0], seed=0)
    assert result.resamples == 10_000
    assert result.n_queries == 3
    assert result.seed == 0
    assert result.confidence == 0.95
    assert result.interval == "one-sided-lower"
    assert result.method == "paired-percentile-bootstrap"
    assert result.percentile == 5.0
    assert result.interpolation == "linear"


def test_the_resample_count_and_seed_are_honoured_not_ignored() -> None:
    """A `resamples` argument that is recorded but not used would be the worst kind of
    provenance: a receipt that lies. Two different counts must give two different
    bootstrap distributions, hence two different bounds."""
    control = _reciprocal_rank_arm(300, seed=31)
    treatment = _reciprocal_rank_arm(300, seed=32)
    small = paired_bootstrap(control, treatment, seed=4, resamples=250)
    large = paired_bootstrap(control, treatment, seed=4, resamples=10_000)
    assert small.resamples == 250
    assert large.resamples == 10_000
    assert small.lower_bound != large.lower_bound
    assert small.point_estimate == large.point_estimate


def test_the_result_is_frozen_and_serialises_its_whole_convention() -> None:
    """A receipt field that can be mutated after the fact is not provenance."""
    result = paired_bootstrap([0.0, 1.0], [1.0, 1.0], seed=0, resamples=100)
    with pytest.raises((AttributeError, TypeError)):
        result.lower_bound = 0.9  # type: ignore[misc]
    receipt = result.as_receipt()
    assert set(receipt) == {f.name for f in PairedBootstrapResult.__dataclass_fields__.values()}
    assert receipt["method"] == "paired-percentile-bootstrap"
    assert receipt["resamples"] == 100


# ---------------------------------------------------------------------------------------
# Refusals. Every one of these would otherwise produce a plausible number.
# ---------------------------------------------------------------------------------------


def test_mismatched_arm_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap([0.0, 1.0, 1.0], [1.0, 1.0], seed=0)


def test_an_empty_contrast_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one query"):
        paired_bootstrap([], [], seed=0)


def test_a_non_positive_resample_count_is_refused() -> None:
    with pytest.raises(ValueError, match="resamples must be"):
        paired_bootstrap([0.0, 1.0], [1.0, 1.0], seed=0, resamples=0)


def test_alignment_reorders_the_treatment_arm_onto_the_control_order() -> None:
    """The arms may arrive in different query orders; only the ids say which value pairs
    with which. Zipping them positionally would pair the wrong queries and still return a
    number."""
    control, treatment = align_paired_arms(
        ["q1", "q2", "q3"], [0.0, 1.0, 0.0], ["q3", "q1", "q2"], [1.0, 1.0, 0.0]
    )
    assert control.tolist() == [0.0, 1.0, 0.0]
    assert treatment.tolist() == [1.0, 0.0, 1.0]


def test_alignment_refuses_a_length_mismatch_instead_of_truncating() -> None:
    with pytest.raises(ValueError, match="same queries"):
        align_paired_arms(["q1", "q2"], [0.0, 1.0], ["q1"], [1.0])


def test_alignment_refuses_a_different_query_set_of_the_same_size() -> None:
    """The dangerous case: equal lengths, so a length check alone waves it through."""
    with pytest.raises(ValueError, match="different queries"):
        align_paired_arms(["q1", "q2"], [0.0, 1.0], ["q1", "q9"], [1.0, 1.0])


def test_alignment_refuses_ids_and_values_of_different_lengths() -> None:
    with pytest.raises(ValueError, match="control arm has"):
        align_paired_arms(["q1", "q2"], [0.0], ["q1", "q2"], [1.0, 1.0])
    with pytest.raises(ValueError, match="treatment arm has"):
        align_paired_arms(["q1", "q2"], [0.0, 1.0], ["q1", "q2"], [1.0])


def test_alignment_refuses_a_repeated_query_id() -> None:
    """A duplicated id makes "the same query in both arms" ambiguous, and the dict-based
    lookup would silently keep only the last one."""
    with pytest.raises(ValueError, match="control arm repeats"):
        align_paired_arms(["q1", "q1"], [0.0, 1.0], ["q1", "q2"], [1.0, 1.0])
    with pytest.raises(ValueError, match="treatment arm repeats"):
        align_paired_arms(["q1", "q2"], [0.0, 1.0], ["q1", "q1"], [1.0, 1.0])
