"""The paired bootstrap: the decision rule for a two-arm retrieval contrast.

WHY THIS EXISTS
A retrieval arm is judged by a difference, and the difference between two arms measured
on the SAME queries is not the difference between two independent samples. At p = 0.2 and
n = 500, one arm's Success@10 has a standard error of 0.0179 and an unpaired difference
of two arms about 0.0253 -- so an unpaired 95% bar would sit near 0.050 and a real
improvement of 0.03 would be unreportable. Almost all of that variance is BETWEEN
QUERIES: some FiQA questions are answerable by any encoder and some by none, and both
arms see exactly the same ones. Pairing removes that shared term, which is what makes a
0.02 bar falsifiable rather than optimistic.

WHY A BOOTSTRAP AND NOT A t-TEST
The per-query outcome for Success@k is a 0/1 indicator, its paired difference lives in
{-1, 0, +1}, and the quantity of interest is the mean of 500 of them. A bootstrap makes
no distributional claim about that; it resamples the queries actually measured. It is
also the same procedure for MRR, whose per-query terms are 1/rank and badly non-normal,
so one convention covers the primary metric and its secondaries.

WHAT THE PAIRING ACTUALLY MEANS IN THE CODE
The resampled index set is drawn ONCE per replicate and applied to BOTH arms
(:func:`paired_bootstrap` resamples the difference vector, which is the same thing and
makes the invariant impossible to get wrong). Resampling each arm independently would
reintroduce exactly the between-query variance the pairing exists to remove, and would
silently widen every interval. That is the whole method, and it is one line.

WHAT THIS MODULE DOES NOT DO
It does not know the gate. The pre-registered rule -- a one-sided 95% lower bound above
0.02 on both seeds -- lives in the pre-registration; this returns the bound and the
provenance needed to record how it was made, and the caller compares.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

RESAMPLES = 10_000
"""Bootstrap replicates, pre-registered. Not a tuning knob: at 10,000 the Monte-Carlo
noise on a 5th percentile is small against the 0.02 bar, and changing it after seeing a
result would be choosing the resample count that gives the answer."""

CONFIDENCE = 0.95
"""One-sided confidence level. The gate reads a LOWER bound, so the interval is
one-sided by construction -- a two-sided 95% interval's lower end is the 2.5th
percentile and would be a different, stricter number reported under the same name."""

LOWER_PERCENTILE = 5.0
"""The percentile of the bootstrap distribution taken as the one-sided lower bound,
i.e. `100 * (1 - CONFIDENCE)`. Written as a literal rather than computed: the arithmetic
gives 5.000000000000004 in binary floating point, and a percentile a receipt records as
"5.000000000000004" invites a reader to wonder what convention that was."""


@dataclass(frozen=True)
class PairedBootstrapResult:
    """A paired-bootstrap bound plus everything needed to say how it was produced.

    The provenance fields are not decoration. A bound of 0.024 means one thing from a
    one-sided 95% percentile bootstrap over 10,000 resamples of 500 paired queries and
    another entirely from a two-sided interval, a different resample count, or an
    unpaired resample -- and a receipt that records only the number cannot tell those
    apart later. They are literal fields so a receipt carries them without the emitter
    having to remember to write them down.

    Attributes:
        n_queries: Paired queries the bound is over -- the shared queries, after
            alignment, not either arm's own count.
        resamples: Bootstrap replicates drawn.
        seed: Seed handed to `numpy.random.default_rng`; replays the exact bound.
        point_estimate: Mean paired difference, treatment minus control, on the observed
            data. Not the mean of the bootstrap distribution, which is a resampling
            artefact and is slightly different.
        lower_bound: The one-sided lower confidence bound on that mean.
        confidence: Confidence level of the bound (0.95).
        interval: `"one-sided-lower"` -- which end of which interval this is.
        method: `"paired-percentile-bootstrap"` -- percentile method, no BCa correction
            and no bias/acceleration estimate.
        percentile: The percentile of the bootstrap distribution taken as the bound
            (5.0 at 95% one-sided).
        interpolation: `numpy.percentile`'s interpolation, `"linear"`. Named because the
            bound falls between two of 10,000 order statistics and the choice moves it.
    """

    n_queries: int
    resamples: int
    seed: int
    point_estimate: float
    lower_bound: float
    confidence: float = CONFIDENCE
    interval: str = "one-sided-lower"
    method: str = "paired-percentile-bootstrap"
    percentile: float = LOWER_PERCENTILE
    interpolation: str = "linear"

    def as_receipt(self) -> dict[str, Any]:
        """This result as a plain dict, for embedding in a run receipt.

        Returns:
            Every field by name, including the convention fields -- a receipt that keeps
            only `lower_bound` records a number nobody can re-derive.
        """
        return asdict(self)


def align_paired_arms(
    control_ids: Sequence[str],
    control_values: Sequence[float],
    treatment_ids: Sequence[str],
    treatment_values: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Put two arms' per-query values on the same queries, in the same order, or refuse.

    WHY THIS REFUSES INSTEAD OF INTERSECTING
    The obvious convenience -- take the queries the two arms share and carry on -- is the
    failure this guards. A paired contrast is only interpretable over the pre-registered
    query set; silently dropping the queries one arm failed to score turns "the treatment
    is better" into "the treatment is better on the queries it managed", which is a
    different and much easier claim. If an arm is missing queries, that is an incomplete
    evaluation and the run should be fixed, not the comparison narrowed.

    Args:
        control_ids: Query ids for the control arm, in the order its values are given.
        control_values: The control arm's per-query metric, one entry per id.
        treatment_ids: Query ids for the treatment arm.
        treatment_values: The treatment arm's per-query metric.

    Returns:
        `(control, treatment)` as float64 `[Q]` arrays, both in `control_ids`' order.
        float64 because everything downstream is a difference of differences; the
        float32 pinning that matters is on the metric itself, upstream of here (see
        `metrics.recall_at_k`), and the values arrive already rounded to it.

    Raises:
        ValueError: If either arm's ids and values disagree in length, if an arm repeats
            a query id (which would make "the same query in both arms" ambiguous), or if
            the two arms do not cover exactly the same query ids.
    """
    if len(control_ids) != len(control_values):
        raise ValueError(f"control arm has {len(control_ids)} ids but {len(control_values)} values")
    if len(treatment_ids) != len(treatment_values):
        raise ValueError(
            f"treatment arm has {len(treatment_ids)} ids but {len(treatment_values)} values"
        )
    if len(control_ids) != len(treatment_ids):
        raise ValueError(
            f"paired arms must cover the same queries: control has {len(control_ids)}, "
            f"treatment has {len(treatment_ids)}. Refusing to truncate to the overlap -- "
            f"that would silently change which queries the contrast is about."
        )

    control_order = {qid: i for i, qid in enumerate(control_ids)}
    if len(control_order) != len(control_ids):
        raise ValueError("control arm repeats a query id; a paired arm needs one row per query")
    treatment_order = {qid: i for i, qid in enumerate(treatment_ids)}
    if len(treatment_order) != len(treatment_ids):
        raise ValueError("treatment arm repeats a query id; a paired arm needs one row per query")

    missing = sorted(set(control_order) - set(treatment_order))
    if missing:
        raise ValueError(
            f"paired arms cover different queries: {len(missing)} in control and not in "
            f"treatment, e.g. {missing[:5]}"
        )

    control = np.asarray(control_values, dtype=np.float64)
    treatment = np.asarray(
        [treatment_values[treatment_order[qid]] for qid in control_ids], dtype=np.float64
    )
    return control, treatment


def paired_bootstrap(
    control: Sequence[float],
    treatment: Sequence[float],
    *,
    seed: int,
    resamples: int = RESAMPLES,
) -> PairedBootstrapResult:
    """One-sided 95% lower bound on the mean paired difference, treatment minus control.

    THE CONVENTION, STATED
    Percentile bootstrap. Query INDICES are drawn with replacement, `resamples` times,
    each replicate the same size as the query set; the SAME index set is applied to both
    arms, which is what "paired" means and is why the difference vector is what gets
    resampled. The reported bound is the `100*(1-confidence)`th percentile -- the **5th**
    at 95% -- of the resampled mean differences, with `numpy.percentile`'s default linear
    interpolation between order statistics. No BCa correction is applied.

    Args:
        control: Per-query metric for the control arm, in query order.
        treatment: Per-query metric for the treatment arm, in the SAME query order --
            use :func:`align_paired_arms` if the two arms carry their own id lists.
        seed: Seed for `numpy.random.default_rng`. Required, not defaulted: an unseeded
            bound is a number that cannot be reproduced, and the receipt records this
            value so the bound can be replayed byte-for-byte in another process.
        resamples: Bootstrap replicates. Defaults to the pre-registered 10,000.

    Returns:
        A :class:`PairedBootstrapResult` carrying the point estimate, the bound, and the
        convention that produced it.

    Raises:
        ValueError: If the arms differ in length (they are not paired), if either is
            empty (nothing to resample), or if `resamples` is below 1.
    """
    if len(control) != len(treatment):
        raise ValueError(
            f"paired arms must be the same length: control {len(control)}, "
            f"treatment {len(treatment)}"
        )
    if len(control) == 0:
        raise ValueError("paired bootstrap needs at least one query")
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")

    differences = np.asarray(treatment, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    n = differences.size

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n, size=(resamples, n), dtype=np.int64)
    replicate_means = differences[picks].mean(axis=1)

    return PairedBootstrapResult(
        n_queries=n,
        resamples=resamples,
        seed=seed,
        point_estimate=float(differences.mean()),
        lower_bound=float(np.percentile(replicate_means, LOWER_PERCENTILE, method="linear")),
    )
