"""Representation geometry: does quantization move the encoder's OUTPUT, not just its
score on a linear probe.

WHY THIS EXISTS
`docs/design/evidence/visual-ptq-sensitivity-2026-09-06/README.md` measured every
linear-probe read-out this project has (the pooled EuroSAT probe, a Fashion transfer
probe, a pre-pool token-surface probe) staying flat within noise across the visual
region's whole 3-to-8-bit PTQ ladder, while the encoder's actual output geometry moved
by an order of magnitude more over the SAME ladder: per-image cosine similarity to the
fp32 latent fell from 0.999972 at 8-bit to 0.952436 at 3-bit, and top-10
nearest-neighbour identity agreement fell from 99.6% to 90.8%. The verdict there is
"the probe is insensitive, not the encoder" -- a mean-pooled linear classifier's
decision boundary survives a disturbance that a nearest-neighbour retrieval consumer,
or anything reading patch tokens directly, would not. This module is what makes that
measurement a receipt field every quantized region reports, rather than a one-off
evidence script re-run by hand.

WHAT THIS DOES NOT MEASURE
Task accuracy -- see `cogsyndelta.eval.benchmark` / `cogsyndelta.eval.metrics` for
that. Holding the ITEMS fixed, this module answers a narrower question: how far did the
encoder's OUTPUT move when its WEIGHTS were quantized. A `mean_cosine` near 1.0 next to
an unmoved probe score is not proof nothing changed underneath -- it is exactly the
shape the evidence above shows CAN coexist with real, monotonic geometry drift.

NO THRESHOLDS HERE
This module reports numbers; it does not gate on them. The evidence dir's ladder shows
geometry drift is continuous and region/bit-width dependent, and no operator tolerance
has been set yet (see `docs/design/METRICS-METHODOLOGY.md`'s representation-geometry
section). A caller wanting a pass/fail bound must add one explicitly, elsewhere.

G27
`compute_geometry` compares two latent matrices row-for-row: row `i` of
`fp32_latents` and row `i` of `quantized_latents` must be the SAME item, in the SAME
order, or every number this module returns describes a mismatched pairing rather than
quantization. `GeometryReference` names which held-out set a side was computed from
(a content fingerprint plus a row count); `verify_geometry_reference` refuses,
fail-closed, when the two sides disagree -- the same shape G26 already fixed for the
text held-out split (`cogsyndelta.splits.SplitGuardError`), applied here to a pairwise
comparison instead of a train/eval partition.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

#: Neighbours considered for `nn_agreement_at_k` -- matches
#: `docs/design/evidence/visual-ptq-sensitivity-2026-09-06/measure_visual_ptq_sensitivity.py`'s
#: `NN_K`. The receipt field this module's callers write is literally named after this
#: number (`quant.geometry.nn_agreement_at_10`), so changing the default renames the
#: field, not just a threshold.
DEFAULT_NN_K = 10

#: The percentile `p05_cosine` reports -- the evidence dir's own `min_cosine` is a
#: single worst-case item and can be an outlier; the 5th percentile is the smallest
#: value a caller can read as "the bad end of the distribution" without one item
#: dominating it.
P05_QUANTILE = 0.05


class GeometryReferenceError(RuntimeError):
    """G27 fail-closed: the fp32 and quantized sides of a geometry comparison disagree
    on which items (or how many) they were computed over.
    """


@dataclass(frozen=True)
class GeometryReference:
    """Identifies WHICH held-out items one side of a geometry comparison covers.

    Two `GeometryReference`s must be equal (per `verify_geometry_reference`) before
    `compute_geometry` may run on the latents they describe. `split_sha256` is a
    content fingerprint of the item set/order -- the text battery's own
    `split.sha256` (G26) when one exists, or a fingerprint computed over the decoded
    eval tensor for a battery with no split-manifest of its own (visual) -- never a
    path or a config value, both of which can name the same bytes under two different
    strings or two different byte sets under the same string.
    """

    split_sha256: str
    """Content fingerprint of the item set/order this side was computed over."""
    n_items: int
    """Row count of the latent matrix this reference describes -- a second, cheap
    check that catches a truncated or padded batch even on a `split_sha256` collision
    (a 16-byte digest is compared, not re-derived from the latents themselves, so a
    mismatched row count is otherwise invisible to `verify_geometry_reference`)."""


def verify_geometry_reference(
    fp32_reference: GeometryReference, quantized_reference: GeometryReference
) -> None:
    """Refuse, fail-closed, unless both sides of a geometry comparison name the same
    items in the same order.

    Args:
        fp32_reference: what the fp32-side latents were computed over.
        quantized_reference: what the quantized-side latents were computed over.

    Raises:
        GeometryReferenceError: the two references disagree on `split_sha256` or on
            `n_items` -- either means the two latent matrices this call is about to
            compare are not a row-for-row pairing of the same items.
    """
    if fp32_reference.split_sha256 != quantized_reference.split_sha256:
        raise GeometryReferenceError(
            "G27: fp32 and quantized latents were not produced on the same split -- "
            f"fp32 split_sha256={fp32_reference.split_sha256!r} != quantized "
            f"split_sha256={quantized_reference.split_sha256!r}. Refusing to compute "
            "representation-geometry metrics across two different item sets."
        )
    if fp32_reference.n_items != quantized_reference.n_items:
        raise GeometryReferenceError(
            "G27: fp32 and quantized latents disagree on item count -- "
            f"fp32 n_items={fp32_reference.n_items} != quantized "
            f"n_items={quantized_reference.n_items}. Refusing to compute "
            "representation-geometry metrics over a truncated or padded batch on one "
            "side."
        )


def _topk_neighbor_mask(latents: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean `[n, n]` mask: `mask[i, j]` iff `j` is one of `i`'s top-`k` cosine
    neighbours. Self-similarity is excluded before the top-k, so a latent is never its
    own neighbour.

    Identical definition to
    `docs/design/evidence/visual-ptq-sensitivity-2026-09-06/measure_visual_ptq_sensitivity.py`'s
    `_topk_neighbor_mask` -- reproduced here (rather than imported from a `docs/`
    evidence script, which is not a package) so the receipt field and the evidence
    that motivated it mean the identical thing.
    """
    x = F.normalize(latents, dim=1)
    sim = x @ x.T
    sim.fill_diagonal_(-2.0)
    idx = sim.topk(k, dim=1).indices
    mask = torch.zeros_like(sim, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def compute_geometry(
    fp32_latents: torch.Tensor,
    quantized_latents: torch.Tensor,
    *,
    k: int = DEFAULT_NN_K,
) -> dict[str, float]:
    """Per-item cosine drift and neighbourhood-rank churn between two pooled latent
    matrices for the SAME items in the SAME order.

    Deterministic: no sampling, no RNG. `torch.topk`/`torch.quantile` are exact given
    fixed input tensors (ties in a top-k selection are the only source of
    run-to-run variation possible here, and this function is never called on latents
    engineered to tie).

    Args:
        fp32_latents: `[n_items, dim]`, the reference (unquantized) representation.
        quantized_latents: `[n_items, dim]`, the same items through the quantized
            model, in the same row order. Callers MUST establish this pairing via
            `verify_geometry_reference` before calling this function -- it is not
            re-checked here, because this function has no access to either side's
            `GeometryReference` (it sees only the tensors).
        k: neighbours considered for `nn_agreement_at_k`. Defaults to
            `DEFAULT_NN_K` (10), matching the evidence this module derives from.

    Returns:
        A flat `dict[str, float]`, safe to merge directly into a receipt's `metrics`:

        - `mean_cosine`: mean over items of `cosine_similarity(fp32_i, quantized_i)`.
        - `min_cosine`: the single worst item's cosine similarity.
        - `p05_cosine`: the `P05_QUANTILE` (5th-percentile) cosine similarity --
          the smallest value not dominated by one outlier item.
        - `nn_agreement_at_{k}`: mean over items of `|top-k(fp32_i) ∩ top-k(quantized_i)|
          / k` -- IDENTITY agreement, not Jaccard. Both neighbour sets have exactly
          `k` members by construction, so this fraction is simultaneously each set's
          precision and recall against the other; true Jaccard
          (`|intersection| / |union|`) would read SMALLER for the identical case
          (`|union| = 2k - |intersection|`) whenever the two sets are not equal. This
          is the same quantity
          `measure_visual_ptq_sensitivity.py`'s `_geometry_vs_fp32` reports as
          `nn_agreement_at_k`, computed the same way, so the two numbers mean the
          same thing.
        - `latent_std_ratio`: `quantized_latents.std(dim=0).mean() /
          fp32_latents.std(dim=0).mean()` -- a second, cheap collapse signal on this
          exact population (1.0 means the quantized population is exactly as spread
          out as the fp32 one; the evidence dir's `eval_latents_std_ratio` is this
          same formula).

    Raises:
        ValueError: the two latent matrices disagree in shape, are not 2-D, or there
            are not more items than `k` (an `nn_agreement_at_k` over `n <= k` items
            would need every OTHER item as a neighbour, which is not what "top-k" means).
    """
    if fp32_latents.shape != quantized_latents.shape:
        raise ValueError(
            f"fp32_latents shape {tuple(fp32_latents.shape)} != quantized_latents "
            f"shape {tuple(quantized_latents.shape)} -- refusing to compare mismatched "
            "latents"
        )
    if fp32_latents.ndim != 2:
        raise ValueError(f"expected latents shaped [n_items, dim], got {tuple(fp32_latents.shape)}")
    n = fp32_latents.shape[0]
    if n <= k:
        raise ValueError(f"nn_agreement_at_{k} needs more than {k} items, got {n}")

    fp32 = fp32_latents.detach().to(torch.float32)
    quantized = quantized_latents.detach().to(torch.float32)

    cos = F.cosine_similarity(fp32, quantized, dim=1)
    fp32_mask = _topk_neighbor_mask(fp32, k)
    quantized_mask = _topk_neighbor_mask(quantized, k)
    agreement = (fp32_mask & quantized_mask).sum(dim=1).to(torch.float32) / k

    fp32_std = fp32.std(dim=0).mean().item()
    quantized_std = quantized.std(dim=0).mean().item()

    return {
        "mean_cosine": float(cos.mean().item()),
        "min_cosine": float(cos.min().item()),
        "p05_cosine": float(torch.quantile(cos, P05_QUANTILE).item()),
        f"nn_agreement_at_{k}": float(agreement.mean().item()),
        "latent_std_ratio": float(quantized_std / max(1e-9, fp32_std)),
    }
