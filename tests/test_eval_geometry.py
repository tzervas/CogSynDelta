"""Unit tests for `cogsyndelta.eval.geometry` -- deterministic on synthetic data.

Two shapes are covered against an INDEPENDENT reference, not against the module's own
formulas read back: the identity case (fp32 and quantized latents are the same
tensor -- every metric must read as "nothing moved"), and a hand-constructed
known-degraded case scored against a plain-Python reference implementation (nested
loops, no torch, no shared code with `geometry.py`) so a bug in the vectorized
implementation cannot also be baked into the check. The fail-closed reference guard
(`verify_geometry_reference` refusing a mismatched pair) is proved separately in
`tests/test_guards_can_fail.py`, per that file's own "add to it" convention.
"""

from __future__ import annotations

import math

import pytest
import torch

from cogsyndelta.eval.geometry import (
    GeometryReference,
    compute_geometry,
    verify_geometry_reference,
)

pytestmark = pytest.mark.cpu


def test_identity_latents_report_perfect_agreement() -> None:
    """fp32 and quantized are the SAME tensor: every metric must read as untouched."""
    torch.manual_seed(0)
    latents = torch.randn(15, 6)

    result = compute_geometry(latents, latents.clone(), k=3)

    assert result["mean_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert result["min_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert result["p05_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert result["nn_agreement_at_3"] == pytest.approx(1.0, abs=1e-9)
    assert result["latent_std_ratio"] == pytest.approx(1.0, abs=1e-6)


def _reference_geometry(
    fp32_rows: list[list[float]], quant_rows: list[list[float]], k: int
) -> dict:
    """Independent, unvectorized reference: plain Python loops, no torch, no shared
    formula with `cogsyndelta.eval.geometry`. Ties in top-k are avoided by construction
    in the case this is used on, so `sorted(..., reverse=True)` needs no tie-break rule
    to agree with `compute_geometry`'s `torch.topk`.
    """

    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    def norm(a: list[float]) -> float:
        return math.sqrt(dot(a, a))

    def cosine(a: list[float], b: list[float]) -> float:
        return dot(a, b) / (norm(a) * norm(b))

    n = len(fp32_rows)
    per_item_cosine = [cosine(fp32_rows[i], quant_rows[i]) for i in range(n)]

    def topk_neighbors(rows: list[list[float]], i: int, k: int) -> set[int]:
        sims = sorted(
            ((j, cosine(rows[i], rows[j])) for j in range(len(rows)) if j != i),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return {j for j, _ in sims[:k]}

    agreements = []
    for i in range(n):
        fp32_nn = topk_neighbors(fp32_rows, i, k)
        quant_nn = topk_neighbors(quant_rows, i, k)
        agreements.append(len(fp32_nn & quant_nn) / k)

    def sample_std(values: list[float]) -> float:
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(var)

    dim = len(fp32_rows[0])
    fp32_col_std = [sample_std([row[d] for row in fp32_rows]) for d in range(dim)]
    quant_col_std = [sample_std([row[d] for row in quant_rows]) for d in range(dim)]
    fp32_mean_std = sum(fp32_col_std) / dim
    quant_mean_std = sum(quant_col_std) / dim

    sorted_cos = sorted(per_item_cosine)
    pos = 0.05 * (n - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    frac = pos - lo
    p05 = sorted_cos[lo] + frac * (sorted_cos[hi] - sorted_cos[lo])

    return {
        "mean_cosine": sum(per_item_cosine) / n,
        "min_cosine": min(per_item_cosine),
        "p05_cosine": p05,
        f"nn_agreement_at_{k}": sum(agreements) / n,
        "latent_std_ratio": quant_mean_std / fp32_mean_std,
    }


def test_known_degraded_case_matches_an_independent_reference_implementation() -> None:
    """Four unit vectors; only one is perturbed on the quantized side, changing that
    one item's nearest neighbour and depressing its own cosine to 0 while the other
    three are untouched. Scored against `_reference_geometry` above, a from-scratch
    Python implementation sharing no code with `compute_geometry`.
    """
    fp32 = [
        [1.0, 0.0],  # A
        [0.0, 1.0],  # B
        [-1.0, 0.0],  # C
        [0.6, 0.8],  # D
    ]
    # Only C moves (to the exact opposite of B); A, B, D are byte-identical to fp32.
    quantized = [
        [1.0, 0.0],  # A, unchanged
        [0.0, 1.0],  # B, unchanged
        [0.0, -1.0],  # C, perturbed
        [0.6, 0.8],  # D, unchanged
    ]

    expected = _reference_geometry(fp32, quantized, k=1)
    result = compute_geometry(torch.tensor(fp32), torch.tensor(quantized), k=1)

    assert result.keys() == expected.keys()
    for key, value in expected.items():
        assert result[key] == pytest.approx(value, abs=1e-6), key

    # Sanity on the construction itself, independent of both implementations: C's own
    # cosine collapsed to 0 (perpendicular, not just "lower"), and it is the minimum.
    assert result["min_cosine"] == pytest.approx(0.0, abs=1e-9)
    # C's nearest neighbour changed (B -> A); the other three items' did not, so 3 of 4
    # items keep perfect top-1 agreement and one drops to zero.
    assert result["nn_agreement_at_1"] == pytest.approx(0.75, abs=1e-9)


def test_shape_mismatch_is_refused() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_geometry(torch.randn(10, 4), torch.randn(9, 4))


def test_non_2d_input_is_refused() -> None:
    with pytest.raises(ValueError, match="n_items, dim"):
        compute_geometry(torch.randn(10), torch.randn(10))


def test_too_few_items_for_k_is_refused() -> None:
    with pytest.raises(ValueError, match="nn_agreement_at_5"):
        compute_geometry(torch.randn(5, 4), torch.randn(5, 4), k=5)


def test_the_neighbour_field_name_carries_k() -> None:
    """The receipt field name IS `nn_agreement_at_{k}` -- callers writing a fixed k=10
    field must see that key literally, not a generically-named one they rename."""
    result = compute_geometry(torch.randn(20, 4), torch.randn(20, 4), k=7)
    assert "nn_agreement_at_7" in result
    assert "nn_agreement_at_10" not in result


def test_verify_geometry_reference_accepts_a_matching_pair() -> None:
    """Positive control for the guard proved (failing) in test_guards_can_fail.py --
    the ordinary case, both sides describing the same items, must not raise."""
    ref = GeometryReference(split_sha256="abc123", n_items=42)
    verify_geometry_reference(ref, GeometryReference(split_sha256="abc123", n_items=42))
