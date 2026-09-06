"""Lane IC-7 -- `cogsyndelta.interconnect.losses`: shapes, masks, determinism, boundary cases.

`docs/design/INTERCONNECT-MODULE-SPEC.md` section 5 Table 9, `losses.py` row: "each loss is
finite on random inputs; `L_B` is zero when `s_hat = a`; the FLOPs penalty is zero at target and
positive above it." The Table 9 "all" row additionally requires "construction and forward under
`torch.manual_seed(0)` are bitwise identical across two runs on CPU," reproduced here as
`test_determinism_under_fixed_seed`.

`losses.py` has no Table 8 guard rows of its own (none of G27-G36 fires inside this file), so
this file carries no `test_guards_can_fail.py`-style cases.
"""

from __future__ import annotations

import pytest
import torch

from cogsyndelta.interconnect.losses import DistilLoss, FlopsPenalty, RankLoss, UnifyLoss

pytestmark = pytest.mark.cpu

_D_W = 512
_K = 32
_I = 4
_R = 5
_POOLED_DIMS = {"language": 256, "memory": 256, "reasoning": 256, "visual": 384}


def _seeded(batch: int, *shape: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, *shape)


# ---------------------------------------------------------------------------
# RankLoss
# ---------------------------------------------------------------------------


def test_rank_loss_finite_on_random_inputs() -> None:
    loss = RankLoss()
    scores = _seeded(6, _K, seed=1)
    target = torch.randint(0, _K, (6,))
    value = loss(scores, target)
    assert value.shape == ()
    assert torch.isfinite(value)


def test_rank_loss_weight_scales_linearly() -> None:
    scores = _seeded(4, _K, seed=2)
    target = torch.randint(0, _K, (4,))
    unweighted = RankLoss(weight=1.0)(scores, target)
    weighted = RankLoss(weight=3.0)(scores, target)
    assert torch.allclose(weighted, 3.0 * unweighted)


def test_rank_loss_null_target_is_a_valid_index() -> None:
    """The general-bin correct answer is `NULL` at index 0 (spec section 3 step 12)."""
    scores = _seeded(3, _K, seed=3)
    target = torch.zeros(3, dtype=torch.int64)
    value = RankLoss()(scores, target)
    assert torch.isfinite(value)


# ---------------------------------------------------------------------------
# UnifyLoss
# ---------------------------------------------------------------------------


def _probe_and_targets(batch: int, seed: int) -> tuple[dict, dict]:
    torch.manual_seed(seed)
    probes = {name: torch.randn(batch, dim) for name, dim in _POOLED_DIMS.items()}
    targets = {name: torch.randn(batch, dim) for name, dim in _POOLED_DIMS.items()}
    return probes, targets


def test_unify_loss_finite_on_random_inputs() -> None:
    probes, targets = _probe_and_targets(5, seed=4)
    value = UnifyLoss(weight=0.5)(probes, targets)
    assert value.shape == ()
    assert torch.isfinite(value)


def test_unify_loss_zero_when_probes_equal_targets() -> None:
    """`1 - cos(x, x) == 0` for every region, so the summed loss is exactly zero."""
    probes, _ = _probe_and_targets(4, seed=5)
    value = UnifyLoss(weight=1.0)(probes, probes)
    assert torch.allclose(value, torch.zeros(()), atol=1e-6)


def test_unify_loss_missing_target_region_raises() -> None:
    probes, targets = _probe_and_targets(2, seed=6)
    del targets["visual"]
    with pytest.raises(KeyError):
        UnifyLoss(weight=1.0)(probes, targets)


def test_unify_loss_admitted_mask_excludes_unadmitted_regions() -> None:
    """A region masked out for every item drops out of the sum regardless of its cosine term."""
    probes, targets = _probe_and_targets(3, seed=7)
    all_admitted = {name: torch.ones(3, dtype=torch.bool) for name in _POOLED_DIMS}
    none_admitted = {name: torch.zeros(3, dtype=torch.bool) for name in _POOLED_DIMS}
    full = UnifyLoss(weight=1.0)(probes, targets, admitted=all_admitted)
    dropped = UnifyLoss(weight=1.0)(probes, targets, admitted=none_admitted)
    assert torch.allclose(dropped, torch.zeros(()), atol=1e-6)
    assert not torch.allclose(full, dropped)


def test_unify_loss_default_admitted_matches_all_true() -> None:
    """`admitted=None` (phase A's dense schedule) equals every region explicitly admitted."""
    probes, targets = _probe_and_targets(3, seed=8)
    all_admitted = {name: torch.ones(3, dtype=torch.bool) for name in _POOLED_DIMS}
    default = UnifyLoss(weight=1.0)(probes, targets)
    explicit = UnifyLoss(weight=1.0)(probes, targets, admitted=all_admitted)
    assert torch.allclose(default, explicit)


# ---------------------------------------------------------------------------
# DistilLoss
# ---------------------------------------------------------------------------


def _distil_inputs(
    batch: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    teacher = torch.randn(batch, _I, _R)
    student = torch.randn(batch, _I, _R)
    active = torch.ones(batch, _I, dtype=torch.bool)
    b_hat = torch.rand(batch, _R)
    return teacher, student, active, b_hat


def test_distil_loss_finite_on_random_inputs() -> None:
    teacher, student, active, b_hat = _distil_inputs(4, seed=9)
    value = DistilLoss(temperature=1.0, beta=0.1)(teacher, student, active, b_hat, b_read=64.0)
    assert value.shape == ()
    assert torch.isfinite(value)


def test_distil_loss_zero_when_student_equals_teacher_and_budget_conserved() -> None:
    """`L_B` is zero when `s_hat = a` (Table 9) and `Sum_r b_hat_r == B_read` exactly."""
    teacher, _student, active, _b_hat = _distil_inputs(3, seed=10)
    b_read = 64.0
    b_hat = torch.full((3, _R), b_read / _R)
    value = DistilLoss(temperature=0.5, beta=1.0)(teacher, teacher, active, b_hat, b_read=b_read)
    assert torch.allclose(value, torch.zeros(()), atol=1e-5)


def test_distil_loss_inactive_iterations_do_not_contribute() -> None:
    """Zeroing `active` for an iteration removes its KL contribution."""
    teacher, student, _active, b_hat = _distil_inputs(2, seed=11)
    b_hat = torch.full_like(b_hat, 64.0 / _R)
    all_active = torch.ones(2, _I, dtype=torch.bool)
    none_active = torch.zeros(2, _I, dtype=torch.bool)
    with_kl = DistilLoss(temperature=1.0, beta=0.0)(teacher, student, all_active, b_hat, 64.0)
    without_kl = DistilLoss(temperature=1.0, beta=0.0)(teacher, student, none_active, b_hat, 64.0)
    assert torch.allclose(without_kl, torch.zeros(()), atol=1e-6)
    assert not torch.allclose(with_kl, without_kl)


def test_distil_loss_budget_term_scales_with_beta() -> None:
    teacher, student, active, _b_hat = _distil_inputs(2, seed=12)
    # Same tensor on both sides zeroes the KL term, isolating the budget term.
    b_hat = torch.full((2, _R), 20.0)  # sums to 100, off target
    low_beta = DistilLoss(temperature=1.0, beta=0.1)(teacher, teacher, active, b_hat, b_read=64.0)
    high_beta = DistilLoss(temperature=1.0, beta=1.0)(teacher, teacher, active, b_hat, b_read=64.0)
    assert torch.isfinite(low_beta)
    assert torch.isfinite(high_beta)
    assert high_beta > low_beta


def test_distil_loss_halt_term_defaults_to_skipped() -> None:
    teacher, student, active, b_hat = _distil_inputs(2, seed=13)
    b_hat = torch.full_like(b_hat, 64.0 / _R)
    without_halt = DistilLoss(temperature=1.0, beta=0.0)(teacher, student, active, b_hat, 64.0)
    halt_logits = torch.randn(2, _I)
    halt_target = torch.zeros(2, dtype=torch.int64)
    with_halt = DistilLoss(temperature=1.0, beta=0.0, halt_weight=1.0)(
        teacher, student, active, b_hat, 64.0, halt_logits=halt_logits, halt_target=halt_target
    )
    assert torch.isfinite(with_halt)
    assert not torch.allclose(with_halt, without_halt)


def test_distil_loss_nonpositive_temperature_raises() -> None:
    with pytest.raises(ValueError):
        DistilLoss(temperature=0.0, beta=0.0)


# ---------------------------------------------------------------------------
# FlopsPenalty
# ---------------------------------------------------------------------------


def test_flops_penalty_finite_on_random_inputs() -> None:
    torch.manual_seed(14)
    A = torch.rand(4, _I, 4)
    phi = torch.rand(4) * 100
    ctx = torch.rand(4, 4) * 64
    value = FlopsPenalty(weight=1.0, flops_target=1_000_000.0)(A, phi, ctx)
    assert value.shape == ()
    assert torch.isfinite(value)


def test_flops_penalty_zero_at_target() -> None:
    """The FLOPs penalty is zero at target (Table 9)."""
    A = torch.ones(2, _I, 2)
    phi = torch.tensor([10.0, 20.0])
    ctx = torch.tensor([[5.0, 5.0], [5.0, 5.0]])
    # FLOPs per item = I * (phi[0]*ctx[0] + phi[1]*ctx[1]) = 4 * (50 + 100) = 600
    flops_target = 4 * (10.0 * 5.0 + 20.0 * 5.0)
    value = FlopsPenalty(weight=1.0, flops_target=flops_target)(A, phi, ctx)
    assert torch.allclose(value, torch.zeros(()), atol=1e-6)


def test_flops_penalty_positive_above_target() -> None:
    """The FLOPs penalty is positive above target (Table 9)."""
    A = torch.ones(2, _I, 2)
    phi = torch.tensor([10.0, 20.0])
    ctx = torch.tensor([[5.0, 5.0], [5.0, 5.0]])
    exact_flops = 4 * (10.0 * 5.0 + 20.0 * 5.0)
    value = FlopsPenalty(weight=1.0, flops_target=exact_flops * 0.5)(A, phi, ctx)
    assert value > 0.0
    assert torch.isfinite(value)


def test_flops_penalty_scales_with_weight() -> None:
    A = torch.ones(1, _I, 2)
    phi = torch.tensor([10.0, 20.0])
    ctx = torch.tensor([[5.0, 5.0]])
    target = 4 * (10.0 * 5.0 + 20.0 * 5.0) * 0.5
    low = FlopsPenalty(weight=1.0, flops_target=target)(A, phi, ctx)
    high = FlopsPenalty(weight=2.0, flops_target=target)(A, phi, ctx)
    assert torch.allclose(high, 2.0 * low)


def test_flops_penalty_mismatched_region_axis_raises() -> None:
    A = torch.ones(1, _I, 3)
    phi = torch.tensor([1.0, 2.0])  # wrong length
    ctx = torch.ones(1, 3)
    with pytest.raises(ValueError):
        FlopsPenalty(weight=1.0, flops_target=1.0)(A, phi, ctx)


def test_flops_penalty_nonpositive_target_raises() -> None:
    with pytest.raises(ValueError):
        FlopsPenalty(weight=1.0, flops_target=0.0)


# ---------------------------------------------------------------------------
# Determinism (Table 9 "all" row)
# ---------------------------------------------------------------------------


def test_determinism_under_fixed_seed() -> None:
    """Construction and forward under `torch.manual_seed(0)` are bitwise identical across runs."""

    def _run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        scores = torch.randn(4, _K)
        target = torch.randint(0, _K, (4,))
        rank = RankLoss()(scores, target)

        probes, targets = _probe_and_targets(4, seed=0)
        unify = UnifyLoss(weight=0.3)(probes, targets)

        teacher, student, active, b_hat = _distil_inputs(4, seed=0)
        distil = DistilLoss(temperature=1.0, beta=0.1)(teacher, student, active, b_hat, 64.0)

        A = torch.rand(4, _I, 4)
        phi = torch.rand(4) * 100
        ctx = torch.rand(4, 4) * 64
        flops = FlopsPenalty(weight=1.0, flops_target=1_000_000.0)(A, phi, ctx)
        return rank, unify, distil, flops

    first = _run()
    second = _run()
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)
