"""Tests for `cogsyndelta.interconnect.controller` -- spec section 5 Table 9's
`controller.py` row.

Covers: both simplexes stay inside their bounds for 10,000 random inputs and for
adversarial inputs built to starve a region; `box_integerise` sums to `B_read` exactly
and respects every box, including the three fixtures spec section 5 Table 9 names by
value (the draft-1 counter-example, the cascading-cap case, and the two dense
allocations with their tie-breaks); `Σ_i A[i, r] >= 1`; `halt_at <= max_iters` for any
logits; an input exists whose `halt_at < n_iter`; construction and forward under
`torch.manual_seed(0)` are bitwise identical across two runs on CPU (spec section 5
Table 9's "all" row).
"""

from __future__ import annotations

import pytest
import torch

from cogsyndelta.interconnect.controller import (
    ControllerConfigError,
    ControllerParticipant,
    FlooredSimplex,
    ThalamicController,
    _admit,
    _halt,
    box_integerise,
)

pytestmark = pytest.mark.cpu

B_READ = 256
B_KV = 3_221_225_472
ETA = 0.15
N_ITER = 4


def _v1_participants() -> dict[str, ControllerParticipant]:
    """Spec Table 1 / Table 4a's five v1 participants, in Table 1 order."""
    return {
        "language": ControllerParticipant(
            summary_dim=256,
            ctx_min=8,
            ctx_max=96,
            kv_bytes_per_token=4096,
            token_budget_min=1,
            token_budget_max=96,
        ),
        "memory": ControllerParticipant(
            summary_dim=256,
            ctx_min=8,
            ctx_max=96,
            kv_bytes_per_token=4096,
            token_budget_min=1,
            token_budget_max=96,
        ),
        "reasoning": ControllerParticipant(
            summary_dim=256,
            ctx_min=8,
            ctx_max=256,
            kv_bytes_per_token=4096,
            token_budget_min=1,
            token_budget_max=256,
        ),
        "visual": ControllerParticipant(
            summary_dim=384,
            ctx_min=8,
            ctx_max=64,
            kv_bytes_per_token=9216,
            token_budget_min=1,
            token_budget_max=64,
        ),
        "episodic_store": ControllerParticipant(
            summary_dim=4,
            ctx_min=None,
            ctx_max=None,
            kv_bytes_per_token=None,
            token_budget_min=8,
            token_budget_max=256,
        ),
    }


def _r4_participants() -> dict[str, ControllerParticipant]:
    """The W5 participant list (no store), spec section 1: "W5 trains at R = 4"."""
    p = _v1_participants()
    del p["episodic_store"]
    return p


def _controller(**overrides) -> ThalamicController:
    kwargs = {
        "participants": _v1_participants(),
        "d_ctrl": 256,
        "depth_ctrl": 2,
        "heads_ctrl": 4,
        "n_iter": N_ITER,
        "B_read": B_READ,
        "B_kv": B_KV,
        "eta": ETA,
    }
    kwargs.update(overrides)
    return ThalamicController(**kwargs)


def _raw_summary(batch: int, participants: dict[str, ControllerParticipant]) -> dict:
    return {name: torch.randn(batch, p.summary_dim) for name, p in participants.items()}


# --------------------------------------------------------------------------- params ---


def test_param_count_matches_spec_table4():
    """Table 4: "thalamic controller | ... | 1,683,978"."""
    ctrl = _controller()
    assert sum(p.numel() for p in ctrl.parameters()) == 1_683_978


def test_text_summary_projection_is_identity_zero_params():
    """Table 5: "text summary | ... -> 256 | 0 | borrowed and frozen"."""
    ctrl = _controller()
    for name in ("language", "memory", "reasoning"):
        proj = ctrl.summary_proj[name]
        assert isinstance(proj, torch.nn.Identity)
        assert sum(p.numel() for p in proj.parameters()) == 0


def test_visual_and_store_summary_projection_param_counts():
    """Table 5: visual summary 98,560; store summary 1,280."""
    ctrl = _controller()
    visual_params = sum(p.numel() for p in ctrl.summary_proj["visual"].parameters())
    store_params = sum(p.numel() for p in ctrl.summary_proj["episodic_store"].parameters())
    assert visual_params == 98_560
    assert store_params == 1_280


def test_slot_embeddings_shape():
    """Table 5: "slot embeddings | [R+1, 256] | 1,536"."""
    ctrl = _controller()
    assert ctrl.slot_embed.shape == (6, 256)
    assert ctrl.slot_embed.numel() == 1_536


def test_heads_param_counts():
    """Table 5: ctx_head+b_head 514; A_head 1,028; halt_head 1,028."""
    ctrl = _controller()
    assert sum(p.numel() for p in ctrl.ctx_head.parameters()) == 257
    assert sum(p.numel() for p in ctrl.b_head.parameters()) == 257
    assert sum(p.numel() for p in ctrl.A_head.parameters()) == 257 * N_ITER
    assert sum(p.numel() for p in ctrl.halt_head.parameters()) == 257 * N_ITER


# ------------------------------------------------------------------- box_integerise ---


def test_box_integerise_dense_r5_tie_break():
    """Spec section 5 Table 9: dense R=5 is [52, 51, 51, 51, 51], tie on language."""
    lo = torch.tensor([8.0, 8.0, 8.0, 8.0, 8.0])
    hi = torch.tensor([96.0, 96.0, 256.0, 64.0, 256.0])
    target = torch.full((5,), B_READ / 5)
    result = box_integerise(target, lo, hi)
    assert result.tolist() == [52, 51, 51, 51, 51]


def test_box_integerise_dense_r4():
    """Spec section 5 Table 9: dense R=4 is [64, 64, 64, 64]."""
    lo = torch.tensor([10.0, 10.0, 10.0, 10.0])
    hi = torch.tensor([96.0, 96.0, 256.0, 64.0])
    target = torch.full((4,), B_READ / 4)
    result = box_integerise(target, lo, hi)
    assert result.tolist() == [64, 64, 64, 64]


def test_box_integerise_draft1_counterexample():
    """Spec section 5 Table 9: β_b = [8, 8, 222, 11, 7] / 256 -> [15, 15, 195, 17, 14]."""
    lo = torch.tensor([8.0, 8.0, 8.0, 8.0, 8.0])
    hi = torch.tensor([96.0, 96.0, 256.0, 64.0, 256.0])
    target = torch.tensor([8.0, 8.0, 222.0, 11.0, 7.0])
    result = box_integerise(target, lo, hi)
    assert result.tolist() == [15, 15, 195, 17, 14]
    assert result.sum().item() == B_READ


def test_box_integerise_cascading_cap():
    """Spec section 5 Table 9: the cascading-cap case, two rounds of capping."""
    lo = torch.tensor([8.0, 8.0, 8.0, 8.0, 8.0])
    hi = torch.tensor([96.0, 96.0, 256.0, 64.0, 256.0])
    beta = torch.tensor([0.7766, 0.0313, 0.0317, 0.108, 0.0524])
    target = beta * B_READ
    result = box_integerise(target, lo, hi)
    assert result.tolist() == [96, 27, 28, 64, 41]
    assert result.sum().item() == B_READ


def test_box_integerise_ties_go_to_earlier_participant():
    """Spec section 3 step 2: "ties on equal remainders going to the earlier
    participant in Table 1 order" -- constructed so two entries land on an identical
    fractional remainder and only one extra unit remains to distribute.
    """
    lo = torch.tensor([0.0, 0.0, 0.0])
    hi = torch.tensor([10.0, 10.0, 10.0])
    # equal shares -> equal remainders after flooring; pool of 1 leftover unit.
    target = torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3])
    result = box_integerise(target, lo, hi)
    assert result.tolist() == [1, 0, 0]
    assert result.sum().item() == 1


def test_box_integerise_random_sums_and_bounds():
    """10,000 random targets: sum matches and every entry respects its box."""
    torch.manual_seed(0)
    r = 5
    lo = torch.tensor([8.0, 8.0, 8.0, 8.0, 8.0])
    hi = torch.tensor([96.0, 96.0, 256.0, 64.0, 256.0])
    n = 10_000
    weights = torch.rand(n, r) + 1e-6
    weights = weights / weights.sum(dim=-1, keepdim=True)
    target = weights * B_READ
    result = box_integerise(target, lo, hi)
    assert result.shape == (n, r)
    assert torch.all(result.sum(dim=-1) == B_READ)
    assert torch.all(result >= lo.to(torch.int64))
    assert torch.all(result <= hi.to(torch.int64))


def test_box_integerise_adversarial_starves_one_region():
    """A weight vector that puts nearly all mass on one region still respects every
    other region's floor and no region's ceiling.
    """
    lo = torch.tensor([8.0, 8.0, 8.0, 8.0, 8.0])
    hi = torch.tensor([96.0, 96.0, 256.0, 64.0, 256.0])
    weights = torch.tensor([1e-6, 1e-6, 0.999996, 1e-6, 1e-6])
    target = weights * B_READ
    result = box_integerise(target, lo, hi)
    assert result.sum().item() == B_READ
    assert torch.all(result >= lo.to(torch.int64))
    assert torch.all(result <= hi.to(torch.int64))
    # the starved regions still get their floor, never zero.
    assert result[0].item() >= 8
    assert result[1].item() >= 8


def test_box_integerise_batched():
    lo = torch.tensor([8.0, 8.0, 8.0, 8.0, 8.0])
    hi = torch.tensor([96.0, 96.0, 256.0, 64.0, 256.0])
    target = torch.stack([torch.full((5,), B_READ / 5), torch.full((5,), B_READ / 5)])
    result = box_integerise(target, lo, hi)
    assert result.shape == (2, 5)
    assert result[0].tolist() == result[1].tolist() == [52, 51, 51, 51, 51]


# ------------------------------------------------------------------- FlooredSimplex ---


def test_floored_simplex_sums_to_one_and_respects_floor():
    simplex = FlooredSimplex()
    torch.manual_seed(1)
    logits = torch.randn(1000, 5)
    probs = simplex(logits, 0.15)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1000), atol=1e-5)
    assert torch.all(probs >= 0.15 / 5 - 1e-6)


def test_floored_simplex_adversarial_floor_holds():
    """One category driven to -inf-scale logits still gets >= eta/N."""
    simplex = FlooredSimplex()
    logits = torch.tensor([[-1e6, 0.0, 0.0, 0.0, 0.0]])
    probs = simplex(logits, 0.15)
    assert probs[0, 0].item() >= 0.15 / 5 - 1e-9
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]), atol=1e-5)


def test_floored_simplex_eta_zero_is_plain_softmax():
    simplex = FlooredSimplex()
    logits = torch.randn(4, 3)
    probs = simplex(logits, 0.0)
    assert torch.allclose(probs, torch.softmax(logits, dim=-1), atol=1e-6)


# --------------------------------------------------------------------- admission A ---


def test_admit_forces_admission_when_threshold_admits_nowhere():
    """Spec section 3 step 3: "forced so every declared region is admitted at least
    once at its argmax iteration."
    """
    # every logit far below the sigmoid>0.5 threshold (i.e. very negative).
    A_logits = torch.full((1, 3, N_ITER), -10.0)
    A_logits[0, 1, 2] = -1.0  # region 1's least-negative (argmax) iteration is 2
    admitted = _admit(A_logits)
    assert admitted.sum(dim=-1).ge(1).all()
    assert admitted[0, 1].tolist() == [False, False, True, False]


def test_admit_adversarial_starve_one_region_among_many():
    """10,000 random rows plus a region built to never cross the threshold: every
    region is admitted at least once regardless.
    """
    torch.manual_seed(2)
    n_regions = 5
    A_logits = torch.randn(10_000, n_regions, N_ITER)
    A_logits[:, 2, :] = -50.0  # region 2 starved in every row
    admitted = _admit(A_logits)
    assert torch.all(admitted.sum(dim=-1) >= 1)


def test_admit_no_forcing_when_threshold_already_admits():
    A_logits = torch.zeros(1, 1, N_ITER)
    A_logits[0, 0, 1] = 10.0  # sigmoid(10) > 0.5, no forcing needed
    admitted = _admit(A_logits)
    assert admitted[0, 0].tolist() == [False, True, False, False]


def test_admit_random_10000_always_covers_every_region():
    torch.manual_seed(3)
    A_logits = torch.randn(10_000, 5, N_ITER) * 5.0
    admitted = _admit(A_logits)
    assert torch.all(admitted.sum(dim=-1) >= 1)


# ------------------------------------------------------------------------- halt_at ---


def test_halt_at_bounds_random_logits():
    torch.manual_seed(4)
    halt_logits = torch.randn(1000, N_ITER)
    halt_at, active = _halt(halt_logits, N_ITER, 1e-3)
    assert torch.all(halt_at >= 1)
    assert torch.all(halt_at <= N_ITER)
    assert active.shape == (1000, N_ITER)
    assert torch.equal(active.sum(dim=-1), halt_at)


def test_halt_at_can_be_less_than_n_iter():
    """Spec section 5 Table 9: "an input exists whose halt_at < n_iter"."""
    halt_logits = torch.tensor([[20.0, 0.0, 0.0, 0.0]])  # sigmoid(20) ~ 1
    halt_at, active = _halt(halt_logits, N_ITER, 1e-3)
    assert halt_at.item() == 1
    assert halt_at.item() < N_ITER
    assert active[0].tolist() == [True, False, False, False]


def test_halt_at_falls_back_to_n_iter_when_never_reached():
    halt_logits = torch.full((1, N_ITER), -20.0)  # sigmoid ~ 0, cumsum never reaches
    halt_at, active = _halt(halt_logits, N_ITER, 1e-3)
    assert halt_at.item() == N_ITER
    assert active[0].tolist() == [True, True, True, True]


def test_halt_at_single_iteration_ceiling():
    halt_logits = torch.randn(50, 1)
    halt_at, active = _halt(halt_logits, 1, 1e-3)
    assert torch.all(halt_at == 1)
    assert active.shape == (50, 1)


# --------------------------------------------------------------------- full forward ---


def test_forward_shapes():
    ctrl = _controller()
    raw = _raw_summary(3, _v1_participants())
    out = ctrl(raw)
    assert out.ctx.shape == (3, 4)
    assert out.b.shape == (3, 5)
    assert out.A.shape == (3, N_ITER, 5)
    assert out.halt_at.shape == (3,)
    assert out.active.shape == (3, N_ITER)
    assert out.halt_logits.shape == (3, N_ITER)
    assert out.ctx_logits.shape == (3, 4)
    assert out.b_logits.shape == (3, 5)
    assert out.A_logits.shape == (3, 5, N_ITER)
    assert out.ctx.dtype == torch.int64
    assert out.b.dtype == torch.int64
    assert out.A.dtype == torch.bool
    assert out.halt_at.dtype == torch.int64
    assert out.active.dtype == torch.bool


def test_forward_b_sums_to_b_read_and_respects_box():
    ctrl = _controller()
    torch.manual_seed(5)
    raw = _raw_summary(16, _v1_participants())
    out = ctrl(raw)
    assert torch.all(out.b.sum(dim=-1) == B_READ)
    lo_hi = _v1_participants()
    for i, name in enumerate(ctrl.participant_names):
        p = lo_hi[name]
        assert torch.all(out.b[:, i] <= p.token_budget_max)


def test_forward_ctx_respects_bounds():
    ctrl = _controller()
    torch.manual_seed(6)
    raw = _raw_summary(16, _v1_participants())
    out = ctrl(raw)
    ctx_participants = [
        n for n in ctrl.participant_names if _v1_participants()[n].ctx_min is not None
    ]
    for i, name in enumerate(ctx_participants):
        p = _v1_participants()[name]
        assert torch.all(out.ctx[:, i] >= p.ctx_min)
        assert torch.all(out.ctx[:, i] <= p.ctx_max)


def test_forward_every_region_admitted_at_least_once():
    ctrl = _controller()
    torch.manual_seed(7)
    raw = _raw_summary(64, _v1_participants())
    out = ctrl(raw)
    assert torch.all(out.A.sum(dim=1) >= 1)


def test_forward_halt_at_bounds():
    ctrl = _controller()
    torch.manual_seed(8)
    raw = _raw_summary(64, _v1_participants())
    out = ctrl(raw)
    assert torch.all(out.halt_at >= 1)
    assert torch.all(out.halt_at <= N_ITER)


def test_forward_r4_no_store():
    """W5 trains at R = 4 without the store (spec section 1)."""
    ctrl = _controller(participants=_r4_participants(), B_read=256)
    raw = _raw_summary(2, _r4_participants())
    out = ctrl(raw)
    assert out.b.shape == (2, 4)
    assert out.ctx.shape == (2, 4)
    assert out.A.shape == (2, N_ITER, 4)
    assert torch.all(out.b.sum(dim=-1) == 256)


def test_forward_bad_raw_summary_keys_raises():
    ctrl = _controller()
    raw = _raw_summary(2, _v1_participants())
    del raw["visual"]
    with pytest.raises(ControllerConfigError):
        ctrl(raw)


# --------------------------------------------------------------- determinism (seed) ---


def test_construction_and_forward_deterministic_under_fixed_seed():
    """Spec section 5 Table 9's "all" row: construction and forward under
    `torch.manual_seed(0)` are bitwise identical across two runs on CPU.
    """
    torch.manual_seed(0)
    ctrl1 = _controller()
    raw1 = _raw_summary(4, _v1_participants())
    out1 = ctrl1(raw1)

    torch.manual_seed(0)
    ctrl2 = _controller()
    raw2 = _raw_summary(4, _v1_participants())
    out2 = ctrl2(raw2)

    assert torch.equal(out1.ctx, out2.ctx)
    assert torch.equal(out1.b, out2.b)
    assert torch.equal(out1.A, out2.A)
    assert torch.equal(out1.halt_at, out2.halt_at)
    assert torch.equal(out1.active, out2.active)
    assert torch.allclose(out1.halt_logits, out2.halt_logits, atol=0.0, rtol=0.0)


# ------------------------------------------------------------------- config errors ---


def test_empty_participants_raises():
    with pytest.raises(ControllerConfigError):
        ThalamicController(participants={})


def test_heads_not_dividing_d_ctrl_raises():
    with pytest.raises(ControllerConfigError):
        _controller(d_ctrl=250, heads_ctrl=4)


def test_eta_out_of_range_raises():
    with pytest.raises(ControllerConfigError):
        _controller(eta=1.5)
    with pytest.raises(ControllerConfigError):
        _controller(eta=-0.1)


def test_ctx_min_unreachable_within_floor_share_raises():
    """Spec section 3 step 2: construction refuses a config where
    eta/R_ctx * B_kv < c_r * ctx_min for any r.
    """
    participants = _v1_participants()
    with pytest.raises(ControllerConfigError):
        _controller(participants=participants, B_kv=1_000, eta=0.01)


def test_b_box_floor_sum_exceeds_b_read_raises():
    participants = _v1_participants()
    with pytest.raises(ControllerConfigError):
        _controller(participants=participants, B_read=10)


def test_b_box_ceiling_sum_falls_short_raises():
    # token_budget_max pinned to token_budget_min for every participant, so the
    # declared box stays valid (min <= max) while the ceiling sum (12) falls far
    # short of B_read.
    participants = {
        name: ControllerParticipant(
            summary_dim=p.summary_dim,
            ctx_min=p.ctx_min,
            ctx_max=p.ctx_max,
            kv_bytes_per_token=p.kv_bytes_per_token,
            token_budget_min=p.token_budget_min,
            token_budget_max=p.token_budget_min,
        )
        for name, p in _v1_participants().items()
    }
    with pytest.raises(ControllerConfigError):
        _controller(participants=participants, B_read=B_READ)


def test_controller_participant_bad_ctx_nullity_raises():
    with pytest.raises(ControllerConfigError):
        ControllerParticipant(
            summary_dim=256,
            ctx_min=8,
            ctx_max=None,
            kv_bytes_per_token=4096,
            token_budget_min=1,
            token_budget_max=96,
        )


def test_controller_participant_ctx_min_gt_max_raises():
    with pytest.raises(ControllerConfigError):
        ControllerParticipant(
            summary_dim=256,
            ctx_min=100,
            ctx_max=8,
            kv_bytes_per_token=4096,
            token_budget_min=1,
            token_budget_max=96,
        )


def test_controller_participant_budget_min_gt_max_raises():
    with pytest.raises(ControllerConfigError):
        ControllerParticipant(
            summary_dim=256,
            ctx_min=8,
            ctx_max=96,
            kv_bytes_per_token=4096,
            token_budget_min=100,
            token_budget_max=10,
        )
