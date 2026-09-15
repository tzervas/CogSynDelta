"""IC-2 -- `RegionAdapter`, `TopKSelect`, `ConditioningPrefix`, and this file's share of G31.

`docs/design/INTERCONNECT-MODULE-SPEC.md` section 5 Table 9's `adapters.py` row: "output
shapes for `d_r in {256, 384}`; top-k returns exactly `b_r` positions that are both real
and inside the budget mask, never a masked one, ties by position; the straight-through
gate passes gradient to the scorer; prefix shape `[B, 8, d_r]`; a uniform query
reproduces the mean pool to 1e-6." Plus the general "all" row: construction and forward
under a fixed seed are bitwise identical across two runs on CPU. Plus section 4 Table 8's
G31 row and section 5's guard-test instruction: "Every guard test constructs the
condition the guard exists to catch and asserts that it fires, following
`tests/test_guards_can_fail.py`."

This is a lane-owned unit-test file (spec section 7, lane IC-2); `tests/interconnect/`
has no `__init__.py` or `conftest.py` yet (both are lane IC-8's), so every fixture this
file needs is built locally.
"""

from __future__ import annotations

import pytest
import torch

from cogsyndelta.interconnect.adapters import (
    ConditioningPrefix,
    RegionAdapter,
    TopKSelect,
    assert_float_tract,
)

pytestmark = pytest.mark.cpu

D_W = 512


# ---------------------------------------------------------------------------
# assert_float_tract / G31
# ---------------------------------------------------------------------------


def test_assert_float_tract_accepts_float() -> None:
    assert_float_tract(torch.randn(2, 3), "t")  # must not raise


def test_g31_fires_on_integer_h_r_into_region_adapter() -> None:
    """G31's constructed violation: a region wrapper returning ids instead of latents."""
    adapter = RegionAdapter(token_dim=256, D_w=D_W)
    ids = torch.randint(0, 1000, (2, 4, 256), dtype=torch.long)
    with pytest.raises(TypeError, match="G31"):
        adapter(ids)


def test_g31_fires_on_integer_h_r_into_topk_select() -> None:
    scorer = TopKSelect(token_dim=256)
    ids = torch.randint(0, 1000, (2, 4, 256), dtype=torch.long)
    mask = torch.ones(2, 4, dtype=torch.bool)
    with pytest.raises(TypeError, match="G31"):
        scorer(ids, mask, b_r=2)


def test_g31_fires_on_integer_z_into_conditioning_prefix() -> None:
    prefix = ConditioningPrefix(D_w=D_W, token_dim=256, n_cond=8)
    ids = torch.randint(0, 1000, (2, 16, D_W), dtype=torch.long)
    with pytest.raises(TypeError, match="G31"):
        prefix(ids)
    with pytest.raises(TypeError, match="G31"):
        prefix.pool(ids)


# ---------------------------------------------------------------------------
# RegionAdapter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token_dim", [256, 384])
def test_region_adapter_output_shape(token_dim: int) -> None:
    adapter = RegionAdapter(token_dim=token_dim, D_w=D_W)
    h = torch.randn(3, 5, token_dim)
    out = adapter(h)
    assert out.shape == (3, 5, D_W)
    assert torch.is_floating_point(out)


def test_region_adapter_deterministic_under_fixed_seed() -> None:
    torch.manual_seed(0)
    a1 = RegionAdapter(token_dim=256, D_w=D_W)
    h1 = torch.randn(2, 4, 256)
    out1 = a1(h1)

    torch.manual_seed(0)
    a2 = RegionAdapter(token_dim=256, D_w=D_W)
    h2 = torch.randn(2, 4, 256)
    out2 = a2(h2)

    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# TopKSelect
# ---------------------------------------------------------------------------


def test_topk_select_returns_exactly_b_r_positions() -> None:
    scorer = TopKSelect(token_dim=256)
    h = torch.randn(4, 10, 256)
    mask = torch.ones(4, 10, dtype=torch.bool)
    selected_h, selected_mask, indices = scorer(h, mask, b_r=3)
    assert selected_h.shape == (4, 3, 256)
    assert selected_mask.shape == (4, 3)
    assert indices.shape == (4, 3)


def test_topk_select_never_returns_a_masked_position_when_enough_real_ones_exist() -> None:
    torch.manual_seed(1)
    scorer = TopKSelect(token_dim=8)
    h = torch.randn(2, 10, 8)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, :6] = True  # only positions 0..5 are real/inside budget
    _selected_h, selected_mask, indices = scorer(h, mask, b_r=4)
    assert selected_mask.all()
    assert (indices < 6).all()


def test_topk_select_pads_with_masked_positions_when_too_few_real_ones_exist() -> None:
    scorer = TopKSelect(token_dim=8)
    h = torch.randn(1, 10, 8)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[:, :2] = True  # only 2 real positions, but we ask for 5
    _selected_h, selected_mask, _indices = scorer(h, mask, b_r=5)
    assert selected_mask.sum().item() == 2
    assert (~selected_mask[:, 2:]).all()


def test_topk_select_keeps_the_highest_scoring_real_positions() -> None:
    scorer = TopKSelect(token_dim=1)
    # scorer is Linear(1, 1); make its weight positive and bias zero so score == value,
    # a direct read of which positions get kept.
    with torch.no_grad():
        scorer.scorer.weight.fill_(1.0)
        scorer.scorer.bias.zero_()
    h = torch.tensor([[[0.0], [5.0], [1.0], [9.0], [2.0]]])  # [1, 5, 1]
    mask = torch.ones(1, 5, dtype=torch.bool)
    _selected_h, _selected_mask, indices = scorer(h, mask, b_r=2)
    assert set(indices[0].tolist()) == {1, 3}  # positions with values 5.0 and 9.0


def test_topk_select_breaks_ties_by_earlier_position() -> None:
    scorer = TopKSelect(token_dim=1)
    with torch.no_grad():
        scorer.scorer.weight.fill_(1.0)
        scorer.scorer.bias.zero_()
    h = torch.tensor([[[3.0], [3.0], [3.0], [3.0]]])  # every position ties
    mask = torch.ones(1, 4, dtype=torch.bool)
    _selected_h, _selected_mask, indices = scorer(h, mask, b_r=2)
    assert indices[0].tolist() == [0, 1]


def test_topk_select_straight_through_gate_passes_gradient_to_the_scorer() -> None:
    scorer = TopKSelect(token_dim=8)
    h = torch.randn(2, 10, 8, requires_grad=False)
    mask = torch.ones(2, 10, dtype=torch.bool)
    selected_h, _selected_mask, _indices = scorer(h, mask, b_r=4)
    loss = selected_h.sum()
    loss.backward()
    assert scorer.scorer.weight.grad is not None
    assert torch.any(scorer.scorer.weight.grad != 0)


def test_topk_select_forward_value_is_unaffected_by_the_gate_to_high_precision() -> None:
    """The gate is exactly 1 in the forward pass -- STE, not a soft rescaling."""
    scorer = TopKSelect(token_dim=8)
    h = torch.randn(2, 10, 8)
    mask = torch.ones(2, 10, dtype=torch.bool)
    selected_h, _selected_mask, indices = scorer(h, mask, b_r=4)
    gathered = torch.gather(h, 1, indices.unsqueeze(-1).expand(-1, -1, 8))
    torch.testing.assert_close(selected_h, gathered)


def test_topk_select_rejects_b_r_out_of_range() -> None:
    scorer = TopKSelect(token_dim=8)
    h = torch.randn(1, 5, 8)
    mask = torch.ones(1, 5, dtype=torch.bool)
    with pytest.raises(ValueError):
        scorer(h, mask, b_r=6)
    with pytest.raises(ValueError):
        scorer(h, mask, b_r=-1)


def test_topk_select_deterministic_under_fixed_seed() -> None:
    torch.manual_seed(0)
    s1 = TopKSelect(token_dim=8)
    h1 = torch.randn(2, 10, 8)
    mask1 = torch.ones(2, 10, dtype=torch.bool)
    out1 = s1(h1, mask1, b_r=4)[0]

    torch.manual_seed(0)
    s2 = TopKSelect(token_dim=8)
    h2 = torch.randn(2, 10, 8)
    mask2 = torch.ones(2, 10, dtype=torch.bool)
    out2 = s2(h2, mask2, b_r=4)[0]

    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# ConditioningPrefix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token_dim", [256, 384])
def test_conditioning_prefix_output_shape(token_dim: int) -> None:
    prefix = ConditioningPrefix(D_w=D_W, token_dim=token_dim, n_cond=8)
    z = torch.randn(3, 64, D_W)
    cond = prefix(z)
    assert cond.shape == (3, 8, token_dim)
    assert torch.is_floating_point(cond)


def test_conditioning_prefix_uniform_query_reproduces_the_mean_pool() -> None:
    """`self.query` is initialised to zero, so `pool_r` starts as a plain mean pool."""
    prefix = ConditioningPrefix(D_w=D_W, token_dim=256, n_cond=8)
    z = torch.randn(4, 64, D_W)
    pooled = prefix.pool(z)
    expected = z.mean(dim=1)
    torch.testing.assert_close(pooled, expected, atol=1e-6, rtol=1e-6)


def test_conditioning_prefix_nonzero_query_departs_from_the_mean_pool() -> None:
    prefix = ConditioningPrefix(D_w=D_W, token_dim=256, n_cond=8)
    with torch.no_grad():
        prefix.query.normal_()
    z = torch.randn(4, 64, D_W)
    pooled = prefix.pool(z)
    expected = z.mean(dim=1)
    assert not torch.allclose(pooled, expected, atol=1e-3)


def test_conditioning_prefix_deterministic_under_fixed_seed() -> None:
    torch.manual_seed(0)
    p1 = ConditioningPrefix(D_w=D_W, token_dim=256, n_cond=8)
    z1 = torch.randn(2, 64, D_W)
    out1 = p1(z1)

    torch.manual_seed(0)
    p2 = ConditioningPrefix(D_w=D_W, token_dim=256, n_cond=8)
    z2 = torch.randn(2, 64, D_W)
    out2 = p2(z2)

    assert torch.equal(out1, out2)
