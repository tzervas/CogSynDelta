"""Lane IC-4 -- `cogsyndelta.interconnect.readout`: shapes, masks/gates, determinism.

`docs/design/INTERCONNECT-MODULE-SPEC.md` section 5 Table 9, `readout.py` row: "`f`
shape; two distinct `z` give distinct `f`; cosine scores inside `[-1, 1]`; the `NULL`
candidate sits at index 0 and receives gradient; probe outputs at each `pooled_dim`;
the `z_affect` gate is zero and adds no parameter." The Table 9 "all" row additionally
requires "construction and forward under `torch.manual_seed(0)` are bitwise identical
across two runs on CPU," reproduced here as `test_determinism_under_fixed_seed`.

`readout.py` has no mask-shaped inputs of its own (`z` is the dense `[B, L, D_w]`
workspace state with no padding, per Table 3) and no guard rows in Table 8 that fire
inside this file (G36 fires in `gates.py` over a receipt; see the module docstring),
so this file carries no `test_guards_can_fail.py`-style cases.

Every test uses `D_w = 512`, the v1 workspace width, so the parameter-count assertions
below double as a direct check against section 2.3 Table 4's arithmetic.
"""

from __future__ import annotations

import pytest
import torch

from cogsyndelta.interconnect.readout import FrontalReadout, NullCandidate, RankHead, UnifyProbes

pytestmark = pytest.mark.cpu

_D_W = 512
_L = 64
_K = 32
_POOLED_DIMS = {"language": 256, "memory": 256, "reasoning": 256, "visual": 384}


def _z(batch: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, _L, _D_W)


def _content_candidates(batch: int, seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, _K - 1, _D_W)


# ---------------------------------------------------------------------------
# FrontalReadout
# ---------------------------------------------------------------------------


def test_frontal_readout_f_shape() -> None:
    """`f = FrontalReadout(z_N)` is `[B, D_w]` (spec section 3 step 12)."""
    model = FrontalReadout(_D_W)
    f = model(_z(batch=5))
    assert f.shape == (5, _D_W)


def test_frontal_readout_param_count_matches_table_4() -> None:
    """3,150,848 params at `D_w=512`, `mlp_ratio=4` (spec section 2.3 Table 4)."""
    model = FrontalReadout(_D_W, mlp_ratio=4)
    assert sum(p.numel() for p in model.parameters()) == 3_150_848


def test_frontal_readout_distinct_z_give_distinct_f() -> None:
    """Two distinct `z` must not collapse to the same `f`."""
    model = FrontalReadout(_D_W).eval()
    f1 = model(_z(batch=2, seed=10))
    f2 = model(_z(batch=2, seed=11))
    assert not torch.allclose(f1, f2)


def test_frontal_readout_z_affect_gate_is_zero_and_adds_no_parameter() -> None:
    """The declared `z_affect` seam changes neither `f` nor the parameter count."""
    model = FrontalReadout(_D_W).eval()
    n_params_before = sum(p.numel() for p in model.parameters())
    z = _z(batch=3)

    f_without = model(z)
    f_with_zero = model(z, z_affect=torch.zeros(3, _D_W))
    f_with_arbitrary = model(z, z_affect=torch.randn(3, _D_W) * 1e6)

    assert sum(p.numel() for p in model.parameters()) == n_params_before
    assert torch.equal(f_without, f_with_zero)
    assert torch.equal(f_without, f_with_arbitrary)
    assert not any("affect" in name for name, _ in model.named_parameters())


def test_frontal_readout_z_affect_wrong_width_raises() -> None:
    """A `z_affect` whose last dim is not `D_w` is refused, not silently broadcast."""
    model = FrontalReadout(_D_W)
    with pytest.raises(ValueError, match="D_w"):
        model(_z(batch=2), z_affect=torch.randn(2, _D_W - 1))


# ---------------------------------------------------------------------------
# NullCandidate
# ---------------------------------------------------------------------------


def test_null_candidate_expand_shape_and_sharing() -> None:
    """`expand` broadcasts one learned vector; every batch row is identical."""
    null = NullCandidate(_D_W)
    expanded = null.expand(batch=4)
    assert expanded.shape == (4, 1, _D_W)
    for i in range(1, 4):
        assert torch.equal(expanded[0], expanded[i])


def test_null_candidate_param_count() -> None:
    """One learned vector of 512 (spec section 2.3 Table 4)."""
    null = NullCandidate(_D_W)
    assert sum(p.numel() for p in null.parameters()) == 512


# ---------------------------------------------------------------------------
# RankHead
# ---------------------------------------------------------------------------


def test_rank_head_scores_shape() -> None:
    """Scores are `[B, k]`, `NULL` at index 0 (spec Table 3, "rank scores")."""
    head = RankHead(_D_W, k=_K, temperature=1.0)
    f = FrontalReadout(_D_W)(_z(batch=3))
    scores = head(f, _content_candidates(batch=3))
    assert scores.shape == (3, _K)


def test_rank_head_param_count_matches_table_4() -> None:
    """262,656 (the shared linear) + 512 (`NULL`) = 263,168 (spec section 2.3 Table 4)."""
    head = RankHead(_D_W, k=_K, temperature=1.0)
    assert sum(p.numel() for p in head.parameters()) == 262_656 + 512


def test_rank_head_cosine_scores_inside_unit_interval() -> None:
    """At `temperature=1.0` the returned scores are raw cosine similarities."""
    head = RankHead(_D_W, k=_K, temperature=1.0)
    f = FrontalReadout(_D_W)(_z(batch=6))
    scores = head(f, _content_candidates(batch=6))
    assert torch.all(scores >= -1.0 - 1e-5)
    assert torch.all(scores <= 1.0 + 1e-5)


def test_rank_head_null_at_index_0_receives_gradient() -> None:
    """`NullCandidate`'s embedding gets a non-zero gradient through `RankHead`."""
    head = RankHead(_D_W, k=_K, temperature=0.5)
    f = FrontalReadout(_D_W)(_z(batch=4)).detach().requires_grad_(True)
    scores = head(f, _content_candidates(batch=4))
    scores.sum().backward()
    assert head.null.embedding.grad is not None
    assert torch.any(head.null.embedding.grad != 0)


def test_rank_head_null_is_index_0_of_the_assembled_candidates() -> None:
    """Zeroing the content candidates isolates `NULL`'s own score at index 0."""
    head = RankHead(_D_W, k=4, temperature=1.0)
    f = head.proj.weight.new_zeros(1, _D_W)
    f[0, 0] = 1.0
    content = f.new_zeros(1, 3, _D_W)
    scores = head(f, content)
    null_only_score = torch.nn.functional.cosine_similarity(
        head.proj(f), head.proj(head.null.expand(1).squeeze(1)), dim=-1
    )
    assert torch.allclose(scores[:, 0], null_only_score)


def test_rank_head_rejects_non_positive_temperature() -> None:
    """`temperature` has no default and must be positive (spec section 4)."""
    with pytest.raises(ValueError, match="temperature"):
        RankHead(_D_W, k=_K, temperature=0.0)


def test_rank_head_rejects_wrong_candidate_count() -> None:
    """`content_candidates` must be exactly `k - 1` wide."""
    head = RankHead(_D_W, k=_K, temperature=1.0)
    f = FrontalReadout(_D_W)(_z(batch=2))
    with pytest.raises(ValueError, match="k - 1"):
        head(f, _content_candidates(batch=2)[:, :-1, :])


# ---------------------------------------------------------------------------
# UnifyProbes
# ---------------------------------------------------------------------------


def test_unify_probes_output_shape_per_pooled_dim() -> None:
    """One `[B, pooled_dim_r]` output per constructor-time region name."""
    probes = UnifyProbes(_D_W, _POOLED_DIMS)
    f = FrontalReadout(_D_W)(_z(batch=3))
    out = probes(f)
    assert set(out) == set(_POOLED_DIMS)
    for name, dim in _POOLED_DIMS.items():
        assert out[name].shape == (3, dim)


def test_unify_probes_param_count_matches_table_4() -> None:
    """590,976 params for the four `R_ctx = 4` encoding participants (Table 4)."""
    probes = UnifyProbes(_D_W, _POOLED_DIMS)
    assert sum(p.numel() for p in probes.parameters()) == 590_976


def test_unify_probes_rejects_empty_pooled_dims() -> None:
    """An empty region map is refused rather than silently building nothing."""
    with pytest.raises(ValueError, match="pooled_dims"):
        UnifyProbes(_D_W, {})


# ---------------------------------------------------------------------------
# Determinism (Table 9 "all" row)
# ---------------------------------------------------------------------------


def test_determinism_under_fixed_seed() -> None:
    """Construction and forward under `torch.manual_seed(0)` are bitwise identical."""

    def _run() -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        torch.manual_seed(0)
        frontal = FrontalReadout(_D_W)
        head = RankHead(_D_W, k=_K, temperature=1.0)
        probes = UnifyProbes(_D_W, _POOLED_DIMS)
        z = torch.randn(2, _L, _D_W)
        content = torch.randn(2, _K - 1, _D_W)
        f = frontal(z)
        scores = head(f, content)
        probe_out = probes(f)
        return f, scores, probe_out

    f1, scores1, probes1 = _run()
    f2, scores2, probes2 = _run()

    assert torch.equal(f1, f2)
    assert torch.equal(scores1, scores2)
    for name in probes1:
        assert torch.equal(probes1[name], probes2[name])
