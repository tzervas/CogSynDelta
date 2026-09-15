"""IC-3 -- `LatentBank`, `WorkspaceBlock`, `Workspace`.

`docs/design/INTERCONNECT-MODULE-SPEC.md` section 5 Table 9's `workspace.py` row:
"`z` keeps its shape across iterations; `a[b, i, :]` sums to 1 over admitted regions on
active iterations and is zero otherwise; SDPA and explicit-softmax paths agree to 1e-4;
the severance mask gives zero gradient through the masked slots." Plus the general
"all" row: construction and forward under a fixed seed are bitwise identical across two
runs on CPU.

This is a lane-owned unit-test file (spec section 7, lane IC-3); `tests/interconnect/`
has no `__init__.py` or `conftest.py` yet (both are lane IC-8's), so every fixture this
file needs is built locally.
"""

from __future__ import annotations

import pytest
import torch

from cogsyndelta.interconnect.workspace import LatentBank, Workspace, WorkspaceBlock

pytestmark = pytest.mark.cpu


def _toy_bank(
    batch_size: int,
    n_iter: int,
    n_slots: int,
    d_w: int,
    n_regions: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a toy `(bank_k, bank_v, key_mask, slot_region)` fixture.

    Slots are laid out in fixed region order, `n_slots // n_regions` per region (any
    remainder goes to region 0), all attendable -- callers mask specific slots or
    regions out on top of this when a test needs it.
    """
    gen = torch.Generator().manual_seed(seed)
    bank_k = torch.randn(batch_size, n_iter, n_slots, d_w, generator=gen)
    bank_v = torch.randn(batch_size, n_iter, n_slots, d_w, generator=gen)
    key_mask = torch.ones(batch_size, n_iter, n_slots, dtype=torch.bool)
    per_region = n_slots // n_regions
    region_row = torch.full((n_slots,), n_regions - 1, dtype=torch.long)
    for r in range(n_regions):
        region_row[r * per_region : (r + 1) * per_region] = r
    slot_region = region_row.view(1, 1, n_slots).expand(batch_size, n_iter, n_slots).clone()
    return bank_k, bank_v, key_mask, slot_region


class TestLatentBank:
    def test_shape(self) -> None:
        """`LatentBank(L, D_w)(B)` returns `[B, L, D_w]` (spec Table 3, `z`)."""
        bank = LatentBank(L=8, D_w=16)
        z0 = bank(batch_size=3)
        assert z0.shape == (3, 8, 16)

    def test_shared_across_batch(self) -> None:
        """Every batch item starts from the identical learned latents."""
        bank = LatentBank(L=4, D_w=8)
        z0 = bank(batch_size=5)
        for i in range(1, 5):
            assert torch.equal(z0[0], z0[i])


class TestWorkspaceBlockAttentionPaths:
    """Table 9: "SDPA and explicit-softmax paths agree to 1e-4"."""

    def _block_and_inputs(
        self, seed: int = 0
    ) -> tuple[WorkspaceBlock, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        block = WorkspaceBlock(D_w=32, heads=4, mlp_ratio=4.0).eval()
        z = torch.randn(2, 6, 32)
        bank_k = torch.randn(2, 10, 32)
        bank_v = torch.randn(2, 10, 32)
        key_mask = torch.ones(2, 10, dtype=torch.bool)
        key_mask[1, 7:] = False
        return block, z, bank_k, bank_v, key_mask

    def test_sdpa_and_explicit_softmax_agree(self) -> None:
        block, z, bank_k, bank_v, key_mask = self._block_and_inputs()
        z_sdpa, weights_sdpa = block(z, bank_k, bank_v, key_mask, return_attention=False)
        z_explicit, weights_explicit = block(z, bank_k, bank_v, key_mask, return_attention=True)
        assert weights_sdpa is None
        assert weights_explicit is not None
        assert torch.allclose(z_sdpa, z_explicit, atol=1e-4)

    def test_explicit_weights_shape_and_row_sums(self) -> None:
        block, z, bank_k, bank_v, key_mask = self._block_and_inputs()
        _, weights = block(z, bank_k, bank_v, key_mask, return_attention=True)
        assert weights.shape == (2, 4, 6, 10)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_masked_keys_receive_zero_weight(self) -> None:
        block, z, bank_k, bank_v, key_mask = self._block_and_inputs()
        _, weights = block(z, bank_k, bank_v, key_mask, return_attention=True)
        assert torch.equal(weights[1, :, :, 7:], torch.zeros_like(weights[1, :, :, 7:]))

    def test_no_mask_matches_all_true_mask(self) -> None:
        block, z, bank_k, bank_v, _ = self._block_and_inputs()
        all_true = torch.ones(2, 10, dtype=torch.bool)
        z_none, _ = block(z, bank_k, bank_v, None, return_attention=True)
        z_masked, _ = block(z, bank_k, bank_v, all_true, return_attention=True)
        assert torch.allclose(z_none, z_masked, atol=1e-6)

    def test_heads_must_divide_dw(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            WorkspaceBlock(D_w=10, heads=3, mlp_ratio=4.0)


class TestWorkspaceShape:
    """Table 9: "`z` keeps its shape across iterations"."""

    @pytest.mark.parametrize("n_iter", [1, 2, 4])
    def test_z_and_history_shapes(self, n_iter: int) -> None:
        torch.manual_seed(0)
        ws = Workspace(D_w=16, L=5, n_iter=n_iter, heads=4, mlp_ratio=4.0).eval()
        bank_k, bank_v, key_mask, slot_region = _toy_bank(
            batch_size=3, n_iter=n_iter, n_slots=12, d_w=16, n_regions=3
        )
        z, a, z_history = ws(bank_k, bank_v, key_mask, slot_region, n_regions=3)
        assert z.shape == (3, 5, 16)
        assert a.shape == (3, n_iter, 3)
        assert len(z_history) == n_iter
        for z_i in z_history:
            assert z_i.shape == (3, 5, 16)
        assert torch.equal(z, z_history[-1])

    def test_shape_stable_under_partial_halting(self) -> None:
        """An item halted early still yields full-shape `z` and history entries."""
        torch.manual_seed(0)
        ws = Workspace(D_w=16, L=5, n_iter=4, heads=4, mlp_ratio=4.0).eval()
        bank_k, bank_v, key_mask, slot_region = _toy_bank(
            batch_size=2, n_iter=4, n_slots=12, d_w=16, n_regions=3
        )
        active = torch.tensor([[True, False, False, False], [True, True, True, True]])
        z, a, z_history = ws(bank_k, bank_v, key_mask, slot_region, n_regions=3, active=active)
        assert z.shape == (2, 5, 16)
        assert a.shape == (2, 4, 3)
        assert all(z_i.shape == (2, 5, 16) for z_i in z_history)


class TestAttentionMassExport:
    """Table 9: "`a[b, i, :]` sums to 1 over admitted regions on active iterations and is
    zero otherwise"."""

    def test_active_iterations_sum_to_one(self) -> None:
        torch.manual_seed(0)
        ws = Workspace(D_w=16, L=5, n_iter=3, heads=4, mlp_ratio=4.0).eval()
        bank_k, bank_v, key_mask, slot_region = _toy_bank(
            batch_size=4, n_iter=3, n_slots=12, d_w=16, n_regions=3
        )
        _, a, _ = ws(bank_k, bank_v, key_mask, slot_region, n_regions=3)
        assert torch.allclose(a.sum(dim=-1), torch.ones(4, 3), atol=1e-4)

    def test_inactive_iterations_are_zero(self) -> None:
        torch.manual_seed(0)
        ws = Workspace(D_w=16, L=5, n_iter=4, heads=4, mlp_ratio=4.0).eval()
        bank_k, bank_v, key_mask, slot_region = _toy_bank(
            batch_size=2, n_iter=4, n_slots=12, d_w=16, n_regions=3
        )
        active = torch.tensor([[True, True, False, False], [True, True, True, False]])
        _, a, _ = ws(bank_k, bank_v, key_mask, slot_region, n_regions=3, active=active)
        assert torch.equal(a[0, 2:], torch.zeros(2, 3))
        assert torch.equal(a[1, 3:], torch.zeros(1, 3))
        assert torch.allclose(a[0, :2].sum(dim=-1), torch.ones(2), atol=1e-4)
        assert torch.allclose(a[1, :3].sum(dim=-1), torch.ones(3), atol=1e-4)

    def test_default_active_is_all_true_dense_schedule(self) -> None:
        """No `active` argument means the phase-A dense schedule: every iteration active."""
        torch.manual_seed(0)
        ws = Workspace(D_w=16, L=5, n_iter=3, heads=4, mlp_ratio=4.0).eval()
        bank_k, bank_v, key_mask, slot_region = _toy_bank(
            batch_size=2, n_iter=3, n_slots=12, d_w=16, n_regions=3
        )
        _, a, _ = ws(bank_k, bank_v, key_mask, slot_region, n_regions=3)
        assert (a.sum(dim=-1) > 0).all()

    def test_unattributable_slots_excluded(self) -> None:
        """`slot_region == -1` slots contribute no mass to any region."""
        torch.manual_seed(0)
        ws = Workspace(D_w=16, L=5, n_iter=1, heads=4, mlp_ratio=4.0).eval()
        bank_k, bank_v, key_mask, slot_region = _toy_bank(
            batch_size=2, n_iter=1, n_slots=12, d_w=16, n_regions=3
        )
        # Mark half the slots as empty (unattributable) and unattendable.
        slot_region[:, :, 6:] = -1
        key_mask[:, :, 6:] = False
        _, a, _ = ws(bank_k, bank_v, key_mask, slot_region, n_regions=3)
        assert torch.allclose(a.sum(dim=-1), torch.ones(2, 1), atol=1e-4)

    def test_unadmitted_region_gets_zero_mass(self) -> None:
        """Masking one region's slots at iteration `i` removes it from `a[:, i, :]`."""
        torch.manual_seed(0)
        ws = Workspace(D_w=16, L=5, n_iter=1, heads=4, mlp_ratio=4.0).eval()
        bank_k, bank_v, key_mask, slot_region = _toy_bank(
            batch_size=2, n_iter=1, n_slots=12, d_w=16, n_regions=3
        )
        # Region 1 occupies slots [4:8); mask them out as "not admitted at i".
        key_mask[:, :, 4:8] = False
        _, a, _ = ws(bank_k, bank_v, key_mask, slot_region, n_regions=3)
        assert torch.allclose(a[:, 0, 1], torch.zeros(2), atol=1e-6)
        assert torch.allclose(a.sum(dim=-1), torch.ones(2, 1), atol=1e-4)


class TestSeveranceGradient:
    """Table 9: "the severance mask gives zero gradient through the masked slots"."""

    def test_masked_slots_receive_zero_gradient(self) -> None:
        torch.manual_seed(0)
        ws = Workspace(D_w=16, L=5, n_iter=1, heads=4, mlp_ratio=4.0)
        bank_k, bank_v, key_mask, slot_region = _toy_bank(
            batch_size=2, n_iter=1, n_slots=12, d_w=16, n_regions=3
        )
        key_mask[:, :, 8:] = False  # sever region 2's slots (indices [8:12))
        bank_k.requires_grad_(True)
        bank_v.requires_grad_(True)
        z, _, _ = ws(bank_k, bank_v, key_mask, slot_region, n_regions=3)
        z.sum().backward()
        assert torch.equal(bank_k.grad[:, :, 8:], torch.zeros_like(bank_k.grad[:, :, 8:]))
        assert torch.equal(bank_v.grad[:, :, 8:], torch.zeros_like(bank_v.grad[:, :, 8:]))
        # Sanity: the unmasked region does receive gradient, so the zero above is the
        # severance mask, not a dead graph.
        assert bank_k.grad[:, :, :8].abs().sum() > 0
        assert bank_v.grad[:, :, :8].abs().sum() > 0

    def test_inactive_iteration_blocks_all_gradient(self) -> None:
        """No block at or beyond an item's halted iteration receives gradient."""
        torch.manual_seed(0)
        ws = Workspace(D_w=16, L=5, n_iter=3, heads=4, mlp_ratio=4.0)
        bank_k, bank_v, key_mask, slot_region = _toy_bank(
            batch_size=1, n_iter=3, n_slots=12, d_w=16, n_regions=3
        )
        active = torch.tensor([[True, False, False]])
        bank_k.requires_grad_(True)
        z, _, _ = ws(bank_k, bank_v, key_mask, slot_region, n_regions=3, active=active)
        z.sum().backward()
        assert torch.equal(bank_k.grad[:, 1:], torch.zeros_like(bank_k.grad[:, 1:]))
        assert bank_k.grad[:, 0].abs().sum() > 0


class TestDeterminism:
    """The "all" row: construction and forward under a fixed seed are bitwise identical."""

    def _run(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        ws = Workspace(D_w=16, L=5, n_iter=3, heads=4, mlp_ratio=4.0).eval()
        bank_k, bank_v, key_mask, slot_region = _toy_bank(
            batch_size=2, n_iter=3, n_slots=12, d_w=16, n_regions=3
        )
        z, a, _ = ws(bank_k, bank_v, key_mask, slot_region, n_regions=3)
        return z, a

    def test_two_runs_are_bitwise_identical(self) -> None:
        z1, a1 = self._run()
        z2, a2 = self._run()
        assert torch.equal(z1, z2)
        assert torch.equal(a1, a2)
