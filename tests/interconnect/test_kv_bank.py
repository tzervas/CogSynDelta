"""IC-2 -- `KVBank`, `StoreProjection`, and this file's share of G31.

`docs/design/INTERCONNECT-MODULE-SPEC.md` section 5 Table 9's `kv_bank.py` row: "the
dense and 10,000 random allocations pack without overflow; `slot_region` and `key_mask`
agree; an unadmitted region's slots are masked at iteration `i`; an empty store
partition yields zero unmasked store slots; `bank_k == bank_v` off the store slots;
`W_k` and `W_v` carry no bias." Plus the general "all" row: construction and forward
under a fixed seed are bitwise identical across two runs on CPU. Plus section 4 Table 8's
G31 row.

This is a lane-owned unit-test file (spec section 7, lane IC-2); `tests/interconnect/`
has no `__init__.py` or `conftest.py` yet (both are lane IC-8's), so every fixture this
file needs is built locally.
"""

from __future__ import annotations

import random

import pytest
import torch

from cogsyndelta.interconnect.kv_bank import STORE_PARTICIPANT, KVBank, StoreProjection

pytestmark = pytest.mark.cpu

D_W = 16
B_READ = 32
PARTICIPANTS_NO_STORE = ["language", "memory", "reasoning", "visual"]
PARTICIPANTS_WITH_STORE = [*PARTICIPANTS_NO_STORE, STORE_PARTICIPANT]


def _dense_budget(participants: list[str], b_read: int) -> dict[str, int]:
    base = b_read // len(participants)
    rem = b_read - base * len(participants)
    budget = dict.fromkeys(participants, base)
    for name in participants[:rem]:
        budget[name] += 1
    return budget


def _adapted(
    batch: int, b_r: int, d_w: int, all_real: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randn(batch, b_r, d_w)
    mask = (
        torch.ones(batch, b_r, dtype=torch.bool)
        if all_real
        else torch.zeros(batch, b_r, dtype=torch.bool)
    )
    return tokens, mask


# ---------------------------------------------------------------------------
# StoreProjection
# ---------------------------------------------------------------------------


def test_store_projection_output_shape() -> None:
    proj = StoreProjection(D_w=D_W)
    z = torch.randn(3, 8, D_W)
    k, v = proj(z)
    assert k.shape == (3, 8, D_W)
    assert v.shape == (3, 8, D_W)


def test_store_projection_carries_no_bias() -> None:
    proj = StoreProjection(D_w=D_W)
    assert proj.w_k.bias is None
    assert proj.w_v.bias is None


def test_store_projection_g31_fires_on_integer_input() -> None:
    proj = StoreProjection(D_w=D_W)
    ids = torch.randint(0, 100, (2, 4, D_W), dtype=torch.long)
    with pytest.raises(TypeError, match="G31"):
        proj(ids)


def test_store_projection_deterministic_under_fixed_seed() -> None:
    torch.manual_seed(0)
    p1 = StoreProjection(D_w=D_W)
    z1 = torch.randn(2, 4, D_W)
    out1 = p1(z1)

    torch.manual_seed(0)
    p2 = StoreProjection(D_w=D_W)
    z2 = torch.randn(2, 4, D_W)
    out2 = p2(z2)

    assert torch.equal(out1[0], out2[0])
    assert torch.equal(out1[1], out2[1])


# ---------------------------------------------------------------------------
# KVBank construction
# ---------------------------------------------------------------------------


def test_kv_bank_rejects_empty_participants() -> None:
    with pytest.raises(ValueError):
        KVBank(participants=[], D_w=D_W, B_read=B_READ)


def test_kv_bank_rejects_duplicate_participants() -> None:
    with pytest.raises(ValueError):
        KVBank(participants=["language", "language"], D_w=D_W, B_read=B_READ)


def test_kv_bank_without_store_has_no_store_projection() -> None:
    bank = KVBank(participants=PARTICIPANTS_NO_STORE, D_w=D_W, B_read=B_READ)
    assert bank.store_index is None
    assert bank.store_projection is None


def test_kv_bank_with_store_has_a_store_projection() -> None:
    bank = KVBank(participants=PARTICIPANTS_WITH_STORE, D_w=D_W, B_read=B_READ)
    assert bank.store_index == len(PARTICIPANTS_WITH_STORE) - 1
    assert isinstance(bank.store_projection, StoreProjection)


# ---------------------------------------------------------------------------
# KVBank forward -- shapes, masks, dense and random allocations
# ---------------------------------------------------------------------------


def test_kv_bank_dense_allocation_packs_without_overflow() -> None:
    bank = KVBank(participants=PARTICIPANTS_NO_STORE, D_w=D_W, B_read=B_READ)
    budget = _dense_budget(PARTICIPANTS_NO_STORE, B_READ)
    adapted = {name: _adapted(2, budget[name], D_W) for name in PARTICIPANTS_NO_STORE}
    bank_k, bank_v, key_mask, slot_region = bank(budget, adapted)
    assert bank_k.shape == (2, B_READ, D_W)
    assert bank_v.shape == (2, B_READ, D_W)
    assert key_mask.shape == (2, B_READ)
    assert slot_region.shape == (2, B_READ)
    assert key_mask.all()
    assert (slot_region >= 0).all()


def test_kv_bank_10000_random_allocations_pack_without_overflow() -> None:
    rng = random.Random(0)  # noqa: S311 -- deterministic test fixture, not cryptographic use
    bank = KVBank(participants=PARTICIPANTS_NO_STORE, D_w=D_W, B_read=B_READ)
    n_participants = len(PARTICIPANTS_NO_STORE)
    for _ in range(10_000):
        cuts = sorted(rng.sample(range(1, B_READ), n_participants - 1))
        bounds = [0, *cuts, B_READ]
        counts = [bounds[i + 1] - bounds[i] for i in range(n_participants)]
        budget = dict(zip(PARTICIPANTS_NO_STORE, counts, strict=True))
        adapted = {
            name: _adapted(1, budget[name], D_W)
            for name in PARTICIPANTS_NO_STORE
            if budget[name] > 0
        }
        bank_k, bank_v, key_mask, slot_region = bank(budget, adapted)
        assert bank_k.shape == (1, B_READ, D_W)
        assert int(key_mask.sum().item()) == sum(budget.values())
        assert (slot_region >= 0).all()


def test_kv_bank_slot_region_and_key_mask_agree_on_admitted_regions() -> None:
    bank = KVBank(participants=PARTICIPANTS_NO_STORE, D_w=D_W, B_read=B_READ)
    budget = _dense_budget(PARTICIPANTS_NO_STORE, B_READ)
    adapted = {name: _adapted(1, budget[name], D_W) for name in PARTICIPANTS_NO_STORE}
    _bank_k, _bank_v, key_mask, slot_region = bank(budget, adapted)
    for r_idx, name in enumerate(PARTICIPANTS_NO_STORE):
        this_region = slot_region[0] == r_idx
        assert bool(key_mask[0][this_region].all())


def test_kv_bank_unadmitted_region_slots_are_masked_at_this_iteration() -> None:
    bank = KVBank(participants=PARTICIPANTS_NO_STORE, D_w=D_W, B_read=B_READ)
    budget = _dense_budget(PARTICIPANTS_NO_STORE, B_READ)
    # Drop "reasoning" from adapted_tokens -- not admitted at this iteration.
    admitted = [n for n in PARTICIPANTS_NO_STORE if n != "reasoning"]
    adapted = {name: _adapted(1, budget[name], D_W) for name in admitted}
    bank_k, bank_v, key_mask, slot_region = bank(budget, adapted)

    reasoning_idx = PARTICIPANTS_NO_STORE.index("reasoning")
    reasoning_slots = slot_region[0] == reasoning_idx
    assert reasoning_slots.any()
    assert not bool(key_mask[0][reasoning_slots].any())
    assert torch.equal(bank_k[0][reasoning_slots], torch.zeros_like(bank_k[0][reasoning_slots]))
    assert torch.equal(bank_v[0][reasoning_slots], torch.zeros_like(bank_v[0][reasoning_slots]))

    # Every other region stays admitted and fully masked-in.
    for name in admitted:
        r_idx = PARTICIPANTS_NO_STORE.index(name)
        this_region = slot_region[0] == r_idx
        assert bool(key_mask[0][this_region].all())


def test_kv_bank_bank_k_equals_bank_v_off_store_slots() -> None:
    bank = KVBank(participants=PARTICIPANTS_WITH_STORE, D_w=D_W, B_read=B_READ)
    budget = _dense_budget(PARTICIPANTS_WITH_STORE, B_READ)
    adapted = {name: _adapted(2, budget[name], D_W) for name in PARTICIPANTS_NO_STORE}
    store_z = torch.randn(2, budget[STORE_PARTICIPANT], D_W)
    store_mask = torch.ones(2, budget[STORE_PARTICIPANT], dtype=torch.bool)
    bank_k, bank_v, _key_mask, slot_region = bank(budget, adapted, (store_z, store_mask))

    store_idx = PARTICIPANTS_WITH_STORE.index(STORE_PARTICIPANT)
    off_store = slot_region != store_idx
    torch.testing.assert_close(bank_k[off_store], bank_v[off_store])


def test_kv_bank_store_can_differ_between_k_and_v() -> None:
    bank = KVBank(participants=PARTICIPANTS_WITH_STORE, D_w=D_W, B_read=B_READ)
    budget = _dense_budget(PARTICIPANTS_WITH_STORE, B_READ)
    adapted = {name: _adapted(1, budget[name], D_W) for name in PARTICIPANTS_NO_STORE}
    store_z = torch.randn(1, budget[STORE_PARTICIPANT], D_W)
    store_mask = torch.ones(1, budget[STORE_PARTICIPANT], dtype=torch.bool)
    bank_k, bank_v, _key_mask, slot_region = bank(budget, adapted, (store_z, store_mask))

    store_idx = PARTICIPANTS_WITH_STORE.index(STORE_PARTICIPANT)
    on_store = slot_region[0] == store_idx
    # W_k and W_v are independently initialised Linear layers, so their outputs on the
    # identical store input differ almost surely.
    assert not torch.allclose(bank_k[0][on_store], bank_v[0][on_store])


def test_kv_bank_empty_store_partition_yields_zero_unmasked_store_slots() -> None:
    bank = KVBank(participants=PARTICIPANTS_WITH_STORE, D_w=D_W, B_read=B_READ)
    budget = _dense_budget(PARTICIPANTS_WITH_STORE, B_READ)
    adapted = {name: _adapted(1, budget[name], D_W) for name in PARTICIPANTS_NO_STORE}
    store_z = torch.randn(1, budget[STORE_PARTICIPANT], D_W)
    store_mask = torch.zeros(1, budget[STORE_PARTICIPANT], dtype=torch.bool)  # empty partition
    _bank_k, _bank_v, key_mask, slot_region = bank(budget, adapted, (store_z, store_mask))

    store_idx = PARTICIPANTS_WITH_STORE.index(STORE_PARTICIPANT)
    on_store = slot_region[0] == store_idx
    assert int(key_mask[0][on_store].sum().item()) == 0


def test_kv_bank_region_slots_get_distinct_sincos_positions() -> None:
    """`type_emb` alone would make every region slot identical; sincos breaks that."""
    bank = KVBank(participants=PARTICIPANTS_NO_STORE, D_w=D_W, B_read=B_READ)
    budget = _dense_budget(PARTICIPANTS_NO_STORE, B_READ)
    zeros = {
        name: (torch.zeros(1, budget[name], D_W), torch.ones(1, budget[name], dtype=torch.bool))
        for name in PARTICIPANTS_NO_STORE
    }
    bank_k, _bank_v, _key_mask, slot_region = bank(budget, zeros)

    r_idx = 0
    this_region = slot_region[0] == r_idx
    rows = bank_k[0][this_region]
    assert rows.shape[0] >= 2
    assert not torch.allclose(rows[0], rows[1])


# ---------------------------------------------------------------------------
# KVBank forward -- validation errors
# ---------------------------------------------------------------------------


def test_kv_bank_rejects_slot_budget_missing_a_participant() -> None:
    bank = KVBank(participants=PARTICIPANTS_NO_STORE, D_w=D_W, B_read=B_READ)
    budget = _dense_budget(PARTICIPANTS_NO_STORE, B_READ)
    del budget["visual"]
    with pytest.raises(ValueError):
        bank(budget, {})


def test_kv_bank_rejects_slot_budget_not_summing_to_b_read() -> None:
    bank = KVBank(participants=PARTICIPANTS_NO_STORE, D_w=D_W, B_read=B_READ)
    budget = _dense_budget(PARTICIPANTS_NO_STORE, B_READ)
    budget["language"] += 1
    with pytest.raises(ValueError):
        bank(budget, {})


def test_kv_bank_rejects_store_latents_without_a_store_participant() -> None:
    bank = KVBank(participants=PARTICIPANTS_NO_STORE, D_w=D_W, B_read=B_READ)
    budget = _dense_budget(PARTICIPANTS_NO_STORE, B_READ)
    adapted = {name: _adapted(1, budget[name], D_W) for name in PARTICIPANTS_NO_STORE}
    store_z = torch.randn(1, 4, D_W)
    store_mask = torch.ones(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError):
        bank(budget, adapted, (store_z, store_mask))


def test_kv_bank_rejects_missing_store_latents_when_store_is_a_participant() -> None:
    bank = KVBank(participants=PARTICIPANTS_WITH_STORE, D_w=D_W, B_read=B_READ)
    budget = _dense_budget(PARTICIPANTS_WITH_STORE, B_READ)
    adapted = {name: _adapted(1, budget[name], D_W) for name in PARTICIPANTS_NO_STORE}
    with pytest.raises(ValueError):
        bank(budget, adapted, None)


def test_kv_bank_rejects_a_tensor_shaped_off_its_own_slot_budget() -> None:
    bank = KVBank(participants=PARTICIPANTS_NO_STORE, D_w=D_W, B_read=B_READ)
    budget = _dense_budget(PARTICIPANTS_NO_STORE, B_READ)
    adapted = {name: _adapted(1, budget[name], D_W) for name in PARTICIPANTS_NO_STORE}
    # Corrupt "visual"'s tensor to the wrong slot count.
    wrong_tokens, wrong_mask = _adapted(1, budget["visual"] + 1, D_W)
    adapted["visual"] = (wrong_tokens, wrong_mask)
    with pytest.raises(ValueError):
        bank(budget, adapted)


# ---------------------------------------------------------------------------
# G31 -- kv_bank.py's share
# ---------------------------------------------------------------------------


def test_kv_bank_g31_fires_on_integer_adapted_tokens() -> None:
    bank = KVBank(participants=PARTICIPANTS_NO_STORE, D_w=D_W, B_read=B_READ)
    budget = _dense_budget(PARTICIPANTS_NO_STORE, B_READ)
    adapted = {name: _adapted(1, budget[name], D_W) for name in PARTICIPANTS_NO_STORE}
    ids = torch.randint(0, 100, (1, budget["language"], D_W), dtype=torch.long)
    adapted["language"] = (ids, torch.ones(1, budget["language"], dtype=torch.bool))
    with pytest.raises(TypeError, match="G31"):
        bank(budget, adapted)


def test_kv_bank_g31_fires_on_integer_store_latents() -> None:
    bank = KVBank(participants=PARTICIPANTS_WITH_STORE, D_w=D_W, B_read=B_READ)
    budget = _dense_budget(PARTICIPANTS_WITH_STORE, B_READ)
    adapted = {name: _adapted(1, budget[name], D_W) for name in PARTICIPANTS_NO_STORE}
    ids = torch.randint(0, 100, (1, budget[STORE_PARTICIPANT], D_W), dtype=torch.long)
    store_mask = torch.ones(1, budget[STORE_PARTICIPANT], dtype=torch.bool)
    with pytest.raises(TypeError, match="G31"):
        bank(budget, adapted, (ids, store_mask))


# ---------------------------------------------------------------------------
# Determinism under a fixed seed
# ---------------------------------------------------------------------------


def test_kv_bank_deterministic_under_fixed_seed() -> None:
    budget = _dense_budget(PARTICIPANTS_NO_STORE, B_READ)

    torch.manual_seed(0)
    bank1 = KVBank(participants=PARTICIPANTS_NO_STORE, D_w=D_W, B_read=B_READ)
    adapted1 = {name: _adapted(2, budget[name], D_W) for name in PARTICIPANTS_NO_STORE}
    out1 = bank1(budget, adapted1)

    torch.manual_seed(0)
    bank2 = KVBank(participants=PARTICIPANTS_NO_STORE, D_w=D_W, B_read=B_READ)
    adapted2 = {name: _adapted(2, budget[name], D_W) for name in PARTICIPANTS_NO_STORE}
    out2 = bank2(budget, adapted2)

    for t1, t2 in zip(out1, out2, strict=True):
        assert torch.equal(t1, t2)
