"""The episodic store as the forward pass actually uses it -- the 2026-09-07 defect set.

WHAT WAS MEASURED, AND WHY THIS FILE EXISTS. On `main` the store read returned a
CONSTANT: max absolute difference `0.0` across items in a batch, across batches, and
before versus after 50 training steps. Mutual information with the target was exactly
zero, so `dL/d(store attention)` was ~0 and the store's attention direction was
unidentified -- it random-walked. Two independent causes, plus one write-side bug:

  1. `EpisodicStore.read` took NO query. It filtered on exact `(principal, session[,
     domain])`, sorted `(-importance, +written_at)`, and returned the first `b_store`.
     Same scope for every item in a batch, therefore the same records, therefore a
     constant.
  2. Uniform importance plus an OLDEST-FIRST tie-break made the primers a permanent
     wall: with 8 primers resident, the model's own writes were never readable at all.
     Measured: an 8-record store and a 32-record store produced bit-identical trained
     results to 17 significant figures.
  3. `WhiteMatter._write_store` wrote every batch item under ONE `logical_key`, so
     last-write-wins collapsed `B` records to one. Measured: after 800 steps at `B = 8`
     the store held 12 records, not 6,408.

The repair is to the READ MECHANISM, not to training: a query-dependent read (cosine
ranking against the item's own `z_N` from a store-free no-grad pre-pass) was measured at
MI excess +1.58 to +1.86 bits and held-out accuracy 0.855-0.996 against a null of 0.29,
with 98.6% of items retrieving a top-1 record of their own class at step 0 -- before any
optimizer step -- over memories whose maximum off-diagonal cosine was 0.9993. Near
collinearity does not destroy the ranking signal.

Per-item SCOPING was measured and is NOT the fix (MI under null, held-out accuracy under
chance at all four seeds), and a content-derived scope key would repurpose
`(principal, session)` -- a security isolation boundary enforced by `derive_scope`/G32/
DEC-64 -- as a content index. Scope stays the isolation axis; the query carries content.

Every test below has a matching CAN-FAIL control that reconstructs the defect, per
`MEMORY.md`'s "verify guards by making them fail": a green assertion about a fixed bug is
worth only as much as the red one beside it.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from cogsyndelta.interconnect.kv_bank import STORE_PARTICIPANT
from cogsyndelta.interconnect.mind import WhiteMatter
from tests.interconnect.conftest import make_toy_inputs

# ---------------------------------------------------------------------------------------
# Defect 3: the write-key collision (`_write_store` / `_logical_keys_for`).
# ---------------------------------------------------------------------------------------


def test_each_batch_item_writes_its_own_store_record(white_matter, toy_scope) -> None:
    """`B` items are `B` independent turns, so one forward pass leaves `B` records.

    Spec section 3 step 13 writes "one record per turn"; a batch item IS a turn. Before
    the fix all four items shared `inputs["logical_key"]`, and `(scope, domain,
    logical_key)` is the store's whole identity, so three of the four writes were
    overwritten by the fourth under last-write-wins.
    """
    inputs = make_toy_inputs(batch_size=4, scope=toy_scope, seed=3)
    out = white_matter(inputs)

    assert out.write_receipt is not None
    keys = [receipt.logical_key for receipt in out.write_receipt]
    assert len(keys) == 4
    assert len(set(keys)) == 4, f"batch items collided on one identity: {keys}"

    _latents, mask = white_matter.store.read(toy_scope, domain="general", b_store=16)
    assert int(mask.sum().item()) == 4


def test_one_shared_logical_key_collapses_the_batch_to_a_single_record(
    white_matter, toy_scope
) -> None:
    """The can-fail control: hand the pass the key set the bug produced and the measured
    collapse comes straight back -- four writes, one surviving record. This is the exact
    shape that left the 800-step run holding 12 records instead of 6,408.
    """
    inputs = make_toy_inputs(batch_size=4, scope=toy_scope, seed=3)
    inputs["logical_keys"] = ["turn-0"] * 4
    white_matter(inputs)

    _latents, mask = white_matter.store.read(toy_scope, domain="general", b_store=16)
    assert int(mask.sum().item()) == 1, "last-write-wins is still the contract on one key"


def test_explicit_per_item_logical_keys_are_used_verbatim(white_matter, toy_scope) -> None:
    """A caller with real episode ids passes `logical_keys`; nothing is appended to them."""
    inputs = make_toy_inputs(batch_size=3, scope=toy_scope, seed=4)
    inputs["logical_keys"] = ["episode-a", "episode-b", "episode-c"]
    out = white_matter(inputs)

    assert out.write_receipt is not None
    assert [r.logical_key for r in out.write_receipt] == ["episode-a", "episode-b", "episode-c"]


def test_logical_keys_of_the_wrong_length_is_refused(white_matter, toy_scope) -> None:
    """Mirrors `_scopes_for`'s own length check: a per-item sequence that is not per item
    is a caller bug, and a silent broadcast would put two turns under one identity again.
    """
    inputs = make_toy_inputs(batch_size=3, scope=toy_scope, seed=5)
    inputs["logical_keys"] = ["only-one"]
    with pytest.raises(ValueError, match="logical_keys"):
        white_matter(inputs)


def test_successive_turns_accumulate_rather_than_replace(white_matter, toy_scope) -> None:
    """Two forward passes under distinct request-wide keys leave `2 * B` records.

    The suffix is appended to the caller's own base key, so `turn-0#0` and `turn-1#0` are
    different identities: the store grows with the run instead of being rewritten by it.
    """
    for turn in range(2):
        inputs = make_toy_inputs(batch_size=4, scope=toy_scope, seed=6 + turn)
        inputs["logical_key"] = f"turn-{turn}"
        white_matter(inputs)

    _latents, mask = white_matter.store.read(toy_scope, domain="general", b_store=32)
    assert int(mask.sum().item()) == 8


# ---------------------------------------------------------------------------------------
# Defects 1 and 2 through the forward pass: is the bank the workspace sees a constant?
# ---------------------------------------------------------------------------------------


def _observed_store_bank(white_matter: WhiteMatter, inputs: dict) -> torch.Tensor:
    """The `[B, b_store, D_w]` store bank the EXECUTION pass actually read.

    Taken by spying on `_read_store` rather than by recomputing it, so the test measures
    what the workspace was handed. `_store_query`'s pre-pass supplies no store bank of its
    own (it runs against `_empty_store_bank`), so `_read_store` is called exactly once per
    forward pass and there is nothing to disambiguate.
    """
    seen: list[torch.Tensor] = []
    original = white_matter._read_store

    def spy(*args, **kwargs):
        latents, mask = original(*args, **kwargs)
        seen.append(latents.detach().clone())
        return latents, mask

    white_matter._read_store = spy  # type: ignore[method-assign]
    try:
        white_matter(inputs)
    finally:
        white_matter._read_store = original  # type: ignore[method-assign]
    assert len(seen) == 1, f"_read_store ran {len(seen)} times, expected once per pass"
    return seen[0]


def _prime(white_matter: WhiteMatter, toy_scope, *, records: int) -> None:
    """Write `records` distinct episodes under the shared scope, as `cli.prime_store` does."""
    generator = torch.Generator().manual_seed(11)
    for i in range(records):
        white_matter.store.write(
            toy_scope,
            "general",
            f"primer-{i}",
            torch.randn(white_matter.config.workspace_dim, generator=generator),
        )


def test_the_store_bank_differs_across_items_of_one_batch(white_matter, toy_scope) -> None:
    """THE assertion this whole change exists for.

    Every item of a batch shares one scope by construction (`cli.synthetic_batches` does
    this deliberately, so the store is not empty at step 0). Before the query, the read was
    a pure function of that scope, so the bank was identical for every item -- max absolute
    difference across items 0.0, mutual information with the target exactly zero, and
    therefore `dL/d(store attention)` ~ 0. It must not be a constant any more.
    """
    _prime(white_matter, toy_scope, records=16)
    inputs = make_toy_inputs(batch_size=4, scope=toy_scope, seed=12)
    bank = _observed_store_bank(white_matter, inputs)

    spread = (bank - bank[0:1]).abs().max()
    assert float(spread) > 0.0, "the store bank is still constant across items of one batch"


def test_the_store_bank_is_constant_again_with_the_query_switched_off(
    toy_config, fake_faculties, fake_store, toy_scope
) -> None:
    """The can-fail control: `store_query=False` restores the measured defect exactly.

    Same records, same scope, same seed -- only the query is gone. If this arm did NOT go
    constant, the test above would be measuring something other than the query.
    """
    wm = WhiteMatter(
        dataclasses.replace(toy_config, store_query=False),
        fake_faculties,
        fake_store,  # type: ignore[arg-type]
    )
    _prime(wm, toy_scope, records=16)
    inputs = make_toy_inputs(batch_size=4, scope=toy_scope, seed=12)
    bank = _observed_store_bank(wm, inputs)

    spread = (bank - bank[0:1]).abs().max()
    assert float(spread) == 0.0, "the no-query control is not the constant it used to be"


def test_the_store_bank_changes_when_the_request_changes(white_matter, toy_scope) -> None:
    """Different inputs, same scope, same store: the retrieved records must differ.

    The batch-item assertion above could in principle be satisfied by a read that varied
    with position rather than with content. This one varies the CONTENT and holds everything
    else fixed.
    """
    _prime(white_matter, toy_scope, records=16)
    first = _observed_store_bank(
        white_matter, make_toy_inputs(batch_size=2, scope=toy_scope, seed=13)
    )
    second = _observed_store_bank(
        white_matter, make_toy_inputs(batch_size=2, scope=toy_scope, seed=14)
    )
    assert float((first - second).abs().max()) > 0.0


def test_the_query_is_the_pre_passes_z_n_and_carries_no_gradient(white_matter, toy_scope) -> None:
    """`_store_query` returns `[B, D_w]`, detached from the graph the real pass builds.

    The store read is a SELECTION. A query wired into autograd would put a sort on the
    backward path, and the pre-pass exists precisely so the selection is decided before the
    pass that trains begins.
    """
    _prime(white_matter, toy_scope, records=8)
    inputs = make_toy_inputs(batch_size=3, scope=toy_scope, seed=15)
    schedule = white_matter._dense_schedule()
    ctx, b, admitted, halt_at = white_matter._unpack_schedule(schedule)

    query = white_matter._store_query(
        inputs,
        ctx=ctx,
        b=b,
        admitted=admitted,
        halt_at=halt_at,
        batch_size=3,
        n_regions=len(white_matter.participant_names),
        device=torch.device("cpu"),
    )
    assert query is not None
    assert query.shape == (3, white_matter.config.workspace_dim)
    assert not query.requires_grad
    assert float((query - query[0:1]).abs().max()) > 0.0, "the query itself is a constant"


def test_the_pre_pass_sees_no_store_slots(white_matter, toy_scope) -> None:
    """The pre-pass is STORE-FREE: its bank has zero unmasked store slots.

    If the pre-pass could read the store, the query would depend on what the store already
    returned and the read would be defined in terms of itself. `_empty_store_bank` is what
    keeps the query a function of the request alone.
    """
    _prime(white_matter, toy_scope, records=8)
    masks: list[torch.Tensor] = []
    original = white_matter.kv_bank.forward

    def spy(slot_budget, adapted_tokens, store_latents=None):
        if not torch.is_grad_enabled() and store_latents is not None:
            masks.append(store_latents[1].detach().clone())
        return original(slot_budget, adapted_tokens, store_latents)

    white_matter.kv_bank.forward = spy  # type: ignore[method-assign]
    try:
        white_matter(make_toy_inputs(batch_size=2, scope=toy_scope, seed=16))
    finally:
        white_matter.kv_bank.forward = original  # type: ignore[method-assign]

    assert masks, "the pre-pass never assembled a bank"
    for mask in masks:
        assert not bool(mask.any()), "the pre-pass read the store it exists to precede"


def test_the_pre_pass_costs_exactly_one_extra_encode_round(
    toy_config, fake_faculties, fake_store, toy_scope
) -> None:
    """The pre-pass's price, pinned rather than absorbed: one more encode of each region.

    `_store_query` deliberately does not share `_iterate`'s `h_r` cache with the execution
    pass -- the pre-pass builds its tensors under `no_grad`, and handing those to the pass
    that trains would silently cut the gradient path through `TopKSelect` and
    `RegionAdapter`. That costs a second encode, and this test is where the cost is visible.
    """
    counts = {"pre": 0, "exec": 0}
    text_fac = fake_faculties["language"]
    original_tokens = text_fac.tokens
    exec_ctx = toy_config.participants["language"].ctx_max

    def spy(inputs, *, context_tokens, condition=None):
        if context_tokens == exec_ctx:
            counts["exec" if torch.is_grad_enabled() else "pre"] += 1
        return original_tokens(inputs, context_tokens=context_tokens, condition=condition)

    text_fac.tokens = spy  # type: ignore[method-assign]
    wm = WhiteMatter(toy_config, fake_faculties, fake_store)  # type: ignore[arg-type]
    wm(make_toy_inputs(batch_size=2, scope=toy_scope, seed=17))

    assert counts["exec"] == toy_config.n_iter, "write-back re-encodes once per iteration"
    assert counts["pre"] == counts["exec"], "the pre-pass is one extra run of the same loop"


def test_the_store_participant_is_still_the_one_the_bank_budgets_for(
    white_matter, toy_scope
) -> None:
    """The query changes WHICH records fill the store slots, never how many there are.

    `b_store` is a request-wide budget (`KVBank`'s static slot layout), so a repair that
    quietly changed the slot count would break the bank's contract rather than the store's.
    """
    _prime(white_matter, toy_scope, records=16)
    inputs = make_toy_inputs(batch_size=2, scope=toy_scope, seed=18)
    out = white_matter(inputs)
    bank = _observed_store_bank(white_matter, inputs)

    assert bank.shape == (2, out.b[STORE_PARTICIPANT], white_matter.config.workspace_dim)
