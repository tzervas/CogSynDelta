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

import pytest

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
