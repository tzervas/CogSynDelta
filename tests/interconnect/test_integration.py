"""Integration tests for `WhiteMatter` -- lane IC-8, spec section 5's "Integration test"
paragraph. Every assertion named there gets its own test function below, named after
the clause it covers.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from cogsyndelta.interconnect.controller import box_integerise
from cogsyndelta.interconnect.gates import check_attention_mass_floor
from cogsyndelta.interconnect.mind import WhiteMatter
from cogsyndelta.interconnect.schedule import (
    OutputSpec,
    Schedule,
    StepBudget,
    read_token_floor,
)
from tests.interconnect.conftest import FakeText, make_toy_inputs


def _custom_schedule(
    wm: WhiteMatter,
    *,
    halt_at: int,
    admitted: dict[str, tuple[bool, ...]] | None = None,
) -> Schedule:
    """Build a validated `Schedule` off `wm`'s own config, at a caller-chosen
    `halt_at`/`admitted` -- everything else follows the dense allocation (spec section
    3 step 2), the same arithmetic `WhiteMatter._dense_schedule` uses.
    """
    cfg = wm.config
    names = wm.participant_names
    n = len(names)
    if admitted is None:
        admitted = {name: tuple([True] * cfg.n_iter) for name in names}
    uniform_target = torch.full((n,), cfg.budget_total_read_tokens / n, dtype=torch.float64)
    lo = torch.tensor(
        # spec section 3 step 2's lo_r, through the same shared definition
        # `_dense_schedule` uses -- not a second copy that can drift from it.
        [
            read_token_floor(
                cfg.participants[name].token_budget_min,
                cfg.floor_eta,
                n,
                cfg.budget_total_read_tokens,
            )
            for name in names
        ],
        dtype=torch.float64,
    )
    hi = torch.tensor(
        [cfg.participants[name].token_budget_max for name in names], dtype=torch.float64
    )
    b_vec = box_integerise(uniform_target, lo, hi).tolist()

    context_tokens, read_tokens, condition, precision, priority = {}, {}, {}, {}, {}
    total_kv = 0
    for idx, name in enumerate(names):
        spec = cfg.participants[name]
        is_store = name == "episodic_store"
        ctx_r = 0 if is_store else int(spec.ctx_max or 0)
        context_tokens[name] = ctx_r
        read_tokens[name] = int(b_vec[idx])
        condition[name] = False
        precision[name] = cfg.precision
        priority[name] = idx
        if not is_store:
            total_kv += wm.faculties[name].kv_bytes_per_token * ctx_r

    region_token_flops = cfg.n_iter * sum(
        cfg.participants[name].phi * context_tokens[name] for name in names
    )
    step_budget = StepBudget(
        max_iters=cfg.n_iter,
        kv_bytes=total_kv,
        read_tokens=cfg.budget_total_read_tokens,
        wall_ms=cfg.wall_ms_budget,
        flops_ceiling=wm._flops_ceiling,
    )
    output = OutputSpec(modalities=tuple(cfg.resident_heads))
    schedule = Schedule.assemble(
        context_tokens=context_tokens,
        read_tokens=read_tokens,
        admitted=admitted,
        condition=condition,
        precision=precision,
        priority=priority,
        region_token_flops=region_token_flops,
        step_budget=step_budget,
        output=output,
        halt_at=halt_at,
    )
    return wm.schedule_validator.validate(schedule)


def test_phase_a_loss_falls_over_20_steps(white_matter, toy_scope) -> None:
    """Spec: "runs phase A for 20 steps on toy items with a planted rule and asserts
    that the loss falls." Planted rule: the gold candidate index is the fake text ids'
    checksum mod `k`. `[lane]`: the same toy batch is reused every step (an ordinary
    overfitting check) rather than a fresh random batch per step -- with a random,
    un-normalised token-sum target and no positional signal in a masked-mean pool,
    a fresh random batch each step is a different, unrelated 4-way problem every time
    and Adam has no consistent gradient to descend; the spec's own "loss falls" claim
    is about optimisation actually working end to end, which a fixed toy batch already
    demonstrates over 20 steps without conflating it with generalisation (out of scope
    for a CPU smoke-shaped integration test).
    """
    torch.manual_seed(0)
    opt = torch.optim.Adam([p for p in white_matter.parameters() if p.requires_grad], lr=5e-2)
    inputs = make_toy_inputs(batch_size=4, scope=toy_scope, seed=0)
    target = inputs["language"].sum(dim=1) % 4
    losses = []
    for _step in range(20):
        out = white_matter(inputs)
        loss = F.cross_entropy(out.scores, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_a_is_a_distribution_over_admitted_regions(white_matter, toy_scope) -> None:
    """Spec Table 3: "`a[b, i, :]` sums to 1 over admitted regions ... where
    `active[b, i]`" -- the dense schedule admits everyone at every iteration.
    """
    inputs = make_toy_inputs(batch_size=3, scope=toy_scope, seed=1)
    out = white_matter(inputs)
    totals = out.a.sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-4)


def test_attention_mass_meets_the_floor_on_every_region(white_matter, toy_scope) -> None:
    """G29 (`gates.py`), spec section 4 Table 8: every region's mean admitted
    attention mass over active iterations must be at least `eta / R`. Summing to 1
    (the test above) alone does not catch a collapse -- a fully collapsed `a` (100% of
    mass on one region, 0% on the rest) still sums to 1 -- so this test routes the
    integrated module's own exported `a` through the real G29 guard, the same function
    the compose-stage script calls over a receipt (IC-R1 skeptic F4: nothing in this
    module's own test suite previously did). A mutant that collapses attention to one
    slot (`WorkspaceBlock._cross_attend` forced to slot 0) must fail THIS test, not
    only the severance-gradient unit tests in `test_workspace.py`.

    Calls `white_matter` twice on the same scope first: `episodic_store` legitimately
    gets zero mass on the very first read of a scope with nothing written yet (spec,
    `test_store_mass_only_after_a_write_in_the_same_scope` above) -- that is correct
    behaviour, not a collapse, and must not make this guard fire on an honest empty
    store. The second call's store slots have something to read, so all three
    participants have genuinely earned mass to check the floor against.
    """
    inputs = make_toy_inputs(batch_size=3, scope=toy_scope, seed=1)
    white_matter(inputs)  # primes episodic_store's write in this scope
    out = white_matter(inputs)
    r = len(white_matter.participant_names)
    mean_a = out.a.mean(dim=(0, 1))  # [R]: mean over batch and active iterations
    for name, mean_mass in zip(white_matter.participant_names, mean_a.tolist(), strict=True):
        check_attention_mass_floor(name, mean_mass, white_matter.config.floor_eta, r)


def test_write_back_changes_h_at_iteration_1_not_0(
    toy_config, fake_faculties, fake_store, toy_scope
) -> None:
    """Spec: "write-back changes `h_r` at iteration 1 and not at iteration 0.""" ""
    calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    text_fac = fake_faculties["language"]
    original_tokens = text_fac.tokens
    exec_ctx = toy_config.participants["language"].ctx_max  # distinguishes execution
    # calls (ctx_max) from the controller's own cheap ctx_min summary call (tension 1).

    def spy(inputs, *, context_tokens, condition=None):
        h, mask = original_tokens(inputs, context_tokens=context_tokens, condition=condition)
        if context_tokens == exec_ctx:
            calls.append(
                (h.detach().clone(), None if condition is None else condition.detach().clone())
            )
        return h, mask

    text_fac.tokens = spy  # type: ignore[method-assign]
    wm = WhiteMatter(toy_config, fake_faculties, fake_store)
    inputs = make_toy_inputs(batch_size=2, scope=toy_scope, seed=2)
    wm(inputs)

    assert len(calls) == 2, "language must be encoded fresh at both iterations under write-back"
    h0, cond0 = calls[0]
    h1, cond1 = calls[1]
    assert cond0 is None, "iteration 0 must never condition (spec section 3 step 5)"
    assert cond1 is not None, "iteration 1 must condition when write_back is on"
    # Same ids, same ctx_r truncation at both iterations -> the unconditioned embedding
    # is identical to h0; h1 differing from h0 is exactly "write-back changed h_r".
    assert not torch.allclose(h0, h1)


def test_store_mass_only_after_a_write_in_the_same_scope(
    toy_config, fake_faculties, fake_store, toy_scope
) -> None:
    """Spec: "the store's slots receive mass only after a write in the same scope."""
    wm = WhiteMatter(toy_config, fake_faculties, fake_store)
    store_idx = wm.participant_names.index("episodic_store")

    inputs_no_scope = make_toy_inputs(batch_size=2, scope=None, seed=3)
    out_no_scope = wm(inputs_no_scope)
    assert torch.allclose(
        out_no_scope.a[..., store_idx], torch.zeros_like(out_no_scope.a[..., store_idx])
    )

    inputs_scoped = make_toy_inputs(batch_size=2, scope=toy_scope, seed=4)
    out_first_write = wm(inputs_scoped)  # this call's own read precedes its own write
    assert torch.allclose(
        out_first_write.a[..., store_idx], torch.zeros_like(out_first_write.a[..., store_idx])
    )

    out_after_write = wm(inputs_scoped)  # now the read sees the previous call's write
    assert out_after_write.a[..., store_idx].sum().item() > 0.0


def test_emitted_schedule_validates(white_matter, toy_scope) -> None:
    """Spec: "the emitted `Schedule` validates." """
    inputs = make_toy_inputs(batch_size=2, scope=toy_scope, seed=5)
    out = white_matter(inputs)
    # Re-validating a schedule that already validated once must be pure/idempotent.
    revalidated = white_matter.schedule_validator.validate(out.schedule)
    assert revalidated is out.schedule


def test_write_back_off_omits_edges(toy_config, fake_faculties, fake_store, toy_scope) -> None:
    """Spec: "switching `write_back` off omits `edges`." """
    import dataclasses

    on_wm = WhiteMatter(toy_config, fake_faculties, fake_store)
    inputs = make_toy_inputs(batch_size=2, scope=toy_scope, seed=6)
    out_on = on_wm(inputs)
    assert out_on.edges is not None

    off_config = dataclasses.replace(toy_config, write_back=False)
    off_wm = WhiteMatter(off_config, fake_faculties, fake_store)
    out_off = off_wm(inputs)
    assert out_off.edges is None


def test_bank_is_load_bearing(white_matter, toy_scope) -> None:
    """Spec: "the bank is load-bearing, `∂f/∂bank_v != 0`." """
    captured: dict[str, torch.Tensor] = {}
    original_forward = white_matter.kv_bank.forward

    def spy(slot_budget, adapted_tokens, store_latents=None):
        bank_k, bank_v, key_mask, slot_region = original_forward(
            slot_budget, adapted_tokens, store_latents
        )
        bank_v.retain_grad()
        captured["bank_v"] = bank_v
        return bank_k, bank_v, key_mask, slot_region

    white_matter.kv_bank.forward = spy  # type: ignore[method-assign]
    inputs = make_toy_inputs(batch_size=2, scope=toy_scope, seed=7)
    out = white_matter(inputs)
    out.f.sum().backward()

    assert "bank_v" in captured
    assert captured["bank_v"].grad is not None
    assert captured["bank_v"].grad.abs().sum().item() > 0.0


def test_halting_is_reachable_and_no_gradient_past_halt_at(white_matter, toy_scope) -> None:
    """Spec: "halting is reachable and no block at or beyond `halt_at` receives
    gradient." `n_iter=2`; halting at 1 must never run (or gradient-touch) block 1.
    """
    schedule = _custom_schedule(white_matter, halt_at=1)
    inputs = make_toy_inputs(batch_size=2, scope=toy_scope, seed=8)
    out = white_matter(inputs, schedule=schedule)
    assert len(out.z_history) == 1

    for p in white_matter.workspace.blocks[1].parameters():
        assert p.grad is None
    out.f.sum().backward()
    for p in white_matter.workspace.blocks[1].parameters():
        assert p.grad is None, "block 1 sits at/beyond halt_at=1 and must never receive gradient"
    assert any(p.grad is not None for p in white_matter.workspace.blocks[0].parameters())


def test_unadmitted_region_is_unencoded_and_unread_at_that_iteration(
    toy_config, fake_faculties, fake_store, toy_scope
) -> None:
    """Spec: "an `A` with some `A[i, r] = 0` leaves that region unencoded and unread at
    `i`." `visual` is admitted at iteration 0 only.
    """
    call_count = {"n": 0}
    visual_fac = fake_faculties["visual"]
    original_tokens = visual_fac.tokens

    def spy(inputs, *, context_tokens, condition=None):
        call_count["n"] += 1
        return original_tokens(inputs, context_tokens=context_tokens, condition=condition)

    visual_fac.tokens = spy  # type: ignore[method-assign]
    wm = WhiteMatter(toy_config, fake_faculties, fake_store)
    admitted = {
        "language": (True, True),
        "visual": (True, False),
        "episodic_store": (True, True),
    }
    schedule = _custom_schedule(wm, halt_at=2, admitted=admitted)
    inputs = make_toy_inputs(batch_size=2, scope=toy_scope, seed=9)
    wm(inputs, schedule=schedule)

    assert call_count["n"] == 1, "visual must be encoded exactly once (iteration 0 only)"


def test_distinct_z_give_distinct_f(white_matter) -> None:
    """Spec: "two distinct `z` give distinct `f`." """
    torch.manual_seed(0)
    z1 = torch.randn(2, white_matter.config.latents, white_matter.config.workspace_dim)
    z2 = torch.randn(2, white_matter.config.latents, white_matter.config.workspace_dim)
    f1 = white_matter.frontal_readout(white_matter.final_norm(z1))
    f2 = white_matter.frontal_readout(white_matter.final_norm(z2))
    assert not torch.allclose(f1, f2)


def test_frozen_schedule_reemits_byte_identically_while_f_changes(white_matter, toy_scope) -> None:
    """Spec: "the frozen-schedule arm, `forward(inputs, schedule=s0)` with the tokens
    content-swapped, re-emits `s0` byte-identically while `f` changes."
    """
    inputs_a = make_toy_inputs(batch_size=2, scope=toy_scope, seed=10)
    out_a = white_matter(inputs_a)
    s0 = out_a.schedule

    inputs_b = make_toy_inputs(batch_size=2, scope=toy_scope, seed=11)
    out_b = white_matter(inputs_b, schedule=s0)

    assert out_b.schedule.to_json() == s0.to_json()
    assert not torch.allclose(out_a.f, out_b.f)


# ---------------------------------------------------------------------------
# Table 3 shape and mask contracts (IC-R2 skeptic item 4)
# ---------------------------------------------------------------------------


def test_a_is_padded_to_n_iter_so_receipts_stack_across_halt_points(
    white_matter, toy_scope
) -> None:
    """Spec Table 3: `a` is `[B, I, R]`, "all zero where not [active]" -- not
    `[B, halt_at, R]`. IC-R2 skeptic item 4b: returning only the executed iterations
    made the second axis request-dependent, so stacking receipts from two items with
    different `halt_at` values raised instead of producing `[N, I, R]`. This test runs
    the same module at `halt_at = n_iter` and at `halt_at = 1` and stacks the two.
    """
    inputs = make_toy_inputs(batch_size=2, scope=toy_scope, seed=20)
    full = white_matter(inputs)
    n_iter = white_matter.config.n_iter
    assert full.a.shape == (2, n_iter, len(white_matter.participant_names))

    out_early = white_matter(inputs, schedule=_custom_schedule(white_matter, halt_at=1))
    assert out_early.halt_at == 1
    assert out_early.a.shape == full.a.shape
    assert torch.all(out_early.a[:, 1:] == 0), "iterations past halt_at must be zero-padded"
    assert torch.any(out_early.a[:, 0] != 0), "the executed iteration must carry mass"

    stacked = torch.stack([full.a, out_early.a])
    assert stacked.shape == (2, *full.a.shape)


def test_budget_mask_is_anded_in_before_topkselect_sees_the_mask(
    toy_config, fake_faculties, fake_store, toy_scope
) -> None:
    """Spec Table 3's `budget_mask_r`: "`[B, T_r]` bool, `True` where `t < ctx_r[b]`;
    ANDed into `mask_r`", consumed by `TopKSelect`. IC-R2 skeptic item 4d: it was never
    constructed, so nothing but the faculty's own good behaviour kept out-of-budget
    positions out of the bank -- and the frozen-region contract gives this module no way
    to verify that behaviour.

    The faculty here deliberately ignores `context_tokens` and returns all eight input
    positions with an all-`True` mask, while `ctx_max = 4` caps the budget at four. The
    mask `TopKSelect` actually receives is captured and must be `False` from position
    four onwards, which can only happen if `WhiteMatter` built the budget mask itself.
    """

    class IgnoresItsBudget(FakeText):
        """A `Faculty` that returns every input position regardless of its budget."""

        def tokens(self, inputs, *, context_tokens, condition=None):
            """Deliberately ignores `context_tokens` (see the test's docstring)."""
            mask = torch.ones(inputs.shape, dtype=torch.bool, device=inputs.device)
            return self.embed(inputs), mask

    capped = dataclasses.replace(
        toy_config,
        participants={
            **toy_config.participants,
            "language": dataclasses.replace(toy_config.participants["language"], ctx_max=4),
        },
    )
    faculties = dict(fake_faculties)
    faculties["language"] = IgnoresItsBudget()
    wm = WhiteMatter(capped, faculties, fake_store)

    seen: list[torch.Tensor] = []
    inner = wm.selectors["language"].forward

    def record(h, mask, b_r):
        seen.append(mask.clone())
        return inner(h, mask, b_r)

    wm.selectors["language"].forward = record  # type: ignore[method-assign]

    inputs = make_toy_inputs(batch_size=2, scope=toy_scope, seed=21)
    assert inputs["language"].shape[1] == 8
    wm(inputs)

    assert seen, "TopKSelect was never called for `language`"
    for mask in seen:
        assert mask.shape[1] == 8
        assert torch.all(mask[:, :4]), "in-budget positions must stay selectable"
        assert not torch.any(mask[:, 4:]), "positions past ctx_r must be masked out"
