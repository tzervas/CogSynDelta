"""Construction-time tests for `WhiteMatter`/`InterconnectConfig` -- lane IC-8.

IC-R1 skeptic F3: spec section 2.1's last sentence on `flops_ceiling` -- "construction
refuses a smaller value, so G30's dense fallback can never be refused by its own
validator." This file is the first place that sentence gets its own test; per-guard
enforcement for the ten Table 8 guards already lives in
`tests/test_interconnect_guards_can_fail.py`, but `InterconnectConfigError` is not one
of Table 8's rows -- it is `WhiteMatter.__init__`'s own construction-time refusal,
mirroring `ScheduleValidatorConfigError` and `ControllerConfigError`'s precedent
(`schedule.py`, `controller.py`).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from cogsyndelta.interconnect.controller import box_integerise
from cogsyndelta.interconnect.mind import (
    InterconnectConfig,
    InterconnectConfigError,
    ParticipantSpec,
    WhiteMatter,
)
from cogsyndelta.interconnect.schedule import ScheduleViolationError, read_token_floor
from tests.interconnect.conftest import make_toy_inputs


def test_flops_ceiling_below_the_dense_region_token_flops_is_refused(
    toy_config: InterconnectConfig, fake_faculties, fake_store
) -> None:
    """`toy_config`'s dense `region_token_flops` is `n_iter=2 * (phi_language=1.0 *
    ctx_max=8 + phi_visual=1.5 * ctx_max=6 + phi_store=0.0 * 0) == 34.0` -- a
    `flops_ceiling` of `1.0` is far under it and must be refused at construction,
    before any forward pass, rather than surfacing later as a `ScheduleViolationError`
    from inside `forward()` when the dense fallback trips G30 against itself.
    """
    bad_config = dataclasses.replace(toy_config, flops_ceiling=1.0)
    with pytest.raises(InterconnectConfigError, match="flops_ceiling"):
        WhiteMatter(bad_config, fake_faculties, fake_store)  # type: ignore[arg-type]


def test_flops_ceiling_defaults_to_the_dense_value_and_the_dense_schedule_validates(
    toy_config: InterconnectConfig, fake_faculties, fake_store, toy_scope
) -> None:
    """`flops_ceiling=None` (the default, `toy_config`'s own value) must resolve to
    exactly the dense schedule's own `region_token_flops` -- not merely something large
    enough -- and a real forward pass through the `schedule=None` dense-fallback arm
    must validate against it without raising.
    """
    wm = WhiteMatter(toy_config, fake_faculties, fake_store)  # type: ignore[arg-type]
    expected_dense_flops = toy_config.n_iter * sum(
        spec.phi * (spec.ctx_max or 0) for spec in toy_config.participants.values()
    )
    assert wm._flops_ceiling == pytest.approx(expected_dense_flops)

    inputs = make_toy_inputs(batch_size=2, scope=toy_scope, seed=2)
    out = wm(inputs)  # schedule=None -> _dense_schedule() -> must validate cleanly
    assert out.schedule.step_budget.flops_ceiling == pytest.approx(expected_dense_flops)


def _floor_binding_config(toy_config: InterconnectConfig) -> InterconnectConfig:
    """`toy_config` with the read-token floor made load-bearing.

    At `toy_config`'s own numbers (`eta=0.15`, `R=3`, `B_read=16`) the floor is
    `ceil(0.15/3 * 16) = 1`, below every participant's `token_budget_min=2`, so
    `max(token_budget_min, 0)` and `max(token_budget_min, ceil(eta/R * B_read))` agree
    and the bug this file pins is invisible. At `eta=0.5` the floor is
    `ceil(0.5/3 * 16) = 3`, above the regions' `token_budget_min=1`, and the store's
    `token_budget_min=10` soaks up enough of `B_read` that the two expressions produce
    DIFFERENT allocations -- which is the whole point.
    """
    return dataclasses.replace(
        toy_config,
        floor_eta=0.5,
        participants={
            "language": ParticipantSpec(
                ctx_min=2, ctx_max=8, token_budget_min=1, token_budget_max=8, phi=1.0
            ),
            "visual": ParticipantSpec(
                ctx_min=2, ctx_max=6, token_budget_min=1, token_budget_max=8, phi=1.5
            ),
            "episodic_store": ParticipantSpec(
                ctx_min=None, ctx_max=None, token_budget_min=10, token_budget_max=16, phi=0.0
            ),
        },
    )


def test_dense_allocation_is_pinned_and_validates_against_its_own_validator(
    toy_config: InterconnectConfig, fake_faculties, fake_store
) -> None:
    """IC-R2 skeptic item 1, measured: `_dense_schedule` built `lo` from
    `max(token_budget_min, 0)`, dropping spec section 3 step 2's
    `ceil(eta/R * B_read)` term that BOTH `ThalamicController.__init__` and
    `ScheduleValidator.__init__` enforce -- so on any config where that term binds, the
    dense fallback the module builds for `schedule=None` was refused by its own
    validator ("G30: memory: read_tokens=7 outside [8, 96]" on the v1 participant set).

    Two things are pinned here, on both the default and the floor-binding config:
    the exact `read_tokens` the dense allocation produces, and that
    `schedule_validator.validate` accepts it. `_dense_schedule` validates internally, so
    a regression raises rather than returning a wrong answer; validating the returned
    object a second time is the explicit form of the claim, and `validate` is pure
    (spec section 5 Table 9) so the second call is free of side effects.
    """
    wm = WhiteMatter(toy_config, fake_faculties, fake_store)  # type: ignore[arg-type]
    dense = wm._dense_schedule()
    assert {node.region: node.read_tokens for node in dense.nodes} == {
        "language": 6,
        "visual": 5,
        "episodic_store": 5,
    }
    assert wm.schedule_validator.validate(dense) is dense

    binding = _floor_binding_config(toy_config)
    wm_binding = WhiteMatter(binding, fake_faculties, fake_store)  # type: ignore[arg-type]
    dense_binding = wm_binding._dense_schedule()
    assert {node.region: node.read_tokens for node in dense_binding.nodes} == {
        "language": 3,
        "visual": 3,
        "episodic_store": 10,
    }
    assert wm_binding.schedule_validator.validate(dense_binding) is dense_binding
    for node in dense_binding.nodes:
        assert node.read_tokens >= read_token_floor(
            binding.participants[node.region].token_budget_min,
            binding.floor_eta,
            len(binding.participants),
            binding.budget_total_read_tokens,
        )


def test_the_dropped_floor_term_really_is_what_the_validator_refuses(
    toy_config: InterconnectConfig, fake_faculties, fake_store
) -> None:
    """The guard-can-fail half of the test above (`MEMORY.md`: verify a guard by making
    it fail). Rebuilds the dense allocation with the exact expression the bug used --
    `lo = max(token_budget_min, 0)` -- on the floor-binding config, and asserts the
    validator refuses the result. If `read_token_floor`'s term were ever dropped again,
    THIS is the allocation `_dense_schedule` would hand its own validator, so the test
    above would go red rather than silently pinning a wrong number.
    """
    binding = _floor_binding_config(toy_config)
    wm = WhiteMatter(binding, fake_faculties, fake_store)  # type: ignore[arg-type]
    names = wm.participant_names
    n = len(names)
    target = torch.full((n,), binding.budget_total_read_tokens / n, dtype=torch.float64)
    buggy_lo = torch.tensor(
        [max(binding.participants[name].token_budget_min, 0) for name in names],
        dtype=torch.float64,
    )
    hi = torch.tensor(
        [binding.participants[name].token_budget_max for name in names], dtype=torch.float64
    )
    buggy_b = box_integerise(target, buggy_lo, hi).tolist()
    assert [int(v) for v in buggy_b] == [2, 2, 12], "the bug's own allocation changed"

    dense = wm._dense_schedule()
    buggy_nodes = tuple(
        dataclasses.replace(node, read_tokens=int(buggy_b[names.index(node.region)]))
        for node in dense.nodes
    )
    buggy_schedule = dataclasses.replace(dense, nodes=buggy_nodes)
    with pytest.raises(ScheduleViolationError, match="read_tokens"):
        wm.schedule_validator.validate(buggy_schedule)
