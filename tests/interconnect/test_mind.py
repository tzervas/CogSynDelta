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

from cogsyndelta.interconnect.mind import InterconnectConfig, InterconnectConfigError, WhiteMatter
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
