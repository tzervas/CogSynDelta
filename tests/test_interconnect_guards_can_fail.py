"""Every guard fires, and can be told apart from a guard that fires on everything --
lane IC-9, spec `docs/design/INTERCONNECT-MODULE-SPEC.md` section 5's "Guard-can-fail
tests" paragraph: "holds one constructed failure per row of Table 8 ... Each test also
asserts the positive control passes, so a guard that refuses everything is caught."

WHAT THIS FILE IS, AND WHAT IT IS NOT
This is the enforcement test suite for every guard built across IC-1 (`schedule.py`,
G30), IC-2 (`adapters.py`, G31), IC-6 (`episodic_store.py`, G32) and IC-10 (`gates.py`,
G27-G29, G33-G36) -- ten guards, ten test functions below, each named after its
G-number. It defines no new guard logic (`gates.py`, `schedule.py`, `adapters.py` and
`episodic_store.py` already do); it constructs the EXACT failing scenario spec section
5's own "Guard-can-fail tests" paragraph names, wherever a concrete number is given
there, and asserts a neighbouring positive control still passes. Per-guard unit tests
already live beside each guard's own file (`tests/interconnect/test_gates.py`,
`test_schedule.py`, `test_adapters.py`, `test_kv_bank.py`, `test_episodic_store.py`);
this file is the single place Table 8's ten rows are enumerated together, following
`tests/test_guards_can_fail.py`'s own convention for this repo's guard registry.

Every guard here fails closed: it raises, it never returns a boolean a caller could
forget to check (module docstring precedent: `tests/test_guards_can_fail.py`, "Reading
[a guard] confirmed intent, not behaviour").
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from cogsyndelta.interconnect.adapters import RegionAdapter, TopKSelect, assert_float_tract
from cogsyndelta.interconnect.episodic_store import (
    InMemoryStoreStub,
    Scope,
    StoreScopeError,
    derive_scope,
)
from cogsyndelta.interconnect.gates import (
    GateFailure,
    check_attention_mass_floor,
    check_frozen_set_identity,
    check_null_gate,
    check_overfit_gate,
    check_phase_d_revert,
    check_topology_agreement,
    check_write_back_gate,
)
from cogsyndelta.interconnect.schedule import (
    OutputSpec,
    ParticipantBudget,
    Schedule,
    ScheduleNode,
    ScheduleValidator,
    ScheduleViolationError,
    StepBudget,
)

# ----------------------------------------------------------------------------------
# G27 -- W5b write-back gate, gates.py
# ----------------------------------------------------------------------------------


def test_g27_write_back_gate_fires_on_a_1_5_point_own_bin_drop() -> None:
    """Spec section 5: "a receipt pair with a 1.5-point own-bin drop (G27)."""
    with pytest.raises(GateFailure, match="G27"):
        check_write_back_gate(
            "language",
            own_bin_before=0.800,
            own_bin_after=0.785,  # 1.5-point drop
            composed_before=0.700,
            composed_after=0.705,  # composed metric DOES improve -- own-bin alone must fire
        )


def test_g27_positive_control_small_drop_and_improved_composed_passes() -> None:
    check_write_back_gate(
        "language",
        own_bin_before=0.800,
        own_bin_after=0.795,  # 0.5-point drop, under the 1-point limit
        composed_before=0.700,
        composed_after=0.710,
    )


# ----------------------------------------------------------------------------------
# G28 -- topology agreement, gates.py
# ----------------------------------------------------------------------------------


def test_g28_topology_agreement_fires_at_90_percent_agreement() -> None:
    """Spec section 5: "two derivations disagreeing on 10% of items (G28)."""
    with pytest.raises(GateFailure, match="G28"):
        check_topology_agreement(0.90, item_count=100)


def test_g28_positive_control_97_percent_agreement_passes() -> None:
    check_topology_agreement(0.97, item_count=100)


# ----------------------------------------------------------------------------------
# G29 -- attention-mass floor, gates.py
# ----------------------------------------------------------------------------------


def test_g29_fires_when_a_region_is_masked_to_zero_mass() -> None:
    """Spec section 5: "a phase-A run with one region's keys masked so its mass is
    zero" -- `eta=0.15`, `R=4` -> floor `3.75%`.
    """
    with pytest.raises(GateFailure, match="G29"):
        check_attention_mass_floor("reasoning", mean_a=0.0, eta=0.15, r=4)


def test_g29_fires_when_the_receipt_prints_the_wrong_floor() -> None:
    """Spec section 5: "a receipt printing 3.75% at `R = 5`" -- the true floor at
    `eta=0.15, R=5` is `3.0%`, not `3.75%` (that is `R=4`'s floor).
    """
    with pytest.raises(GateFailure, match="G29"):
        check_attention_mass_floor("language", mean_a=0.05, eta=0.15, r=5, printed_floor=0.0375)


def test_g29_fires_on_the_store_falsifier_at_its_legal_floor() -> None:
    """Spec section 5: "the in-contract store falsifier of TAX:2662, `b_store` pinned
    at its legal floor of 8 on episodes with no recall dependency, asserting
    `mean(a_store) < 3.0%` (G29)" -- `eta=0.15, R=5` -> floor `3.0%`.
    """
    with pytest.raises(GateFailure, match="G29"):
        check_attention_mass_floor("store", mean_a=0.02, eta=0.15, r=5)


def test_g29_positive_control_mass_above_floor_and_correct_printed_floor_passes() -> None:
    check_attention_mass_floor("language", mean_a=0.20, eta=0.15, r=4, printed_floor=0.0375)


# ----------------------------------------------------------------------------------
# G30 -- Schedule validator, schedule.py
# ----------------------------------------------------------------------------------


def _minimal_validator_and_schedule() -> tuple[ScheduleValidator, Schedule]:
    """One participant, one valid `Schedule` -- the seed every G30 mutation below
    starts from (`dataclasses.replace` keeps every field but the one under test).
    """
    participants = {
        "language": ParticipantBudget(
            ctx_min=4,
            ctx_max=16,
            kv_bytes_per_token=64,
            token_budget_min=4,
            token_budget_max=16,
            accepts_condition=True,
        )
    }
    validator = ScheduleValidator(
        participants,
        B_read=8,
        B_kv=10_000,
        n_iter=2,
        eta=0.15,
        flops_ceiling=1_000.0,
        allowed_modalities=("text",),
        resident_heads=("text",),
    )
    node = ScheduleNode(
        region="language",
        active=True,
        depth=0,
        admitted=(True, True),
        context_tokens=8,
        read_tokens=8,
        condition=False,
        priority=0,
        precision="fp32",
        resident=True,
        codec="none",
    )
    schedule = Schedule(
        nodes=(node,),
        step_budget=StepBudget(
            max_iters=2, kv_bytes=512, read_tokens=8, wall_ms=100.0, flops_ceiling=64.0
        ),
        region_token_flops=64.0,
        output=OutputSpec(modalities=("text",)),
        halt_at=2,
    )
    return validator, schedule


def test_g30_positive_control_the_seed_schedule_validates() -> None:
    validator, schedule = _minimal_validator_and_schedule()
    assert validator.validate(schedule) is schedule


def test_g30_fires_one_token_over_b_read() -> None:
    """Spec section 5: "a `Schedule` one token over `B_read`"."""
    validator, schedule = _minimal_validator_and_schedule()
    node = dataclasses.replace(schedule.nodes[0], read_tokens=9)  # B_read=8
    bad = dataclasses.replace(
        schedule,
        nodes=(node,),
        step_budget=dataclasses.replace(schedule.step_budget, read_tokens=9),
    )
    with pytest.raises(ScheduleViolationError, match="G30"):
        validator.validate(bad)


def test_g30_fires_one_over_the_flops_ceiling() -> None:
    """Spec section 5: "one over the FLOP ceiling"."""
    validator, schedule = _minimal_validator_and_schedule()
    bad = dataclasses.replace(schedule, region_token_flops=1_000.0 + 1.0)
    with pytest.raises(ScheduleViolationError, match="G30"):
        validator.validate(bad)


def test_g30_fires_an_all_zero_admitted_row() -> None:
    """Spec section 5: "one with an `admitted` row of all zeros"."""
    validator, schedule = _minimal_validator_and_schedule()
    node = dataclasses.replace(schedule.nodes[0], admitted=(False, False))
    bad = dataclasses.replace(schedule, nodes=(node,))
    with pytest.raises(ScheduleViolationError, match="G30"):
        validator.validate(bad)


def test_g30_fires_a_depth_that_disagrees_with_admitted() -> None:
    """Spec section 5: "one whose `depth` disagrees with `admitted`"."""
    validator, schedule = _minimal_validator_and_schedule()
    node = dataclasses.replace(schedule.nodes[0], depth=1)  # admitted[0] is True -> depth must be 0
    bad = dataclasses.replace(schedule, nodes=(node,))
    with pytest.raises(ScheduleViolationError, match="G30"):
        validator.validate(bad)


def test_g30_fires_a_schedule_carrying_trace_id() -> None:
    """Spec section 5: "one carrying `trace_id`" -- caught structurally by
    `Schedule.from_dict`, the other half of G30 (`ScheduleValidator.validate` only
    ever sees already-parsed `Schedule`s, which cannot carry the field at all)."""
    _validator, schedule = _minimal_validator_and_schedule()
    payload = schedule.to_dict()
    payload["trace_id"] = "req-123"
    with pytest.raises(ScheduleViolationError, match="G30"):
        Schedule.from_dict(payload)


def test_g30_fires_a_modality_naming_speech_with_no_resident_head() -> None:
    """Spec section 5: "one naming `speech` with no head"."""
    validator, schedule = _minimal_validator_and_schedule()
    bad = dataclasses.replace(schedule, output=OutputSpec(modalities=("speech",)))
    with pytest.raises(ScheduleViolationError, match="G30"):
        validator.validate(bad)


# ----------------------------------------------------------------------------------
# G31 -- latent-tract assertion, adapters.py / kv_bank.py
# ----------------------------------------------------------------------------------


def test_g31_fires_on_argmax_ids_and_a_prefix_built_from_ids() -> None:
    """Spec section 5: "a region wrapper returning `argmax` ids and a prefix built
    from ids (G31)" -- an integer-typed tensor at the `h_r` / adapted-token boundary.
    """
    ids = torch.randint(0, 100, (2, 5))  # int64 "argmax ids", not position latents
    with pytest.raises(TypeError, match="not floating point"):
        assert_float_tract(ids, "h_r")

    adapter = RegionAdapter(token_dim=16, D_w=32)
    with pytest.raises(TypeError, match="not floating point"):
        adapter(ids.float().long())  # still integer dtype after a no-op cast

    selector = TopKSelect(token_dim=16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    with pytest.raises(TypeError, match="not floating point"):
        selector(torch.randint(0, 100, (2, 5, 16)), mask, b_r=3)


def test_g31_positive_control_float_tensors_pass() -> None:
    h = torch.randn(2, 5, 16)
    assert_float_tract(h, "h_r")
    adapter = RegionAdapter(token_dim=16, D_w=32)
    out = adapter(h)
    assert out.shape == (2, 5, 32)


# ----------------------------------------------------------------------------------
# G32 -- store scope, episodic_store.py
# ----------------------------------------------------------------------------------


def test_g32_fires_on_a_request_supplied_scope() -> None:
    """Spec section 5: "a request carrying its own scope (G32)" -- a value NOT built by
    `derive_scope`, e.g. a raw string a handler forwarded verbatim.
    """
    store = InMemoryStoreStub(domain_enum={"general"}, half_life_s=3600.0, importance_default=0.5)
    forged_scope = "principal:alice"  # a request-supplied string, not a Scope
    with pytest.raises(StoreScopeError, match="G32"):
        store.write(forged_scope, "general", "k1", torch.randn(8))  # type: ignore[arg-type]
    with pytest.raises(StoreScopeError, match="G32"):
        store.read(forged_scope, domain="general", b_store=4)  # type: ignore[arg-type]


def test_g32_fires_on_write_with_no_server_derived_scope() -> None:
    """`scope=None` refuses `write` (no legal partition for an unknown principal)."""
    store = InMemoryStoreStub(domain_enum={"general"}, half_life_s=3600.0, importance_default=0.5)
    with pytest.raises(StoreScopeError, match="G32"):
        store.write(None, "general", "k1", torch.randn(8))


def test_g32_positive_control_a_derived_scope_writes_and_reads() -> None:
    store = InMemoryStoreStub(domain_enum={"general"}, half_life_s=3600.0, importance_default=0.5)
    scope: Scope = derive_scope("alice", session="s1")
    store.write(scope, "general", "k1", torch.randn(8))
    latents, mask = store.read(scope, domain="general", b_store=4)
    assert mask[0].item() is True
    assert latents.shape == (4, 8)


# ----------------------------------------------------------------------------------
# G33 -- frozen-set identity, gates.py
# ----------------------------------------------------------------------------------


def test_g33_fires_on_a_flipped_checkpoint_byte() -> None:
    """Spec section 5: "a checkpoint with one flipped byte against its receipt"."""
    with pytest.raises(GateFailure, match="G33"):
        check_frozen_set_identity(
            [
                {
                    "name": "language",
                    "checkpoint_sha256": "aa" * 32,
                    "receipt_checkpoint_sha256": "ab" * 32,  # one byte differs
                    "kind": "contrastive_encoder",
                    "parametric": True,
                }
            ]
        )


def test_g33_fires_on_a_parametric_region_claiming_nonparametric_store() -> None:
    """Spec section 5: "a parametric region claiming `nonparametric_store` (G33)"."""
    sha = "cc" * 32
    with pytest.raises(GateFailure, match="G33"):
        check_frozen_set_identity(
            [
                {
                    "name": "language",
                    "checkpoint_sha256": sha,
                    "receipt_checkpoint_sha256": sha,
                    "kind": "nonparametric_store",
                    "parametric": True,
                }
            ]
        )


def test_g33_fires_on_a_nonparametric_store_row_missing_parametric() -> None:
    """IC-R1 skeptic F1: a row that declares `kind: "nonparametric_store"` but omits
    `parametric` entirely must not get the store's identity exemption for free -- the
    guard used to read both fields with `.get(...)`, so a missing `parametric` silently
    evaluated as falsy and the row was ACCEPTED. Fails closed now: refused.
    """
    sha = "ee" * 32
    with pytest.raises(GateFailure, match="G33"):
        check_frozen_set_identity(
            [
                {
                    "name": "language",
                    "checkpoint_sha256": sha,
                    "receipt_checkpoint_sha256": sha,
                    "kind": "nonparametric_store",
                    # "parametric" deliberately omitted
                }
            ]
        )


def test_g33_positive_control_matching_sha_and_honest_kind_passes() -> None:
    sha = "dd" * 32
    check_frozen_set_identity(
        [
            {
                "name": "language",
                "checkpoint_sha256": sha,
                "receipt_checkpoint_sha256": sha,
                "kind": "contrastive_encoder",
                "parametric": True,
            },
            {
                "name": "episodic_store",
                "checkpoint_sha256": "n/a",
                "receipt_checkpoint_sha256": "n/a",
                "kind": "nonparametric_store",
                "parametric": False,
            },
        ]
    )


# ----------------------------------------------------------------------------------
# G34 -- phase-D revert, gates.py
# ----------------------------------------------------------------------------------


def test_g34_fires_when_d_is_worse_than_c() -> None:
    """Spec section 5: "a D receipt worse than C (G34)"."""
    with pytest.raises(GateFailure, match="G34"):
        check_phase_d_revert(d_metric=0.70, c_metric=0.72, d_flops=900.0, c_flops=1_000.0)


def test_g34_positive_control_d_beats_c_at_lower_flops_passes() -> None:
    check_phase_d_revert(d_metric=0.75, c_metric=0.72, d_flops=800.0, c_flops=1_000.0)


# ----------------------------------------------------------------------------------
# G35 -- overfit gate, gates.py
# ----------------------------------------------------------------------------------


def test_g35_fires_on_a_6_point_gap() -> None:
    """Spec section 5: "a 6-point gap (G35)"."""
    with pytest.raises(GateFailure, match="G35"):
        check_overfit_gate(train_metric=0.80, held_out_metric=0.74)


def test_g35_positive_control_a_3_point_gap_passes() -> None:
    check_overfit_gate(train_metric=0.80, held_out_metric=0.77)


def test_g35_fires_on_a_float_noisy_exact_5_point_gap() -> None:
    """IC-R1 skeptic F2: `0.35 - 0.30 == 0.049999999999999996` in IEEE 754 float
    subtraction -- one ulp under the 5.00-point threshold, not over it. The bare
    `gap >= 5 * POINT` comparison this guard used to make read that as a passing 4.9999
    -point gap and ACCEPTED it; G27 already carries an `isclose` tolerance at this exact
    boundary (`gates.py:104`) for exactly this reason. Must fire.
    """
    with pytest.raises(GateFailure, match="G35"):
        check_overfit_gate(train_metric=0.35, held_out_metric=0.30)


# ----------------------------------------------------------------------------------
# G36 -- NULL gate, gates.py
# ----------------------------------------------------------------------------------


def test_g36_fires_on_low_null_recall_and_high_off_bin_fpr() -> None:
    """Spec section 5: "a general-bin `NULL` recall of 0.40 and an off-bin `NULL`
    false-positive rate of 0.10 (G36)"."""
    with pytest.raises(GateFailure, match="G36"):
        check_null_gate(general_null_recall=0.40, per_bin_null_fpr={"code": 0.02})
    with pytest.raises(GateFailure, match="G36"):
        check_null_gate(general_null_recall=0.60, per_bin_null_fpr={"code": 0.10})


def test_g36_positive_control_healthy_recall_and_fpr_pass() -> None:
    check_null_gate(general_null_recall=0.65, per_bin_null_fpr={"code": 0.02, "reasoning": 0.03})
