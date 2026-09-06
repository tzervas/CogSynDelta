"""Tests for `cogsyndelta.interconnect.schedule` -- spec §5 Table 9's `schedule.py` row.

Covers: every Table 3 bound accepted at the boundary and refused one past it (including
`region_token_flops`, a malformed `admitted`, and a `depth` that disagrees with
`admitted`); `trace_id` refused; modality refusal in both directions; JSON round-trip
equality; the validator is pure and deterministic. Every guard test below follows
`tests/test_guards_can_fail.py`'s pattern: construct the exact condition G30 exists to
catch, assert it fires, and assert the neighbouring positive control still passes.
"""

from __future__ import annotations

import copy
import json

import pytest

from cogsyndelta.interconnect.schedule import (
    CODEC_V1,
    OutputSpec,
    ParticipantBudget,
    Schedule,
    ScheduleNode,
    ScheduleValidator,
    ScheduleValidatorConfigError,
    ScheduleViolationError,
)

pytestmark = pytest.mark.cpu

N_ITER = 4
B_READ = 16
B_KV = 100_000
ETA = 0.2
FLOPS_CEILING = 1_000.0


def _participants() -> dict[str, ParticipantBudget]:
    """Three encoding participants plus one store participant, at toy scale."""
    return {
        "language": ParticipantBudget(
            ctx_min=2,
            ctx_max=8,
            kv_bytes_per_token=64,
            token_budget_min=1,
            token_budget_max=10,
            accepts_condition=True,
        ),
        "visual": ParticipantBudget(
            ctx_min=2,
            ctx_max=8,
            kv_bytes_per_token=96,
            token_budget_min=1,
            token_budget_max=10,
            accepts_condition=True,
        ),
        "episodic_store": ParticipantBudget(
            ctx_min=None,
            ctx_max=None,
            kv_bytes_per_token=None,
            token_budget_min=1,
            token_budget_max=10,
            accepts_condition=False,
        ),
    }


def _validator(**overrides: object) -> ScheduleValidator:
    kwargs: dict[str, object] = {
        "participants": _participants(),
        "B_read": B_READ,
        "B_kv": B_KV,
        "n_iter": N_ITER,
        "eta": ETA,
        "flops_ceiling": FLOPS_CEILING,
        "allowed_modalities": {"text", "speech"},
        "resident_heads": {"text"},
    }
    kwargs.update(overrides)
    return ScheduleValidator(**kwargs)  # type: ignore[arg-type]


def _valid_schedule() -> Schedule:
    """A schedule that satisfies every B1 bound under `_validator()`'s configuration.

    read_tokens sum to B_READ=16 (6 + 6 + 4); kv bytes = 4*64 + 4*96 = 640 <= B_KV.
    """
    return Schedule.assemble(
        context_tokens={"language": 4, "visual": 4, "episodic_store": 0},
        read_tokens={"language": 6, "visual": 6, "episodic_store": 4},
        admitted={
            "language": (True, False, False, False),
            "visual": (True, False, False, False),
            "episodic_store": (True, False, False, False),
        },
        condition={"language": False, "visual": False, "episodic_store": False},
        precision={"language": "fp32", "visual": "fp32", "episodic_store": "fp32"},
        priority={"language": 0, "visual": 1, "episodic_store": 2},
        region_token_flops=640.0,
        step_budget=_step_budget(),
        output=OutputSpec(modalities=("text",)),
        halt_at=1,
    )


def _step_budget(**overrides: object):
    from cogsyndelta.interconnect.schedule import StepBudget

    kwargs: dict[str, object] = {
        "max_iters": 1,
        "kv_bytes": 640,
        "read_tokens": 16,
        "wall_ms": 500.0,
        "flops_ceiling": FLOPS_CEILING,
    }
    kwargs.update(overrides)
    return StepBudget(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------
# Boundary: every bound accepted at its edge, refused one past it.
# ---------------------------------------------------------------------------------------


def test_valid_schedule_at_every_boundary_is_accepted() -> None:
    validator = _validator()
    schedule = _valid_schedule()
    assert validator.validate(schedule) is schedule


def test_context_tokens_accepted_at_ctx_max_refused_one_past() -> None:
    validator = _validator()
    schedule = _valid_schedule()

    at_max = _replace_node(schedule, "language", context_tokens=8)  # ctx_max=8
    # language's kv share grows from 4*64 to 8*64; step_budget.kv_bytes must track it,
    # or the (unrelated) step_budget-consistency check would fire instead.
    at_max = _replace_step_budget(at_max, kv_bytes=8 * 64 + 4 * 96)
    validator.validate(at_max)

    over_max = _replace_node(schedule, "language", context_tokens=9)
    with pytest.raises(ScheduleViolationError, match="context_tokens"):
        validator.validate(over_max)


def test_read_tokens_below_floor_is_refused() -> None:
    validator = _validator()
    schedule = _valid_schedule()
    # lo_r = max(token_budget_min=1, ceil(eta/R * B_read)) = max(1, ceil(0.2/3*16)) = 2.
    # The per-node floor check fires while walking `nodes`, before any step_budget-total
    # consistency check runs, so step_budget need not be kept in sync for this case.
    under_floor = _replace_node(schedule, "language", read_tokens=1)
    with pytest.raises(ScheduleViolationError, match="read_tokens"):
        validator.validate(under_floor)


def test_read_tokens_over_ceiling_is_refused() -> None:
    validator = _validator()
    schedule = _valid_schedule()
    over_ceiling = _replace_node(schedule, "language", read_tokens=11)  # token_budget_max=10
    with pytest.raises(ScheduleViolationError, match="read_tokens"):
        validator.validate(over_ceiling)


def test_sum_read_tokens_must_equal_b_read() -> None:
    validator = _validator()
    schedule = _valid_schedule()
    off_by_one = _replace_node(schedule, "visual", read_tokens=5)  # sum drops to 15
    with pytest.raises(ScheduleViolationError, match="B_read"):
        validator.validate(off_by_one)


def test_kv_bytes_over_b_kv_is_refused() -> None:
    # ctx_min=0 for the ctx-bearing participants so the construction-time ctx_min
    # feasibility guard (a separate, legitimate check -- see the tests below) cannot
    # itself fire at a tiny B_kv, isolating the runtime kv-byte bound under test.
    lenient = {
        name: ParticipantBudget(
            ctx_min=0 if p.ctx_min is not None else None,
            ctx_max=p.ctx_max,
            kv_bytes_per_token=p.kv_bytes_per_token,
            token_budget_min=p.token_budget_min,
            token_budget_max=p.token_budget_max,
            accepts_condition=p.accepts_condition,
        )
        for name, p in _participants().items()
    }
    tight = _validator(participants=lenient, B_kv=100)
    schedule = _valid_schedule()  # 640 declared kv bytes > 100
    with pytest.raises(ScheduleViolationError, match="kv bytes"):
        tight.validate(schedule)


def test_region_token_flops_accepted_at_ceiling_refused_one_past() -> None:
    validator = _validator(flops_ceiling=640.0)
    schedule = _valid_schedule()
    # The schedule's own declared step_budget.flops_ceiling must not itself exceed the
    # validator's, or that (unrelated) bound fires first -- bring both to the ceiling
    # under test so only region_token_flops is being exercised.
    schedule = _replace_step_budget(schedule, flops_ceiling=640.0)
    validator.validate(schedule)  # exactly at the ceiling: accepted

    over_budget = _replace(schedule, region_token_flops=640.0001)
    with pytest.raises(ScheduleViolationError, match="region_token_flops"):
        validator.validate(over_budget)


def test_halt_at_over_max_iters_is_refused() -> None:
    validator = _validator()
    schedule = _valid_schedule()
    bad = _replace(schedule, halt_at=schedule.step_budget.max_iters + 1)
    with pytest.raises(ScheduleViolationError, match="halt_at"):
        validator.validate(bad)


def test_max_iters_over_n_iter_is_refused() -> None:
    validator = _validator()  # n_iter=4
    schedule = _valid_schedule()
    bad = _replace_step_budget(schedule, max_iters=N_ITER + 1)
    with pytest.raises(ScheduleViolationError, match="max_iters"):
        validator.validate(bad)


# ---------------------------------------------------------------------------------------
# Malformed `admitted`, and depth disagreeing with admitted -- G30's own named cases.
# ---------------------------------------------------------------------------------------


def test_all_zero_admitted_row_is_refused() -> None:
    """G30: an admitted row of all zeros -- the exact IC-9 guard-can-fail case."""
    validator = _validator()
    schedule = _valid_schedule()
    never_admitted = _replace_node(
        schedule, "visual", admitted=(False, False, False, False), depth=0
    )
    with pytest.raises(ScheduleViolationError, match="admitted is all-zero"):
        validator.validate(never_admitted)


def test_admitted_wrong_length_is_refused() -> None:
    validator = _validator()
    schedule = _valid_schedule()
    short = _replace_node(schedule, "visual", admitted=(True, False, False))
    with pytest.raises(ScheduleViolationError, match="length"):
        validator.validate(short)


def test_depth_disagreeing_with_admitted_is_refused() -> None:
    validator = _validator()
    schedule = _valid_schedule()
    wrong_depth = _replace_node(schedule, "language", admitted=(True, False, False, False), depth=2)
    with pytest.raises(ScheduleViolationError, match="depth"):
        validator.validate(wrong_depth)

    # Positive control: the honestly-derived depth for the same admitted row passes.
    right_depth = _replace_node(schedule, "language", admitted=(True, False, False, False), depth=0)
    validator.validate(right_depth)


def test_active_disagreeing_with_admitted_is_refused() -> None:
    validator = _validator()
    schedule = _valid_schedule()
    node = schedule.nodes[0]
    assert node.region == "language"
    tampered = ScheduleNode(**{**node.__dict__, "active": False})
    bad = _swap_node(schedule, tampered)
    with pytest.raises(ScheduleViolationError, match="active"):
        validator.validate(bad)


# ---------------------------------------------------------------------------------------
# trace_id and store-namespace refusal (from_dict boundary).
# ---------------------------------------------------------------------------------------


def test_trace_id_is_refused_at_top_level() -> None:
    raw = _valid_schedule().to_dict()
    raw["trace_id"] = "req-123"
    with pytest.raises(ScheduleViolationError, match="forbidden"):
        Schedule.from_dict(raw)


def test_trace_id_is_refused_on_a_node() -> None:
    raw = _valid_schedule().to_dict()
    raw["nodes"][0]["trace_id"] = "req-123"
    with pytest.raises(ScheduleViolationError, match="forbidden"):
        Schedule.from_dict(raw)


@pytest.mark.parametrize("key", ["scope", "namespace", "store_namespace"])
def test_store_namespace_key_is_refused(key: str) -> None:
    raw = _valid_schedule().to_dict()
    raw[key] = "principal-42"
    with pytest.raises(ScheduleViolationError, match="forbidden"):
        Schedule.from_dict(raw)


def test_from_dict_positive_control_round_trips_without_forbidden_keys() -> None:
    schedule = _valid_schedule()
    assert Schedule.from_dict(schedule.to_dict()) == schedule


def test_unrecognised_key_is_refused() -> None:
    raw = _valid_schedule().to_dict()
    raw["nodes"][0]["extra_junk_field"] = 1
    with pytest.raises(ScheduleViolationError, match="unrecognised"):
        Schedule.from_dict(raw)


def test_missing_required_key_is_refused() -> None:
    raw = _valid_schedule().to_dict()
    del raw["step_budget"]["wall_ms"]
    with pytest.raises(ScheduleViolationError, match="missing required field"):
        Schedule.from_dict(raw)


# ---------------------------------------------------------------------------------------
# Modality refusal in both directions.
# ---------------------------------------------------------------------------------------


def test_modality_not_in_allowed_modalities_is_refused() -> None:
    validator = _validator(allowed_modalities={"text"}, resident_heads={"text"})
    schedule = _valid_schedule()
    speech = _replace(schedule, output=OutputSpec(modalities=("speech",)))
    with pytest.raises(ScheduleViolationError, match="allowed_modalities"):
        validator.validate(speech)


def test_modality_allowed_but_no_resident_head_is_refused() -> None:
    validator = _validator(allowed_modalities={"text", "speech"}, resident_heads={"text"})
    schedule = _valid_schedule()
    speech = _replace(schedule, output=OutputSpec(modalities=("speech",)))
    with pytest.raises(ScheduleViolationError, match="resident head"):
        validator.validate(speech)


def test_modality_allowed_with_resident_head_is_accepted() -> None:
    validator = _validator(allowed_modalities={"text", "speech"}, resident_heads={"text", "speech"})
    schedule = _valid_schedule()
    speech = _replace(schedule, output=OutputSpec(modalities=("speech",)))
    validator.validate(speech)


# ---------------------------------------------------------------------------------------
# Table 8a inert-field bounds: precision, resident, codec, stream, latency fields.
# ---------------------------------------------------------------------------------------


def test_precision_outside_closed_vocabulary_is_refused() -> None:
    validator = _validator()
    schedule = _valid_schedule()
    bad = _replace_node(schedule, "language", precision="fp16")
    with pytest.raises(ScheduleViolationError, match="precision"):
        validator.validate(bad)


def test_non_resident_node_is_refused() -> None:
    validator = _validator()
    schedule = _valid_schedule()
    bad = _replace_node(schedule, "language", resident=False)
    with pytest.raises(ScheduleViolationError, match="resident=False"):
        validator.validate(bad)


def test_non_none_codec_is_refused() -> None:
    validator = _validator()
    schedule = _valid_schedule()
    bad = _replace_node(schedule, "language", codec="zstd")
    with pytest.raises(ScheduleViolationError, match="codec"):
        validator.validate(bad)


def test_stream_true_is_refused() -> None:
    validator = _validator()
    schedule = _valid_schedule()
    bad = _replace(schedule, output=OutputSpec(modalities=("text",), stream=True))
    with pytest.raises(ScheduleViolationError, match="stream"):
        validator.validate(bad)


def test_non_null_latency_fields_are_refused() -> None:
    validator = _validator()
    schedule = _valid_schedule()
    bad = _replace(schedule, output=OutputSpec(modalities=("text",), first_token_ms=12.0))
    with pytest.raises(ScheduleViolationError, match="null"):
        validator.validate(bad)


def test_condition_without_accepts_condition_is_refused() -> None:
    validator = _validator()
    schedule = _valid_schedule()
    bad = _replace_node(schedule, "episodic_store", condition=True)  # accepts_condition=False
    with pytest.raises(ScheduleViolationError, match="accepts_condition"):
        validator.validate(bad)


# ---------------------------------------------------------------------------------------
# ScheduleValidator construction-time refusals (spec §3 step 2).
# ---------------------------------------------------------------------------------------


def test_construction_refuses_unreachable_ctx_min() -> None:
    starved = dict(_participants())
    starved["language"] = ParticipantBudget(
        ctx_min=8,
        ctx_max=8,
        kv_bytes_per_token=10_000_000,
        token_budget_min=1,
        token_budget_max=10,
        accepts_condition=True,
    )
    with pytest.raises(ScheduleValidatorConfigError, match="ctx_min"):
        _validator(participants=starved, B_kv=1)


def test_construction_refuses_infeasible_floor_sum() -> None:
    greedy = {
        name: ParticipantBudget(
            ctx_min=p.ctx_min,
            ctx_max=p.ctx_max,
            kv_bytes_per_token=p.kv_bytes_per_token,
            token_budget_min=100,
            token_budget_max=200,
            accepts_condition=p.accepts_condition,
        )
        for name, p in _participants().items()
    }
    with pytest.raises(ScheduleValidatorConfigError, match="floors"):
        _validator(participants=greedy)


def test_construction_refuses_infeasible_ceiling_sum() -> None:
    stingy = {
        name: ParticipantBudget(
            ctx_min=p.ctx_min,
            ctx_max=p.ctx_max,
            kv_bytes_per_token=p.kv_bytes_per_token,
            token_budget_min=0,
            token_budget_max=1,
            accepts_condition=p.accepts_condition,
        )
        for name, p in _participants().items()
    }
    with pytest.raises(ScheduleValidatorConfigError, match="ceilings"):
        _validator(participants=stingy, B_read=B_READ)


def test_construction_refuses_empty_participants() -> None:
    with pytest.raises(ScheduleValidatorConfigError, match="participant"):
        _validator(participants={})


@pytest.mark.parametrize("bad_eta", [-0.1, 1.1])
def test_construction_refuses_eta_outside_unit_interval(bad_eta: float) -> None:
    with pytest.raises(ScheduleValidatorConfigError, match="eta"):
        _validator(eta=bad_eta)


def test_participant_budget_rejects_partial_ctx_axis() -> None:
    with pytest.raises(ScheduleValidatorConfigError, match="ctx"):
        ParticipantBudget(
            ctx_min=4,
            ctx_max=None,
            kv_bytes_per_token=64,
            token_budget_min=1,
            token_budget_max=10,
            accepts_condition=True,
        )


# ---------------------------------------------------------------------------------------
# JSON round-trip equality.
# ---------------------------------------------------------------------------------------


def test_json_round_trip_is_exact() -> None:
    schedule = _valid_schedule()
    restored = Schedule.from_json(schedule.to_json())
    assert restored == schedule


def test_json_round_trip_survives_serialisation_of_nulls() -> None:
    schedule = _valid_schedule()
    with_nulls = json.loads(schedule.to_json(indent=2))
    assert with_nulls["output"]["first_token_ms"] is None
    restored = Schedule.from_dict(with_nulls)
    assert restored == schedule


# ---------------------------------------------------------------------------------------
# Purity and determinism.
# ---------------------------------------------------------------------------------------


def test_validate_does_not_mutate_the_schedule() -> None:
    validator = _validator()
    schedule = _valid_schedule()
    before = copy.deepcopy(schedule.to_dict())
    validator.validate(schedule)
    after = copy.deepcopy(schedule.to_dict())
    assert before == after


def test_validate_is_deterministic_across_repeated_calls_and_validators() -> None:
    schedule = _valid_schedule()
    v1 = _validator()
    v2 = _validator()  # a second, independently constructed, identically configured validator
    r1 = v1.validate(schedule)
    r2 = v1.validate(schedule)
    r3 = v2.validate(schedule)
    assert r1 is schedule
    assert r2 is schedule
    assert r3 is schedule


def test_validate_is_deterministic_on_a_failing_schedule_too() -> None:
    validator = _validator()
    schedule = _valid_schedule()
    bad = _replace_node(schedule, "language", context_tokens=999)
    first_message = _raised_message(validator, bad)
    second_message = _raised_message(validator, bad)
    assert first_message == second_message


def _raised_message(validator: ScheduleValidator, schedule: Schedule) -> str:
    with pytest.raises(ScheduleViolationError) as excinfo:
        validator.validate(schedule)
    return str(excinfo.value)


# ---------------------------------------------------------------------------------------
# Schedule.assemble -- the emission half.
# ---------------------------------------------------------------------------------------


def test_assemble_derives_active_and_depth_from_admitted() -> None:
    schedule = Schedule.assemble(
        context_tokens={"language": 4},
        read_tokens={"language": 6},
        admitted={"language": (False, True, False, False)},
        condition={"language": False},
        precision={"language": "fp32"},
        priority={"language": 0},
        region_token_flops=10.0,
        step_budget=_step_budget(max_iters=4, read_tokens=6, kv_bytes=256),
        output=OutputSpec(modalities=("text",)),
        halt_at=2,
    )
    node = schedule.nodes[0]
    assert node.active is True
    assert node.depth == 1


def test_assemble_defaults_resident_true_and_codec_none() -> None:
    schedule = Schedule.assemble(
        context_tokens={"language": 4},
        read_tokens={"language": 6},
        admitted={"language": (True, False, False, False)},
        condition={"language": False},
        precision={"language": "fp32"},
        priority={"language": 0},
        region_token_flops=10.0,
        step_budget=_step_budget(max_iters=4, read_tokens=6, kv_bytes=256),
        output=OutputSpec(modalities=("text",)),
        halt_at=1,
    )
    node = schedule.nodes[0]
    assert node.resident is True
    assert node.codec == CODEC_V1


def test_assemble_rejects_mismatched_participant_sets() -> None:
    with pytest.raises(ScheduleViolationError, match="participant"):
        Schedule.assemble(
            context_tokens={"language": 4, "visual": 4},
            read_tokens={"language": 6},  # missing "visual"
            admitted={
                "language": (True, False, False, False),
                "visual": (True, False, False, False),
            },
            condition={"language": False, "visual": False},
            precision={"language": "fp32", "visual": "fp32"},
            priority={"language": 0, "visual": 1},
            region_token_flops=10.0,
            step_budget=_step_budget(),
            output=OutputSpec(modalities=("text",)),
            halt_at=1,
        )


# ---------------------------------------------------------------------------------------
# Test helpers.
# ---------------------------------------------------------------------------------------


def _replace_node(schedule: Schedule, region: str, **changes: object) -> Schedule:
    """A copy of `schedule` with one named node's fields changed."""
    new_nodes = []
    for node in schedule.nodes:
        if node.region == region:
            fields = {**node.__dict__, **changes}
            new_nodes.append(ScheduleNode(**fields))
        else:
            new_nodes.append(node)
    return Schedule(
        nodes=tuple(new_nodes),
        step_budget=schedule.step_budget,
        region_token_flops=schedule.region_token_flops,
        output=schedule.output,
        halt_at=schedule.halt_at,
    )


def _swap_node(schedule: Schedule, replacement: ScheduleNode) -> Schedule:
    new_nodes = tuple(replacement if n.region == replacement.region else n for n in schedule.nodes)
    return Schedule(
        nodes=new_nodes,
        step_budget=schedule.step_budget,
        region_token_flops=schedule.region_token_flops,
        output=schedule.output,
        halt_at=schedule.halt_at,
    )


def _replace_step_budget(schedule: Schedule, **changes: object) -> Schedule:
    from cogsyndelta.interconnect.schedule import StepBudget

    fields = {**schedule.step_budget.__dict__, **changes}
    return _replace(schedule, step_budget=StepBudget(**fields))  # type: ignore[arg-type]


def _replace(schedule: Schedule, **changes: object) -> Schedule:
    """A copy of `schedule` with one or more top-level fields changed."""
    fields = {**schedule.__dict__, **changes}
    return Schedule(**fields)  # type: ignore[arg-type]
