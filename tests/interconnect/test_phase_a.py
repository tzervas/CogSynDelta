"""Phase A of the DEC-50 training contract -- spec section 4, Table 6 row "A dense".

Three claims this file exists to make checkable, none of which the module could make
before it had a trainer:

1. The trainable and frozen sets are what Table 6 row A says, established by an explicit
   partition rather than by reading `requires_grad` back off the module.
2. The frozen sets are VERIFIABLY frozen -- every region parameter and every controller
   parameter has `grad is None` after a real backward pass, and none of them moves. A
   `requires_grad` flag is a declaration; a `None` gradient after `loss.backward()` is
   evidence.
3. The receipt records what was in force AT the loss site, not what was parsed from
   arguments. Two tests mutate the live temperature and the live `delta` after the trainer
   is built and assert the receipt follows the mutation.

Run directly:
    CUDA_VISIBLE_DEVICES="" python -m pytest tests/interconnect/test_phase_a.py -q
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from cogsyndelta.interconnect.cli import (
    V1_PARTICIPANTS,
    V1_REGION_WIDTHS,
    V1_STORE_SPEC,
)
from cogsyndelta.interconnect.episodic_store import InMemoryStoreStub
from cogsyndelta.interconnect.gates import GateFailure
from cogsyndelta.interconnect.kv_bank import STORE_PARTICIPANT
from cogsyndelta.interconnect.mind import InterconnectConfig, WhiteMatter
from cogsyndelta.interconnect.phase_a import (
    CONTROLLER_R5,
    PHASE_A_TRAINABLE_R5,
    TABLE_4_TOTAL_R5,
    PhaseABatch,
    PhaseAConfig,
    PhaseATrainer,
    loss_decreased,
    phase_a_parameter_partition,
)
from tests.interconnect.conftest import make_toy_inputs

# ----------------------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------------------


class _V1Faculty(nn.Module):
    """A Table 1 participant's construction-time surface, for the parameter-count build."""

    def __init__(self, name: str, token_dim: int, kv_bytes_per_token: int) -> None:
        super().__init__()
        self.name = name
        self.faculty = name
        self.token_dim = token_dim
        self.pooled_dim = token_dim
        self.kv_bytes_per_token = kv_bytes_per_token
        self.accepts_condition = True

    def tokens(
        self, inputs: Tensor, *, context_tokens: int, condition: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def pool(self, h: Tensor, mask: Tensor) -> Tensor:
        raise NotImplementedError


def build_v1(*, with_store: bool, write_back: bool = True) -> WhiteMatter:
    """Build `WhiteMatter` at the REAL v1 dimensions Table 4 was derived from."""
    participants = dict(V1_PARTICIPANTS)
    if with_store:
        participants[STORE_PARTICIPANT] = V1_STORE_SPEC
    faculties = {
        name: _V1Faculty(name, token_dim, kv_bytes)
        for name, (token_dim, kv_bytes) in V1_REGION_WIDTHS.items()
    }
    store = (
        InMemoryStoreStub(domain_enum={"general"}, half_life_s=3600.0, importance_default=0.5)
        if with_store
        else None
    )
    config = InterconnectConfig(participants=participants, write_back=write_back)
    return WhiteMatter(config, faculties, store)  # type: ignore[arg-type]


def honest_frozen_set(white_matter: WhiteMatter) -> list[dict[str, Any]]:
    """Table 7 frozen-set rows that G33 accepts: every sha cites itself, all fields present."""
    rows: list[dict[str, Any]] = []
    for name in white_matter.participant_names:
        is_store = name == STORE_PARTICIPANT
        # A real 64-hex-digit digest, not `hash()`: `hash()` of a str is salted per
        # process, so a flipped-byte test built on it is non-deterministic across runs.
        sha = hashlib.sha256(name.encode()).hexdigest()
        rows.append(
            {
                "name": name,
                "checkpoint_sha256": sha,
                "receipt_checkpoint_sha256": sha,
                "receipt_path": f"receipts/{name}.json",
                "status": "frozen",
                "kind": "nonparametric_store" if is_store else "contrastive_encoder",
                "parametric": not is_store,
            }
        )
    return rows


def toy_batch(white_matter: WhiteMatter, toy_scope: Any, *, seed: int = 0) -> PhaseABatch:
    """One planted-rule toy batch at the spec's integration dimensions."""
    inputs = make_toy_inputs(batch_size=4, scope=toy_scope, seed=seed, k=4)
    target = inputs["language"].sum(dim=1) % 4
    return PhaseABatch(inputs=inputs, target=target)


@pytest.fixture
def trainer(white_matter: WhiteMatter) -> PhaseATrainer:
    """A `PhaseATrainer` over the toy `white_matter`, with an honest frozen set."""
    return PhaseATrainer(
        white_matter, PhaseAConfig(lr=3e-3, steps=8), honest_frozen_set(white_matter)
    )


def identity_for(white_matter: WhiteMatter) -> dict[str, Any]:
    """Table 7's identity group, minus the two stamps `write_receipt` applies."""
    return {
        "corpus_fingerprint": "f" * 64,
        "fingerprint_scheme": "sha256-test",
        "battery_id": "plumbing",
        "k": white_matter.config.k_candidates,
        "pooling": "frontal",
        "checkpoint_sha256": "c" * 64,
        "region": "white_matter",
        "seed": 0,
        "split_sha256": "s" * 64,
    }


# ----------------------------------------------------------------------------------
# Claim 1 -- the trainable and frozen sets are Table 6 row A's two columns
# ----------------------------------------------------------------------------------


def test_the_partition_is_total_and_disjoint_over_named_parameters(
    white_matter: WhiteMatter,
) -> None:
    """Every parameter lands in exactly one Table 6 column, and none in both or neither."""
    trainable, frozen = phase_a_parameter_partition(white_matter)
    assert not (trainable.keys() & frozen.keys())
    assert trainable.keys() | frozen.keys() == set(dict(white_matter.named_parameters()))


def test_trainable_count_at_v1_is_table_4_total_minus_the_controller() -> None:
    """The number the task pins, and the number Table 6 actually trains, in one assertion.

    Section 2.3 Table 4's last core row is labelled "trainable in phase A, heads included,
    `R = 5`" and gives 28,376,334. Measured, that is EVERY parameter `WhiteMatter` owns at
    `R = 5` -- it includes the thalamic controller, which Table 6 row A's own "frozen"
    column lists as "controller (bypassed)". Both numbers are asserted here, with the
    subtraction shown, so the discrepancy between the two tables is recorded rather than
    resolved silently.
    """
    white_matter = build_v1(with_store=True)
    assert sum(p.numel() for p in white_matter.parameters()) == TABLE_4_TOTAL_R5

    trainable, frozen = phase_a_parameter_partition(white_matter)
    measured_trainable = sum(p.numel() for p in trainable.values())
    measured_frozen = sum(p.numel() for p in frozen.values())

    assert measured_trainable == PHASE_A_TRAINABLE_R5
    assert measured_frozen == CONTROLLER_R5
    assert measured_trainable + measured_frozen == TABLE_4_TOTAL_R5
    assert TABLE_4_TOTAL_R5 - CONTROLLER_R5 == PHASE_A_TRAINABLE_R5


def test_the_frozen_column_at_v1_is_exactly_the_controller() -> None:
    """Table 6 row A's "frozen" column, minus the regions, is one submodule and no other."""
    _, frozen = phase_a_parameter_partition(build_v1(with_store=True))
    assert {name.split(".")[0] for name in frozen} == {"controller"}


def test_the_heads_table_4_lists_separately_are_in_the_trainable_column() -> None:
    """Row A names "rank head, `NULL`, unify probes" among what receives gradient."""
    trainable, _ = phase_a_parameter_partition(build_v1(with_store=True))
    head_params = sum(
        param.numel()
        for name, param in trainable.items()
        if name.startswith(("rank_head.", "unify_probes."))
    )
    # Table 4's three separately-listed head rows: 262,656 + 512 + 590,976.
    assert head_params == 262_656 + 512 + 590_976
    assert any(name.startswith("rank_head.null") for name in trainable)


def test_write_back_off_moves_the_prefixes_into_the_frozen_column() -> None:
    """Row A: "prefixes only when write-back is enabled"."""
    on_trainable, _ = phase_a_parameter_partition(build_v1(with_store=True, write_back=True))
    off_trainable, off_frozen = phase_a_parameter_partition(
        build_v1(with_store=True, write_back=False)
    )
    prefix_params = sum(
        p.numel() for name, p in on_trainable.items() if name.startswith("conditioning.")
    )
    # Table 4's conditioning-prefix row (4,727,808) plus its prefix attention-pool
    # queries row (2,048).
    assert prefix_params == 4_727_808 + 2_048
    assert not any(name.startswith("conditioning.") for name in off_trainable)
    assert sum(p.numel() for p in off_frozen.values()) == CONTROLLER_R5 + prefix_params


def test_store_projections_are_trainable_only_at_r5() -> None:
    """Row A: "`W_k`/`W_v` in E2" -- i.e. only once the store is an admitted participant."""
    r5_trainable, _ = phase_a_parameter_partition(build_v1(with_store=True))
    r4_trainable, r4_frozen = phase_a_parameter_partition(build_v1(with_store=False))
    store_key = "kv_bank.store_projection."
    assert sum(p.numel() for n, p in r5_trainable.items() if n.startswith(store_key)) == 524_288
    assert not any(n.startswith(store_key) for n in r4_trainable)
    assert sum(p.numel() for n, p in r4_frozen.items() if n.startswith(store_key)) == 524_288


def test_a_submodule_in_neither_column_is_refused_rather_than_untrained(
    white_matter: WhiteMatter,
) -> None:
    """A parameter added to `mind.py` later must fail loudly, not be silently untrained."""
    white_matter.register_parameter("some_future_head", nn.Parameter(torch.zeros(4)))
    with pytest.raises(GateFailure, match="matched neither Table 6 column"):
        phase_a_parameter_partition(white_matter)


def test_optimizer_holds_exactly_the_trainable_column(trainer: PhaseATrainer) -> None:
    """The set the partition names is the set AdamW was actually handed."""
    held = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert held == {id(p) for p in trainer.trainable.values()}
    assert trainer.trainable_parameter_count() == sum(p.numel() for p in trainer.trainable.values())


# ----------------------------------------------------------------------------------
# Claim 2 -- the frozen sets are verifiably frozen, by gradient and not by flag
# ----------------------------------------------------------------------------------


def test_regions_receive_no_gradient_after_a_real_backward(
    white_matter: WhiteMatter, toy_scope: Any
) -> None:
    """Not `requires_grad`: `grad is None` after `loss.backward()`.

    The faculties are deliberately UNFROZEN first, undoing the conftest fixture, so this
    tests the trainer's own `freeze_regions` rather than the fixture's setup. Without it
    the assertion would hold for a trainer that froze nothing at all.
    """
    for faculty in white_matter.faculties.values():
        for param in faculty.parameters():
            param.requires_grad_(True)

    trainer = PhaseATrainer(white_matter, PhaseAConfig(steps=1), honest_frozen_set(white_matter))
    trainer.step(toy_batch(white_matter, toy_scope), step_index=0)

    checked = 0
    for faculty_name, param_name, param in trainer.region_parameters():
        assert param.grad is None, f"region {faculty_name}.{param_name} received a gradient"
        assert not param.requires_grad
        checked += 1
    assert checked > 0, "the toy faculties have no parameters; this test proves nothing"


def test_the_controller_receives_no_gradient_after_a_real_backward(
    trainer: PhaseATrainer, white_matter: WhiteMatter, toy_scope: Any
) -> None:
    """Row A's "frozen: controller (bypassed)", checked the same way."""
    trainer.step(toy_batch(white_matter, toy_scope), step_index=0)
    for name, param in white_matter.controller.named_parameters():
        assert param.grad is None, f"controller.{name} received a gradient"


def test_every_trainable_parameter_receives_a_finite_gradient(
    trainer: PhaseATrainer, white_matter: WhiteMatter, toy_scope: Any
) -> None:
    """The other side of the same coin: nothing in the trainable column is dead weight."""
    trainer.step(toy_batch(white_matter, toy_scope), step_index=0)
    for name, param in trainer.trainable.items():
        assert param.grad is not None, f"trainable parameter {name!r} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name!r} has a non-finite gradient"


def test_frozen_parameters_do_not_move_over_a_run(
    trainer: PhaseATrainer, white_matter: WhiteMatter, toy_scope: Any
) -> None:
    """A no-gradient parameter could still be moved by weight decay if it reached AdamW."""
    before = {name: param.detach().clone() for name, param in trainer.frozen.items()}
    region_before = {
        f"{f}.{p}": param.detach().clone() for f, p, param in trainer.region_parameters()
    }
    trainer.run([toy_batch(white_matter, toy_scope)], [], steps=4)
    for name, param in trainer.frozen.items():
        assert torch.equal(param, before[name]), f"frozen parameter {name!r} moved"
    for faculty_name, param_name, param in trainer.region_parameters():
        assert torch.equal(param, region_before[f"{faculty_name}.{param_name}"])


# ----------------------------------------------------------------------------------
# Claim 3 -- L_A, the loss curve, and what the receipt records
# ----------------------------------------------------------------------------------


def test_loss_decreases_over_a_handful_of_steps(
    trainer: PhaseATrainer, white_matter: WhiteMatter, toy_scope: Any
) -> None:
    """A tiny fixed batch, a handful of steps on CPU: `L_A` must fall."""
    batch = toy_batch(white_matter, toy_scope)
    result = trainer.run([batch], [batch], steps=12)
    assert len(result.loss_curve) == 12
    assert all(torch.isfinite(torch.tensor(value)) for value in result.loss_curve)
    assert loss_decreased(result.loss_curve), result.loss_curve
    assert result.loss_curve[-1] < result.loss_curve[0]


def test_l_a_is_l_task_plus_delta_times_l_unify(white_matter: WhiteMatter, toy_scope: Any) -> None:
    """`L_A = L_task + delta*L_unify`: at `delta = 0` the total collapses onto `L_task`."""
    frozen_set = honest_frozen_set(white_matter)
    batch = toy_batch(white_matter, toy_scope)

    zero = PhaseATrainer(white_matter, PhaseAConfig(delta=0.0), frozen_set)
    out = zero.white_matter(batch.inputs)
    total, l_task, l_unify = (t.detach() for t in zero.compute_loss(batch, out))
    assert float(l_unify) == pytest.approx(0.0)
    assert float(total) == pytest.approx(float(l_task))

    two = PhaseATrainer(white_matter, PhaseAConfig(delta=2.0), frozen_set)
    out2 = two.white_matter(batch.inputs)
    total2, l_task2, l_unify2 = (t.detach() for t in two.compute_loss(batch, out2))
    assert float(total2) == pytest.approx(float(l_task2) + float(l_unify2), rel=1e-5)
    # UnifyLoss carries `delta` internally, so `l_unify2` is already `delta * sum(...)`.
    assert float(l_unify2) > 0.0


def test_observed_loss_site_refuses_before_any_step(trainer: PhaseATrainer) -> None:
    """A receipt built before a step would be describing arguments, not a run."""
    with pytest.raises(RuntimeError, match="no loss has been computed yet"):
        _ = trainer.observed_loss_site


def test_receipt_records_the_temperature_in_force_not_the_argument(
    trainer: PhaseATrainer, white_matter: WhiteMatter, toy_scope: Any, tmp_path: Path
) -> None:
    """The silent-invalidation trap: `RankHead.temperature` is mutated after construction.

    `InterconnectConfig.rank_temperature` was 1.0 when the module was built, and a receipt
    that reported the parsed argument would still say 1.0. What actually divided the cosine
    scores at the loss site is 4.0, and that is what the receipt has to carry.
    """
    white_matter.rank_head.temperature = 4.0
    result = trainer.run([toy_batch(white_matter, toy_scope)], [], steps=2)

    assert white_matter.config.rank_temperature == 1.0
    assert result.loss_site["rank_temperature"] == 4.0

    path = trainer.write_receipt(result, identity_for(white_matter), tmp_path)
    on_disk = json.loads(path.read_text())
    assert on_disk["budgets"]["loss_site"]["rank_temperature"] == 4.0
    assert on_disk["budgets"]["loss_site"]["source"] == "observed at the loss site"


def test_receipt_records_the_delta_in_force_not_the_argument(
    white_matter: WhiteMatter, toy_scope: Any, tmp_path: Path
) -> None:
    """The same trap on the loss weights: `PhaseAConfig.delta` says 1.0, the site says 0.25."""
    trainer = PhaseATrainer(white_matter, PhaseAConfig(delta=1.0), honest_frozen_set(white_matter))
    trainer.unify_loss.weight = 0.25
    trainer.rank_loss.weight = 3.0
    result = trainer.run([toy_batch(white_matter, toy_scope)], [], steps=2)

    assert trainer.config.delta == 1.0
    assert result.loss_site["delta"] == 0.25
    assert result.loss_site["task_weight"] == 3.0

    on_disk = json.loads(
        trainer.write_receipt(result, identity_for(white_matter), tmp_path).read_text()
    )
    assert on_disk["budgets"]["loss_site"]["delta"] == 0.25
    assert on_disk["budgets"]["loss_site"]["task_weight"] == 3.0


def test_receipt_is_written_with_the_right_envelope_and_every_table_7_group(
    trainer: PhaseATrainer, white_matter: WhiteMatter, toy_scope: Any, tmp_path: Path
) -> None:
    """Section 4's "Receipts": envelope `model-pipeline-receipt/v1`, `stage: "compose"`,
    the `csd-metrics/v2` stamp applied once, and all eleven Table 7 groups present.
    """
    batch = toy_batch(white_matter, toy_scope)
    result = trainer.run([batch], [batch], steps=3)
    path = trainer.write_receipt(result, identity_for(white_matter), tmp_path)
    on_disk = json.loads(path.read_text())

    assert on_disk["schema"] == "model-pipeline-receipt/v1"
    assert on_disk["stage"] == "compose"
    assert on_disk["metrics_schema"] == "csd-metrics/v2"
    assert path.read_text().count('"metrics_schema"') == 1
    assert "code_revision" in on_disk
    for group in (
        "frozen_set",
        "budgets",
        "composed_metric",
        "baselines",
        "attention_mass",
        "write_back_topology",
        "scheduler",
        "latency",
        "store",
        "verdicts",
        "placement_knobs",
    ):
        assert group in on_disk, f"Table 7 group {group!r} missing"

    floor = on_disk["budgets"]["collapse_floor"]
    assert floor["expression"] == "eta/R"
    assert floor["R"] == len(white_matter.participant_names)
    assert floor["value"] == pytest.approx(
        white_matter.config.floor_eta / len(white_matter.participant_names)
    )
    assert on_disk["attention_mass"]["collapsed_in_phase_A"] == []
    assert on_disk["frozen_set"]["R"] == len(white_matter.participant_names)
    assert on_disk["budgets"]["trainable_parameters"] == trainer.trainable_parameter_count()


def test_attention_mass_is_a_distribution_over_active_iterations_only(
    trainer: PhaseATrainer, white_matter: WhiteMatter, toy_scope: Any
) -> None:
    """`a[b, i, :]` sums to 1 on active iterations, so the per-region means must too.

    This is what makes the floor `eta/R` meaningful: if the zero-padded iterations past
    `halt_at` were folded into the mean, every region would drift toward the floor for a
    reason that has nothing to do with the run.
    """
    batch = toy_batch(white_matter, toy_scope)
    result = trainer.run([batch], [batch], steps=3)
    assert sum(result.mean_per_region.values()) == pytest.approx(1.0, abs=1e-4)
    assert set(result.mean_per_region) == set(white_matter.participant_names)
