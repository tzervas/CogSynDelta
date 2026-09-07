"""Row W3's episode harness, and its three gates constructed to fire.

WHAT IS ASSERTED HERE, AND WHY EACH ASSERTION HAS A RED TWIN. The W3 cell does not ask for
three properties to be claimed; it asks for three gates *"all constructed to fire"*. A green
assertion about a property nobody has broken is worth what the red one beside it is worth --
`MEMORY.md`'s "verify guards by making them fail", and the reason three CSD guards once turned
out to be structurally incapable of firing. So every gate below is paired:

  gate (i)   negative control        `test_negative_control_fails_the_item`
             CAN-FAIL                `test_negative_control_cannot_fire_when_the_donor_fact_collides`
  gate (ii)  DEC-38 disjointness     `test_dec38_holds_on_a_well_formed_episode`
             CAN-FAIL                `test_dec38_fires_when_the_turns_share_a_source_row`
  gate (iii) partition reset         `test_turn1_less_variant_still_fails_after_a_full_run`
             CAN-FAIL                `test_the_turn1_less_variant_starts_passing_when_the_reset_is_off`

and the ordering contract -- turn 1's write commits before turn 2 is scored -- gets the same
treatment: `test_turn1_commit_is_observable_before_turn_2` against
`test_an_uncommitted_turn1_leaves_turn2_with_nothing_to_read`, which swaps in a store whose
write is acknowledged but never lands. That control is the design's own sentence made
executable: *"a turn 2 scored against an uncommitted write is measuring nothing."*

THE ITEM FIXTURES ARE SYNTHETIC, ON PURPOSE. `/akula-data/csd/reserve/x7/` is not on a CI
runner, so the gates run against hand-built records in the landed `csd-x7-episode/v1` shape.
One mount-gated test parses the real shard, so the synthetic shape cannot drift away from the
artefact without something going red.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from cogsyndelta.eval.episode_harness import (
    ABSTAIN,
    Dec38ViolationError,
    EpisodeArm,
    EpisodeHarness,
    HarnessContractError,
    HashingTurnEncoder,
    StoreProbe,
    assert_disjoint_source_rows,
    harness_receipt,
    load_episodes,
    parse_episode,
    verify_negative_control,
    verify_partition_reset,
)
from cogsyndelta.interconnect.episodic_store import (
    InMemoryStoreStub,
    Scope,
    WriteReceipt,
    derive_scope,
)
from cogsyndelta.interconnect.mind import InterconnectConfig, ParticipantSpec, WhiteMatter
from cogsyndelta.reserve.episodes import EPISODE_SCHEMA
from tests.interconnect.conftest import FakeText

X7_EVAL_SHARD = Path("/akula-data/csd/reserve/x7/x7-eval.jsonl")
DOMAIN = "episode"
D_W = 64
N_OPTIONS = 5


# ---------------------------------------------------------------------------------------
# Fixtures: an X7-shaped item, and a two-text-faculty mind with the store admitted.
# ---------------------------------------------------------------------------------------


def make_record(
    *,
    item_id: str = "item-0000",
    fact: int = 33,
    probe_answer: int = 216,
    donor_facts: tuple[int, ...] = (50, 60, 90, 11),
    substituted_index: int = 0,
    probe_fingerprint: str = "probe-row",
) -> dict[str, Any]:
    """Build one record in the landed `csd-x7-episode/v1` shape.

    Options are `donor_fact + probe_answer` for the episode's own fact and each donor's, in
    ascending numeric order -- the builder's own construction, restated here so a synthetic
    item binds exactly the way a landed one does.

    Args:
        item_id: The item's id.
        fact: The quantity turn 1 commits.
        probe_answer: Turn 2's own answer, `Y`.
        donor_facts: The four unrelated turn-1 facts supplying the wrong options.
        substituted_index: Which donor the negative control substitutes.
        probe_fingerprint: Turn 2's source-row fingerprint; overridden to construct a
            DEC-38 violation.

    Returns:
        A decoded X7 record.
    """
    facts = [fact, *donor_facts]
    options = sorted({f + probe_answer for f in facts})
    substituted = donor_facts[substituted_index]
    return {
        "schema": EPISODE_SCHEMA,
        "item_id": item_id,
        "shape": "X7",
        "variant": "passage_claim",
        "bin": "episodic_store x memory",
        "split": "eval",
        "chance": 1 / N_OPTIONS,
        "episode": {
            "turns": [
                {
                    "index": 0,
                    "role": "commit",
                    "query": f"TURN 1 ({item_id}). The passage resolves to {fact}.",
                    "fact_value": str(fact),
                    "key_claim": f"passage resolves its query and arrives at {fact}",
                },
                {
                    "index": 1,
                    "role": "probe",
                    "question": (
                        f"TURN 2 ({item_id}). Let X be the quantity you committed. "
                        f"The probe's own answer is {probe_answer}. Report X + Y."
                    ),
                    "probe_answer": str(probe_answer),
                    "options": [str(value) for value in options],
                    "binding": "expected = turn0.fact_value + turn1.probe_answer",
                },
            ]
        },
        "expected": {"turn": 1, "option_index": options.index(fact + probe_answer)},
        "counterfactual": {
            "donor_fact_values": [str(value) for value in donor_facts],
            "expected_option_index": options.index(substituted + probe_answer),
            "expected_outcome": "FAIL",
            "substituted_turn1": {
                "fact_value": str(substituted),
                "key_claim": f"passage resolves its query and arrives at {substituted}",
                "pair_fingerprint": f"{item_id}-donor-{substituted_index}",
            },
            "why_it_must_fail": "the substituted fact selects a different option",
        },
        "source_rows": [
            {"turn": 0, "role": "commit", "pair_fingerprint": f"{item_id}-commit"},
            {"turn": 1, "role": "probe", "pair_fingerprint": probe_fingerprint},
            *[
                {
                    "turn": 0,
                    "role": "counterfactual_donor",
                    "pair_fingerprint": f"{item_id}-donor-{index}",
                }
                for index in range(len(donor_facts))
            ],
        ],
    }


@pytest.fixture
def item():
    """One well-formed episode."""
    return parse_episode(make_record())


@pytest.fixture
def episode_config() -> InterconnectConfig:
    """Two TEXT participants plus the store, and `k_candidates` sized for five options.

    `language` and `memory` are X7's own two ablation-pair partners (`episodic_store x
    memory`, `episodic_store x language`), so the toy mind has the participant set the shape
    was written for rather than `conftest`'s language/visual pair. `k_candidates = 6` is
    `NULL` plus one slot per option; the harness refuses any other value, which is what makes
    a mis-sized mind a loud failure instead of a silent truncation.
    """
    spec = ParticipantSpec(ctx_min=2, ctx_max=8, token_budget_min=2, token_budget_max=8, phi=1.0)
    return InterconnectConfig(
        participants={
            "language": spec,
            "memory": spec,
            "episodic_store": ParticipantSpec(
                ctx_min=None, ctx_max=None, token_budget_min=2, token_budget_max=8, phi=0.0
            ),
        },
        workspace_dim=D_W,
        latents=8,
        n_iter=2,
        heads=4,
        mlp_ratio=2,
        budget_total_read_tokens=16,
        budget_total_kv_bytes=100_000,
        floor_eta=0.15,
        n_cond=4,
        controller_dim=32,
        controller_depth=1,
        controller_heads=2,
        write_back=True,
        k_candidates=N_OPTIONS + 1,
        rank_temperature=1.0,
        allowed_modalities=("text",),
        resident_heads=("text",),
    )


def build_harness(
    config: InterconnectConfig,
    *,
    store=None,
    reset_partitions: bool = True,
) -> tuple[EpisodeHarness, StoreProbe]:
    """Assemble a mind, a probe and a harness over them.

    Args:
        config: The interconnect configuration.
        store: The store to wrap; a fresh `InMemoryStoreStub` when omitted.
        reset_partitions: Passed through to the harness -- `False` is gate (iii)'s can-fail
            control.

    Returns:
        `(harness, probe)`.
    """
    torch.manual_seed(0)
    faculties = {"language": FakeText(), "memory": FakeText()}
    for faculty in faculties.values():
        faculty.name = "text"
        for parameter in faculty.parameters():
            parameter.requires_grad_(False)
    probe = StoreProbe(
        store
        if store is not None
        else InMemoryStoreStub(domain_enum={DOMAIN}, half_life_s=3600.0, importance_default=0.5)
    )
    mind = WhiteMatter(config, faculties, probe)
    encoder = HashingTurnEncoder({"language": 64, "memory": 64}, seq_len=8, workspace_dim=D_W)
    harness = EpisodeHarness(mind, probe, encoder, domain=DOMAIN, reset_partitions=reset_partitions)
    return harness, probe


class NeverCommitsStore(InMemoryStoreStub):
    """A store that ACKS every write and lands none of them.

    Not a hypothetical: it is the shape of every fire-and-forget write path, and it is exactly
    what the W3 cell's ordering requirement forbids. `WriteReceipt` is frozen with
    `committed: Literal[True]`, so a caller that trusts the receipt alone cannot tell this
    store apart from a working one -- which is why the harness reads the partition back.
    """

    def write(
        self,
        scope: Scope | None,
        domain: str,
        logical_key: str,
        latent: Tensor,
        *,
        importance: float | None = None,
        provenance: str | None = None,
    ) -> WriteReceipt:
        """Return a receipt without storing anything.

        Args:
            scope: Server-derived scope.
            domain: One value of the domain enum.
            logical_key: Record identity.
            latent: The latent that will not be stored.
            importance: Ignored.
            provenance: Ignored.

        Returns:
            A `WriteReceipt` that is a lie about durability, and true about nothing else.
        """
        return WriteReceipt(
            scope=scope,
            domain=domain,
            logical_key=logical_key,
            importance=self.importance_default if importance is None else importance,
            written_at=0.0,
        )


# ---------------------------------------------------------------------------------------
# The ordering contract: turn 1 commits BEFORE turn 2 is scored.
# ---------------------------------------------------------------------------------------


def test_turn1_commit_is_observable_before_turn_2(item, episode_config) -> None:
    """W3's headline clause, checked at the partition rather than at the receipt.

    The receipt says `committed=True` by construction, so the receipt alone proves nothing
    about durability. What is asserted here is that a read of the partition -- performed after
    turn 1 returns and before turn 2 runs -- finds the record.
    """
    harness, _probe = build_harness(episode_config)
    result = harness.run_episode(item, EpisodeArm.FULL)

    assert result.turn1_committed is True
    assert result.turn1_logical_key == f"{item.item_id}#t0"
    assert result.resident_before_turn2 == 1
    assert result.commit_observed_before_turn2 is True


def test_an_uncommitted_turn1_leaves_turn2_with_nothing_to_read(item, episode_config) -> None:
    """The can-fail twin: acknowledge the write, land nothing, and the episode collapses.

    Every symptom the design predicts appears at once -- an empty partition before turn 2, a
    read that returns no record, an abstaining oracle, and a receipt whose
    `store_is_load_bearing` is `False`. This is what a green harness over a dead store looks
    like from the inside, and it is why `read_returned_turn1_record` is a reported field.
    """
    store = NeverCommitsStore(domain_enum={DOMAIN}, half_life_s=3600.0, importance_default=0.5)
    harness, _probe = build_harness(episode_config, store=store)
    result = harness.run_episode(item, EpisodeArm.FULL)

    assert result.turn1_committed is True, "the receipt still claims a commit"
    assert result.resident_before_turn2 == 0
    assert result.commit_observed_before_turn2 is False
    assert result.read_returned_turn1_record is False
    assert result.oracle_choice == ABSTAIN
    assert result.oracle_pass is False
    receipt = harness_receipt([result])
    assert receipt["store"]["store_is_load_bearing"] is False


# ---------------------------------------------------------------------------------------
# The instrument: turn 2's read really did return turn 1's record.
# ---------------------------------------------------------------------------------------


def test_turn2_read_returns_turn1_record_at_rank_zero(item, episode_config) -> None:
    """The one measurement that keeps every other number honest.

    Not "the store was called" and not "the bank was non-zero": the latent turn 2's step-8
    read came back with is bit-equal to the latent turn 1 wrote, and it came back FIRST. The
    cosine is reported beside the identity so a near-miss cannot be mistaken for a hit.
    """
    harness, _probe = build_harness(episode_config)
    result = harness.run_episode(item, EpisodeArm.FULL)

    assert result.read_returned_turn1_record is True
    assert result.read_rank_of_turn1_record == 0
    assert result.read_top1_cosine_to_turn1 == pytest.approx(1.0, abs=1e-5)
    assert result.store_read_records == 1
    assert result.oracle_choice == item.gold_index
    assert result.oracle_pass is True


def test_a_colliding_latent_is_attributed_to_this_episode_not_another(episode_config) -> None:
    """Two identical turn-1 latents must not make one episode claim another's record.

    MEASURED, 2026-09-07: the `wrong_turn1` arm stages the donor's `key_claim` as its text and
    donors recur across items, so 466 of 2,048 runs committed a latent bit-identical to
    another episode's. The verdict was unaffected -- identical inputs commit identical facts --
    but `read_record_logical_key` named the wrong episode, which is the very field a reader
    would use to check the harness's own claim.

    Constructed here with two items whose turn-1 text is character-identical, run in order.
    The second episode must attribute its read to its OWN key.
    """
    shared = "TURN 1 (shared). The passage resolves to 33."
    first = parse_episode(make_record(item_id="first", fact=33))
    second = parse_episode(make_record(item_id="second", fact=33, probe_answer=100))
    first = dataclasses.replace(first, turn1_text=shared)
    second = dataclasses.replace(second, turn1_text=shared)

    harness, _probe = build_harness(episode_config)
    results = harness.run([first, second], [EpisodeArm.FULL])

    assert results[0].read_record_logical_key == "first#t0"
    assert results[1].read_record_logical_key == "second#t0", (
        "the second episode was credited with the first episode's colliding record"
    )
    assert all(row.oracle_pass for row in results)


def test_the_step8_read_is_the_one_the_forward_pass_made(item, episode_config) -> None:
    """The evidence is the call, not a re-derivation of it.

    `forward` reads the store twice per item: `_raw_summary`'s occupancy proxy
    (`global_query=True`, one slot) and step 8's domain-scoped bank. The probe sees both, the
    harness reports on the second, and this test pins that separation -- if a future change
    makes step 8 a global query, the harness must fail loudly rather than start reporting on
    the occupancy read.
    """
    harness, probe = build_harness(episode_config)
    harness.run_episode(item, EpisodeArm.FULL)

    scoped = [read for read in probe.reads if not read.global_query]
    occupancy = [read for read in probe.reads if read.global_query]
    assert len(scoped) == 1
    assert scoped[0].domain == DOMAIN
    assert scoped[0].had_query is True, "PR #86's query-dependent read is what makes it vary"
    assert len(occupancy) == 1
    assert occupancy[0].b_store == 1


def test_a_batched_turn_is_refused_rather_than_reported_on(item, episode_config) -> None:
    """Read provenance at `B > 1` would be a reconstruction, so `B` is pinned to 1.

    Recorded ambiguity 4 in the module docstring: batching is legal for `mind.py` -- it takes
    per-item `scopes` and `logical_keys` precisely so a batch of episodes works -- and is W5's
    optimisation to make. This harness sends exactly one `logical_key` per turn, so an encoder
    that emits a wider batch is refused by `mind.py`'s own length check rather than silently
    broadcast, which is the same guard that closed the write-key collision.
    """

    class TwoItemEncoder(HashingTurnEncoder):
        """An encoder that emits `B = 2`, to construct the refusal."""

        def encode_turn(self, text: str) -> dict[str, Tensor]:
            """Emit a two-row batch.

            Args:
                text: The turn's document.

            Returns:
                `{faculty: [2, seq_len] int64}`.
            """
            single = super().encode_turn(text)
            return {name: value.repeat(2, 1) for name, value in single.items()}

    harness, _probe = build_harness(episode_config)
    harness.encoder = TwoItemEncoder({"language": 64, "memory": 64}, seq_len=8, workspace_dim=D_W)
    with pytest.raises(ValueError, match=r"logical_keys.* has 1 entries, expected 2"):
        harness.run_episode(item, EpisodeArm.FULL)


def test_a_second_store_reader_on_the_forward_path_is_refused(item, episode_config) -> None:
    """The harness reports on ONE identifiable read, or it reports nothing.

    `forward` reads the store twice per item today -- `_raw_summary`'s occupancy proxy
    (`global_query=True`) and step 8's domain-scoped bank -- and `global_query` separates
    them. Construct the case that separation cannot survive: a second component on the forward
    path that also reads the partition. The harness must refuse rather than pick one of the
    two arbitrarily and call it the read it is reporting on.
    """

    class DoubleReadingMind(WhiteMatter):
        """A mind with a second domain-scoped store reader wired into its forward pass."""

        def forward(self, inputs, schedule=None):
            """Read the partition once more, then run the real pass.

            Args:
                inputs: The request mapping.
                schedule: An optional frozen schedule.

            Returns:
                The real pass's output.
            """
            self.store.read(inputs["scope"], domain=inputs["domain"], b_store=1)
            return super().forward(inputs, schedule)

    torch.manual_seed(0)
    faculties = {"language": FakeText(), "memory": FakeText()}
    probe = StoreProbe(
        InMemoryStoreStub(domain_enum={DOMAIN}, half_life_s=3600.0, importance_default=0.5)
    )
    mind = DoubleReadingMind(episode_config, faculties, probe)
    encoder = HashingTurnEncoder({"language": 64, "memory": 64}, seq_len=8, workspace_dim=D_W)
    harness = EpisodeHarness(mind, probe, encoder, domain=DOMAIN)
    with pytest.raises(HarnessContractError, match="exactly one domain-scoped store read"):
        harness.run_episode(item, EpisodeArm.FULL)


def test_a_mind_without_a_store_is_refused(episode_config) -> None:
    """A store-free mind would report a clean sweep of abstentions that looks like a leak.

    Refusing at construction is what keeps "every episode abstained" a diagnosis of the store
    rather than of the configuration.
    """
    config = InterconnectConfig(
        **{
            **{
                field: getattr(episode_config, field)
                for field in episode_config.__dataclass_fields__
            },
            "participants": {
                name: spec
                for name, spec in episode_config.participants.items()
                if name != "episodic_store"
            },
        }
    )
    faculties = {"language": FakeText(), "memory": FakeText()}
    mind = WhiteMatter(config, faculties, None)
    probe = StoreProbe(
        InMemoryStoreStub(domain_enum={DOMAIN}, half_life_s=3600.0, importance_default=0.5)
    )
    encoder = HashingTurnEncoder({"language": 64, "memory": 64}, seq_len=8, workspace_dim=D_W)
    with pytest.raises(HarnessContractError, match="declares no 'episodic_store' participant"):
        EpisodeHarness(mind, probe, encoder, domain=DOMAIN)


def test_an_item_wider_than_the_rank_head_is_refused(episode_config) -> None:
    """Six options against `k_candidates = 6` would silently drop one; it raises instead."""
    wide = parse_episode(make_record(donor_facts=(50, 60, 90, 11, 77)))
    harness, _probe = build_harness(episode_config)
    with pytest.raises(HarnessContractError, match="k_candidates = 7"):
        harness.run_episode(wide, EpisodeArm.FULL)


# ---------------------------------------------------------------------------------------
# Gate (i): the negative control on every episode.
# ---------------------------------------------------------------------------------------


def test_negative_control_fails_the_item(item, episode_config) -> None:
    """W3 gate (i) in its strong form: FAIL, and fail onto the declared wrong option.

    "Did not pass" is satisfied by an abstention, and an abstention is also what a dead store
    produces, so the weak form cannot distinguish a working control from a broken one. The
    substituted fact selects an option that is *present and wrong*, so the control's verdict
    is checkable to the index.
    """
    harness, _probe = build_harness(episode_config)
    result = harness.run_episode(item, EpisodeArm.WRONG_TURN1)

    assert result.read_returned_turn1_record is True, "the control must be READ to be a control"
    assert result.oracle_choice == item.counterfactual_index
    assert result.oracle_choice != item.gold_index
    assert result.oracle_pass is False
    assert verify_negative_control([result])["gate_fires"] is True


def test_negative_control_cannot_fire_when_the_donor_fact_collides(episode_config) -> None:
    """The can-fail twin, and it reconstructs a defect the builder already guards.

    `reserve.episodes`'s `fact_value_collision` rejects an episode whose counterfactual donor
    commits the SAME fact, because the substituted turn 1 then selects the SAME option and the
    control cannot fail the item. Build exactly that episode -- which is only possible here
    because the fixture bypasses `build_episode` -- and the control arm PASSES, and
    `verify_negative_control` refuses to call the gate fired.
    """
    colliding = parse_episode(
        make_record(item_id="collide", fact=33, donor_facts=(33, 60, 90, 11, 77))
    )
    assert len(colliding.options) == N_OPTIONS, "five options, one of them shared by two facts"
    assert colliding.counterfactual_index == colliding.gold_index, "the fixture is the defect"

    harness, _probe = build_harness(episode_config)
    result = harness.run_episode(colliding, EpisodeArm.WRONG_TURN1)

    assert result.oracle_pass is True, "the control passed the item it was meant to fail"
    verdict = verify_negative_control([result])
    assert verdict["gate_fires"] is False
    assert verdict["passed_and_should_not_have"] == ["collide"]


def test_verify_negative_control_does_not_fire_on_an_empty_arm(item, episode_config) -> None:
    """A gate that passes vacuously is a gate that never ran.

    Handing `verify_negative_control` a run with no `WRONG_TURN1` rows must report
    `gate_fires: False` -- "nothing violated it" is not "it fired".
    """
    harness, _probe = build_harness(episode_config)
    result = harness.run_episode(item, EpisodeArm.FULL)
    assert verify_negative_control([result])["gate_fires"] is False


# ---------------------------------------------------------------------------------------
# Gate (ii): no episode's two turns share a source row (DEC-38).
# ---------------------------------------------------------------------------------------


def test_dec38_holds_on_a_well_formed_episode(item) -> None:
    """The green half: a landed-shape item's turn-1 and turn-2 rows are disjoint."""
    assert_disjoint_source_rows(item)


def test_dec38_fires_when_the_turns_share_a_source_row() -> None:
    """The red half: give turn 2 turn 1's own row and the episode is refused.

    An episode that leaks against itself cannot be placed on one side of the split, so it is
    unscorable however it was constructed -- which is why the harness asserts this against the
    ARTEFACT it reads rather than trusting the builder that wrote it.
    """
    leaking = parse_episode(make_record(item_id="leak", probe_fingerprint="leak-commit"))
    with pytest.raises(Dec38ViolationError, match="dec38_shared_source_row"):
        assert_disjoint_source_rows(leaking)


def test_dec38_fires_when_a_donor_row_is_reused_as_the_commit_row() -> None:
    """The second shape DEC-38 covers: one turn-1 row doing two jobs inside one episode."""
    record = make_record(item_id="dup")
    record["source_rows"][0]["pair_fingerprint"] = "dup-donor-0"
    with pytest.raises(Dec38ViolationError, match="as both its own commit"):
        assert_disjoint_source_rows(parse_episode(record))


def test_running_an_episode_asserts_dec38_before_it_scores_anything(episode_config) -> None:
    """Gate (ii) is asserted per item at run time, not only by a separate audit pass."""
    leaking = parse_episode(make_record(item_id="leak2", probe_fingerprint="leak2-commit"))
    harness, _probe = build_harness(episode_config)
    with pytest.raises(Dec38ViolationError):
        harness.run_episode(leaking, EpisodeArm.FULL)


# ---------------------------------------------------------------------------------------
# Gate (iii): the partition really resets between episodes.
# ---------------------------------------------------------------------------------------


def test_turn1_less_variant_still_fails_after_a_full_run(item, episode_config) -> None:
    """W3 gate (iii), run exactly as the cell words it.

    *"verified by re-running one episode and asserting its turn-1-less variant still fails."*
    The `FULL` arm runs first and commits; the `NO_TURN1` arm then runs the same item's turn 2
    alone. With the reset intact it starts against an empty partition, its read returns
    nothing, and it fails by abstention.
    """
    harness, _probe = build_harness(episode_config)
    results = harness.run([item], [EpisodeArm.FULL, EpisodeArm.NO_TURN1])
    full, without = results

    assert full.oracle_pass is True
    assert without.partition_empty_at_start is True
    assert without.resident_before_turn2 == 0
    assert without.store_read_records == 0
    assert without.read_returned_turn1_record is False
    assert without.oracle_choice == ABSTAIN
    assert without.oracle_pass is False
    assert verify_partition_reset(results)["gate_fires"] is True


def test_the_turn1_less_variant_starts_passing_when_the_reset_is_off(item, episode_config) -> None:
    """The can-fail twin, and it is the design's own prediction verbatim.

    *"If state leaks across episodes, that variant would start passing."* Pin every episode to
    one session, run the same two arms, and the turn-1-less variant reads the `FULL` arm's
    record and answers correctly -- a perfect score produced entirely by a leak.
    """
    harness, _probe = build_harness(episode_config, reset_partitions=False)
    results = harness.run([item], [EpisodeArm.FULL, EpisodeArm.NO_TURN1])
    full, without = results

    assert full.oracle_pass is True
    assert without.partition_empty_at_start is False
    assert without.turn1_logical_key is None, "this arm committed nothing of its own"
    assert without.read_returned_turn1_record is True, "it read the previous episode's memory"
    assert without.read_record_logical_key == f"{item.item_id}#t0"
    assert without.oracle_pass is True, "the leak makes a turn-1-less episode look correct"
    assert verify_partition_reset(results)["gate_fires"] is False


def test_two_episodes_cannot_read_each_others_memories(episode_config) -> None:
    """The reset stated as isolation rather than as a re-run.

    Episode B's turn 2 must not see episode A's commit. Constructed with two items whose facts
    differ, so a leak would be visible as B answering with A's fact rather than abstaining.
    """
    first = parse_episode(make_record(item_id="a", fact=33))
    second = parse_episode(make_record(item_id="b", fact=41))
    harness, _probe = build_harness(episode_config)
    results = harness.run([first, second], [EpisodeArm.FULL])

    assert all(row.partition_empty_at_start for row in results)
    assert all(row.store_read_records == 1 for row in results)
    assert all(row.oracle_pass for row in results)
    assert results[0].scope_session != results[1].scope_session


def test_a_dirty_partition_is_refused_rather_than_scored(item, episode_config) -> None:
    """If the reset is meant to hold and does not, the harness stops instead of reporting.

    Constructed by writing into the exact partition the next episode will derive, then running
    it. A harness that scored on regardless would be reporting another episode's memory as
    this episode's recall.
    """
    harness, probe = build_harness(episode_config)
    session = harness._session_for(item, EpisodeArm.FULL)
    probe.inner.write(
        derive_scope(harness.principal, session=session),
        DOMAIN,
        "squatter",
        torch.zeros(D_W),
    )
    harness._ordinal = iter([0])
    with pytest.raises(HarnessContractError, match="already holds 1 record"):
        harness.run_episode(item, EpisodeArm.FULL)


# ---------------------------------------------------------------------------------------
# The receipt.
# ---------------------------------------------------------------------------------------


def test_receipt_reports_every_gate_and_the_store_evidence(item, episode_config) -> None:
    """One run, three arms, and a receipt that cannot report accuracy without the evidence."""
    harness, _probe = build_harness(episode_config)
    results = harness.run([item], [EpisodeArm.FULL, EpisodeArm.WRONG_TURN1, EpisodeArm.NO_TURN1])
    receipt = harness_receipt(results, config={"omp_num_threads": "1"})

    assert receipt["config"] == {"omp_num_threads": "1"}
    assert set(receipt["arms"]) == {"full", "wrong_turn1", "no_turn1"}
    assert receipt["arms"]["full"]["read_returned_turn1_record"] == 1
    assert receipt["arms"]["full"]["read_rank_0"] == 1
    assert receipt["arms"]["no_turn1"]["oracle_abstain"] == 1
    assert receipt["gates"]["negative_control"]["gate_fires"] is True
    assert receipt["gates"]["partition_reset"]["gate_fires"] is True
    assert receipt["store"]["store_is_load_bearing"] is True
    assert len(receipt["episodes"]) == 3
    assert receipt["episode_rows"] == {"recorded": 3, "of": 3, "truncated": False}
    json.dumps(receipt)  # a receipt that cannot be written down is not a receipt


def test_truncating_the_row_listing_does_not_truncate_the_gates(item, episode_config) -> None:
    """A full-shard receipt bounds its ROW LISTING; it must not bound its evidence.

    `max_episode_rows` exists so a 3,072-run receipt is reviewable. If it also narrowed the
    gates, a truncated receipt could report a clean sweep over rows it kept while the rows it
    dropped failed -- so the summaries and both gates are asserted to be computed over the
    whole run, with `episode_rows` recording what was cut.
    """
    harness, _probe = build_harness(episode_config)
    results = harness.run([item], [EpisodeArm.FULL, EpisodeArm.WRONG_TURN1, EpisodeArm.NO_TURN1])
    receipt = harness_receipt(results, max_episode_rows=1)

    assert receipt["episode_rows"] == {"recorded": 1, "of": 3, "truncated": True}
    assert len(receipt["episodes"]) == 1
    assert receipt["gates"]["negative_control"]["episodes"] == 1
    assert receipt["gates"]["negative_control"]["gate_fires"] is True
    assert receipt["gates"]["partition_reset"]["episodes_total"] == 3
    assert receipt["gates"]["partition_reset"]["gate_fires"] is True
    assert sum(arm["episodes"] for arm in receipt["arms"].values()) == 3


def test_rank_head_choice_is_reported_and_is_not_the_gate(item, episode_config) -> None:
    """The model scorer is wired and is explicitly not evidence on untrained weights.

    It must produce a legal choice (an option index, or `ABSTAIN` for `NULL`) so W5 inherits a
    working seam; it must not be what any gate keys on. Asserted by checking that the gates
    fire on the `WRONG_TURN1` arm regardless of what the untrained head happened to say.
    """
    harness, _probe = build_harness(episode_config)
    result = harness.run_episode(item, EpisodeArm.WRONG_TURN1)

    assert ABSTAIN <= result.rank_head_choice < N_OPTIONS
    assert verify_negative_control([result])["gate_fires"] is True


# ---------------------------------------------------------------------------------------
# The landed artefact.
# ---------------------------------------------------------------------------------------


def test_parse_refuses_an_unknown_schema() -> None:
    """A schema this harness has not seen may bind its answer differently."""
    record = make_record()
    record["schema"] = "csd-x7-episode/v2"
    with pytest.raises(ValueError, match="is not"):
        parse_episode(record)


def test_parse_refuses_an_episode_that_is_not_two_turns() -> None:
    """ "Episode" means exactly two turns in §5.5(e); anything else is a different shape."""
    record = make_record()
    record["episode"]["turns"].append(dict(record["episode"]["turns"][1]))
    with pytest.raises(ValueError, match="exactly two turns"):
        parse_episode(record)


@pytest.mark.skipif(not X7_EVAL_SHARD.exists(), reason="the X7 reserve is not mounted here")
def test_the_landed_shard_parses_and_runs(episode_config) -> None:
    """The synthetic fixtures above cannot drift away from the real artefact unnoticed.

    Parses the head of the landed eval shard, asserts DEC-38 on every item it read, and runs
    all three arms of the first one end to end -- so a schema change in `csd-build-x7-episodes`
    turns this red rather than leaving the gates green against a shape nothing produces.
    """
    items = load_episodes(X7_EVAL_SHARD, limit=8)
    assert len(items) == 8
    for landed in items:
        assert_disjoint_source_rows(landed)
        assert len(landed.options) == N_OPTIONS
        assert landed.options[landed.gold_index] != landed.options[landed.counterfactual_index]

    harness, _probe = build_harness(episode_config)
    results = harness.run(items[:1], [EpisodeArm.FULL, EpisodeArm.WRONG_TURN1, EpisodeArm.NO_TURN1])
    assert verify_negative_control(results)["gate_fires"] is True
    assert verify_partition_reset(results)["gate_fires"] is True
    assert harness_receipt(results)["store"]["store_is_load_bearing"] is True
