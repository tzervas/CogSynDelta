"""X7 episode construction: the schema, and a constructed failure for every guard.

Every gate in `cogsyndelta.reserve.episodes` gets a case that makes it FIRE. This
programme has found four guards that were "correct in reasoning and wrong in scope,
reporting success they could not have detected the absence of"; a test suite that only
builds valid items would be a fifth.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from cogsyndelta.reserve.episodes import (
    EPISODE_SCHEMA,
    N_OPTIONS,
    REJECTION_REASONS,
    EpisodeRejectedError,
    FactDonor,
    ProbeSource,
    SplitKeyMissingError,
    build_episode,
    content_hash,
    format_value,
    generator_identity,
    reference_solver_rank1,
    source_row_split,
    split_key_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "x7_episodes_sample.jsonl"
K_SPLIT = b"test-key-not-the-real-one"

SPLIT_KEY = {"scheme": "test", "key_id": "0" * 16, "eval_permille": 100}
GENERATOR = generator_identity("0" * 64, {"test": True})


def donor(tag: str, value: float, *, pool: tuple[str, ...] = ()) -> FactDonor:
    """A turn-1 donor with a distinct source row and a distinct fact.

    Args:
        tag: Makes the source-row fingerprint unique.
        value: The fact this donor commits.
        pool: Extra source-row fingerprints in its retrieval pool.

    Returns:
        The donor.
    """
    return FactDonor(
        source_dataset="deepmind/aqua_rat",
        source_revision="pinned",
        row_index=sum(ord(c) for c in tag),
        pair_fingerprint=f"fp-{tag}",
        columns=("question", "rationale"),
        licence_tier="permissive",
        query=f"TURN 1 retrieval for {tag}",
        key_claim=f"passage resolves its query and arrives at {value:g}",
        fact_value=value,
        pool_fingerprints=pool,
        pool_row_indices=tuple(range(len(pool))),
    )


def probe(question: str = "Problem: how many geese fly north? Report X + Y.") -> ProbeSource:
    """A turn-2 probe whose own answer is 7.

    Args:
        question: The rendered probe text.

    Returns:
        The probe.
    """
    return ProbeSource(
        source_dataset="deepmind/aqua_rat",
        source_revision="pinned",
        row_index=999,
        pair_fingerprint="fp-probe",
        columns=("question", "rationale"),
        licence_tier="permissive",
        question=question,
        answer_value=7.0,
    )


def valid_episode(**overrides: object) -> dict[str, object]:
    """Build one well-formed episode.

    Args:
        **overrides: Passed through to ``build_episode``.

    Returns:
        The item.
    """
    kwargs: dict[str, object] = {
        "variant": "passage_claim",
        "split": "eval",
        "split_key": SPLIT_KEY,
        "generator": GENERATOR,
    }
    kwargs.update(overrides)
    return build_episode(
        donor("gold", 41),
        probe(),
        [donor("d1", 52), donor("d2", 63), donor("d3", 74), donor("d4", 85)],
        **kwargs,
    )


class TestSplitKey:
    """DEC-39: the split is keyed, and it fails closed without a key."""

    def test_missing_key_refuses_rather_than_falling_back(self) -> None:
        """No key must mean no split -- never a seeded or sequential fallback."""
        with pytest.raises(SplitKeyMissingError):
            source_row_split("fp-a", b"")
        with pytest.raises(SplitKeyMissingError):
            split_key_id(b"")

    def test_assignment_is_deterministic_and_key_dependent(self) -> None:
        """The same row under the same key lands on the same side, and only then."""
        rows = [f"fp-{i}" for i in range(400)]
        first = [source_row_split(r, K_SPLIT) for r in rows]
        assert first == [source_row_split(r, K_SPLIT) for r in rows]
        other = [source_row_split(r, b"a-different-key") for r in rows]
        assert first != other, "a different k_split must produce a different assignment"

    def test_eval_share_tracks_the_requested_permille(self) -> None:
        """A 10% request lands near 10% over a few thousand rows."""
        rows = [f"fp-{i}" for i in range(4000)]
        share = sum(source_row_split(r, K_SPLIT) == "eval" for r in rows) / len(rows)
        assert 0.07 < share < 0.13

    def test_key_id_does_not_reveal_the_key(self) -> None:
        """The published id is an HMAC under a public label, not a hash of the key."""
        import hashlib

        assert split_key_id(K_SPLIT) != hashlib.sha256(K_SPLIT).hexdigest()[:16]
        assert split_key_id(K_SPLIT) == split_key_id(K_SPLIT)


class TestGuardsFire:
    """Every construction gate, made to refuse."""

    def test_dec38_probe_row_reused_in_turn1(self) -> None:
        """W3 gate (ii): an episode whose two turns share a source row is rejected."""
        with pytest.raises(EpisodeRejectedError) as caught:
            build_episode(
                donor("gold", 41, pool=("fp-probe",)),
                probe(),
                [donor("d1", 52), donor("d2", 63), donor("d3", 74), donor("d4", 85)],
                variant="passage_claim",
                split="eval",
                split_key=SPLIT_KEY,
                generator=GENERATOR,
            )
        assert caught.value.reason == "dec38_shared_source_row"

    def test_dec38_donor_row_used_twice(self) -> None:
        """The same donor twice would put one source row on both sides of the control."""
        with pytest.raises(EpisodeRejectedError) as caught:
            build_episode(
                donor("gold", 41),
                probe(),
                [donor("d1", 52), donor("d1", 63), donor("d3", 74), donor("d4", 85)],
                variant="passage_claim",
                split="eval",
                split_key=SPLIT_KEY,
                generator=GENERATOR,
            )
        assert caught.value.reason == "dec38_shared_source_row"

    def test_a_substituted_fact_that_changes_nothing_is_rejected(self) -> None:
        """THE negative control: if the wrong turn 1 gives the same answer, refuse.

        This is §5.5(e) property 1 in its sharpest form. An episode whose counterfactual
        answer equals its gold answer PASSES its own negative control, which means the
        control cannot fail it, which means the item is not recall-dependent.
        """
        with pytest.raises(EpisodeRejectedError) as caught:
            build_episode(
                donor("gold", 41),
                probe(),
                [donor("d1", 41), donor("d2", 63), donor("d3", 74), donor("d4", 85)],
                variant="passage_claim",
                split="eval",
                split_key=SPLIT_KEY,
                generator=GENERATOR,
            )
        assert caught.value.reason == "fact_value_collision"

    def test_fact_present_in_turn2_is_rejected(self) -> None:
        """X7 requires the fact to be absent from turn 2's context window."""
        with pytest.raises(EpisodeRejectedError) as caught:
            build_episode(
                donor("gold", 41),
                probe("Problem: a train leaves at 41 past the hour. Report X + Y."),
                [donor("d1", 52), donor("d2", 63), donor("d3", 74), donor("d4", 85)],
                variant="passage_claim",
                split="eval",
                split_key=SPLIT_KEY,
                generator=GENERATOR,
            )
        assert caught.value.reason == "fact_in_probe_window"

    def test_answer_present_in_turn2_is_rejected(self) -> None:
        """An answer readable off turn 2 needs no recall at all."""
        with pytest.raises(EpisodeRejectedError) as caught:
            build_episode(
                donor("gold", 41),
                probe("Problem: 48 geese fly north; how many? Report X + Y."),
                [donor("d1", 52), donor("d2", 63), donor("d3", 74), donor("d4", 85)],
                variant="passage_claim",
                split="eval",
                split_key=SPLIT_KEY,
                generator=GENERATOR,
            )
        assert caught.value.reason == "answer_in_probe_window"

    def test_lexically_leaky_item_is_rejected_by_the_reference_solver(self) -> None:
        """A gold option a bag of words can find from a WRONG turn 1 is rejected.

        The substituted claim below carries the gold option's own digits, so BM25 ranks
        gold first on the wrong-turn-1 arm -- which is exactly the shortcut the screen
        exists to catch.
        """
        leaky = FactDonor(
            source_dataset="deepmind/aqua_rat",
            source_revision="pinned",
            row_index=1,
            pair_fingerprint="fp-leaky",
            columns=("question", "rationale"),
            licence_tier="permissive",
            query="TURN 1",
            key_claim="the passage mentions 48 repeatedly",
            fact_value=52,
        )
        with pytest.raises(EpisodeRejectedError) as caught:
            build_episode(
                donor("gold", 41),
                probe(),
                [leaky, donor("d2", 63), donor("d3", 74), donor("d4", 85)],
                variant="passage_claim",
                split="eval",
                split_key=SPLIT_KEY,
                generator=GENERATOR,
            )
        assert caught.value.reason == "negative_control_solved_with_wrong_turn1"

    def test_wrong_donor_count_refuses(self) -> None:
        """X7 is a five-way choice; anything else is a different shape."""
        with pytest.raises(ValueError, match="distractor donors"):
            build_episode(
                donor("gold", 41),
                probe(),
                [donor("d1", 52)],
                variant="passage_claim",
                split="eval",
                split_key=SPLIT_KEY,
                generator=GENERATOR,
            )

    def test_undeclared_rejection_reason_refuses(self) -> None:
        """A rejection the manifest cannot tally is a rejection nobody can audit."""
        with pytest.raises(ValueError, match="undeclared rejection reason"):
            raise EpisodeRejectedError("made_up_reason")


class TestReferenceSolver:
    """The pinned lexical stand-in for "answerable"."""

    def test_no_lexical_signal_is_not_a_choice(self) -> None:
        """Bare numeric options share no token with the probe, so the solver abstains."""
        assert reference_solver_rank1("a probe with words only", ["11", "22", "33"]) == -1

    def test_signal_is_reported(self) -> None:
        """When an option's token really is in the anchor, the solver names it."""
        assert reference_solver_rank1("the answer is 22 exactly", ["11", "22", "33"]) == 1

    def test_empty_option_set_refuses(self) -> None:
        """Ranking nothing is a bug, not an abstention."""
        with pytest.raises(ValueError, match="empty option set"):
            reference_solver_rank1("anything", [])


class TestWellFormedEpisode:
    """What a built item promises."""

    def test_shape_and_binding(self) -> None:
        """Two turns, five options, and the gold option is fact + probe answer."""
        item = valid_episode()
        assert item["schema"] == EPISODE_SCHEMA
        assert item["shape"] == "X7"
        assert item["chance"] == 1.0 / N_OPTIONS
        turns = item["episode"]["turns"]
        assert [t["role"] for t in turns] == ["commit", "probe"]
        options = turns[1]["options"]
        assert len(options) == N_OPTIONS
        expected = item["expected"]
        assert options[expected["option_index"]] == expected["value"]
        assert expected["value"] == format_value(41 + 7)

    def test_counterfactual_names_a_wrong_option(self) -> None:
        """The substituted turn 1 must select an option that is present and wrong."""
        item = valid_episode()
        counterfactual = item["counterfactual"]
        options = item["episode"]["turns"][1]["options"]
        index = counterfactual["expected_option_index"]
        assert index != item["expected"]["option_index"]
        assert options[index] == format_value(52 + 7)
        assert counterfactual["expected_outcome"] == "FAIL"

    def test_provenance_names_no_model(self) -> None:
        """W3's acceptance clause: no generating model, or one named with a licence."""
        item = valid_episode()
        assert item["generator"]["model"] == "none"
        assert item["generator"]["config_hash"]

    def test_every_source_row_is_fingerprinted(self) -> None:
        """DEC-38: the manifest carries the fingerprint of every row the item consumed."""
        item = valid_episode()
        rows = item["source_rows"]
        assert len(rows) == 2 + (N_OPTIONS - 1)
        assert all(row["pair_fingerprint"] for row in rows)
        turn0 = {r["pair_fingerprint"] for r in rows if r["turn"] == 0}
        turn1 = {r["pair_fingerprint"] for r in rows if r["turn"] == 1}
        assert not (turn0 & turn1)

    def test_turn1_pool_rows_are_recorded_too(self) -> None:
        """The pool is RENDERED into turn 1, so DEC-38 counts it as consumed."""
        item = build_episode(
            donor("gold", 41, pool=("fp-p1", "fp-p2", "fp-gold")),
            probe(),
            [donor("d1", 52), donor("d2", 63), donor("d3", 74), donor("d4", 85)],
            variant="passage_claim",
            split="eval",
            split_key=SPLIT_KEY,
            generator=GENERATOR,
        )
        pool_rows = [r for r in item["source_rows"] if r["role"] == "turn1_pool"]
        assert {r["pair_fingerprint"] for r in pool_rows} == {"fp-p1", "fp-p2"}

    def test_options_that_collide_only_after_the_shift_are_rejected(self) -> None:
        """Distinct facts do NOT imply distinct options, and this gate is what sees it.

        `format_value` rounds to four decimals, and rounding does not commute with
        addition. Facts 0.00004 and 0.00006 render "0" and "0.0001", so
        `fact_value_collision` passes them; with a probe answer of 0.00003 both options
        render "0.0001". This fires through the PUBLIC api -- `build_episode` accepts any
        float fact -- so the gate guards a live hazard, not a hypothetical one.
        """
        colliding = ProbeSource(
            source_dataset="deepmind/aqua_rat",
            source_revision="pinned",
            row_index=1234,
            pair_fingerprint="fp-probe",
            columns=("question", "rationale"),
            licence_tier="permissive",
            question="Problem: how many geese fly north? Report X + Y.",
            answer_value=0.00003,
        )
        with pytest.raises(EpisodeRejectedError) as caught:
            build_episode(
                donor("gold", 0.00004),
                colliding,
                [
                    donor("d1", 0.00006),
                    donor("d2", 63),
                    donor("d3", 74),
                    donor("d4", 85),
                ],
                variant="passage_claim",
                split="eval",
                split_key=SPLIT_KEY,
                generator=GENERATOR,
            )
        assert caught.value.reason == "option_value_collision"

    def test_the_facts_gate_really_does_pass_that_pair(self) -> None:
        """Proof the case above reaches the right gate, not merely *a* gate.

        Without this, the test above could be passing because `fact_value_collision`
        refused first, and `option_value_collision` would still be untested.
        """
        assert format_value(0.00004) != format_value(0.00006)
        assert format_value(0.00004 + 0.00003) == format_value(0.00006 + 0.00003)


class TestBindingInvariant:
    """The assumption that keeps `option_value_collision` quiet in the shipped build."""

    def test_integral_facts_make_the_shift_injective(self) -> None:
        """The builder admits only positive integers, and on those the shift is safe.

        This is the invariant `option_value_collision`'s zero tally rests on. If the
        builder's integer filter is ever relaxed, this test is what tells the next reader
        that the gate above stops being a formality and starts being load-bearing.
        """
        for fact_a in range(0, 120):
            for fact_b in range(fact_a + 1, 120):
                for answer in (0, 1, 7, 99):
                    assert format_value(fact_a + answer) != format_value(fact_b + answer)

    def test_non_integral_facts_break_it(self) -> None:
        """The same sweep over decimals does NOT hold -- which is why the gate exists."""
        grid = [i / 100000 for i in range(1, 12)]
        collisions = [
            (a, b, y)
            for a in grid
            for b in grid
            for y in grid
            if a < b
            and format_value(a) != format_value(b)
            and format_value(a + y) == format_value(b + y)
        ]
        assert collisions, "the hazard this gate guards must be demonstrable"

    def test_content_hash_tracks_content_not_bookkeeping(self) -> None:
        """Re-splitting an item is not source drift; changing a turn is."""
        item = valid_episode()
        assert item["content_hash"] == content_hash(item)
        resplit = valid_episode(split="train")
        assert resplit["content_hash"] == item["content_hash"]
        drifted = dict(item)
        drifted["expected"] = {"turn": 1, "option_index": 0, "value": "999"}
        assert content_hash(drifted) != item["content_hash"]

    def test_item_id_is_stable_across_builds(self) -> None:
        """The same rows must always produce the same item id."""
        assert valid_episode()["item_id"] == valid_episode()["item_id"]


class TestFixture:
    """The committed sample, checked against the same invariants as a fresh build."""

    def test_fixture_exists_and_is_wellformed(self) -> None:
        """A tracked sample keeps the schema honest without shipping the whole reserve."""
        assert FIXTURE.exists(), "run scripts/csd-build-x7-episodes.py to regenerate"
        items = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
        assert items
        for item in items:
            assert item["schema"] == EPISODE_SCHEMA
            assert item["shape"] == "X7"
            assert item["branch"] == "text-only-slip"
            assert item["admission"]["status"] == "pending"
            assert item["generator"]["model"] == "none"
            assert item["content_hash"] == content_hash(item)
            turns = item["episode"]["turns"]
            assert len(turns) == 2
            options = turns[1]["options"]
            assert len(options) == N_OPTIONS == len(set(options))
            fact = float(turns[0]["fact_value"])
            answer = float(turns[1]["probe_answer"])
            assert options[item["expected"]["option_index"]] == format_value(fact + answer)
            counterfactual = item["counterfactual"]
            substituted = float(counterfactual["substituted_turn1"]["fact_value"])
            assert substituted != fact
            assert options[counterfactual["expected_option_index"]] == format_value(
                substituted + answer
            )
            turn0 = {r["pair_fingerprint"] for r in item["source_rows"] if r["turn"] == 0}
            turn1 = {r["pair_fingerprint"] for r in item["source_rows"] if r["turn"] == 1}
            assert not (turn0 & turn1), "DEC-38: an episode's two turns share a source row"
            assert format_value(fact) not in turns[1]["question"]
            roles = {r["role"] for r in item["source_rows"]}
            assert roles == {"commit", "probe", "counterfactual_donor", "turn1_pool"}


def load_builder():
    """Import the builder script by path, the way this repo's script tests do.

    Returns:
        The module.
    """
    path = REPO_ROOT / "scripts" / "csd-build-x7-episodes.py"
    spec = importlib.util.spec_from_file_location("csd_build_x7_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestBuilderHelpers:
    """The corpus-facing half, exercised without a corpus."""

    def test_unknown_licence_refuses_rather_than_guessing(self) -> None:
        """Mirror metadata lies; a tier this generator has not been told is a refusal."""
        builder = load_builder()
        assert builder.licence_tier("mit (verified via HF dataset_info tags)") == "permissive"
        with pytest.raises(builder.BuildRefusedError, match="no policy tier"):
            builder.licence_tier("cc-by-nc-4.0")

    def test_aqua_gold_value_only_accepts_a_bare_quantity(self) -> None:
        """A "None of these" answer cannot be bound into an arithmetic join."""
        builder = load_builder()
        options = ["A)21", "B)21.5", "C)22", "D)22.5", "E)23"]
        assert builder.aqua_gold_value(options, "E") == 23.0
        assert builder.aqua_gold_value(["A)none of these", "B)2"], "A") is None
        assert builder.aqua_gold_value(options, "Z") is None

    def test_magnitude_bands_keep_options_comparable(self) -> None:
        """Facts drawn from one band cannot leave one option a visible outlier."""
        builder = load_builder()
        assert builder.magnitude_bucket(7) == 1
        assert builder.magnitude_bucket(9999) == 4
        assert builder.magnitude_bucket(10**9) == builder.MAX_BUCKET

    def test_missing_corpus_root_refuses(self) -> None:
        """No corpus means no build; there is deliberately no synthetic fallback."""
        builder = load_builder()
        with pytest.raises(builder.BuildRefusedError, match="not a directory"):
            builder.resolve_corpus_root(Path("/nonexistent/corpus/root"))

    def test_every_declared_rejection_reason_is_tallied(self) -> None:
        """The manifest's rejection block must cover every reason that can be raised."""
        assert len(set(REJECTION_REASONS)) == len(REJECTION_REASONS)
        for reason in REJECTION_REASONS:
            assert EpisodeRejectedError(reason).reason == reason
