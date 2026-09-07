"""X7 -- recall-dependent two-turn episodes, and the guards that reject a bad one.

WHAT AN X7 ITEM IS (design §5.4 shape table, §5.5(e))
A two-turn episode. **Turn 1** presents a document and asks a retrieval question whose
answer commits a fact to the episodic store. **Turn 2** presents a *different* document
and asks a question answerable only from turn 1's fact -- a fact that is not in turn 2's
context window and not recoverable from turn 2's own corpus. It is reported in the
``episodic_store x memory`` bin and makes ``{episodic_store, memory}`` and
``{episodic_store, language}`` load-bearing.

WHY THE NEGATIVE CONTROL IS PART OF THE ITEM, NOT PART OF THE EVAL
§5.5(e) property 1: *"An item whose turn 2 is answerable without turn 1 is not
recall-dependent, it is a single-turn item in two parts, and it is rejected at
construction, not discovered at W6."* So the counterfactual -- the identical turn 2 with
turn 1 replaced by an unrelated episode -- is a field of the item and a gate on it.

HOW THE CONTROL IS MADE TO FIRE BY CONSTRUCTION
Turn 2's answer is ``fact + probe_answer``, and **every one of the five options is
``some_donor_fact + probe_answer``**: one from the episode's own turn 1, four from
unrelated episodes. A reader that solves turn 2's problem knows ``probe_answer`` and
still cannot rank the options, because each is consistent with a different committed
fact. A reader handed the WRONG turn 1 lands on that donor's option, which is a wrong
answer -- the negative control fails the item, by arithmetic rather than by hope.
Chance is therefore exactly ``1/5`` and is printed on the item.

WHAT THIS MODULE CANNOT DECIDE, STATED SO IT IS NOT OVERCLAIMED
The sufficient form of the control is *"run turn 2 through the mind with a substituted
turn 1 and require a wrong answer"*, and that needs the episode harness (W3's other
half) and trained weights. What runs here is the **necessary** form: the structural
guarantee above, plus a screen with a pinned, model-free reference solver
(``cogsyndelta.eval.lexical``'s BM25, ``csd-lexical/v1``). An item this module admits can
still be rejected by the harness; an item it rejects can never be recall-dependent.

NO MODEL IS IN THE PROVENANCE CHAIN. Every field is a deterministic function of rows
already held and of the pinned generator source. ``generator.model`` is the explicit
string ``"none"``, per W3's acceptance clause.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cogsyndelta.eval import lexical

EPISODE_SCHEMA = "csd-x7-episode/v1"
"""Schema stamp on every item. Bump when a field changes meaning."""

LEDGER_SCHEMA = "csd-reserve-source-rows/v1"
"""Schema stamp on the source-row ledger DEC-38 asks the train-time guard to read."""

MANIFEST_SCHEMA = "csd-x7-manifest/v1"
"""Schema stamp on the build manifest."""

GENERATOR_NAME = "x7_recall_dependent_episode"
"""DEC-39 ledger generator id. One spelling, recorded on every item."""

SPLIT_KEY_SCHEME = "HMAC-SHA256(k_split, pair_fingerprint(source_row)) mod 1000"
"""DEC-39's keyed split assignment, at DEC-38's source-row granularity."""

SPLIT_KEY_ID_LABEL = b"csd-split-key-id/v1"
"""Public label HMAC'd under ``k_split`` to publish a key id without publishing the key.

A bare ``sha256(k_split)`` would be an offline-guessing oracle for a low-entropy key;
an HMAC under a fixed public label is not.
"""

N_OPTIONS = 5
"""Options per probe turn: the episode's own fact plus four unrelated donors' facts."""

CHANCE = 1.0 / N_OPTIONS
"""Chance level, printed on the item because §5.3 scores in chance-normalised units."""

DEFAULT_EVAL_PERMILLE = 100
"""Per-mille of source rows assigned to the eval side. 100 => a 10% eval split."""

REFERENCE_SOLVER = f"{lexical.SCORER_VERSION}:bm25"
"""The pinned, model-free stand-in for "answerable" used by the construction screen."""

REJECTION_REASONS = (
    "dec38_shared_source_row",
    "fact_value_collision",
    "option_value_collision",
    "fact_in_probe_window",
    "answer_in_probe_window",
    "negative_control_solved_without_turn1",
    "negative_control_solved_with_wrong_turn1",
)
"""Every reason ``build_episode`` may refuse an item. The build manifest tallies these.

They are listed here rather than raised anonymously so a run that rejects nothing can be
told apart from a run whose guards cannot fire -- the failure mode this programme has
found four times (design §5.3, "verify by making it fail").
"""


class SplitKeyMissingError(RuntimeError):
    """Raised when no ``k_split`` is available.

    DEC-39: *"no key => the loader refuses to build a split at all, rather than falling
    back to sequential or seeded assignment."* A generator that can choose which rows land
    in eval defeats every downstream gate, so this fails closed.
    """


class EpisodeRejectedError(ValueError):
    """Raised by :func:`build_episode` when a candidate episode fails a construction gate.

    Attributes:
        reason: One of :data:`REJECTION_REASONS`.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        """Record which gate refused, and why.

        Args:
            reason: One of :data:`REJECTION_REASONS`.
            detail: Human-readable specifics, for the build log.

        Raises:
            ValueError: If ``reason`` is not a declared rejection reason.
        """
        if reason not in REJECTION_REASONS:
            raise ValueError(f"undeclared rejection reason {reason!r}")
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True, slots=True)
class FactDonor:
    """A turn-1 candidate: a retrieval item whose answer commits one fact to the store.

    Attributes:
        source_dataset: Pinned dataset id, e.g. ``deepmind/aqua_rat``.
        source_revision: The revision the manifest pins (never a bare name, §5.6).
        row_index: Row ordinal within the pinned shard.
        pair_fingerprint: DEC-38 source-row key, ``pair_fingerprint`` over the two
            columns the corpus manifest declares.
        columns: The two declared columns the fingerprint was taken over.
        licence_tier: Policy tier this row's licence sits in.
        query: Turn 1's rendered document-plus-question.
        key_claim: What the retrieved record asserts -- the fact, in words.
        fact_value: The numeric quantity the key claim carries. Turn 2 binds to this.
        pool_fingerprints: Source-row fingerprints of turn 1's retrieval pool. The pool's
            rows are *rendered into turn 1's text*, so DEC-38 counts them as consumed:
            they go through the disjointness check and into the item's ``source_rows``.
        pool_row_indices: Row ordinals aligned with ``pool_fingerprints``.
        gold_pool_index: Which pool member is the one to retrieve.
    """

    source_dataset: str
    source_revision: str
    row_index: int
    pair_fingerprint: str
    columns: tuple[str, str]
    licence_tier: str
    query: str
    key_claim: str
    fact_value: float
    pool_fingerprints: tuple[str, ...] = ()
    pool_row_indices: tuple[int, ...] = ()
    gold_pool_index: int = 0


@dataclass(frozen=True, slots=True)
class ProbeSource:
    """Turn 2: an admitted multi-hop item whose answer is deferred behind turn 1's fact.

    Attributes:
        source_dataset: Pinned dataset id.
        source_revision: The revision the manifest pins.
        row_index: Row ordinal within the pinned shard.
        pair_fingerprint: DEC-38 source-row key.
        columns: The two declared columns the fingerprint was taken over.
        licence_tier: Policy tier this row's licence sits in.
        question: Turn 2's rendered document-plus-question, with the turn-1 fact deferred.
        answer_value: The probe's own answer -- the second hop, stated in turn 2's corpus.
    """

    source_dataset: str
    source_revision: str
    row_index: int
    pair_fingerprint: str
    columns: tuple[str, str]
    licence_tier: str
    question: str
    answer_value: float


def split_key_id(k_split: bytes) -> str:
    """Publishable identifier for a split key, which never reveals the key.

    Args:
        k_split: The split key.

    Returns:
        16 hex characters of ``HMAC(k_split, SPLIT_KEY_ID_LABEL)``.

    Raises:
        SplitKeyMissingError: If the key is empty.
    """
    if not k_split:
        raise SplitKeyMissingError("no k_split: refusing to identify an absent key")
    return hmac.new(k_split, SPLIT_KEY_ID_LABEL, hashlib.sha256).hexdigest()[:16]


def source_row_split(
    fingerprint: str, k_split: bytes, *, eval_permille: int = DEFAULT_EVAL_PERMILLE
) -> str:
    """Assign one SOURCE ROW to a split, keyed, so a generator cannot steer it.

    DEC-38 puts the split at source-row granularity; DEC-39 keys the assignment under a
    ``k_split`` held outside the repo. Both together are what make the ``<5``-point
    overfit gate mean anything: a generator that cannot choose which rows land in eval
    cannot defeat any downstream gate.

    Args:
        fingerprint: The source row's ``pair_fingerprint``.
        k_split: The split key, read from the vault at run time and never logged.
        eval_permille: Per-mille of rows assigned to eval.

    Returns:
        ``"train"`` or ``"eval"``.

    Raises:
        SplitKeyMissingError: If ``k_split`` is empty. *Fails closed* -- there is
            deliberately no seeded fallback.
    """
    if not k_split:
        raise SplitKeyMissingError(
            "no k_split available: refusing to assign a split. DEC-39 forbids a "
            "sequential or seeded fallback -- fetch the key with "
            "`secret exec K_SPLIT=akula/csd-k-split -- ...`."
        )
    digest = hmac.new(k_split, fingerprint.encode("utf-8"), hashlib.sha256).digest()
    bucket = int.from_bytes(digest[:8], "big") % 1000
    return "eval" if bucket < eval_permille else "train"


def generator_identity(
    source_sha256: str, config: Mapping[str, object], *, name: str = GENERATOR_NAME
) -> dict[str, object]:
    """The DEC-39 generator row: name, source SHA-256, config hash, and no model.

    Args:
        source_sha256: SHA-256 of the generator's own source, so a poisoned generator
            diff is at least *visible*; DEC-39 records the residual risk that a poisoner
            with write access to generator source defeats content hashing.
        config: The build configuration. Hashed canonically.
        name: Generator id.

    Returns:
        The identity block written onto every item and onto the manifest.
    """
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "name": name,
        "revision": source_sha256,
        "source_sha256": source_sha256,
        "config_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "model": "none",
        "licence": "none -- no generating model is in the provenance chain",
    }


def episode_content_payload(record: Mapping[str, object]) -> str:
    """Canonical, content-bearing projection of an item, for :func:`content_hash`.

    Only the parts a later re-derivation must reproduce byte-for-byte are included:
    the rendered turns, the options, the expected answer and the source-row keys. Split
    assignment and generator identity are deliberately excluded, so re-keying the split
    does not look like source drift.

    Args:
        record: A built item.

    Returns:
        Canonical JSON.
    """
    episode = record["episode"]
    assert isinstance(episode, dict)
    return json.dumps(
        {
            "schema": record["schema"],
            "variant": record["variant"],
            "turns": episode["turns"],
            "expected": record["expected"],
            "source_rows": record["source_rows"],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def content_hash(record: Mapping[str, object]) -> str:
    """SHA-256 over :func:`episode_content_payload`, so source drift is detectable.

    Args:
        record: A built item.

    Returns:
        64 hex characters.
    """
    return hashlib.sha256(episode_content_payload(record).encode("utf-8")).hexdigest()


def reference_solver_rank1(anchor: str, options: Sequence[str]) -> int:
    """Index the pinned lexical reference solver ranks first, or ``-1`` for no signal.

    The solver is ``cogsyndelta.eval.lexical``'s BM25 -- the same tokeniser, the same
    ``k1``/``b`` that the reason-region diagnosis and every text eval receipt use.
    Reusing it rather than writing a second scorer is the point: two spellings of "what a
    bag of words can do" that drift apart is how a guard ends up checking something other
    than what it claims to.

    **A tie is NOT a choice.** When no option shares a token with the anchor every score
    is zero, and ``argmax`` over a fixed jitter would then always name the same position
    -- which would reject roughly a fifth of all items for a reason that has nothing to do
    with the item. ``-1`` is returned instead, and the caller records "no lexical signal"
    rather than a spurious hit.

    Args:
        anchor: The reader's whole context -- probe text, plus whatever fact it was
            handed (or nothing, for the no-turn-1 arm).
        options: The candidate answers, in item order.

    Returns:
        Index into ``options``, or ``-1`` when the solver has no signal.

    Raises:
        ValueError: If ``options`` is empty.
    """
    if not options:
        raise ValueError("refusing to rank an empty option set")
    scores = lexical.score_bm25([lexical.tokenize(anchor)], [lexical.tokenize(o) for o in options])
    row = np.asarray(scores)[0]
    best = float(row.max())
    if best <= 0.0 or int((row >= best).sum()) > 1:
        return -1
    return int(np.argmax(row))


def format_value(value: float) -> str:
    """Render a bound numeric answer the one way the whole pipeline renders it.

    Args:
        value: The quantity.

    Returns:
        An integer-looking string when the value is integral, else 4 decimal places.
    """
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _all_source_fingerprints(donor: FactDonor) -> set[str]:
    """Every source row a donor turn consumes, its retrieval pool included.

    Args:
        donor: The turn-1 candidate.

    Returns:
        The set of source-row fingerprints.
    """
    return {donor.pair_fingerprint, *donor.pool_fingerprints}


def _check_dec38(gold: FactDonor, probe: ProbeSource, donors: Sequence[FactDonor]) -> None:
    """Assert no episode's two turns share a source row (W3 gate (ii), DEC-38).

    The check covers the counterfactual donors and turn 1's retrieval pool as well as the
    two headline rows: a probe row that also sits in turn 1's pool leaks the fact, and a
    donor row shared with turn 2 leaks the counterfactual.

    Args:
        gold: Turn 1.
        probe: Turn 2.
        donors: The unrelated turn-1 episodes supplying the wrong-fact options.

    Raises:
        EpisodeRejectedError: ``dec38_shared_source_row`` if any two turns share a row.
    """
    turn1 = _all_source_fingerprints(gold)
    for donor in donors:
        turn1 |= _all_source_fingerprints(donor)
    if probe.pair_fingerprint in turn1:
        raise EpisodeRejectedError(
            "dec38_shared_source_row",
            f"probe row {probe.pair_fingerprint} also appears in turn 1",
        )
    seen: set[str] = set()
    for donor in (gold, *donors):
        if donor.pair_fingerprint in seen:
            raise EpisodeRejectedError(
                "dec38_shared_source_row",
                f"donor row {donor.pair_fingerprint} used twice in one episode",
            )
        seen.add(donor.pair_fingerprint)


def _check_facts_distinct(gold: FactDonor, donors: Sequence[FactDonor]) -> None:
    """Reject when two donors commit the same fact, so the control cannot discriminate.

    If the substituted turn 1 commits the same value the real one did, the counterfactual
    answer equals the gold answer and the negative control is a no-op. That is exactly the
    "guard structurally incapable of firing" pattern, so it is a rejection.

    Args:
        gold: Turn 1.
        donors: The counterfactual donors.

    Raises:
        EpisodeRejectedError: ``fact_value_collision``.
    """
    values = [gold.fact_value, *(d.fact_value for d in donors)]
    if len({format_value(v) for v in values}) != len(values):
        raise EpisodeRejectedError(
            "fact_value_collision",
            f"donor facts are not pairwise distinct: {[format_value(v) for v in values]}",
        )


def _check_options_distinct(options: Sequence[float]) -> None:
    """Reject when two options RENDER identically, which distinct facts do not prevent.

    WHAT THIS GUARDS THAT ``fact_value_collision`` CANNOT. That gate inspects the facts
    **before** the shift; this one inspects the options **after** it, and the two are not
    the same question, because :func:`format_value` is **lossy**. It rounds to four
    decimals, and rounding does not commute with addition: ``0.00004`` and ``0.00006``
    render as ``"0"`` and ``"0.0001"`` -- distinct, so ``fact_value_collision`` passes --
    yet with ``probe_answer = 0.00003`` both options render ``"0.0001"``. A brute-force
    sweep of a 39x39x39 grid of small decimals finds 2,414 such triples, so this is a
    family and not a fluke.

    It matters because the rendered string **is** the item: options are stored rendered
    and the harness scores a string choice. Two options that render the same make the item
    unanswerable and make the negative control non-discriminating -- silently.

    WHY ITS TALLY IS NONETHELESS ZERO IN THE SHIPPED BUILD, which is a fact about the
    *builder*, not about this gate. ``csd-build-x7-episodes.py`` admits only positive
    integral facts and probe answers, and on integers ``format_value`` is exact, so the
    shift is injective there (0 collisions over a 400x400x200 sweep). That filter is an
    invariant this gate depends on, so ``tests/test_x7_episodes.py`` asserts it directly:
    if the filter is ever relaxed, the test names this gate as the thing that must catch
    the result.

    Args:
        options: The option values, in item order.

    Raises:
        EpisodeRejectedError: ``option_value_collision``.
    """
    rendered = [format_value(v) for v in options]
    if len(set(rendered)) != len(rendered):
        raise EpisodeRejectedError("option_value_collision", f"options collide: {rendered}")


def _check_fact_absent_from_probe(gold: FactDonor, probe: ProbeSource) -> None:
    """Assert turn 1's fact is not in turn 2's context window.

    X7's own words: the answer must depend on a fact *"not in turn 2's context window and
    not recoverable from turn 2's own corpus"*. The corpus half is `s_r`'s job at
    admission (§5.5(e) property 2, W2b); the window half is decidable here and is decided
    here.

    Args:
        gold: Turn 1.
        probe: Turn 2.

    Raises:
        EpisodeRejectedError: ``fact_in_probe_window``.
    """
    haystack = probe.question
    needle = format_value(gold.fact_value)
    if needle in haystack:
        raise EpisodeRejectedError(
            "fact_in_probe_window", f"fact {needle} appears verbatim in turn 2's window"
        )
    claim_tokens = set(lexical.tokenize(gold.key_claim))
    probe_tokens = set(lexical.tokenize(haystack))
    shared = {t for t in claim_tokens & probe_tokens if len(t) >= 6}
    if len(shared) >= 3:
        raise EpisodeRejectedError(
            "fact_in_probe_window",
            f"turn 2 shares {len(shared)} distinctive tokens with turn 1's key claim",
        )


def _check_answer_absent_from_probe(expected: float, probe: ProbeSource) -> None:
    """Assert the expected answer is not readable off turn 2's document.

    Args:
        expected: The gold value.
        probe: Turn 2.

    Raises:
        EpisodeRejectedError: ``answer_in_probe_window``.
    """
    if format_value(expected) in probe.question:
        raise EpisodeRejectedError(
            "answer_in_probe_window",
            f"the answer {format_value(expected)} appears verbatim in turn 2's window",
        )


def _check_negative_control(
    probe_text: str, option_texts: Sequence[str], gold_index: int, wrong_fact_text: str
) -> dict[str, object]:
    """Run the two construction-time arms of the negative control and refuse a pass.

    Arm A is turn 2 with **no** turn 1 at all; arm B is turn 2 with an **unrelated**
    episode's turn 1, which is the arm §5.5(e) property 1 names. In both, the reference
    solver must NOT land on the gold option: an item a bag of words can answer without the
    right fact is not recall-dependent whatever a trained model would do with it.

    A ``-1`` from either arm means the solver had no lexical signal at all, which is the
    designed case: every option is ``some_donor_fact + probe_answer``, so a bag of words
    has nothing to rank them by. The screen therefore catches residual artefacts -- a gold
    option whose digits happen to appear in the probe text or in the substituted claim --
    rather than doing the control's main work, which the option construction does.

    Args:
        probe_text: Turn 2's rendered text.
        option_texts: The rendered options, in item order.
        gold_index: Index of the gold option.
        wrong_fact_text: The substituted episode's key claim, as turn 1 would have
            committed it.

    Returns:
        What each arm chose, for the item's ``counterfactual.verified_by`` block.

    Raises:
        EpisodeRejectedError: ``negative_control_solved_without_turn1`` or
            ``negative_control_solved_with_wrong_turn1``.
    """
    no_turn1 = reference_solver_rank1(probe_text, option_texts)
    if no_turn1 == gold_index:
        raise EpisodeRejectedError(
            "negative_control_solved_without_turn1",
            f"{REFERENCE_SOLVER} ranks the gold option first from turn 2 alone",
        )
    wrong_turn1 = reference_solver_rank1(f"{wrong_fact_text}\n{probe_text}", option_texts)
    if wrong_turn1 == gold_index:
        raise EpisodeRejectedError(
            "negative_control_solved_with_wrong_turn1",
            f"{REFERENCE_SOLVER} ranks the gold option first from a substituted turn 1",
        )
    return {
        "reference_solver": REFERENCE_SOLVER,
        "arm_no_turn1_choice": no_turn1,
        "arm_wrong_turn1_choice": wrong_turn1,
        "gold_index": gold_index,
        "solver_found_gold": False,
    }


def build_episode(
    gold: FactDonor,
    probe: ProbeSource,
    distractor_donors: Sequence[FactDonor],
    *,
    variant: str,
    split: str,
    split_key: Mapping[str, object],
    generator: Mapping[str, object],
    branch: str = "text-only-slip",
) -> dict[str, object]:
    """Stage one X7 episode, or refuse it.

    Args:
        gold: The turn-1 retrieval item whose answer commits the fact.
        probe: The turn-2 multi-hop item whose answer is deferred behind that fact.
        distractor_donors: ``N_OPTIONS - 1`` unrelated episodes. Each supplies one wrong
            option *and* is a usable counterfactual turn 1, which is what makes the
            negative control fire arithmetically rather than by assertion.
        variant: ``"passage_claim"`` or ``"api_record"`` -- X7's two named fact types.
        split: ``"train"`` or ``"eval"``, already decided at source-row granularity.
        split_key: The recorded split-key block (scheme, key id, eval per-mille).
        generator: :func:`generator_identity`'s output.
        branch: The pre-committed §5.4 branch this item is built under.

    Returns:
        The item record.

    Raises:
        EpisodeRejectedError: If any construction gate refuses.
        ValueError: If the wrong number of distractor donors is supplied.
    """
    if len(distractor_donors) != N_OPTIONS - 1:
        raise ValueError(f"X7 needs exactly {N_OPTIONS - 1} distractor donors")

    _check_dec38(gold, probe, distractor_donors)
    _check_facts_distinct(gold, distractor_donors)

    expected_value = gold.fact_value + probe.answer_value
    option_values = [expected_value] + [
        d.fact_value + probe.answer_value for d in distractor_donors
    ]
    _check_options_distinct(option_values)
    _check_fact_absent_from_probe(gold, probe)
    _check_answer_absent_from_probe(expected_value, probe)

    item_id = hashlib.sha256(
        "\x00".join(
            [
                EPISODE_SCHEMA,
                variant,
                gold.pair_fingerprint,
                probe.pair_fingerprint,
                *(d.pair_fingerprint for d in distractor_donors),
            ]
        ).encode("utf-8")
    ).hexdigest()

    order = list(range(N_OPTIONS))
    # Reproducibility, not cryptography (ruff S311): the option order must be REPRODUCIBLE from the item id, so a reviewer can
    # rebuild the item and get the same gold index. A CSPRNG would defeat that; nothing
    # here is a secret, and the split -- the one thing an adversary would want to steer --
    # is keyed under `k_split` instead.
    random.Random(int(item_id[:16], 16)).shuffle(order)  # noqa: S311
    shuffled = [option_values[i] for i in order]
    option_texts = [format_value(v) for v in shuffled]
    gold_index = order.index(0)
    primary_counterfactual_index = order.index(1)

    verified = _check_negative_control(
        probe.question, option_texts, gold_index, distractor_donors[0].key_claim
    )

    source_rows = [
        {
            "turn": 0,
            "role": "commit",
            "dataset": gold.source_dataset,
            "revision": gold.source_revision,
            "row_index": gold.row_index,
            "pair_fingerprint": gold.pair_fingerprint,
            "columns": list(gold.columns),
            "licence_tier": gold.licence_tier,
        },
        {
            "turn": 1,
            "role": "probe",
            "dataset": probe.source_dataset,
            "revision": probe.source_revision,
            "row_index": probe.row_index,
            "pair_fingerprint": probe.pair_fingerprint,
            "columns": list(probe.columns),
            "licence_tier": probe.licence_tier,
        },
    ]
    source_rows += [
        {
            "turn": 0,
            "role": "turn1_pool",
            "dataset": gold.source_dataset,
            "revision": gold.source_revision,
            "row_index": index,
            "pair_fingerprint": fingerprint,
            "columns": list(gold.columns),
            "licence_tier": gold.licence_tier,
        }
        for fingerprint, index in zip(gold.pool_fingerprints, gold.pool_row_indices, strict=False)
        if fingerprint != gold.pair_fingerprint
    ]
    source_rows += [
        {
            "turn": 0,
            "role": "counterfactual_donor",
            "dataset": d.source_dataset,
            "revision": d.source_revision,
            "row_index": d.row_index,
            "pair_fingerprint": d.pair_fingerprint,
            "columns": list(d.columns),
            "licence_tier": d.licence_tier,
        }
        for d in distractor_donors
    ]

    record: dict[str, object] = {
        "schema": EPISODE_SCHEMA,
        "item_id": item_id,
        "shape": "X7",
        "variant": variant,
        "branch": branch,
        "bin": "episodic_store x memory",
        "ablation_pairs": ["episodic_store x memory", "episodic_store x language"],
        "chance": CHANCE,
        "split": split,
        "split_key": dict(split_key),
        "episode": {
            "turns": [
                {
                    "index": 0,
                    "role": "commit",
                    "query": gold.query,
                    "key_claim": gold.key_claim,
                    "fact_value": format_value(gold.fact_value),
                    "pool_size": len(gold.pool_fingerprints),
                    "gold_pool_index": gold.gold_pool_index,
                },
                {
                    "index": 1,
                    "role": "probe",
                    "question": probe.question,
                    "options": option_texts,
                    "probe_answer": format_value(probe.answer_value),
                    "binding": "expected = turn0.fact_value + turn1.probe_answer",
                },
            ]
        },
        "expected": {
            "turn": 1,
            "option_index": gold_index,
            "value": format_value(expected_value),
        },
        "counterfactual": {
            "substituted_turn1": {
                "pair_fingerprint": distractor_donors[0].pair_fingerprint,
                "key_claim": distractor_donors[0].key_claim,
                "fact_value": format_value(distractor_donors[0].fact_value),
            },
            "expected_option_index": primary_counterfactual_index,
            "expected_outcome": "FAIL",
            "why_it_must_fail": (
                "the substituted fact selects a different option, and that option is "
                "present and wrong"
            ),
            "donor_fact_values": [format_value(d.fact_value) for d in distractor_donors],
            "verified_by": verified,
        },
        "source_rows": source_rows,
        "generator": dict(generator),
    }
    record["content_hash"] = content_hash(record)
    return record
