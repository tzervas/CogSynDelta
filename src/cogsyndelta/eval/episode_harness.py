"""Row W3's episode harness: the thing that makes a two-turn eval item scorable.

WHAT THIS IS, IN THE DESIGN'S OWN WORDS (`docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md`,
row **W3**): *"The harness is a named deliverable and not an implied one: an eval item may be
a sequence of turns; turn 1's write must commit through the store's `learn()` and return a
`LearnReceipt` before turn 2 is scored; the partition is reset between episodes."*

WHY IT COULD NOT BE SKIPPED. Revision 2's objection to the episodic store was that *"a single
item eval forward pass has nothing to read"*. §5.5(e) records the verdict on that objection:
it is **correct**, and it *"is not answered by declaring a shape; it is answered by building
the harness."* Every other reserve shape is one forward pass. X7 is two, in order, with a
durable commit between them, and nothing in this repo could run that until this module.

WHAT AN EPISODE RUN IS HERE, STEP BY STEP.

  1. A **fresh partition**. `derive_scope(principal, session=...)` with a session nobody else
     uses. A fresh session is the only reset the store API offers -- there is no `clear`, and
     `stop()` is terminal -- so "reset" is spelled "never reuse a session", and the harness
     ASSERTS the partition is empty at episode start rather than trusting that spelling
     (`EpisodeResult.partition_empty_at_start`).
  2. **Turn 1**, one `WhiteMatter.forward`. The pass writes at step 13; the harness requires a
     `WriteReceipt` back, and `WriteReceipt.committed` is `Literal[True]` by construction --
     a receipt cannot exist in a non-committed state. On `EpisodicStoreImpl` that receipt is
     returned by `learn`, which commits through `StoreBackend.put` BEFORE returning, so on
     `SqliteBackend` the record is durable at that point and not merely queued.
  3. **The commit is observed, not assumed.** Before turn 2 runs, the harness reads the
     partition itself and requires turn 1's record to be resident and bit-equal to what the
     pass wrote (`EpisodeResult.commit_observed_before_turn2`).
  4. **Turn 2**, a second `WhiteMatter.forward` under the same scope and domain and a
     different `logical_key`. Its step-8 read is captured by `StoreProbe`.
  5. **The verdict**, plus the evidence for it.

THE FAILURE THIS MODULE EXISTS TO PREVENT. A harness that reports a green number while the
store contributes nothing is worse than no harness: `mean(a_store)` passed G29 on a store
whose mutual information with the target was exactly zero. So the store read is INSTRUMENTED,
not inferred. `StoreProbe` records every `read`/`write` the forward pass makes, and every
`EpisodeResult` carries `read_returned_turn1_record`, the rank it came back at, and its
cosine to the record turn 1 actually wrote. A run whose episodes score well while
`read_returned_turn1_record` is `False` is reporting a lie, and the receipt makes that
visible in one field rather than in an argument.

TWO SCORERS, AND WHY BOTH (they are not redundant, and only one of them is evidence today).

  - **`oracle_choice` -- the RECALL ORACLE.** Turn 2's answer is chosen from the record turn
    2's read ACTUALLY returned: the harness matches each unmasked read slot against the
    latents it watched turn 1 commit, takes the top-ranked match, looks up the fact that turn
    1 committed, and renders `fact + probe_answer` into one of the item's five options. It
    abstains (`-1`) when no returned slot is a record it watched being written. This is a
    HARNESS INSTRUMENT, not a model: its accuracy is exactly the fidelity of the store path
    and is `0` when the store is not read. That is precisely what makes the three W3 gates
    fire deterministically today, before any region is retrained.
  - **`rank_head_choice` -- the MODEL.** `RankHead`'s argmax over the item's options, index 0
    dropped because it is `NULL`. This is the seam W5 scores through. **On untrained weights
    it is noise and is NOT evidence of anything**; it is reported so the receipt shows what
    the model said next to what the store supplied, and so W5 inherits a wired scorer rather
    than a TODO.

THE THREE W3 GATES, AND WHERE EACH ONE FIRES.

  (i)   **The negative control on every episode.** `EpisodeArm.WRONG_TURN1` re-runs the
        episode with turn 1 replaced by the item's own declared counterfactual donor, whose
        fact selects a DIFFERENT option that is present and wrong. The item must FAIL, and
        the harness checks the sharper form the item supports: the oracle must land exactly
        on `counterfactual.expected_option_index`. `verify_negative_control` returns the
        tally.
  (ii)  **No episode's two turns share a source row** (DEC-38). `assert_disjoint_source_rows`
        re-runs the check against the LANDED item's own `source_rows` block, not against the
        builder that produced it -- a construction-time guard cannot vouch for an artefact
        read off disk months later.
  (iii) **The partition really resets between episodes.** `EpisodeArm.NO_TURN1` runs turn 2
        alone; with the reset intact its read returns nothing, the oracle abstains, and the
        item fails. `EpisodeHarness(reset_partitions=False)` is the can-fail control: it
        pins every episode to one session, the earlier arm's record is still resident, and
        the turn-1-less variant starts PASSING. That switch is the whole point -- a reset
        nobody can break is a reset nobody has tested.

WHERE THE W3 CELL AND §5.5(e) LEFT A CHOICE, RECORDED RATHER THAN TAKEN SILENTLY.

  1. **`LearnReceipt` has no symbol in this repo.** The W3 cell and §5.5(e) both name one.
     `learn` (lifecycle verb 2/6, `interconnect/episodic/store.py`) returns `WriteReceipt`,
     and `WriteReceipt`'s own docstring records that it *is* memory-gate's `LearnReceipt`
     ported under the E0 protocol's name. Read as the same object; no symbol was invented.
  2. **"the partition is reset between episodes" does not say per ARM.** An arm is a
     re-staging of one episode, and two arms of one item sharing a partition would let the
     `FULL` arm's record satisfy the `NO_TURN1` arm -- which is the leak gate (iii) exists to
     catch. So the session key is per `(item, arm, run ordinal)`: strictly finer than the
     text requires, and the direction that cannot hide a leak.
  3. **The negative control's turn-1 TEXT is not in the item.** `counterfactual.
     substituted_turn1` carries `fact_value`, `key_claim` and `pair_fingerprint` -- no
     rendered query. The control is defined by the fact that gets committed, so the harness
     stages the donor's `key_claim` as turn 1's text (`substituted_turn1_text`) and commits
     `fact_value`. Recovering the donor's full rendered query would mean re-reading the
     pinned corpus shard at eval time, which no other scorer in this repo does.
  4. **This module is NOT re-exported from `cogsyndelta.eval.__init__`**, unlike its siblings.
     Importing it pulls the whole interconnect -- `WhiteMatter`, the workspace, the
     controller, the schedule validator -- into every `import cogsyndelta.eval`, and that
     package is imported today by lexical-baseline paths that need none of it. Same care
     `reserve/__init__`'s docstring takes over parquet, and the same reason
     `tests/test_import_hygiene.py` exists: import it as
     `cogsyndelta.eval.episode_harness`.
  5. **Batch size is one episode per forward pass.** Nothing forbids batching -- `mind.py`
     takes per-item `scopes` and `logical_keys` precisely so a batch of episodes works -- but
     with `B = 1` the step-8 read of each turn is unambiguously one read, so read provenance
     is an observation rather than a reconstruction. Batching is W5's optimisation to make.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from cogsyndelta.interconnect.episodic_store import (
    EpisodicStore,
    Scope,
    WriteReceipt,
    derive_scope,
)
from cogsyndelta.interconnect.mind import WhiteMatter
from cogsyndelta.reserve.episodes import EPISODE_SCHEMA, format_value

__all__ = [
    "ABSTAIN",
    "Dec38ViolationError",
    "EpisodeArm",
    "EpisodeHarness",
    "EpisodeItem",
    "EpisodeResult",
    "HarnessContractError",
    "HashingTurnEncoder",
    "StoreProbe",
    "StoreRead",
    "StoreWrite",
    "TurnEncoder",
    "assert_disjoint_source_rows",
    "harness_receipt",
    "load_episodes",
    "parse_episode",
    "verify_negative_control",
    "verify_partition_reset",
]

ABSTAIN = -1
"""The choice a scorer returns when it has nothing to choose from.

Not a sentinel borrowed from `RankHead`: index `0` there is the `NULL` candidate, a real
option the model can rank, whereas `ABSTAIN` means the scorer never got an input. Keeping
them distinct is what lets a receipt tell "the store returned nothing" apart from "the model
preferred `NULL`", and those are different failures.
"""

_LATENT_MATCH_ATOL = 1e-5
"""Tolerance for "this read slot IS the record turn 1 wrote".

Exact equality would be defensible on `InMemoryBackend`, which holds the tensor by value, but
`SqliteBackend` round-trips the latent through a float32 blob and `EpisodicStoreImpl.learn`
clones it. A tolerance keeps one identity test correct over both durability oracles, which is
the same reason the E1 conformance suite runs over both. It is TIGHT on purpose: two records
this close are the same write, not two similar memories -- the loose, graded question is what
`read_top1_cosine_to_turn1` reports separately.
"""


class HarnessContractError(RuntimeError):
    """The harness's own preconditions were violated, so nothing it reports would mean anything.

    Distinct from an item failing: a failed item is a RESULT, and this is the harness refusing
    to produce one. Raised when turn 1 comes back with no `WriteReceipt`, when a partition that
    must be empty is not, when the mind's candidate width cannot hold the item's options, or
    when the forward pass made no domain-scoped store read to observe.
    """


class Dec38ViolationError(ValueError):
    """W3 gate (ii): an episode's two turns share a source row under DEC-38.

    A ValueError rather than a harness error because the ITEM is wrong, not the run: an
    episode that leaks against itself cannot be placed on one side of the train/eval split, so
    it is unscorable wherever it is read from.
    """


class EpisodeArm(str, Enum):
    """The three stagings of one episode the W3 gates need.

    `str` mixin so an arm serialises into a receipt as its own name without a conversion at
    every call site -- the same choice `Lifecycle` makes in `interconnect/episodic/store.py`.
    """

    FULL = "full"
    """Turn 1 then turn 2, the episode as constructed. Expected to PASS once the regions can
    actually recall; today it is the arm that proves the store path carries the fact."""

    WRONG_TURN1 = "wrong_turn1"
    """Gate (i): turn 1 replaced by the item's declared counterfactual donor. The item MUST
    fail, and the oracle must land on `counterfactual.expected_option_index`."""

    NO_TURN1 = "no_turn1"
    """Gate (iii)'s instrument: turn 2 alone against a partition that should be empty. Fails
    by abstention when the reset holds; starts passing the moment it does not."""


@dataclass(frozen=True, slots=True)
class EpisodeItem:
    """One `csd-x7-episode/v1` record, reduced to what running it needs.

    The landed item carries provenance, admission and split blocks this harness does not
    score against; `source_rows` is kept because gate (ii) is asserted against the artefact
    itself, and `raw_split`/`raw_bin` because a receipt that cannot say which half and which
    ablation pair an episode came from cannot be reconciled with §5.4's arithmetic.
    """

    item_id: str
    variant: str
    ablation_bin: str
    split: str
    chance: float
    turn1_text: str
    turn1_fact: str
    turn2_text: str
    probe_answer: str
    options: tuple[str, ...]
    gold_index: int
    substituted_turn1_text: str
    substituted_turn1_fact: str
    counterfactual_index: int
    source_rows: tuple[Mapping[str, Any], ...]

    def fact_for(self, arm: EpisodeArm) -> str | None:
        """The fact this arm's turn 1 commits, or `None` when the arm runs no turn 1.

        Args:
            arm: Which staging is being run.

        Returns:
            The rendered fact value, or `None` for `NO_TURN1`.
        """
        if arm is EpisodeArm.FULL:
            return self.turn1_fact
        if arm is EpisodeArm.WRONG_TURN1:
            return self.substituted_turn1_fact
        return None

    def text_for(self, arm: EpisodeArm) -> str | None:
        """Turn 1's text for this arm, or `None` when the arm runs no turn 1.

        Args:
            arm: Which staging is being run.

        Returns:
            The turn-1 document, or `None` for `NO_TURN1`.
        """
        if arm is EpisodeArm.FULL:
            return self.turn1_text
        if arm is EpisodeArm.WRONG_TURN1:
            return self.substituted_turn1_text
        return None

    def option_index_for_fact(self, fact: str) -> int:
        """Which option a reader who recalled `fact` would choose, or `ABSTAIN`.

        The binding is the item's own: `expected = turn0.fact_value + turn1.probe_answer`,
        rendered by `format_value` -- reused from `reserve.episodes` rather than restated,
        because the options were rendered by that exact function and a second implementation
        of "how a number becomes a string" is a defect waiting for a fractional fact.

        Args:
            fact: The rendered fact value recalled from the store.

        Returns:
            The index in `options`, or `ABSTAIN` when the bound value is not among them.
        """
        bound = format_value(float(fact) + float(self.probe_answer))
        if bound not in self.options:
            return ABSTAIN
        return self.options.index(bound)


def parse_episode(record: Mapping[str, Any]) -> EpisodeItem:
    """Reduce one landed X7 JSON record to an `EpisodeItem`.

    Args:
        record: A decoded line of `x7-train.jsonl` / `x7-eval.jsonl`.

    Returns:
        The parsed item.

    Raises:
        ValueError: The record does not declare `EPISODE_SCHEMA`, or does not carry exactly
            two turns. Both are refusals rather than best-effort reads: a schema this harness
            has not seen may bind its answer differently, and "exactly two turns" is what the
            word "episode" means in §5.5(e).
    """
    schema = record.get("schema")
    if schema != EPISODE_SCHEMA:
        raise ValueError(f"episode schema {schema!r} is not {EPISODE_SCHEMA!r}")
    turns = record["episode"]["turns"]
    if len(turns) != 2:
        raise ValueError(f"an X7 episode has exactly two turns, got {len(turns)}")
    commit, probe = turns
    counterfactual = record["counterfactual"]
    substituted = counterfactual["substituted_turn1"]
    return EpisodeItem(
        item_id=record["item_id"],
        variant=record["variant"],
        ablation_bin=record["bin"],
        split=record["split"],
        chance=float(record["chance"]),
        turn1_text=commit["query"],
        turn1_fact=str(commit["fact_value"]),
        turn2_text=probe["question"],
        probe_answer=str(probe["probe_answer"]),
        options=tuple(str(option) for option in probe["options"]),
        gold_index=int(record["expected"]["option_index"]),
        substituted_turn1_text=(
            f"TURN 1 (retrieval, negative control). "
            f"The retrieved record asserts: {substituted['key_claim']}."
        ),
        substituted_turn1_fact=str(substituted["fact_value"]),
        counterfactual_index=int(counterfactual["expected_option_index"]),
        source_rows=tuple(dict(row) for row in record["source_rows"]),
    )


def load_episodes(path: str | Path, *, limit: int | None = None) -> list[EpisodeItem]:
    """Read a JSONL shard of X7 items.

    Args:
        path: `x7-eval.jsonl` or `x7-train.jsonl`.
        limit: Stop after this many items. `None` reads the shard.

    Returns:
        The parsed items, in file order -- the order is the shard's, so a truncated run is a
        prefix of the full one and two runs at the same `limit` see the same episodes.
    """
    items: list[EpisodeItem] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(items) >= limit:
                break
            stripped = line.strip()
            if stripped:
                items.append(parse_episode(json.loads(stripped)))
    return items


def assert_disjoint_source_rows(item: EpisodeItem) -> None:
    """W3 gate (ii): no episode's two turns share a source row (DEC-38).

    Asserted against the LANDED item's own `source_rows` block. `reserve.episodes` runs the
    equivalent check at construction, over the builder's in-memory donors; this one runs over
    the artefact an eval actually reads, which is the only version a scorer can vouch for.
    Turn 1's retrieval pool and the counterfactual donors count as turn-1 rows because §5.5(e)
    counts them: a pool row is rendered into turn 1's text, and a donor row supplies a wrong
    option, so either one shared with turn 2 leaks.

    Args:
        item: The episode to check.

    Raises:
        Dec38ViolationError: A fingerprint appears on both turns, or a turn-1 row is used
            twice inside one episode.
    """
    turn1 = [row for row in item.source_rows if int(row["turn"]) == 0]
    turn2 = [row for row in item.source_rows if int(row["turn"]) == 1]
    turn1_prints = {str(row["pair_fingerprint"]) for row in turn1}
    turn2_prints = {str(row["pair_fingerprint"]) for row in turn2}
    shared = turn1_prints & turn2_prints
    if shared:
        raise Dec38ViolationError(
            f"dec38_shared_source_row: episode {item.item_id} uses source row(s) "
            f"{sorted(shared)} on both turns; it leaks against itself and cannot be "
            "placed on one side of the split."
        )
    headline = [str(row["pair_fingerprint"]) for row in turn1 if row["role"] != "turn1_pool"]
    if len(headline) != len(set(headline)):
        raise Dec38ViolationError(
            f"dec38_shared_source_row: episode {item.item_id} uses one turn-1 source row "
            "as both its own commit and a counterfactual donor."
        )


@dataclass(frozen=True, slots=True)
class StoreRead:
    """One `EpisodicStore.read` the forward pass made, with what came back."""

    scope: Scope | None
    domain: str | None
    global_query: bool
    b_store: int
    had_query: bool
    latents: Tensor
    mask: Tensor


@dataclass(frozen=True, slots=True)
class StoreWrite:
    """One `EpisodicStore.write` the forward pass made, with the latent it committed."""

    scope: Scope | None
    domain: str
    logical_key: str
    latent: Tensor
    receipt: WriteReceipt


class StoreProbe:
    """An `EpisodicStore` that records what the forward pass asked it for.

    WHY A WRAPPER AND NOT A RE-DERIVATION. The alternative to observing the read is
    recomputing it: rebuild the query the way `WhiteMatter._store_query` does, call `read`
    again, and assume the two agree. That assumption is the exact thing under test -- it is
    how a harness ends up reporting on a read the model never made. Wrapping the store means
    the evidence is the call itself.

    Every method of the E0 protocol is delegated explicitly rather than through `__getattr__`,
    so this class structurally satisfies `EpisodicStore` for a type checker and so a method
    added to the protocol later fails here loudly instead of silently forwarding.
    """

    def __init__(self, store: EpisodicStore) -> None:
        """Wrap a store.

        Args:
            store: The store to observe. Kept as `inner`, so a caller can make its own
                instrumentation reads without polluting this probe's log -- which is what
                `EpisodeHarness` does for its partition-empty and commit-observed checks.
        """
        self.inner = store
        self.reads: list[StoreRead] = []
        self.writes: list[StoreWrite] = []

    def clear(self) -> None:
        """Drop the log. Called between turns so a turn's reads are exactly its own."""
        self.reads.clear()
        self.writes.clear()

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
        """Record and delegate one write.

        Args:
            scope: Server-derived scope, or `None`.
            domain: One value of the store's domain enum.
            logical_key: Episode identity within `(scope, domain)`.
            latent: `[D]` float tensor.
            importance: Per-record override.
            provenance: Sidecar text.

        Returns:
            The wrapped store's `WriteReceipt`.
        """
        receipt = self.inner.write(
            scope, domain, logical_key, latent, importance=importance, provenance=provenance
        )
        self.writes.append(
            StoreWrite(
                scope=scope,
                domain=domain,
                logical_key=logical_key,
                latent=latent.detach().clone(),
                receipt=receipt,
            )
        )
        return receipt

    def read(
        self,
        scope: Scope | None,
        *,
        domain: str | None = None,
        global_query: bool = False,
        b_store: int,
        query: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Record and delegate one read.

        Args:
            scope: Server-derived scope, or `None`.
            domain: One domain, or `None` with `global_query=True`.
            global_query: Read across every domain under `scope`.
            b_store: Slots to fill.
            query: `[D]` float, what the read is looking for.

        Returns:
            `(latents [b_store, D], mask [b_store])`.
        """
        latents, mask = self.inner.read(
            scope, domain=domain, global_query=global_query, b_store=b_store, query=query
        )
        self.reads.append(
            StoreRead(
                scope=scope,
                domain=domain,
                global_query=global_query,
                b_store=b_store,
                had_query=query is not None,
                latents=latents.detach().clone(),
                mask=mask.detach().clone(),
            )
        )
        return latents, mask

    def capacity_bytes(self, host: str) -> int:
        """Delegate DEC-63's capacity query.

        Args:
            host: Host label.

        Returns:
            The wrapped store's answer.
        """
        return self.inner.capacity_bytes(host)

    def evict(self) -> int:
        """Delegate DEC-63's scored eviction.

        Returns:
            The number of records the wrapped store evicted.
        """
        return self.inner.evict()

    def promote(self, scope: Scope, domain: str, logical_key: str) -> None:
        """Delegate DEC-65's tier promotion.

        Args:
            scope: Server-derived scope.
            domain: One value of the domain enum.
            logical_key: Record identity.
        """
        self.inner.promote(scope, domain, logical_key)

    def bind_session(self, scope: Scope) -> Scope:
        """Delegate DEC-64's session bounding.

        Args:
            scope: The scope to bound.

        Returns:
            The wrapped store's bounded scope.
        """
        return self.inner.bind_session(scope)

    def consolidate(self) -> None:
        """Delegate DEC-66's consolidation."""
        self.inner.consolidate()


class TurnEncoder(Protocol):
    """How one turn's text becomes this mind's per-faculty inputs.

    Deliberately a protocol and not a class. The regions X7 is written against
    (`language`, `memory`) have not been retrained -- W1's token-rank result made that
    mandatory and W7v's rebuild is corpus-blocked -- so there is no single correct tokeniser
    to hard-code here. Whatever encodes a turn today, the harness's contract with it is two
    methods, and W5 swaps in the real one without touching this file.
    """

    def encode_turn(self, text: str) -> dict[str, Tensor]:
        """Encode one turn.

        Args:
            text: The turn's rendered document-plus-question.

        Returns:
            `{faculty_name: tensor}` with a leading batch axis of 1, one entry per non-store
            participant the mind declares.
        """
        ...

    def encode_options(self, options: Sequence[str]) -> Tensor:
        """Embed an item's answer options as `RankHead` content candidates.

        Args:
            options: The item's rendered options, in item order.

        Returns:
            `[1, len(options), D_w]`.
        """
        ...


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One episode run, its verdict, and the evidence the verdict rests on.

    The evidence fields are not diagnostics bolted on afterwards. `read_returned_turn1_record`
    is the field that separates this harness from the one the design warns about, and
    `harness_receipt` refuses to report accuracy without it.
    """

    item_id: str
    arm: EpisodeArm
    variant: str
    ablation_bin: str
    split: str
    scope_session: str
    partition_empty_at_start: bool
    turn1_committed: bool
    turn1_logical_key: str | None
    commit_observed_before_turn2: bool
    resident_before_turn2: int
    store_read_slots: int
    store_read_records: int
    read_returned_turn1_record: bool
    read_rank_of_turn1_record: int
    read_top1_cosine_to_turn1: float
    read_record_logical_key: str | None
    oracle_choice: int
    oracle_pass: bool
    rank_head_choice: int
    rank_head_pass: bool
    gold_index: int
    counterfactual_index: int

    def as_dict(self) -> dict[str, Any]:
        """Flatten to JSON-safe primitives for a receipt.

        Returns:
            One receipt row.
        """
        return {
            "item_id": self.item_id,
            "arm": self.arm.value,
            "variant": self.variant,
            "bin": self.ablation_bin,
            "split": self.split,
            "scope_session": self.scope_session,
            "partition_empty_at_start": self.partition_empty_at_start,
            "turn1_committed": self.turn1_committed,
            "turn1_logical_key": self.turn1_logical_key,
            "commit_observed_before_turn2": self.commit_observed_before_turn2,
            "resident_before_turn2": self.resident_before_turn2,
            "store_read_slots": self.store_read_slots,
            "store_read_records": self.store_read_records,
            "read_returned_turn1_record": self.read_returned_turn1_record,
            "read_rank_of_turn1_record": self.read_rank_of_turn1_record,
            "read_top1_cosine_to_turn1": self.read_top1_cosine_to_turn1,
            "read_record_logical_key": self.read_record_logical_key,
            "oracle_choice": self.oracle_choice,
            "oracle_pass": self.oracle_pass,
            "rank_head_choice": self.rank_head_choice,
            "rank_head_pass": self.rank_head_pass,
            "gold_index": self.gold_index,
            "counterfactual_index": self.counterfactual_index,
        }


@dataclass(slots=True)
class CommittedRecord:
    """A latent the harness watched turn 1 write, and the fact it stands for.

    Absent from `__all__`, and NOT named with a leading underscore -- the quality gate's
    PascalCase rule rejects `_CommittedRecord`, the same way it rejected `episodic_store.py`'s
    internal `Record`. "Internal" is expressed by not exporting it, not by the name.

    The oracle needs an identity for a read slot, and `EpisodicStore.read` returns latents and
    a mask -- no keys. That is not a gap to route around: the read's contract is a bank of
    vectors for the workspace to attend over, and adding keys to it for a scorer's benefit
    would put eval bookkeeping on the forward path. The harness instead remembers what it
    watched being written, which it can do because it is the thing that ran turn 1.
    """

    latent: Tensor
    fact: str
    logical_key: str


@dataclass(slots=True)
class TurnRun:
    """What one `WhiteMatter.forward` produced that the harness cares about.

    Internal like `CommittedRecord` above, and unexported for the same reason and by the same
    means.
    """

    write_receipts: list[WriteReceipt] = field(default_factory=list)
    written_latents: list[Tensor] = field(default_factory=list)
    step8_read: StoreRead | None = None
    scores: Tensor | None = None


class EpisodeHarness:
    """Runs X7 episodes against a `WhiteMatter` and reports what the store actually did.

    Construction takes the mind and the probe SEPARATELY even though the mind holds the probe,
    because the harness needs both surfaces: the probe's log (what the pass read) and the
    probe's `inner` (its own reads, which must not appear in that log).
    """

    def __init__(
        self,
        mind: WhiteMatter,
        probe: StoreProbe,
        encoder: TurnEncoder,
        *,
        principal: str = "w3-episode-harness",
        domain: str = "episode",
        session_prefix: str = "x7",
        reset_partitions: bool = True,
    ) -> None:
        """Build a harness over an already-constructed mind.

        Args:
            mind: The `WhiteMatter` to run. Must declare `episodic_store` as a participant and
                hold `probe` as its store; both are checked, because a harness silently
                running a store-free mind would report a clean sweep of abstentions and look
                like a broken reset rather than a misconfiguration.
            probe: The `StoreProbe` wrapping the store `mind` was built with.
            encoder: Turns text into this mind's inputs.
            principal: The authenticated principal every scope is derived for. One principal
                is correct here: `scope` is DEC-64's ISOLATION axis, and episodes are not
                different users. Separation between episodes is the `session` segment.
            domain: The store domain both turns write and read under. Must be in the store's
                own `domain_enum`.
            session_prefix: Leading segment of every session key, so one store can hold two
                runs without them colliding.
            reset_partitions: `True` gives every episode run its own session -- the reset
                gate (iii) asserts. `False` pins every run to one session and is the CAN-FAIL
                CONTROL for that gate: it reconstructs the leak, and the `NO_TURN1` arm starts
                passing. It exists to be used by a test, not by a run.

        Raises:
            HarnessContractError: `mind` has no store, or holds a different one than `probe`.
        """
        if mind.store is None:
            raise HarnessContractError(
                "EpisodeHarness: this WhiteMatter declares no 'episodic_store' participant, "
                "so no turn can commit anything and every episode would abstain."
            )
        if mind.store is not probe:
            raise HarnessContractError(
                "EpisodeHarness: `probe` is not the store `mind` was built with, so its log "
                "would describe reads the forward pass never made."
            )
        self.mind = mind
        self.probe = probe
        self.encoder = encoder
        self.principal = principal
        self.domain = domain
        self.session_prefix = session_prefix
        self.reset_partitions = reset_partitions
        self._ordinal: Iterator[int] = itertools.count()
        self._committed: list[CommittedRecord] = []
        """Every record this harness has watched a turn 1 commit, for the WHOLE run.

        Run-wide, not per episode, and that is the point. A per-episode list can only ever
        match a record the current episode wrote, so a leak from an EARLIER episode -- the
        exact failure gate (iii) exists to catch -- would come back as "the read returned
        nothing" and the gate would report a clean reset while state was crossing between
        episodes. Matching against every record the run committed is what lets the
        `NO_TURN1` arm say *whose* memory it just read (`read_record_logical_key`), which is
        the difference between detecting a leak and being fooled by one.
        """

    # -- scope plumbing ------------------------------------------------------------------

    def _session_for(self, item: EpisodeItem, arm: EpisodeArm) -> str:
        """The session segment this episode run owns.

        Args:
            item: The episode.
            arm: Which staging is being run.

        Returns:
            A session key unique to this run, or the one shared key when
            `reset_partitions` is off.
        """
        if not self.reset_partitions:
            return f"{self.session_prefix}-shared"
        return f"{self.session_prefix}-{item.item_id[:16]}-{arm.value}-{next(self._ordinal)}"

    def _resident(self, scope: Scope, b_store: int) -> int:
        """Count the records resident in one partition, without touching the probe's log.

        Args:
            scope: The partition to count.
            b_store: How many slots to ask for; the count saturates at this many.

        Returns:
            The number of unmasked slots the store returned.
        """
        _latents, mask = self.probe.inner.read(scope, domain=self.domain, b_store=b_store)
        return int(mask.sum().item())

    # -- one turn ------------------------------------------------------------------------

    def _run_turn(
        self,
        text: str,
        scope: Scope,
        logical_key: str,
        *,
        options: Sequence[str] | None,
    ) -> TurnRun:
        """One `WhiteMatter.forward`, with its store traffic captured.

        Args:
            text: The turn's document.
            scope: The episode's partition.
            logical_key: This turn's identity within the partition.
            options: The item's options when this turn is scored, `None` when it is not.
                Turn 1 is never scored -- §5.5(e) scores the probe -- so it passes `None` and
                `RankHead` is not run at all.

        Returns:
            The turn's receipts, written latents, step-8 read and scores.

        Raises:
            HarnessContractError: The mind's `k_candidates` cannot hold the item's options, or
                the pass made no domain-scoped store read.
        """
        inputs: dict[str, Any] = dict(self.encoder.encode_turn(text))
        inputs["scope"] = scope
        inputs["domain"] = self.domain
        inputs["logical_keys"] = [logical_key]
        if options is not None:
            expected_k = len(options) + 1
            if self.mind.config.k_candidates != expected_k:
                raise HarnessContractError(
                    f"EpisodeHarness: the item has {len(options)} options, so RankHead needs "
                    f"k_candidates = {expected_k} (NULL plus one slot per option); this mind "
                    f"declares {self.mind.config.k_candidates}."
                )
            inputs["candidates"] = self.encoder.encode_options(options)

        self.probe.clear()
        output = self.mind(inputs)

        # `forward` reads the store twice per item: once inside `_raw_summary` (an occupancy
        # proxy, `domain=None, global_query=True, b_store=1`) and once at step 8 (the
        # load-bearing bank, domain-scoped). Only the second is the read this harness reports
        # on, and `global_query` separates them without depending on call order.
        scoped = [read for read in self.probe.reads if not read.global_query]
        if len(scoped) != 1:
            raise HarnessContractError(
                f"EpisodeHarness: expected exactly one domain-scoped store read per turn at "
                f"B = 1, saw {len(scoped)}; the read this harness reports on is not "
                "identifiable, so its evidence would be a guess."
            )
        receipts = list(output.write_receipt or [])
        return TurnRun(
            write_receipts=receipts,
            written_latents=[write.latent for write in self.probe.writes],
            step8_read=scoped[0],
            scores=output.scores,
        )

    # -- one episode ---------------------------------------------------------------------

    def run_episode(self, item: EpisodeItem, arm: EpisodeArm) -> EpisodeResult:
        """Run one episode in one staging, and report the verdict with its evidence.

        The ORDER is the contract, not an implementation detail: turn 1 runs, its receipt is
        required, the commit is observed in the partition, and only then does turn 2 run. A
        turn 2 scored against an uncommitted write is measuring nothing, which is why the
        commit is read back rather than inferred from the receipt alone.

        Args:
            item: The episode.
            arm: Which staging to run.

        Returns:
            The result, including whether turn 2's read returned turn 1's record.

        Raises:
            Dec38ViolationError: Gate (ii) -- the episode's turns share a source row.
            HarnessContractError: Turn 1 produced no receipt, or a partition that must be
                empty was not.
        """
        assert_disjoint_source_rows(item)

        session = self._session_for(item, arm)
        scope = derive_scope(self.principal, session=session)
        b_store = self.mind.config.participants["episodic_store"].token_budget_max or 8
        resident_at_start = self._resident(scope, b_store)
        partition_empty_at_start = resident_at_start == 0
        if self.reset_partitions and not partition_empty_at_start:
            raise HarnessContractError(
                f"EpisodeHarness: partition {session!r} already holds {resident_at_start} "
                "record(s) at episode start; the per-episode reset did not hold, and every "
                "verdict from this point could be another episode's memory."
            )

        own: list[CommittedRecord] = []
        turn1_key: str | None = None
        turn1_committed = False
        turn1_text = item.text_for(arm)
        if turn1_text is not None:
            turn1_key = f"{item.item_id}#t0"
            turn1 = self._run_turn(turn1_text, scope, turn1_key, options=None)
            if not turn1.write_receipts:
                raise HarnessContractError(
                    "EpisodeHarness: turn 1 returned no WriteReceipt, so nothing committed "
                    "and turn 2 would be scored against an empty partition."
                )
            receipt = turn1.write_receipts[0]
            turn1_committed = bool(receipt.committed)
            fact = item.fact_for(arm)
            assert fact is not None  # arm has a turn 1, so it has a fact
            own = [
                CommittedRecord(latent=latent, fact=fact, logical_key=receipt.logical_key)
                for latent in turn1.written_latents
            ]
            self._committed.extend(own)

        resident_before_turn2 = self._resident(scope, b_store)
        commit_observed = turn1_committed and resident_before_turn2 >= 1

        turn2 = self._run_turn(item.turn2_text, scope, f"{item.item_id}#t1", options=item.options)
        read = turn2.step8_read
        assert read is not None  # _run_turn raises when it is not exactly one
        rank, matched, cosine = _match_read_to_commits(read, own, self._committed)
        returned = rank >= 0

        oracle_choice = ABSTAIN
        if matched is not None:
            oracle_choice = item.option_index_for_fact(matched.fact)
        rank_head_choice = _rank_head_choice(turn2.scores)

        return EpisodeResult(
            item_id=item.item_id,
            arm=arm,
            variant=item.variant,
            ablation_bin=item.ablation_bin,
            split=item.split,
            scope_session=session,
            partition_empty_at_start=partition_empty_at_start,
            turn1_committed=turn1_committed,
            turn1_logical_key=turn1_key,
            commit_observed_before_turn2=commit_observed,
            resident_before_turn2=resident_before_turn2,
            store_read_slots=int(read.mask.shape[-1]),
            store_read_records=int(read.mask.sum().item()),
            read_returned_turn1_record=returned,
            read_rank_of_turn1_record=rank,
            read_top1_cosine_to_turn1=cosine,
            read_record_logical_key=None if matched is None else matched.logical_key,
            oracle_choice=oracle_choice,
            oracle_pass=oracle_choice == item.gold_index,
            rank_head_choice=rank_head_choice,
            rank_head_pass=rank_head_choice == item.gold_index,
            gold_index=item.gold_index,
            counterfactual_index=item.counterfactual_index,
        )

    def run(self, items: Iterable[EpisodeItem], arms: Sequence[EpisodeArm]) -> list[EpisodeResult]:
        """Run every arm of every episode, in item order then arm order.

        Args:
            items: The episodes.
            arms: Which stagings to run, in the order they should run. `FULL` before
                `NO_TURN1` is what makes gate (iii)'s can-fail control able to leak: with the
                reset off, the `NO_TURN1` arm then has an earlier record of its own item to
                find.

        Returns:
            One result per (item, arm).
        """
        return [self.run_episode(item, arm) for item in items for arm in arms]


def _match_read_to_commits(
    read: StoreRead,
    own: Sequence[CommittedRecord],
    run_wide: Sequence[CommittedRecord],
) -> tuple[int, CommittedRecord | None, float]:
    """Find which committed record turn 2's read returned, and at what rank.

    THE EPISODE'S OWN RECORDS ARE SEARCHED FIRST, AND THAT ORDER IS LOAD-BEARING (measured
    2026-09-07). Two turn-1 latents can be bit-identical when their inputs are: the
    `WRONG_TURN1` arm stages the donor's `key_claim` as its text, and donors recur across
    items, so 466 of 2,048 runs matched a run-wide record written by a DIFFERENT episode with
    the same latent. The verdict was unaffected -- identical inputs commit identical facts --
    but `read_record_logical_key` named the wrong episode, which is exactly the field a reader
    would use to check the harness's own claim. Searching this episode's records first makes
    the attribution correct without weakening leak detection: a leaked record is by definition
    NOT in `own`, so it still comes back from `run_wide` and still says whose it was.

    Args:
        read: The step-8 read.
        own: The records THIS episode's turn 1 committed.
        run_wide: Every record the harness has watched a turn 1 write in this run.

    Returns:
        `(slot rank, the matched record, its cosine to the returned slot)`, or
        `(-1, None, 0.0)` when no unmasked slot is one of them. The rank is the SLOT ORDER the
        store returned, so `0` means the read put that record FIRST -- which is the claim PR
        #86's cosine ranking makes, and the one worth reporting per episode.
    """
    seen = {id(record) for record in own}
    committed = [*own, *(record for record in run_wide if id(record) not in seen)]
    if not committed:
        return -1, None, 0.0
    latents = read.latents
    mask = read.mask
    for slot in range(int(mask.shape[-1])):
        if not bool(mask[slot]):
            continue
        candidate = latents[slot]
        for record in committed:
            if candidate.shape != record.latent.shape:
                continue
            if torch.allclose(candidate, record.latent, rtol=0.0, atol=_LATENT_MATCH_ATOL):
                cosine = torch.nn.functional.cosine_similarity(
                    candidate.flatten(), record.latent.flatten(), dim=0
                )
                return slot, record, float(cosine.item())
    return -1, None, 0.0


def _rank_head_choice(scores: Tensor | None) -> int:
    """Turn `RankHead`'s `[1, k]` scores into an option index.

    Index 0 is `NULL`, not an option, so a mind that prefers `NULL` has declined to choose and
    the result is `ABSTAIN` rather than option 0 -- which would silently credit or blame the
    first option for a refusal.

    Args:
        scores: `WhiteMatterOutput.scores`, or `None` when the turn carried no candidates.

    Returns:
        The chosen option index, or `ABSTAIN`.
    """
    if scores is None:
        return ABSTAIN
    best = int(torch.argmax(scores[0]).item())
    return ABSTAIN if best == 0 else best - 1


def verify_negative_control(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    """W3 gate (i): every `WRONG_TURN1` episode must FAIL, on the declared wrong option.

    §5.5(e) property 1 in its strong form. The weak form -- "the item did not pass" -- is
    satisfied by an abstention, and an abstention is also what a broken store produces, so it
    cannot tell a working control from a dead one. The strong form the item supports is that
    the substituted fact selects a specific option that is *present and wrong*, so this
    reports both: how many failed, and how many failed FOR THE RIGHT REASON.

    Args:
        results: Results from any set of arms; only `WRONG_TURN1` rows are considered.

    Returns:
        `episodes`, `failed`, `landed_on_counterfactual`, `passed_and_should_not_have`, and
        `gate_fires` -- `True` only when every control episode failed and every one of them
        landed on its declared counterfactual option.
    """
    arm = [row for row in results if row.arm is EpisodeArm.WRONG_TURN1]
    failed = [row for row in arm if not row.oracle_pass]
    landed = [row for row in arm if row.oracle_choice == row.counterfactual_index]
    passed = [row for row in arm if row.oracle_pass]
    return {
        "episodes": len(arm),
        "failed": len(failed),
        "landed_on_counterfactual": len(landed),
        "passed_and_should_not_have": [row.item_id for row in passed],
        "gate_fires": bool(arm) and len(failed) == len(arm) and len(landed) == len(arm),
    }


def verify_partition_reset(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    """W3 gate (iii): with the reset intact, every turn-1-less episode still fails.

    Two independent observations, because either one alone can be satisfied by accident. The
    turn-1-less arm must fail (the design's own test), AND every episode of every arm must
    have started against an empty partition (the mechanism that makes it fail). A run where
    the second holds and the first does not is a scoring bug; a run where the first holds and
    the second does not passed by luck.

    Args:
        results: Results from any set of arms.

    Returns:
        `episodes`, `failed`, `read_a_record_anyway`, `partitions_empty_at_start`,
        `episodes_total` and `gate_fires`.
    """
    arm = [row for row in results if row.arm is EpisodeArm.NO_TURN1]
    failed = [row for row in arm if not row.oracle_pass]
    leaked = [row for row in arm if row.read_returned_turn1_record]
    empty = [row for row in results if row.partition_empty_at_start]
    return {
        "episodes": len(arm),
        "failed": len(failed),
        "read_a_record_anyway": [row.item_id for row in leaked],
        "partitions_empty_at_start": len(empty),
        "episodes_total": len(results),
        "gate_fires": (bool(arm) and len(failed) == len(arm) and len(empty) == len(results)),
    }


def harness_receipt(
    results: Sequence[EpisodeResult],
    *,
    config: Mapping[str, Any] | None = None,
    max_episode_rows: int | None = None,
) -> dict[str, Any]:
    """Summarise a run, with the store-read evidence beside every accuracy.

    THE FIELD THAT MATTERS. `store.read_returned_turn1_record` is reported per arm, next to
    that arm's accuracy, and `store.store_is_load_bearing` is `False` whenever an arm that ran
    a turn 1 scored above chance without its reads returning that turn's record. An accuracy
    reported without it is the exact failure mode `mean(a_store)` passing G29 on a
    zero-mutual-information store already demonstrated once.

    Args:
        results: Every result in the run.
        config: Run identity to stamp -- thread pinning, item source, mind configuration.
            Recorded verbatim; the harness does not interpret it.
        max_episode_rows: Keep only this many per-episode rows in `episodes`, so a
            full-shard run does not produce a receipt too large to review. EVERY summary and
            EVERY gate is still computed over the whole run -- only the row listing is cut,
            and `episode_rows` says by how much. `None` keeps them all.

    Returns:
        A JSON-safe receipt: `config`, `arms`, `gates`, `store`, `episode_rows` and
        `episodes`.
    """
    arms: dict[str, Any] = {}
    for arm in EpisodeArm:
        rows = [row for row in results if row.arm is arm]
        if not rows:
            continue
        read_hits = sum(1 for row in rows if row.read_returned_turn1_record)
        arms[arm.value] = {
            "episodes": len(rows),
            "oracle_pass": sum(1 for row in rows if row.oracle_pass),
            "oracle_abstain": sum(1 for row in rows if row.oracle_choice == ABSTAIN),
            "rank_head_pass": sum(1 for row in rows if row.rank_head_pass),
            "turn1_committed": sum(1 for row in rows if row.turn1_committed),
            "commit_observed_before_turn2": sum(
                1 for row in rows if row.commit_observed_before_turn2
            ),
            "read_returned_turn1_record": read_hits,
            "read_rank_0": sum(1 for row in rows if row.read_rank_of_turn1_record == 0),
        }
    with_turn1 = [row for row in results if row.turn1_committed]
    load_bearing = bool(with_turn1) and all(row.read_returned_turn1_record for row in with_turn1)
    rows_kept = results if max_episode_rows is None else results[:max_episode_rows]
    return {
        "config": dict(config or {}),
        "arms": arms,
        "gates": {
            "negative_control": verify_negative_control(results),
            "partition_reset": verify_partition_reset(results),
        },
        "store": {
            "episodes_with_a_turn1": len(with_turn1),
            "read_returned_turn1_record": sum(
                1 for row in with_turn1 if row.read_returned_turn1_record
            ),
            "store_is_load_bearing": load_bearing,
            "note": (
                "store_is_load_bearing is False whenever an episode committed a turn 1 whose "
                "record turn 2's read did not return. Any accuracy reported beside a False "
                "here was not produced by recall."
            ),
        },
        "episode_rows": {
            "recorded": len(rows_kept),
            "of": len(results),
            "truncated": len(rows_kept) < len(results),
        },
        "episodes": [row.as_dict() for row in rows_kept],
    }


class HashingTurnEncoder:
    """A deterministic, untrained stand-in `TurnEncoder`.

    WHAT IT IS NOT. Not a tokeniser and not an embedding model. It hashes a string to a seed
    and draws token ids and candidate vectors from it, so identical text gives identical
    inputs on every host and different text gives unrelated inputs. Nothing it produces
    carries meaning, and no accuracy measured through it is evidence about a region.

    WHY IT IS IN `src/` RATHER THAN A TEST FILE. The harness has to be runnable before the
    regions it will eventually score are retrained -- W1's token-rank result made a
    `language`/`memory` retrain mandatory and W7v's rebuild is corpus-blocked -- and the
    receipt has to be able to NAME the encoder that produced its numbers. A stand-in nobody
    can name is a stand-in that gets mistaken for the real thing. It holds no weights: the
    faculties are the models, and they stay outside this module.

    `blake2b`, not `hash()`: Python salts `str.__hash__` per process, so a run today and a
    rerun tomorrow would encode the same turn differently and no two receipts would compare.
    """

    name = "hashing-stand-in/v1"
    """Identifier for the receipt's `config` block. Bump it if the derivation changes."""

    def __init__(
        self,
        text_faculties: Mapping[str, int],
        *,
        seq_len: int,
        workspace_dim: int,
        seed: int = 0,
    ) -> None:
        """Configure the encoder for one mind's participant set.

        Args:
            text_faculties: `{participant name: vocabulary size}` for every non-store
                participant. Every one gets the same ids from the same text, which is correct
                for a stand-in: any per-faculty difference here would be invented signal.
            seq_len: Token ids per turn.
            workspace_dim: `D_w`, the width `RankHead` scores candidates at.
            seed: Mixed into every draw, so two runs can be given different stand-in inputs
                deliberately rather than by accident.
        """
        self.text_faculties = dict(text_faculties)
        self.seq_len = seq_len
        self.workspace_dim = workspace_dim
        self.seed = seed

    def _generator(self, text: str) -> torch.Generator:
        """Seed a generator from a string, reproducibly across processes and hosts.

        Args:
            text: The string to derive a seed from.

        Returns:
            A CPU generator seeded by `blake2b(text)` mixed with `self.seed`.
        """
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return torch.Generator().manual_seed(
            (int.from_bytes(digest, "big") ^ self.seed) % (2**63 - 1)
        )

    def encode_turn(self, text: str) -> dict[str, Tensor]:
        """Encode one turn as token ids for every text faculty.

        Args:
            text: The turn's document.

        Returns:
            `{faculty: [1, seq_len] int64}`.
        """
        generator = self._generator(text)
        return {
            name: torch.randint(0, vocab, (1, self.seq_len), generator=generator)
            for name, vocab in self.text_faculties.items()
        }

    def encode_options(self, options: Sequence[str]) -> Tensor:
        """Embed the options as `RankHead` content candidates.

        Args:
            options: The item's rendered options, in item order.

        Returns:
            `[1, len(options), workspace_dim]`.
        """
        rows = [
            torch.randn(self.workspace_dim, generator=self._generator(f"option:{option}"))
            for option in options
        ]
        return torch.stack(rows, dim=0).unsqueeze(0)
