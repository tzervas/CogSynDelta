"""The episodic store's E0 interface, an in-memory stub, and its named contract gaps.

WHAT THIS IS, IN THE SPEC'S OWN WORDS (`docs/design/INTERCONNECT-MODULE-SPEC.md` Table 2,
row `episodic_store.py`): *"Defines the `EpisodicStore` Protocol (E0 interface), an
`InMemoryStoreStub`, and `ContractGap` exceptions naming each unimplemented DEC-63..66
clause."* This is lane IC-6 of the interconnect's file-disjoint lane plan (spec section 7,
Table 10): wave 1, depends only on IC-0.

WHAT THIS IS NOT. Not the store's build -- that is row E1 (`docs/design/
REGION-TAXONOMY-AND-INTERCONNECT.md` line 2649): byte capacity computed per host per
scheduler tick from live `nvidia-smi` state, scored eviction, GPU/RAM/disk tiering, SQLite
durability, and the six lifecycle verbs with refusing backpressure at `max_in_flight = 32`.
This module ships only what wave-2/3 lanes need to build against: the interface shape and a
CPU, in-process stand-in that is honest about what it does not do.

THE PARTITION AXIS (DEC-64): `(scope, domain, logical_key)`. `scope` is the authenticated
principal, derived server-side and NEVER accepted verbatim from a request -- `derive_scope`
below is the only sanctioned constructor of a `Scope`, and `session` is the "sub-segment
the request may name" that DEC-64 describes. `domain` is memory-gate's closed task-taxonomy
axis, ported unchanged (`domain_enum` at construction). `logical_key` identifies one episode
within a `(scope, domain)` partition; a second write under the same triple is last-write-wins
(memory-gate's own contract, ported here as a green test rather than a gap).

INDEX-NOT-BYTES, PARTIALLY (DEC-65). The taxonomy's model is an index over latent bytes the
*runtime* owns -- residency (`Gpu`/`Ram`/`Disk`) is a metadata tag on a pointer, and the
store never becomes a second VRAM allocator. `InMemoryStoreStub` cannot honour that: it is a
CPU, in-process conformance stand-in with nothing outside itself to point *into*, so it holds
the latent tensor by value. That is the one place this stub knowingly diverges from DEC-65's
own model rather than silently pretending to implement it -- named here, and the residency
machinery (`promote`) that the real, tiered store would need is a `ContractGap`, not a
best-effort partial implementation of tiering over one flat dict.

FOUR NAMED GAPS, ONE `ContractGap` EACH. `ContractGap.dec` is always one of these four; see
each method's docstring for why it, specifically, cites that DEC:

  - DEC-63 (dynamic byte capacity + staleness-scored eviction): `capacity_bytes`, `evict`.
  - DEC-64 (the scope axis's own "server MAY bound `session`" clause -- the axis itself is
    adopted and fully implemented below; only the *bounding* is unbuilt): `bind_session`.
  - DEC-65 (residency / tiering over the runtime-owned bytes this stub instead holds by
    value): `promote`.
  - DEC-66 (learned consolidation, explicitly deferred for the whole of v1 rather than
    unbuilt-but-intended): `consolidate`.

WHERE THE SPEC LEFT A CHOICE THIS MODULE HAD TO MAKE (recorded here, and again in the lane
report, per the task's instruction to record spec ambiguity and how it was resolved):

  1. The nine E0 fixtures split five ways across DEC-63/65, not evenly across all four,
     because the taxonomy's own "Test-pinned behaviours to port" list
     (`admit_spills_lowest_importance`, `gpu_hint_protects_from_spill`,
     `promote_returns_span_to_ram`, `retrieve_merges_tiers`, `disk_prune_drops_lowest`,
     TAX:683) sits entirely inside its `(2) Eviction` clause, which DEC-63's own text
     explicitly annexes ("gap (a) is EXTENDED to cover the decay function itself"). This
     module reads the three purely-eviction-scoring fixtures (spill order, the GPU-residency
     tie-break, disk-tier pruning) as DEC-63, and the two purely-tiering fixtures (promotion,
     cross-tier merge) as DEC-65, since those need the residency/pointer model DEC-65 owns
     rather than the capacity/staleness formula DEC-63 owns. DEC-64's scope axis and DEC-66's
     prune-only policy are not eviction-scoring fixtures at all, so neither gets one of the
     five.
  2. `read()` operates on ONE item's `scope`, returning `[b_store, D]` + `[b_store]` mask,
     not the batched `[B, b_store, D]` Table 3 shows for the module as a whole. Batching
     `B` distinct scopes into one bank slot range is `kv_bank.py`'s / `mind.py`'s job
     (IC-2/IC-8, later waves) -- looping this call per batch item and stacking. Building
     that loop here would be scope creep into files this lane does not own.
  3. `read()` ranks matches by `importance` descending, ties by LATER `written_at` (a
     stable, RNG-free order) rather than the taxonomy's full residency score `importance +
     gpu_resident_bonus - staleness_penalty(last_accessed)` (TAX:683's clause 2): the GPU
     bonus and the staleness half-life are exactly DEC-63's un-adopted formula, so ranking by
     it here would silently ship a DEC-63 answer through the back door of `read()` instead of
     through a `ContractGap`. Importance-descending is the smallest honest substitute that
     keeps `read()` usable by later lanes without pre-empting DEC-63. AMENDED 2026-09-07:
     the tie-break was `written_at` ASCENDING, which under a uniform importance made the
     oldest records permanently unreachable-past; see `_tie_break` for the measurement.
  4. The constructor signature is fixed by the spec at `InMemoryStoreStub(domain_enum,
     half_life_s, importance_default)` -- no `dim` argument. `D` is therefore inferred from
     the first write; a `read()` against a partition that has never been written (including
     the G32 unknown-principal case) returns `[b_store, 0]` rather than guessing a width, and
     an all-`False` mask, matching step 8's "an empty partition yields zero unmasked store
     slots, which is the no-store configuration" for any consumer that checks the mask
     before the width.
  5. `half_life_s` is accepted and stored (it is part of the fixed constructor signature,
     DEC-63's config constant) but not yet READ by anything -- exponential staleness decay is
     exactly DEC-63's gap. Recording it unused rather than omitting it is what keeps the
     constructor signature exact, per this lane's brief.
  6. `read()` TAKES A QUERY (added 2026-09-07, a change to a merged contract). It did not,
     and that was measured to make the store dead weight: with no query, a read is a pure
     function of `(scope, domain)`, so every item of a batch sharing one scope got the same
     records, and the returned bank was a CONSTANT -- max absolute difference `0.0` across
     items, across batches, and before versus after 50 training steps. Mutual information
     with the target was exactly zero, `dL/d(store attention)` was therefore ~0, and the
     store's attention direction was unidentified and random-walked. `query` is optional and
     defaults to `None`, which reproduces the original ordering exactly, so no caller
     written before this date changes behaviour.

     WHAT THE QUERY IS, AND WHAT IT IS NOT. It is CONTENT: `[D]`, compared to each resident
     record by cosine, best first. `scope` remains the ISOLATION axis -- a server-derived
     `(principal, session)` that `derive_scope`/G32/DEC-64 make a security boundary. Per-item
     SCOPING was measured as an alternative fix and failed below its own noise floor (MI
     under null, held-out accuracy under chance at all four seeds), and a content-derived
     scope key would have been worse than useless: it repurposes an isolation boundary as a
     content index, putting different principals' memories in one partition. Content goes in
     the query; identity stays in the scope.

     WHAT IT BOUGHT. With the query being the item's own `z_N` from a store-free no-grad
     pre-pass (`mind.py`'s `_store_query`): MI excess +1.58 to +1.86 bits, held-out accuracy
     0.855-0.996 against a null of 0.29, and 98.6% of items retrieving a top-1 record of
     their own class at step 0 -- before any optimizer step. The repair is to the read
     mechanism, not to training.

G32 -- STORE SCOPE (Table 8): *"a read or write with no server-derived scope, or a scope
supplied by the request. Effect: refused; an unknown principal reads an empty partition."*
Two failure shapes, not one:

  - `scope=None` ("no server-derived scope", an unknown/unauthenticated principal): `write`
    REFUSES (`StoreScopeError`) -- there is no legal partition to write an anonymous record
    into. `read` does NOT raise; it degrades to `_empty_partition`, because a caller with no
    identity should see nothing, not crash the request.
  - any value that is not a `Scope` built by `derive_scope` ("a scope supplied by the
    request" -- a raw string, dict, or other payload a handler forwarded verbatim instead of
    deriving one): BOTH `read` and `write` refuse. This module cannot verify authentication
    (that boundary is architectural, outside this file); what it CAN and does enforce is the
    *shape* of the violation -- an un-derived value reaching the store at all -- which is the
    smallest honest check available at this layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

__all__ = [
    "ContractGap",
    "EpisodicStore",
    "InMemoryStoreStub",
    "Scope",
    "StoreScopeError",
    "WriteReceipt",
    "derive_scope",
]


@dataclass(frozen=True, slots=True)
class Scope:
    """A server-derived `(principal, session)` pair -- DEC-64's partition axis, segment one.

    Never construct this directly from a request payload; use `derive_scope`. The frozen,
    slotted dataclass exists so a request handler cannot mutate a `Scope` in place after
    deriving it and so `isinstance(x, Scope)` is a meaningful check in `_require_scope`
    below -- a plain `NamedTuple` or `TypedDict` would let a client-supplied mapping of the
    same shape pass that check by accident.
    """

    principal: str
    """The authenticated principal. Never empty -- `derive_scope` enforces this."""

    session: str | None = None
    """DEC-64's "sub-segment the request may name and the server may bound." Naming is
    implemented (it is one more key segment below); bounding is `bind_session`'s gap."""


def derive_scope(principal: str, *, session: str | None = None) -> Scope:
    """The one sanctioned way to build a `Scope` (DEC-64: "derived server-side and never
    client-supplied"). Call this after authentication, never with a value read straight out
    of a request body -- passing THIS function's own output straight back through unchanged
    is exactly what makes a caller's scope "server-derived" rather than "request-supplied"
    from `_require_scope`'s point of view.

    Args:
        principal: The authenticated principal id. Must be non-empty.
        session: Optional session sub-segment (DEC-64).

    Returns:
        A `Scope` that `EpisodicStore.read`/`write` will accept.

    Raises:
        ValueError: `principal` is empty.
    """
    if not principal:
        raise ValueError("derive_scope: principal must be non-empty")
    return Scope(principal=principal, session=session)


class StoreScopeError(RuntimeError):
    """G32 fail-closed (Table 8): a store call lacked a server-derived scope, or carried one
    the request supplied itself. See the module docstring's G32 section for the two shapes
    this covers and why `read` and `write` respond to each differently.
    """


class ContractGap(RuntimeError):  # noqa: N818 -- spec Table 2 fixes this exact class name
    """A store capability `InMemoryStoreStub` does not implement, naming its DEC.

    Table 2: *"every DEC-63..66 gap raises `ContractGap` naming its DEC."* Raising here
    rather than silently no-op'ing or returning a fabricated answer is the same fail-closed
    discipline as this repo's guards (`SplitGuardError`, `src/cogsyndelta/splits.py`) applied
    to an unbuilt feature instead of a violated invariant: a caller that catches nothing
    still learns, from the traceback, exactly which taxonomy clause it depended on that E1
    has not built yet -- never a plausible-looking zero or an empty list standing in for "not
    implemented."
    """

    def __init__(self, dec: str, clause: str) -> None:
        """Name the unimplemented capability in the exception itself.

        Args:
            dec: The decision this gap belongs to, e.g. `"DEC-63"`.
            clause: One sentence naming the specific unimplemented behaviour.
        """
        self.dec = dec
        """Which decision this gap belongs to, e.g. `"DEC-63"` -- always one of DEC-63..66
        for this module (module docstring, "four named gaps")."""
        self.clause = clause
        """One sentence naming the specific unimplemented behaviour, for a reader who only
        sees the exception message."""
        super().__init__(
            f"{dec}: {clause} -- not implemented by InMemoryStoreStub "
            f"(row E1's build, docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md:2649)."
        )


@dataclass(frozen=True, slots=True)
class WriteReceipt:
    """One write's outcome. `committed` is always `True`: a `WriteReceipt` that exists at all
    means the write happened, matching memory-gate's `LearnReceipt` ("frozen with
    `committed: Literal[True]`, so a receipt cannot exist in a non-committed state", TAX:683)
    -- there is no droppable fire-and-forget handle to construct one otherwise.
    """

    scope: Scope
    domain: str
    logical_key: str
    importance: float
    written_at: float
    committed: bool = True


@dataclass(slots=True)
class Record:
    """Internal storage row (absent from `__all__`; the leading underscore this class used
    to carry is what the quality gate's PascalCase rule rejects). Not exported: callers see
    `WriteReceipt` and `read()`'s tensors,
    never this type, so `caller_cannot_mutate` (Table 9) is enforced by `read()` cloning
    rather than by anything on this class.
    """

    scope: Scope
    domain: str
    logical_key: str
    latent: Tensor
    importance: float
    written_at: float
    last_accessed: float
    provenance: str | None


def _tie_break(record: Record) -> tuple[float, float, str]:
    """The RNG-free order a read falls back to: importance down, then NEWEST first, then key.

    THE MEASURED DEFECT THIS CLOSES (2026-09-07). The tie-break used to be `written_at`
    ASCENDING -- oldest first. Combined with a uniform `importance_default` (every write
    from `mind.py` takes the default; nothing on the forward path sets one), that made the
    priming records a permanent wall: with 8 primers resident and `b_store = 8`, every
    later write the model made ranked strictly below them and was never readable. Measured:
    an 8-record store and a 32-record store trained to bit-identical results, to 17
    significant figures, because the extra 24 records could not be reached.

    Newest-first is the smallest honest repair. It keeps the order total, deterministic and
    seed-free (ambiguity note 3's actual requirement), and it agrees with the direction the
    real store's eviction already scores in -- `episodic/store.py::_enforce_capacity` spills
    the OLDER record when scores tie, so preferring the older one on a read was the read
    and the eviction disagreeing about the same record. `logical_key` last keeps two records
    written inside the same clock tick in a fixed order.

    Args:
        record: The stored record to key.

    Returns:
        A sort key; `sorted` over it is the read's ranking when no query is supplied.
    """
    return (-record.importance, -record.written_at, record.logical_key)


@runtime_checkable
class EpisodicStore(Protocol):
    """The E0 interface every store implementation (this stub, and eventually E1's real
    store) must satisfy. `mind.py` (IC-8, a later wave) is typed against this, not against
    `InMemoryStoreStub` directly, so E1's real store is a drop-in replacement.

    Only `read` and `write` are load-bearing for the forward pass (spec section 3 step 8 and
    step 13); `capacity_bytes`, `evict`, `promote`, `bind_session` and `consolidate` exist so
    a caller can discover a DEC-63..66 gap by calling the interface itself rather than by
    reading this module's source -- `InMemoryStoreStub` implements all of them, four by
    raising `ContractGap`.
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
        """Write one episode latent under `(scope, domain, logical_key)` (DEC-64/DEC-65).

        Spec section 3 step 13: one record per turn, the mean over the workspace's 64
        latents of the final-normed `z_N`, with the default importance and a provenance
        sidecar "never on the read path." A second write at the same identity replaces the
        first (last-write-wins, E0 fixture `test_double_put_same_identity_last_write_wins`).

        Args:
            scope: A `Scope` from `derive_scope`, or `None` for an unknown principal.
            domain: One value of this store's `domain_enum`.
            logical_key: Identifies the episode within `(scope, domain)`.
            latent: `[D]` float tensor -- the mean-pooled workspace latent (`D = D_w = 512`
                at v1, but this stub infers `D` from whatever is written; see the module
                docstring's ambiguity note 4).
            importance: Overrides `importance_default` for this record when given.
            provenance: Human-readable sidecar text (DEC-65) -- kept for inspection, never
                returned by `read`.

        Returns:
            A `WriteReceipt` confirming the commit.

        Raises:
            StoreScopeError: G32 -- `scope` is `None` or not server-derived.
            ValueError: `domain` is outside `domain_enum`, or `latent` is not a 1-D
                floating-point tensor (mirrors G31's latent-tract assertion at this
                module's own boundary).
        """
        ...

    def read(
        self,
        scope: Scope | None,
        *,
        domain: str | None = None,
        global_query: bool = False,
        b_store: int,
        query: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Read up to `b_store` episodes for one item's scope (spec section 3 step 8).

        Args:
            scope: A `Scope` from `derive_scope`, or `None` for an unknown principal.
            domain: Restrict to one domain, or `None` with `global_query=True` for every
                domain under `scope`.
            global_query: Must be `True` when `domain` is `None` (E0 fixture
                `test_query_requires_domain_or_global_flag`).
            b_store: The workspace's floored store budget (`lo_store = 8` at v1, spec step
                8). The read pads or truncates to exactly this many slots.
            query: `[D]` float, WHAT this read is looking for -- the resident set is ranked
                by cosine similarity to it (ambiguity note 6). `None` keeps the original,
                query-free order, so every caller written before 2026-09-07 is unchanged.
                `query` is content; `scope` is isolation. They are different axes and must
                stay different: a content-derived scope would put two principals' memories
                in one partition and break G32/DEC-64's boundary.

        Returns:
            `(latents [b_store, D], mask [b_store])`, `mask` `True` for a real record and
            `False` for padding, ranked by cosine to `query` when one is given and by
            descending importance otherwise (ambiguity notes 3 and 6). Zero unmasked slots
            is "the no-store configuration" (spec step 8) whether because the partition is
            genuinely empty or because `scope` was `None` (G32).

        Raises:
            StoreScopeError: G32 -- `scope` is a value the request supplied directly rather
                than one built by `derive_scope`. (`scope=None` does NOT raise here.)
            ValueError: `domain` is `None` and `global_query` is `False`, `domain` is
                outside `domain_enum`, or `query` is not a 1-D floating-point tensor of the
                store's established width.
        """
        ...

    def capacity_bytes(self, host: str) -> int:
        """Report DEC-63's dynamic byte capacity for one host. Always raises in the stub.

        Raises:
            ContractGap: always, naming `"DEC-63"`.
        """
        ...

    def evict(self) -> int:
        """DEC-63's scored eviction (`importance + gpu_resident_bonus -
        staleness(last_accessed)`). Always raises in the stub.

        Raises:
            ContractGap: always, naming `"DEC-63"`.
        """
        ...

    def promote(self, scope: Scope, domain: str, logical_key: str) -> None:
        """DEC-65's tier promotion (`promote_returns_span_to_ram`, `retrieve_merges_tiers`
        -- ambiguity note 1). Always raises in the stub, which has one flat tier.

        Raises:
            ContractGap: always, naming `"DEC-65"`.
        """
        ...

    def bind_session(self, scope: Scope) -> Scope:
        """DEC-64's "the server MAY bound `session`" clause. Always raises in the stub.

        Raises:
            ContractGap: always, naming `"DEC-64"`.
        """
        ...

    def consolidate(self) -> None:
        """DEC-66: v1 is prune-only; a learned consolidation pass is a phase-3 candidate,
        not a v1 feature under construction. Always raises in the stub.

        Raises:
            ContractGap: always, naming `"DEC-66"`.
        """
        ...


def _require_scope(scope: Scope | None, *, action: str) -> Scope | None:
    """G32's shared check. Returns the validated `Scope`, or `None` for an unknown
    principal -- callers decide what `None` means for their own action (`write` refuses,
    `read` degrades), which is why this helper never raises on `None` itself.
    """
    if scope is None:
        return None
    if not isinstance(scope, Scope):
        raise StoreScopeError(
            f"G32: {action} refused -- scope must come from derive_scope(); got "
            f"{type(scope).__name__!r}, which reads as a request supplying its own scope "
            f"(docs/design/INTERCONNECT-MODULE-SPEC.md Table 8, G32)."
        )
    return scope


class InMemoryStoreStub:
    """The E0 conformance stub: a flat, in-process dict keyed by the full partition tuple.

    Explicitly NOT a durability claim (matching memory-gate's own in-memory tier, TAX:683
    clause 4: "explicitly NOT a durability claim") and not index-not-bytes compliant (module
    docstring) -- it exists so wave-2/3 lanes (`kv_bank.py`, `mind.py`) and this file's own
    tests have something conforming to `EpisodicStore` to run against on CPU, today.
    """

    def __init__(
        self,
        domain_enum: frozenset[str] | set[str] | tuple[str, ...] | list[str],
        half_life_s: float,
        importance_default: float,
    ) -> None:
        """Args:
        domain_enum: The closed set of legal `domain` values (DEC-64: "retained as
            memory-gate's task axis"). A `domain` outside this set is refused at `write`/
            `read`, matching memory-gate's Rust closed enum rather than a free string.
        half_life_s: DEC-63's staleness half-life constant, in seconds. Stored so the
            constructor signature matches the spec exactly (Table 2); not yet read by
            anything -- see the module docstring's ambiguity note 5.
        importance_default: Importance assigned to a `write()` that does not override it.
        """
        self._domain_enum = frozenset(domain_enum)
        self.half_life_s = half_life_s
        self.importance_default = importance_default
        self._records: dict[tuple[str, str | None, str, str], Record] = {}
        self._dim: int | None = None

    def _check_domain(self, domain: str | None, *, global_query: bool = False) -> None:
        if domain is None:
            if not global_query:
                raise ValueError(
                    "query_requires_domain_or_global_flag: domain is None and "
                    "global_query is False -- name a domain or pass global_query=True."
                )
            return
        if domain not in self._domain_enum:
            raise ValueError(f"domain {domain!r} is not in this store's domain_enum")

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
        """See `EpisodicStore.write`."""
        checked = _require_scope(scope, action="write")
        if checked is None:
            raise StoreScopeError(
                "G32: write refused -- no server-derived scope (an unknown principal "
                "cannot write; there is no legal partition to write into)."
            )
        self._check_domain(domain)
        if not torch.is_floating_point(latent):
            raise ValueError(
                f"write() latent has dtype {latent.dtype}, which is not floating point "
                "(the same DEC-47 shape rule G31 enforces at the interconnect's other "
                "tracts: an episodic latent is never an integer-typed payload)."
            )
        if latent.dim() != 1:
            raise ValueError(f"write() latent must be 1-D [D], got shape {tuple(latent.shape)}")
        if self._dim is None:
            self._dim = latent.shape[0]
        elif latent.shape[0] != self._dim:
            raise ValueError(
                f"write() latent width {latent.shape[0]} disagrees with this store's "
                f"established width {self._dim} (every record must share one D)."
            )

        now = time.time()
        key = (checked.principal, checked.session, domain, logical_key)
        record_importance = self.importance_default if importance is None else importance
        self._records[key] = Record(
            scope=checked,
            domain=domain,
            logical_key=logical_key,
            latent=latent.detach().clone(),
            importance=record_importance,
            written_at=now,
            last_accessed=now,
            provenance=provenance,
        )
        return WriteReceipt(
            scope=checked,
            domain=domain,
            logical_key=logical_key,
            importance=record_importance,
            written_at=now,
        )

    def _empty_partition(self, b_store: int) -> tuple[Tensor, Tensor]:
        """Spec step 8: "an empty partition yields zero unmasked store slots." Also G32's
        answer for an unknown principal, and this store's answer before any write ever sets
        `self._dim` (ambiguity note 4).
        """
        dim = self._dim or 0
        return torch.zeros(b_store, dim), torch.zeros(b_store, dtype=torch.bool)

    def _check_query(self, query: Tensor) -> None:
        """Hold `query` to the same shape rule `write` holds a record's latent to."""
        if not torch.is_floating_point(query):
            raise ValueError(
                f"read() query has dtype {query.dtype}, which is not floating point "
                "(the same DEC-47 shape rule G31 enforces at the interconnect's other "
                "tracts: a store query is never an integer-typed payload)."
            )
        if query.dim() != 1:
            raise ValueError(f"read() query must be 1-D [D], got shape {tuple(query.shape)}")
        if self._dim is not None and query.shape[0] != self._dim:
            raise ValueError(
                f"read() query width {query.shape[0]} disagrees with this store's "
                f"established width {self._dim} (a query is compared against records)."
            )

    def _rank(self, matches: list[Record], query: Tensor | None) -> list[Record]:
        """Order one partition's matches: by cosine to `query`, else by `_tie_break` alone.

        Cosine, not dot product: the records are mean-pooled `z_N` latents whose NORMS carry
        how long the turn was and how confident the workspace was, neither of which is a
        relevance signal. Ranking on the direction alone is what survived the measurement --
        98.6% of items retrieved a top-1 record of their own class at step 0, before any
        optimizer step, over a store whose maximum off-diagonal cosine was 0.9993. Near
        collinearity compresses the scores; it does not destroy their ORDER, and only the
        order reaches the caller.

        The whole partition is stacked into one `[N, D]` tensor per read. That is the same
        `O(N)` this method already paid to build `matches`, and this stub is a CPU
        conformance stand-in -- a store that needs an index over `N` needs `EpisodicStoreImpl`
        and, past that, DEC-63's own machinery.

        Args:
            matches: The partition's records; non-empty.
            query: `[D]` float, or `None` for the query-free order.

        Returns:
            `matches`, ordered best-first.
        """
        if query is None:
            return sorted(matches, key=_tie_break)
        self._check_query(query)
        with torch.no_grad():
            bank = torch.stack([record.latent for record in matches]).to(torch.float32)
            vector = query.detach().to(dtype=torch.float32, device=bank.device)
            similarity = torch.cosine_similarity(bank, vector.unsqueeze(0), dim=1)
        scored = list(zip(similarity.tolist(), matches))
        scored.sort(key=lambda pair: (-pair[0], *_tie_break(pair[1])))
        return [record for _similarity, record in scored]

    def read(
        self,
        scope: Scope | None,
        *,
        domain: str | None = None,
        global_query: bool = False,
        b_store: int,
        query: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """See `EpisodicStore.read`."""
        checked = _require_scope(scope, action="read")
        if checked is None:
            return self._empty_partition(b_store)
        self._check_domain(domain, global_query=global_query)

        matches = [
            record
            for record in self._records.values()
            if record.scope.principal == checked.principal
            and record.scope.session == checked.session
            and (domain is None or record.domain == domain)
        ]
        if not matches:
            return self._empty_partition(b_store)

        matches = self._rank(matches, query)
        now = time.time()
        selected = matches[:b_store]
        for record in selected:
            record.last_accessed = now

        dim = self._dim or 0
        latents = torch.zeros(b_store, dim)
        mask = torch.zeros(b_store, dtype=torch.bool)
        for i, record in enumerate(selected):
            latents[i] = record.latent
            mask[i] = True
        return latents, mask

    def capacity_bytes(self, host: str) -> int:
        """See `EpisodicStore.capacity_bytes`. `host` is accepted (and unused) so the
        signature matches DEC-63's `capacity_bytes(host, tick)` shape at the one call site
        that will ever exercise it before raising -- there is no "tick" argument because a
        stub with no capacity model has no tick to be dynamic against, either.
        """
        raise ContractGap(
            "DEC-63",
            f"capacity_bytes({host!r}): dynamic byte capacity from live VRAM/KV state",
        )

    def evict(self) -> int:
        """See `EpisodicStore.evict`."""
        raise ContractGap("DEC-63", "evict(): staleness- and importance-scored eviction")

    def promote(self, scope: Scope, domain: str, logical_key: str) -> None:
        """See `EpisodicStore.promote`."""
        raise ContractGap(
            "DEC-65",
            f"promote({scope!r}, {domain!r}, {logical_key!r}): GPU/RAM/disk tier promotion",
        )

    def bind_session(self, scope: Scope) -> Scope:
        """See `EpisodicStore.bind_session`."""
        raise ContractGap("DEC-64", f"bind_session({scope!r}): server-side session bounding")

    def consolidate(self) -> None:
        """See `EpisodicStore.consolidate`."""
        raise ContractGap("DEC-66", "consolidate(): learned consolidation (deferred to phase 3)")
