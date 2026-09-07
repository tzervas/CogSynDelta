"""`EpisodicStoreImpl` -- row E1's store: identity, byte capacity, eviction, lifecycle.

WHAT ROW E1 ASKS FOR, and where each clause is answered here:

  - *"`(scope, domain, logical_key)` identity with the scope segment derived server-side"* --
    `partition_scope_key` below, called by every verb on a `Scope` that `derive_scope` built.
    A request-supplied value never reaches a key.
  - *"BYTE capacity from section 8 gap (a)'s formula computed per host per scheduler tick"* --
    a `capacity_provider` callable, invoked on every admission and never memoised. See
    `capacity.py` for the formula and `_enforce_placement` for the one place its answer is
    consumed.
  - *"scored eviction `importance + gpu_resident_bonus - staleness(last_accessed)` with ties
    by older timestamp then key"* -- `_score` and `_enforce_placement`.
  - *"the six lifecycle verbs with REFUSING backpressure at `max_in_flight = 32`"* --
    `start`/`learn`/`retrieve`/`drain`/`flush`/`stop`, and `AdmissionGuard`.
  - *"SQLite as the durability oracle with in-memory as the conformance oracle"* -- both are
    `StoreBackend`s in `backends.py`; this class holds one and does not know which.

PLACEMENT IS HOST-DEPENDENT; MEMBERSHIP IS NOT (DEC-63 as amended 2026-09-07). `capacity_bytes`
is derived from live VRAM, so every decision it drives differs card by card -- which is correct
for WHERE a span sits, and was wrong for WHETHER a record exists. The two now have separate
methods and separate triggers: `_enforce_placement` demotes against the VRAM-derived capacity
and deletes nothing, and `_enforce_membership` deletes against `TierBudget.max_records`, a
declared record count that no card can move. `_enforce_placement` does not call
`_enforce_membership`; `learn` and `evict` invoke both, membership first. Demotion under VRAM
pressure is untouched and stays -- DEC-70 ratifies it, and it costs latency, not membership.

AND WHAT IT ASKS NOT TO BUILD, which is the harder half of the row. *"The store's two
projections are NOT trained here -- they are white-matter parameters and E2 trains them. This
row builds a container and proves it is a correct container; it makes no claim about
usefulness."* So: nothing in this module is an `nn.Module`, nothing has parameters, nothing
imports `StoreProjection`, and no test in this row measures whether reading the store helps
anything. `consolidate()` remains a `ContractGap` on DEC-66 for the same reason -- v1 is
prune-only, and a learned consolidation pass would be building past this row.

THE INDEX IS SEPARATE FROM THE BACKEND, and that is DEC-65's shape rather than a cache.
Section 1.3 clause (6): the store owns *"an INDEX AND A POLICY over latent bytes owned by the
runtime, not a second allocator."* `self._index` is that index -- span size, importance,
residency and the two timestamps per key -- and it is what eviction scores over. The backend
owns the bytes. Two consequences worth naming: scoring never deserialises a latent (a 10,000
record fuzz would otherwise rebuild 10,000 tensors per eviction pass), and a SQLite-backed
store rebuilds its index from the database at construction, so the index is derived state and
never the source of truth.

THE SIX VERBS AND WHICH OF THEM BACKPRESSURE CAN REFUSE. Section 1.3 clause (3) ports
`start`/`learn`/`drain`/`flush`/`stop` from `memory_gateway.py` and clause (5) names the read
path (`retrieve_context`); CSD's six are `start`, `learn`, `retrieve`, `drain`, `flush`,
`stop`. Backpressure *"refuses rather than queues"*: exceeding `max_in_flight = 32`
concurrent operations raises `StoreBackpressureError` synchronously at submit, and no work is
enqueued anywhere. The bound applies to every verb that touches the backend -- `learn`,
`retrieve`, `flush`, `promote`, `bind_session`, `evict`. It deliberately does NOT apply to
`drain` and `stop`, and this is a divergence stated rather than hidden: those two verbs exist
to unwind in-flight work, so a store saturated at 32 in-flight operations is exactly the state
in which they must still be callable. A `stop()` that backpressure can refuse is a store that
cannot be shut down under load, which is a worse failure than the one the bound prevents.
`start` touches no backend and is a state transition only.

WHAT IS NOT IMPLEMENTED HERE, AND IS NOT PRETENDED TO BE. The differential record encoding,
the record format and temporal versioning have their own design with open decisions
outstanding. This store holds whatever latent it is handed, whole, with no delta chain and no
version graph -- so a record here is one episode at one time, and the eventual format will
change what a record CARRIES without changing the identity, capacity or eviction contracts
this row proves.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from cogsyndelta.interconnect.episodic.backends import (
    GPU_TIERS,
    RecordKey,
    Residency,
    StoreBackend,
    StoredRecord,
    TierBudget,
)
from cogsyndelta.interconnect.episodic.capacity import CapacityDecision
from cogsyndelta.interconnect.episodic_store import (
    ContractGap,
    Scope,
    StoreScopeError,
    WriteReceipt,
)

__all__ = [
    "DEFAULT_HALF_LIFE_S",
    "DEFAULT_IMPORTANCE",
    "GPU_RESIDENT_BONUS",
    "MAX_IN_FLIGHT",
    "CapacityProvider",
    "EpisodicStoreImpl",
    "Lifecycle",
    "StoreBackpressureError",
    "StoreDurabilityError",
    "StoreLifecycleError",
    "partition_scope_key",
    "staleness_penalty",
]

MAX_IN_FLIGHT = 32
"""Section 1.3 clause (3), ported unchanged: *"`max_in_flight = 32`, and exceeding it raises
... synchronously at submit time rather than spawning unboundedly"* (`memory_gateway.py:
118-133`)."""

GPU_RESIDENT_BONUS = 1.0
"""The eviction score's residency term, ported from `tiered.rs:150-178`: *"`+1.0` if the span
is `Residency::Gpu`-tagged"*. E1's gate (iv) asserts a GPU-resident low-importance entry
outlives a host-resident higher-importance one *by exactly this bonus*, so it is a named
constant that a test can add to an importance and compare against, not a literal in `_score`."""

DEFAULT_HALF_LIFE_S = 3600.0
"""Section 8 gap (a) extended, recommended default: *"exponential half-life on
`last_accessed`, half-life a config constant starting at the same order as the workspace's own
context window in wall-clock time."* One hour is that order for a conversational context; the
receipt records the value that ran, which is how gap (a) says this closes."""

DEFAULT_IMPORTANCE = 0.5
"""Importance assigned to a `learn()` that does not override it."""

CapacityProvider = Callable[[], int]
"""What `capacity_bytes(host, tick)` looks like from the store's side: a zero-argument
callable it invokes at every admission. A caller wires this to `capacity.capacity_decision`
over a live `nvidia-smi` reading and the scheduler's current budgets. It is a CALLABLE rather
than an int precisely so that DEC-63's *"never cached"* is structural: there is nowhere in
this class to put a cached capacity."""

_SESSION_MAX_LEN = 64
_SESSION_ALLOWED = re.compile(r"[^A-Za-z0-9._:-]")


class StoreLifecycleError(RuntimeError):
    """A verb was called in a lifecycle state that does not accept it.

    `learn`/`retrieve` before `start()` or after `stop()` raise this rather than silently
    working: a store that accepts writes before it has started is a store whose `start()` hook
    is decorative.
    """


class StoreBackpressureError(RuntimeError):
    """More than `max_in_flight` operations were in flight; this one is REFUSED, not queued.

    The upstream name is `MemoryBackendUnavailableError`; the behaviour is what section 9.9 B2
    cares about -- *"a store that queues instead of refusing is a second unbounded object,
    which is the thing B2 exists to prevent."*
    """


class StoreDurabilityError(RuntimeError):
    """A `drain` or `flush` failed, or `stop` could not reach a durable terminal state.

    Section 1.3 clause (3): `stop()` is *"refuse-new -> drain -> flush -> stopped, re-raising
    any drain or flush failure as `MemoryDurabilityError`."*
    """


class Lifecycle(str, Enum):
    """The store's lifecycle states, in the only order they occur."""

    NEW = "new"
    """Constructed, not started. `learn`/`retrieve` refuse."""

    RUNNING = "running"
    """`start()` has run; every verb is accepted, subject to backpressure."""

    STOPPING = "stopping"
    """Inside `stop()`: new work is refused while in-flight work drains."""

    STOPPED = "stopped"
    """Terminal. Drained and flushed; every data verb refuses."""


def _escape_segment(segment: str) -> str:
    """Percent-escape a key segment so joined segments cannot collide.

    A naive `f"{principal}:{session}"` makes `("a:b", None)` and `("a", "b")` the same
    partition, which is a cross-scope read with no attacker required. Escaping the separator
    and the escape character makes the join injective.
    """
    return segment.replace("%", "%25").replace("/", "%2F")


def partition_scope_key(scope: Scope) -> str:
    """Derive DEC-64's `scope` key segment SERVER-SIDE from an authenticated principal.

    Section 9.9 B2, and row E1's own words: *"identity with the scope segment derived
    server-side."* The derivation is here, in the store, and its only input is a `Scope` that
    `derive_scope` built after authentication -- there is no code path from a request field to
    a key segment, which is the structural version of the rule rather than a validation of it.

    The `session` sub-segment DEC-64 allows a request to name is included, and `None` (no
    session) is encoded distinctly from the empty string, so "the whole principal" and "the
    principal's unnamed session" are different partitions rather than the same one.

    Args:
        scope: A `Scope` from `derive_scope`.

    Returns:
        The scope key segment, e.g. `"alice/-"` or `"alice/=web-42"`.

    Raises:
        StoreScopeError: `scope` is not a `Scope` built by `derive_scope` (G32's "a scope
            supplied by the request" shape), or its principal is empty.
    """
    if not isinstance(scope, Scope):
        raise StoreScopeError(
            f"G32: scope must come from derive_scope(); got {type(scope).__name__!r}, which "
            "reads as a request supplying its own scope."
        )
    if not scope.principal:
        raise StoreScopeError("G32: scope.principal is empty; there is no partition to key.")
    session = "-" if scope.session is None else "=" + _escape_segment(scope.session)
    return f"{_escape_segment(scope.principal)}/{session}"


def staleness_penalty(last_accessed: float, now: float, half_life_s: float) -> float:
    """Section 8 gap (a) extended: exponential half-life decay on `last_accessed`.

    Bounded in `[0, 1)` and monotone in age, which is what makes the `+1.0` GPU bonus a
    meaningful unit: a GPU-resident record can never be out-scored by staleness alone, only by
    importance. A never-accessed record's penalty tends to 1 and never reaches it.

    Args:
        last_accessed: Wall-clock time of the record's last read or promotion.
        now: The scoring tick's wall-clock time.
        half_life_s: Seconds after which an untouched record has accrued half the penalty.

    Returns:
        The penalty to subtract from the record's score.

    Raises:
        ValueError: `half_life_s` is not positive.
    """
    if half_life_s <= 0:
        raise ValueError(f"half_life_s must be > 0, got {half_life_s}")
    age = max(0.0, now - last_accessed)
    return 1.0 - 0.5 ** (age / half_life_s)


@dataclass(slots=True)
class IndexRow:
    """The store index row: everything eviction scores over, and no bytes.

    Not exported (see this package export list): callers see tensors and receipts, never this
    type. The leading underscore a private class would carry is what the repo quality gate
    rejects as non-PascalCase, which is why the name is bare -- the same resolution that
    `episodic_store.Record` already took.
    """

    span_bytes: int
    importance: float
    residency: Residency
    written_at: float
    last_accessed: float


class AdmissionGuard:
    """The in-flight bound. Refuses at the cap; never queues, never spawns.

    A counting guard rather than a semaphore-with-timeout on purpose: a semaphore that blocks
    is a queue with the queue hidden in the scheduler, and the contract is that the caller is
    told "no" synchronously so it can shed load itself.
    """

    def __init__(self, max_in_flight: int) -> None:
        """Args:
        max_in_flight: The cap. Must be positive.
        """
        if max_in_flight <= 0:
            raise ValueError(f"max_in_flight must be > 0, got {max_in_flight}")
        self.max_in_flight = max_in_flight
        self._condition = threading.Condition()
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        """Operations currently holding a slot.

        Returns:
            The current in-flight count.
        """
        with self._condition:
            return self._in_flight

    def acquire(self, verb: str) -> None:
        """Take a slot, or refuse.

        Args:
            verb: Name of the verb asking, for the error message.

        Raises:
            StoreBackpressureError: the cap is already reached.
        """
        with self._condition:
            if self._in_flight >= self.max_in_flight:
                raise StoreBackpressureError(
                    f"{verb}: refused -- {self._in_flight} operations already in flight and "
                    f"max_in_flight is {self.max_in_flight}. The store refuses rather than "
                    "queues (section 1.3 clause (3))."
                )
            self._in_flight += 1

    def release(self) -> None:
        """Give a slot back and wake anything waiting in `drain`."""
        with self._condition:
            self._in_flight -= 1
            self._condition.notify_all()

    def wait_idle(self, timeout_s: float) -> bool:
        """Block until no slots are held, without cancelling anything.

        Args:
            timeout_s: Seconds to wait before giving up.

        Returns:
            `True` if the store reached zero in-flight within the timeout.
        """
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._in_flight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    @contextmanager
    def slot(self, verb: str) -> Iterator[None]:
        """Hold a slot for the duration of a block.

        Args:
            verb: Name of the verb asking, for the error message.

        Yields:
            Nothing; the slot is released on exit, including on exception.

        Raises:
            StoreBackpressureError: the cap is already reached.
        """
        self.acquire(verb)
        try:
            yield
        finally:
            self.release()


class EpisodicStoreImpl:
    """Row E1's episodic store: a correct container over one durability backend.

    Satisfies `EpisodicStore` (E0's protocol), so `mind.py` and every wave-2/3 consumer typed
    against that protocol takes this in place of `InMemoryStoreStub` with no change. Five of
    the stub's six `ContractGap`s are now implemented (`capacity_bytes`, `evict`, `promote`,
    `bind_session`, and the tiering `promote` implies); `consolidate` stays a gap because
    DEC-66 defers it, not because E1 ran out of room.
    """

    def __init__(
        self,
        domain_enum: Iterable[str],
        *,
        backend: StoreBackend,
        capacity_provider: CapacityProvider,
        host: str = "localhost",
        half_life_s: float = DEFAULT_HALF_LIFE_S,
        importance_default: float = DEFAULT_IMPORTANCE,
        max_in_flight: int = MAX_IN_FLIGHT,
        tier_budget: TierBudget | None = None,
    ) -> None:
        """Args:
        domain_enum: The closed set of legal `domain` values (DEC-64's task axis, ported as
            a closed enum rather than a free string).
        backend: The durability oracle. `InMemoryBackend` for conformance, `SqliteBackend`
            for durability; the store does not know which it holds.
        capacity_provider: Zero-argument callable returning DEC-63's `capacity_bytes` for
            this host at the current tick. Invoked on every admission, never cached.
        host: Host label this store's capacity is computed for; `capacity_bytes(host)`
            refuses a different one rather than answering for the wrong card.
        half_life_s: Staleness half-life, in seconds.
        importance_default: Importance for a `learn()` that does not override it.
        max_in_flight: Refusing-backpressure cap.
        tier_budget: Item budgets for the residency ladder; upstream's defaults if omitted.
        """
        self._domain_enum = frozenset(domain_enum)
        self._backend = backend
        self._capacity_provider = capacity_provider
        self.host = host
        self.half_life_s = half_life_s
        self.importance_default = importance_default
        self.tier_budget = tier_budget or TierBudget()
        self._admission = AdmissionGuard(max_in_flight)
        self._state = Lifecycle.NEW
        self._lock = threading.RLock()
        self._index: dict[RecordKey, IndexRow] = {}
        self._resident_bytes = 0
        self._resident_count = 0
        self._disk_count = 0
        self._dim: int | None = None
        self._reindex()

    # -- index -------------------------------------------------------------------------

    def _reindex(self) -> None:
        """Rebuild the in-memory index from the backend (a SQLite store may already hold rows)."""
        self._index.clear()
        self._resident_bytes = 0
        self._resident_count = 0
        self._disk_count = 0
        for record in self._backend.records():
            self._index_put(
                record.key,
                IndexRow(
                    span_bytes=record.span_bytes,
                    importance=record.importance,
                    residency=record.residency,
                    written_at=record.written_at,
                    last_accessed=record.last_accessed,
                ),
            )
            if self._dim is None:
                self._dim = int(record.latent.shape[0])

    def _tally(self, meta: IndexRow, sign: int) -> None:
        """Add (`sign=+1`) or remove (`sign=-1`) one index row from the running totals.

        The totals exist so that an admission under budget is O(1) rather than a scan of the
        whole index: at 10,000 records, gate (iii)'s fuzz would otherwise spend its time
        re-summing the index rather than exercising partition isolation.
        """
        if meta.residency in GPU_TIERS:
            self._resident_bytes += sign * meta.span_bytes
            self._resident_count += sign
        else:
            # Index accounting only. Since DEC-63's 2026-09-07 amendment this tally no longer
            # triggers anything: deletion counts every record against a declared ceiling
            # (`_enforce_membership`), not the spilled subset against a tier bound.
            self._disk_count += sign

    def _index_put(self, key: RecordKey, meta: IndexRow) -> None:
        """Insert or replace one index row, keeping the totals exact."""
        existing = self._index.get(key)
        if existing is not None:
            self._tally(existing, -1)
        self._index[key] = meta
        self._tally(meta, +1)

    def _index_drop(self, key: RecordKey) -> None:
        """Remove one index row, keeping the totals exact."""
        existing = self._index.pop(key, None)
        if existing is not None:
            self._tally(existing, -1)

    @property
    def resident_bytes(self) -> int:
        """Bytes currently counted against `capacity_bytes` (GPU- and RAM-resident spans).

        Returns:
            The resident byte total. Disk-resident spans are excluded: the capacity bound is
            section 8's VRAM residual, and a spilled span occupies none of it.
        """
        with self._lock:
            return self._resident_bytes

    @property
    def in_flight(self) -> int:
        """Operations currently holding an admission slot.

        Returns:
            The in-flight count.
        """
        return self._admission.in_flight

    @property
    def state(self) -> Lifecycle:
        """Current lifecycle state.

        Returns:
            The state.
        """
        return self._state

    def resident_keys(self) -> set[RecordKey]:
        """The keys that survived eviction, i.e. are GPU- or RAM-resident.

        Returns:
            A set of `(scope_key, domain, logical_key)`. E1's gate (iv) compares this against
            the top-scored set it constructed.
        """
        with self._lock:
            return {k for k, m in self._index.items() if m.residency in GPU_TIERS}

    def admission_slot(self, verb: str = "caller") -> AbstractContextManager[None]:
        """Hold one admission slot, for a caller that needs it across a compound operation.

        Exposed so the `max_in_flight` bound is observable and testable without threads: a
        test can occupy the cap and assert the next verb is refused rather than delayed.

        Args:
            verb: Label used in the refusal message.

        Returns:
            A context manager holding one slot.
        """
        return self._admission.slot(verb)

    # -- validation --------------------------------------------------------------------

    def _check_domain(self, domain: str | None, *, global_query: bool = False) -> None:
        """Reject a domain outside the closed enum, and a query that names neither."""
        if domain is None:
            if not global_query:
                raise ValueError(
                    "query_requires_domain_or_global_flag: domain is None and "
                    "global_query is False -- name a domain or pass global_query=True."
                )
            return
        if domain not in self._domain_enum:
            raise ValueError(f"domain {domain!r} is not in this store's domain_enum")

    def _check_running(self, verb: str) -> None:
        """Refuse a data verb outside `RUNNING`."""
        if self._state is not Lifecycle.RUNNING:
            raise StoreLifecycleError(
                f"{verb}: refused -- store is {self._state.value}, not running. "
                "start() before writing or reading; after stop() the store is terminal."
            )

    @staticmethod
    def _check_latent(latent: Tensor) -> None:
        """Reject a latent that is not a 1-D floating-point tensor (G31's rule at this edge)."""
        if not torch.is_floating_point(latent):
            raise ValueError(
                f"learn() latent has dtype {latent.dtype}, which is not floating point."
            )
        if latent.dim() != 1:
            raise ValueError(f"learn() latent must be 1-D [D], got shape {tuple(latent.shape)}")

    def _check_dim(self, latent: Tensor) -> None:
        """Hold every record in the store to one width."""
        if self._dim is None:
            self._dim = int(latent.shape[0])
        elif int(latent.shape[0]) != self._dim:
            raise ValueError(
                f"learn() latent width {latent.shape[0]} disagrees with this store's "
                f"established width {self._dim} (every record must share one D)."
            )

    # -- lifecycle verb 1/6: start -----------------------------------------------------

    def start(self) -> None:
        """Begin accepting work. Idempotent while running; refuses to restart a stopped store.

        Raises:
            StoreLifecycleError: the store has already been stopped.
        """
        with self._lock:
            if self._state is Lifecycle.RUNNING:
                return
            if self._state in (Lifecycle.STOPPING, Lifecycle.STOPPED):
                raise StoreLifecycleError(
                    f"start: refused -- store is {self._state.value}; a stopped store is "
                    "terminal and is replaced, not restarted."
                )
            self._state = Lifecycle.RUNNING

    # -- lifecycle verb 2/6: learn -----------------------------------------------------

    def learn(
        self,
        scope: Scope | None,
        domain: str,
        logical_key: str,
        latent: Tensor,
        *,
        importance: float | None = None,
        provenance: str | None = None,
        residency: Residency = Residency.RAM,
        span_bytes: int | None = None,
    ) -> WriteReceipt:
        """Commit one episode, then enforce the byte capacity. Durable-first.

        The order is section 1.3 clause (4)'s hard rule: the backend commit precedes the index
        placement and the eviction pass, *"so eviction from hot changes residency and never
        loses an acked write."* An over-capacity write therefore triggers EVICTION, not a
        refusal and not an allocation -- E1's gate (ii), in its own words.

        Args:
            scope: A `Scope` from `derive_scope`. `None` is refused (G32: an unknown principal
                has no legal partition to write into).
            domain: One value of `domain_enum`.
            logical_key: Identifies the episode within `(scope, domain)`.
            latent: `[D]` float tensor.
            importance: Overrides `importance_default` for this record.
            provenance: Sidecar text; never on the read path.
            residency: DEC-65's tag for the span this record indexes.
            span_bytes: Size of the runtime-owned span this record indexes. Defaults to the
                latent's own `nbytes`, which is exact when the store holds the tensor itself.

        Returns:
            A frozen `WriteReceipt`; its existence means the write committed.

        Raises:
            StoreScopeError: G32 -- no scope, or a scope the request supplied.
            StoreLifecycleError: the store is not running.
            StoreBackpressureError: `max_in_flight` operations are already in flight.
            ValueError: bad domain, non-float or non-1-D latent, width disagreement, or a
                non-positive `span_bytes`.
        """
        if scope is None:
            raise StoreScopeError(
                "G32: learn refused -- no server-derived scope (an unknown principal cannot "
                "write; there is no legal partition to write into)."
            )
        scope_key = partition_scope_key(scope)
        with self._admission.slot("learn"):
            self._check_running("learn")
            self._check_domain(domain)
            self._check_latent(latent)
            with self._lock:
                self._check_dim(latent)
                if span_bytes is None:
                    cost = int(latent.numel() * latent.element_size())
                else:
                    cost = int(span_bytes)
                if cost <= 0:
                    raise ValueError(f"span_bytes must be > 0, got {cost}")
                now = time.time()
                record = StoredRecord(
                    scope_key=scope_key,
                    domain=domain,
                    logical_key=logical_key,
                    latent=latent.detach().clone(),
                    span_bytes=cost,
                    importance=self.importance_default if importance is None else importance,
                    residency=residency,
                    written_at=now,
                    last_accessed=now,
                    provenance=provenance,
                )
                self._backend.put(record)
                self._index_put(
                    record.key,
                    IndexRow(
                        span_bytes=cost,
                        importance=record.importance,
                        residency=residency,
                        written_at=now,
                        last_accessed=now,
                    ),
                )
                # Membership before placement: the declared record ceiling decides WHAT the
                # store holds, host-invariantly, and only then does the VRAM-derived capacity
                # decide WHERE the survivors sit (DEC-63 as amended 2026-09-07).
                self._enforce_membership()
                self._enforce_placement(now)
            return WriteReceipt(
                scope=scope,
                domain=domain,
                logical_key=logical_key,
                importance=record.importance,
                written_at=now,
            )

    # -- lifecycle verb 3/6: retrieve --------------------------------------------------

    def retrieve(
        self,
        scope: Scope | None,
        *,
        domain: str | None = None,
        global_query: bool = False,
        b_store: int,
        query: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Read up to `b_store` episodes for one scope, merged across residency tiers.

        `retrieve_merges_tiers` (E0 fixture 9/9) is this sentence: a spilled record is still a
        record, so the read path does not filter on residency. With no `query`, ranking is
        the same score eviction uses, so what survives eviction is what a read prefers -- one
        policy, not two. With a `query`, cosine to it ranks first and that score becomes the
        tie-break: relevance decides WHICH record answers this request, the eviction score
        decides which records exist to answer it at all.

        Args:
            scope: A `Scope` from `derive_scope`, or `None` for an unknown principal (which
                reads an empty partition rather than raising -- G32's read shape).
            domain: Restrict to one domain, or `None` with `global_query=True`.
            global_query: Must be `True` when `domain` is `None`.
            b_store: Workspace store budget; the read pads or truncates to exactly this many.
            query: `[D]` float, what this read is looking for; `None` keeps the original
                score-only order. See `episodic_store.py`'s ambiguity note 6 for the
                measurement that added it, and for why content belongs here rather than in
                the scope.

        Returns:
            `(latents [b_store, D], mask [b_store])`, `mask` `True` for a real record.

        Raises:
            StoreScopeError: G32 -- a scope the request supplied rather than derived.
            StoreLifecycleError: the store is not running.
            StoreBackpressureError: `max_in_flight` operations are already in flight.
            ValueError: `domain` is `None` without `global_query`, is outside the enum, or
                `query` is not a 1-D floating-point tensor of the store's width.
        """
        with self._admission.slot("retrieve"):
            self._check_running("retrieve")
            self._check_domain(domain, global_query=global_query)
            if scope is None:
                return self._empty_partition(b_store)
            scope_key = partition_scope_key(scope)
            with self._lock:
                matches = self._backend.partition(scope_key, domain)
                if not matches:
                    return self._empty_partition(b_store)
                now = time.time()
                selected = self._rank(matches, query, now)[:b_store]
                # A read touches `last_accessed` in the INDEX only, not in the backend. The
                # index is what eviction scores over, so recency is exact where it is used;
                # writing it through would turn every read into N durable writes for a field
                # no contract clause requires to survive a restart. A reopened store therefore
                # sees each record's last durable touch (its write or its last promotion),
                # which makes it slightly MORE eager to evict than it was before the restart
                # -- the conservative direction, and it is recorded here rather than
                # discovered later.
                for record in selected:
                    meta = self._index.get(record.key)
                    if meta is not None:
                        meta.last_accessed = now
                return self._pack(selected, b_store)

    def _empty_partition(self, b_store: int) -> tuple[Tensor, Tensor]:
        """Zero unmasked slots -- the no-store configuration."""
        dim = self._dim or 0
        return torch.zeros(b_store, dim), torch.zeros(b_store, dtype=torch.bool)

    def _pack(self, selected: list[StoredRecord], b_store: int) -> tuple[Tensor, Tensor]:
        """Pad or truncate the selected records into `[b_store, D]` plus a mask."""
        dim = self._dim or 0
        latents = torch.zeros(b_store, dim)
        mask = torch.zeros(b_store, dtype=torch.bool)
        for i, record in enumerate(selected[:b_store]):
            latents[i] = record.latent
            mask[i] = True
        return latents, mask

    # -- lifecycle verb 4/6: drain -----------------------------------------------------

    def drain(self, *, timeout_s: float = 30.0) -> None:
        """Await in-flight work to a terminal state WITHOUT cancelling it.

        Not subject to backpressure: see the module docstring. Draining is what a saturated
        store needs to be able to do.

        Args:
            timeout_s: Seconds to wait for in-flight work to finish.

        Raises:
            StoreDurabilityError: in-flight work did not finish within `timeout_s`.
        """
        if not self._admission.wait_idle(timeout_s):
            raise StoreDurabilityError(
                f"drain: {self._admission.in_flight} operations still in flight after "
                f"{timeout_s}s; nothing was cancelled."
            )

    # -- lifecycle verb 5/6: flush -----------------------------------------------------

    def flush(self) -> None:
        """Invoke the backend's durability barrier.

        Raises:
            StoreBackpressureError: `max_in_flight` operations are already in flight.
            StoreDurabilityError: the backend's barrier failed.
        """
        with self._admission.slot("flush"):
            try:
                self._backend.flush()
            except Exception as exc:
                raise StoreDurabilityError(f"flush: backend barrier failed: {exc}") from exc

    # -- lifecycle verb 6/6: stop ------------------------------------------------------

    def stop(self, *, timeout_s: float = 30.0) -> None:
        """Refuse-new, drain, flush, stopped -- re-raising any failure as a durability error.

        Args:
            timeout_s: Seconds allowed for the drain.

        Raises:
            StoreDurabilityError: the drain or the flush failed. The store is left `STOPPING`
                in that case, not `STOPPED`: a store that could not flush has not stopped.
        """
        with self._lock:
            if self._state is Lifecycle.STOPPED:
                return
            self._state = Lifecycle.STOPPING
        self.drain(timeout_s=timeout_s)
        try:
            self._backend.flush()
        except Exception as exc:
            raise StoreDurabilityError(f"stop: flush failed: {exc}") from exc
        with self._lock:
            self._state = Lifecycle.STOPPED

    # -- capacity and eviction ---------------------------------------------------------

    def capacity_bytes(self, host: str) -> int:
        """DEC-63's capacity for this host at THIS tick, read from the provider every call.

        Args:
            host: Host label. Must match the label this store was constructed for.

        Returns:
            Bytes the store may hold resident right now.

        Raises:
            ValueError: `host` is not the host this store was built for -- answering for the
                wrong card is the failure mode gate (ii) exists to catch.
            StoreBackpressureError: `max_in_flight` operations are already in flight.
        """
        if host != self.host:
            raise ValueError(
                f"capacity_bytes({host!r}): this store was constructed for host "
                f"{self.host!r}; capacity is per host and is not answered by proxy."
            )
        with self._admission.slot("capacity_bytes"):
            return int(self._capacity_provider())

    def _score_meta(self, meta: IndexRow, now: float) -> float:
        """`importance + gpu_resident_bonus - staleness(last_accessed)`, over an index row."""
        bonus = GPU_RESIDENT_BONUS if meta.residency is Residency.GPU else 0.0
        penalty = staleness_penalty(meta.last_accessed, now, self.half_life_s)
        return meta.importance + bonus - penalty

    def _rank_key(self, record: StoredRecord, now: float) -> tuple[float, float, RecordKey]:
        """Read ranking: eviction score descending, then NEWEST first, then key.

        Scored off the INDEX where the record has one, so a read and an eviction never
        disagree about the same record -- one policy, evaluated once per tick.

        AMENDED 2026-09-07. The timestamp tie-break used to be ascending (older first),
        which is the direction `_enforce_placement` SPILLS in: the read preferred exactly the
        record eviction had judged least valuable, so the two halves of "one policy, not
        two" pointed opposite ways. Under a uniform importance -- which is what every write
        through `mind.py` carries, since nothing on the forward path sets one -- that made a
        resident set of primers a permanent wall in front of everything written afterwards.
        `episodic_store.py::_tie_break` carries the measurement; this is the same repair in
        the store that replaces that stub. Eviction's own ordering is untouched: it sorts in
        `_enforce_placement`/`_enforce_membership` with their own keys, not through this method.
        """
        meta = self._index.get(record.key)
        if meta is not None:
            score = self._score_meta(meta, now)
        else:
            bonus = GPU_RESIDENT_BONUS if record.residency is Residency.GPU else 0.0
            score = (
                record.importance
                + bonus
                - staleness_penalty(record.last_accessed, now, self.half_life_s)
            )
        return (-score, -record.written_at, record.key)

    def _check_query(self, query: Tensor) -> None:
        """Hold a read's query to the same shape rule `learn` holds a record's latent to."""
        if not torch.is_floating_point(query):
            raise ValueError(
                f"retrieve() query has dtype {query.dtype}, which is not floating point."
            )
        if query.dim() != 1:
            raise ValueError(f"retrieve() query must be 1-D [D], got {tuple(query.shape)}")
        if self._dim is not None and int(query.shape[0]) != self._dim:
            raise ValueError(
                f"retrieve() query width {query.shape[0]} disagrees with this store's "
                f"established width {self._dim} (a query is compared against records)."
            )

    def _rank(
        self, matches: list[StoredRecord], query: Tensor | None, now: float
    ) -> list[StoredRecord]:
        """Order one partition's matches best-first: cosine to `query`, then `_rank_key`.

        Cosine on the DIRECTION only: a record is a mean-pooled `z_N`, whose norm carries
        turn length and workspace confidence rather than relevance. `_rank_key` stays as the
        tie-break, so two records the query cannot separate are still separated by the same
        eviction score the rest of this store is ordered by.

        Args:
            matches: The partition's records; non-empty.
            query: `[D]` float, or `None` for the score-only order.
            now: The scoring tick, shared with `_rank_key`.

        Returns:
            `matches`, ordered best-first.
        """
        if query is None:
            return sorted(matches, key=lambda record: self._rank_key(record, now))
        self._check_query(query)
        with torch.no_grad():
            bank = torch.stack([record.latent for record in matches]).to(torch.float32)
            vector = query.detach().to(dtype=torch.float32, device=bank.device)
            similarity = torch.cosine_similarity(bank, vector.unsqueeze(0), dim=1)
        scored = list(zip(similarity.tolist(), matches))
        scored.sort(key=lambda pair: (-pair[0], *self._rank_key(pair[1], now)))
        return [record for _similarity, record in scored]

    def score(self, key: RecordKey, *, now: float | None = None) -> float:
        """The eviction score of one record, exposed so a gate can construct a known ordering.

        Args:
            key: `(scope_key, domain, logical_key)`.
            now: Scoring tick; defaults to the current wall clock.

        Returns:
            The record's score.

        Raises:
            KeyError: the key is not in the store.
        """
        with self._lock:
            meta = self._index[key]
            return self._score_meta(meta, time.time() if now is None else now)

    def _set_residency(self, key: RecordKey, residency: Residency) -> None:
        """Move a record's residency tag in both the index and the backend. Moves no bytes."""
        meta = self._index[key]
        if meta.residency is not residency:
            self._tally(meta, -1)
            meta.residency = residency
            self._tally(meta, +1)
        record = self._backend.get(key)
        if record is not None:
            record.residency = residency
            self._backend.put(record)

    def _enforce_placement(self, now: float) -> int:
        """Spill the lowest-scored resident records until the byte and item budgets hold.

        PLACEMENT ONLY. This method's bound is `capacity_bytes`, which is derived from live
        VRAM and is therefore different on every card by design -- that is what makes the
        capacity dynamic, and it is the right kind of host dependence: which tier a span sits
        in costs LATENCY. It must never decide EXISTENCE, so nothing here deletes, and this
        method does not call `_enforce_membership`. See DEC-63 as amended 2026-09-07.

        One sort, not one scan per victim: the score of a record does not change while this
        runs (`now` is fixed), so the spill order is a single ascending sort by
        `(score, written_at, key)` -- lowest score first, ties to the older record, then to
        the lexicographically smaller key. That is `pick_spill_victim`'s order applied
        repeatedly, computed once.

        Args:
            now: The scoring tick.

        Returns:
            How many records were demoted to disk.
        """
        capacity = int(self._capacity_provider())
        if (
            self._resident_bytes <= capacity
            and self._resident_count <= self.tier_budget.ram_max_items
        ):
            return 0
        resident = [(k, m) for k, m in self._index.items() if m.residency in GPU_TIERS]
        total = self._resident_bytes

        order = sorted(
            resident,
            key=lambda km: (self._score_meta(km[1], now), km[1].written_at, km[0]),
        )
        spilled = 0
        count = len(resident)
        for key, meta in order:
            if total <= capacity and count <= self.tier_budget.ram_max_items:
                break
            self._set_residency(key, Residency.DISK)
            total -= meta.span_bytes
            count -= 1
            spilled += 1
        return spilled

    def _enforce_membership(self) -> int:
        """Hard-delete the lowest-importance records once the DECLARED record ceiling overflows.

        Section 1.3 clause (2): *"overflow at the lower tier HARD-DELETES the lowest-importance
        rows -- the only data-loss path in the store"* (`tiered.rs:180-198`). Importance, not
        the full score: upstream prunes on importance, the GPU bonus is deliberately excluded,
        and neither `residency` nor `span_bytes` appears in the key. The SCORE was always
        clean; the TRIGGER was not.

        WHAT THE TRIGGER USED TO BE, AND WHY IT WAS WRONG (DEC-63 as amended 2026-09-07). The
        overflow was `disk_count - disk_max_items`, and `disk_count` is the spill rate, which
        follows `capacity_bytes`, which follows live VRAM. A smaller card spilled more, hit the
        ceiling at a smaller working set, and hard-deleted rows a larger card still held: the
        same input sequence produced a different STORE depending on which card ran it, through
        the one path the store documents as its only data-loss path. Counting every record
        against a declared ceiling makes deletion a function of the workload and the
        configuration alone. Demotion under VRAM pressure is untouched (DEC-70 ratifies it, and
        it is correct: demotion costs latency, not membership).

        Returns:
            How many records were deleted.
        """
        overflow = len(self._index) - self.tier_budget.max_records
        if overflow <= 0:
            return 0
        order = sorted(
            self._index.items(), key=lambda km: (km[1].importance, km[1].written_at, km[0])
        )
        for key, _meta in order[:overflow]:
            self._backend.delete(key)
            self._index_drop(key)
        return overflow

    def evict(self) -> int:
        """Run one eviction pass at the current tick.

        Membership first, then placement: what the store holds is decided before where it sits,
        so a spill never feeds a deletion. Both are invoked here explicitly rather than one
        calling the other -- an eviction pass is a caller's request for both, not evidence that
        they are one policy.

        Returns:
            How many records were hard-deleted or spilled to disk.

        Raises:
            StoreBackpressureError: `max_in_flight` operations are already in flight.
        """
        with self._admission.slot("evict"), self._lock:
            deleted = self._enforce_membership()
            return deleted + self._enforce_placement(time.time())

    # -- residency ---------------------------------------------------------------------

    def mark_gpu(self, scope: Scope, domain: str, logical_key: str) -> None:
        """Tag a record's span as GPU-resident. Records residency; MOVES NO BYTES.

        `tiered.rs:99-106`'s `mark_gpu`, ported: the effect is on the eviction score (the
        `+1.0` bonus), not on where anything lives.

        Args:
            scope: A `Scope` from `derive_scope`.
            domain: The record's domain.
            logical_key: The record's logical key.

        Raises:
            KeyError: no such record.
            StoreScopeError: G32 -- a scope the request supplied.
        """
        key = (partition_scope_key(scope), domain, logical_key)
        with self._lock:
            if key not in self._index:
                raise KeyError(f"mark_gpu: no record at {key}")
            self._set_residency(key, Residency.GPU)

    def promote(self, scope: Scope, domain: str, logical_key: str) -> None:
        """Return a spilled span to RAM -- `promote_returns_span_to_ram`, caller-invoked.

        Promotion re-enters the resident set, so the capacity is re-enforced immediately
        afterwards: promoting into a full store spills something else rather than
        overcommitting.

        Args:
            scope: A `Scope` from `derive_scope`.
            domain: The record's domain.
            logical_key: The record's logical key.

        Raises:
            KeyError: no such record.
            StoreScopeError: G32 -- a scope the request supplied.
            StoreBackpressureError: `max_in_flight` operations are already in flight.
        """
        key = (partition_scope_key(scope), domain, logical_key)
        with self._admission.slot("promote"), self._lock:
            if key not in self._index:
                raise KeyError(f"promote: no record at {key}")
            now = time.time()
            self._index[key].last_accessed = now
            self._set_residency(key, Residency.RAM)
            # Placement only. A promotion admits no record, so membership cannot have changed.
            self._enforce_placement(now)

    # -- DEC-64's session bounding -----------------------------------------------------

    def bind_session(self, scope: Scope) -> Scope:
        """Bound the `session` sub-segment server-side -- DEC-64's *"the server MAY bound"*.

        A request names its session; the server decides what a session id may be. Bounding is
        length and character set, applied here so that an unbounded client string cannot
        become an unbounded key space: without it, `session` is a free partition axis a caller
        can inflate at will, which is the unbounded-growth shape section 9.9 B2 refuses.

        Args:
            scope: A `Scope` from `derive_scope`.

        Returns:
            A `Scope` with the same principal and a bounded session (unchanged when the
            session is already legal, `None` when there is none).

        Raises:
            StoreScopeError: G32 -- a scope the request supplied rather than derived.
            StoreBackpressureError: `max_in_flight` operations are already in flight.
        """
        with self._admission.slot("bind_session"):
            partition_scope_key(scope)  # G32 shape check; the derived key is not needed here
            if scope.session is None:
                return scope
            bounded = _SESSION_ALLOWED.sub("_", scope.session)[:_SESSION_MAX_LEN]
            return Scope(principal=scope.principal, session=bounded)

    # -- DEC-66, still deferred ---------------------------------------------------------

    def consolidate(self) -> None:
        """DEC-66: v1 is prune-only; learned consolidation is a phase-3 candidate.

        Raises:
            ContractGap: always, naming `"DEC-66"`. E1 implements the container, and DEC-66
                is a decision to defer rather than an unbuilt intention -- so this stays a
                gap on the built store too, and says which decision put it there.
        """
        raise ContractGap("DEC-66", "consolidate(): learned consolidation (deferred to phase 3)")

    # -- E0 protocol aliases -------------------------------------------------------------

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
        """`EpisodicStore.write` -- the protocol's name for `learn`.

        Args:
            scope: A `Scope` from `derive_scope`, or `None`.
            domain: One value of `domain_enum`.
            logical_key: Identifies the episode within `(scope, domain)`.
            latent: `[D]` float tensor.
            importance: Overrides `importance_default`.
            provenance: Sidecar text.

        Returns:
            The `WriteReceipt` `learn` returned.
        """
        return self.learn(
            scope,
            domain,
            logical_key,
            latent,
            importance=importance,
            provenance=provenance,
        )

    def read(
        self,
        scope: Scope | None,
        *,
        domain: str | None = None,
        global_query: bool = False,
        b_store: int,
        query: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """`EpisodicStore.read` -- the protocol's name for `retrieve`.

        Args:
            scope: A `Scope` from `derive_scope`, or `None`.
            domain: Restrict to one domain, or `None` with `global_query=True`.
            global_query: Must be `True` when `domain` is `None`.
            b_store: Workspace store budget.
            query: `[D]` float, what this read is looking for; `None` for the score-only
                order.

        Returns:
            `(latents [b_store, D], mask [b_store])`.
        """
        return self.retrieve(
            scope, domain=domain, global_query=global_query, b_store=b_store, query=query
        )

    # -- receipts -------------------------------------------------------------------------

    def capacity_receipt(self, decision: CapacityDecision) -> dict[str, object]:
        """Stamp a capacity decision with what this store did with it.

        Args:
            decision: The `CapacityDecision` this store's provider is returning.

        Returns:
            The decision's dict, plus the store's resident bytes, record count and the
            half-life that scored its evictions -- the fields section 8 gap (a) says close the
            staleness question by measurement.
        """
        payload = dict(decision.to_dict())
        payload["resident_bytes"] = self.resident_bytes
        payload["records"] = self._backend.count()
        payload["half_life_s"] = self.half_life_s
        payload["backend_durability"] = self._backend.durability
        return payload
