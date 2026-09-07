"""Row E1's episodic store: a correct container, and nothing more.

WHAT THIS PACKAGE IS. The build of row E1 (`docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md`
line 2649): *"Build the store: partition, byte-capacity, eviction, lifecycle -- plus the
DYNAMIC-CAPACITY PROBE on both GPUs."* Three files, one per concern:

  - `capacity.py`  -- DEC-63's `capacity_bytes(host, tick)`, section 8 gap (a)'s formula, its
    pre-committed floor branch, and the `nvidia-smi` probe that supplies `VRAM_total`. Pure
    stdlib: it imports no torch, so the GPU CI runner can execute the probe in an image that
    has only a Python and a driver.
  - `backends.py` -- the durability ladder. `InMemoryBackend` is the CONFORMANCE oracle and
    explicitly not a durability claim; `SqliteBackend` is the DURABILITY oracle and enforces
    `journal_mode=WAL` + `synchronous=FULL` at connect time, raising if either is not what it
    asked for. Both satisfy one `StoreBackend` protocol, so one contract suite runs over both.
  - `store.py`    -- `EpisodicStoreImpl`: `(scope, domain, logical_key)` identity with the
    scope segment derived server-side, byte capacity read from a provider on every admission,
    scored eviction, the residency ladder, and the six lifecycle verbs with refusing
    backpressure.

WHAT THIS PACKAGE IS NOT, stated because the row states it. *"This row builds a container and
proves it is a correct container; it makes no claim about usefulness, which is E2's job."*
Concretely, three things are deliberately absent:

  - **The store's two projections are not trained here.** `W_k`, `W_v` (2 x 512 x 512 =
    524,288 parameters) live in `kv_bank.py` as `StoreProjection`; they are white-matter
    parameters and row E2 trains them. Nothing in this package touches them, and nothing here
    imports `StoreProjection`.
  - **The differential encoding, the record format and temporal versioning are not here.** A
    separate design carries those and still has open decisions. This package stores whatever
    latent it is handed, by value in the conformance backend and as a float32 blob in the
    durability backend, with no delta chain and no version graph.
  - **`consolidate()` stays a `ContractGap`.** DEC-66 defers learned consolidation to phase 3;
    v1 is prune-only. E1 building it would be building past its own row.

WHY THE E0 STUB IS NOT MODIFIED. `episodic_store.py` ships E0's `EpisodicStore` protocol,
`InMemoryStoreStub`, `Scope`, `derive_scope` and `ContractGap`, and E0's gate is that its nine
named fixtures run against that stub and fail red on the clauses the stub does not implement.
E1's own gate (i) is that those fixtures go green *and that the diff making them green touches
no test file*. Editing the stub to implement DEC-63 would turn E0's file red, which is the
same failure mode by a different door: a build that rewrites its own gate. So E1 adds a second
implementation of the same protocol beside the stub, leaves E0's module and E0's test file
byte-identical, and re-runs the nine fixtures against `EpisodicStoreImpl` over both durability
oracles in `tests/interconnect/test_e1_store_conformance.py`. `git diff` on E1's own branch
shows `tests/interconnect/test_episodic_store.py` untouched; that was gate (i)'s evidence.

WHAT CHANGED SINCE, AND WHY IT IS NOT THAT MOVE (2026-09-07). The stub and its test file
have since been edited -- deliberately, and for a reason gate (i) does not cover. Gate (i)
forbids making a RED gate green by editing the gate; the 2026-09-07 change fixes a MEASURED
defect in a merged contract (the read returned a constant, so the store contributed no
learnable signal) and adds its own tests, red before the fix and green after. Both stores
carry the repair, because both implement one protocol: the read's tie-break now prefers the
newer record, and `read`/`retrieve` take an optional `query` and rank by cosine against it.
None of E0's nine `ContractGap` fixtures moved.
"""

from __future__ import annotations

from cogsyndelta.interconnect.episodic.backends import (
    InMemoryBackend,
    Residency,
    SqliteBackend,
    StoreBackend,
    StoredRecord,
    TierBudget,
)
from cogsyndelta.interconnect.episodic.capacity import (
    DEFAULT_ACTIVATION_RESERVE_BYTES,
    DEFAULT_KV_BYTES_PER_TOKEN_PER_REGION,
    DEFAULT_SAFETY_MARGIN_BYTES,
    DEFAULT_STORE_FLOOR_BYTES,
    GIB,
    CapacityBranch,
    CapacityDecision,
    HostVram,
    SchedulerBudgets,
    capacity_decision,
    probe_host_vram,
)
from cogsyndelta.interconnect.episodic.store import (
    GPU_RESIDENT_BONUS,
    MAX_IN_FLIGHT,
    EpisodicStoreImpl,
    Lifecycle,
    StoreBackpressureError,
    StoreDurabilityError,
    StoreLifecycleError,
    partition_scope_key,
    staleness_penalty,
)

__all__ = [
    "DEFAULT_ACTIVATION_RESERVE_BYTES",
    "DEFAULT_KV_BYTES_PER_TOKEN_PER_REGION",
    "DEFAULT_SAFETY_MARGIN_BYTES",
    "DEFAULT_STORE_FLOOR_BYTES",
    "GIB",
    "GPU_RESIDENT_BONUS",
    "MAX_IN_FLIGHT",
    "CapacityBranch",
    "CapacityDecision",
    "EpisodicStoreImpl",
    "HostVram",
    "InMemoryBackend",
    "Lifecycle",
    "Residency",
    "SchedulerBudgets",
    "SqliteBackend",
    "StoreBackend",
    "StoreBackpressureError",
    "StoreDurabilityError",
    "StoreLifecycleError",
    "StoredRecord",
    "TierBudget",
    "capacity_decision",
    "partition_scope_key",
    "probe_host_vram",
    "staleness_penalty",
]
