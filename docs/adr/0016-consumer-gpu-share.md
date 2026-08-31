# ADR-0016: MIG-like GPU share on consumer cards

**Status**: Proposed (horizon). Do not implement while Phase 1 product
goals are open.
**Date**: 2026-08-31
**Decision Makers**: tzervas
**Technical Story**: `docs/program/GPU-SHARE.md` (`G-SHARE`, `G-1080`)

## Context

The lab wants up to **256 microscopic specialized models** resident at
once **when they fit**, with hybrid split/share: intern identical
read-only weights (disk symlink / same inode; GPU content-hash tensor
pool), isolate only mutables (KV, unique overlays, scratch contents).
Compose with LAN pooling (`G-POOL`, ADR-0015). Cards are GeForce
(3090 Ti, 5080, later 1080 Ti). None implement NVIDIA MIG.

A disk symlink does not dedupe VRAM across processes. Exclusive
time-share (today's lock) cannot run a matrix of tiny specialists
concurrently.

## Decision

The system will, when unblocked:

1. Load many tiny specialists into a **read-only interned weight pool**
   (content hash). Identical tensors occupy VRAM once; unique tensors
   stay private. Writable buffers never intern (CoW or refuse).
2. Disk: symlink/hardlink/mmap the same file. GPU: intern in one
   process (or CUDA IPC). Isolation is KV + unique adapters, not
   duplicate weights. **Parallelize** ops that hit the same interned
   block (batched GEMM), not 256 serial forwards.
3. Approximate leftover MIG-like SM/memory caps with CUDA MPS / green
   contexts **after** intern is measured. Probe; do not assume Pascal
   or Ampere green contexts.
4. Compose with `G-POOL`: tight autodev decode stays exclusive on the
   3090; tiny `ok-lan` specialists may share a card or the pool.
5. Tag a future **1080 Ti** (`pascal` / `sm_61` / `11gb`) in the 5080
   workstation only after `nvidia-smi` sees it. Factory cooler first.
6. Never call this MIG. Never schedule sm_120 work on Pascal/Ampere.

## Rationale

### Why This Approach

Intern-by-hash plus private overlays is how 256 *different* tiny
models can fit: pay unique bytes, not n × sizeof(model). Disk
symlink is necessary but not sufficient. MPS is only the extra
process-cap layer. Capability tags already exist (ADR-0015).

### Alternatives Considered

#### Option 1: Wait for MIG-capable hardware

- **Pros**: Real isolation.
- **Cons**: Not the cards we have.
- **Why Rejected**: Operator asked for closest-robust on this lab.

#### Option 2: 256 independent llama-server processes

- **Pros**: Process isolation.
- **Cons**: 256 VRAM uploads even if files are symlinked.
- **Why Rejected**: Symlink is host-side; GPU copies unless interned.

#### Option 3: Only time-slice (status quo locks)

- **Pros**: Simple, already live.
- **Cons**: No concurrent matrix eval.
- **Why Rejected**: Does not meet the 256-tenant goal. Keep as fallback
  for exclusive CUDA.

## Consequences

### Positive

- Matrix eval / mixed-effort experiments become possible without a
  datacenter GPU.
- 1080 Ti can join as a tagged leftover node instead of sitting in a box.

### Negative

- MPS is not a fault or security boundary.
- **Mitigation**: isolation tests; do not mix untrusted tenants with
  autodev; KV quotas.
- Will not match real MIG quality.
- **Mitigation**: honest STATUS rows; no marketing claims.

### Neutral

- Autodev 14B remains exclusive on the 3090.

## Implementation

Horizon pack: `docs/program/GPU-SHARE.md`. Catalog stub:
`future_hosts.gpu5080-1080ti`. Router must ignore `future_hosts` until
promoted. Autoloop must not start `G-SHARE` / `G-1080`.

## References

- ADR-0015 LAN GPU inference pool
- NVIDIA MPS documentation (memory limit / thread percentage)
- llama.cpp parallel slots / server slot API
