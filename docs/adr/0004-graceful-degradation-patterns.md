# ADR-0004: Graceful Degradation Patterns

**Status**: Accepted
**Date**: 2026-01-18
**Decision Makers**: @tzervas
**Technical Story**: System resilience and fault tolerance

## Context

CogSynDelta operates as a complex system with multiple interdependent components:
- Memory tiers (active, short-term, long-term)
- External services (potentially)
- Quantum backends (when available)
- GPU acceleration (when available)

In production, various components may be unavailable or fail:
- Memory entries may be archived/compacted and no longer directly accessible
- GPU may not be available (CPU fallback needed)
- Quantum libraries unavailable (see ADR-0003)
- Network services may timeout

The system must continue functioning despite partial failures.

## Decision

We will implement **graceful degradation patterns** across the codebase:

1. **Silent skip for non-critical operations** when data is unavailable
2. **Fallback implementations** for optional accelerators
3. **Clear logging** for debugging without user-facing errors
4. **Bounded retry** with exponential backoff for transient failures

### Specific Pattern: Memory Temporal Continuity

In `_ensure_temporal_continuity()`, when a memory ID needed for temporal context no longer exists:

```python
for memory_id in continuity_ids:
    if memory_id not in active_ids:
        try:
            embedding, _ = self.retrieve(memory_id)
            if len(self.active_memory) < self.active_capacity * 1.2:
                self.active_memory[memory_id] = embedding
        except KeyError:
            # Memory not found in any tier - skip silently
            # WHY: Temporal continuity is best-effort. The memory may have been
            # archived, compacted, or deleted. Failing here would break the
            # entire memory system for a non-critical optimization.
            pass
```

This is **intentional**, not a bug.

## Rationale

### Why Silent Skip (for this case)

1. **Temporal continuity is an optimization**, not a requirement
2. **Memory lifecycle**: Memories are legitimately archived/deleted over time
3. **System stability**: Raising exceptions here would cascade failures
4. **Logging exists**: Debug logs capture skipped memories for troubleshooting

### Alternatives Considered

#### Option 1: Raise exception on missing memory

- **Pros**: Explicit failure, easier to debug
- **Cons**: System stops working when any memory is archived
- **Why Rejected**: Would make the system unusable over time

#### Option 2: Return placeholder/null object

- **Pros**: No exception, maintains contract
- **Cons**: Pollutes active memory with invalid data
- **Why Rejected**: Could cause downstream issues

#### Option 3: Log warning on every skip

- **Pros**: Visible in logs
- **Cons**: Log spam during normal operation
- **Why Rejected**: Too noisy; archived memories are expected

## Consequences

### Positive

- System remains stable during partial failures
- Memory lifecycle (creation → archival → deletion) works naturally
- GPU/quantum fallbacks are transparent to callers

### Negative

- Silent failures can hide bugs if not properly bounded
- Mitigation: Clear documentation, specific exception types, debug logging
- Mitigation: Periodic health checks that validate system state

### Neutral

- Pattern requires discipline to apply consistently
- Each graceful degradation point must document WHY it's appropriate

## Implementation

### Documentation Requirement

Every graceful degradation point must include a comment explaining:
1. **What** is being skipped/degraded
2. **Why** this is acceptable (not a bug)
3. **When** this code path is expected to execute

### Code Pattern

```python
try:
    result = potentially_failing_operation()
except SpecificException:
    # GRACEFUL DEGRADATION: [component]
    # WHY: [rationale for why this is acceptable]
    # WHEN: [conditions under which this occurs normally]
    logger.debug("Skipped operation: %s", context)
    result = fallback_value  # or continue/pass
```

## References

- `src/cogsyndelta/memory/active_memory.py` - Memory skip example
- `src/cogsyndelta/quantum/quantum_compute.py` - Quantum stubbing
- Netflix's "Chaos Engineering" principles
- Google SRE book on graceful degradation
