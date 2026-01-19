# Feature Specification: Silent Error Logging Infrastructure

**Feature Branch**: `feat/specs-benchmarks-logging-infrastructure`
**Created**: 2026-01-18
**Status**: In Review
**Author**: CogSynDelta Team

## Summary

Implement configurable logging for graceful degradation points (silent error paths) across the memory subsystem. Currently, per ADR-0004, the system silently skips non-critical operations but provides no visibility into these skips. This feature adds structured logging with configurable verbosity (DEV default, DEBUG override), skip metrics aggregation, and JSON output for automated analysis—enabling investigation, debugging, and future automated error correction.

## User Scenarios & Testing

### User Story 1 - Monitor Silent Skips in Development (Priority: P1)

A developer wants to understand how often graceful degradation is occurring during normal operation to identify potential issues.

**Why this priority**: Without visibility, silent failures accumulate unnoticed and may indicate deeper problems.

**Acceptance Scenarios**:

1. **Given** default DEV logging level, **When** `_ensure_temporal_continuity()` skips a memory ID, **Then** skip is counted in metrics but not logged to console.
2. **Given** DEV level, **When** session ends or metrics are queried, **Then** aggregated skip counts are available (e.g., "temporal_continuity_missing: 42").
3. **Given** `COGSYNDELTA_LOG_LEVEL=DEBUG`, **When** any graceful degradation occurs, **Then** detailed log entry written with context.

### User Story 2 - Debug Silent Failures (Priority: P1)

A developer investigating unexpected behavior wants to see exactly which operations were skipped and why.

**Why this priority**: Critical for debugging—can't fix what you can't see.

**Acceptance Scenarios**:

1. **Given** DEBUG level, **When** memory retrieval fails in any graceful degradation path, **Then** log includes: operation name, memory_id, exception type, timestamp.
2. **Given** log file `logs/memory_debug.jsonl`, **When** parsing with standard tools, **Then** each line is valid JSON with consistent schema.
3. **Given** DEBUG enabled, **When** reviewing logs after a session, **Then** can reconstruct which skips occurred and in what order.

### User Story 3 - Automated Analysis Pipeline (Priority: P2)

A system administrator wants to analyze skip patterns across multiple runs to identify systemic issues.

**Why this priority**: Enables proactive issue detection before they become critical.

**Acceptance Scenarios**:

1. **Given** structured JSON logs, **When** aggregating multiple log files, **Then** can compute skip rates, identify patterns, detect anomalies.
2. **Given** metrics API, **When** querying current session, **Then** returns dict with all skip categories and counts.
3. **Given** threshold configuration, **When** skip rate exceeds threshold, **Then** warning logged (not error—graceful degradation is intentional).

### Edge Cases

- What if log directory doesn't exist? → Create automatically
- What if disk is full? → Fall back to stderr, don't crash
- What if logging itself throws? → Swallow error, system stability > logging

## Requirements

### Functional Requirements

- **FR-001**: System MUST support configurable log levels: DEV (default), DEBUG
- **FR-002**: DEV level MUST count skips without console output
- **FR-003**: DEBUG level MUST log each skip with full context
- **FR-004**: All graceful degradation points in memory module MUST be instrumented
- **FR-005**: Log output MUST be structured JSON (one object per line)
- **FR-006**: System MUST provide SkipMetrics class for aggregated counts
- **FR-007**: Log level MUST be configurable via environment variable
- **FR-008**: Log level MUST be configurable via function argument (for testing)

### Non-Functional Requirements

- **NFR-001**: Logging overhead < 1% of operation time
- **NFR-002**: JSON schema consistent across all log entries
- **NFR-003**: No logging-related crashes (all logging failures handled gracefully)

### Key Entities

- **LogConfig**: Singleton configuration (level, output path, format)
- **SkipMetrics**: Counter dict by skip category with query methods
- **SkipLogEntry**: Structured log entry (timestamp, operation, context, exception)

## Success Criteria

### Measurable Outcomes

- **SC-001**: All 4 identified graceful degradation points instrumented
- **SC-002**: DEBUG logs parseable by `jq` (valid JSONL)
- **SC-003**: Logging adds < 1% overhead (benchmark validated)
- **SC-004**: Tests verify both DEV and DEBUG behavior

## Technical Notes

Per ADR-0004, graceful degradation points identified:
1. `_ensure_temporal_continuity()` in `tiered_memory.py` - KeyError on missing memory
2. `_archive_to_long_term()` in `tiered_memory.py` - KeyError/RuntimeError on corrupted entry
3. `_load_from_archive()` in `memory_persistence.py` - FileNotFoundError/pickle errors
4. Potential others in compression modules

Log schema:
```json
{
  "timestamp": "2026-01-18T12:34:56.789Z",
  "level": "DEBUG",
  "operation": "_ensure_temporal_continuity",
  "category": "temporal_continuity_missing",
  "memory_id": "abc123",
  "exception_type": "KeyError",
  "message": "Memory not found in any tier",
  "context": {"tier_checked": ["active", "short_term", "long_term"]}
}
```

## Related

- ADR: [0004-graceful-degradation-patterns.md](../../docs/adr/0004-graceful-degradation-patterns.md) - documents the pattern this implements
- Constitution: "System handles failures silently when appropriate"
- Depends on: None (foundational infrastructure)

---

*Template based on [GitHub Spec-Kit](https://github.com/github/spec-kit)*
