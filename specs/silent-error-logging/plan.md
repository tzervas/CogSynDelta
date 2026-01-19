# Implementation Plan: Silent Error Logging Infrastructure

## Overview

Implement configurable logging infrastructure for graceful degradation points in the memory subsystem, providing visibility into silent skips while maintaining system stability.

## Phase 1: Core Infrastructure (Week 1)

**Duration**: 1 week
**Goal**: Establish logging foundation and configuration system

**Activities**:
1. Create LogConfig singleton for level management
2. Implement SkipMetrics counter class
3. Define JSON log schema and output format
4. Add environment variable configuration
5. Create basic logging infrastructure

**Deliverables**:
- `src/cogsyndelta/core/logging_config.py` - Configuration management
- `src/cogsyndelta/core/skip_metrics.py` - Metrics aggregation
- Unit tests for configuration system

### Key Decisions

- **Log Levels**: DEV (default, metrics only), DEBUG (detailed logging)
- **Output Format**: JSON Lines (JSONL) for easy parsing
- **Configuration**: Environment variable `COGSYNDELTA_LOG_LEVEL`
- **Storage**: Automatic log directory creation

## Phase 2: Memory Instrumentation (Week 2)

**Duration**: 1 week
**Goal**: Instrument all identified graceful degradation points

**Activities**:
1. Instrument `_ensure_temporal_continuity()` in tiered_memory.py
2. Instrument `_archive_to_long_term()` in tiered_memory.py
3. Instrument `_load_from_archive()` in memory_persistence.py
4. Add context information to log entries
5. Test logging in both DEV and DEBUG modes

**Deliverables**:
- Modified memory modules with logging instrumentation
- Comprehensive test coverage for logging behavior
- Log file validation (JSON schema compliance)

### Instrumentation Points

Per ADR-0004, the following points need logging:
1. **Temporal continuity skips**: KeyError on missing memory IDs
2. **Archive failures**: KeyError/RuntimeError on corrupted entries
3. **Load failures**: FileNotFoundError/pickle errors
4. **Future points**: Any additional graceful degradation discovered

## Phase 3: Testing & Validation (Week 3)

**Duration**: 1 week
**Goal**: Comprehensive testing and performance validation

**Activities**:
1. Write unit tests for both DEV and DEBUG logging
2. Create integration tests for metrics aggregation
3. Performance benchmark logging overhead (< 1%)
4. Test log file parsing and analysis
5. Validate JSON schema consistency

**Deliverables**:
- Complete test suite (unit + integration)
- Performance benchmark results
- Log analysis tools/scripts

### Test Scenarios

- **DEV mode**: Verify metrics counting without console output
- **DEBUG mode**: Verify detailed logging with full context
- **Error handling**: Verify logging failures don't crash system
- **Performance**: Benchmark overhead on memory operations

## Phase 4: Documentation & Integration (Week 4)

**Duration**: 1 week
**Goal**: Complete documentation and merge to main branch

**Activities**:
1. Update inline code documentation
2. Create troubleshooting guide section
3. Write developer documentation
4. Create example log analysis scripts
5. Final integration testing

**Deliverables**:
- Updated documentation
- Log analysis utilities
- Migration guide for developers

## Technical Architecture

### LogConfig Class

```python
class LogConfig:
    level: LogLevel = LogLevel.DEV
    output_path: Path = Path("logs/memory_debug.jsonl")
    format: str = "jsonl"

    @classmethod
    def from_env(cls) -> 'LogConfig':
        # Environment variable parsing
```

### SkipMetrics Class

```python
class SkipMetrics:
    _counters: dict[str, int] = defaultdict(int)

    def increment(self, category: str) -> None:
        self._counters[category] += 1

    def get_all(self) -> dict[str, int]:
        return dict(self._counters)
```

### Log Entry Schema

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

## Risk Mitigation

### Performance Risk
- **Mitigation**: Extensive benchmarking to ensure <1% overhead
- **Fallback**: Logging can be disabled entirely if needed

### Stability Risk
- **Mitigation**: All logging operations wrapped in try/except
- **Testing**: Comprehensive error handling tests

### Complexity Risk
- **Mitigation**: Simple, focused implementation
- **Documentation**: Clear usage examples and guidelines

## Dependencies

- **Python**: 3.14+ (for enhanced error handling)
- **Libraries**: Standard library only (json, pathlib, datetime)
- **Testing**: pytest fixtures for log level configuration

## Success Criteria

### Functional Criteria
- [ ] DEV level counts skips without console output
- [ ] DEBUG level logs detailed context for each skip
- [ ] All graceful degradation points instrumented
- [ ] JSON output parseable by standard tools
- [ ] Environment variable configuration works
- [ ] Metrics API provides aggregated counts

### Non-Functional Criteria
- [ ] Logging overhead < 1% of operation time
- [ ] No logging-related system crashes
- [ ] JSON schema consistent across all entries
- [ ] Memory operations remain stable under failure

## Timeline

| Phase | Duration | Activities | Deliverables |
|-------|----------|------------|--------------|
| Infrastructure | 1 week | Config, metrics, basic logging | Core classes, unit tests |
| Instrumentation | 1 week | Memory module changes | Instrumented code, integration tests |
| Testing | 1 week | Comprehensive validation | Test suite, performance benchmarks |
| Documentation | 1 week | Docs, examples, integration | Complete documentation, merge |

**Total Duration**: 4 weeks
**Effort**: ~2-3 developer-weeks

## Resources Required

- **Personnel**: 1 developer with logging and memory system experience
- **Testing**: Access to test environment for validation
- **Documentation**: Technical writing for developer guides

---

*Created: January 18, 2026*
*Last Updated: January 18, 2026*
