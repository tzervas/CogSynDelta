# Implementation Plan: [FEATURE NAME]

**Spec**: [specs/feature-name/spec.md](../specs/feature-name/spec.md)
**Created**: [DATE]
**Status**: Draft | In Progress | Complete
**Author**: [NAME]
**Estimated Effort**: [T-shirt size: S/M/L/XL or days]

## Overview

[Brief summary of what's being implemented and the approach]

## Prerequisites

- [ ] [Dependency 1 - e.g., ADR approved]
- [ ] [Dependency 2 - e.g., spec reviewed]
- [ ] [Dependency 3 - e.g., baseline benchmarks captured]

## Implementation Phases

### Phase 1: [Foundation/Setup] (Est: [X days])

**Goal**: [What this phase achieves]

**Changes**:
1. [File/component]: [What changes and why]
2. [File/component]: [What changes and why]

**Validation**:
- [ ] [How to verify this phase is complete]
- [ ] [Tests that should pass]

### Phase 2: [Core Implementation] (Est: [X days])

**Goal**: [What this phase achieves]

**Changes**:
1. [File/component]: [What changes and why]
2. [File/component]: [What changes and why]

**Validation**:
- [ ] [How to verify this phase is complete]
- [ ] [Tests that should pass]

### Phase 3: [Integration/Polish] (Est: [X days])

**Goal**: [What this phase achieves]

**Changes**:
1. [File/component]: [What changes and why]

**Validation**:
- [ ] [How to verify this phase is complete]
- [ ] [All acceptance criteria from spec met]

## Technical Approach

### Architecture Changes

[Describe any architectural changes, with diagrams if helpful]

```
[Component] --> [New Component] --> [Existing Component]
```

### Key Design Decisions

1. **[Decision]**: [Rationale]
2. **[Decision]**: [Rationale]

### Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| [Risk 1] | High/Med/Low | High/Med/Low | [How to address] |
| [Risk 2] | High/Med/Low | High/Med/Low | [How to address] |

## Testing Strategy

### Unit Tests
- [What new unit tests are needed]

### Integration Tests
- [What integration tests are needed]

### Benchmark Validation
- [What benchmarks validate this change]
- [Expected performance impact]

## Rollout Plan

1. **Feature Branch**: All development in `feat/[feature-name]`
2. **Code Review**: PR requires [N] approvals
3. **Staging**: Deploy to staging, run full test suite
4. **Benchmarks**: Compare against baseline, document deltas
5. **Main**: Merge after all validations pass

## Documentation Updates

- [ ] Update [README.md] if user-facing
- [ ] Update [docs/ARCHITECTURE.md] if architectural
- [ ] Update [CHANGELOG.md] with changes
- [ ] Create ADR if decision is significant

## Related

- Spec: [link]
- Tasks: [specs/feature-name/tasks.md](../specs/feature-name/tasks.md)
- ADRs: [list relevant ADRs]
- Issues: #[issue numbers]

---

*Template based on [GitHub Spec-Kit](https://github.com/github/spec-kit)*
