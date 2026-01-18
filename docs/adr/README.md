# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for CogSynDelta.

## What is an ADR?

An Architecture Decision Record captures an important architectural decision made along with its context and consequences. ADRs help:

- Document the "why" behind technical decisions
- Provide context for future developers
- Track the evolution of the architecture
- Enable informed reconsideration of past decisions

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](0001-use-uv-package-manager.md) | Use UV Package Manager | Accepted | 2026-01-18 |
| [0002](0002-tiered-memory-architecture.md) | Tiered Memory Architecture | Accepted | 2026-01-18 |
| [0003](0003-quantum-module-stubbing.md) | Quantum Module Stubbing | Accepted | 2026-01-18 |
| [0004](0004-graceful-degradation-patterns.md) | Graceful Degradation Patterns | Accepted | 2026-01-18 |
| [0005](0005-conventional-commits.md) | Conventional Commits Standard | Accepted | 2026-01-18 |
| [0006](0006-google-style-docstrings.md) | Google Style Docstrings | Accepted | 2026-01-18 |

## Creating New ADRs

1. Copy `templates/adr-template.md` to this directory
2. Name it `NNNN-title-with-dashes.md` (next sequential number)
3. Fill in all sections
4. Submit PR for review
5. Update this index

## ADR Statuses

- **Proposed**: Under discussion
- **Accepted**: Decision made and in effect
- **Deprecated**: No longer applies but preserved for history
- **Superseded**: Replaced by a newer ADR (link to replacement)

## References

- [Michael Nygard's ADR article](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR GitHub organization](https://adr.github.io/)
