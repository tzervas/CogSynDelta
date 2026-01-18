# Development Standards

> Comprehensive guide to CogSynDelta development practices, requirements, and conventions.

## Overview

This document defines the opinionated requirements and standards for contributing to CogSynDelta. All contributors (human and AI agents) must follow these guidelines.

**Philosophy**: Every piece of code must answer **What**, **Why**, and **How**.

---

## Code Standards

### Python Version

- **Required**: Python 3.14+
- Use modern syntax (match statements, type union `|`, etc.)
- No compatibility shims for older Python versions

### Type Hints

**All functions must have complete type annotations.**

```python
# ✅ Good
def process_memory(
    memory_id: str,
    embedding: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    ...

# ❌ Bad - missing types
def process_memory(memory_id, embedding, metadata=None):
    ...
```

### Docstrings (Google Style)

**All public functions, classes, and modules must have docstrings.**

```python
def retrieve_memory(memory_id: str, threshold: float = 0.7) -> torch.Tensor:
    """Retrieve a memory embedding by ID with similarity threshold.

    Searches through memory tiers (active → short-term → long-term) to find
    the requested memory. Returns the embedding if found and similarity
    exceeds threshold.

    Why: Tiered search ensures hot data is checked first for performance,
    while still allowing access to archived memories when needed.

    Args:
        memory_id: Unique identifier for the memory to retrieve.
        threshold: Minimum similarity score (0.0-1.0) for retrieval.
            Defaults to 0.7 for balanced precision/recall.

    Returns:
        The memory embedding tensor if found and above threshold.

    Raises:
        KeyError: If memory_id not found in any tier.
        ValueError: If threshold not in valid range.

    Example:
        >>> embedding = retrieve_memory("mem_123", threshold=0.8)
        >>> embedding.shape
        torch.Size([768])
    """
```

### Line Length

- **Maximum**: 100 characters
- Enforced by ruff

### Imports

```python
# Order: stdlib → third-party → local
# Sorted alphabetically within groups

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import torch
import torch.nn as nn
from fastapi import FastAPI

from cogsyndelta.core import IntegratedSystem
from cogsyndelta.memory import ActiveMemory

if TYPE_CHECKING:
    from cogsyndelta.agents import SelfImprovingAgent
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `ActiveMemoryManager` |
| Functions | snake_case | `retrieve_memory` |
| Variables | snake_case | `memory_id` |
| Constants | UPPER_SNAKE | `MAX_MEMORY_SIZE` |
| Private | `_prefix` | `_internal_cache` |
| Type Variables | Single uppercase or descriptive | `T`, `MemoryT` |

---

## Git Standards

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body explaining WHAT and WHY]

[optional footer with refs]
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`

**Scopes**: `core`, `memory`, `agents`, `api`, `quantum`, `benchmarks`, `tests`, `ci`, `docs`

### Branch Naming

```
<type>/<description>
```

Examples:
- `feat/add-memory-compression`
- `fix/api-timeout-handling`
- `docs/update-architecture`

### Branching Strategy

See `docs/BRANCHING_STRATEGY.md` for full details.

```
main        →  Production (protected, signed commits required)
  ↑
staging     →  Pre-release testing (protected)
  ↑
develop     →  Development integration (protected)
  ↑
feat/*      →  Feature branches (unprotected)
```

---

## Documentation Standards

### What Must Be Documented

1. **All public APIs** - Functions, classes, modules
2. **Configuration options** - In config files and code
3. **Architecture decisions** - Via ADRs in `docs/adr/`
4. **Setup procedures** - In INSTALL.md and README.md

### Documentation Philosophy

Every piece of documentation must answer:

1. **What**: What does this do? What is it?
2. **Why**: Why does it exist? Why this approach?
3. **How**: How do you use it? How does it work?

### No Unsubstantiated Claims

❌ **Bad**: "This is 10x faster than the previous implementation."
✅ **Good**: "Benchmarks show 10x improvement (see `benchmark_results/compression_benchmark.json`)."

All performance claims must reference:
- Benchmark results
- Test data
- Reproducible methodology

---

## Testing Standards

### Test Categories

| Category | Location | Purpose |
|----------|----------|---------|
| Unit | `tests/test_unit.py` | Isolated component tests |
| Integration | `tests/test_comprehensive.py` | Cross-component tests |
| API | `tests/test_api.py` | Endpoint tests |
| Benchmarks | `benchmarks/` | Performance validation |

### Test Requirements

- All new features require tests
- Bug fixes require regression tests
- Performance claims require benchmark validation
- Minimum 80% coverage for new code

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ -v --cov=cogsyndelta --cov-report=term

# Specific test file
uv run pytest tests/test_memory.py -v
```

---

## Quality Gates

All PRs must pass these checks:

### Automated (CI)

- [ ] `ruff check` - No linting errors
- [ ] `ruff format --check` - Code formatted
- [ ] `mypy --strict` - Type checking passes
- [ ] `pytest` - All tests pass
- [ ] `quality_control.py --fail-under 90` - Quality score ≥ 90

### Manual (Review)

- [ ] Docstrings on all public functions
- [ ] "Why" explained for non-obvious code
- [ ] No unsubstantiated claims
- [ ] CHANGELOG updated (for features/fixes)
- [ ] ADR created (for architectural changes)

---

## Pre-commit Hooks

The project uses pre-commit for automated checks. Install with:

```bash
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

### Hooks Enabled

1. **Trailing whitespace** - Auto-fixed
2. **End of file fixer** - Auto-fixed
3. **YAML/TOML/JSON check** - Syntax validation
4. **Large file check** - Prevents accidental large commits
5. **Merge conflict check** - Blocks conflicted files
6. **Private key detection** - Security
7. **Ruff lint** - With auto-fix
8. **Ruff format** - Code formatting
9. **Mypy** - Type checking
10. **Conventional commits** - Commit message validation

---

## Security Standards

### Secrets

- **Never commit secrets** to the repository
- Use environment variables or `.env` files (gitignored)
- Pre-commit hook detects private keys

### Dependencies

- All dependencies must have compatible licenses (see `LICENSES/LICENSE_TRACKER.md`)
- Security scanning enabled in CI (CodeQL when enabled)
- Dependency review on PRs

---

## Performance Standards

### Benchmarks

All performance-critical code should have benchmarks in `benchmarks/`.

```bash
# Run benchmarks
uv run cogsyndelta-benchmark

# Results saved to benchmark_results/
```

### Targets

| Metric | Target | Measured By |
|--------|--------|-------------|
| API response time | < 200ms p95 | `benchmarks/` |
| Memory footprint | Bounded active memory | Unit tests |
| Startup time | < 5s to healthy | Integration tests |

---

## Architecture Decision Records (ADRs)

Significant technical decisions must be documented as ADRs.

### When to Create an ADR

- New major component or system
- Significant dependency addition
- Breaking API changes
- Performance optimization approach
- Security-related decisions

### ADR Template

Use `templates/adr-template.md` and place in `docs/adr/`.

---

## AI Agent Guidelines

For AI coding agents (Copilot, Claude, Cursor, etc.):

1. **Read the constitution first**: `memory/constitution.md`
2. **Check existing ADRs**: `docs/adr/`
3. **Follow docstring format exactly**: Google style with "Why" section
4. **Run tests before submitting**: `uv run pytest tests/ -v`
5. **Use conventional commits**: `type(scope): description`

See `AGENTS.md` for comprehensive agent guidelines.

---

## Quick Reference

### Essential Commands

```bash
# Setup
uv sync --group dev

# Testing
uv run pytest tests/ -v

# Linting
uv run ruff check src/ tests/

# Type checking
uv run mypy src/

# Quality score
uv run python scripts/quality_control.py src/ --fail-under 90

# Pre-commit (all files)
uv run pre-commit run --all-files
```

### Key Files

| File | Purpose |
|------|---------|
| `AGENTS.md` | AI agent guidelines |
| `memory/constitution.md` | Project principles |
| `docs/BRANCHING_STRATEGY.md` | Git workflow |
| `docs/adr/` | Architecture decisions |
| `LICENSES/LICENSE_TRACKER.md` | Dependency licenses |
| `.gitmessage` | Commit template |

---

*Last Updated: January 18, 2026*
