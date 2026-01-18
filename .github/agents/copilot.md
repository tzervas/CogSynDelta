---
description: "GitHub Copilot agent instructions for CogSynDelta project"
---

# CogSynDelta Copilot Agent

You are assisting with the CogSynDelta project - a PCN-VAE-GAN hybrid self-improving AI system.

## Project Context

- **Language**: Python 3.14+
- **Package Manager**: uv (not pip)
- **Framework**: PyTorch 2.9+, FastAPI
- **Testing**: pytest with pytest-asyncio
- **Type Checking**: mypy (strict mode)
- **Linting**: ruff

## Key Files

- `memory/constitution.md` - Project principles (READ FIRST)
- `AGENTS.md` - Agent development guidelines
- `docs/adr/` - Architecture Decision Records
- `ROADMAP.md` - Current project status
- `CHANGELOG.md` - Version history

## Coding Standards

### Docstrings (Google Style - MANDATORY)

```python
def function_name(param1: str, param2: int) -> bool:
    """Brief description of what the function does.

    Longer description explaining the why - rationale for this approach,
    edge cases handled, and any non-obvious behavior.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is empty.

    Example:
        >>> function_name("test", 42)
        True
    """
```

### Type Hints (REQUIRED)

All functions must have complete type annotations:

```python
from typing import TypeVar, Generic
from collections.abc import Sequence

T = TypeVar("T")

def process_items(items: Sequence[T], transform: Callable[[T], T]) -> list[T]:
    ...
```

### Imports

```python
# Standard library
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

# Third-party
import torch
import torch.nn as nn

# Local
from cogsyndelta.core import IntegratedSystem

if TYPE_CHECKING:
    from cogsyndelta.memory import ActiveMemory
```

## Commands

### Running Tests
```bash
uv run pytest tests/ -v
```

### Type Checking
```bash
uv run mypy src/
```

### Linting
```bash
uv run ruff check src/ tests/
```

### Quality Check
```bash
uv run python scripts/quality_control.py src/ --fail-under 90
```

## Architecture

```
src/cogsyndelta/
├── core/           # PCN-VAE-GAN, VL-JEPA, mHC, Interconnect
├── memory/         # Tiered memory (active → short → long)
├── agents/         # Self-improving agents
├── api/            # FastAPI server, Google ADK adapter
├── quantum/        # Quantum computing (stubbed)
└── optimization/   # CUDA kernels
```

## Important Notes

1. **Quantum features are STUBBED** - Real quantum backends blocked by Python 3.14 compatibility
2. **Memory "skip silently"** - Intentional graceful degradation in `_ensure_temporal_continuity()`
3. **No unsubstantiated claims** - Performance claims require benchmark validation
4. **Conventional commits** - Use `type(scope): description` format

## Before Submitting Code

- [ ] Docstrings on all public functions (Google style)
- [ ] Type hints on all functions
- [ ] Tests for new functionality
- [ ] `uv run ruff check .` passes
- [ ] `uv run mypy src/` passes
- [ ] No trailing whitespace
- [ ] Conventional commit message
