---
description: "GitHub Copilot agent instructions for CogSynDelta project"
---

# CogSynDelta Copilot Agent

You are assisting with the CogSynDelta project - a PCN-VAE-GAN hybrid self-improving AI system with specialized submodel composition.

## Project Context

- **Language**: Python 3.14+
- **Package Manager**: uv (not pip)
- **Framework**: PyTorch 2.9+, FastAPI
- **Testing**: pytest with pytest-asyncio
- **Type Checking**: mypy (strict mode)
- **Linting**: ruff
- **License**: Proprietary
- **Copyright**: © 2026 Tyler Zervas (tzervas) and Average Joe's Labs (AJL)

---

## Quick Reference Index

### Core Documentation

| Document | Path | Purpose |
|----------|------|---------|
| Constitution | `memory/constitution.md` | Project principles - **READ FIRST** |
| Agent Guide | `AGENTS.md` | Agent development guidelines |
| Roadmap | `ROADMAP.md` | Current project status |
| Changelog | `CHANGELOG.md` | Version history |

### Architecture & Design

| Document | Path | Purpose |
|----------|------|---------|
| Architecture | `docs/ARCHITECTURE.md` | System architecture overview |
| ADRs | `docs/adr/` | Architecture Decision Records |
| API Reference | `docs/API_REFERENCE.md` | REST API documentation |
| Configuration | `docs/CONFIGURATION.md` | Config options |

### Development

| Document | Path | Purpose |
|----------|------|---------|
| Contributing | `docs/CONTRIBUTING.md` | Contribution guidelines |
| Dev Standards | `docs/DEVELOPMENT_STANDARDS.md` | Coding standards |
| Troubleshooting | `docs/TROUBLESHOOTING.md` | Common issues |
| License Tracker | `LICENSES/LICENSE_TRACKER.md` | Third-party licenses |

---

## Source Code Structure

```
src/cogsyndelta/
├── core/                    # Neural architecture core
│   ├── integrated_system.py # Main system entry point
│   ├── pcn_vae_gan.py       # PCN-VAE-GAN hybrid
│   ├── vl_jepa.py           # Vision-Language JEPA
│   ├── mhc.py               # Memory-augmented Hopfield
│   ├── interconnect_manager.py # Cross-module routing
│   ├── progressive_loader.py   # Selective submodel loading
│   └── model_sectioning.py  # Model partitioning
├── memory/                  # Tiered memory system
│   ├── active_memory.py     # Hot data, immediate access
│   ├── short_term.py        # Recently accessed
│   └── long_term.py         # Compressed persistent storage
├── agents/                  # Self-improving agent framework
│   └── base_agent.py        # Agent base class
├── api/                     # FastAPI REST server
│   ├── server.py            # Main server
│   └── routes/              # API endpoints
├── quantum/                 # Quantum computing (STUBBED)
└── optimization/            # CUDA/Triton kernels
```

### Specialized Libraries (`libs/`)

```
libs/
├── vsa/                     # Vector Symbolic Architecture
│   └── src/vsa/
│       ├── core.py          # FHRR encoder
│       ├── memory/hopfield.py  # Modern Hopfield networks
│       └── memory/resonator.py # Resonator networks
├── compression/             # Compression pipeline
│   └── src/compression/
│       ├── balanced_ternary/ # {-1, 0, +1} quantization
│       ├── bitnet/          # BitNet-style ternary
│       ├── rvq/             # Residual Vector Quantization
│       └── mrl/             # Matryoshka Representation
└── algebraic-training/      # Algebraic training utilities
    └── src/algebraic_training/
```

---

## Subagent Workflow

When delegating to subagents (`runSubagent`), provide clear context:

### Research Tasks
```
Prompt structure for research:
1. Specific question or topic
2. Files/directories to search
3. Expected output format
4. What NOT to modify (research only)
```

### Code Implementation Tasks
```
Prompt structure for implementation:
1. Feature specification
2. Relevant files to read first
3. Files to modify
4. Test requirements
5. Reference: docs/DEVELOPMENT_STANDARDS.md for style
```

### Example Subagent Prompts

**Research prompt:**
```
Search the codebase for all usages of `IntelligentInterconnectManager`.
Look in: src/cogsyndelta/core/, tests/
Return: List of files, line numbers, and how each usage initializes the manager.
DO NOT modify any files - research only.
```

**Implementation prompt:**
```
Implement a new method `get_routing_stats()` in src/cogsyndelta/core/interconnect_manager.py

Requirements:
1. Read existing class structure first
2. Follow Google-style docstrings (see docs/DEVELOPMENT_STANDARDS.md)
3. Add type hints
4. Create test in tests/unit/test_interconnect_manager.py

Reference: Check similar methods like `get_memory_usage()` for patterns.
```

---

## Design Philosophy: Progressive Loading

CogSynDelta uses **pseudo-generalized highly specialized models** with selective loading:

- **Total submodels**: 10-20 specialized modules
- **Simultaneously active**: 4-8 submodels (~8GB GPU)
- **Target GPU**: RTX 5080 (16GB VRAM)

This approach favors composing smaller, specialized models dynamically rather than scaling to massive monolithic models.

Key file: `src/cogsyndelta/core/progressive_loader.py`

---

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

```python
from typing import TypeVar
from collections.abc import Sequence, Callable

T = TypeVar("T")

def process_items(items: Sequence[T], transform: Callable[[T], T]) -> list[T]:
    ...
```

### Import Order

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

---

## Commands Reference

### Testing
```bash
uv run pytest tests/ -v                    # All tests
uv run pytest tests/unit/ -v               # Unit tests only
uv run pytest tests/ -k "test_memory" -v   # Filter by name
```

### Quality Checks
```bash
uv run ruff check .                        # Lint all files
uv run ruff format .                       # Format all files
uv run mypy src/                           # Type checking
uv run python scripts/quality_control.py src/ --fail-under 90
```

### Running
```bash
uv run cogsyndelta-server                  # Start API server
uv run cogsyndelta-benchmark               # Run benchmarks
```

---

## Critical Notes

1. **Quantum features are STUBBED** - Real quantum backends blocked by Python 3.14 compatibility
2. **Memory "skip silently"** - Intentional graceful degradation in `_ensure_temporal_continuity()`
3. **No unsubstantiated claims** - Performance claims require benchmark validation in `benchmark_results/`
4. **Conventional commits** - Use `type(scope): description` format
5. **Dual attribution** - Always attribute to Tyler Zervas (tzervas) AND Average Joe's Labs (AJL)
6. **Device auto-detection** - Use `torch.cuda.is_available()` pattern, never default to "cuda"
7. **Progressive loading defaults** - 8 active submodels from pool of 10-20

---

## Pre-Submission Checklist

- [ ] Docstrings on all public functions (Google style)
- [ ] Type hints on all functions
- [ ] Tests for new functionality
- [ ] `uv run ruff check .` passes
- [ ] `uv run ruff format .` passes (no changes)
- [ ] `uv run mypy src/` passes
- [ ] No trailing whitespace
- [ ] Conventional commit message
- [ ] Performance claims backed by benchmarks in `benchmark_results/`
