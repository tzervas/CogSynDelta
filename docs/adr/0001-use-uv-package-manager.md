# ADR-0001: Use UV Package Manager

**Status**: Accepted
**Date**: 2026-01-18
**Decision Makers**: @tzervas
**Technical Story**: Migration from pip/setuptools to UV

## Context

The project initially used pip with setuptools for dependency management. As the project grew and adopted Python 3.14, several challenges emerged:

1. **Slow dependency resolution**: pip's resolver became increasingly slow with complex dependency trees
2. **Reproducibility issues**: pip-tools and pip-compile didn't guarantee identical installs across environments
3. **Python 3.14 support**: Needed a tool that could handle cutting-edge Python versions
4. **Developer experience**: Wanted faster, more reliable installs for CI and local development

The Python packaging ecosystem has evolved significantly, with new tools like UV (from Astral, the ruff creators) offering substantial improvements.

## Decision

We will use **UV** as the primary package manager for CogSynDelta.

Specifically:
- `uv sync` replaces `pip install -r requirements.txt`
- `uv.lock` provides deterministic dependency resolution
- `uv run` replaces virtualenv activation for running commands
- `uv build` replaces `python -m build`

## Rationale

### Why UV

1. **10-100x faster** than pip for dependency resolution and installation
2. **Built-in lockfile** (`uv.lock`) for reproducible builds
3. **Python version management** built-in (`uv python install 3.14`)
4. **Drop-in compatible** with existing pyproject.toml
5. **Maintained by Astral** (same team as ruff, which we already use)

### Alternatives Considered

#### Option 1: Continue with pip + pip-tools

- **Pros**: Familiar, widely documented
- **Cons**: Slow, Python 3.14 issues, no integrated lockfile
- **Why Rejected**: Performance and reproducibility issues

#### Option 2: Poetry

- **Pros**: Good lockfile, nice CLI
- **Cons**: Slower than UV, different pyproject.toml format, lagging Python 3.14 support
- **Why Rejected**: Performance and compatibility concerns

#### Option 3: PDM

- **Pros**: PEP 582 support, good lockfile
- **Cons**: Smaller community, less tooling integration
- **Why Rejected**: Ecosystem concerns

## Consequences

### Positive

- CI runs are 3-5x faster due to improved install times
- Reproducible builds via `uv.lock`
- Simpler commands (`uv run pytest` vs `source .venv/bin/activate && pytest`)
- Multi-index support for PyTorch CUDA wheels works seamlessly

### Negative

- Developers need to install UV (one-time setup)
- Some documentation/tutorials assume pip
- Mitigation: Clear INSTALL.md instructions and contributing guide

### Neutral

- pyproject.toml remains the source of truth (compatible with pip fallback)
- Both `requirements.txt` and `uv.lock` maintained for compatibility

## Implementation

Commit `a7f0aea`: "feat: migrate to UV package manager with Python 3.14 and fix all tests"

Key changes:
- Added `[tool.uv]` section to pyproject.toml
- Created `uv.lock` for dependency pinning
- Updated CI workflows to use `uv sync` and `uv run`
- Updated INSTALL.md with UV instructions
- Maintained `requirements.txt` for pip fallback compatibility

## References

- [UV Documentation](https://docs.astral.sh/uv/)
- [Astral Blog: UV Announcement](https://astral.sh/blog/uv)
- Commit: a7f0aea31b1f35afc043f1f7c8c8c579e35522f7
