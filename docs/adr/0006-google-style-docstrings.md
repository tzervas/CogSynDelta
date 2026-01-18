# ADR-0006: Google Style Docstrings

**Status**: Accepted
**Date**: 2026-01-18
**Decision Makers**: @tzervas
**Technical Story**: Standardize documentation for code and auto-generation

## Context

CogSynDelta requires comprehensive documentation for:
1. **API reference generation**: Auto-generate docs from code
2. **IDE support**: Better autocomplete and hover information
3. **Onboarding**: New developers need to understand the codebase
4. **AI agents**: Coding agents need clear context about function behavior

The project needs a consistent docstring format that:
- Is readable in source code
- Renders well in documentation tools
- Integrates with type hints
- Supports the "what and why" documentation philosophy

## Decision

We will use **[Google Style Python Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)** for all public functions, classes, and modules.

### Required Sections

```python
def function_name(param1: str, param2: int) -> bool:
    """Brief one-line description of what the function does.

    Longer description explaining the WHY - rationale for this approach,
    edge cases handled, and any non-obvious behavior. This section should
    help readers understand not just what the code does, but why it exists
    and why it was implemented this way.

    Args:
        param1: Description of param1, including valid values/constraints.
        param2: Description of param2.

    Returns:
        Description of return value. For complex returns, describe structure.

    Raises:
        ValueError: When param1 is empty or invalid.
        RuntimeError: When the operation fails due to [specific condition].

    Example:
        >>> function_name("test", 42)
        True

    Note:
        Any important caveats or implementation details.

    See Also:
        related_function: For similar functionality.
    """
```

### Minimum Required

All public functions MUST have:
- Brief description (first line)
- Args section (if parameters exist)
- Returns section (if not None)
- Raises section (if exceptions raised)

### Recommended

- Example section for complex functions
- "Why" explanation in description
- See Also for related functions

## Rationale

### Why Google Style

1. **Readable**: Clean, scannable format
2. **Tooling**: Supported by Sphinx, pdoc, mkdocs, IDEs
3. **Type hints**: Complements Python type hints well
4. **Popular**: Widely used, familiar to Python developers

### Alternatives Considered

#### Option 1: NumPy Style

- **Pros**: Very detailed, good for scientific code
- **Cons**: More verbose, more sections to maintain
- **Why Rejected**: Overkill for most functions

#### Option 2: Sphinx/reST Style

- **Pros**: Native to Sphinx
- **Cons**: Less readable in source, `:param:` syntax verbose
- **Why Rejected**: Harder to read in code

#### Option 3: Epytext

- **Pros**: Compact
- **Cons**: Outdated, less tooling support
- **Why Rejected**: Legacy format

## Consequences

### Positive

- Consistent documentation across codebase
- Auto-generated API reference
- Better IDE experience
- Clear context for AI coding agents

### Negative

- More typing for developers
- Mitigation: Snippet templates, enforcement in CI makes it habitual
- Mitigation: Quality control script checks coverage

### Neutral

- Requires ongoing maintenance as code changes
- Pre-commit hooks help catch missing docs

## Implementation

### Enforcement

1. **Pre-commit**: pydocstyle checks Google style compliance
2. **CI**: Quality control script measures docstring coverage
3. **Code Review**: Reviewers verify docstrings exist and explain "why"

### Pre-commit Configuration

```yaml
- repo: https://github.com/pycqa/pydocstyle
  rev: 6.3.0
  hooks:
    - id: pydocstyle
      args: [--convention=google]
      files: ^src/
```

### pyproject.toml Configuration

```toml
[tool.pydocstyle]
convention = "google"
add-ignore = ["D100", "D104"]  # Allow missing module/package docstrings
```

### Quality Gate

The quality control script (`scripts/quality_control.py`) measures:
- Percentage of functions with docstrings
- Docstring completeness (Args, Returns, Raises)
- Target: 90%+ coverage

## References

- [Google Python Style Guide - Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [PEP 257 - Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)
- `scripts/quality_control.py` - Docstring coverage checking
- `AGENTS.md` - Agent documentation requirements
