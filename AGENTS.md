# AGENTS.md

## About CogSynDelta and AI Agent Development

**CogSynDelta** is a PCN-VAE-GAN hybrid self-improving AI system. This document provides guidance for AI coding agents working on this codebase, following [GitHub Spec-Kit](https://github.com/github/spec-kit) standards for Spec-Driven Development.

---

## General Practices

### Code Standards

- **Python Version**: 3.14+ (use modern syntax, type hints everywhere)
- **Docstrings**: Google-style docstrings are **mandatory** for all public functions, classes, and modules
  - Must include: Brief description, Args, Returns, Raises, and a "Why" section for non-obvious implementations
- **Type Hints**: All functions must have complete type annotations (enforced by mypy strict mode)
- **Line Length**: 100 characters maximum
- **Imports**: Sorted by ruff, no unused imports permitted

### Documentation Requirements

Every piece of code must answer:
1. **What**: What does this do? (docstring first line)
2. **Why**: Why does it exist? Why this approach? (docstring body or inline comments)
3. **How**: How does it work? (code + docstrings for complex logic)

### Commit Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`

**Scopes**: `core`, `memory`, `agents`, `api`, `quantum`, `benchmarks`, `tests`, `ci`

### Branch Naming

- Features: `feat/description-of-feature`
- Fixes: `fix/description-of-fix`
- Docs: `docs/what-documented`
- Refactor: `refactor/what-refactored`

---

## Supported AI Agents

| Agent | Directory | Format | Status |
|-------|-----------|--------|--------|
| **GitHub Copilot** | `.github/agents/` | Markdown | ✅ Full Support |
| **Claude Code** | `.claude/commands/` | Markdown | 🔄 Planned |
| **Cursor** | `.cursor/rules/` | Markdown | 🔄 Planned |

---

## Project Architecture

### Core Components

```
src/cogsyndelta/
├── core/           # Core neural architecture (PCN-VAE-GAN, VL-JEPA, mHC)
├── memory/         # Tiered memory system (active → short → long-term)
├── agents/         # Self-improving agent framework
├── api/            # FastAPI REST server
├── quantum/        # Quantum computing integration (stubbed for Python 3.14)
└── optimization/   # CUDA/Triton kernel optimizations
```

### Key Design Decisions

All major architectural decisions are documented in `docs/adr/`. Before making significant changes:

1. Check existing ADRs for context
2. Propose new ADRs for significant architectural changes
3. Reference ADRs in commit messages and PRs

### Memory Architecture

The memory system uses a tiered approach for efficiency:

- **Active Memory**: Hot data, immediate access, bounded capacity
- **Short-term Memory**: Recently accessed, fast retrieval
- **Long-term Memory**: Compressed storage, slower but persistent

**Important**: The "memory not found - skip silently" behavior in `_ensure_temporal_continuity()` is **intentional graceful degradation**. Memory IDs may reference archived/compacted data.

---

## Development Workflow

### Before Making Changes

1. **Read the Constitution** (`memory/constitution.md`)
2. **Check ADRs** (`docs/adr/`) for relevant decisions
3. **Review ROADMAP.md** for planned vs backlogged features
4. **Run Tests**: `uv run pytest tests/ -v`

### Making Changes

1. Create feature branch from `develop`
2. Write tests first (TDD strongly encouraged)
3. Implement with proper docstrings
4. Run quality checks: `uv run ruff check . && uv run mypy src/`
5. Commit with conventional commit messages

### Quality Gates

All PRs must pass:
- [ ] All tests pass (`pytest`)
- [ ] Type checking (`mypy --strict`)
- [ ] Linting (`ruff check`)
- [ ] Code quality score ≥ 90 (`scripts/quality_control.py`)
- [ ] No unsubstantiated claims in documentation

---

## Spec-Driven Development

This project follows Spec-Driven Development (SDD). For new features:

1. **Spec First**: Create specification in `specs/[feature]/spec.md`
2. **Plan**: Generate implementation plan in `specs/[feature]/plan.md`
3. **Tasks**: Break into atomic tasks in `specs/[feature]/tasks.md`
4. **Implement**: Code against the spec
5. **Validate**: Verify against acceptance criteria

### Templates

- `templates/spec-template.md` - Feature specification
- `templates/plan-template.md` - Implementation plan
- `templates/tasks-template.md` - Task breakdown
- `templates/adr-template.md` - Architecture Decision Record

---

## Testing Philosophy

### Test Categories

1. **Unit Tests** (`tests/test_unit.py`): Isolated component testing
2. **Integration Tests** (`tests/test_comprehensive.py`): Cross-component interaction
3. **API Tests** (`tests/test_api.py`): REST endpoint validation
4. **Benchmark Tests** (`benchmarks/`): Performance validation

### Test Requirements

- All new features require tests
- Bug fixes require regression tests
- Performance claims require benchmark validation
- **No unsubstantiated claims**: If you claim "10x faster", provide benchmarks

---

## License Compliance

Before adding dependencies:

1. Check `LICENSES/LICENSE_TRACKER.md` for approved licenses
2. MIT, Apache-2.0, BSD-2/3-Clause, PSF-2.0 are pre-approved
3. GPL/LGPL require explicit approval
4. Update tracker when adding new dependencies

---

## Contact & Governance

- **Code Owner**: @tzervas
- **Organization**: Average Joe's Labs (AJL)
- **Constitution**: `memory/constitution.md`
- **Standards**: `docs/DEVELOPMENT_STANDARDS.md`

*Last Updated: January 18, 2026*
