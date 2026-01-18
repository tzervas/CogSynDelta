# Specification Templates

This directory contains templates for Spec-Driven Development following [GitHub Spec-Kit](https://github.com/github/spec-kit) standards.

## Templates

| Template | Purpose | When to Use |
|----------|---------|-------------|
| `spec-template.md` | Feature specifications | Before implementing any new feature |
| `plan-template.md` | Implementation plans | After spec approval, before coding |
| `tasks-template.md` | Task breakdown | To create atomic, trackable work items |
| `adr-template.md` | Architecture decisions | For significant technical decisions |

## Workflow

```
1. spec-template.md   →  Define WHAT we're building
         ↓
2. plan-template.md   →  Define HOW we'll build it
         ↓
3. tasks-template.md  →  Break into ACTIONABLE items
         ↓
4. Implementation     →  Write code against spec
         ↓
5. adr-template.md    →  Document significant decisions
```

## Usage

### For New Features

1. Copy `spec-template.md` to `specs/[feature-name]/spec.md`
2. Fill in all sections, especially acceptance criteria
3. Get spec reviewed and approved
4. Create implementation plan using `plan-template.md`
5. Break down into tasks using `tasks-template.md`

### For Architecture Decisions

1. Copy `adr-template.md` to `docs/adr/NNNN-title.md`
2. Use sequential numbering (0001, 0002, etc.)
3. Document the context, decision, and rationale
4. Link to related specs and issues

## Principles

- **No unsubstantiated claims**: All performance claims need benchmarks
- **What AND Why**: Every section explains both
- **Testable acceptance criteria**: If you can't test it, rewrite it
- **Living documents**: Update specs as understanding evolves

## See Also

- `AGENTS.md` - Agent development guidelines
- `memory/constitution.md` - Project principles
- `docs/DEVELOPMENT_STANDARDS.md` - Coding standards
