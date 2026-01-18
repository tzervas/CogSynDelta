# ADR-0005: Conventional Commits Standard

**Status**: Accepted
**Date**: 2026-01-18
**Decision Makers**: @tzervas
**Technical Story**: Standardize commit messages for automation

## Context

As the project grows, we need:
1. **Readable history**: Easy to understand what changed and why
2. **Automated changelogs**: Generate CHANGELOG.md from commits
3. **Semantic versioning**: Determine version bumps from commit types
4. **Better reviews**: Consistent format helps reviewers

The team was using inconsistent commit message formats, making history hard to parse.

## Decision

We will adopt the **[Conventional Commits](https://www.conventionalcommits.org/) specification** for all commit messages.

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description | Version Bump |
|------|-------------|--------------|
| `feat` | New feature | MINOR |
| `fix` | Bug fix | PATCH |
| `docs` | Documentation only | None |
| `style` | Formatting, no code change | None |
| `refactor` | Code change, no feature/fix | None |
| `perf` | Performance improvement | PATCH |
| `test` | Adding/updating tests | None |
| `build` | Build system changes | None |
| `ci` | CI configuration changes | None |
| `chore` | Maintenance tasks | None |

### Breaking Changes

```
feat(api)!: change response format

BREAKING CHANGE: API responses now use camelCase instead of snake_case.
```

The `!` or `BREAKING CHANGE:` footer triggers a MAJOR version bump.

### Scopes

Project-specific scopes:
- `core` - Core neural architecture
- `memory` - Memory systems
- `agents` - Self-improving agents
- `api` - REST API
- `quantum` - Quantum module
- `benchmarks` - Performance testing
- `tests` - Test suite
- `ci` - CI/CD pipelines
- `docs` - Documentation

## Rationale

### Why Conventional Commits

1. **Industry standard**: Widely adopted, well-documented
2. **Tooling support**: Works with semantic-release, commitizen, changelog generators
3. **Clear intent**: Type immediately tells you what kind of change
4. **Automation ready**: Machine-parseable format

### Alternatives Considered

#### Option 1: Free-form messages with guidelines

- **Pros**: Flexible, no learning curve
- **Cons**: Inconsistent, can't automate
- **Why Rejected**: Doesn't enable automation

#### Option 2: Gitmoji

- **Pros**: Visual, fun
- **Cons**: Harder to parse, emoji rendering issues
- **Why Rejected**: Less tooling support, accessibility concerns

#### Option 3: Angular commit conventions

- **Pros**: Mature, well-documented
- **Cons**: Conventional Commits is the evolution of Angular's format
- **Why Rejected**: Conventional Commits is newer standard

## Consequences

### Positive

- Automated CHANGELOG generation
- Semantic version bumps from commits
- Consistent, readable history
- Better PR descriptions

### Negative

- Learning curve for contributors
- Mitigation: Pre-commit hook validates format, `.gitmessage` template

### Neutral

- Requires discipline to write good descriptions
- Scope is optional but recommended

## Implementation

### Pre-commit Hook

```yaml
# In .pre-commit-config.yaml
- repo: https://github.com/compilerla/conventional-pre-commit
  rev: v3.6.0
  hooks:
    - id: conventional-pre-commit
      stages: [commit-msg]
```

### Commit Template

`.gitmessage` provides a template with instructions.

Configure with:
```bash
git config commit.template .gitmessage
```

### CI Validation

PR titles and commit messages validated in CI.

## References

- [Conventional Commits Spec](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- `.gitmessage` - Commit template
- `CHANGELOG.md` - Generated from commits
