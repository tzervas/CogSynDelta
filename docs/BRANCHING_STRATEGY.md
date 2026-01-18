# Branching Strategy

CogSynDelta follows a modified GitFlow branching model optimized for iterative development with quality gates.

## Branch Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                          main                                    │
│  Production-ready code only. Releases tagged here.              │
│  Protection: Require PR, 1 approval, signed commits, CI pass    │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ merge --no-ff (release)
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        staging                                   │
│  Integration testing. Pre-release validation.                   │
│  Protection: Require PR, CI pass                                │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ merge --no-ff (integration)
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        develop                                   │
│  Active development. Features integrate here first.             │
│  Protection: Require PR, CI pass                                │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ merge (squash or regular)
                              │
┌─────────────────────────────────────────────────────────────────┐
│              feat/* | fix/* | docs/* | refactor/*               │
│  Working branches. Short-lived. One purpose per branch.         │
│  No protection.                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Branch Types

### Protected Branches

| Branch | Purpose | Merges From | Merges To | Protection |
|--------|---------|-------------|-----------|------------|
| `main` | Production releases | `staging` | - | Strictest |
| `staging` | Pre-release testing | `develop` | `main` | Strict |
| `develop` | Development integration | `feat/*`, `fix/*` | `staging` | Standard |

### Working Branches

| Prefix | Purpose | Merges To | Example |
|--------|---------|-----------|---------|
| `feat/` | New features | `develop` | `feat/add-memory-compression` |
| `fix/` | Bug fixes | `develop` | `fix/api-timeout-handling` |
| `docs/` | Documentation | `develop` | `docs/update-api-reference` |
| `refactor/` | Code improvements | `develop` | `refactor/simplify-embeddings` |
| `test/` | Test improvements | `develop` | `test/add-integration-tests` |
| `chore/` | Maintenance | `develop` | `chore/update-dependencies` |
| `hotfix/` | Urgent production fixes | `main` + `develop` | `hotfix/critical-security-fix` |

## Workflow

### Standard Feature Development

```bash
# 1. Start from develop
git checkout develop
git pull origin develop

# 2. Create feature branch
git checkout -b feat/my-new-feature

# 3. Make changes with conventional commits
git add .
git commit -m "feat(core): implement new feature"

# 4. Push and create PR to develop
git push -u origin feat/my-new-feature
# Create PR via GitHub

# 5. After PR merge, delete branch
git checkout develop
git pull
git branch -d feat/my-new-feature
```

### Release Process

```bash
# 1. Merge develop to staging for testing
# (via PR on GitHub)

# 2. Test in staging environment

# 3. If tests pass, merge staging to main
# (via PR on GitHub, requires approval)

# 4. Tag the release (automated via semantic-release)
# v0.3.0 created automatically
```

### Hotfix Process

```bash
# 1. Create hotfix from main
git checkout main
git checkout -b hotfix/critical-fix

# 2. Make fix
git commit -m "fix(core): resolve critical issue"

# 3. PR to main (emergency review)
# 4. After merge, also merge to develop
git checkout develop
git merge main
```

## Protection Rules

### main

- ✅ Require pull request before merging
- ✅ Require 1 approval
- ✅ Require signed commits
- ✅ Require status checks to pass (CI, tests)
- ✅ Require linear history
- ✅ Restrict who can push (code owners only)
- ✅ Do not allow bypassing settings

### staging

- ✅ Require pull request before merging
- ✅ Require status checks to pass (CI, tests)
- ✅ Require branches to be up to date

### develop

- ✅ Require pull request before merging
- ✅ Require status checks to pass (CI, tests)
- ⬜ Allow squash merging

## Naming Conventions

### Branch Names

```
<type>/<short-description>
```

- Use lowercase
- Use hyphens (not underscores)
- Keep short but descriptive
- Include issue number if applicable: `feat/42-add-caching`

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

See `.gitmessage` template for details.

## Version Tagging

Versions follow [Semantic Versioning](https://semver.org/):

- **Tags on main only**: `v0.2.0`, `v0.3.0`, `v1.0.0`
- **Pre-release tags**: `v0.3.0-rc.1` (release candidates on staging)
- **Dev versions**: Not tagged, tracked via commit SHA

## Quick Reference

```bash
# Configure commit template
git config commit.template .gitmessage

# Configure GPG signing (recommended)
git config commit.gpgsign true

# View current branch
git branch --show-current

# List recent branches
git branch --sort=-committerdate | head -10

# Clean up merged branches
git branch --merged develop | grep -v "main\|staging\|develop" | xargs git branch -d
```

## See Also

- `AGENTS.md` - Agent development guidelines
- `memory/constitution.md` - Project principles
- `docs/COMMIT_SIGNING.md` - GPG signing setup
