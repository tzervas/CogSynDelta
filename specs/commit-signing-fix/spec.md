# Specification: Fix Unverified Commits in PR #4

## Problem Statement

Two commits in PR #4 (staging → main) are showing as "Unverified" on GitHub:

1. **`af0c1c4`** - "Initial plan"
   - **Author**: `copilot-swe-agent[bot]`
   - **Signed**: YES (with Tyler's GPG key `476C15DDFDEFF71932EA0E313470C2807DD80A2B`)
   - **GitHub Status**: May show "Unverified" due to author/signer mismatch
   - **Note**: This is a bot-authored commit that was later signed

2. **`169f4b2`** - "benchmarks(gpu): add RTX 5080 model baseline"
   - **Author**: Tyler Zervas
   - **Signed**: NO (commit is completely unsigned)
   - **GitHub Status**: Shows "Unverified"
   - **This is the critical commit to fix**

## Additional Context from Analysis

- **Total commits in feature branch**: 135 (from `9a769ec` to HEAD)
- **Commits with good signatures (G)**: 46
- **Commits unsigned (N)**: 43
- **Commits with verification errors (E)**: 46 (signed with key `B5690EEEBB952194` not available locally)

The `E` status commits are likely signed via GitHub web interface (GPG key from GitHub's web commit signing) and should show as "Verified" on GitHub even though they can't be verified locally.

## Branch Topology Analysis

```
main (39312ac)
    │
    └──> staging (02ad78f)
              │
              └──> feat/specs-benchmarks-logging-infrastructure (1302e2d)
                        │
                        └──> fix/ci-precommit-checks (a513e4a)

Problematic commits exist in:
- staging (remote + local)
- feat/specs-benchmarks-logging-infrastructure (remote + local)
- fix/ci-precommit-checks (remote only)

NOT in main (good - these are only in feature branches)
```

## Critical Constraints

1. **Cannot rewrite history on main** - main is protected
2. **PR #4 is staging → main** - these commits will enter main via this PR
3. **Multiple branches depend on these commits** - all feature branches descend from them
4. **Force-push will be required** - amending commits changes SHA hashes

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Divergent histories | HIGH | Coordinate force-push, notify collaborators |
| Lost commits | HIGH | Create backup branches before rewrite |
| Broken PRs | MEDIUM | May need to recreate PRs after rewrite |
| CI re-runs | LOW | Expected, acceptable |

## Solution Options

### Option A: Rebase and Re-sign All Commits (RECOMMENDED)

**Strategy**: Use `git rebase -i` with `exec` to re-sign commits starting from the first unsigned one.

**Pros**:
- Clean history with all commits properly signed
- Maintains correct author attribution
- Consistent verification status

**Cons**:
- Requires force-push to all affected branches
- Changes all commit SHAs downstream
- PRs may need updates

### Option B: Cherry-pick to New Branch

**Strategy**: Create new branches by cherry-picking and signing each commit.

**Pros**:
- Safer - doesn't modify existing branches
- Can validate before switching

**Cons**:
- Creates duplicate history
- More manual work
- Still requires closing/recreating PRs

### Option C: Squash Merge with Single Signed Commit

**Strategy**: Squash all staging changes into a single signed commit.

**Pros**:
- Simplest approach
- Single clean commit

**Cons**:
- Loses individual commit granularity
- Not preferred for feature development

## Recommended Implementation Plan

### Phase 1: Backup (Safety First)

```bash
# Create backup branches
git checkout staging
git branch staging-backup-$(date +%Y%m%d)

git checkout feat/specs-benchmarks-logging-infrastructure
git branch feat-backup-$(date +%Y%m%d)
```

### Phase 2: Identify the Rebase Point

The rebase needs to start from **before** `af0c1c4`:
- `9a769ec` - "Initial commit" (the parent of af0c1c4)
- This is the safe point to start the interactive rebase

### Phase 3: Rebase and Re-sign Feature Branch

```bash
git checkout feat/specs-benchmarks-logging-infrastructure
git pull origin feat/specs-benchmarks-logging-infrastructure

# Interactive rebase from before the problematic commits
git rebase -i 9a769ec --exec "git commit --amend --no-edit -S"
```

### Phase 4: Force-push Feature Branch

```bash
git push --force-with-lease origin feat/specs-benchmarks-logging-infrastructure
```

### Phase 5: Update Staging Branch

```bash
git checkout staging
git reset --hard origin/main
git merge feat/specs-benchmarks-logging-infrastructure
git push --force-with-lease origin staging
```

### Phase 6: Update Dependent Branches

For `fix/ci-precommit-checks`:
```bash
git checkout fix/ci-precommit-checks
git rebase feat/specs-benchmarks-logging-infrastructure
git push --force-with-lease origin fix/ci-precommit-checks
```

### Phase 7: Verify and Cleanup

1. Verify all commits show as "Verified" on GitHub
2. Check PR #4 is still valid
3. Delete backup branches after confirmation

## Success Criteria

- [ ] All commits in PR #4 show "Verified" badge on GitHub
- [ ] No divergent branch histories
- [ ] All PRs remain functional
- [ ] CI passes on all branches

## Alternative: GitHub Web-based Commit Amending

For the `af0c1c4` "Initial plan" commit specifically, since it was made by copilot-swe-agent[bot], GitHub may have special handling. However, this typically cannot be changed after the fact.

## Timing Recommendation

Execute this fix **before** merging PR #4 to main, otherwise:
- The unverified commits will enter main
- main history will have permanent unverified commits
- Cannot rewrite main history later

## Author

Tyler Zervas (@tzervas) - January 18, 2026
