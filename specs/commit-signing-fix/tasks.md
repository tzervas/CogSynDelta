# Tasks: Fix Unverified Commits

## Task Checklist

### Pre-Execution Verification

- [ ] **T0.1** Verify GPG key is available: `gpg --list-secret-keys 476C15DDFDEFF71932EA0E313470C2807DD80A2B`
- [ ] **T0.2** Verify git signing config: `git config --get commit.gpgsign` (should be `true`)
- [ ] **T0.3** Verify user has push access to all branches
- [ ] **T0.4** Ensure no other collaborators are actively working on these branches

---

### Phase 1: Backups

- [ ] **T1.1** Fetch all remote updates: `git fetch --all`
- [ ] **T1.2** Create backup of staging: `git branch backup/staging-20260118 origin/staging`
- [ ] **T1.3** Create backup of feature branch: `git branch backup/feat-20260118 origin/feat/specs-benchmarks-logging-infrastructure`
- [ ] **T1.4** Create backup of fix branch: `git branch backup/fix-ci-20260118 origin/fix/ci-precommit-checks`
- [ ] **T1.5** Verify all backups created: `git branch | grep backup`

---

### Phase 2: Fix Feature Branch

- [ ] **T2.1** Checkout feature branch: `git checkout feat/specs-benchmarks-logging-infrastructure`
- [ ] **T2.2** Pull latest: `git pull origin feat/specs-benchmarks-logging-infrastructure`
- [ ] **T2.3** Start interactive rebase from parent of first problematic commit:
  ```bash
  git rebase -i 9a769ec
  ```
- [ ] **T2.4** In editor, mark `af0c1c4` as `edit`, keep others as `pick`
- [ ] **T2.5** When rebase stops at af0c1c4, amend with correct author and signature:
  ```bash
  git commit --amend --author="Tyler Zervas <tz-dev@vectorweight.com>" -S
  git rebase --continue
  ```
- [ ] **T2.6** For commit 169f4b2, amend with signature when rebase reaches it:
  ```bash
  git commit --amend --no-edit -S
  git rebase --continue
  ```
- [ ] **T2.7** Complete rebase for all remaining commits
- [ ] **T2.8** Verify all commits now signed:
  ```bash
  git log --show-signature -5
  ```
- [ ] **T2.9** Force-push feature branch:
  ```bash
  git push --force-with-lease origin feat/specs-benchmarks-logging-infrastructure
  ```

---

### Phase 3: Rebuild Staging

- [ ] **T3.1** Checkout staging: `git checkout staging`
- [ ] **T3.2** Reset to main: `git reset --hard origin/main`
- [ ] **T3.3** Merge fixed feature branch: `git merge feat/specs-benchmarks-logging-infrastructure`
- [ ] **T3.4** Verify merge is clean (no conflicts)
- [ ] **T3.5** Force-push staging: `git push --force-with-lease origin staging`

---

### Phase 4: Fix Dependent Branches

- [ ] **T4.1** Checkout fix branch: `git checkout -B fix/ci-precommit-checks origin/fix/ci-precommit-checks`
- [ ] **T4.2** Rebase onto fixed feature branch: `git rebase feat/specs-benchmarks-logging-infrastructure`
- [ ] **T4.3** Resolve any conflicts if needed
- [ ] **T4.4** Force-push: `git push --force-with-lease origin fix/ci-precommit-checks`

---

### Phase 5: Verification

- [ ] **T5.1** Check local signatures on staging:
  ```bash
  git log --show-signature origin/staging -10
  ```
- [ ] **T5.2** Check for any unsigned commits:
  ```bash
  git log --format="%H %G?" origin/staging 2>/dev/null | grep -v "G"
  ```
- [ ] **T5.3** Navigate to GitHub PR #4 and verify all commits show "Verified"
- [ ] **T5.4** Wait for CI to complete on all branches
- [ ] **T5.5** Confirm CI passes

---

### Phase 6: Cleanup

- [ ] **T6.1** Delete backup branches after successful verification:
  ```bash
  git branch -D backup/staging-20260118
  git branch -D backup/feat-20260118
  git branch -D backup/fix-ci-20260118
  ```
- [ ] **T6.2** Sync local main: `git checkout main && git pull`
- [ ] **T6.3** Sync local develop: `git checkout develop && git pull`
- [ ] **T6.4** Sync local staging: `git checkout staging && git pull`

---

### Post-Execution

- [ ] **T7.1** Update status report with completion
- [ ] **T7.2** Close this spec as completed
- [ ] **T7.3** Continue with remaining project tasks

---

## Task Dependencies

```
T0.* (Prerequisites)
  │
  ├─> T1.* (Backups)
  │     │
  │     └─> T2.* (Fix Feature Branch)
  │           │
  │           ├─> T3.* (Rebuild Staging)
  │           │
  │           └─> T4.* (Fix Dependent Branches)
  │                 │
  │                 └─> T5.* (Verification)
  │                       │
  │                       └─> T6.* (Cleanup)
  │                             │
  │                             └─> T7.* (Post-Execution)
```

## Rollback Tasks (If Needed)

- [ ] **R1** Restore staging: `git checkout staging && git reset --hard backup/staging-20260118 && git push --force-with-lease origin staging`
- [ ] **R2** Restore feature: `git checkout feat/specs-benchmarks-logging-infrastructure && git reset --hard backup/feat-20260118 && git push --force-with-lease origin feat/specs-benchmarks-logging-infrastructure`
- [ ] **R3** Restore fix branch: `git checkout fix/ci-precommit-checks && git reset --hard backup/fix-ci-20260118 && git push --force-with-lease origin fix/ci-precommit-checks`

---

## Estimated Completion

| Phase | Est. Time | Actual |
|-------|-----------|--------|
| Prerequisites | 2 min | |
| Backups | 2 min | |
| Fix Feature | 10 min | |
| Rebuild Staging | 5 min | |
| Fix Dependents | 5 min | |
| Verification | 5 min | |
| Cleanup | 2 min | |
| **Total** | **31 min** | |

---

*Last Updated: January 18, 2026*
