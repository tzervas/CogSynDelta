# Branch Status Tracker

**Last Updated:** January 20, 2026

This document tracks the status of all active branches, their merge status, and pending work.

---

## ⚠️ BRANCH POLICY - CRITICAL

**NO DIRECT MERGES TO `main` BRANCH**

All development work flows through the following hierarchy:

```
feature/* / fix/* → develop → staging → main
         ↑                               ↑
    (work here)                    (release only)
```

**Rules:**
1. **Never** merge directly into `main` - releases only via CI/CD pipeline
2. **Never** merge directly into `staging` - automated after `develop` validation
3. **Always** create feature/fix branches from `develop`
4. **Always** merge PRs into `develop` first
5. Feature branches may merge between each other if dependencies exist

---

## Branch Hierarchy

```
main (production) ← PROTECTED - NO DIRECT MERGES
  └── staging (pre-production) ← PROTECTED - AUTOMATED ONLY
        └── develop (integration) ← ALL PRs TARGET HERE
              ├── feat/algebraic-training
              ├── feat/ternary-implementations
              ├── feat/specs-benchmarks-logging-infrastructure
              └── refactor/libs-submodule-structure
```

---

## Active Branches

### Protected Branches

| Branch | Last Commit | Status | Notes |
|--------|-------------|--------|-------|
| `main` | `c5f3403` release: CogSynDelta infrastructure improvements | ✅ Stable | Production |
| `staging` | `8749e6d` Merge branch 'main' into staging | ✅ Synced | Pre-production |
| `develop` | `b22c1c2` chore(hooks): add pre-push hook | ✅ Updated | Integration |

### Feature Branches

| Branch | Base | Status | PR | Merge Target | Priority |
|--------|------|--------|-----|--------------|----------|
| `refactor/libs-submodule-structure` | develop | 🟡 In Review | [PR #11](https://github.com/tzervas/CogSynDelta/pull/11) | develop | **High** |
| `feat/algebraic-training` | develop | 🟡 Ready for Review | Pending | develop | **High** |
| `feat/ternary-implementations` | develop | 🔵 In Progress | N/A | develop | Medium |
| `feat/specs-benchmarks-logging-infrastructure` | develop | 🔵 In Progress | N/A | develop | Medium |

### Status Legend

- ✅ **Stable/Synced** - Production ready, no pending merges
- 🟢 **Merged** - Successfully merged, branch can be deleted
- 🟡 **Ready for Review** - PR ready or needs PR creation
- 🔵 **In Progress** - Active development
- 🔴 **Blocked** - Waiting on dependency or issue

---

## Pending Merges

### Phase 1: Current Sprint

1. **`refactor/libs-submodule-structure` → `develop`**
   - Content: Extract algebraic-training to libs/ submodule
   - ADR: 0011-libs-submodule-structure.md
   - Tests: 22 passing
   - Lint: ✅ Clean
   - Status: **Ready for PR**

2. **`feat/algebraic-training` → `develop`**
   - Content: Algebraic training ADR-0009, research docs
   - Tests: Passing
   - Status: **Ready for merge** (after libs extraction merges)

### Phase 2: Next Sprint

3. **`feat/ternary-implementations` → `develop`**
   - Content: BitNet exploration ADR-0010
   - Status: In progress

4. **`feat/specs-benchmarks-logging-infrastructure` → `develop`**
   - Content: Memory compactors, benchmarks
   - Status: In progress

### Phase 3: Release

5. **`develop` → `staging`** (automated after Phase 1 & 2 complete, all tests pass)
6. **`staging` → `main`** (automated release pipeline only - NO manual merge)

> **Note:** Merges to `staging` and `main` are automated via CI/CD.
> Do not manually merge into these branches.

---

## Branch Details

### refactor/libs-submodule-structure

**Purpose:** Extract algebraic training into standalone library for potential external release

**Key Changes:**
- Created `libs/algebraic-training/` standalone package
- Split `algebraic_training.py` into focused modules:
  - `ntk.py` - NTKPredictor
  - `fisher.py` - FisherInformationPredictor
  - `spectral.py` - SpectralWeightPredictor
  - `optimizer.py` - AlgebraicOptimizer
  - `trainer.py` - UnifiedAlgebraicTrainer
  - `weight_distribution.py` - WeightDistributionPredictor
  - `functional.py` - Functional API
- Created `_integration.py` for CogSynDelta-specific wrappers
- 22 tests including accuracy parity tests

**Files Changed:** 17 files, +3300 lines

**Commit:** `ea321b9` - GPG signed

---

### feat/algebraic-training

**Purpose:** Implement algebraic training methods (ADR-0009)

**Key Changes:**
- ADR-0009: Algebraic Training Architecture
- Research documentation
- Initial implementation (superseded by libs refactor)

**Status:** Research/docs complete, implementation moved to libs

---

### feat/ternary-implementations

**Purpose:** Explore BitNet and ternary weight implementations (ADR-0010)

**Key Changes:**
- ADR-0010: BitNet Exploration
- Specs for ternary weight systems
- Initial research

**Status:** Early exploration phase

---

### feat/specs-benchmarks-logging-infrastructure

**Purpose:** Improve memory compression and benchmarking

**Key Changes:**
- HighFidelityCompactor (≥0.95 cosine similarity)
- HybridAdaptiveCompactor (multi-mode compression)
- ResidualBoostCompactor
- LosslessCompactor (OOM protection for <24GB GPUs)
- Benchmark improvements:
  - Fixed PCN_VAE_GAN, InterconnectManager, mHCGate import errors
  - Added OOM protection for LosslessCompactor
  - Fixed CompactorAdapter for DenseEmbeddingEncoder encode/decode
  - **NEW**: Untrained model detection (fidelity < 0.3 flagged)
  - **NEW**: Detailed statistics report (`--detailed` flag)
  - **NEW**: Benchmark visualization with sparklines and trends
- Context-efficient memory techniques:
  - ChunkedCompactor (memory-efficient chunked processing)
  - ImportanceContextPruner (ToMe-inspired token pruning)
  - LatentCache (LRU caching with invalidation)
- ADRs:
  - ADR-0015: Context-Efficient Memory Techniques
  - ADR-0016: Model Training Roadmap

**Status:** Active development - benchmark infrastructure complete

---

## Merge Checklist

Before merging any branch:

- [ ] All CI checks passing
- [ ] GPG signed commits
- [ ] Tests pass (`uv run pytest`)
- [ ] Lint clean (`uv run ruff check`)
- [ ] Type check clean (`uv run mypy src/`)
- [ ] Documentation updated
- [ ] CHANGELOG updated (for releases)
- [ ] ADR created (for architectural changes)

---

### Cleanup Tasks

### Branches Deleted ✅

| Branch | Status | Notes |
|--------|--------|-------|
| `backup/staging-20260118` | ✅ Deleted | Local backup removed |
| `backup/feat-20260118` | ✅ Deleted | Local backup removed |
| `backup/fix-ci-20260118` | ✅ Deleted | Local backup removed |
| `fix/ci-precommit-checks` | ✅ Pruned | Remote ref cleaned |
| `copilot/create-pcn-vae-gan-hybrid` | ✅ Pruned | Remote ref cleaned |

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-19 | 1.1 | PR #11 created, branches cleaned, hook added |
| 2026-01-19 | 1.0 | Initial tracker creation |

---

*Maintained by: Tyler Zervas (@tzervas)*
