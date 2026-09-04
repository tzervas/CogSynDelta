# Phase 0 Forgejo keep/drop audit — 2026-08-30

> Official git copy of the 2026-08-30 Codex Phase 0 audit. Vault twin:
> `akula-csd-kb/Architecture/Phase-0-Forgejo-Keep-Drop-2026-08-30.md`.
> `STATUS.md` still wins for measured claims. GitHub CLOSE rows are
> recommendations only — never bot-close GitHub.com PRs.

## Decision

The Python PoC line is the program line. Preserve narrow, measured work and freeze or close the
two sedimentary programs. No branch was merged, pushed, deleted, or renamed during this audit.

- **KEEP**: active program/reference line; continue or rebase it.
- **CHERRY-PICK**: take only the named narrow value onto a clean branch from current `main`.
- **ARCHIVE**: freeze as provenance; do not merge into the program line.
- **CLOSE**: reject as noise or an obsolete proposal. GitHub entries are recommendations only.

Source snapshot: authenticated Forgejo REST plus fully fetched `refs/heads/*` on 2026-08-30.
Forgejo contained **126 heads** and **zero open PR objects** across the three repositories. GitHub
is an operator mirror and contained 61 open PRs; the PR tables below are read-only recommendations.

## `tzervas/CogSynDelta`

Baseline: `forgejo/main` at `f835cf0`. Heads: **59**. Adjudicated counts: **KEEP 3**, **ARCHIVE
56**. The former `develop` program is not to be merged into `main`.

| Forgejo head | Verdict | One-line reason |
|---|---|---|
| `archive/20260715-develop` | ARCHIVE | Dated snapshot of the separate Program B baseline. |
| `archive/20260715-main` | ARCHIVE | Dated pre-PoC snapshot; current `main` contains it. |
| `ci/akula-gpu-runner` | KEEP | Two focused PoC CUDA/runner commits; rebase onto current `main` and require honest CI. |
| `ci/self-hosted-only` | ARCHIVE | 36 commits behind; its incomplete runner selector should be recreated if needed. |
| `claude/balanced-ternary-5wbyn` | ARCHIVE | Pre-PoC balanced-ternary research, unsupported by measured status. |
| `claude/balanced-ternary-clean-5wbyn` | ARCHIVE | Cleaned ternary fossil on the same superseded lineage. |
| `claude/compression-pipeline-5wbyn` | ARCHIVE | Legacy compression experiment without current test evidence. |
| `claude/progressive-loading-5wbyn` | ARCHIVE | Legacy progressive-loading program, divergent from PoC architecture. |
| `claude/vsa-library-5wbyn` | ARCHIVE | VSA research belongs to Program B sediment. |
| `copilot/create-pcn-vae-gan-hybrid` | ARCHIVE | Independent README-era hybrid history with no merge base to current `main`. |
| `dev` | ARCHIVE | Fully contained stale fleet/badge-era branch. |
| `develop` | ARCHIVE | VSA/algebraic Program B line, 43 behind/14 ahead; explicitly non-program. |
| `feat/agent-harness` | KEEP | Five focused commits for operator harness, guards, and docs; no AgentFleet runtime. |
| `feat/automated-maintenance-review-11775001417740100142` | ARCHIVE | Automated Program B maintenance descendant. |
| `feat/compression-benchmark-infra` | ARCHIVE | Pre-PoC benchmark-report infrastructure, not a current experiment. |
| `feat/maintenance-quality-review-1105009105384052571` | ARCHIVE | Divergent Program B CI/maintenance branch. |
| `feat/maintenance-review-4878658191108315390` | ARCHIVE | Divergent Program B workflow-maintenance branch. |
| `feat/maintenance-review-5265908703401954886` | ARCHIVE | Divergent blending/quality branch. |
| `feat/maintenance-review-6673365789579649169` | ARCHIVE | Algebraic-training maintenance, outside measured PoC. |
| `feat/poc-cuda-measure` | ARCHIVE | Fully merged; CUDA measurement work is already on `main`. |
| `feat/poc-e0-e2-foundation` | ARCHIVE | Fully merged PoC-1 substrate. |
| `feat/poc-region-registry` | ARCHIVE | Fully merged PoC-2 registry/router. |
| `feat/poc-train-route` | ARCHIVE | Fully merged PoC-3 trained routing. |
| `feat/silent-error-logging` | ARCHIVE | One-commit January logging experiment on pre-PoC history. |
| `feat/specs-benchmarks-logging-infrastructure` | ARCHIVE | Obsolete independent legacy history with no merge base. |
| `fix/ci-mypy-type-errors` | ARCHIVE | Broad legacy assessment branch, not a current PoC fix. |
| `fix/maintenance-review-improvements-15552731785882124168` | ARCHIVE | Program B optimization/quality churn. |
| `jules-11663694647749852728-e9cd8d28` | ARCHIVE | Jules chunk-blending/maintenance sediment. |
| `jules-1247812531793930591-2172b6e5` | ARCHIVE | Jules quality/CI sediment. |
| `jules-1247845497477431656-b6237bb8` | ARCHIVE | Jules quality-review sediment. |
| `jules-13703441076878881046-95e2cb0f` | ARCHIVE | Jules progressive-loader/algebraic sediment. |
| `jules-15592453777948541705-4ce36ee5` | ARCHIVE | Jules blending/quality sediment. |
| `jules-1668043902756157847-f0c6832a` | ARCHIVE | Jules maintenance/security churn. |
| `jules-18422569073778592940-d0d75b23` | ARCHIVE | Jules algebraic-complexity churn. |
| `jules-4074474147775153444-21fc1b20` | ARCHIVE | Jules chunk-blending/quality churn. |
| `jules-4285862908706332243-e4d1adbd` | ARCHIVE | Jules maintenance/runner-disk churn. |
| `jules-5732845235305945309-f0a7af81` | ARCHIVE | Jules optimization-maintenance churn. |
| `jules-5988585040434365535-4afdb6cf` | ARCHIVE | Jules progressive-loader/algebraic churn. |
| `jules-6202098072530946860-4e8abd18` | ARCHIVE | Jules core/optimization churn. |
| `jules-7298396753358203747-31d965ab` | ARCHIVE | Jules optimization-quality churn. |
| `jules-9014215542651342316-3943b6a9` | ARCHIVE | Jules automated-quality churn. |
| `jules-9765958131619533452-98e0c760` | ARCHIVE | Jules fleet-workflow sediment. |
| `main` | KEEP | Authoritative `STATUS.md`, PoC tests, and `src/cogsyndelta/poc/` line. |
| `maintenance/comprehensive-review-and-quality-improvements-7404159461572907830` | ARCHIVE | Program B broad maintenance descendant. |
| `maintenance/quality-refactoring-and-blending-8432855283715579363` | ARCHIVE | Program B blending/quality descendant. |
| `refactor/automated-maintenance-review-4463605815494777710` | ARCHIVE | Program B runner-disk/maintenance churn. |
| `refactor/comprehensive-maintenance-review-4910817366475062095` | ARCHIVE | Program B broad core-quality refactor. |
| `refactor/enhance-code-quality-scores-16907359726872318662` | ARCHIVE | Unmeasured “100/100” score churn. |
| `refactor/maintenance-quality-review-1492356723521003984` | ARCHIVE | Program B static-quality refactor. |
| `refactor/maintenance-quality-review-4428541114331801030` | ARCHIVE | Program B CI disk/caching maintenance. |
| `refactor/maintenance-review-6912658892127868145` | ARCHIVE | Program B broad maintenance review. |
| `refactor/maintenance-review-quality-enhancements-16048535916728849168` | ARCHIVE | Program B CI disk/quality churn. |
| `refactor/maintenance-review-quality-improvements-12931597326089652696` | ARCHIVE | Program B automated maintenance refactor. |
| `refactor/maintenance-review-quality-pass-925525883652807835` | ARCHIVE | Program B quality-pass churn. |
| `refactor/quality-and-complexity-improvements-5039841576911041607` | ARCHIVE | Program B algebraic-complexity work. |
| `refactor/quality-control-improvements-3511371844288069406` | ARCHIVE | Program B CI cleanup/refactor. |
| `refactor/quality-control-maintenance-11765386244386255808` | ARCHIVE | Program B CI-maintenance tip. |
| `refactor/quality-improvement-maintenance-9956075923500594985` | ARCHIVE | Program B algebraic-quality work. |
| `staging` | ARCHIVE | Fully merged January release-era ancestor. |

### GitHub mirror PR recommendations (30)

| PR | Base ← head | Verdict | One-line reason |
|---|---|---|---|
| #27 | `develop` ← `feat/maintenance-review-4878658191108315390` | CLOSE | Program B maintenance. |
| #28 | `develop` ← `refactor/enhance-code-quality-scores-16907359726872318662` | CLOSE | Unmeasured quality-score refactor. |
| #29 | `develop` ← `feat/maintenance-review-5265908703401954886` | CLOSE | Program B blending/maintenance. |
| #30 | `develop` ← `jules-9014215542651342316-3943b6a9` | CLOSE | Jules maintenance. |
| #32 | `develop` ← `refactor/maintenance-quality-review-4428541114331801030` | CLOSE | Program B CI maintenance. |
| #34 | `develop` ← `jules-1668043902756157847-f0c6832a` | CLOSE | Jules maintenance/security churn. |
| #35 | `develop` ← `jules-4074474147775153444-21fc1b20` | CLOSE | Jules blending/quality churn. |
| #36 | `develop` ← `feat/maintenance-review-6673365789579649169` | CLOSE | Program B algebraic maintenance. |
| #37 | `develop` ← `refactor/maintenance-review-6912658892127868145` | CLOSE | Program B maintenance with failing historical checks. |
| #38 | `main` ← `ci/self-hosted-only` | CLOSE | Obsolete, incomplete runner selector; recreate only if still needed. |
| #39 | `develop` ← `jules-5732845235305945309-f0a7af81` | CLOSE | Jules optimization maintenance. |
| #40 | `develop` ← `refactor/automated-maintenance-review-4463605815494777710` | CLOSE | Automated Program B maintenance. |
| #41 | `develop` ← `jules-1247812531793930591-2172b6e5` | CLOSE | Jules quality/CI maintenance. |
| #42 | `develop` ← `jules-7298396753358203747-31d965ab` | CLOSE | Jules algebraic-quality churn. |
| #43 | `develop` ← `refactor/quality-and-complexity-improvements-5039841576911041607` | CLOSE | Program B algebraic refactor. |
| #44 | `develop` ← `refactor/quality-improvement-maintenance-9956075923500594985` | CLOSE | Program B score/algebraic churn. |
| #45 | `develop` ← `jules-15592453777948541705-4ce36ee5` | CLOSE | Jules maintenance/quality. |
| #46 | `develop` ← `jules-5988585040434365535-4afdb6cf` | CLOSE | Jules progressive-loader/algebraic churn. |
| #47 | `develop` ← `refactor/comprehensive-maintenance-review-4910817366475062095` | CLOSE | Program B broad maintenance. |
| #48 | `develop` ← `jules-6202098072530946860-4e8abd18` | CLOSE | Jules core/optimization churn. |
| #49 | `develop` ← `jules-18422569073778592940-d0d75b23` | CLOSE | Jules algebraic-complexity churn. |
| #50 | `develop` ← `jules-4285862908706332243-e4d1adbd` | CLOSE | Jules maintenance/runner-disk churn. |
| #51 | `develop` ← `fix/maintenance-review-improvements-15552731785882124168` | CLOSE | Program B optimization/docstring churn. |
| #59 | `main` ← `ci/akula-gpu-runner` | KEEP | Sole viable PoC CI proposal; rebase on current `main`, then require honest green checks. |
| #61 | `develop` ← `feat/automated-maintenance-review-11775001417740100142` | CLOSE | Program B automated maintenance. |
| #62 | `develop` ← `maintenance/quality-refactoring-and-blending-8432855283715579363` | CLOSE | Program B blending/quality. |
| #63 | `develop` ← `refactor/maintenance-review-quality-enhancements-16048535916728849168` | CLOSE | Program B CI/maintenance. |
| #64 | `develop` ← `refactor/maintenance-quality-review-1492356723521003984` | CLOSE | Program B static-quality maintenance. |
| #65 | `develop` ← `maintenance/comprehensive-review-and-quality-improvements-7404159461572907830` | CLOSE | Program B maintenance with a historical failure. |
| #67 | `develop` ← `refactor/quality-control-maintenance-11765386244386255808` | CLOSE | Program B CI maintenance. |

## `tzervas/memory-gate`

Baseline: `forgejo/main` at `95f24f3`. Heads: **33**. Counts: **KEEP 2**, **CHERRY-PICK 3**,
**ARCHIVE 28**. Local `main` is 10 behind/1 ahead; its sole local commit is already preserved as
`local/kang-main-wip` and must not be overwritten.

| Forgejo head | Verdict | One-line reason |
|---|---|---|
| `chore/comprehensive-maintenance-review-15268910131248222000` | ARCHIVE | Dev-rooted maintenance stack, not an atomic Phase 1 change. |
| `chore/fleet-ci-harden-catchup` | ARCHIVE | Mixes useful hygiene with an explicit always-green gate. |
| `chore/p26-fleet-standards` | ARCHIVE | Patch-equivalent content already on Forgejo `main`. |
| `chore/secret-hygiene` | CHERRY-PICK | Narrow gitleaks/history hygiene; review its allowlist before landing. |
| `chore/version-policy-cz-alignment` | ARCHIVE | Dev/maintenance stack, not a standalone version change. |
| `ci/align-workflow-triggers-dev` | ARCHIVE | Expands the excluded `dev` program. |
| `ci/false-green-sweep` | CHERRY-PICK | Removes masked pytest/Trivy failures; land before feature work. |
| `ci/host-homelab-product-ci` | ARCHIVE | Adds invalid `scribe-cpu-build`; re-author with canonical six labels. |
| `ci/oss-self-hosted-tools` | ARCHIVE | Suppresses scans and permits a GitHub bot push, both out of policy. |
| `ci/self-hosted-only` | ARCHIVE | Partial selector still lacks required CPU/homelab labels. |
| `copilot/review-task-tracker-docs` | ARCHIVE | Dev-derived documentation/maintenance stack. |
| `dependabot/uv/uv-c20c29ea69` | ARCHIVE | Unvalidated stale Torch/lock refresh. |
| `dev` | ARCHIVE | Explicitly excluded second program, 10 behind/15 ahead. |
| `docs/pm-suite` | CHERRY-PICK | Select archaeology only after revalidating every current-state claim. |
| `feat/multi-model-embeddings` | ARCHIVE | Same head as embed binding; substantive catalog already landed on `main`. |
| `feat/self-hosted-podman-ci-triggers` | ARCHIVE | Older runner/trigger variant superseded by baseline. |
| `fix/embed-model-binding` | ARCHIVE | Duplicate of catalog/binding content already on `main`. |
| `fix/local-validation-commitizen` | ARCHIVE | Older quality stack superseded by baseline. |
| `fix/reopen-issues-yaml-block-scalar` | ARCHIVE | Repairs YAML by adding prohibited `ubuntu-latest`; re-author if needed. |
| `jules-11523506872801838573-3f1d6a1a` | ARCHIVE | Jules maintenance stack. |
| `jules-12840983974763317780-2b2fdaa5` | ARCHIVE | Jules maintenance stack. |
| `jules-12957321351251519916-72d51e53` | ARCHIVE | Jules maintenance stack. |
| `jules-15945367036246028596-9a7d7bed` | ARCHIVE | Jules maintenance stack. |
| `jules-5999440526354225705-b3910e48` | ARCHIVE | Jules maintenance stack. |
| `jules-7870355771063655523-85dcdc48` | ARCHIVE | Jules maintenance stack. |
| `local/kang-main-wip` | KEEP | Preserves local `RUST_PORT_PROMPT.md` divergence; retain but do not integrate. |
| `main` | KEEP | Canonical Python Phase 1 baseline. |
| `maintenance-review-12816755469846097293` | ARCHIVE | Maintenance stack. |
| `maintenance-review-fixes-4406937931998988862` | ARCHIVE | Maintenance stack. |
| `maintenance/review-fixes-8803522786581151071` | ARCHIVE | Maintenance stack. |
| `maintenance/review-quality-improvements-10357553646650570129` | ARCHIVE | Maintenance stack. |
| `maintenance/review-reconciliation-9827415239851104756` | ARCHIVE | Maintenance stack. |
| `merge/main-into-dev` | ARCHIVE | Excluded dev reconciliation, not a main candidate. |

### GitHub mirror PR recommendations (18)

| PR | Verdict | One-line reason |
|---|---|---|
| #57 | CLOSE | Maintenance reconciliation into `dev`. |
| #56 | CLOSE | Comprehensive maintenance into `dev`. |
| #55 | CLOSE | Jules maintenance into `dev`. |
| #54 | CLOSE | Jules maintenance into `dev`. |
| #53 | CLOSE | Maintenance into `dev`. |
| #52 | CLOSE | Maintenance-quality stack into `dev`. |
| #51 | CLOSE | Jules maintenance into `dev`. |
| #50 | CLOSE | Maintenance fixes into `dev`. |
| #49 | CLOSE | Homelab selector includes invalid `scribe-cpu-build`; re-author. |
| #46 | CLOSE | Expands the excluded `dev` program. |
| #43 | CHERRY-PICK | Take the focused false-green correction without merging mirror PR. |
| #40 | CLOSE | `main`-into-`dev` reconciliation. |
| #38 | CLOSE | Dev-rooted version/maintenance stack. |
| #37 | CHERRY-PICK | Take only revalidated archaeology. |
| #36 | CLOSE | Uses prohibited `ubuntu-latest`; re-author if needed. |
| #35 | CLOSE | Suppressed scans plus GitHub bot push. |
| #34 | CHERRY-PICK | Narrow secret hygiene candidate. |
| #32 | CLOSE | Unvalidated stale dependency refresh. |

## `tzervas/memory-gate-rs`

Baseline: `forgejo/main` at `02bcced`. Heads: **34**. Counts: **KEEP 2**, **ARCHIVE 23**,
**CLOSE 9**. Rust is read-only reference until Python Phase 3 exits.

| Forgejo head | Verdict | One-line reason |
|---|---|---|
| `archive/20260715-dev` | ARCHIVE | Dated historical snapshot contained by main. |
| `archive/20260715-main` | ARCHIVE | Dated historical snapshot contained by main. |
| `chore/commitizen` | ARCHIVE | Superseded one-file release tooling. |
| `chore/p16-trit-vsa-debt` | ARCHIVE | VSA debt note, not a Python feature request. |
| `chore/p24f-reuse-bootstrap` | ARCHIVE | Overtaken license bootstrap. |
| `chore/p26-fleet-standards` | ARCHIVE | Workflow/template divergence with no memory behavior. |
| `chore/semver-0x-compliance` | ARCHIVE | Deferred Rust release/version churn. |
| `chore/semver-baseline-v1.0.1` | ARCHIVE | Deeply stale W2-era baseline. |
| `chore/tero-index-cabal-ready` | ARCHIVE | Historical contained indexing kickoff. |
| `chore/w2-code-wiring-facade` | ARCHIVE | Old CommonMemory facade sketch. |
| `chore/w2-commonmemory-impl-sketch` | ARCHIVE | Explicit implementation sketch with no current consumer. |
| `chore/w2-more-wiring` | ARCHIVE | Wiring notes/facade drift. |
| `chore/w2-rollout-docs-wiring` | ARCHIVE | Docs/index wiring only. |
| `ci/oss-self-hosted-tools` | ARCHIVE | CI proposal; Rust landing is deferred. |
| `dependabot/cargo/minor-and-patch-07256db2e0` | CLOSE | Stale lockfile-only bulk bot update. |
| `dependabot/cargo/rand-0.10.2` | CLOSE | Stale dependency proposal. |
| `dependabot/cargo/rusqlite-0.40.1` | CLOSE | Stale dependency proposal. |
| `dependabot/cargo/sha2-0.11.0` | CLOSE | Stale dependency proposal. |
| `dependabot/github_actions/actions/checkout-7` | CLOSE | Mirror bot workflow bump. |
| `dependabot/github_actions/actions/setup-python-7` | CLOSE | Stale workflow bump. |
| `dependabot/github_actions/astral-sh/setup-uv-7` | CLOSE | Stale workflow bump. |
| `dependabot/github_actions/codecov/codecov-action-7` | CLOSE | Stale workflow bump. |
| `dev` | ARCHIVE | Diverged July fleet/doc line. |
| `docs/pm-suite` | ARCHIVE | Stale current-state and embedding ADR claims. |
| `feat/ci-push-pr-triggers` | ARCHIVE | Superseded CI trigger proposal. |
| `feat/hypha-kv-tiers` | KEEP | One focused RAM/disk tier-policy commit; read as Python gap reference, do not merge. |
| `feat/integration-facade` | ARCHIVE | Facade behavior is effectively represented in later `main`. |
| `feat/multi-model-embeddings` | ARCHIVE | Old experiment that omits later gateway/eval work. |
| `feat/self-hosted-podman-runner` | ARCHIVE | Superseded workflow line. |
| `feat/wave-c-golden-recall` | ARCHIVE | Recall harness landed on `main`; endpoint otherwise regresses files. |
| `feature/mint-m1-domain-facade` | ARCHIVE | Contained historical domain/facade work. |
| `fix/embed-model-binding` | ARCHIVE | Useful binding reference bundled with stale multi-model work. |
| `main` | KEEP | Canonical gateway/store/consolidation/metrics/golden-recall reference. |
| `merge/main-into-dev` | CLOSE | Transient July dev reconciliation. |

### GitHub mirror PR recommendations (13)

All thirteen should be recommended **CLOSE** on the GitHub operator mirror. Preserve the Forgejo
`feat/hypha-kv-tiers` head as read-only reference even while closing mirror PR #62.

| PR | Head | Verdict | One-line reason |
|---|---|---|---|
| #63 | `dependabot/cargo/minor-and-patch-07256db2e0` | CLOSE | Lockfile-only Rust update during the Python-first freeze. |
| #62 | `feat/hypha-kv-tiers` | CLOSE | Keep Forgejo reference; do not advance a Rust mirror PR now. |
| #59 | `dependabot/github_actions/astral-sh/setup-uv-7` | CLOSE | Stale workflow bump. |
| #58 | `dependabot/github_actions/actions/setup-python-7` | CLOSE | Stale workflow bump. |
| #57 | `dependabot/github_actions/actions/checkout-7` | CLOSE | Mirror-only bot workflow bump. |
| #56 | `dependabot/github_actions/codecov/codecov-action-7` | CLOSE | Stale workflow bump. |
| #55 | `merge/main-into-dev` | CLOSE | Non-program dev reconciliation. |
| #54 | `docs/pm-suite` | CLOSE | Stale roadmap/current-state claims. |
| #53 | `chore/semver-0x-compliance` | CLOSE | Deferred Rust release work. |
| #52 | `ci/oss-self-hosted-tools` | CLOSE | No Rust landing in this phase. |
| #50 | `dependabot/cargo/sha2-0.11.0` | CLOSE | Stale dependency bump. |
| #49 | `dependabot/cargo/rusqlite-0.40.1` | CLOSE | Stale dependency bump. |
| #48 | `dependabot/cargo/rand-0.10.2` | CLOSE | Stale dependency bump. |

## Audit checks and blockers

- CSD PoC CPU slice: `CUDA_VISIBLE_DEVICES='' uv run pytest tests/test_poc_*.py` → **21 passed,
  5 skipped** in 0.58 s. No VL-JEPA, mHC, quantum, or compression claim is implied.
- Rust reference at `feat/hypha-kv-tiers`: `cargo test --all-features` → **121 passed, 32
  ignored** across unit/integration/doctests. Golden recall and backend/model-download tests remain
  ignored; this is not a quality measurement.
- Python Forgejo `main`, with CUDA hidden: **135 passed, 8 warnings** in 65.62 s. Three benchmark
  cases and five lifecycle/consolidation/gateway cases emitted unawaited-coroutine warnings; treat
  the suite as useful coverage evidence, not an honest performance or warning-clean gate.
- Homelab CPU runner service `forgejo-runner-cpu` was **inactive**, and the Forgejo admin runner API
  returned no registered runners. Phase 1 cannot claim CI green until this is repaired and a real
  run completes.
- Private HF model `tzervas/cogsyndelta` and dataset `tzervas/cogsyndelta-eval` were verified
  accessible and private. No card or checkpoint was changed.
- 3090 LocalAI remained resident. The 5080 exclusive-seq queue was empty. No GPU service was paused.
- The three canonical Markdown deliverables were written to `akula-csd-kb`. Qdrant collection
  `akula-csd-kb` was created green at 1024 dimensions. Vector reindex is **not complete**: ComfyUI is
  healthy and resident on the 5080 at roughly 11.4 GiB, while the remote worker has neither the RAG
  index script/runtime nor the vault/Qdrant path. The current `csd-kb-index` wrapper passes a
  prime-only venv path through SSH, so it fails remotely. An initially misrouted prime-state queue
  entry was lock-safely cleared after discovery; no executable job is being represented as queued.
  The Markdown vault remains canonical and the collection currently has zero points.

## Immediate operator actions after approval

1. Rebase CSD `ci/akula-gpu-runner` onto current `main`; do not merge until checks are truly green.
2. Restore/register `homelab-cpu`, then land the Python false-green correction and canonical
   `[self-hosted, linux, x64, podman, compute-cpu, host-homelab]` selectors as a fresh atomic PR.
3. Close the listed GitHub mirror PRs only by operator action; do not bot-push GitHub.
4. Leave every archive head in place until an explicit remote-retention operation is approved.
5. Repair 5080 indexing as one operations change before claiming reindex success: use the remote
   state file watched by `gpu-timeshare-5080.timer`; enforce admission against resident Comfy;
   provide a pinned Qwen3-1024 index runtime, index script, and read-only staged vault on gpu5080;
   route the job to that indexer rather than `model-kb-ingest`; connect back to the prime Qdrant
   through an explicit authenticated route; then verify dimension 1024 and a nonzero point count.
