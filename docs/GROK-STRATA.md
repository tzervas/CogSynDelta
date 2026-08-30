# CogSynDelta strata (branches, PRs, what fits)

Observed 2026-08-30 from `origin` + GitHub PRs. Not `docs/BRANCH_STATUS.md` (that file is January 2026 and stale).

## Two programs (do not mix)

| Program | Tip | What it is | Fit with PoC-on-main |
|---|---|---|---|
| **A — measured PoC** | `origin/main` `f835cf0` (#66 README on top of #56–#60) | LatentVAE + RegionRegistry + softmax train-route. `STATUS.md` is truth. Python `>=3.12,<3.14`. | **This is the working tree.** |
| **B — develop sediment** | `origin/develop` `fd2cbd2` (2026-07-15, #14 VSA/compression) | Algebraic training, VSA, quality-score bots, README-era “production-ready” stack | **Does not fit** until a deliberate, tested merge. `main...develop` is a large two-way divergence. |

`origin/dev` (`28d0454`, #23 fleet badges) **is an ancestor of main** (fleet CI only). `origin/staging` is also ancestral and stale (Jan 2026).

## What landed on main (fits)

| PR | Head | Merged | Role |
|---|---|---|---|
| #56 | `feat/poc-e0-e2-foundation` | 2026-08-20 | PoC-1 substrate |
| #57 | `feat/poc-region-registry` | 2026-08-20 | PoC-2 registry + softmax |
| #58 | `feat/poc-cuda-measure` | 2026-08-20 | CUDA measure tests |
| #60 | `feat/poc-train-route` | 2026-08-20 | PoC-3 Switch aux LB train |
| #66 | `tzervas-patch-1` | 2026-08-25 | README only |

Local leftover branches `feat/poc-*` are pre-merge tips. Prefer `origin/main`. Worktrees under `CogSynDelta-wt-poc*` are prunable once you do not need the old SHAs.

## Open PRs

### Fits (rebase onto current main first)

| PR | Base ← head | Why |
|---|---|---|
| [#59](https://github.com/tzervas/CogSynDelta/pull/59) | `main` ← `ci/akula-gpu-runner` | PoC CUDA tests + akula-prime GPU smoke. Checks include lint, gitleaks, PoC pytest, GPU runner smoke (SUCCESS). About **1 commit behind main** (#66). |

### Does not fit the PoC program

| Cluster | Base | Count (open) | Why not |
|---|---|---|---|
| Jules / `refactor/*` / `maintenance/*` / `feat/maintenance-*` / `feat/automated-maintenance-*` | **develop** | ~28 | Automated “100/100 quality” and docstring churn on Program B (algebraic training / VSA). Merging them into main would revive sediment and fight `STATUS.md`. |
| [#38](https://github.com/tzervas/CogSynDelta/pull/38) `ci/self-hosted-only` | main | 1 | Fleet CI from July; **far behind** PoC main. Revisit only as a CI patch on current main, not a merge of that tip. |

Claude/copilot remote branches (`claude/*`, `copilot/create-pcn-vae-gan-hybrid`) are **research fossils** (ternary, VSA library, PCN-VAE-GAN hybrid). Cite; do not treat as next increment.

## How they fit the code-thropology layers

| Layer | Branch/PR home | Live? |
|---|---|---|
| `STATUS.md` + `src/cogsyndelta/poc/` | main #56–#60 | Yes |
| Akula GPU CI | #59 | Almost; rebase |
| VSA / holographic compression ADRs 0012/0008 | develop #14 + `claude/vsa-*` | Sediment |
| Algebraic training ADR 0009 + `libs/` | develop #10/#11 | Sediment |
| Quality-bot PRs | develop #27–#67 | Noise on sediment |
| README VL-JEPA / quantum / agent fleet | marketing on all tips | Not measured |

## Next honest increment (one slice)

1. Rebase #59 onto `origin/main` and land GPU CI if the operator wants it.
2. Dig Program A only (`/code-thropology`, `/csd-code-thropology`).
3. Do **not** merge develop or Jules PRs into main to “catch up.”
