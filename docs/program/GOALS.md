# Goal ledger — CogSynDelta / memory-gate

Living list for `/csd-goal-loop` and `/csd-python-first-drive`. Update when
evidence exists. Horizon stays **blocked** until Phase 1 exit. Not `STATUS.md`.

**Drive split:** hosted Grok = plan / WAN / merge-when-green / research feed.
Workflow implement = `grok-4.6` (host cannot spawn `local/code`). Keep 3090
`local/code` loaded anyway. Self-hosted does the function+test lift when a
session can call it. Wake on Forgejo CI (`scripts/forgejo-pr-watch.py` + Stop
hook), not a 2h timer.

## Active (Phase 1)

| ID | Goal | Status | Blocker | Sandbox? | Refs |
|---|---|---|---|---|---|
| G-CI | Fail-closed Forgejo CI on memory-gate | met | PR #1 `7c1cdeba` | n/a | P1-00 |
| G-SPEC | Lifecycle/data contract docs | met | PR #2 `91cc0f79` | n/a | P1-01 |
| G-ERR | Public `memory_gate.errors` + mapper | met | PR #4 `81af7f98` | n/a | P1-02 |
| G-FLEET | No cargo jobs on Python-only fleet-ci | met | PR #5 `7595bd53` | n/a | fleet-ci.yml |
| G-STORE | Store protocol + in-memory oracle | met | PR #6 `58934e6` | n/a | P1-04 |
| G-RAG | `akula-csd-kb` 1024-d points > 0 | met | 83 points / 20 files on 5080. Never 3090 | n/a | vault |
| G-ACK | Durable learn acknowledgement | met | PR #7 merged `6c0c6fb5`. Jobs ran (quality, security, fleet-ci python, tests, gitleaks, trivy, commitizen). | n/a | P1-03 |
| G-SQL | SQLite+sqlite-vec durable store | open | **Next.** After G-ACK. | yes | P1-06 |
| G-QD | Qdrant 1024-d Qwen3, fail-closed 384 | open | After G-STORE; 5080 for embed measure | CPU yes | P1-07 |
| G-LIFE | learn → retrieve → consolidate → persona through restart | open | After G-SQL, G-QD, CLS, persona | later | P1-16 |

## Lab / ops (keep true)

| ID | Goal | Status | Notes |
|---|---|---|---|
| G-GPU | 3090 `local/code` 32k + 5080 autodev | met | 32k ~16570 used / 5993 free. Comfy masked. Share-small leftover. Never dual 14B. |
| G-CAP | Safeguard budgets fit lab GPUs | met | 14 GiB default (5080), 20 GiB ceiling (3090 Ti). |
| G-GPUCI | 5080 Forgejo GPU runner + lock queue | open | Runner `gpu5080-tzervas` labels gpu/5080/host-gpu5080. `gpu5080.lock` vs timeshare. Workflow `.github/workflows/gpu-5080.yml`. |
| G-WHEEL | Homelab offline PyPI (+ crates dir) | open | `/data/pypi-offline` (24 wheels starter). No WAN on 5080. CUDA torch = Comfy image. |

## Blocked / horizon

| ID | Goal | Status | Unblock |
|---|---|---|---|
| G-CHROMA | Reintegrate Chroma | blocked | Named patched RC/stable in GHSA. PR #3 stays red/unmerged. |
| G-TRAIN | Region train → bedrock → foundation | blocked | After `G-LIFE`. Per-region `region-<id>` data, then `common` for overall. Two HF weight artifacts. One-GPU Python first. |
| G-SPLIT | Pool 3090+5080 for one CSD mind | blocked | Phase 5. After Phase 3 one-GPU proof. |

## Priority

`G-SQL` → `G-QD` → … → `G-LIFE`. Skip stalled. Never start
`G-TRAIN` or `G-SPLIT` from an autoloop. Never reset `kang-main-wip`.
