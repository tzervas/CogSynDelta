# Goal ledger — CogSynDelta / memory-gate

Living list for `/csd-goal-loop` and `/csd-python-first-drive`. Update when
evidence exists. Horizon stays **blocked** until Phase 1 exit. Not `STATUS.md`.

**Drive split (quota):** hosted Grok = **planner / safety / unblock / context
pack only**. Do **not** use grok-4.6 as the implementer. Autodev =
`scripts/csd-autodev-loop` + `csd-model-router` + `csd-localai-queue` →
3090 `local/code`. GPU splits: `scripts/csd-gpu-plan`. 5080 =
CUDA/tests/helpers behind the lock. WebUIs on homelab. Stop hosted
`/csd-python-first-drive` implement runs. Pack: `docs/program/GPU-POOL.md`.

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
| G-SQL | SQLite+sqlite-vec durable store | met | PR #8 merged `a1f648d` (head `9c6c09d`). 15 jobs ran (quality, security, fleet-ci python, unit/integration/regression, CI Complete, gitleaks, trivy, commitizen). | n/a | P1-06 |
| G-QD | Qdrant 1024-d Qwen3, fail-closed 384 | met | PR #9 merged `a823268` (head `5546ad7`). 15 jobs ran (quality, security, fleet-ci python, unit/integration/regression, CI Complete, gitleaks, trivy, commitizen). | n/a | P1-07 |
| G-LIFE | learn → retrieve → consolidate → persona through restart | open | After G-SQL, G-QD, CLS, persona | later | P1-16 |

## Lab / ops (keep true)

| ID | Goal | Status | Notes |
|---|---|---|---|
| G-GPU | 3090 `local/code` 32k + 5080 autodev | met | 32k ~16570 used / 5993 free. Comfy masked. Share-small leftover. Never dual 14B. |
| G-CAP | Safeguard budgets fit lab GPUs | met | 14 GiB default (5080), 20 GiB ceiling (3090 Ti). Pooled one-model may use almost full VRAM minus 1536 MiB headroom/card. |
| G-GPUCI | 5080 Forgejo GPU runner + lock queue | open | Runner `gpu5080-tzervas` labels gpu/5080/host-gpu5080. `gpu5080.lock` vs timeshare. Workflow `.github/workflows/gpu-5080.yml`. |
| G-WHEEL | Homelab offline PyPI (+ crates dir) | open | `/data/pypi-offline` (24 wheels starter). No WAN on 5080. CUDA torch = Comfy image. |
| G-POOL | LAN pool 3090+5080 inference (~40 GiB) | open | Stage A live. Stage B: large **MoE** / `pool/large` via mixed llama.cpp+vLLM+bitnet-cpp after images/GGUFs. Later: many small CSD region models (`G-SHARE`). Not `G-SPLIT`. |
| G-SCALE | Tiny CPU → small 5080 → medium when measured | open | Doc `CSD-SCALE-LADDER.md`. Grafana sat only if `scale_ladder.json` `green=true`. `hf/autodev` missing → **mint HF**. Never copy `gpu/huggingface-token`. |

## Blocked / horizon

| ID | Goal | Status | Unblock |
|---|---|---|---|
| G-CHROMA | Reintegrate Chroma | blocked | Named patched RC/stable in GHSA. PR #3 stays red/unmerged. |
| G-TRAIN | Region train → bedrock → foundation | blocked | After `G-LIFE`. Per-region `region-<id>` data, then `common` for overall. Two HF weight artifacts. One-GPU Python first. |
| G-SPLIT | Pool 3090+5080 for one CSD mind | blocked | Phase 5. After Phase 3 one-GPU proof. Distinct from `G-POOL` (serving). |
| G-SHARE | 256 micro specialists + interned RO weights | blocked | After `G-POOL` Stage A measured. Dedupe identical tensors; isolate KV; **batch parallel** GEMMs on interned blocks. Policy: `scripts/csd-weight-intern`. Pack: `docs/program/GPU-SHARE.md`. ADR-0016. |
| G-1080 | 1080 Ti in 5080 workstation as tagged node | blocked | Operator hardware first (slot, 1300 W PSU, factory cooler). Then `pascal`/`sm_61` caps. Do not assume the card exists. |

## Priority

`G-QD` met. P1-08 merged `2c11c3f` (PR #11). Next closeable: P1-09
retrieve/domain (not `G-LIFE` yet). Skip stalled. Never start
`G-TRAIN`, `G-SPLIT`, `G-SHARE`, or `G-1080` from an autoloop.
Never reset `kang-main-wip`.
