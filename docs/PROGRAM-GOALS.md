# CogSynDelta program goals (persistent driver)

This file is the **planning and allocation contract**. Codex, Grok, and any
smaller implementer model must treat it as the current program, not as
marketing. Measured truth still lives in `STATUS.md` and in tests.

`docs/HANDOFF-ARCHITECTURE.md` is unvetted ChatGPT context. Reconcile every
claim with live Python source before building on it.

## North star

A **MoE-adjacent single brain with specialized centers**, not a swarm of
complete agents:

| MoE idea | CogSynDelta analog |
|---|---|
| Expert | Cognitive **region** (one module of one mind) |
| Gating / router | Softmax top-k + Switch aux load-balance (PoC-3). Interconnect / mHC later |
| Shared residual stream | Shared latent + compressed memory substrate |
| Control loop | Explore → cull → meta-optimize |

**Python-first.** Prove memory, personas, regions, and routing in Python with
the PyTorch / Triton ecosystem. **Rust later** (`memory-gate-rs`, then a CSD
rewrite on Burn / CubeCL / Candle) only after the Python path is production-ready
and exhaustively validated.

## What we will not do in this program

- Merge `develop` sediment, Jules “100/100 quality” bots, or README-era VL-JEPA /
  quantum / agent-fleet claims into `main` to “catch up.”
- Start the Rust rewrite before Python memory-gate **and** Python CogSynDelta
  are measured complete.
- Push GGUFs or checkpoints to Forgejo. Weights go to private Hugging Face.
- Pause LocalAI on the 3090 Ti for RAG **search**, or run two 14B GGUFs at once.
- Use GitHub.com as the day-to-day forge. Operator mirrors to GitHub when ready.

## Repositories (Forgejo is the working forge)

Full **deep trees** (every branch + tag, not a shallow `main`) live on
`https://git.vectorweight.com`:

| Repo | Role now | Later |
|---|---|---|
| `tzervas/memory-gate` | **Phase 1 vehicle.** Python dynamic memory + persona backend | Stay canonical Python |
| `tzervas/memory-gate-rs` | **Reference + rewrite target.** VSA / `HolographicStore` / CLS | Port *after* Python is proven |
| `tzervas/CogSynDelta` | **Phase 2–3 vehicle.** Regions, router, Triton kernels, integration | Rust rewrite after Python CSD is proven |

Working branch for harness + this program: `feat/agent-harness`.
Program-of-record for measured PoC: `main` (`STATUS.md`).
Do not author on `main` / `staging` / `develop` / `dev`.

**Self-review, no required approvals.** Forgejo PRs on these tzervas repos may
merge when CI is honestly green and a self-review COMMENT exists. Cabal never
APPROVEs. Operator still owns the GitHub.com mirror.

## Phase 0 — Plan, inventory, keep/drop (**complete** 2026-08-30)

Official artifacts (implement against these, not the ChatGPT handoff):

- [docs/program/PYTHON-FIRST-PLAN.md](program/PYTHON-FIRST-PLAN.md)
- [docs/program/PHASE-0-KEEP-DROP.md](program/PHASE-0-KEEP-DROP.md)
- [docs/program/PYTHON-MEMORY-GATE-GAP-MAP.md](program/PYTHON-MEMORY-GATE-GAP-MAP.md)
- [docs/program/PHASE-1-TASK-BOARD.md](program/PHASE-1-TASK-BOARD.md)

Phase 0 Codex planning is accepted. **Implementation is in progress** (Goal 0
ops + Phase 1 board). Drive one ID at a time with `/csd-python-first-drive`.

**Do not close GitHub PRs from the agent.** CLOSE rows are operator
recommendations. Operator closes on GitHub.com when they mirror.

### Keep / drop protocol (mandatory)

Dig **all** branches. A full clone is on Forgejo so you can. Verdicts:

| Verdict | Meaning | Typical examples |
|---|---|---|
| **KEEP** | Lands on the Python-first program after rebase onto current `main` | CSD `feat/agent-harness`, CSD `#59` `ci/akula-gpu-runner`, unique memory-gate CI that actually fails closed, `feat/hypha-kv-tiers` *as a later-rs idea* |
| **CHERRY-PICK** | One commit or file is useful; the branch as a whole is not | OSS SAST, host-homelab `runs-on`, secret hygiene, a real test |
| **ARCHIVE** | Research fossil. Cite, do not merge | CSD `develop` / `claude/vsa-*` / copilot hybrid; mg-rs `archive/*`; VSA holographic sketches |
| **CLOSE** | Noise. Close PR, leave branch until operator deletes | Jules / `refactor/*` / `maintenance/*` “quality 100/100” on `develop` or `dev`; Dependabot that fights the Python-first pin |

Starting hypotheses (must be **re-validated** against the live trees, not copied blindly):

**CogSynDelta (30 open GitHub PRs; 55 heads)**

- KEEP working: `main` (PoC-1..3), `feat/agent-harness`, rebase `#59`.
- ARCHIVE: `develop`, `archive/*`, `claude/*`, `copilot/*`, algebraic-training `libs/`.
- CLOSE: ~28 Jules/maintenance PRs onto `develop`. `#38` `ci/self-hosted-only` is far behind — cherry-pick labels only.

**memory-gate Python (18 open PRs; 32 heads)** — **Phase 1 source of truth**

- KEEP: `main` (production-intent Python). CI PRs on `main` that make gates able to fail (`ci/false-green-sweep`, `ci/oss-self-hosted-tools`, `ci/host-homelab-product-ci`) after rebase.
- CHERRY-PICK: `feat/multi-model-embeddings` / `fix/embed-model-binding` only if they match the Qwen3-1024 / akula-csd-kb plane; `docs/pm-suite` if honest.
- ARCHIVE/CLOSE: Jules + `maintenance/*` onto `dev`. `dev` is **not** CSD `develop`, but treat it as a second program until the audit says otherwise. Do not merge `dev` into `main` to “catch up.”

**memory-gate-rs (13 open PRs; 34 heads)** — **not the implementation vehicle this phase**

- KEEP as reference: `main`, `feat/hypha-kv-tiers` (tiered KV — read, do not port yet).
- ARCHIVE: `archive/*`, `chore/p16-trit-vsa-debt`, `chore/w2-*` sketches, `feature/mint-m1-domain-facade` until Python needs that interface.
- CLOSE/defer: Dependabot on GitHub Actions that we will not run here; `merge/main-into-dev`.
- Use this tree to **write the gap list for Python**, not to add features in Rust.

## Phase 1 — Python memory-gate complete (**current**)

Goal: Python `memory-gate` is the **full dynamic memory and persona backend**,
validated, production-ready, wired to akula knowledge planes.

Clone: `/home/kang/code/personal/tzervas/python-ai/memory-gate` (working tree).
Forgejo: `git.vectorweight.com/tzervas/memory-gate` (deep tree).

Must close, with tests, not README:

1. Honest inventory vs `memory-gate-rs`: gateway, adapters, stores (in-memory,
   vector, sqlite, qdrant), consolidation (fast/slow CLS), domain filter,
   metrics, MCP/persona hooks, holographic/VSA **only if** the Python design
   needs it (default: no — keep VSA on the rs reference until measured).
2. Persona backend: instantiate + basin without mixing operator KB / model KB /
   gap-kb. Talk to `/akula-data/obsidian/akula-personas` as the store, same
   rules as `scripts/persona`.
3. Persistence that survives process restart. No “create_task store and forget”
   without backpressure and error surfacing (today’s gateway fire-and-forget
   is a gap).
4. Retrieval quality: golden recall set on CPU first, then 5080 embeddings
   exclusive-seq. Collection dim **1024 Qwen3** if it shares akula Qdrant.
   Never mix 384-d.
5. Production: typed public API, ruff + mypy, pytest that can fail, Forgejo
   Actions `runs-on: [self-hosted, linux, x64, podman, compute-cpu, host-homelab]`.
6. Checkpoints / embedding caches → `tzervas/cogsyndelta` (model) or
   `tzervas/cogsyndelta-eval` (dataset), private, with a real card.

**Exit:** operator can run learn → retrieve → consolidate → persona instantiate
on homelab CPU + 5080 embed without data loss, and a written gap table vs
memory-gate-rs is empty of *required* Python holes.

## Phase 2 — Integrate memory into CogSynDelta (Python)

1. Replace or wrap `src/cogsyndelta/memory/` with the Python memory-gate API.
2. Region registry + softmax router consume memory as the shared substrate
   (compressed), not as a second agent.
3. Personas select basin / domain; they do not become MoE experts.
4. Keep PoC-1..3 green. New tests for the integration, CUDA measure on 5080.

**Exit:** CSD train-route + memory round-trip measured on CPU and 5080.

## Phase 3 — CogSynDelta Python complete

Close remaining Python gaps on `feat/*` off `main`:

- More regions (still one brain), real load-balance, culling, meta-opt loop.
- **Triton fused kernels** on the 5080 for hot paths (attention-like mix,
  quant, compact). 3090 stays LocalAI assist unless a job **must** own it
  (pause with restore trap — `docs/CODEX-OPS.md`).
- Exhaustive eval: seed, steps, device in `STATUS.md` and HF `model-index`.
- No VL-JEPA / mHC / quantum claims unless a test you ran supports them.

**Exit:** STATUS tables cover the intended Python architecture; CI green on
homelab CPU; CUDA jobs via `with-gpu-5080 --mode exclusive-seq`.

## Phase 4 — Progressive Rust rewrite

Only after Phase 3 exit.

1. Port memory-gate Python → `memory-gate-rs` (Burn / Candle where tensors
   exist; CubeCL for GPU kernels). Preserve API semantics; do not invent a
   third memory model.
2. Port CogSynDelta hot paths. Triton kernels become CubeCL/Burn.
3. Dual-run eval vs Python until numbers match within agreed tolerance.

## Allocation (how work is sized)

Phase 0 is **planning** (this model / Codex). Implementation is **right-sized
smaller models** with matching reasoning:

| Slice | Who | Reasoning |
|---|---|---|
| Plan, strata, keep/drop, ADRs, HF card, GPU/Forgejo ops | Codex / Grok | High |
| Single test + one function on memory-gate | Small code model (`local/code` on 3090) | Low–medium |
| Triton kernel, CUDA measure, router train | Medium + 5080 exclusive-seq | Medium–high |
| Branch CLOSE recommendations (Jules noise) | Small | Low |
| Rust port (Phase 4 only) | Codex + smaller for mechanical modules | High then low |

One architecture change per PR. Conventional commits. Never push GitHub as a bot.

## Lab (non-negotiable)

See `docs/CODEX-OPS.md`.

- **3090 Ti (akula-prime):** one LocalAI GGUF, default `local/code`. Assist while
  5080 trains. Pause only if a CSD job must own the whole card; `trap` restore.
- **5080:** exclusive-seq CUDA / `csd-kb-index` / Comfy. Queue via timeshare.
  Parallelism = 3090 inference **plus** 5080 job, not two jobs on one card.
- **Homelab:** Forgejo CPU runner `homelab-cpu` labels
  `self-hosted,linux,x64,podman,compute-cpu,host-homelab`. Never `gpu`.
- **Hugging Face:** `tzervas/cogsyndelta` (private model) +
  `tzervas/cogsyndelta-eval` (private dataset). Card YAML is Hub metadata.
- **Knowledge:** RO `tzervas-dev-kb` + `akula-model-kb`; RW gap-kb; RW+reindex
  `akula-csd-kb` on 5080. Keyword retrieve on `:8091`/`:8092` is CPU.

## Success for the whole endeavor

1. Python memory-gate is the production memory/persona backend.
2. CogSynDelta Python is a working MoE-adjacent single brain, tested on both GPUs.
3. Every branch on the three Forgejo remotes has a keep/drop verdict.
4. Weights and cards live on private HF; code and CI live on Forgejo.
5. Rust rewrite has a measured Python baseline to beat — not a second design.
