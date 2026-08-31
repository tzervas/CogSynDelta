# Self-hosted drive targets

**Status**: horizon contract (operator 2026-08-31). **Not measured.** `STATUS.md`
is still the only place for PoC-1/2/3 numbers. Self-hosted implementers drive
**toward** this document after Phase 1 memory-gate exits. Do not train CogSynDelta
weights or claim VL-JEPA / 10× compression from this file.

Vault twin: `akula-csd-kb/Program/Self-Hosted-Drive-Targets.md`.

## What we are building

One **mind** with specialized **regions**, not a swarm of agents. Regions are
experts of one brain (MoE-adjacent submodels). Personas select basins.
Memory-gate is the dynamic learning substrate they share.

**Region specialization (Phase 3, not now):** each region is a submodel.
Train it at high fidelity (large param count, large data) **then quantize**
so its VRAM/latency sits at the measured performance/size balance on *this*
lab. After every region and subregion is dialed, compose the full CogSynDelta
stack: quantized regions + interconnect/gate + memory. Goal is
**foundation-model quality, hardware-dialed residency** — not a 70B that
cannot load. Same rule as the assist GPU matrix: pack what fits; never dual
14B; never claim PoC ResidualMLP/LatentVAE already *is* those specialists.
`STATUS.md` remains the only measured PoC. **Do not train CSD weights until
`G-LIFE`.**

**Split across mismatched GPUs (after one-GPU Python proof):** treating the
3090 Ti + 5080 as one pool, routing shards of *one* mind, is an experimental
follow-on. Prove the Python stack on **one GPU** first (Phase 1–3). Do not
start pipeline-parallel / custom dispatch until that bar is green.

The eventual stack is a **symphony**:

| Layer | Role | Persistence |
|---|---|---|
| Small / specialized models (SLM, task heads) | Region specialists (code, retrieve, route, compress) | Checkpoints on private HF `tzervas/cogsyndelta` |
| Larger LLMs | Occasional high-reasoning / orchestration assist | Hosted Grok is gatekeeper; 3090 `local/code` 32k + share-small leftover |
| Classical ML / other internals | Routers, compressors, load-balance, eval probes | Code in git; weights only if measured |
| Interconnect / gate | Softmax top-k + Switch aux (PoC-3). mHC later | Trained when 5080 exclusive-seq is free |
| Dynamic memory | Python memory-gate: learn → retrieve → consolidate → persona | SQLite+vec local; Qdrant 1024-d Akula |

## Sequencing (hard)

1. **Now — Phase 1**: Python `memory-gate` board `P1-00`…`P1-17`. Honest CI.
   SQLite+vec + Qdrant. No Chroma nightly. Merge only **legitimately green**.
2. **Phase 2**: Wire that memory into CogSynDelta `memory/`. Keep PoC-1..3 green.
3. **Phase 3 — this target**: Automate **region specialization** off Hugging Face
   data (copy public sets into private `tzervas/cogsyndelta-eval` when a row
   needs them). Train high-fidelity regions, **quantize** to lab VRAM, train
   interconnect. Offload storage. Eval/bench along the way. Prove the full
   Python stack on **one GPU** (5080 exclusive-seq for train/eval; 3090 stays
   assist). Never pause 3090 LocalAI for RAG.
4. **Phase 4**: Rust rewrite after Python CSD is production-ready.
5. **Phase 5 (experimental)**: pool mismatched GPUs (3090 Ti + 5080) for
   split inference/training of one CSD mind. Not before Phase 3 one-GPU proof.

## Hardware and storage

| Resource | Persistent | Ephemeral |
|---|---|---|
| Git (Forgejo deep trees) | Code, specs, ADRs, eval scripts | Worktree checkouts |
| Private HF | Checkpoints, golden eval dumps | Download caches |
| Qdrant `akula-csd-kb` 1024-d | Indexed experiment notes | CUDA index jobs |
| Obsidian `akula-csd-kb` | Plans, handoffs, measured rows | Session chatter |
| 3090 LocalAI GGUF | `local/code` Q4 native 32k (~16.5 GiB measured) + share-small leftover | Dual 14B |
| 5080 | Measured train/eval; idle embed helper | Comfy (masked for autodev until unmask) |
| `/tmp`, CI workdirs, failed runs | — | Delete after the log is in Forgejo |

Never mix 384-d into shared Qdrant. Never write `tzervas-dev-kb` or
`akula-model-kb`.

## Eval bar

Every region or interconnect change needs: pinned data revision, metric, and a
row in `STATUS.md` or the eval card. A skip is not a bench. No 10× / VL-JEPA /
quantum claims without a test you ran.

## Self-hosted worker rules

- Read `docs/program/IMPLEMENTER-HANDOFF.md` first.
- No WAN. Lab Forgejo, localhost Qdrant/LocalAI, and the csd-kb vault only.
- If you need the internet (PyPI, GitHub chroma RC, HF download), **stop and
  ask the hosted Grok gatekeeper**. Do not curl the public net.
- One architecture change per Forgejo PR. Do not merge unless required checks
  **legitimately green**.
- Ask the gatekeeper to index durable notes into `akula-csd-kb` (keyword now;
  CUDA index via `./scripts/csd-kb-index` enqueue, never 3090).
