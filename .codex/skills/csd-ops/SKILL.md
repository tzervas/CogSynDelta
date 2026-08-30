---
name: csd-ops
description: Use 3090/5080 the right way, pause/restore LocalAI, Forgejo git, Hugging Face private model cards. Use when Codex needs GPU, CI runners, remotes, or to upload checkpoints.
---

# CogSynDelta lab ops

Read `docs/CODEX-OPS.md`. Short form:

- **3090:** one LocalAI GGUF (`local/code`). Do not pause for RAG search. Pause (docker stop/start trap) only if a CSD job must own the 3090; prefer 5080 for CUDA.
- **5080:** `akula-ai-platform/scripts/with-gpu-5080 --mode exclusive-seq -- <cmd>`. Timeshare enqueue if Comfy is up. Never `compute-cpu` here.
- **Index:** `./scripts/csd-kb-index` (5080). Never `docker stop akula-localai` for that.
- **Git:** Forgejo `git.vectorweight.com` **deep trees** (`tzervas/CogSynDelta`, `memory-gate`, `memory-gate-rs`). Never shallow-main. Push with `akula-ai-platform/scripts/cabal-forgejo-push-full-tree`. `.github/workflows` with `runs-on: [self-hosted, linux, x64, podman, compute-cpu, host-homelab]`. No `.forgejo/workflows/`. No GitHub.com bot push. Self-review, no required approvals.
- **Branches:** Audit every head/PR: KEEP / CHERRY-PICK / ARCHIVE / CLOSE. See `docs/PROGRAM-GOALS.md`. Do not merge Jules/`develop` sediment.
- **Weights:** private HF `tzervas/cogsyndelta` + dataset `tzervas/cogsyndelta-eval`. Card = `templates/hf-model-card.md`. Not Forgejo LFS for GGUFs.
- **Python first.** memory-gate Python is Phase 1. `memory-gate-rs` is reference + later rewrite. No Rust port this pass.
