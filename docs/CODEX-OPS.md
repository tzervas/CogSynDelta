# Codex ops: GPUs, pause, Forgejo, Hugging Face

Host agent on **akula-prime**. Two cards, no MIG. Timeslice is a **queue**, not extra VRAM. Parallelism is **across hosts**: 3090 text + 5080 media/train at the same time.

Python-first. This file is how to use the lab, not a claim that CogSynDelta already trains at 14B.

## Hardware

| Host | IP | GPU | Role for Codex |
|---|---|---|---|
| akula-prime | 192.168.1.98 | RTX 3090 Ti ~23028 MiB | **One** LocalAI GGUF (`local/code` for assist). Never two 14Bs. |
| gpu5080 | 192.168.1.252 | RTX 5080 ~16303 MiB | Exclusive-seq: CogSynDelta CUDA train/eval, RAG **index**, Comfy. Never `compute-cpu`. |
| homelab | 192.168.1.170 | none | Always-up UI + **Forgejo CPU Actions**. No GPU jobs. |

Akula scripts live in `/home/kang/code/personal/tzervas/akula-ai-platform`. Prefix `AKULA=…` below.

## 3090 Ti — inference assist (stay loaded)

Codex should **keep** LocalAI on the 3090 for chat/code while it works:

```bash
# health
curl -sS -H "Authorization: Bearer $LOCALAI_API_KEY" http://127.0.0.1:8080/v1/models
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv

# pick one alias (ttl unloads idle). Default assist: local/code
# OpenAI-compatible:
#   base  http://127.0.0.1:8080/v1
#   key   secret exec LOCALAI_API_KEY=gpu/localai-api-key
```

| Alias | When |
|---|---|
| `local/code` | Edits, tests, review (Qwen2.5-Coder-14B) |
| `local/fast` / `local/general` | Chat/summarize (Ministral 14B) |
| `local/reasoning` | Plans (gpt-oss-20b). Exclusive swap. |
| `local/code-heavy` | Only if 14B is not enough. Exclusive. |

**Do not** `docker stop akula-localai` for RAG **search**. Keyword-cpu: `AKULA/scripts/gpu-alloc --kind retrieve` then `:8091` / `:8092`.

### Pause the 3090 (only if CSD must own the whole card)

Prefer the 5080 for PyTorch. If you truly need the 3090 for a CogSynDelta CUDA experiment:

```bash
AKULA=/home/kang/code/personal/tzervas/akula-ai-platform
# queue, do not steal blindly
"$AKULA/scripts/gpu-timeshare" enqueue --kind harness-review --subject csd-3090 --host akula-prime
# exclusive: stop LocalAI, run, start. Restore even on failure.
docker stop akula-localai
trap 'docker start akula-localai' EXIT
# … CSD train/eval on 3090 …
```

Never leave LocalAI stopped. Never pause it for `csd-kb-index` (that is 5080).

### Load / swap a GGUF on the 3090

One resident file. Pull then let LocalAI load the yaml `name:`:

```bash
cd "$AKULA"
./scripts/pull-models --alias local/code --yes    # only with operator --yes
# yaml in config/localai/models/*.yaml — name: local/code, ttl: 180
# POST /v1/chat/completions {"model":"local/code",…}  # loads, unloads previous
```

Do not add a second `download_files` and dual-load. Uncensored aliases need `--yes` and are operator-selected.

## 5080 — train, index, media (exclusive-seq)

Never stop LocalAI on prime from 5080 jobs.

```bash
AKULA=/home/kang/code/personal/tzervas/akula-ai-platform
# one job; unloads whatever was on 5080
"$AKULA/scripts/with-gpu-5080" --mode exclusive-seq -- \
  bash -lc 'cd /path/to/CogSynDelta && uv run python -m cogsyndelta.poc.cli train --device cuda --steps 20'

# RAG / CSD vault index (this repo) — enqueues on 5080 local timeshare state via SSH.
# Does not SSH prime-only venv paths. Does not pause 3090 LocalAI. Does not stop Comfy.
./scripts/csd-kb-index

# Equivalent manual enqueue (must target 5080 state, not prime /akula-data/cabal/…):
ssh -o BatchMode=yes gpu5080 \
  env AKULA_TIMESHARE=/home/tzervas/akula-harness/state/gpu-timeshare.json \
  python3 /home/tzervas/akula-harness/scripts/gpu-timeshare \
  enqueue --kind rag-index --subject csd-kb --host gpu5080
ssh -o BatchMode=yes gpu5080 \
  env AKULA_TIMESHARE=/home/tzervas/akula-harness/state/gpu-timeshare.json \
  python3 /home/tzervas/akula-harness/scripts/gpu-timeshare status
```

Prime `gpu-timeshare` state (`/akula-data/cabal/gpu-timeshare.json`) is **not** the 5080 worker queue. The 5080 timer uses `AKULA_TIMESHARE=/home/tzervas/akula-harness/state/gpu-timeshare.json`. Enqueue there, or the job never claims.

`share-small` only if `nvidia-smi` free ≥ 8192 MiB and the helper is ≤ ~6 GiB. Otherwise refuse (exit 3). No MIG.

**Maximal leverage:** Codex infers on 3090 (`local/code`) **while** a 5080 job trains or indexes. That is the parallelism. Do not run Comfy and CSD CUDA on the 5080 at once. Reindex needs: indexer + CUDA venv on 5080, vault mount/sync, Qdrant reachable from 5080 (today loopback-only on prime), and free VRAM after Comfy.

## Homelab + Forgejo CPU

Git for **code**: `https://git.vectorweight.com` (Forgejo). **Not** GitHub.com for day-to-day. Cabal/agent never pushes GitHub as a bot. Operator mirrors to GitHub when *they* are ready.

Working repos (full history — every branch and tag, **not** a shallow `main`):

| Forgejo | GitHub origin (read/mirror only) | Local tree |
|---|---|---|
| `tzervas/CogSynDelta` | `github.com/tzervas/CogSynDelta` | `/home/kang/code/personal/tzervas/CogSynDelta` |
| `tzervas/memory-gate` | `github.com/tzervas/memory-gate` | `/home/kang/code/personal/tzervas/python-ai/memory-gate` |
| `tzervas/memory-gate-rs` | `github.com/tzervas/memory-gate-rs` | `/home/kang/code/personal/tzervas/memory-gate-rs` |

`memory-gate-rs` on GitHub was often cloned **main-only**. Always:

```bash
git remote set-branches origin '*'
git fetch origin --prune --tags
# never: git clone --depth 1   or   git fetch origin main
```

Push the **deep tree** (all heads + tags + local-only branches):

```bash
AKULA=/home/kang/code/personal/tzervas/akula-ai-platform
secret exec TOKEN=git/cabal-forgejo-admin -- \
  python3 "$AKULA/scripts/cabal-forgejo-push-full-tree" \
    --repo /home/kang/code/personal/tzervas/CogSynDelta \
    --dest tzervas/CogSynDelta
# same for python-ai/memory-gate → tzervas/memory-gate
# same for memory-gate-rs → tzervas/memory-gate-rs
```

Day-to-day after that: `git remote add forgejo https://git.vectorweight.com/tzervas/CogSynDelta.git` and `git push -u forgejo HEAD`. PRs/issues/Actions on git.vectorweight.com. `secret exec TOKEN=git/cabal-forgejo-agent` — never argv.

```yaml
# .github/workflows stay canonical (Forgejo runs them). Never add .forgejo/workflows/.
runs-on: [self-hosted, linux, x64, podman, compute-cpu, host-homelab]
# Never: gpu, 5080, ubuntu-latest, bare self-hosted
```

No required reviews on these tzervas repos. Self-review COMMENT + honestly green CI. Never APPROVE as Cabal.

**Branch archaeology is part of the job.** Every head and every open PR on all three remotes gets KEEP / CHERRY-PICK / ARCHIVE / CLOSE. Protocol: `docs/PROGRAM-GOALS.md` Phase 0. Starting map: `docs/GROK-STRATA.md` (CSD only — must re-validate). Do not merge Jules/`develop` sediment into `main`.

Forgejo LFS is fine for small fixtures. **Weights and GGUFs do not go to Forgejo** — Hugging Face private (below).

Homelab runner unit: `forgejo-runner-cpu` on **192.168.1.170** (`homelab-cpu`, always-up). Prime may also have `akula-prime-cpu`. Both must stay `compute-cpu`, never `gpu`.

## Hugging Face (models only, private)

Code → Forgejo. **Checkpoints, GGUFs, datasets of weights → Hugging Face private repos.** Do not git-push 14B files to git.vectorweight.com.

Private family already exists (do not create a second umbrella):

- Model: [`tzervas/cogsyndelta`](https://huggingface.co/tzervas/cogsyndelta)
- Dataset: [`tzervas/cogsyndelta-eval`](https://huggingface.co/datasets/tzervas/cogsyndelta-eval)

```bash
# token already on this host (huggingface-cli whoami → tzervas). Never argv.
hf upload tzervas/cogsyndelta path/to/ckpt.safetensors --repo-type model
hf upload tzervas/cogsyndelta-eval path/to/eval.jsonl --repo-type dataset
# card: templates/hf-model-card.md → README.md on the Hub (YAML frontmatter IS metadata)
```

### Model card = `README.md`

YAML frontmatter **is** the Hub metadata (search, widgets, license badge). Body is the human card.

```yaml
---
library_name: pytorch          # or transformers, diffusers, …
pipeline_tag: other            # or text-generation, feature-extraction, …
license: mit
license_link: https://github.com/tzervas/CogSynDelta/blob/main/LICENSE
language: [en]
tags:
  - cogsyndelta
  - moe
  - pytorch
  - private
base_model: []                 # Hub ids if fine-tune/quant of another repo
datasets: []                   # Hub dataset ids only
metrics: []
private: true
---
```

Required in the Markdown body (honest numbers only from `STATUS.md` or a run you logged):

1. **Summary** — one paragraph, Python PoC vs intended architecture.
2. **Use** — exact `uv run` / `from_pretrained` snippet that works.
3. **Training data** — what, license, no scraped secrets.
4. **Metrics** — table with seed, steps, device (`cuda:5080` / `cpu`). Link `STATUS.md`.
5. **Limitations** — not 10× compression unless measured; not VL-JEPA unless tests say so.
6. **Hardware** — 5080 exclusive-seq / 3090 one GGUF.
7. **Citation** + Forgejo code URL.

`model-index` (YAML) for structured evals so the Hub **model-index / Papers with Code** view fills:

```yaml
model-index:
  - name: csd-poc-latentvae
    results:
      - task:
          type: other
          name: LatentVAE reconstruction
        dataset:
          type: synthetic
          name: planted-batch
        metrics:
          - type: loss
            value: 0.00895
            name: routed recon (20 steps, seed 42)
        source:
          name: STATUS.md PoC-3
          url: https://git.vectorweight.com/tzervas/CogSynDelta
```

Hub **Settings** (UI or `hf repos settings`):

| Field | Value |
|---|---|
| Visibility | **Private** until operator says public |
| Gated | optional `auto`/`manual` if sharing inside org |
| License | match YAML `license` |
| Discussions | on if you want issues on HF; code issues stay Forgejo |

**Datasets** (`--type dataset`): separate repo, `dataset` card YAML (`task_categories`, `size_categories`, `pretty_name`). Do not dump training dumps into the model repo.

**Spaces**: not required. If you add a demo, it is a third repo (`--type space`).

Token: `HF_TOKEN` via `secret exec`, never in git, never in LocalAI env. `huggingface-cli whoami` before upload.

After upload, check the Hub **model page** renders: pipeline widget (if tag is standard), files tab, card sections, license, tags. Fix YAML if the widget is wrong (`pipeline_tag`).

## Quick decision table

| Need | Where | How |
|---|---|---|
| Codex code assist | 3090 LocalAI | `local/code`, leave it loaded |
| CSD `train`/`route` CUDA | 5080 | `with-gpu-5080 --mode exclusive-seq` |
| CSD vault reindex | 5080 | `./scripts/csd-kb-index` |
| Keyword RAG | CPU | `:8091` / `:8092`, never pause 3090 |
| pytest / ruff / uv | homelab or prime CPU | Forgejo Actions `compute-cpu` |
| Comfy still running | 5080 busy | enqueue timeshare; do not kill unless exclusive-seq |
| Save weights | HF `tzervas/cogsyndelta` private | card YAML + LFS; code stays Forgejo |
| Eval dumps | HF `tzervas/cogsyndelta-eval` | dataset card, not the model repo |
| Full git history | Forgejo `tzervas/{CogSynDelta,memory-gate,memory-gate-rs}` | `cabal-forgejo-push-full-tree`, never shallow |
