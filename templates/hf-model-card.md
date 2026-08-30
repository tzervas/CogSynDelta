---
# Hugging Face Hub reads this YAML as repo metadata (search, license badge,
# pipeline widget, model-index). Keep it honest. Empty lists are better than lies.
language:
  - en
license: mit
license_link: https://git.vectorweight.com/tzervas/CogSynDelta/src/branch/main/LICENSE
library_name: pytorch
pipeline_tag: other
tags:
  - cogsyndelta
  - moe
  - pytorch
  - latent-vae
  - private
base_model: []
datasets: []
metrics: []
pretty_name: CogSynDelta (private)
# Fill model-index from a run you logged (STATUS.md or eval receipt). Never invent.
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
            name: routed recon (20 steps, seed 42, CPU)
        source:
          name: STATUS.md PoC-3
          url: https://git.vectorweight.com/tzervas/CogSynDelta
---

# {model_id}

Private CogSynDelta checkpoint family (`tzervas/cogsyndelta`).

**Code (deep git tree):** [git.vectorweight.com/tzervas/CogSynDelta](https://git.vectorweight.com/tzervas/CogSynDelta)
**Memory backend (Python):** [tzervas/memory-gate](https://git.vectorweight.com/tzervas/memory-gate)
**Eval dumps (dataset repo):** [tzervas/cogsyndelta-eval](https://huggingface.co/datasets/tzervas/cogsyndelta-eval)
**Measured claims:** `STATUS.md` in the code repo. This card must not exceed that file.

This is a **MoE-adjacent single brain** (cognitive regions + softmax router), not a
multi-agent swarm. Python-first. Rust (`memory-gate-rs`, Burn/CubeCL/Candle) is a
later rewrite.

## Model Details

- **Developed by:** Tyler Zervas / Average Joe's Labs
- **Model type:** PyTorch LatentVAE region + softmax top-k router (PoC-1..3)
- **Language(s):** n/a (latent / reconstruction); English docs
- **License:** MIT
- **Finetuned from:** none (trained from the CogSynDelta PoC, not a Hub base model)
- **Checkpoint files:** `*.safetensors` / `*.pt` via Git LFS. No GGUF of LocalAI aliases here.

## Uses

### Direct Use

```bash
# CUDA on gpu5080 only, exclusive-seq. Do not co-run Comfy.
AKULA=/home/kang/code/personal/tzervas/akula-ai-platform
"$AKULA/scripts/with-gpu-5080" --mode exclusive-seq -- \
  uv run python -m cogsyndelta.poc.cli train-route --device cuda --steps 20
```

```python
# CPU path (homelab / prime without pausing LocalAI)
# uv run python -m cogsyndelta.poc.cli train-route --device cpu --steps 20
```

### Downstream Use

Region weights as initialization for later CogSynDelta Python experiments.
Not a chat model. Not a drop-in `transformers` generator.

### Out-of-Scope

- VL-JEPA / V-JEPA 2, quantum backends, “10× compression,” agent fleets
- Serving as a LocalAI GGUF on the 3090 (that card runs **one** assist alias)
- Public demo / Hub widget unless `pipeline_tag` is a real transformers task

## How to Get Started with the Model

1. Clone **code** from Forgejo (full history): `git clone https://git.vectorweight.com/tzervas/CogSynDelta.git`
2. `hf download tzervas/cogsyndelta --repo-type model` (needs a token with read on this private repo)
3. Point the PoC CLI / `from_pretrained` helper at the downloaded `*.safetensors`

## Training Details

| Field | Value |
|---|---|
| Seed | 42 (PoC-3 STATUS) |
| Steps | 20 |
| Device | CPU measured; RTX 5080 exclusive-seq for CUDA jobs |
| Batch | 8 |
| Stream dim | 16 |
| Aux | Switch `N * sum(f_i * P_i)` |

Fill new rows from the run that produced **this** file. Do not copy STATUS numbers
onto a different seed or architecture.

### Training Data

Synthetic planted batches in-repo. No scraped user data. If a real dataset is
used, it lives in `tzervas/cogsyndelta-eval` with its own dataset card
(`task_categories`, `size_categories`, `pretty_name`, license).

### Preprocessing / Speeds

Report wall time, GPU, `nvidia-smi` peak MiB. 5080 ~16303 MiB exclusive.

## Evaluation

Paste STATUS-compatible numbers only:

| Claim | Target | Measured | Device |
|---|---|---|---|
| Routed recon last < first | 20 steps | 0.08774 → 0.00895 | CPU |
| Softmax weights sum | 1.0 | 1.000 | CPU |

`model-index` YAML above must match this table or the Hub “model-index” view
will lie. Update both together.

## Bias, Risks, Limitations

- Tiny PoC dims. Not a foundation model.
- Compression ratios in STATUS are **honest and small** (~1.3–1.6× basis residual),
  not 10×. Do not advertise otherwise.
- Private repo: do not assume Hub widgets, Papers-with-Code, or community evals.

## Hardware / Environmental

| Host | GPU | Role |
|---|---|---|
| akula-prime | RTX 3090 Ti | One LocalAI GGUF (`local/code`). Do not dual-load. |
| gpu5080 | RTX 5080 | Exclusive-seq train / index. Never `compute-cpu`. |
| homelab | CPU | Forgejo Actions, pytest, keyword RAG. |

## Citation

```
@software{cogsyndelta,
  author = {Zervas, Tyler},
  title = {CogSynDelta},
  year = {2026},
  url = {https://git.vectorweight.com/tzervas/CogSynDelta}
}
```

## Model Card Authors / Contact

Tyler Zervas. Code issues on Forgejo, not on Hugging Face Discussions unless
Settings → Discussions is enabled for a reason.

---

### Hub Settings checklist (UI or `hf repo settings`)

- Visibility: **Private** until the operator says public
- Gated: optional
- License: MIT (must match YAML)
- Discussions: off by default (code issues stay Forgejo)
- Files: LFS for `*.safetensors *.gguf *.pt *.bin *.onnx`
- Do not put training dumps in the model repo — use the dataset repo
- After upload: open the model page and confirm card sections, license badge,
  tags, Files, and (if `pipeline_tag` is standard) the widget. Fix YAML if the
  widget is wrong.
