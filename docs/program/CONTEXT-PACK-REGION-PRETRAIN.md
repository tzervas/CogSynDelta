# Region-pretrain — one live PoC region on a public split

Worktree: `/home/kang/code/personal/tzervas/CogSynDelta` (`feat/agent-harness`).
**Never** touch `python-ai/memory-gate` (`kang-main-wip`).
**Never** write into `memory-gate-wt-p1-09`.

Regions are modules of **one mind** (MoE experts), not a swarm of agents.
`STATUS.md`: PoC-1 LatentVAE; PoC-2 ResidualMLP + LatentVAE + softmax top-k;
PoC-3 trains the gate. Not mHC. Not VL-JEPA. Not 10×.

This is **not** `G-TRAIN` (bedrock / foundation / 14B). One existing region
only. Autodev implements. Hosted Grok researched the split.

## Dataset (pinned)

| Field | Value |
|---|---|
| Region | **LatentVAE** (`stream_vae`; `cogsyndelta.poc.train.train_latent_vae`) |
| Why this region | Isolated train loop already exists. ResidualMLP only trains via `train-route` |
| Dataset | `Salesforce/wikitext` |
| Config | `wikitext-2-raw-v1` |
| Split | **train** only (36,718 lines). Never test / validation SGD |
| Revision | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| Size | ~7.75 MB download |
| License | Hub `cc-by-sa-3.0` + `gfdl` (Wikipedia-derived; card body also CC-BY-SA-4.0) |

Do **not** start SNLI-10k this tick (needs a pair encoder). Do **not** pull
WikiText-103, C4, The Stack, or a 14B.

Catalog: [CSD-REGION-DATASETS.md](CSD-REGION-DATASETS.md).

## This tick — one failing test only

Path **must** be `tests/test_poc_region_pretrain.py`.

- `pytest`. **No** `unittest`, **no** `pass`.
- One test: `test_latent_vae_pretrain_loss_decreases_on_wikitext2_train`.
- Import a helper that **does not exist yet** (name
  `cogsyndelta.poc.train.train_latent_vae_on_public_split`).
- CPU. seed **42**. steps **≥20**. batch_size **8**. `input_dim` matches PoC.
- Assert `result["last_loss"] < result["first_loss"]` and
  `result["split"] == "train"`.
- The test **must fail** because `train_latent_vae` still draws `torch.rand`
  synthetic batches (`src/cogsyndelta/poc/train.py`) and has no WikiText-2
  path.
- Include `assert`. Empty stubs are rejected.

Do **not** implement the trainer this tick.

## Next tick (not now)

Replace synthetic `torch.rand` with encoded WikiText-2 **train** lines as
`[B, D]`. Deterministic CPU encode (hash / bag-of-bytes). Do not pause 3090
LocalAI. Do not pull Qwen3. Tiny CUDA later on **5080** exclusive-seq only.

## Lab

| Work | Where |
|---|---|
| Implement this test | 3090 `local/code`. LocalAI stays loaded. Never dual 14B |
| Tiny CUDA later | 5080 `192.168.1.251` exclusive-seq. Comfy masked |
| RAG / index | 1080 Ti guest `192.168.1.243` **live** (`nvidia-smi` GTX 1080 Ti `GPU-4df3ba11-fd12-3550-bb97-ad00b0b00569`). No train / PoC CUDA on Pascal |

## Forbidden

- Whole-model 14B, `G-TRAIN` bedrock/foundation, dual 14B
- GitHub bot push; merge skip-theatre
- Memory-gate P1-09 / chroma P1-05 this steer
- Writing CSD `docs/program` into memory-gate worktrees
- Unmask Comfy; pause LocalAI; `0.0.0.0` WAN
