# CogSynDelta scale ladder

Not `STATUS.md`. Contract for **when** a size may train, eval, and (if
minted) land on private Hub. Hosted Grok does not implement. Autodev
on `feat/agent-harness` may run **tiny** CPU PoC and **small** 5080
CUDA recipes already in `src/cogsyndelta/poc/`. Do **not** start
medium/larger from an autoloop. `G-TRAIN` stays blocked until
`G-LIFE`.

Identity: **autodev**. Never `main` / `staging` / `develop` / `dev`.
Never dual 14B. Never pause 3090 LocalAI. Never unmask Comfy. Never
GitHub bot push. Never `0.0.0.0` WAN. Never mix 384-d. Never write
operator or model vaults.

Python-first: memory-gate Phase 1, then CSD PoC regions/router. 5080
CUDA is exclusive-seq on **gpu5080 `192.168.1.251`**. 1080 Ti guest
(`192.168.1.243`) is retrieve-index-light only — not a train/eval
card, not a 5080 substitute. Do not ham PoC CUDA on Pascal.

Do **not** claim VL-JEPA, mHC, quantum, or 10× compression. Measured
PoC numbers live only in `STATUS.md`. This file does not mint a
Grafana sat.

## Ladder

| N | Size | Device | Status now | Private HF (when `hf/autodev`) |
|---|---|---|---|---|
| 0 | `tiny` | CPU (prime / homelab) | Historical pass in `STATUS.md` 2026-08-19. **Not** a Grafana sat | `tzervas/cogsyndelta-tiny` |
| 1 | `small` | CUDA RTX 5080 exclusive-seq | Historical pass in `STATUS.md` CUDA table. **Not** a Grafana sat | `tzervas/cogsyndelta-small` |
| 2 | `medium` | CUDA 5080 exclusive-seq, larger when measured | **not-yet.** Blocked on `G-LIFE` / Phase 3 | `tzervas/cogsyndelta-medium` |

Receipt file: [`benchmark_results/scale_ladder.json`](../../benchmark_results/scale_ladder.json).
`green: true` is written **only** after a run this tree records (CLI
JSON + pytest). Historical `STATUS.md` rows are `status:
historical-pass` with `green: false` so Grafana cannot page a fake sat.

Hub publish is **private** `tzervas/cogsyndelta-<size>` and needs CSD
vault `hf/autodev`. Today that name is **missing** (`gap=mint HF`).
Until mint: no upload. Never copy `gpu/huggingface-token`.

## Rung 0 — tiny (CPU / PoC)

**Why:** prove one LatentVAE region + two-region softmax gate on CPU
without a GPU grant. This is the PoC already in-tree, not a 14B.

### Train recipe

```bash
# homelab CPU or prime. No CUDA. No 1080 Ti. No 3090 pause.
uv run python -m cogsyndelta.poc.cli train --device cpu --steps 20 \
  --checkpoint /tmp/csd-tiny-vae.pt
uv run python -m cogsyndelta.poc.cli train-route --device cpu --steps 20
```

Fixed knobs (match `STATUS.md` PoC-3 CPU): seed **42**, train-route
batch **8**, stream_dim **16** (RouteTrainConfig default). Synthetic
batch, not a Hub corpus.

Smoke: `./scripts/poc_smoke.sh cpu`.

### Eval

| Gate | How |
|---|---|
| LatentVAE last loss < first (≥20 steps) | `train` CLI JSON `improved` |
| Routed recon last < first (≥20 steps) | `train-route` CLI |
| Switch aux finite, toward 1.0 uniform | CLI aux field |
| Load not `{1.0, 0.0}` on batch 8 | CLI load split |
| Pytest | `uv run pytest tests/test_poc_train.py tests/test_poc_compress.py tests/test_poc_registry.py tests/test_poc_route_train.py tests/test_poc_contracts.py -q` |

Do **not** write `green: true` unless that receipt is saved next to
`scale_ladder.json`. Do not copy 2026-08-19 `STATUS.md` numbers into
the Grafana gauge.

### HF

`tzervas/cogsyndelta-tiny` (private model) **when** `hf/autodev`
exists. Card from `templates/hf-model-card.md`. No GGUF of LocalAI
aliases. No upload while `gap=mint HF`.

## Rung 1 — small (CUDA on 5080)

**Why:** same PoC on Blackwell sm_120. Exclusive-seq. 3090 stays
`local/code`. Comfy stays masked. 1080 Ti stays RAG/index.

### Train recipe

```bash
# gpu5080 LAN 192.168.1.251. Never 0.0.0.0. Never dual 14B.
AKULA=/home/kang/code/personal/tzervas/akula-ai-platform
"$AKULA/scripts/with-gpu-5080" --mode exclusive-seq -- \
  uv run python -m cogsyndelta.poc.cli train --device cuda --steps 20 \
  --checkpoint /tmp/csd-small-vae.pt
"$AKULA/scripts/with-gpu-5080" --mode exclusive-seq -- \
  uv run python -m cogsyndelta.poc.cli train-route --device cuda --steps 20
```

Same seed 42. If `gpu5080.lock` is held, enqueue — do not steal Comfy
by unmasking, do not pause LocalAI.

Smoke: `./scripts/poc_smoke.sh cuda` **on gpu5080 only**.

### Eval

Same gates as tiny, plus `tests/test_poc_cuda.py` (skips when CUDA is
absent — a skip is **not** small-rung green). Occupied VRAM must be
reported; do not kill ollama/LocalAI to make it fit.

### HF

`tzervas/cogsyndelta-small` (private) when minted. Same card rules.

## Rung 2 — medium (larger **when measured**)

**Not now.** Do not train from `/csd-autodev-loop`. Unblock after
memory-gate `G-LIFE` / Phase 3 (`docs/PROGRAM-GOALS.md`). Then one
architecture change per PR.

Intended shape (horizon, not a claim):

- More than two regions still one mind (code / retrieve / compress +
  gate). Catalog: [CSD-REGION-DATASETS.md](CSD-REGION-DATASETS.md).
- Real tagged mix, not only synthetic tokens. Pin `dataset_id` +
  `config` + `revision`. Eval stays out of the train mix.
- Train on 5080 exclusive-seq. Measure steps, seed, device, recon,
  aux, load, VRAM. Write `STATUS.md` **and** a receipt. Only then
  `green: true` for `n=2`.
- Never 1080 Ti for this rung. Never dual 14B. Never VL-JEPA / quantum
  / 10× unless a test you ran supports it.

Example **after** unblock (do not run today):

```bash
AKULA=/home/kang/code/personal/tzervas/akula-ai-platform
"$AKULA/scripts/with-gpu-5080" --mode exclusive-seq -- \
  uv run python -m cogsyndelta.poc.cli train-route --device cuda \
  --steps 200 --batch-size 32
```

Longer steps / larger batch are **not** green until measured. Foundation
`tzervas/cogsyndelta` (no size suffix) is a later overall pretrain, not
rung 2.

### Eval (when a run exists)

| Gate | How |
|---|---|
| Routed recon last < first on the logged recipe | CLI JSON |
| Aux finite; load not collapsed on batch ≥ 8 | CLI + pytest |
| Region-tagged top-1 on a labeled batch of 8+ | new test, not README |
| CUDA VRAM + seed + steps in receipt | `scale_ladder.json` + `STATUS.md` |

### HF

`tzervas/cogsyndelta-medium` (private) when minted **and** green.

## Hugging Face gap (`hf/autodev`)

Checked 2026-08-31 CSD vault `/akula-data/cabal/csd-vault` `secret ls`:
`csd/apply-token`, `git/autodev`, `gpu/localai-api-key` only. `hf/`
exists **empty**. Hugging Face has **no** API to mint a fine-grained
write token. GET `/api/goals` reports `hf.gap = "mint HF"`.

Operator mint (do **not** copy `gpu/huggingface-token`):

```bash
# huggingface.co/settings/tokens → fine-grained write on
# tzervas/cogsyndelta* only (tiny / small / medium / eval / data)
printf '%s' 'hf_…' | SECRET_VAULT=/akula-data/cabal/csd-vault \
  SOPS_AGE_KEY_FILE=$SECRET_VAULT/age.txt secret set hf/autodev
```

Until that exists: **no** Hub publish. Public sets stay at canonical
URLs. Goals API `hf.publish` stays blocked.

Upload after mint **and** a green receipt:

```bash
SECRET_VAULT=/akula-data/cabal/csd-vault \
  SOPS_AGE_KEY_FILE=$SECRET_VAULT/age.txt \
  secret exec TOKEN=hf/autodev -- \
  hf upload tzervas/cogsyndelta-<size> path/to/ckpt.safetensors --repo-type model
```

Private only. Never Forgejo LFS for weights. Never GitHub.

## Grafana sat (one rule, no fake page)

Contact: `maintainers-email` → `maintainers@vectorweight.com`
([contact-points.yaml](../../deploy/grafana/provisioning/alerting/contact-points.yaml)).

Rule `CSDScaleLadderRungGreen` watches the metric **this tree
creates**:

```promql
csd_scale_ladder_rung_green == 1
```

Source: `benchmark_results/scale_ladder.json` → lab console `GET /metrics`
(`scripts/csd-lab-console`). `green: false` for every rung today, so
the query is empty / 0. `noDataState: OK` — **does not fire**. Do not
provision a dummy series of 1. Do not email a historical PoC.

When a measured receipt sets `green: true` for rung N, the gauge is 1
and the rule may page maintainers. That is the only sat.

Scrape of `:9118/metrics` is **not** added to
`deploy/o11y/vm-scrape.yml` (no new job until operator installs). Until
then the rule stays no-data OK. Do not sudo-restart Grafana from
autodev to “prove” mail.

## Goals API

`GET /api/goals` (lab console, LAN bind, never `0.0.0.0`) includes
`scale_ladder` (this receipt) and `hf` (live CSD-vault presence, not
the operator vault). UI Goals/Todos tab renders it. Not Open WebUI.

## Hard no

- Do not merge skip-theatre. Required checks must have **run**.
- Do not train medium from this doc alone.
- Do not promote 1080 Ti as a train device. Guest `nvidia-smi` is a
  RAG/index signal only.
- Do not dual-load 14B GGUFs. Do not pause `akula-localai`.
