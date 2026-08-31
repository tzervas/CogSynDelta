# CogSynDelta scale ladder

Not `STATUS.md`. Contract for **when** a rung may train, eval, and (if
minted) land on private Hub. Hosted Grok does not implement. Autodev
on `feat/agent-harness` may run **R0** CPU / 5080-smoke **one
specialist** recipes already in `src/cogsyndelta/poc/`. Do **not**
start R3+ from an autoloop. `G-TRAIN` stays blocked until `G-LIFE`.

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

Regions are specialized **submodels of one mind**. They **must**
pretrain on their own datasets **before** whole-model train. Catalog:
[CSD-REGION-DATASETS.md](CSD-REGION-DATASETS.md). Map:
[CSD-BRAIN-REGIONS.md](CSD-BRAIN-REGIONS.md).

Live PoC (`STATUS.md`): two regions (`residual_mlp`, `stream_vae`) +
softmax router. Not mHC.

## Ladder

Curriculum first, param count later. R0 is **one specialist**, not a
tiny whole mind.

| N | Size | Stage | Device | Status now | Private HF (when `hf/autodev`) |
|---|---|---|---|---|---|
| 0 | `region_pretrain` | tiny region-pretrain (one specialist) | CPU / PoC or 5080 smoke | Historical PoC-1 `train` in `STATUS.md` 2026-08-19. **Not** a Grafana sat | `tzervas/cogsyndelta-region-<name>-tiny` |
| 1 | `router` | router / interconnect: frozen regions, then joint | CPU or 5080 smoke | Historical PoC-3 `train-route` in `STATUS.md`. **Not** a Grafana sat. Not mHC | `tzervas/cogsyndelta-region-route-tiny` |
| 2 | `tiny_mind` | assembled tiny whole-mind | CPU or 5080 smoke | **not-yet.** Needs R0+R1 green receipts | `tzervas/cogsyndelta-tiny` |
| 3 | `small` | larger param counts | CUDA 5080 exclusive-seq | **not-yet.** Blocked on `G-LIFE` / Phase 3 | `tzervas/cogsyndelta-small` |
| 4 | `medium` | still larger, then foundation | CUDA 5080 exclusive-seq | **not-yet.** After R3 measured | `tzervas/cogsyndelta-medium` then `tzervas/cogsyndelta` |

Receipt file: [`benchmark_results/scale_ladder.json`](../../benchmark_results/scale_ladder.json).
`green: true` is written **only** after a run this tree records (CLI
JSON + pytest **or** a new `STATUS.md` row from that run). Historical
`STATUS.md` rows are `status: historical-pass` with `green: false` so
Grafana cannot page a fake sat.

Hub publish is **private**. Region rungs use
`tzervas/cogsyndelta-region-<name>-<size>` **then** assembled
`tzervas/cogsyndelta-<size>`. Needs CSD vault `hf/autodev`. Today that
name is **missing** (`gap=mint HF`). Until mint: no upload. Never copy
`gpu/huggingface-token`.

## Rung 0 — tiny region-pretrain (one specialist)

**Why:** one cognitive region must learn its job **alone** before a
gate can route to it. PoC analog is LatentVAE (`stream_vae`) via
`poc.cli train`. Named code/retrieve/compress/residual experts wait
for Phase 3 public splits. Not a 14B. Not `train-route`.

CPU/PoC on prime or homelab. 5080 exclusive-seq smoke is allowed.
Never 1080 Ti. Never pause 3090.

### Train recipe

```bash
# homelab CPU or prime. No CUDA. No 1080 Ti. No 3090 pause.
uv run python -m cogsyndelta.poc.cli train --device cpu --steps 20 \
  --checkpoint /tmp/csd-r0-stream_vae.pt

# optional 5080 smoke (exclusive-seq). Never dual 14B.
AKULA=/home/kang/code/personal/tzervas/akula-ai-platform
"$AKULA/scripts/with-gpu-5080" --mode exclusive-seq -- \
  uv run python -m cogsyndelta.poc.cli train --device cuda --steps 20 \
  --checkpoint /tmp/csd-r0-stream_vae.pt
```

Fixed knobs (match `STATUS.md` PoC-1): seed **42**. Synthetic batch
today, not a Hub corpus. Later (after `G-LIFE`): one public
**pretrain** split from
[CSD-REGION-DATASETS.md](CSD-REGION-DATASETS.md) — e.g. SNLI 10k for
`compress` / `stream_vae`, WikiText-2 for `residual_mlp`. Do not mix
`common`. Do not ingest eval-only rows.

Smoke: `./scripts/poc_smoke.sh cpu` covers `train` plus later CLIs;
R0 green requires the **train** receipt, not `train-route`.

### Eval

| Gate | How |
|---|---|
| LatentVAE last loss < first (≥20 steps) | `train` CLI JSON `improved` |
| Pytest (one specialist) | `uv run pytest tests/test_poc_train.py tests/test_poc_compress.py tests/test_poc_contracts.py -q` |
| 5080 smoke (optional) | same on CUDA; `tests/test_poc_cuda.py` skip is **not** R0 green for CUDA |

Do **not** write `green: true` unless that receipt is saved next to
`scale_ladder.json` **or** a new `STATUS.md` row is logged from the
same run. Do not copy 2026-08-19 `STATUS.md` numbers into the Grafana
gauge.

### HF

`tzervas/cogsyndelta-region-<name>-tiny` (private model) **when**
`hf/autodev` exists. PoC names: `stream_vae`, `residual_mlp`. Later:
`code`, `retrieve`, `compress`, `residual`. Card from
`templates/hf-model-card.md`. Stage = region-pretrain / bedrock
specialist, **not** foundation. No GGUF of LocalAI aliases. No upload
while `gap=mint HF`. Whole-mind `tzervas/cogsyndelta-tiny` is **R2**,
not this rung.

## Rung 1 — router / interconnect (frozen, then joint)

**Why:** after specialists exist, train the softmax gate so regions
activate **selectively**. Frozen regions first (`--gate-only`), then
joint. Switch aux `N * sum(f_i * P_i)` keeps load off `{1.0, 0.0}`.
Interconnect / mHC is **later**, not this rung. Idle regions must not
run `activate()`.

Requires R0 checkpoints (or PoC default two-region mind on synthetic).
CPU or 5080 smoke. Never 1080 Ti.

### Train recipe

```bash
# frozen regions, train the gate only
uv run python -m cogsyndelta.poc.cli train-route --device cpu \
  --steps 20 --gate-only

# then joint (unfreeze region params)
uv run python -m cogsyndelta.poc.cli train-route --device cpu --steps 20

# 5080 smoke, exclusive-seq
AKULA=/home/kang/code/personal/tzervas/akula-ai-platform
"$AKULA/scripts/with-gpu-5080" --mode exclusive-seq -- \
  uv run python -m cogsyndelta.poc.cli train-route --device cuda \
  --steps 20 --gate-only
"$AKULA/scripts/with-gpu-5080" --mode exclusive-seq -- \
  uv run python -m cogsyndelta.poc.cli train-route --device cuda --steps 20
```

Fixed knobs (match `STATUS.md` PoC-3 CPU): seed **42**, batch **8**,
stream_dim **16** (RouteTrainConfig default). Synthetic batch today.
Later: `region-route` v0 tagged mix from **already-pretrained**
catalogs. Drop NC / eval-only. `top_k=1` until load is measured
non-collapsed on a labeled batch of 8+.

If `gpu5080.lock` is held, enqueue — do not steal Comfy by unmasking,
do not pause LocalAI.

### Eval

| Gate | How |
|---|---|
| Routed recon last < first (≥20 steps) | `train-route` CLI |
| Switch aux finite, toward 1.0 uniform | CLI aux field |
| Load not `{1.0, 0.0}` on batch 8 | CLI load split |
| Frozen-then-joint | `--gate-only` run logged **before** joint |
| Pytest | `uv run pytest tests/test_poc_route_train.py tests/test_poc_registry.py tests/test_poc_contracts.py -q` |
| CUDA | `tests/test_poc_cuda.py` skip is **not** R1 CUDA green |

Same `green: true` rule as R0. Historical PoC-3 is **not** a sat.

### HF

`tzervas/cogsyndelta-region-route-tiny` for the gate, plus the R0
region repos left frozen/copied. Not `tzervas/cogsyndelta-tiny` yet.

## Rung 2 — assembled tiny whole-mind

**Why:** only after R0 specialists and R1 gate exist, train the
**assembled** tiny mind (regions + gate) as one artifact. Bedrock
analog: no foundation `common`, no C4 Switch pretrain, no 14B.

**Not now** as a Hub publish. Do not train from `/csd-autodev-loop`
until R0+R1 have real receipts. PoC `train-route` without prior R0
isolation is **not** this rung.

### Train recipe

```bash
# after R0 checkpoints + R1 gate. CPU first.
uv run python -m cogsyndelta.poc.cli train-route --device cpu --steps 20

# 5080 smoke only if CUDA is required
AKULA=/home/kang/code/personal/tzervas/akula-ai-platform
"$AKULA/scripts/with-gpu-5080" --mode exclusive-seq -- \
  uv run python -m cogsyndelta.poc.cli train-route --device cuda --steps 20
```

Later (after `G-LIFE`): tagged mix val of each specialist; pin
`dataset_id` + `config` + `revision`. Eval stays out of the train mix.

### Eval

| Gate | How |
|---|---|
| Routed recon last < first on the logged recipe | CLI JSON |
| Aux finite; load not collapsed on batch ≥ 8 | CLI + pytest |
| Region-tagged top-1 on a labeled batch of 8+ | new test, not README |
| R0 + R1 receipts exist | `scale_ladder.json` n=0 and n=1 |

### HF

`tzervas/cogsyndelta-tiny` (private assembled) when minted **and**
green. Region repos stay at `tzervas/cogsyndelta-region-<name>-tiny`.
Foundation `tzervas/cogsyndelta` (no size suffix) is R3+/overall
pretrain, not this rung.

## Rung 3+ — larger param counts

**Not now.** Do not train from `/csd-autodev-loop`. Unblock after
memory-gate `G-LIFE` / Phase 3 (`docs/PROGRAM-GOALS.md`). Then one
architecture change per PR.

Intended shape (horizon, not a claim):

- Same curriculum, more parameters / more rows from the catalog
  **scale** splits (WikiText-103, full SQuAD train, AllNLI 100k,
  license-filtered Stack-smol). `common` only for overall pretrain.
- More than two regions still one mind (code / retrieve / compress /
  residual + gate).
- Train on 5080 exclusive-seq. Measure steps, seed, device, recon,
  aux, load, VRAM. Write `STATUS.md` **and** a receipt. Only then
  `green: true` for `n=3` (`small`) or `n=4` (`medium`).
- Never 1080 Ti for this rung. Never dual 14B. Never VL-JEPA / quantum
  / 10× unless a test you ran supports it.

Example **after** unblock (do not run today):

```bash
AKULA=/home/kang/code/personal/tzervas/akula-ai-platform
"$AKULA/scripts/with-gpu-5080" --mode exclusive-seq -- \
  uv run python -m cogsyndelta.poc.cli train-route --device cuda \
  --steps 200 --batch-size 32
```

Longer steps / larger batch are **not** green until measured.

### Eval (when a run exists)

| Gate | How |
|---|---|
| Routed recon last < first on the logged recipe | CLI JSON |
| Aux finite; load not collapsed on batch ≥ 8 | CLI + pytest |
| Region-tagged top-1 on a labeled batch of 8+ | new test, not README |
| CUDA VRAM + seed + steps in receipt | `scale_ladder.json` + `STATUS.md` |

### HF

`tzervas/cogsyndelta-small` then `tzervas/cogsyndelta-medium` (private)
when minted **and** green. Region scale copies:
`tzervas/cogsyndelta-region-<name>-small` (and `-medium`) **before**
the assembled size repo. Foundation `tzervas/cogsyndelta` only after
overall pretrain.

## Hugging Face gap (`hf/autodev`)

Checked 2026-08-31 CSD vault `/akula-data/cabal/csd-vault` `secret ls`:
`csd/apply-token`, `git/autodev`, `gpu/localai-api-key` only. `hf/`
exists **empty**. Hugging Face has **no** API to mint a fine-grained
write token. GET `/api/goals` reports `hf.gap = "mint HF"`.

Operator mint (do **not** copy `gpu/huggingface-token`):

```bash
# huggingface.co/settings/tokens → fine-grained write on
# tzervas/cogsyndelta* only (region-* / tiny / small / medium / eval / data)
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
  hf upload tzervas/cogsyndelta-region-<name>-<size> path/to/ckpt.safetensors \
  --repo-type model
# assembled, after region repos:
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

`maintainers-email` may page **only** when a rung has a **real**
`STATUS.md` row **or** pytest/CUDA receipt written this tree, and that
receipt sets `green: true`. Historical 2026-08-19 tables are not a sat.

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
- Do not skip R0. Do not start at R3.
- Do not train R3+ from this doc alone.
- Do not promote 1080 Ti as a train device. Guest `nvidia-smi` is a
  RAG/index signal only.
- Do not dual-load 14B GGUFs. Do not pause `akula-localai`.
