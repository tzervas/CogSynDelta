# CogSynDelta Status (PoC-1 + PoC-2 + PoC-3)

Single source of truth for **targets vs measured vs gaps**.
Generated from local container CI 2026-08-19.

## Architecture framing (authoritative)

CogSynDelta is **MoE-adjacent, not multi-agent**:

| MoE concept | CSD analog |
|---|---|
| Expert | Cognitive **region** (module of one mind) |
| Gating / router | Softmax top-k + Switch aux LB (PoC-3). Interconnect / mHC **later** |
| Shared residual stream | Shared latent + compressed memory substrate |
| — | Explore → cull → meta-optimize control loop |

PoC-1 proves the **substrate**: one region (LatentVAE) + measured memory compression + DeviceContext.
PoC-2 proves **two regions + a real MoE gate**. It is not mHC.
PoC-3 **trains** that gate (and region params) so routed reconstruction falls
with a real Switch-Transformer aux term (`N * sum(f_i * P_i)`). Still not mHC.

## PoC-1 scope

| In scope | Out of scope |
|---|---|
| DeviceContext (`cpu` / `cuda` / `auto`) | Full JEPA / V-JEPA 2 |
| One LatentVAE region (loss decreases) | Swarm of complete agents (SWE/QA/security) |
| Measured compression (bytes round-trip) | Quantum backends |
| Home-lab CPU CI | RTX 5080 as CI requirement |

## PoC-1 claims (measured 2026-08-19, CPU)

| Claim | Target | Measured | Status |
|---|---|---|---|
| LatentVAE train loss decreases | last < first over ≥20 steps | 68.96 → 66.48 (20 steps) | **pass** |
| Basis residual fidelity | ≥0.99 cosine | **1.000** | **pass** |
| Basis residual ratio (stored bytes) | report true ratio | **~1.3–1.6×** (not 10×) | **pass** (honest) |
| Calibrated 8-bit quant fidelity | ≥0.90 cosine | **0.99999** | **pass** |
| Device CPU path | works without CUDA | 6/6 tests green | **pass** |
| Device CUDA path | works when CUDA present | RTX 5080 / torch 2.9.1+cu128: see CUDA table below | **pass** |

## PoC-2 scope

| In scope | Out of scope |
|---|---|
| `CognitiveRegion.activate(stream) → [B, D]` | Changing LatentVAE train `forward` tuple |
| `RegionRegistry` register / get / duplicate reject | AgentFleet / marketplace |
| ResidualMLP + stream-dim LatentVAE | More than two regions |
| Softmax top-k mix (MoE gate) | mHC / moderated interconnect |
| Optional 8-bit compact of routed stream | Aux load-balance loss / training the gate |

## PoC-2 claims (measured 2026-08-19, CPU)

| Claim | Target | Measured | Status |
|---|---|---|---|
| Both regions satisfy CognitiveRegion | isinstance + activate shape | ResidualMLP + LatentVAE | **pass** |
| LatentVAE.forward still ELBO tuple | (recon, mu, logvar) | train path unchanged | **pass** |
| Softmax weights sum to 1 | per-token sum == 1 | **1.000** | **pass** |
| Top-k=1 load is a real split | both regions receive tokens | **0.375 / 0.625** (batch 8) | **pass** |
| Routed-stream 8-bit fidelity | ≥0.90 cosine | **0.999992** | **pass** |
| Routed-stream 8-bit ratio | report true ratio | **0.735×** (2785 vs 2048 B; tiny batch expands) | **pass** (honest) |
| PoC tests | all green | **15/15** | **pass** |

## PoC-3 scope

| In scope | Out of scope |
|---|---|
| Train softmax gate + optional region params | mHC / moderated interconnect |
| Switch aux LB `N * sum(f_i * P_i)` | JEPA / V-JEPA 2 |
| Routed reconstruction via `activate` | Changing LatentVAE train `forward` tuple |
| Load not `{1.0, 0.0}` on a batch of 8 | AgentFleet / more than two regions |

## PoC-3 claims (measured 2026-08-19, CPU)

CLI: `python -m cogsyndelta.poc.cli train-route --device cpu --steps 20`
(seed 42, batch 8, stream_dim 16). Same synthetic batch each step.

| Claim | Target | Measured | Status |
|---|---|---|---|
| Routed recon loss decreases | last < first over ≥20 steps | **0.08774 → 0.00895** (20 steps) | **pass** |
| Aux LB is finite | finite `N * sum(f_i * P_i)` | **1.1906 → 1.0114** (toward 1.0 uniform) | **pass** |
| Load not collapsed | not `{1.0, 0.0}` on batch 8 | **0.625 / 0.375** (mlp / vae) | **pass** |
| LatentVAE.forward still ELBO tuple | `(recon, mu, logvar)` | train-route uses `activate` only | **pass** |
| CLI 10-step smoke | last < first | **0.08774 → 0.01192** | **pass** |
| PoC-3 tests | all green | **6/6** CPU + CUDA skip-or-run | **pass** |

## Device CUDA (measured 2026-08-19, RTX 5080)

Hardware: NVIDIA GeForce RTX 5080 (16303 MiB, sm_120), driver 610.57.04,
torch 2.9.1+cu128, CUDA 12.8. CLI:
`python -m cogsyndelta.poc.cli {train,compress,route,train-route} --device cuda`.
`tests/test_poc_cuda.py` skips when `torch.cuda.is_available()` is false.
PoC-3 CUDA used ~unchanged 766 MiB occupied; ollama was not killed.

| Claim | Target | Measured | Status |
|---|---|---|---|
| LatentVAE train loss decreases | last < first over ≥20 steps | 68.85 → 66.67 (20 steps, seed 42) | **pass** |
| Basis residual fidelity | ≥0.99 cosine | **1.000** | **pass** |
| Basis residual ratio (stored bytes) | report true ratio | **1.617×** (20261 vs 32768 B) | **pass** (honest) |
| Calibrated 8-bit quant fidelity | ≥0.90 cosine | **0.999992** | **pass** |
| Calibrated 8-bit quant ratio | report true ratio | **2.673×** (12257 vs 32768 B) | **pass** (honest) |
| Softmax weights sum to 1 | per-token sum == 1 | **1.000** | **pass** |
| Top-k=1 load is a real split | both regions receive tokens | **0.375 / 0.625** (batch 8, stream_dim 32, seed 42) | **pass** |
| Routed-stream 8-bit fidelity | ≥0.90 cosine | **0.999995** | **pass** |
| Routed-stream 8-bit ratio | report true ratio | **0.427×** (2401 vs 1024 B; tiny batch expands) | **pass** (honest) |
| Device CUDA path | kernels run when CUDA present | train/compress/route CLI + tests | **pass** |
| PoC-3 routed recon decreases | last < first over ≥20 steps | **0.07897 → 0.01995** (20 steps, seed 42) | **pass** |
| PoC-3 aux LB finite | finite `N * sum(f_i * P_i)` | **1.1263 → 1.0056** | **pass** |
| PoC-3 load not collapsed | not `{1.0, 0.0}` on batch 8 | **0.375 / 0.625** (mlp / vae) | **pass** |
| PoC-3 CLI 10-step smoke | last < first | **0.07897 → 0.01907** | **pass** |

No 10× compression claim. Ratios are original_bytes / stored_bytes.
PoC-3 losses are routed MSE on a fixed synthetic batch, not ELBO.

## Historical (pre-PoC DenseDifferential, do not market)

| Ratio | Measured fidelity |
|---|---|
| 2× | 0.670 |
| 4× | 0.462 |
| 16× | 0.228 |
