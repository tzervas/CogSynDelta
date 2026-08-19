# CogSynDelta Status (PoC-1 + PoC-2)

Single source of truth for **targets vs measured vs gaps**.
Generated from local container CI 2026-08-19.

## Architecture framing (authoritative)

CogSynDelta is **MoE-adjacent, not multi-agent**:

| MoE concept | CSD analog |
|---|---|
| Expert | Cognitive **region** (module of one mind) |
| Gating / router | Softmax top-k (PoC-2). Interconnect / mHC **later** |
| Shared residual stream | Shared latent + compressed memory substrate |
| — | Explore → cull → meta-optimize control loop |

PoC-1 proves the **substrate**: one region (LatentVAE) + measured memory compression + DeviceContext.
PoC-2 proves **two regions + a real MoE gate**. It is not mHC.

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
| Device CUDA path | works when CUDA present | not run on desktop yet | unknown |

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

## Historical (pre-PoC DenseDifferential, do not market)

| Ratio | Measured fidelity |
|---|---|
| 2× | 0.670 |
| 4× | 0.462 |
| 16× | 0.228 |
