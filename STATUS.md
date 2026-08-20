# CogSynDelta Status (PoC-1)

Single source of truth for **targets vs measured vs gaps**.
Generated from local container CI 2026-08-19.

## Architecture framing (authoritative)

CogSynDelta is **MoE-adjacent, not multi-agent**:

| MoE concept | CSD analog |
|---|---|
| Expert | Cognitive **region** (module of one mind) |
| Gating / router | Interconnect / moderated hyper-connection (later) |
| Shared residual stream | Shared latent + compressed memory substrate |
| — | Explore → cull → meta-optimize control loop |

PoC-1 proves the **substrate**: one region (LatentVAE) + measured memory compression + DeviceContext.
Multi-region routing and the control loop come after the substrate has real numbers.

## PoC-1 scope

| In scope | Out of scope |
|---|---|
| DeviceContext (`cpu` / `cuda` / `auto`) | Full JEPA / V-JEPA 2 |
| One LatentVAE region (loss decreases) | Swarm of complete agents (SWE/QA/security) |
| Measured compression (bytes round-trip) | Quantum backends |
| Home-lab CPU CI | RTX 5080 as CI requirement |

## Claims table (measured 2026-08-19, CPU)

| Claim | Target | Measured | Status |
|---|---|---|---|
| LatentVAE train loss decreases | last < first over ≥20 steps | 68.96 → 66.48 (20 steps) | **pass** |
| Basis residual fidelity | ≥0.99 cosine | **1.000** | **pass** |
| Basis residual ratio (stored bytes) | report true ratio | **~1.3–1.6×** (not 10×) | **pass** (honest) |
| Calibrated 8-bit quant fidelity | ≥0.90 cosine | **0.99999** | **pass** |
| Device CPU path | works without CUDA | 6/6 tests green | **pass** |
| Device CUDA path | works when CUDA present | not run on desktop yet | unknown |

## Historical (pre-PoC DenseDifferential, do not market)

| Ratio | Measured fidelity |
|---|---|
| 2× | 0.670 |
| 4× | 0.462 |
| 16× | 0.228 |
