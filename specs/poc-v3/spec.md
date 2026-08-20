# PoC-3 Spec — Train softmax router with Switch aux load-balance

**Feature Branch**: `feat/poc-train-route`
**Status**: Implemented
**Stacked on**: `feat/poc-cuda-measure`

## Framing

Still **one mind**. Two cognitive regions on a shared stream. Gating is
softmax top-k (MoE-style). This is **not** mHC, not JEPA, not a fleet.

PoC-1 proved the substrate (one LatentVAE region + measured compression).
PoC-2 proved two regions + a real softmax gate (eval-only).
PoC-3 **trains** that gate (and optionally region params) so routed
reconstruction loss falls, with a real auxiliary load-balance term so
token load is not `{1.0, 0.0}`.

`CognitiveRegion.activate(stream) -> [B, D]` remains the routing surface.
`LatentVAE.forward` stays `(recon, mu, logvar)` for the ELBO train path.
Do not force the ELBO tuple through the region protocol.

## Goal

Train the softmax gate so mixed `activate` reconstructions improve, while
Switch-Transformer aux load-balance keeps both regions in use.

## Aux load-balance formula

Switch Transformer (Fedus et al., 2021). For a batch of `T` tokens and
`N` regions:

- `P_i = (1/T) * sum_t softmax(logits_t)_i` — mean router probability
- `f_i = (1/T) * sum_t 1[i ∈ top-k(t)]` — fraction of tokens dispatched
- `L_aux = N * sum_i (f_i * P_i)`

Uniform routing yields `L_aux = 1`. Collapse onto one region yields
`L_aux = N`. The train objective is

`L = MSE(routed_activate, stream) + aux_coef * L_aux`.

The synthetic stream is drawn **once per seed** so first vs last
reconstruction compares the same tokens (not a new rand batch each step).

Implemented in `cogsyndelta.poc.router.switch_aux_load_balance`.

## Definition of done

1. This spec: framing, DoD, non-goals (mHC / JEPA / swarm out)
2. Switch-style `N * sum(f_i * P_i)` aux term
3. `python -m cogsyndelta.poc.cli train-route --device cpu|cuda --steps N`
4. `tests/test_poc_route_train.py`:
   - last routed recon loss < first over ≥20 steps (CPU)
   - aux LB is finite
   - load fractions not `{1.0, 0.0}` on a batch of 8
   - `LatentVAE.forward` still returns the ELBO tuple
5. `STATUS.md` PoC-3 rows are **measured** (CPU, and CUDA if available)
6. `.github/workflows/poc-ci.yml` runs the new tests and CLI smoke
7. Google-style docstrings; `python3 scripts/quality_control.py src/ --fail-under 90`
8. `./scripts/ci_local.sh --poc` (or ruff + pytest PoC + quality) before push

## Non-goals

mHC / moderated interconnect, JEPA / V-JEPA, a fleet of agents, more than
two regions, quantum backends, 10× compression claims.
