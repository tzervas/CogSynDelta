# PoC-2 Spec — RegionRegistry + softmax router

## Framing

Still **one mind**. Two cognitive regions on a shared stream. Gating is
softmax top-k (MoE-style). This is **not** mHC, not JEPA, not a fleet.

## Goal

Register two same-dim stream regions and route tokens between them with a
measured gate. Tie the routed stream back to the PoC-1 memory compactors.

## Contract honesty

`CognitiveRegion.activate(stream) -> [B, D]` is the routing surface.
`LatentVAE.forward` stays `(recon, mu, logvar)` for the ELBO train path.
Do not force the train tuple through the region protocol.

## Definition of done

1. `CognitiveRegion` requires `name` + `activate`; `LatentVAE` and
   `ResidualMLPRegion` both satisfy the protocol
2. `RegionRegistry` register / get / names; duplicate name is an error
3. `SoftmaxRouter` top-k mixes activations; per-token weights sum to 1;
   load fractions are reported
4. `python -m cogsyndelta.poc.cli route --device cpu` prints JSON with
   load + optional compact fidelity
5. `tests/test_poc_registry.py` passes on CPU
6. `STATUS.md` PoC-2 rows are measured; no mHC / JEPA / swarm claims

## Non-goals

mHC interconnect, aux load-balance loss, more than two regions, training
the router, JEPA, quantum, multi-agent orchestration.
