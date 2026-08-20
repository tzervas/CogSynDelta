# PoC-1 Spec

## Framing

CSD ≈ MoE + control loop + shared memory, as **one mind with regions**, not a fleet of agents.

## Goal

Working proof of the substrate: one trainable region + measured compression on CPU (CUDA optional).

## Definition of done

1. `python -m cogsyndelta.poc.cli train --device cpu` → loss decreases, checkpoint written
2. `python -m cogsyndelta.poc.cli compress --device cpu` → JSON with measured ratio + fidelity
3. Same with `--device cuda` when GPU present
4. `tests/test_poc_*.py` pass on CPU
5. STATUS.md reflects measured numbers after bench

## Non-goals

Multi-agent orchestration, JEPA, quantum, 10–100×@0.95 marketing claims, quality-score farming.
