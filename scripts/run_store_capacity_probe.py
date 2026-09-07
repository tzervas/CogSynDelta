#!/usr/bin/env python3
"""Row E1 gate (ii): the DYNAMIC-CAPACITY PROBE, run on one live GPU. Receipt JSON to --out.

WHAT THE GATE ASKS FOR, verbatim (`docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` line
2649): *"compute capacity from a live `nvidia-smi` plus the scheduler's current KV and
activation budgets on the 3090 Ti (24 GiB) and the 5080 (16 GiB), assert the two DIFFER,
assert each is `> 0`, assert each is RESPECTED -- a write that would exceed it triggers
eviction rather than an allocation -- and assert the value CHANGES when the active-region set
changes... A probe returning one number for both cards has measured a constant and failed; a
probe returning two DIFFERENT numbers where one is zero has also failed."*

This script is one card's half. It runs on the host that owns the card, writes a receipt, and
exits non-zero if that card's half fails. The cross-card assertions -- the two numbers differ,
neither is zero -- are made in `tests/interconnect/test_e1_capacity.py` against the two
committed receipts, because no single host can see both cards.

FOUR ARMS, and each exists to answer a different clause of the gate:

  1. `deployment`         -- the headline capacity at the deployment context and the four
                             encoding regions. This is the number the gate compares across
                             cards.
  2. `region_set_changed` -- the same tick with `episodic_store` admitted as a fifth
                             participant. Its capacity MUST differ from arm 1, which is the
                             gate's *"changes when the active-region set changes"*; a probe
                             whose answer is per-host-constant has measured the card, not the
                             tick.
  3. `long_context`       -- a context length large enough to drive the residual to zero on a
                             16 GiB card. This is section 9.11's case, and it is the arm that
                             shows the PRE-COMMITTED FLOOR actually fires rather than being a
                             paragraph. On the 24 GiB card the same arm stays on the residual
                             branch, which is the contrast that makes the branch meaningful.
  4. `respected`          -- admits spans against arm 1's capacity through the real store and
                             asserts the over-capacity write EVICTED rather than allocated.
                             The spans are declared, not allocated: DEC-65's store is an index
                             over runtime-owned bytes, so a 13 GiB capacity is exercised with
                             a handful of small tensors carrying real `span_bytes`. Allocating
                             13 GiB to prove a bound about not allocating would be a strange
                             way to test it.

NOT A pytest FILE, on purpose: the GPU CI image has torch and a driver but no pytest, the same
reason `scripts/run_poc_cuda_gpu.py` is written this way.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogsyndelta.interconnect.episodic.capacity import (
    GIB,
    CapacityBranch,
    SchedulerBudgets,
    capacity_decision,
    probe_host_vram,
)

DEFAULT_REGIONS = ("language", "reason", "retrieve", "visual")
"""The four frozen encoding regions (DEC-48): the active set at a normal tick."""

STORE_PARTICIPANT = "episodic_store"
"""DEC-49's fifth participant. Admitting it is what arm 2 changes."""


def _respected_arm(capacity: int) -> dict[str, Any]:
    """Admit declared spans against `capacity` and assert an over-capacity write evicts.

    Args:
        capacity: The measured capacity in bytes, from arm 1.

    Returns:
        A receipt fragment recording what was admitted, what survived, and whether the bound
        held. `ok` is False if the store ever exceeded the capacity or if nothing was evicted
        when the working set was deliberately built to exceed it.
    """
    import torch

    from cogsyndelta.interconnect.episodic.backends import InMemoryBackend
    from cogsyndelta.interconnect.episodic.store import EpisodicStoreImpl
    from cogsyndelta.interconnect.episodic_store import derive_scope

    span = max(1, capacity // 4)
    store = EpisodicStoreImpl(
        {"chat"},
        backend=InMemoryBackend(),
        capacity_provider=lambda: capacity,
        host="probe",
    )
    store.start()
    scope = derive_scope("probe-principal")
    latent = torch.ones(8)
    peak_resident = 0
    for i in range(6):  # 6 spans of capacity/4 = 1.5x capacity: the bound must bind
        store.learn(scope, "chat", f"span-{i}", latent, importance=float(i), span_bytes=span)
        peak_resident = max(peak_resident, store.resident_bytes)
    survivors = sorted(key[2] for key in store.resident_keys())
    ok = peak_resident <= capacity and len(survivors) < 6
    return {
        "capacity_bytes": capacity,
        "span_bytes_each": span,
        "spans_written": 6,
        "declared_total_bytes": span * 6,
        "peak_resident_bytes": peak_resident,
        "survivors": survivors,
        "evicted": 6 - len(survivors),
        "allocated_bytes_for_spans": int(latent.numel() * latent.element_size()),
        "ok": ok,
    }


def _arm(vram, budgets, args, label: str) -> dict[str, Any]:
    """Evaluate one capacity arm and render it for the receipt."""
    decision = capacity_decision(
        vram,
        budgets,
        safety_margin_bytes=int(args.safety_margin_gib * GIB),
        floor_bytes=int(args.floor_gib * GIB),
    )
    payload = decision.to_dict()
    payload["arm"] = label
    return payload


def main() -> int:
    """Run the probe on this host's card and write the receipt.

    Returns:
        0 if every assertion in this card's half passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="receipt JSON path")
    parser.add_argument("--host", default=socket.gethostname(), help="host label for the receipt")
    parser.add_argument("--index", type=int, default=0, help="CUDA index of the card to read")
    parser.add_argument("--context-len", type=int, default=8192, help="deployment context length")
    parser.add_argument(
        "--long-context-len",
        type=int,
        default=2_000_000,
        help="context length for the floor-branch arm (section 9.11's zero-residual case)",
    )
    parser.add_argument("--safety-margin-gib", type=float, default=2.0)
    parser.add_argument("--activation-reserve-gib", type=float, default=1.0)
    parser.add_argument("--floor-gib", type=float, default=2.0)
    parser.add_argument("--kv-bytes-per-token-per-region", type=int, default=2048)
    args = parser.parse_args()

    receipt: dict[str, Any] = {
        "row": "E1",
        "gate": "ii -- dynamic capacity probe",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": args.host,
        "platform": platform.platform(),
        "formula": (
            "max(0, VRAM_total - KV_reserved(context_len, regions_active) "
            "- activation_reserve - safety_margin)"
        ),
        "inputs": {
            "context_len": args.context_len,
            "long_context_len": args.long_context_len,
            "safety_margin_bytes": int(args.safety_margin_gib * GIB),
            "activation_reserve_bytes": int(args.activation_reserve_gib * GIB),
            "floor_bytes": int(args.floor_gib * GIB),
            "kv_bytes_per_token_per_region": args.kv_bytes_per_token_per_region,
            "regions_deployment": list(DEFAULT_REGIONS),
            "regions_changed": [*DEFAULT_REGIONS, STORE_PARTICIPANT],
        },
        "arms": {},
        "errors": [],
    }

    try:
        vram = probe_host_vram(host=args.host, index=args.index)
    except Exception as exc:
        receipt["errors"].append(f"probe_host_vram: {exc}")
        receipt["ok"] = False
        Path(args.out).write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
        return 1

    receipt["device_name"] = vram.device_name
    receipt["vram_total_bytes"] = vram.total_bytes

    def budgets(context_len: int, regions: tuple[str, ...]) -> SchedulerBudgets:
        return SchedulerBudgets.from_context(
            context_len,
            regions,
            kv_bytes_per_token_per_region=args.kv_bytes_per_token_per_region,
            activation_reserve_bytes=int(args.activation_reserve_gib * GIB),
        )

    deployment = _arm(vram, budgets(args.context_len, DEFAULT_REGIONS), args, "deployment")
    changed = _arm(
        vram,
        budgets(args.context_len, (*DEFAULT_REGIONS, STORE_PARTICIPANT)),
        args,
        "region_set_changed",
    )
    long_context = _arm(vram, budgets(args.long_context_len, DEFAULT_REGIONS), args, "long_context")
    receipt["arms"]["deployment"] = deployment
    receipt["arms"]["region_set_changed"] = changed
    receipt["arms"]["long_context"] = long_context
    receipt["capacity_bytes"] = deployment["capacity_bytes"]
    receipt["branch"] = deployment["branch"]
    receipt["long_context_branch"] = long_context["branch"]

    checks: dict[str, bool] = {}
    checks["capacity_positive"] = deployment["capacity_bytes"] > 0
    checks["region_set_changes_capacity"] = (
        changed["capacity_bytes"] != deployment["capacity_bytes"]
    )
    checks["branch_named"] = deployment["branch"] in {b.value for b in CapacityBranch}
    checks["floor_arm_positive"] = long_context["capacity_bytes"] > 0

    try:
        respected = _respected_arm(int(deployment["capacity_bytes"]))
        receipt["arms"]["respected"] = respected
        checks["capacity_respected"] = bool(respected["ok"])
    except Exception as exc:
        receipt["errors"].append(f"respected arm: {exc}")
        checks["capacity_respected"] = False

    receipt["checks"] = checks
    receipt["ok"] = all(checks.values()) and not receipt["errors"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
