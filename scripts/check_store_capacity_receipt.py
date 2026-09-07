#!/usr/bin/env python3
"""Row E1 gate (ii): assert a freshly measured capacity receipt against the committed pair.

WHY THIS IS A SCRIPT AND NOT A pytest ASSERTION. The GPU runner measures one card, inside a
container, in an image that has torch and a driver but no pytest -- the same constraint
`scripts/run_poc_cuda_gpu.py` was written for. This script runs afterwards on the runner HOST
(outside the container), where only the standard library is available, and turns the fresh
measurement into a pass or a fail.

WHAT IT REFUSES, and why each one is a real failure rather than a formality:

  - a live capacity of zero -- *"a probe returning two DIFFERENT numbers where one is zero has
    also failed"*;
  - a live capacity equal to the OTHER card's -- *"a probe returning one number for both cards
    has measured a constant"*;
  - a live `vram_total` that disagrees with the committed receipt for the same card -- the
    committed evidence has gone stale and the PR body's numbers no longer describe the fleet;
  - a receipt whose own `ok` is false, or whose respect / region-set arms did not pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    """Read one receipt.

    Args:
        path: Receipt JSON path.

    Returns:
        The parsed receipt.
    """
    return json.loads(Path(path).read_text())


def main() -> int:
    """Compare a live receipt against the committed pair and report every failure found.

    Returns:
        0 if every check passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", required=True, help="freshly measured receipt")
    parser.add_argument("--committed-self", required=True, help="committed receipt, same card")
    parser.add_argument("--committed-other", required=True, help="committed receipt, other card")
    args = parser.parse_args()

    live = _load(args.live)
    same = _load(args.committed_self)
    other = _load(args.committed_other)

    failures: list[str] = []
    if live.get("ok") is not True:
        failures.append(f"live receipt is not ok: {live.get('errors')}")
    if int(live.get("capacity_bytes", 0)) <= 0:
        failures.append(f"live capacity is {live.get('capacity_bytes')}, not > 0")
    if int(live.get("capacity_bytes", 0)) == int(other.get("capacity_bytes", -1)):
        failures.append(
            "live capacity equals the other card's -- that is a measured constant, not a "
            "dynamic capacity"
        )
    if int(live.get("vram_total_bytes", 0)) != int(same.get("vram_total_bytes", -1)):
        failures.append(
            f"live VRAM_total {live.get('vram_total_bytes')} disagrees with the committed "
            f"receipt's {same.get('vram_total_bytes')}; the committed evidence is stale"
        )
    for check in ("capacity_respected", "region_set_changes_capacity", "capacity_positive"):
        if live.get("checks", {}).get(check) is not True:
            failures.append(f"live check {check} did not pass")

    print(
        json.dumps(
            {
                "live_capacity_bytes": live.get("capacity_bytes"),
                "live_branch": live.get("branch"),
                "live_long_context_branch": live.get("long_context_branch"),
                "other_card_capacity_bytes": other.get("capacity_bytes"),
                "failures": failures,
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
