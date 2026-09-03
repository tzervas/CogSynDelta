"""Thin adapter between a CSD entry point and the gpu-pack job runner.

gpu-pack (a separate repo -- see the operator's tooling-lives-in-its-own-repo rule) packs
concurrent training/quant jobs onto a shared card by VRAM budget. This module is the two
env vars it hands a launched job through, plus the one line it reads back:

- ``GPU_PACK_BUDGET_MIB``: if set and CUDA is available, cap this process's CUDA
  allocator to that many MiB via `torch.cuda.set_per_process_memory_fraction` (a soft,
  same-process cap -- see the fleet's process-level pseudo-isolation policy, not hard
  MIG isolation). Unset: no cap, matching prior behaviour exactly.
- ``GPU_PACK_PROBE``: when truthy, the caller may cap step count to a short probe run
  (`PROBE_STEPS`) instead of the full config -- used by pretrain.py's entry point.
- `report_peak()`: on completion, prints ``GPU_PACK_PEAK_MIB=<n>`` to stderr so a
  launcher never has to scrape a training log for the number it needs to size the next
  concurrent batch. Unset budget or no CUDA: no line, matching prior behaviour.

Deliberately free of any torch import at module load time is not required here (torch is
already a hard dependency of every caller), but every function below is a no-op unless
its env var is set, so importing this module changes nothing for a run that ignores it.
"""

from __future__ import annotations

import math
import os
import sys

#: Step count a probe run (`GPU_PACK_PROBE=1`) is capped to.
PROBE_STEPS = 20


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in ("", "0", "false", "no")


def apply_budget_from_env(device_index: int = 0) -> float | None:
    """Cap this process's CUDA memory fraction from `GPU_PACK_BUDGET_MIB`, if set.

    No-op (returns ``None``) when the env var is unset, when CUDA is unavailable, or
    when running under a test's mocked `torch.cuda` that reports no devices -- this must
    never be the thing that turns an unrelated run into a CUDA-required one.

    Returns the fraction actually applied, for callers that want to log it.

    Raises:
        ValueError: `GPU_PACK_BUDGET_MIB` is set but not a plain integer.
    """
    raw = os.environ.get("GPU_PACK_BUDGET_MIB")
    if raw is None:
        return None

    try:
        budget_mib = int(raw)
    except ValueError as exc:
        raise ValueError(f"GPU_PACK_BUDGET_MIB must be an integer MiB value, got {raw!r}") from exc

    import torch

    if not torch.cuda.is_available():
        return None

    total_bytes = torch.cuda.get_device_properties(device_index).total_memory
    total_mib = total_bytes / (2**20)
    fraction = min(1.0, budget_mib / total_mib) if total_mib > 0 else 1.0
    torch.cuda.set_per_process_memory_fraction(fraction, device_index)
    return fraction


def probe_requested() -> bool:
    """True when `GPU_PACK_PROBE` asks for a short probe run rather than a full one."""
    return _truthy(os.environ.get("GPU_PACK_PROBE"))


def report_peak(device_index: int = 0) -> int | None:
    """Print ``GPU_PACK_PEAK_MIB=<n>`` to stderr once, if CUDA was used this process.

    No-op (returns ``None``, prints nothing) when CUDA is unavailable -- a CPU-only run
    (tests, the 1080 Ti when its torch build lacks CUDA support, ...) has no peak to
    report and must not gain a spurious line gpu-pack would otherwise try to parse.
    """
    import torch

    if not torch.cuda.is_available():
        return None

    peak_mib = math.ceil(torch.cuda.max_memory_reserved(device_index) / (2**20))
    print(f"GPU_PACK_PEAK_MIB={peak_mib}", file=sys.stderr, flush=True)
    return peak_mib
