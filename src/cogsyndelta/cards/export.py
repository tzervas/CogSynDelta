"""`export_safetensors`: a real `.safetensors` sibling of a torch checkpoint.

WHY THIS EXISTS
The Hub renders its "Safetensors" badge and model-size sidebar only for a file it can
itself parse as safetensors -- `final.pt` (a pickled `torch.save` file) gets neither,
however faithfully `scripts/csd-publish-checkpoint.py` hashes and uploads it. This
module writes a second file, `<checkpoint stem>.safetensors`, alongside the checkpoint,
from the SAME weights -- never a substitute for `final.pt`, which stays the required
primary artifact (see that script's own module docstring).

STATE DICT ONLY, FP32, NO PICKLE -- BY CONSTRUCTION, NOT BY CONVENTION
`safetensors.torch.save_file` accepts only a flat `{name: Tensor}` mapping and refuses
anything else outright -- there is no way to smuggle an arbitrary pickled object through
it the way `torch.save` allows. `_extract_state_dict` below does the narrowing this
project's own checkpoints need (a bare state dict, or -- for a training loop that later
wraps one -- a `{"state_dict": ...}` / `{"model": ...}` / `{"model_state_dict": ...}`
wrapper) and raises `ValueError` for anything else, so a checkpoint shape this function
cannot confidently narrow to tensors-only is refused before `save_file` is ever called,
not silently coerced.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

__all__ = ["export_safetensors", "safetensors_round_trip_matches"]


def _extract_state_dict(loaded: Any) -> dict[str, torch.Tensor]:
    """Narrow whatever `torch.load` handed back to a plain `{name: Tensor}` mapping,
    or raise -- never guess past an ambiguous shape.
    """
    if (
        isinstance(loaded, dict)
        and loaded
        and all(isinstance(v, torch.Tensor) for v in loaded.values())
    ):
        return loaded
    if isinstance(loaded, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            inner = loaded.get(key)
            if (
                isinstance(inner, dict)
                and inner
                and all(isinstance(v, torch.Tensor) for v in inner.values())
            ):
                return inner
    raise ValueError(
        f"{type(loaded).__name__} does not look like a state dict (a plain "
        "{name: Tensor} mapping, or a wrapper dict carrying one under "
        "'state_dict'/'model'/'model_state_dict') -- export_safetensors only ever "
        "writes plain tensors, never an arbitrary pickled object, and refuses rather "
        "than guess which part of an unrecognised shape is the weights"
    )


def export_safetensors(checkpoint: Path) -> Path:
    """Write `checkpoint`'s state dict to `checkpoint.with_suffix(".safetensors")` and
    return that path.

    Every tensor is detached, cast to fp32, made contiguous (safetensors requires
    contiguous storage) and moved to CPU before writing -- the same fp32-only,
    weights-only contract `scripts/csd-publish-checkpoint.py` already applies to
    `final.pt` itself. The written file's metadata records the source checkpoint's own
    sha256 (`source_checkpoint`, `source_sha256`), so the two files' provenance is
    traceable from the safetensors file alone, without re-hashing `final.pt`.

    Raises:
        ValueError: `checkpoint`'s contents do not narrow to a plain state dict (or a
            recognised wrapper around one) -- see `_extract_state_dict`.
    """
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = _extract_state_dict(loaded)
    cpu_fp32 = {
        name: tensor.detach().to(dtype=torch.float32).contiguous().cpu()
        for name, tensor in state_dict.items()
    }

    source_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    out_path = checkpoint.with_suffix(".safetensors")
    save_file(
        cpu_fp32,
        str(out_path),
        metadata={"source_checkpoint": checkpoint.name, "source_sha256": source_sha256},
    )
    return out_path


def safetensors_round_trip_matches(safetensors_path: Path, checkpoint: Path) -> bool:
    """True iff every tensor `load_file(safetensors_path)` returns is bit-identical
    (`torch.equal`, after casting the checkpoint's own tensor to fp32 -- the same cast
    `export_safetensors` applies before writing) to `checkpoint`'s own state dict, under
    the same name, with no extra or missing keys on either side.

    A small, dependency-free round-trip check `export_safetensors`'s own caller (and
    its tests) can use without re-deriving the comparison by hand each time.
    """
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    original = _extract_state_dict(loaded)
    restored = load_file(str(safetensors_path))
    if set(original) != set(restored):
        return False
    return all(
        torch.equal(tensor.detach().to(dtype=torch.float32).contiguous().cpu(), restored[name])
        for name, tensor in original.items()
    )
