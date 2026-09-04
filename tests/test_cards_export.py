"""Tests for `cogsyndelta.cards.export.export_safetensors`.

Covers:
- Round-trip: every tensor `load_file()` reads back from the written `.safetensors`
  file is bit-identical (`torch.equal`) to the source checkpoint's own state dict, same
  names, no extra/missing keys.
- The bare-state-dict shape AND the `{"state_dict": ...}` wrapper shape both narrow
  correctly (`_extract_state_dict`'s two branches).
- Non-fp32 (half) and non-contiguous (transposed view) tensors are cast/copied before
  writing, not rejected -- `safetensors.torch.save_file` itself refuses a
  non-contiguous tensor, so this is load-bearing, not decorative.
- Output path and filename: `<stem>.safetensors` beside the checkpoint, never
  overwriting it.
- Metadata: `source_sha256` on the written file matches the checkpoint's own sha256.
- Refusal (mutation proof): a checkpoint whose contents are not a recognisable state
  dict (a bare tensor, a list, a dict of non-tensors) raises `ValueError` naming why,
  and writes no file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from cogsyndelta.cards.export import export_safetensors, safetensors_round_trip_matches

pytestmark = pytest.mark.cpu


def _make_checkpoint(
    tmp_path: Path, name: str = "final.pt"
) -> tuple[Path, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 4))
    state_dict = model.state_dict()
    path = tmp_path / name
    torch.save(state_dict, path)
    return path, state_dict


# =====================================================================================
# Round trip.
# =====================================================================================


def test_round_trip_bare_state_dict(tmp_path: Path) -> None:
    checkpoint, state_dict = _make_checkpoint(tmp_path)
    out = export_safetensors(checkpoint)

    assert out.is_file()
    restored = load_file(str(out))
    assert set(restored) == set(state_dict)
    for name, tensor in state_dict.items():
        assert torch.equal(tensor.to(dtype=torch.float32), restored[name])


def test_round_trip_helper_agrees_with_manual_check(tmp_path: Path) -> None:
    checkpoint, _ = _make_checkpoint(tmp_path)
    out = export_safetensors(checkpoint)
    assert safetensors_round_trip_matches(out, checkpoint)


def test_round_trip_wrapped_state_dict(tmp_path: Path) -> None:
    """A training loop that wraps the state dict under `{"state_dict": ...}` (a shape
    this project's own checkpoints do not use today, but a plausible future one) must
    still narrow correctly."""
    torch.manual_seed(1)
    model = nn.Linear(8, 8)
    wrapped = {"state_dict": model.state_dict(), "epoch": 3}
    checkpoint = tmp_path / "wrapped.pt"
    torch.save(wrapped, checkpoint)

    out = export_safetensors(checkpoint)
    restored = load_file(str(out))
    assert set(restored) == set(model.state_dict())
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor.to(dtype=torch.float32), restored[name])


# =====================================================================================
# Non-fp32 / non-contiguous tensors are handled, not rejected.
# =====================================================================================


def test_half_precision_tensor_is_upcast_to_fp32(tmp_path: Path) -> None:
    state_dict = {"w": torch.randn(4, 4).half()}
    checkpoint = tmp_path / "half.pt"
    torch.save(state_dict, checkpoint)

    out = export_safetensors(checkpoint)
    restored = load_file(str(out))
    assert restored["w"].dtype == torch.float32
    assert torch.equal(state_dict["w"].to(dtype=torch.float32), restored["w"])


def test_non_contiguous_tensor_is_made_contiguous(tmp_path: Path) -> None:
    """`safetensors.torch.save_file` refuses a non-contiguous tensor outright -- a
    transposed view is the simplest way to construct one. Without the `.contiguous()`
    call in `export_safetensors`, this test's own `export_safetensors` call would raise
    instead of the round trip below ever running."""
    base = torch.randn(4, 6)
    view = base.t()  # transpose: a view, not contiguous
    assert not view.is_contiguous()
    checkpoint = tmp_path / "noncontig.pt"
    torch.save({"w": view}, checkpoint)

    out = export_safetensors(checkpoint)
    restored = load_file(str(out))
    assert torch.equal(view.contiguous(), restored["w"])


# =====================================================================================
# Output path, no overwrite of the source, metadata.
# =====================================================================================


def test_output_path_is_sibling_dot_safetensors(tmp_path: Path) -> None:
    checkpoint, _ = _make_checkpoint(tmp_path, name="final.pt")
    out = export_safetensors(checkpoint)
    assert out == tmp_path / "final.safetensors"
    assert checkpoint.is_file()  # the source checkpoint is untouched


def test_metadata_records_source_sha256(tmp_path: Path) -> None:
    checkpoint, _ = _make_checkpoint(tmp_path)
    expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    out = export_safetensors(checkpoint)

    with out.open("rb") as f:
        import struct

        (header_len,) = struct.unpack("<Q", f.read(8))
        import json as _json

        header = _json.loads(f.read(header_len))
    assert header["__metadata__"]["source_sha256"] == expected_sha
    assert header["__metadata__"]["source_checkpoint"] == checkpoint.name


# =====================================================================================
# Refusal: not a recognisable state dict. Mutation proof built in (three distinct
# unrecognisable shapes, not one).
# =====================================================================================


@pytest.mark.parametrize(
    "payload",
    [
        torch.randn(4, 4),  # a bare tensor, not a mapping at all
        [1, 2, 3],  # a list
        {"a": 1, "b": 2},  # a dict of plain ints, not tensors
        {"state_dict": {"a": 1}},  # wrapper present, but its contents are not tensors
        {},  # empty dict -- nothing to narrow to
    ],
)
def test_unrecognisable_payload_raises_value_error(tmp_path: Path, payload: object) -> None:
    checkpoint = tmp_path / "bad.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="state dict"):
        export_safetensors(checkpoint)

    assert not (tmp_path / "bad.safetensors").exists(), (
        "a refused export must not leave a partial .safetensors file behind"
    )
