"""A checkpoint path read out of a receipt JSON must not be able to run code.

`scripts/csd-quantize.py` and `scripts/csd-benchmark.py` both resolve a checkpoint path
from `receipt["checkpoint"]` / `train_receipt["checkpoint"]` -- a value read out of a
receipt JSON on `/akula-data/csd/receipts`, an NFS export mounted `rw,no_root_squash`.
Anyone who can write to that tree can point the receipt at a file of their choosing, so
loading it with `torch.load(..., weights_only=False)` is arbitrary code execution: pickle
lets an object's `__reduce__` run any callable (`os.system`, `subprocess.run`, ...) as
part of just being *unpickled*, before the caller ever touches the result.

`weights_only=True` closes this: the unpickler only accepts a small allow-list of
types (tensors, plain containers, a few numeric primitives) and raises rather than
constructing anything else. These tests prove that guard actually fires -- not just that
the keyword is present -- and that a real checkpoint still loads under it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.cpu

_REPO_ROOT = Path(__file__).resolve().parent.parent


class _MaliciousReduce:
    """Pickles to a call that appends a marker line to `sentinel` when unpickled.

    Mirrors the real attack shape: the payload doesn't need cooperation from the code
    that calls `torch.load` afterwards. Just unpickling the object under
    `weights_only=False` executes `os.system` -- `model.load_state_dict(...)` is never
    reached.
    """

    def __init__(self, sentinel: Path) -> None:
        self.sentinel = sentinel

    def __reduce__(self):
        # Real payloads would run something worse; `echo` into a scratch file is enough
        # to prove the side effect did or did not happen without doing anything harmful.
        return (os.system, (f"echo pwned >> {self.sentinel}",))


def test_malicious_checkpoint_is_refused_under_weights_only(tmp_path: Path) -> None:
    """The exact load pattern the two scripts use must reject a hostile checkpoint."""
    sentinel = tmp_path / "sentinel.txt"
    ckpt_path = tmp_path / "final.pt"
    torch.save({"model": _MaliciousReduce(sentinel)}, ckpt_path)

    with pytest.raises(Exception, match="Weights only load failed"):
        # Same call shape as the fixed sites: map_location + weights_only=True.
        torch.load(ckpt_path, map_location="cpu", weights_only=True)

    assert not sentinel.exists(), (
        "the malicious __reduce__ ran (os.system executed) even though weights_only=True "
        "should have refused to unpickle it before any callable ran"
    )


def test_legitimate_tensor_checkpoint_still_loads_under_weights_only(tmp_path: Path) -> None:
    """The guard must not be so strict it breaks the checkpoints training actually writes."""
    ckpt_path = tmp_path / "final.pt"
    state = {
        "model": {"weight": torch.randn(4, 4), "bias": torch.zeros(4)},
        "step": 8000,
        "elapsed_s": 123.5,
    }
    torch.save(state, ckpt_path)

    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    assert loaded["step"] == 8000
    assert torch.equal(loaded["model"]["weight"], state["model"]["weight"])
    assert torch.equal(loaded["model"]["bias"], state["model"]["bias"])


@pytest.mark.parametrize(
    ("script", "checkpoint_expr"),
    [
        ("csd-quantize.py", 'receipt["checkpoint"]'),
        ("csd-benchmark.py", 'train_receipt["checkpoint"]'),
    ],
)
def test_production_sites_load_receipt_checkpoints_with_weights_only_true(
    script: str, checkpoint_expr: str
) -> None:
    """Regression guard: the receipt-driven `torch.load` call must stay weights_only=True.

    This is the site the threat model flagged -- `receipt["checkpoint"]` /
    `train_receipt["checkpoint"]` is a path chosen by whatever wrote the receipt JSON,
    not by this process, so it must never be loaded with `weights_only=False` again.
    """
    source = (_REPO_ROOT / "scripts" / script).read_text()
    pattern = re.compile(
        r"torch\.load\(\s*" + re.escape(checkpoint_expr) + r"\s*,([^)]*)\)", re.DOTALL
    )
    match = pattern.search(source)
    assert match is not None, f"expected a torch.load({checkpoint_expr}, ...) call in {script}"
    call_kwargs = match.group(1)
    assert "weights_only=True" in call_kwargs, (
        f"{script} loads {checkpoint_expr} without weights_only=True: {call_kwargs!r}"
    )
    assert "weights_only=False" not in call_kwargs
