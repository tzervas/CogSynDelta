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
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.cpu

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_checkpoint_call_text(source: str, checkpoint_expr: str) -> str:
    """Full text of a `load_checkpoint(<checkpoint_expr>, ...)` call, balanced-paren
    aware -- NOT a `[^)]*` regex, which stops at the FIRST `)` in the call.
    `receipt.get("checkpoint_sha256")`, a nested and perfectly balanced call inside the
    kwargs both production sites pass, hits its own closing paren before
    `load_checkpoint`'s -- a `[^)]*` regex silently truncates the match there, so a
    `weights_only=True` that appears (as it does at both real call sites) AFTER that
    point in the source is never even looked at. Counting paren depth instead captures
    the whole call regardless of what nested, balanced calls appear inside it.
    """
    needle = "load_checkpoint("
    start = source.find(needle)
    while start != -1:
        after = source[start + len(needle) :].lstrip()
        if after.startswith(checkpoint_expr):
            depth = 1
            i = start + len(needle)
            while depth > 0:
                if source[i] == "(":
                    depth += 1
                elif source[i] == ")":
                    depth -= 1
                i += 1
            return source[start:i]
        start = source.find(needle, start + len(needle))
    raise AssertionError(f"no load_checkpoint({checkpoint_expr}, ...) call found in source")


def test_load_checkpoint_call_text_is_not_fooled_by_a_nested_balanced_call() -> None:
    """Proof the helper above does what the module-scope regex it replaced did not: a
    `[^)]*`-style match would stop at the `)` that closes `receipt.get(...)`, never
    reaching `weights_only=True` two lines later. This constructs exactly that shape by
    hand and asserts the full call -- including the part after the nested `)` -- comes
    back."""
    source = (
        "ck = load_checkpoint(\n"
        '    receipt["checkpoint"],\n'
        '    expected_sha256=receipt.get("checkpoint_sha256") or None,\n'
        "    map_location=device,\n"
        "    weights_only=True,\n"
        ")\n"
    )

    call_text = _load_checkpoint_call_text(source, 'receipt["checkpoint"]')

    assert "weights_only=True" in call_text
    assert call_text.startswith('load_checkpoint(\n    receipt["checkpoint"],')
    assert call_text.endswith(")")


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
def test_production_sites_load_receipt_checkpoints_through_load_checkpoint(
    script: str, checkpoint_expr: str
) -> None:
    """Regression guard, updated for `cogsyndelta.regions._checkpoint.load_checkpoint`
    (design row W0c, DEC-40 -- see `tests/test_checkpoint_loader_lint.py`): both sites
    used to call `torch.load(..., weights_only=True)` directly; they now route through
    `load_checkpoint`, plus a content-hash check neither site had before.

    This is still the site the threat model flagged -- `receipt["checkpoint"]` /
    `train_receipt["checkpoint"]` is a path chosen by whatever wrote the receipt JSON,
    not by this process -- so this pins that the call goes through `load_checkpoint`
    (not a raw `torch.load`, which `test_checkpoint_loader_lint.py`'s lint separately
    refuses to allow here at all).

    Earlier versions of this test asserted `"weights_only=True" in call_text` as a
    positive, per-call-site check -- deliberately not just the absence of
    `weights_only=False`, because at the time `load_checkpoint(..., weights_only=flag)`
    forwarded whatever `flag` a caller passed, with no guarantee it was ever `True`.
    bandit's B614 (`Use of unsafe PyTorch load`) flagged exactly that shape: a variable
    `weights_only` reaching `torch.load`, regardless of what any particular caller
    happened to pass. `load_checkpoint` no longer takes a `weights_only` parameter at
    all -- see `test_load_checkpoint_has_no_weights_only_parameter` below -- so a call
    site cannot regress to an unsafe load even by typo; asserting `weights_only=True`
    at the call site would now just assert a `TypeError` never fires, which duplicates
    that structural guarantee instead of adding to it. What this test still needs to
    pin is that the call goes through `load_checkpoint` in the first place.
    """
    source = (_REPO_ROOT / "scripts" / script).read_text()
    call_text = _load_checkpoint_call_text(source, checkpoint_expr)
    assert "weights_only" not in call_text, (
        f"{script}: load_checkpoint({checkpoint_expr}, ...) must not pass weights_only "
        f"-- load_checkpoint no longer accepts it (hardcoded True internally); found "
        f"call: {call_text!r}"
    )


def test_load_checkpoint_has_no_weights_only_parameter() -> None:
    """Structural guarantee, not a convention: no argument exists that could make
    `load_checkpoint` call `torch.load` with anything other than `weights_only=True`.
    This is what makes the per-call-site checks above about a caller passing
    `weights_only=True` unnecessary -- there is no keyword left to pass."""
    import inspect

    from cogsyndelta.regions._checkpoint import load_checkpoint

    assert "weights_only" not in inspect.signature(load_checkpoint).parameters
