"""`load_checkpoint` (design row W0c, DEC-40) must be the ONE place a production
checkpoint is `torch.load`ed, and it must verify a content hash before it does.

WHY THIS EXISTS
`tests/test_checkpoint_load_security.py` proves `weights_only=True` defeats a hostile
pickle payload at the sites that load a checkpoint path read out of a receipt JSON. It
does not prove every such site stays wired that way, and it says nothing about a checkpoint
silently swapped for a different -- still `weights_only`-safe -- file after a receipt
recorded its hash. `cogsyndelta.regions._checkpoint.load_checkpoint` closes both: it is
the one function that calls `torch.load` on a checkpoint in production code, and it
refuses a content-hash mismatch BEFORE the file is ever opened by the unpickler.

This file has three parts:
  1. A lint that greps a SCOPED set of files (the ones this hardening pass actually
     routed through `load_checkpoint`) for a stray `torch.load(` outside
     `_checkpoint.py`'s own definition, and asserts none exist -- run against the real,
     shipped source, so it fails the suite the moment a future edit calls `torch.load`
     directly instead of going through `load_checkpoint`.
  2. A test that the lint's own checker function actually fires: a scratch copy of a
     scoped file with a stray `torch.load(` appended is flagged.
  3. `load_checkpoint` itself: a positive load, a hash-mismatch negative (one byte
     flipped after the expected hash was computed), and a proof that the hash check runs
     BEFORE `torch.load` -- a mismatched hash on a MALICIOUS payload never reaches the
     unpickler at all.

SCOPE, DELIBERATELY NARROW
Several other modules in this project call `torch.load` for unrelated things --
`regions/_tokencache.py`'s token cache, `core/model_sectioning.py`'s section shards,
`contracts/compactor.py`/`memory/memory_persistence.py`'s stored blobs. None of those are
"a training checkpoint read from a receipt-recorded path" -- the threat model this and
`test_checkpoint_load_security.py` exist for -- and none were in scope for this pass. A
repo-wide grep would flag all of them for no reason; the lint below is scoped to exactly
the files this pass touched: `regions/pretrain.py`, `regions/_checkpoint.py` (which
DEFINES the one legitimate call), `scripts/csd-quantize.py`, `scripts/csd-benchmark.py`,
and `regions/retrieve.py` if that file exists (it does not, as of this commit).

WHY PART 1 AND 2 IMPORT NEITHER `torch` NOR `cogsyndelta` AT MODULE SCOPE
`stray_torch_load_calls` is pure `re`/`pathlib` text scanning -- it needs neither. But
`cogsyndelta.regions._checkpoint` (the module part 3 actually exercises) is a *submodule*
of the `regions` package, and Python always runs a package's `__init__.py` before one of
its submodules -- `cogsyndelta.regions.__init__` imports `regions.pretrain`, which
imports `tokenizers` at module scope, even though `_checkpoint.py` itself needs neither
`pyarrow` nor `tokenizers`. A previous version of this file called
`pytest.importorskip(...)` for both at MODULE scope to route around that, which does not
do what it looks like it does: `importorskip` raises `Skipped` during collection, which
skips the entire file, not just the tests that need the train group -- so in a dev-group-
only environment (every CI job in this repo: `.github/workflows/ci.yml`,
`code-quality.yml`, `scripts/ci_local.sh` all `uv sync --group dev`, never `--group
train`) parts 1 and 2 never ran either, despite needing nothing torch/tokenizers-related.
The `ckpt` fixture below does the same two `importorskip` calls, but from INSIDE a
fixture that only the part-3 tests request -- so only those individual tests skip absent
the train group, and the lint (part 1) and its self-test (part 2) run everywhere,
including in CI.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.cpu

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SCOPED_RELPATHS: tuple[str, ...] = (
    "src/cogsyndelta/regions/pretrain.py",
    "src/cogsyndelta/regions/_checkpoint.py",
    "scripts/csd-quantize.py",
    "scripts/csd-benchmark.py",
    # regions/retrieve.py is mentioned in the design row this closes but does not exist
    # yet -- added here (not hardcoded above) so it is covered automatically once it does,
    # without this file silently going stale about a scope that changed.
    "src/cogsyndelta/regions/retrieve.py",
)

_TORCH_LOAD_RE = re.compile(r"torch\.load\(")


def stray_torch_load_calls(path: Path) -> list[int]:
    """1-indexed line numbers of a `torch.load(` call in `path` NOT inside
    `_checkpoint.py`'s own `load_checkpoint` definition -- the one call this whole lint
    exists to fence in. Every other `torch.load(` anywhere in a scoped file, including a
    scratch copy under any other name, is flagged.
    """
    lines = path.read_text().splitlines()
    hits = [i + 1 for i, line in enumerate(lines) if _TORCH_LOAD_RE.search(line)]
    if path.name != "_checkpoint.py":
        return hits

    start = next(
        (i for i, line in enumerate(lines) if line.startswith("def load_checkpoint(")), None
    )
    assert start is not None, "_checkpoint.py must define load_checkpoint"
    end = len(lines)
    for i in range(start + 1, len(lines)):
        # The next top-level `def `/`class ` -- NOT merely "any column-0 character": this
        # project's own formatting (ruff/black) puts a multi-line signature's closing
        # `)` at column 0 too (see load_checkpoint's own signature), which a bare
        # "non-whitespace at column 0" check mistakes for the end of the function.
        if lines[i].startswith(("def ", "class ")):
            end = i
            break
    return [n for n in hits if not (start < n <= end)]


# ---------------------------------------------------------------------------------------
# 1. The lint itself, run against the real shipped source. Needs neither torch nor
#    cogsyndelta -- always collects and runs, dev group alone included.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("relpath", _SCOPED_RELPATHS)
def test_no_stray_torch_load_in_scoped_files(relpath: str) -> None:
    path = _REPO_ROOT / relpath
    if not path.is_file():
        pytest.skip(f"{relpath} does not exist yet")
    violations = stray_torch_load_calls(path)
    assert violations == [], (
        f"{relpath}: torch.load( outside load_checkpoint at line(s) {violations} -- "
        f"route through cogsyndelta.regions._checkpoint.load_checkpoint instead"
    )


# ---------------------------------------------------------------------------------------
# 2. The lint's checker function actually fires -- proven on a scratch copy, never on the
#    real tree (which is asserted clean above). Also needs neither torch nor cogsyndelta:
#    the scratch copies below are plain text edits, never imported or executed.
# ---------------------------------------------------------------------------------------


def test_lint_fires_on_a_stray_torch_load_added_to_a_scratch_copy(tmp_path: Path) -> None:
    original = (_REPO_ROOT / "src/cogsyndelta/regions/pretrain.py").read_text()
    scratch = tmp_path / "pretrain.py"
    scratch.write_text(original + "\n\ndef _stray_direct_load(p):\n    return torch.load(p)\n")

    violations = stray_torch_load_calls(scratch)

    assert violations != []


def test_lint_still_exempts_load_checkpoints_own_call_in_a_scratch_copy_of__checkpoint_py(
    tmp_path: Path,
) -> None:
    """Negative control: copying `_checkpoint.py` verbatim must NOT flag its own,
    legitimate `torch.load(` inside `load_checkpoint` -- the lint targets a call OUTSIDE
    that function, not the file containing it."""
    original = (_REPO_ROOT / "src/cogsyndelta/regions/_checkpoint.py").read_text()
    scratch = tmp_path / "_checkpoint.py"
    scratch.write_text(original)

    assert stray_torch_load_calls(scratch) == []


def test_lint_fires_on_a_stray_torch_load_added_outside_load_checkpoint_in__checkpoint_py(
    tmp_path: Path,
) -> None:
    """The exemption is scoped to `load_checkpoint`'s own body, not the whole file: a
    second, stray `torch.load(` appended elsewhere in a scratch copy of `_checkpoint.py`
    must still be flagged."""
    original = (_REPO_ROOT / "src/cogsyndelta/regions/_checkpoint.py").read_text()
    mutated = original + "\n\ndef _stray_direct_load(p):\n    return torch.load(p)\n"
    scratch = tmp_path / "_checkpoint.py"
    scratch.write_text(mutated)

    violations = stray_torch_load_calls(scratch)

    # Exactly the appended line, not the legitimate one inside load_checkpoint -- computed
    # from the mutated text itself rather than hand arithmetic on `original`'s line count.
    appended_line_no = next(
        i + 1 for i, line in enumerate(mutated.splitlines()) if "return torch.load(p)" in line
    )
    assert violations == [appended_line_no]


# ---------------------------------------------------------------------------------------
# 3. load_checkpoint itself. Everything below requests the `ckpt` fixture, which is the
#    ONLY place torch/tokenizers/pyarrow are imported in this file -- see the module
#    docstring. Absent the train group, only these tests skip.
# ---------------------------------------------------------------------------------------


@pytest.fixture()
def ckpt() -> SimpleNamespace:
    """`torch` plus `cogsyndelta.regions._checkpoint`'s public API, imported HERE (inside
    a fixture, requested only by the tests below) rather than at module scope -- see the
    module docstring. `pytest.importorskip` from inside a fixture skips only the
    requesting test, not the whole file.
    """
    pytest.importorskip("pyarrow", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    import torch as torch_mod

    from cogsyndelta.regions._checkpoint import (
        ChecksumMismatchError,
        load_checkpoint,
        sha256_file,
    )

    return SimpleNamespace(
        torch=torch_mod,
        ChecksumMismatchError=ChecksumMismatchError,
        load_checkpoint=load_checkpoint,
        sha256_file=sha256_file,
    )


def _save_tiny_checkpoint(torch_mod: Any, path: Path, *, step: int = 8000) -> dict:
    state = {
        "model": {"weight": torch_mod.randn(4, 4), "bias": torch_mod.zeros(4)},
        "step": step,
    }
    torch_mod.save(state, path)
    return state


def test_load_checkpoint_loads_a_matching_file(ckpt: SimpleNamespace, tmp_path: Path) -> None:
    ckpt_path = tmp_path / "final.pt"
    state = _save_tiny_checkpoint(ckpt.torch, ckpt_path)
    expected = ckpt.sha256_file(ckpt_path)

    loaded = ckpt.load_checkpoint(ckpt_path, expected_sha256=expected, map_location="cpu")

    assert loaded["step"] == state["step"]
    assert ckpt.torch.equal(loaded["model"]["weight"], state["model"]["weight"])


def test_load_checkpoint_with_no_expected_hash_still_loads(
    ckpt: SimpleNamespace, tmp_path: Path
) -> None:
    """`expected_sha256=None` (the default) is not a broken check -- it is the shape
    every caller with no prior hash to verify against (`load_resumable`, resuming its own
    training loop) legitimately uses."""
    ckpt_path = tmp_path / "final.pt"
    _save_tiny_checkpoint(ckpt.torch, ckpt_path)

    loaded = ckpt.load_checkpoint(ckpt_path, map_location="cpu")

    assert loaded["step"] == 8000


def test_load_checkpoint_refuses_a_checkpoint_with_one_byte_flipped(
    ckpt: SimpleNamespace, tmp_path: Path
) -> None:
    """The regression this whole file exists to close: a checkpoint changed on disk
    AFTER its hash was recorded must be refused, even though it is still a perfectly
    ordinary, `weights_only`-safe tensor file that `torch.load` alone would accept
    without complaint."""
    ckpt_path = tmp_path / "final.pt"
    _save_tiny_checkpoint(ckpt.torch, ckpt_path)
    expected = ckpt.sha256_file(ckpt_path)

    data = bytearray(ckpt_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    ckpt_path.write_bytes(bytes(data))

    with pytest.raises(ckpt.ChecksumMismatchError, match="does not match expected"):
        ckpt.load_checkpoint(ckpt_path, expected_sha256=expected, map_location="cpu")


def test_load_checkpoint_refuses_a_missing_file(ckpt: SimpleNamespace, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ckpt.load_checkpoint(tmp_path / "nope.pt", map_location="cpu")


def test_sha256_out_receives_the_expected_hash_when_one_was_given(
    ckpt: SimpleNamespace, tmp_path: Path
) -> None:
    """`sha256_out` exists so `scripts/csd-benchmark.py` / `scripts/csd-quantize.py`
    can record the checkpoint's hash in their own receipts without hashing the file a
    second time. When `expected_sha256` is given, the value appended must be that same
    verified hash -- not merely `expected_sha256` echoed back uninspected."""
    ckpt_path = tmp_path / "final.pt"
    _save_tiny_checkpoint(ckpt.torch, ckpt_path)
    expected = ckpt.sha256_file(ckpt_path)

    out: list[str] = []
    ckpt.load_checkpoint(ckpt_path, expected_sha256=expected, map_location="cpu", sha256_out=out)

    assert out == [expected]


def test_sha256_out_is_populated_even_with_no_expected_hash(
    ckpt: SimpleNamespace, tmp_path: Path
) -> None:
    """A caller with no prior hash to verify against (`expected_sha256=None`, e.g. a
    pre-R9 training receipt with no recorded `checkpoint_sha256`) must still get the
    checkpoint's real content hash back -- computed here, not skipped."""
    ckpt_path = tmp_path / "final.pt"
    _save_tiny_checkpoint(ckpt.torch, ckpt_path)
    real = ckpt.sha256_file(ckpt_path)

    out: list[str] = []
    ckpt.load_checkpoint(ckpt_path, map_location="cpu", sha256_out=out)

    assert out == [real]


def test_sha256_out_is_not_populated_on_a_mismatch(ckpt: SimpleNamespace, tmp_path: Path) -> None:
    """A checksum mismatch must raise before any caller-visible hash is reported --
    `sha256_out` staying empty on the raised path keeps a caller from ever reading a
    hash for a load that was actually refused."""
    ckpt_path = tmp_path / "final.pt"
    _save_tiny_checkpoint(ckpt.torch, ckpt_path)

    out: list[str] = []
    with pytest.raises(ckpt.ChecksumMismatchError):
        ckpt.load_checkpoint(
            ckpt_path, expected_sha256="0" * 64, map_location="cpu", sha256_out=out
        )

    assert out == []


class _MaliciousReduce:
    """Same shape as `test_checkpoint_load_security.py`'s: pickles to a call that leaves
    a marker file behind when (and only when) it is actually unpickled."""

    def __init__(self, sentinel: Path) -> None:
        self.sentinel = sentinel

    def __reduce__(self):
        import os

        return (os.system, (f"echo pwned >> {self.sentinel}",))


def test_load_checkpoint_checks_the_hash_before_torch_load_ever_opens_the_file(
    ckpt: SimpleNamespace, tmp_path: Path
) -> None:
    """Ordering proof: a MALICIOUS payload, saved under a hash that does NOT match what
    is passed as `expected_sha256`, must be refused by the hash check -- and the
    `__reduce__` side effect must never fire, proving `torch.load` was never reached at
    all. If the hash check ran AFTER (or not before) `torch.load`, this would instead
    raise `Weights only load failed` with the sentinel file created for a heartbeat
    before that error -- a different failure than the one this test pins."""
    sentinel = tmp_path / "sentinel.txt"
    ckpt_path = tmp_path / "final.pt"
    ckpt.torch.save({"model": _MaliciousReduce(sentinel)}, ckpt_path)

    with pytest.raises(ckpt.ChecksumMismatchError):
        ckpt.load_checkpoint(ckpt_path, expected_sha256="0" * 64, map_location="cpu")

    assert not sentinel.exists(), (
        "the malicious __reduce__ ran even though the hash check should have refused "
        "the file before torch.load ever opened it"
    )


def test_load_checkpoint_still_refuses_a_malicious_payload_with_a_matching_hash(
    ckpt: SimpleNamespace, tmp_path: Path
) -> None:
    """`weights_only=True` (`load_checkpoint`'s default) is defense IN DEPTH, not
    superseded by the hash check: an attacker who controls the file also controls its
    hash (they compute it themselves after swapping the file, exactly as a caller with a
    legitimate expected hash would) -- so a matching `expected_sha256` must not be read
    as clearing the payload to unpickle. `weights_only=True` still refuses it."""
    sentinel = tmp_path / "sentinel.txt"
    ckpt_path = tmp_path / "final.pt"
    ckpt.torch.save({"model": _MaliciousReduce(sentinel)}, ckpt_path)
    matching_hash = ckpt.sha256_file(ckpt_path)  # the attacker's own file, honestly hashed

    with pytest.raises(Exception, match="Weights only load failed"):
        ckpt.load_checkpoint(ckpt_path, expected_sha256=matching_hash, map_location="cpu")

    assert not sentinel.exists(), "the malicious __reduce__ ran despite weights_only=True"
