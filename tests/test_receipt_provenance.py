"""`cogsyndelta.regions._receipt` -- code provenance and trainer_defaults, and the guard
that refuses to write a receipt with no `code_revision` block.

WHY THESE GUARDS EXIST
See `_receipt.py`'s own module docstring: three training regions plus `csd-quantize.py`
each wrote their receipt's final JSON inline, and none of the four recorded what code
produced the numbers in it. `write_receipt` is now the single place every one of those
call sites routes through, and it refuses outright (raises, writes nothing) if the
`code_revision` block it is about to stamp turns out to be missing.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RECEIPT_PATH = _REPO_ROOT / "src" / "cogsyndelta" / "regions" / "_receipt.py"


def _load_receipt_module():
    """Load `_receipt.py` directly by file path rather than
    `from cogsyndelta.regions import _receipt`.

    `_receipt` is a *submodule* of the `regions` package, and Python always runs a
    package's `__init__.py` before one of its submodules -- `cogsyndelta.regions.__init__`
    imports `regions.pretrain`, which imports `tokenizers` at module scope, even though
    `_receipt.py` itself imports only `json`, `subprocess`, `pathlib` and `typing` and
    needs neither `pyarrow` nor `tokenizers`. A previous version of this file routed
    around that with a MODULE-scope `pytest.importorskip("pyarrow"/"tokenizers")`, which
    does not do what it looks like it does: `importorskip` raises `Skipped` during
    collection, which skips every test in the file, not just ones that need the train
    group -- so in a dev-group-only environment (every CI job in this repo:
    `.github/workflows/ci.yml`, `code-quality.yml`, `scripts/ci_local.sh` all
    `uv sync --group dev`, never `--group train`) none of this file's tests ran, despite
    none of them needing pyarrow or tokenizers. Loading `_receipt.py` by file path skips
    the package `__init__.py` entirely, so this file needs no skip at all and runs in
    every environment.
    """
    spec = importlib.util.spec_from_file_location("_csd_receipt_under_test", _RECEIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_receipt = _load_receipt_module()
TRAINER_DEFAULT_FIELDS = _receipt.TRAINER_DEFAULT_FIELDS
capture_code_revision = _receipt.capture_code_revision
trainer_defaults = _receipt.trainer_defaults
write_receipt = _receipt.write_receipt


def test_capture_code_revision_reports_this_real_checkout() -> None:
    """Run against the actual repo root (a real git checkout, not a fixture): the SHA is
    a real 40-hex commit id, the branch is a non-empty string, and `dirty` is a bool --
    proving the happy path exercises real `git` subprocesses, not a stub."""
    revision = capture_code_revision(_REPO_ROOT)

    assert revision["git_sha"] != "unknown"
    assert len(revision["git_sha"]) == 40
    assert all(c in "0123456789abcdef" for c in revision["git_sha"])
    assert isinstance(revision["dirty"], bool)
    assert revision["branch"] != "unknown"
    assert revision["branch"] != ""


def test_capture_code_revision_falls_back_when_git_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Outside a git checkout (or with `git` missing from PATH), this must return the
    explicit "unknown" fallback -- `dirty=True`, never raise, and never silently claim a
    clean state it could not verify."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no git binary")),
    )

    revision = capture_code_revision(tmp_path)

    assert revision == {"git_sha": "unknown", "dirty": True, "branch": "unknown"}


def test_capture_code_revision_forces_dirty_true_when_git_commands_fail() -> None:
    """A directory that is not a git repository at all (git exits nonzero, does not
    raise): still the "unknown"/`dirty=True` fallback, not a crash and not a falsely
    clean report."""
    revision = capture_code_revision(Path("/"))

    assert revision == {"git_sha": "unknown", "dirty": True, "branch": "unknown"}


def test_capture_code_revision_ignores_an_ambient_git_dir_pointing_elsewhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression this test file's own CI run tripped over: git sets `GIT_DIR` (and
    `GIT_WORK_TREE`, `GIT_INDEX_FILE`, ...) in ITS OWN environment while running a hook,
    so a hook's nested `git` calls resolve unambiguously -- but that leaks to every
    subprocess the hook's script spawns. `.githooks/pre-push` here runs
    `scripts/ci_local.sh`, which runs `pytest`, which runs THIS test suite -- and before
    the fix this regression test pins, `capture_code_revision`'s subprocess calls
    inherited that ambient `GIT_DIR` by default, so a `repo_root` that is NOT a git
    repository (exactly `test_capture_code_revision_forces_dirty_true_when_git_commands_
    fail`'s case, just above) silently reported the AMBIENT repository -- this project's
    OWN checkout, whatever branch the hook happened to be pushing -- instead of the
    honest "unknown" fallback. This simulates the leak directly rather than relying on
    happening to run inside a hook: `GIT_DIR` set to THIS real repo's actual git-dir,
    `repo_root` an unrelated, definitely-not-a-repo `tmp_path`.
    """
    real_git_dir = subprocess.run(
        ["git", "rev-parse", "--absolute-git-dir"],  # noqa: S607
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert real_git_dir, "could not resolve this checkout's own git-dir to simulate with"
    monkeypatch.setenv("GIT_DIR", real_git_dir)

    revision = capture_code_revision(tmp_path)

    assert revision == {"git_sha": "unknown", "dirty": True, "branch": "unknown"}


def test_write_receipt_stamps_code_revision_and_writes_the_file(tmp_path: Path) -> None:
    receipt: dict = {"region": "test", "held_out": {"recall@1": 0.5}}

    path = write_receipt(receipt, tmp_path, "test-20260101T000000Z.json", repo_root=_REPO_ROOT)

    assert path.is_file()
    assert receipt["receipt_path"] == str(path)
    assert "code_revision" in receipt
    assert set(receipt["code_revision"]) == {"git_sha", "dirty", "branch"}
    import json

    on_disk = json.loads(path.read_text())
    assert "code_revision" in on_disk


def test_write_receipt_refuses_when_capture_returns_none(tmp_path: Path) -> None:
    """The regression this commit fixes: a receipt writer that stamped `code_revision`
    only when capture happened to succeed would let a broken capture function -- one that
    returns `None`, the mutation this test constructs by hand -- through silently, and the
    receipt would land on disk missing the block entirely. `write_receipt` must raise
    instead, and must not write anything."""
    receipt: dict = {"region": "test", "held_out": {"recall@1": 0.5}}
    out_dir = tmp_path / "receipts"

    with pytest.raises(RuntimeError, match="code_revision"):
        write_receipt(
            receipt,
            out_dir,
            "test-20260101T000000Z.json",
            capture=lambda repo_root: None,
        )

    assert not out_dir.exists() or list(out_dir.iterdir()) == []
    assert "code_revision" not in receipt
    assert "receipt_path" not in receipt


def test_write_receipt_refuses_when_capture_returns_an_empty_dict(tmp_path: Path) -> None:
    """`{}` is exactly as falsy as `None` here, and just as much a broken capture."""
    receipt: dict = {"region": "test"}

    with pytest.raises(RuntimeError, match="code_revision"):
        write_receipt(receipt, tmp_path, "test-20260101T000000Z.json", capture=lambda repo_root: {})


class _FieldsOnly:
    """A stand-in config exposing only `steps`/`batch_size`/`lr` -- the shape
    `VLPretrainConfig` actually has: no `bf16`, no `max_len`."""

    steps = 4000
    batch_size = 128
    lr = 1.5e-4


class _AllFields:
    steps = 2000
    batch_size = 256
    lr = 3e-4
    bf16 = True
    max_len = 128


def test_trainer_defaults_pulls_every_field_when_the_config_has_them() -> None:
    result = trainer_defaults(_AllFields())

    assert result == {"steps": 2000, "batch_size": 256, "lr": 3e-4, "bf16": True, "max_len": 128}


def test_trainer_defaults_reports_none_for_a_field_the_config_does_not_define() -> None:
    """The `VLPretrainConfig` shape: `bf16`/`max_len` are not silently omitted from the
    result, and not fabricated from some other config's default -- they read `None`."""
    result = trainer_defaults(_FieldsOnly())

    assert result["steps"] == 4000
    assert result["batch_size"] == 128
    assert result["lr"] == pytest.approx(1.5e-4)
    assert result["bf16"] is None
    assert result["max_len"] is None
    assert set(result) == set(TRAINER_DEFAULT_FIELDS)
