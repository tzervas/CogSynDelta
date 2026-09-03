"""Import hygiene guard (N2): importing `cogsyndelta.quant.ptq` or
`cogsyndelta.regions._checkpoint` must not drag in `cogsyndelta.regions.pretrain` --
and, transitively, `tokenizers` -- merely because both live under the
`cogsyndelta.regions` package and Python always runs a package's `__init__.py`
before any of its submodules.

WHY THIS MATTERS
`scripts/ci_local.sh` -- the pre-push gate -- deliberately runs `uv sync --group
dev`, which EXCLUDES the `train` dependency group; `tokenizers` and `pyarrow` are not
installed in that venv (see pyproject.toml's group split and that script's own header
comment: "pulling pyarrow + tokenizers into every lint job costs minutes for
nothing"). Before this fix, `cogsyndelta/regions/__init__.py` unconditionally did
`from cogsyndelta.regions.pretrain import (...)` at module scope, and Python executes
a package's `__init__.py` as a side effect of importing ANY of its submodules --
so `import cogsyndelta.regions._checkpoint` (which `cogsyndelta.quant.ptq`'s
`save_packed_artifact` imports lazily, and which `tests/test_publish_checkpoint.py` /
`tests/test_quant_ptq.py` import directly) transitively imported `regions.pretrain`,
which imports `tokenizers` at module level (`src/cogsyndelta/regions/pretrain.py`).
Neither test file uses `tokenizers` at all; both failed in ci_local.sh's venv with
`ModuleNotFoundError: tokenizers` regardless.

WHY A SUBPROCESS, NOT AN IN-PROCESS ASSERTION
By the time this test file runs under a normal `pytest tests/`, `cogsyndelta.regions`
(and, in a venv where it is installed, `tokenizers` itself) is very likely already
imported and cached in `sys.modules` -- other test modules import
`cogsyndelta.regions.pretrain` directly. Asserting anything in-process would only be
testing that cache, not a fresh import. A subprocess, with a `sys.meta_path` finder
installed BEFORE `cogsyndelta` is ever touched, is the only way to observe what a
genuinely clean import does -- and it is also what makes this test meaningful
regardless of whether `tokenizers` happens to be installed in whatever venv actually
runs this suite: the finder makes it unavailable inside that one subprocess either way.

WHY THE FINDER ALSO PROVES ITS OWN GUARD CAN FAIL
`test_pretrain_import_still_fails_when_tokenizers_blocked` is not incidental: without
it, the two "succeeds" assertions below could pass for the wrong reason -- a finder
that silently failed to block anything (a typo in `BLOCKED`, a `sys.meta_path` bug, a
mismatched module name) would make every import in this file succeed regardless of
whether `cogsyndelta/regions/__init__.py` is actually lazy, and neither of the
"succeeds" tests would ever notice. Proving `regions.pretrain` -- which genuinely
does need `tokenizers` -- still fails under the SAME finder is what proves the finder
itself works, and therefore that the two passing tests mean something.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

# A minimal sys.meta_path finder that makes `tokenizers` and `pyarrow` (root package
# or any submodule) look uninstalled to everything imported after it, without
# actually needing them uninstalled from whatever venv runs this test. Mirrors
# ci_local.sh's real exclusion (the `train` dependency group), so a pass here means
# the same thing a green ci_local.sh run means for this specific hazard.
_BLOCKER_PREAMBLE = """
import sys

_BLOCKED = {"tokenizers", "pyarrow"}


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _BLOCKED:
            raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}", name=name)
        return None


sys.meta_path.insert(0, _Blocker())
"""


def _run_blocked_import(statement: str) -> subprocess.CompletedProcess[str]:
    """Run `statement` in a fresh subprocess with `tokenizers`/`pyarrow` blocked and
    the worktree's `src/` on the path -- and nothing else assumed about the
    environment (a full `os.environ` copy, so the interpreter's own venv resolves
    exactly as it would for any other invocation)."""
    script = _BLOCKER_PREAMBLE + "\n" + statement + "\n"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=60,
        check=False,  # returncode is asserted explicitly by every caller
    )


def test_ptq_import_survives_tokenizers_blocked() -> None:
    """`cogsyndelta.quant.ptq`'s own module-level imports never touch
    `cogsyndelta.regions` at all (its use of `_checkpoint` is inside a function
    body, imported lazily) -- true before and after this fix. Pinned here as a
    baseline the two tests below build on, and as a regression guard against a
    future edit hoisting that import to module scope."""
    result = _run_blocked_import("import cogsyndelta.quant.ptq")
    assert result.returncode == 0, result.stderr


def test_checkpoint_import_survives_tokenizers_blocked() -> None:
    """The actual N2 regression: `import cogsyndelta.regions._checkpoint` alone
    used to run `cogsyndelta/regions/__init__.py`'s eager `from
    cogsyndelta.regions.pretrain import (...)`, which imports `tokenizers` -- so
    this import raised `ModuleNotFoundError: tokenizers` before this fix (see this
    module's docstring), even though `_checkpoint.py` itself never touches
    `tokenizers`. Must succeed now that `cogsyndelta/regions/__init__.py` is lazy."""
    result = _run_blocked_import("import cogsyndelta.regions._checkpoint")
    assert result.returncode == 0, result.stderr


def test_pretrain_import_still_fails_when_tokenizers_blocked() -> None:
    """The blocker-works guard (see module docstring): `regions.pretrain` genuinely
    imports `tokenizers` at module level and must still fail under the same finder
    the two tests above run under -- proving those two tests would have caught a
    finder that silently blocked nothing."""
    result = _run_blocked_import("import cogsyndelta.regions.pretrain")
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr
    assert "tokenizers" in result.stderr


def test_regions_public_api_still_resolves_lazily() -> None:
    """The laziness fix must not drop any name `cogsyndelta.regions.__all__`
    exported -- `scripts/csd-train-all.py` does `from cogsyndelta.regions import
    PretrainConfig, pretrain_region` (the one caller in the tree that imports from
    the PACKAGE rather than a submodule), which only works if `__getattr__`
    resolves every public name on demand. Run in a subprocess for the same reason
    as the tests above: a name resolved only because some earlier test already
    imported `regions.pretrain` and populated `sys.modules` would be a false pass.
    """
    result = _run_blocked_import(
        "from cogsyndelta.regions import ("
        "PretrainConfig, TextEncoder, TextEncoderConfig, evaluate, evaluate_graded, "
        "info_nce, load_graded_pairs, load_pairs, pretrain_region)"
    )
    # This statement legitimately needs tokenizers (PretrainConfig etc. live in
    # regions.pretrain), so it is expected to fail in THIS blocked subprocess -- the
    # point here is narrower: it must fail for the *same* reason a direct `import
    # cogsyndelta.regions.pretrain` fails (tokenizers missing), not an AttributeError
    # from a `__getattr__` that forgot one of these names.
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr
    assert "tokenizers" in result.stderr
    assert "AttributeError" not in result.stderr


def test_regions_public_api_resolves_when_tokenizers_available() -> None:
    """The positive case for the same property, unblocked: every `__all__` name
    resolves via `__getattr__` to the real object `cogsyndelta.regions.pretrain` /
    `cogsyndelta.regions.text_encoder` define -- not merely "importable", but the
    identical attribute a direct submodule import would give a caller.

    This is an in-process, UNBLOCKED import of `regions.pretrain`, which genuinely
    needs `tokenizers` -- exactly the dependency scripts/ci_local.sh's isolated venv
    deliberately excludes (the `train` group). Skip rather than fail there, the same
    way `tests/test_guards_can_fail.py` and others already skip when an optional
    dependency for the case under test is absent; the two subprocess tests above are
    what actually pin the N2 behaviour regardless of which venv runs this file."""
    pytest.importorskip("tokenizers")
    import cogsyndelta.regions as regions_pkg
    from cogsyndelta.regions import pretrain, text_encoder

    for name in regions_pkg.__all__:
        assert hasattr(regions_pkg, name), f"cogsyndelta.regions has no attribute {name!r}"
    assert regions_pkg.PretrainConfig is pretrain.PretrainConfig
    assert regions_pkg.pretrain_region is pretrain.pretrain_region
    assert regions_pkg.evaluate is pretrain.evaluate
    assert regions_pkg.evaluate_graded is pretrain.evaluate_graded
    assert regions_pkg.load_graded_pairs is pretrain.load_graded_pairs
    assert regions_pkg.load_pairs is pretrain.load_pairs
    assert regions_pkg.TextEncoder is text_encoder.TextEncoder
    assert regions_pkg.TextEncoderConfig is text_encoder.TextEncoderConfig
    assert regions_pkg.info_nce is text_encoder.info_nce


def test_regions_unknown_attribute_still_raises_attribute_error() -> None:
    """`__getattr__` must fall through to a normal `AttributeError` for anything not
    in its lazy-name tables -- not silently return `None`, and not mask a genuine
    typo as some other exception."""
    import cogsyndelta.regions as regions_pkg

    with pytest.raises(AttributeError):
        _ = regions_pkg.definitely_not_a_real_export
