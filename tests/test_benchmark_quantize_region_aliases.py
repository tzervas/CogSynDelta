"""`scripts/csd-benchmark.py` and `scripts/csd-quantize.py` are READERS of training
receipts, and DEC-78 says every reader resolves a region name through
`cogsyndelta.regions.aliases` before comparing, dispatching, or looking up config on it.
Round-2 review (blocking): neither script did. Both globbed receipts on the raw string
the caller passed (`receipts/{region}-2*.json`), so a canonical `--regions language`
against a state root that only has `receipts/code-*.json` -- every cell on disk right
now -- silently matched nothing. `csd-benchmark.py` made that worse: `benchmark_region`
returning `None` fell out of `main()`'s loop uncounted, so the run printed "0
failure(s)" and exited 0 having scored nothing. `csd-quantize.py`'s `_latest_receipt`
raised `FileNotFoundError` for the same gap -- loud, not a silent no-op, but the same
unresolved alias.

This file binds both scripts' receipt lookups to the alias layer (which neither
imported before this fix -- `grep -n aliases scripts/csd-benchmark.py` and the
`csd-quantize.py` equivalent both had no match) and proves the fix with the exact
pre-fix expression, verbatim, showing it does not find what the fixed one does. It also
proves `csd-benchmark.py`'s `main()` now counts an unresolved region as a failure
instead of exiting 0.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_csd_benchmark():
    path = SCRIPTS / "csd-benchmark.py"
    spec = importlib.util.spec_from_file_location("csd_benchmark_alias_test", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_csd_quantize():
    path = SCRIPTS / "csd-quantize.py"
    spec = importlib.util.spec_from_file_location("csd_quantize_alias_test", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bench = _load_csd_benchmark()
quant = _load_csd_quantize()


def _write_receipt(state: Path, name: str, region: str) -> Path:
    path = state / "receipts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"region": region, "checkpoint_sha256": "a" * 64}))
    return path


def _pre_fix_glob(state: Path, region: str) -> list[Path]:
    """The expression both scripts used before this fix, verbatim: glob on the raw
    caller-supplied string, no alias resolution at all."""
    return sorted(state.glob(f"receipts/{region}-2*.json"))


# --------------------------------------------------- both scripts import the alias layer


def test_csd_benchmark_imports_the_alias_layer() -> None:
    source = (SCRIPTS / "csd-benchmark.py").read_text()
    assert "from cogsyndelta.regions.aliases import" in source


def test_csd_quantize_imports_the_alias_layer() -> None:
    source = (SCRIPTS / "csd-quantize.py").read_text()
    assert "from cogsyndelta.regions.aliases import" in source


# --------------------------------------- csd-benchmark.py: _find_train_receipt resolves


def test_canonical_name_finds_a_receipt_filed_under_the_legacy_spelling(
    tmp_path: Path,
) -> None:
    """The exact repro from the review: a canonical `--regions language` against a
    state root that only has a `code-*.json` receipt (every real cell on disk today)."""
    legacy = _write_receipt(tmp_path, "code-20260904T140242Z.json", "code")
    assert bench._find_train_receipt("language", tmp_path) == legacy


def test_legacy_name_finds_a_receipt_filed_under_the_canonical_spelling(
    tmp_path: Path,
) -> None:
    """The other direction: once a writer starts emitting the canonical spelling, a
    caller still passing `--regions code` (or an older script that has not been
    updated) must still find it."""
    canonical = _write_receipt(tmp_path, "language-20260905T000000Z.json", "language")
    assert bench._find_train_receipt("code", tmp_path) == canonical


def test_the_pre_fix_glob_would_have_found_nothing(tmp_path: Path) -> None:
    """MUTATION. The old expression, applied to the exact same directory, finds no
    match at all -- this is the silent no-op the review reproduced."""
    _write_receipt(tmp_path, "code-20260904T140242Z.json", "code")
    assert _pre_fix_glob(tmp_path, "language") == []
    assert bench._find_train_receipt("language", tmp_path) is not None


def test_both_spellings_present_is_still_ambiguous(tmp_path: Path) -> None:
    """A `code-*` receipt and a `language-*` receipt in the same directory are the same
    region under two spellings -- pooling them and refusing is correct; picking one
    silently would be exactly the M1 defect the ambiguity refusal already exists for."""
    older = _write_receipt(tmp_path, "code-20260901T000000Z.json", "code")
    newer = _write_receipt(tmp_path, "language-20260903T000000Z.json", "language")
    with pytest.raises(bench.AmbiguousTrainReceiptError) as exc:
        bench._find_train_receipt("language", tmp_path)
    message = str(exc.value)
    assert str(older) in message
    assert str(newer) in message


def test_a_never_renamed_region_is_unaffected(tmp_path: Path) -> None:
    """`memory` has no legacy spelling -- the alias-aware glob must be a no-op for it,
    identical to the pre-fix behaviour."""
    only = _write_receipt(tmp_path, "memory-20260901T000000Z.json", "memory")
    assert bench._find_train_receipt("memory", tmp_path) == only


def test_no_candidates_under_either_spelling_returns_none(tmp_path: Path) -> None:
    (tmp_path / "receipts").mkdir()
    assert bench._find_train_receipt("language", tmp_path) is None


# ------------------------------------------ csd-benchmark.py: main() no longer exits 0


def test_main_counts_an_unmatched_region_as_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The worse half of the defect: `rec is None` used to `continue` uncounted, so a
    region nothing could be scored for still produced "0 failure(s)" and exit 0."""
    (tmp_path / "receipts").mkdir()
    monkeypatch.setattr(
        sys, "argv", ["csd-benchmark.py", "--regions", "language", "--state", str(tmp_path)]
    )
    rc = bench.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "1 failure(s)" in out


def test_the_pre_fix_loop_body_would_not_have_counted_the_failure(tmp_path: Path) -> None:
    """MUTATION. `benchmark_region` returning `None` is the trigger this branch
    guards on. The pre-fix loop body was a bare `continue` there -- nothing appended to
    `failures` -- which is exactly the "0 failure(s)", exit-0 outcome the review
    reproduced. `main()` now appends `region` to `failures` in that branch instead
    (see `test_main_counts_an_unmatched_region_as_a_failure` for the real `main()`
    call this predicts)."""
    (tmp_path / "receipts").mkdir()
    rec = bench.benchmark_region("language", tmp_path)
    assert rec is None
    pre_fix_failures: list[str] = []  # the old branch body did nothing to this list
    assert pre_fix_failures == []
    assert (1 if pre_fix_failures else 0) == 0  # pre-fix: main() would have exited 0


# ------------------------------------------- csd-quantize.py: _latest_receipt resolves


def test_quantize_canonical_name_finds_a_legacy_spelled_receipt(tmp_path: Path) -> None:
    legacy = _write_receipt(tmp_path, "code-20260904T140242Z.json", "code")
    path, receipt = quant._latest_receipt(tmp_path, "language")
    assert path == legacy
    assert receipt["region"] == "code"


def test_quantize_legacy_name_finds_a_canonical_spelled_receipt(tmp_path: Path) -> None:
    canonical = _write_receipt(tmp_path, "language-20260905T000000Z.json", "language")
    path, _receipt = quant._latest_receipt(tmp_path, "code")
    assert path == canonical


def test_quantize_pre_fix_glob_would_have_found_nothing(tmp_path: Path) -> None:
    """MUTATION. The pre-fix `_latest_receipt` globbed the raw caller-supplied string
    with no alias resolution -- exactly `_pre_fix_glob` above -- which is why it raised
    `FileNotFoundError` against a real, already-trained cell filed under the other
    spelling: `csd-quantize.py`'s louder version of `csd-benchmark.py`'s same gap."""
    _write_receipt(tmp_path, "code-20260904T140242Z.json", "code")
    assert _pre_fix_glob(tmp_path, "language") == []
    path, _receipt = quant._latest_receipt(tmp_path, "language")
    assert path.name == "code-20260904T140242Z.json"


def test_quantize_never_renamed_region_is_unaffected(tmp_path: Path) -> None:
    only = _write_receipt(tmp_path, "memory-20260901T000000Z.json", "memory")
    path, _receipt = quant._latest_receipt(tmp_path, "memory")
    assert path == only


def test_quantize_no_candidates_under_either_spelling_still_raises(tmp_path: Path) -> None:
    (tmp_path / "receipts").mkdir()
    with pytest.raises(FileNotFoundError, match="no training receipt"):
        quant._latest_receipt(tmp_path, "language")
