"""Every REGIONS consumer must resolve through one accessor, and the accessor must be
able to reject the shape that broke them.

Commit 0786a77 turned every `REGIONS[name]` entry into a 4-tuple (added the graded-gate
field, see `scripts/csd-train-all.py`'s `GradedSpec`) without touching
`scripts/csd-quantize.py` or `scripts/csd-benchmark.py`, which each still unpacked
`REGIONS[region]` with their own fixed arity (`sources, _note = ...`). Both scripts now
raised `ValueError: too many values to unpack` before doing anything useful -- and
nothing caught it, because neither script had ever been imported as a module in this
test suite (see tests/test_guards_can_fail.py's rule: an untested path is a comment with
a function signature, not a guarantee).

`scripts/csd-train-all.py`'s `region_spec()` is now the ONE place a `REGIONS` entry is
unpacked; `csd-quantize.py` and `csd-benchmark.py` both resolve a region's sources
through it instead of unpacking `REGIONS[region]` themselves. This file asserts:
  (a) every `REGIONS` entry resolves through `region_spec`;
  (b) `csd-quantize.py` and `csd-benchmark.py` actually use that same accessor, at the
      same call sites a real quantize/benchmark run hits, for every region;
  (c) the accessor itself can fail -- constructing a `REGIONS` entry in the OLD
      (pre-graded-gate) shape and asserting `region_spec` rejects it rather than
      silently misreading it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(module_name: str, filename: str):
    """Import one of the hyphenated `scripts/` files as a module.

    Same pattern as tests/test_reserved_corpus_guard.py: a hyphenated filename is not a
    valid `import` target, so every consumer in this repo loads it via
    `importlib.util.spec_from_file_location`. A distinct `module_name` per call keeps
    `sys.modules` entries from colliding with other test files doing the same thing.
    """
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


train_all = _load("csd_train_all_region_spec_consumers_test", "csd-train-all.py")
quantize = _load("csd_quantize_region_spec_consumers_test", "csd-quantize.py")
benchmark = _load("csd_benchmark_region_spec_consumers_test", "csd-benchmark.py")


# ---------------------------------------------------------------------------------------
# (a) Every REGIONS entry resolves through the accessor.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(train_all.REGIONS))
def test_every_region_resolves_through_the_accessor(name: str) -> None:
    entry = train_all.region_spec(name)
    assert isinstance(entry, train_all.RegionEntry)
    raw = train_all.REGIONS[name]
    assert entry.sources == raw[0]
    assert entry.note == raw[1]
    assert entry.default_max_len == raw[2]
    assert entry.graded == raw[3]
    # graded is either undeclared (None) or a full (glob, columns, name) GradedSpec --
    # never a partially-filled shape that would blow up later inside run_region.
    if entry.graded is not None:
        glob, columns, graded_name = entry.graded
        assert isinstance(glob, str) and glob
        assert isinstance(columns, tuple) and len(columns) == 3
        assert isinstance(graded_name, str) and graded_name


def test_region_spec_shape_matches_regions_annotation() -> None:
    """`REGIONS` today has exactly one region (`compress`) that declares a graded gate --
    if this ever drops to zero, the graded-gate branch above stops being exercised by
    (a) at all, silently. Pin the fact so that regression is visible here first."""
    graded = {name for name, entry in train_all.REGIONS.items() if entry[3] is not None}
    assert graded == {"compress"}


# ---------------------------------------------------------------------------------------
# (b) csd-quantize.py and csd-benchmark.py resolve every region's sources through the
#     SAME accessor, at the same call sites a real run hits.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(train_all.REGIONS))
def test_quantize_resolves_sources_via_the_accessor(name: str) -> None:
    """Exercises exactly the line `quantize_text_region` runs before it ever needs a
    checkpoint or a receipt: `spec["region_spec"](region).sources`. Before this fix, the
    equivalent line was `sources, _note = spec["REGIONS"][region]`, which raised
    `ValueError` for every region once `REGIONS` became a 4-tuple.
    """
    spec = quantize._load_regions_spec()
    sources = spec["region_spec"](name).sources
    assert sources == train_all.REGIONS[name][0]


@pytest.mark.parametrize("name", sorted(train_all.REGIONS))
def test_benchmark_resolves_sources_via_the_accessor(name: str) -> None:
    """Same as above for `benchmark_region`'s `spec["region_spec"](region).sources`."""
    spec = benchmark._regions_spec()
    sources = spec["region_spec"](name).sources
    assert sources == train_all.REGIONS[name][0]


def test_quantize_and_benchmark_expose_region_spec_not_just_raw_regions() -> None:
    """Both loader dicts must carry the accessor itself, not merely the raw `REGIONS`
    mapping -- a future edit that re-adds a bare `spec["REGIONS"][region]` unpack at a
    NEW call site is exactly the failure mode this fix exists to close off, and that
    requires the accessor to actually be reachable from the loaded module dict."""
    assert callable(quantize._load_regions_spec()["region_spec"])
    assert callable(benchmark._regions_spec()["region_spec"])


# ---------------------------------------------------------------------------------------
# (c) The guard can fail: construct the OLD (pre-graded-gate) shape and assert
#     region_spec rejects it with a clear error, rather than silently misreading it.
# ---------------------------------------------------------------------------------------


def test_accessor_rejects_the_old_two_tuple_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shape that broke csd-quantize.py/csd-benchmark.py pre-fix, reconstructed
    deliberately: `REGIONS["code"]` overwritten with the OLD `(sources, note)` 2-tuple,
    missing `default_max_len` and the graded-gate field entirely. If `region_spec` ever
    stopped validating shape -- e.g. someone "simplifies" it back to a bare unpack --
    this is what would start failing.
    """
    old_shape_sources = train_all.REGIONS["code"][0]
    monkeypatch.setitem(
        train_all.REGIONS,
        "code",
        (old_shape_sources, "docstring <-> function; NOT next-token over GitHub"),
    )
    with pytest.raises(ValueError, match="4-tuple"):
        train_all.region_spec("code")


def test_accessor_rejects_the_pre_graded_gate_three_tuple_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shape REGIONS actually had immediately before 0786a77 (sources, note,
    default_max_len) -- one field short of today's graded-gate 4-tuple. This is the more
    realistic "old shape" than a bare 2-tuple, and the accessor must reject it too."""
    old_shape_sources, old_note, old_max_len, _graded = train_all.REGIONS["code"]
    monkeypatch.setitem(train_all.REGIONS, "code", (old_shape_sources, old_note, old_max_len))
    with pytest.raises(ValueError, match="4-tuple"):
        train_all.region_spec("code")


def test_accessor_rejects_a_missing_region() -> None:
    with pytest.raises(KeyError):
        train_all.region_spec("not-a-real-region")


def test_accessor_rejects_a_non_tuple_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not every wrong shape is a short tuple -- a dict-shaped entry (the VL_REGIONS /
    CLASSIFY_REGIONS convention) accidentally assigned into REGIONS must also be caught,
    not just an under-length tuple."""
    monkeypatch.setitem(train_all.REGIONS, "code", {"sources": [], "note": "wrong shape"})
    with pytest.raises(ValueError, match="4-tuple"):
        train_all.region_spec("code")
