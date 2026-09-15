"""Tests for lane IC-11 -- `cogsyndelta.contracts.region_spec`'s interconnect check.

INTERCONNECT-MODULE-SPEC.md section 2.1, Table 10, row IC-11: "replace the
single-stream_dim check with workspace_dim plus per-region token_dim". This module has
no tensors, no masks and nothing random -- it is a construction-time validation over
plain dataclasses -- so "shapes" here means the accepted/rejected `workspace_dim` and
`token_dim` values, and "determinism" means construction is a pure function of its
arguments (asserted directly; there is no seed to fix, since nothing here draws random
numbers).

Covers: `RegionSpec.token_dim` reads `stream_dim`; `MindSpec.workspace_dim` defaults to
`None` and leaves a mind that never sets it unaffected (`tests/test_region_spec.py`
already pins the legacy `stream_dim` warning path); a positive `workspace_dim` with
mismatched region `token_dim`s builds with no warning at all (DEC-15 -- no uniformity is
expected, so nothing should even warn); a non-positive `workspace_dim` is rejected at
construction, before any interconnect adapter could be built from it; `workspace_dim`
round-trips through `to_json`/`from_json` and is omitted-safe from older JSON.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from cogsyndelta.contracts.region_spec import MindSpec, RegionSpec


def _region(name: str, stream_dim: int = 64, **kw: object) -> RegionSpec:
    base: dict[str, object] = {"kind": "residual_mlp", "stream_dim": stream_dim, "hidden_dim": 128}
    base.update(kw)
    return RegionSpec(name=name, **base)  # type: ignore[arg-type]


def test_token_dim_reads_stream_dim() -> None:
    """`token_dim` is the interconnect spec's Table 1 name for the same width this
    dataclass has always stored as `stream_dim` -- not a separate value to keep in
    sync."""
    region = _region("visual", stream_dim=384)
    assert region.token_dim == 384 == region.stream_dim


def test_workspace_dim_defaults_to_none_and_does_not_affect_a_legacy_mind() -> None:
    """A mind that never mentions the interconnect module is unaffected: no
    `workspace_dim`, no IC-11 check runs, and the legacy `stream_dim` warning path
    (pinned by `tests/test_region_spec.py`) is the only one exercised."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = MindSpec(stream_dim=64, regions=[_region("a", stream_dim=64)])
    assert spec.workspace_dim is None


def test_workspace_dim_with_uniform_token_dims_builds_clean() -> None:
    spec = MindSpec(stream_dim=64, regions=[_region("a", stream_dim=512)], workspace_dim=512)
    assert spec.workspace_dim == 512


def test_workspace_dim_with_mismatched_token_dims_builds_with_no_warning() -> None:
    """DEC-15 ('Widths'): regions no longer share one stream width with the workspace
    at all, so a 256-wide region and a 384-wide region under a 512-wide workspace is
    the expected, ordinary case -- not even a warning, unlike the legacy `stream_dim`
    uniformity check this lane replaces for a workspace-declared mind."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = MindSpec(
            stream_dim=999,  # deliberately irrelevant to workspace_dim -- DEC-15
            regions=[_region("language", stream_dim=256), _region("visual", stream_dim=384)],
            workspace_dim=512,
        )
    assert [r.token_dim for r in spec.regions] == [256, 384]
    assert spec.workspace_dim == 512


@pytest.mark.parametrize("bad_workspace_dim", [0, -1, -512])
def test_non_positive_workspace_dim_rejected(bad_workspace_dim: int) -> None:
    with pytest.raises(ValueError, match="workspace_dim"):
        MindSpec(stream_dim=64, regions=[_region("a")], workspace_dim=bad_workspace_dim)


def test_workspace_dim_check_runs_after_duplicate_and_top_k_checks() -> None:
    """Malformed-mind checks stay layered: a duplicate name is still caught before the
    IC-11 check ever runs, so the more specific of two problems is the one reported."""
    with pytest.raises(ValueError, match="duplicate"):
        MindSpec(
            stream_dim=64,
            regions=[_region("a"), _region("a")],
            workspace_dim=-1,
        )


def test_workspace_dim_round_trips_through_json(tmp_path: Path) -> None:
    spec = MindSpec(
        stream_dim=64,
        regions=[_region("language", stream_dim=256)],
        workspace_dim=512,
    )
    path = tmp_path / "mind.json"
    spec.to_json(path)
    reloaded = MindSpec.from_json(path)
    assert reloaded == spec
    assert reloaded.workspace_dim == 512


def test_workspace_dim_absent_from_json_loads_as_none() -> None:
    """Older config written before lane IC-11 has no `workspace_dim` key at all; it
    must still load, as a mind with no interconnect module involved."""
    spec = MindSpec.from_dict(
        {
            "stream_dim": 64,
            "regions": [{"name": "a", "kind": "residual_mlp", "stream_dim": 64, "hidden_dim": 128}],
        }
    )
    assert spec.workspace_dim is None


def test_construction_is_deterministic() -> None:
    """No tensors, no masks, nothing random in this module -- construction from the
    same arguments must produce equal specs every time, not merely specs that pass the
    same checks."""
    kwargs: dict[str, object] = {
        "stream_dim": 64,
        "regions": [_region("language", stream_dim=256), _region("visual", stream_dim=384)],
        "workspace_dim": 512,
    }
    first = MindSpec(**kwargs)  # type: ignore[arg-type]
    second = MindSpec(**kwargs)  # type: ignore[arg-type]
    assert first == second
