"""The compose-reservation guard must fire, not merely exist.

DEC-23 (docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md §5.2, applied
docs/design/CORPUS-CONTRACT.md §2.4): `codeparrot/apps` and `deepmind/code_contests` are
reserved for the composed model, not any region -- they are the only licence-clean paired
natural-language-problem <-> implementation source in the tree, and allocation is
irreversible (once a region trains on a row, it is permanently ineligible for compose).

Per tests/test_guards_can_fail.py's own lesson, a guard with no failing-case test is a
comment with a function signature. This file constructs the exact condition
`scripts/csd-train-all.py`'s `RESERVED_FOR_COMPOSE` guard exists to catch -- a region
resolving a shard under a reserved corpus directory -- at both the unit level
(`_refuse_reserved_shards`) and the entry-point level (`run_region`, dry-run), and asserts
it raises rather than silently training.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu


def _load_csd_train_all():
    """Import scripts/csd-train-all.py the same way scripts/csd-quantize.py does.

    The hyphenated filename is not a valid module name, so every consumer in this repo
    (csd-quantize.py, csd-benchmark.py) loads it via `importlib.util.spec_from_file_location`
    rather than `import` -- this mirrors that convention rather than inventing a new one.
    """
    path = Path(__file__).resolve().parents[1] / "scripts" / "csd-train-all.py"
    spec = importlib.util.spec_from_file_location("csd_train_all_guard_test", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_csd_train_all()


# ---------------------------------------------------------------------------------------
# Unit level -- `_refuse_reserved_shards` directly.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shard",
    [
        # The shared-mount layout `_shards()` resolves REGIONS globs against (CORPUS).
        "/mnt/fleet-datasets/csd/region/code/apps/train.parquet",
        "/mnt/fleet-datasets/csd/region/code/code_contests/train.parquet",
        # The bulk-array layout `csd-corpus-expand.py`'s `Dataset.local` actually fetched
        # them to (LOCAL_CORPUS / REGION_CORPUS_ROOT overrides) -- verified present on
        # gpu5080 at exactly these two paths.
        "/bulk/csd-corpus/code/apps/train.parquet",
        "/bulk/csd-corpus/code/code_contests/train.parquet",
    ],
)
def test_reserved_shard_is_refused_regardless_of_root(shard: str) -> None:
    """A resolved shard under either reserved directory raises, on any root it lives under."""
    with pytest.raises(mod.ReservedSourceError, match="reserved"):
        mod._refuse_reserved_shards("code", "region/code/**/*.parquet", [shard])


def test_reserved_shard_among_clean_ones_still_raises() -> None:
    """One reserved shard in an otherwise-clean list is enough -- this is not a majority vote."""
    shards = [
        "/mnt/fleet-datasets/csd/region/code/codesearchnet-python/00000.parquet",
        "/mnt/fleet-datasets/csd/region/code/codesearchnet-python/00001.parquet",
        "/mnt/fleet-datasets/csd/region/code/apps/train.parquet",
    ]
    with pytest.raises(mod.ReservedSourceError):
        mod._refuse_reserved_shards("code", "region/code/**/*.parquet", shards)


def test_unrelated_sources_are_not_flagged() -> None:
    """The negative control: a guard that fires on everything is as useless as one that
    fires on nothing (see tests/test_guards_can_fail.py's slot-value-template case).

    `codesearchnet-python` is the region's actual, unreserved source. A path component
    that merely CONTAINS "apps" as a substring (`webapps`) must not match either -- the
    guard keys on exact path segments, not substrings, so a future dataset that happens to
    share a substring with a reserved name is not silently swept in.
    """
    clean_shards = [
        "/mnt/fleet-datasets/csd/region/code/codesearchnet-python/00000.parquet",
        "/bulk/csd-corpus/reason/gsm8k-main/train.parquet",
        "/bulk/csd-corpus/classify/webapps-feedback/train.parquet",
    ]
    mod._refuse_reserved_shards("code", "region/code/**/*.parquet", clean_shards)  # no raise


# ---------------------------------------------------------------------------------------
# Entry-point level -- `run_region`, the actual thing an operator or `main()` invokes.
# ---------------------------------------------------------------------------------------


def test_code_region_run_refuses_to_start_when_resolved_sources_include_a_reserved_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Construct a `code` run whose resolved sources include a reserved corpus, and assert
    `run_region` raises before training starts -- the shape DEC-23 and
    docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md §5.6 both require ("the gate ... is
    not 'the ledger exists'. It is: a training run seeded with one reserved row REFUSES TO
    START").

    `_shards` is monkeypatched rather than staging real parquet files: the guard is
    checked immediately after shard resolution, before `_schema_mismatch` ever opens a
    file, so a non-existent path is sufficient to prove the refusal fires ahead of any
    file I/O -- and this is exactly what makes it catch a FUTURE `REGIONS["code"]` entry
    that naively globs the reserved directory, not only today's.
    """
    monkeypatch.setattr(
        mod,
        "_shards",
        lambda pattern, root=mod.CORPUS: ["/mnt/fleet-datasets/csd/region/code/apps/train.parquet"],
    )

    with pytest.raises(mod.ReservedSourceError):
        mod.run_region(
            name="code",
            state=tmp_path,
            steps=1,
            batch=1,
            shard_limit=0,
            dry=True,
        )


def test_code_region_dry_run_is_unaffected_when_sources_stay_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The guard's negative control at the entry-point level: an unreserved resolved shard
    must not trip it, so the dry run reaches its normal `no usable sources` / MISSING path
    instead of a spurious refusal. (The path below does not exist on this host, so this
    exercises the MISSING branch, not a full resolve -- the point is only that
    `ReservedSourceError` is never raised.)
    """
    monkeypatch.setattr(
        mod,
        "_shards",
        lambda pattern, root=mod.CORPUS: [
            "/mnt/fleet-datasets/csd/region/code/codesearchnet-python/00000.parquet"
        ],
    )

    # No ReservedSourceError -- may legitimately return None (schema/columns unresolved
    # against a file that does not exist on this host), which is a different, unrelated
    # code path this test does not assert on.
    try:
        mod.run_region(name="code", state=tmp_path, steps=1, batch=1, shard_limit=0, dry=True)
    except mod.ReservedSourceError:
        pytest.fail("clean source incorrectly refused as reserved")


# ---------------------------------------------------------------------------------------
# Entry-point level -- `run_vl_region` and `run_classify_region`.
#
# `_refuse_reserved_shards` is called from `run_region`'s source loop, but until this
# guard was added `run_vl_region` (train/probe_eval/transfer resolution) and
# `run_classify_region` (shard resolution against `LOCAL_CORPUS` -- the exact root
# `apps`/`code_contests` were fetched under) resolved shards with no check at all. Same
# shape as the `run_region` tests above: monkeypatch `_shards` to return a reserved path,
# assert the runner refuses before `dry=True` would otherwise return `None` cleanly.
# ---------------------------------------------------------------------------------------


def test_vl_region_run_refuses_to_start_when_resolved_sources_include_a_reserved_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`run_vl_region` must refuse a reserved shard exactly like `run_region` does.

    Regression guard for the gap this file's module docstring now (accurately) describes
    as closed: before this test existed, `run_vl_region` never called
    `_refuse_reserved_shards` at all, so a `VL_REGIONS` glob resolving into `apps` or
    `code_contests` would train silently.

    `corpus_source` is monkeypatched to a placeholder name: W7v-cfg's OD-4 gate
    (`VisualCorpusUnsetError`) fires BEFORE shard resolution when it is unset, which
    would otherwise mask the reserved-shard guard this test targets.
    """
    monkeypatch.setitem(mod.VL_REGIONS["vl_latent"], "corpus_source", "test-corpus")
    monkeypatch.setitem(mod.VL_REGIONS["vl_latent"], "manifest", None)
    monkeypatch.setattr(
        mod,
        "_shards",
        lambda pattern, root=mod.CORPUS: ["/mnt/fleet-datasets/csd/region/code/apps/train.parquet"],
    )

    with pytest.raises(mod.ReservedSourceError):
        mod.run_vl_region(name="vl_latent", state=tmp_path, steps=1, batch=1, dry=True)


def test_vl_region_dry_run_is_unaffected_when_sources_stay_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Negative control: an unreserved resolved shard must not trip the VL guard either."""
    monkeypatch.setitem(mod.VL_REGIONS["vl_latent"], "corpus_source", "test-corpus")
    monkeypatch.setitem(mod.VL_REGIONS["vl_latent"], "manifest", None)
    monkeypatch.setattr(
        mod,
        "_shards",
        lambda pattern, root=mod.CORPUS: [
            "/mnt/fleet-datasets/csd/vl/tiny-imagenet/data/train-0.parquet"
        ],
    )

    try:
        mod.run_vl_region(name="vl_latent", state=tmp_path, steps=1, batch=1, dry=True)
    except mod.ReservedSourceError:
        pytest.fail("clean source incorrectly refused as reserved")


def test_classify_region_run_refuses_to_start_when_resolved_sources_include_a_reserved_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`run_classify_region` must refuse a reserved shard, doubly important because it
    resolves against `LOCAL_CORPUS` (`/bulk/csd-corpus`) -- the exact root `apps` and
    `code_contests` were fetched to (see `RESERVED_FOR_COMPOSE`'s docstring). Before this
    test existed, `run_classify_region` never called `_refuse_reserved_shards` at all.
    """
    monkeypatch.setattr(
        mod,
        "_shards",
        lambda pattern, root=mod.LOCAL_CORPUS: ["/bulk/csd-corpus/code/apps/train.parquet"],
    )

    with pytest.raises(mod.ReservedSourceError):
        mod.run_classify_region(
            name="classify_banking77", state=tmp_path, steps=1, batch=1, dry=True
        )


def test_classify_region_dry_run_is_unaffected_when_sources_stay_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Negative control: an unreserved resolved shard must not trip the classify guard."""
    monkeypatch.setattr(
        mod,
        "_shards",
        lambda pattern, root=mod.LOCAL_CORPUS: [
            "/bulk/csd-corpus/classify/banking77/train.parquet"
        ],
    )

    try:
        mod.run_classify_region(
            name="classify_banking77", state=tmp_path, steps=1, batch=1, dry=True
        )
    except mod.ReservedSourceError:
        pytest.fail("clean source incorrectly refused as reserved")
