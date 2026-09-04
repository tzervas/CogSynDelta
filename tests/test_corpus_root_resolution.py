"""`LOCAL_CORPUS` resolution and the MISSING-source hard failure it enables.

WHY THIS FILE EXISTS
`scripts/csd-train-all.py` hardcoded `LOCAL_CORPUS = Path("/bulk/csd-corpus")`. That is
where gpu5080 mounts the bulk array; akula-prime mounts the same data at
`/mnt/bulk/csd-corpus`. On akula-prime, every `reason` source glob resolved zero shards
against the wrong root, and `run_region` used to print "source MISSING", "no usable
sources -- skipping", and return `None` -- which `main()` treats as nothing to report,
exiting 0. `--dry-run --regions reason` looked identical to a clean, empty plan.

This tests both halves of the fix: `resolve_local_corpus_root` (env override, fallback
probing) in isolation, and `run_region` actually raising -- not silently skipping -- when
a source is missing and `--allow-missing` was not passed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu


def _load_csd_train_all():
    """Same loading convention as tests/test_reserved_corpus_guard.py -- the hyphenated
    filename is not a valid module name."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "csd-train-all.py"
    spec = importlib.util.spec_from_file_location("csd_train_all_corpus_root_test", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_csd_train_all()


# ---------------------------------------------------------------------------------------
# `resolve_local_corpus_root` in isolation.
# ---------------------------------------------------------------------------------------


def test_env_override_wins_outright_over_every_candidate(tmp_path: Path) -> None:
    """CSD_CORPUS_ROOT, when set, is returned without even checking `.is_dir()` -- an
    operator who set it explicitly has already made the call."""
    override = tmp_path / "does-not-exist-yet"

    root, how = mod.resolve_local_corpus_root(
        candidates=(tmp_path / "a", tmp_path / "b"),
        env={"CSD_CORPUS_ROOT": str(override)},
    )

    assert root == override
    assert how == "env:CSD_CORPUS_ROOT"


def test_fallback_resolves_to_the_second_candidate_when_the_first_is_absent(
    tmp_path: Path,
) -> None:
    """The exact akula-prime shape: `/bulk/csd-corpus` (first candidate) does not exist,
    `/mnt/bulk/csd-corpus` (second) does -- a fake root layout constructed here rather
    than depending on this host's actual mounts."""
    first = tmp_path / "bulk" / "csd-corpus"  # deliberately never created
    second = tmp_path / "mnt" / "bulk" / "csd-corpus"
    second.mkdir(parents=True)

    root, how = mod.resolve_local_corpus_root(candidates=(first, second), env={})

    assert root == second
    assert "probed" in how
    assert str(second) in how


def test_first_candidate_wins_when_both_exist(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    root, how = mod.resolve_local_corpus_root(candidates=(first, second), env={})

    assert root == first


def test_falls_back_to_the_first_candidate_as_a_deterministic_default_when_none_exist(
    tmp_path: Path,
) -> None:
    """Neither candidate exists (the "nothing present" case): this function itself must
    not raise -- it returns the first candidate anyway (preserving the old hardcoded
    default as SOME answer), and callers (`run_region`) are what turn that into a loud
    failure once they actually try to resolve a source against it."""
    first = tmp_path / "bulk" / "csd-corpus"
    second = tmp_path / "mnt" / "bulk" / "csd-corpus"

    root, how = mod.resolve_local_corpus_root(candidates=(first, second), env={})

    assert root == first
    assert "none of" in how


# ---------------------------------------------------------------------------------------
# `run_region`: a MISSING source is a hard failure by default, opt-out via allow_missing.
# ---------------------------------------------------------------------------------------


def test_run_region_raises_when_a_source_is_missing_and_allow_missing_is_not_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The exact reachable regression: every glob for a region resolves zero shards
    (`/bulk/csd-corpus` absent on this host, as on akula-prime). Before the fix this
    printed two lines and returned `None`; `run_region` must now raise instead."""
    monkeypatch.setattr(mod, "_shards", lambda pattern, root=mod.CORPUS: [])

    with pytest.raises(mod.CorpusSourceMissingError, match="resolved no shards"):
        mod.run_region(name="reason", state=tmp_path, steps=1, batch=1, shard_limit=0, dry=True)


def test_run_region_exception_is_a_runtime_error_main_already_catches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`main()`'s per-region loop catches `Exception` generically and turns it into a
    nonzero exit via `gate_failures` -- `CorpusSourceMissingError` needs no special
    handling there, only to actually be raised. This pins that it is a `RuntimeError`
    subclass, matching `GradedSourceMissingError`/`ReservedSourceError`."""
    assert issubclass(mod.CorpusSourceMissingError, RuntimeError)


def test_run_region_with_allow_missing_warns_and_skips_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Positive control: `--allow-missing` (`allow_missing=True`) restores the old
    behaviour -- no exception -- but must say so with a WARNING line, not the old bare
    "source MISSING" that read as routine output."""
    monkeypatch.setattr(mod, "_shards", lambda pattern, root=mod.CORPUS: [])

    result = mod.run_region(
        name="reason",
        state=tmp_path,
        steps=1,
        batch=1,
        shard_limit=0,
        dry=True,
        allow_missing=True,
    )

    assert result is None  # every source missing -> "no usable sources", not a raise
    printed = capsys.readouterr().out
    assert "WARNING" in printed
    assert "source MISSING" in printed


def test_run_region_allow_missing_trains_on_whichever_sources_do_resolve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`reason` has two sources (gsm8k, aqua_rat). With one resolving and one missing
    under `allow_missing=True`, the dry-run plan must still be produced from the
    resolving source rather than raising or skipping the whole region."""

    def fake_shards(pattern: str, root: Path = mod.CORPUS) -> list[str]:
        if "gsm8k" in pattern:
            return [str(tmp_path / "gsm8k-train.parquet")]
        return []  # aqua_rat: missing

    monkeypatch.setattr(mod, "_shards", fake_shards)
    monkeypatch.setattr(mod, "_schema_mismatch", lambda shards, cols: None)

    result = mod.run_region(
        name="reason",
        state=tmp_path,
        steps=1,
        batch=1,
        shard_limit=0,
        dry=True,
        allow_missing=True,
    )

    assert result is None  # dry run never returns a receipt
    printed = capsys.readouterr().out
    assert "WARNING" in printed
    assert "resolved PretrainConfig" in printed  # reached the plan, did not skip the region


def test_dry_run_plan_records_the_resolved_corpus_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The plan JSON `run_region` prints for `--dry-run` must record which root a
    region's sources resolved against -- the gap that let `reason`'s wrong root hide
    inside an otherwise-normal-looking plan."""
    monkeypatch.setattr(
        mod, "_shards", lambda pattern, root=mod.CORPUS: [str(tmp_path / "shard.parquet")]
    )
    monkeypatch.setattr(mod, "_schema_mismatch", lambda shards, cols: None)

    mod.run_region(name="reason", state=tmp_path, steps=1, batch=1, shard_limit=0, dry=True)

    import json as _json

    printed = capsys.readouterr().out
    plan_json = printed.split("resolved PretrainConfig (dry run, no training started):")[1]
    plan = _json.loads(plan_json)
    assert plan["corpus_root"] == str(mod.LOCAL_CORPUS)
    assert plan["corpus_root_resolution"] == mod.LOCAL_CORPUS_RESOLUTION
