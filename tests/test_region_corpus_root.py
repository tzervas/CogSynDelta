"""Every `REGIONS` consumer must resolve a region's shards against the SAME corpus root.

`REGION_CORPUS_ROOT` (`{"reason": LOCAL_CORPUS}`) used to be consulted in exactly one
place: `run_region`, which read it directly. `csd-quantize.py` and `csd-benchmark.py`
each called `spec["_shards"](g)` with no root argument at all, so both silently defaulted
to the shared `CORPUS` mount regardless of what `REGION_CORPUS_ROOT` said -- for `reason`
(the one region whose shards live under `/bulk/csd-corpus`, not the shared export), that
resolves zero shards through either script even on a host where training itself finds
them fine.

`region_spec()` now resolves `REGION_CORPUS_ROOT` into `RegionEntry.root`, so `run_region`
and both scripts read the SAME value from the SAME place. This file checks:

  1. `region_spec("reason").root` is `LOCAL_CORPUS`, and every other region's `.root` is
     the default `CORPUS`.
  2. `csd-quantize.py`'s `quantize_text_region` and `csd-benchmark.py`'s `benchmark_region`
     both resolve `reason`'s shards under a synthetic root standing in for the real
     `/bulk/csd-corpus` mount (which CI never has) -- driven through the real functions,
     not a reimplementation of their resolution logic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# MUST precede the imports below: quantize_text_region/benchmark_region import
# `tokenizers` at call time (inside the function, not module scope), but calling either
# function at all still needs it installed -- same reasoning as
# tests/test_reserved_corpus_guard.py and tests/test_benchmark_corpus_fingerprint.py.
pytest.importorskip("tokenizers", reason="train group not installed")

from cogsyndelta import corpus as corpus_mod

# MUST be imported here, at module (collection) time, before any test in this file or
# any other gets a chance to monkeypatch corpus_mod.fingerprint_corpus. quantize_text_
# region/benchmark_region both do `from cogsyndelta.regions.pretrain import ...` lazily,
# at call time. If that is pretrain.py's FIRST-EVER import anywhere in the session, it
# runs `from cogsyndelta.corpus import (..., fingerprint_corpus, ...)` at pretrain's own
# module scope while the mock is active, binding pretrain.py's `fingerprint_corpus` name
# to the mock permanently -- monkeypatch only undoes its write to corpus_mod's attribute,
# not that import-time binding, so every later test that relies on pretrain's real
# fingerprint_corpus silently gets the mock instead (order-dependent: whichever test file
# happens to import pretrain first, unpatched, "wins"). Importing it here, at collection
# time -- which always precedes every test's execution across the whole session -- fixes
# the real function in place before any monkeypatching can happen.
from cogsyndelta.regions import pretrain as pretrain_mod

pytestmark = pytest.mark.cpu

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(module_name: str, filename: str):
    """Import one of the hyphenated `scripts/` files as a module (same pattern as
    tests/test_region_spec_consumers.py's `_load`)."""
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_shard(path: Path) -> None:
    """`fingerprint_corpus`/`_shards` only need a real file to exist at the right name
    (glob discovery, `Path.stat().st_size`) -- not valid parquet content -- so this stays
    pyarrow-free and does not need the train group's heavier fixture machinery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stand-in for a parquet shard; only its name and size are read here")


# ---------------------------------------------------------------------------------------
# (1) region_spec(name).root
# ---------------------------------------------------------------------------------------


def test_region_spec_root_is_local_corpus_for_reason() -> None:
    train_all = _load("csd_train_all_region_root_test_1", "csd-train-all.py")
    assert train_all.region_spec("reason").root == train_all.LOCAL_CORPUS
    assert train_all.region_spec("reason").root != train_all.CORPUS


@pytest.mark.parametrize("name", ["code", "compress", "retrieve"])
def test_region_spec_root_defaults_to_corpus_for_other_regions(name: str) -> None:
    train_all = _load(f"csd_train_all_region_root_test_default_{name}", "csd-train-all.py")
    assert train_all.region_spec(name).root == train_all.CORPUS


# ---------------------------------------------------------------------------------------
# (2) csd-quantize.py and csd-benchmark.py resolve `reason` under a synthetic root.
#
# `_load_regions_spec`/`_regions_spec` are monkeypatched to return the SAME
# `REGION_CORPUS_ROOT`-patched `train_all` instance this test controls, rather than the
# real one each would otherwise re-import fresh from disk -- everything downstream
# (`quantize_text_region`/`benchmark_region` calling `spec["_shards"](g, entry.root)`) is
# the real, unmodified code path.
# ---------------------------------------------------------------------------------------


class _CapturedError(Exception):
    """Raised by the patched `fingerprint_corpus` to abort a run right after shard
    resolution, once the shard paths it was asked to fingerprint are in hand -- cheaper
    than staging a full receipt/checkpoint/tokenizer for a check that only needs to prove
    WHICH shards were resolved, not run a training-comparable eval."""


def _make_synthetic_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str):
    train_all = _load(f"csd_train_all_region_root_test_{suffix}", "csd-train-all.py")
    synthetic_root = tmp_path / "synthetic-bulk-csd-corpus"
    monkeypatch.setitem(train_all.REGION_CORPUS_ROOT, "reason", synthetic_root)

    primary_glob = train_all.region_spec("reason").sources[0][
        0
    ]  # "reason/gsm8k-main/train.parquet"
    shard_path = synthetic_root / Path(primary_glob)
    _write_shard(shard_path)

    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    receipt = {
        "corpus": {"shards": [shard_path.name]},
        "config": {"pair_columns": ["question", "answer"]},
    }
    (receipts_dir / "reason-20260101T000000Z.json").write_text(json.dumps(receipt))

    captured: dict[str, list[str]] = {}

    def _capture_and_raise(shards, *, columns=None, extra_sources=None):
        captured["shards"] = list(shards)
        raise _CapturedError

    # Patch the name where it is looked up. quantize_text_region/benchmark_region each
    # do their own `from cogsyndelta.corpus import fingerprint_corpus` at call time, so
    # patching corpus_mod covers them -- but pretrain.py bound its OWN `fingerprint_corpus`
    # name at its own module import (see the top-of-file comment on the pretrain_mod
    # import), and nothing here calls through pretrain.py's copy for this particular path.
    # Patch both anyway: they are the same logical target, and leaving pretrain_mod's copy
    # real while corpus_mod's is mocked is exactly the split state that let a stale mock
    # leak across tests in the first place.
    monkeypatch.setattr(corpus_mod, "fingerprint_corpus", _capture_and_raise)
    monkeypatch.setattr(pretrain_mod, "fingerprint_corpus", _capture_and_raise)

    return train_all, tmp_path, shard_path, captured


def test_quantize_resolves_reason_shards_under_a_synthetic_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_all, state, shard_path, captured = _make_synthetic_reason(
        tmp_path, monkeypatch, "quantize"
    )
    quantize = _load("csd_quantize_region_root_test", "csd-quantize.py")
    monkeypatch.setattr(
        quantize,
        "_load_regions_spec",
        lambda: {
            "REGIONS": train_all.REGIONS,
            "_shards": train_all._shards,
            "region_spec": train_all.region_spec,
        },
    )

    with pytest.raises(_CapturedError):
        quantize.quantize_text_region("reason", state, tolerance=0.05, aggressive=0, max_bits=8)

    assert captured["shards"] == [str(shard_path)]


def test_benchmark_resolves_reason_shards_under_a_synthetic_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_all, state, shard_path, captured = _make_synthetic_reason(
        tmp_path, monkeypatch, "benchmark"
    )
    benchmark = _load("csd_benchmark_region_root_test", "csd-benchmark.py")
    monkeypatch.setattr(
        benchmark,
        "_regions_spec",
        lambda: {
            "REGIONS": train_all.REGIONS,
            "_shards": train_all._shards,
            "region_spec": train_all.region_spec,
        },
    )

    with pytest.raises(_CapturedError):
        benchmark.benchmark_region("reason", state)

    assert captured["shards"] == [str(shard_path)]
