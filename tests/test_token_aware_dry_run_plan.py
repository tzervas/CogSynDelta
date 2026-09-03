"""`scripts/csd-train-all.py`'s dry-run plan must show §4.0's token-aware terms (row W4)
and DEC-24's shared-embedding-table intent, per-region, without importing torch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu


def _load_csd_train_all():
    path = Path(__file__).resolve().parents[1] / "scripts" / "csd-train-all.py"
    spec = importlib.util.spec_from_file_location("csd_train_all_token_aware_plan_test", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_csd_train_all()


def _dry_run_plan(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> dict:
    """Run `run_region(..., dry=True)` with I/O bypassed (same pattern
    `tests/test_guards_can_fail.py` uses) and parse the printed plan back out."""
    monkeypatch.setattr(
        mod, "_shards", lambda pattern, root=mod.CORPUS: [str(tmp_path / "shard.parquet")]
    )
    monkeypatch.setattr(mod, "_schema_mismatch", lambda shards, cols: None)
    result = mod.run_region(name=name, state=tmp_path, steps=10, batch=8, shard_limit=0, dry=True)
    assert result is None  # dry runs never return a receipt
    out = capsys.readouterr().out
    marker = "resolved PretrainConfig (dry run, no training started):"
    assert marker in out, out
    plan_json = out.split(marker, 1)[1]
    # The plan is the first JSON object printed after the marker; everything after it on
    # the same call is nothing (run_region returns immediately), so a direct parse works.
    return json.loads(plan_json)


def test_memory_dry_run_plan_shows_token_aware_terms_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    plan = _dry_run_plan("memory", tmp_path, monkeypatch, capsys)
    assert plan["token_aware"] is True
    assert plan["token_loss_weight"] == mod.TOKEN_AWARE_REGIONS["memory"][0]
    assert plan["decorr_weight"] == mod.TOKEN_AWARE_REGIONS["memory"][1]
    assert plan["token_loss_weight"] > 0.0
    assert plan["decorr_weight"] > 0.0


def test_memory_dry_run_plan_shows_the_shared_embedding_table_design(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    plan = _dry_run_plan("memory", tmp_path, monkeypatch, capsys)
    table = plan["shared_embedding_table"]
    assert table["designed_source"] == "retrieve"
    assert table["applied"] is False
    assert table["init_embedding_from"] is None


def test_memory_dry_run_plan_reflects_a_real_init_embedding_from_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        mod, "_shards", lambda pattern, root=mod.CORPUS: [str(tmp_path / "shard.parquet")]
    )
    monkeypatch.setattr(mod, "_schema_mismatch", lambda shards, cols: None)
    fake_ckpt = str(tmp_path / "retrieve-final.pt")
    result = mod.run_region(
        name="memory",
        state=tmp_path,
        steps=10,
        batch=8,
        shard_limit=0,
        dry=True,
        init_embedding_from=fake_ckpt,
    )
    assert result is None
    out = capsys.readouterr().out
    plan = json.loads(out.split("no training started):", 1)[1])
    assert plan["shared_embedding_table"]["applied"] is True
    assert plan["shared_embedding_table"]["init_embedding_from"] == fake_ckpt


@pytest.mark.parametrize("name", ["code", "compress", "retrieve", "reason"])
def test_every_other_region_dry_run_plan_shows_token_aware_terms_off(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Regression guard: adding `memory` must not turn the terms on for any region
    trained before this feature existed."""
    plan = _dry_run_plan(name, tmp_path, monkeypatch, capsys)
    assert plan["token_aware"] is False
    assert plan["token_loss_weight"] == 0.0
    assert plan["decorr_weight"] == 0.0
    assert plan["shared_embedding_table"]["designed_source"] is None


def test_token_aware_regions_matches_the_memory_modules_own_constants() -> None:
    """MUST match regions/memory.py's TOKEN_LOSS_WEIGHT/DECORR_WEIGHT -- duplicated as
    bare numbers (see TOKEN_AWARE_REGIONS's own docstring for why it cannot be a direct
    import), so this is the guard that keeps the two from silently drifting apart."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions import memory as memory_mod

    assert mod.TOKEN_AWARE_REGIONS["memory"] == (
        memory_mod.TOKEN_LOSS_WEIGHT,
        memory_mod.DECORR_WEIGHT,
    )


def test_shared_embedding_table_source_names_a_real_region() -> None:
    for source in mod.SHARED_EMBEDDING_TABLE_SOURCE.values():
        assert source in mod.REGIONS, f"{source!r} names a region that does not exist"
