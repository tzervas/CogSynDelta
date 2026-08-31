"""Autodev loop routes region-pretrain onto CogSynDelta, not memory-gate."""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "csd-autodev-loop"


def load_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import csd-autodev-loop with isolated steer path."""
    monkeypatch.setenv("CSD_AUTODEV_WT", str(tmp_path / "memory-gate-wt-p1-09"))
    loader = importlib.machinery.SourceFileLoader("csd_autodev_loop", str(SCRIPT))
    spec = importlib.util.spec_from_loader("csd_autodev_loop", loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_dest_worktree_region_pretrain_is_cogsyndelta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Steer region-pretrain must land in this repo, never memory-gate."""
    mod = load_loop(tmp_path, monkeypatch)
    dest = mod.dest_worktree("region-pretrain")
    assert dest == mod.ROOT
    assert dest != mod.MG_DEFAULT
    assert "memory-gate" not in str(dest)
    assert mod.apply_worktree_name("region-pretrain") == "region-pretrain"
    assert mod.is_region_pretrain("P1-region-pretrain") is True
    assert mod.is_region_pretrain("P1-09") is False


def test_dest_worktree_p1_09_stays_memory_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1-09 still targets the memory-gate worktree."""
    mod = load_loop(tmp_path, monkeypatch)
    dest = mod.dest_worktree("P1-09")
    assert dest == mod.MG_WT
    assert dest != mod.ROOT
    assert mod.apply_worktree_name("P1-09") == "p1-09"


def test_next_open_goal_reads_steer_region_pretrain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CSD_STEER next_goal region-pretrain wins over GOALS.md."""
    mod = load_loop(tmp_path, monkeypatch)
    steer = tmp_path / "csd-steer.json"
    steer.write_text(
        '{"pause": false, "next_goal": "region-pretrain"}',
        encoding="utf-8",
    )
    mod.STEER = steer
    assert mod.next_open_goal() == "region-pretrain"
    assert mod.is_region_pretrain(mod.next_open_goal()) is True
