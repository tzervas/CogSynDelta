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
    monkeypatch.setenv("CSD_GROK_NEED", str(tmp_path / "csd-need-grok.json"))
    monkeypatch.setenv("CSD_AUTODEV_STALL", str(tmp_path / "autodev-stall.json"))
    monkeypatch.setenv("CSD_GPU_PLAN", str(tmp_path / "gpu-plan.json"))
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



def test_cap_ping_drops_transcripts_and_stays_under_2kib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ping JSON is identity autodev, no dumps, at most 2048 bytes."""
    mod = load_loop(tmp_path, monkeypatch)
    rec = mod.cap_ping(
        {
            "need": True,
            "question": "x" * 400,
            "evidence": "y" * 800,
            "transcript": "secret chat",
            "stdout": "pytest dump",
            "tried": ["a"] * 40,
            "paths": ["src/x.py"] * 40,
        }
    )
    assert rec["identity"] == "autodev"
    assert "transcript" not in rec
    assert "stdout" not in rec
    raw = __import__("json").dumps(rec, separators=(",", ":")).encode()
    assert len(raw) <= 2048


def test_write_need_grok_fingerprint_does_not_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same stall fingerprint must not rewrite the mailbox (no tick mail)."""
    mod = load_loop(tmp_path, monkeypatch)
    mod.GROK_NEED = tmp_path / "csd-need-grok.json"
    first = mod.write_need_grok(
        question="CI red twice?",
        evidence="blocker: pytest; paths: tests/t.py; tried: wait",
        goal="P1-09",
        blocker="required pytest red",
        paths=["tests/t.py"],
        tried=["wait"],
    )
    blob1 = mod.GROK_NEED.read_bytes()
    second = mod.write_need_grok(
        question="CI red twice?",
        evidence="blocker: pytest; paths: tests/t.py; tried: wait",
        goal="P1-09",
        blocker="required pytest red",
        paths=["tests/t.py"],
        tried=["wait"],
    )
    blob2 = mod.GROK_NEED.read_bytes()
    assert first["need"] is True
    assert second["fp"] == first["fp"]
    assert blob2 == blob1


def test_maybe_ping_stall_waits_for_n_equals_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First failure is not a ping; second same fingerprint sets need=true."""
    mod = load_loop(tmp_path, monkeypatch)
    mod.GROK_NEED = tmp_path / "csd-need-grok.json"
    mod.STALL = tmp_path / "autodev-stall.json"
    first = mod.maybe_ping_stall(
        "P1-09",
        "required pytest red",
        "contract or runner?",
        "blocker: CI; paths: scripts/csd-autodev-loop; tried: wait",
        paths=["scripts/csd-autodev-loop"],
        tried=["wait"],
    )
    assert first["need"] is False
    assert first["n"] == 1
    second = mod.maybe_ping_stall(
        "P1-09",
        "required pytest red",
        "contract or runner?",
        "blocker: CI; paths: scripts/csd-autodev-loop; tried: wait",
        paths=["scripts/csd-autodev-loop"],
        tried=["wait"],
    )
    assert second["need"] is True
    assert second["identity"] == "autodev"


def test_run_refuses_hosted_grok_and_github(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loop never spawns grok or GitHub remotes."""
    mod = load_loop(tmp_path, monkeypatch)
    rc, text = mod._run(["grok", "--print"])
    assert rc == 2
    assert "refusing hosted grok" in text
    rc, text = mod._run(["git", "push", "https://github.com/tzervas/CogSynDelta.git"])
    assert rc == 2
    assert "never GitHub" in text


def test_ti_ok_requires_live_and_guest_smi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1080 Ti only when catalog live=true and guest smi lists 1080."""
    mod = load_loop(tmp_path, monkeypatch)
    plan = tmp_path / "gpu-plan.json"
    mod.GPU_PLAN = plan
    plan.write_text("{}", encoding="utf-8")
    assert mod.ti_ok() is False
    plan.write_text(
        '{"gpu5080-1080ti": {"live": true, "name": "NVIDIA GeForce GTX 1080 Ti"}}',
        encoding="utf-8",
    )
    # catalog in-repo currently live=true; both gates must pass
    assert mod.ti_ok() is True
    plan.write_text(
        '{"gpu5080-1080ti": {"live": true, "name": "NVIDIA GeForce RTX 5080"}}',
        encoding="utf-8",
    )
    assert mod.ti_ok() is False



def test_repo_for_region_pretrain_is_cogsyndelta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Region-pretrain PRs target CogSynDelta, never memory-gate."""
    mod = load_loop(tmp_path, monkeypatch)
    assert mod.repo_for("region-pretrain") == "CogSynDelta"
    assert mod.repo_for("P1-region-pretrain") == "CogSynDelta"
    assert mod.repo_for("P1-09") == "memory-gate"


def test_land_git_skips_when_staged_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unrelated dirty files must not force a no-op commit."""
    mod = load_loop(tmp_path, monkeypatch)
    dest = tmp_path / "wt"
    dest.mkdir()
    monkeypatch.setattr(mod, "dest_worktree", lambda _g: dest)

    def fake_git(args: list[str], cwd: Path) -> dict:
        if args[:2] == ["rev-parse", "--abbrev-ref"]:
            return {"ok": True, "stdout": "feat/agent-harness", "rc": 0}
        if args[0] == "add":
            return {"ok": True, "stdout": "", "rc": 0}
        if args[:3] == ["diff", "--cached", "--name-only"]:
            return {"ok": True, "stdout": "", "rc": 0}
        if args[0] == "commit":
            raise AssertionError("must not commit when staged empty")
        if args[0] == "push":
            raise AssertionError("must not push when staged empty")
        return {"ok": True, "stdout": "", "rc": 0}

    monkeypatch.setattr(mod, "git_cmd", fake_git)
    rec = mod.land_git(
        "region-pretrain",
        {"ok": True, "path": "docs/program/GOAL-LOOP.md"},
    )
    assert rec["ok"] is True
    assert rec["skipped"] == "clean"
    assert rec["repo"] == "CogSynDelta"


def test_launch_local_workflow_dispatches_not_grok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Queue-failed fallback is Forgejo csd-python-first-drive, not hosted Grok."""
    mod = load_loop(tmp_path, monkeypatch)
    seen: list[list[str]] = []

    def fake_run(argv: list[str], **_k: object) -> tuple[int, str]:
        seen.append(list(argv))
        return 0, '{"ok": true, "http": 204}'

    monkeypatch.setattr(mod, "_run", fake_run)
    rec = mod.launch_local_workflow("region-pretrain", "queue-failed")
    assert rec["ok"] is True
    assert rec["hosted_grok"] is False
    assert rec["workflow"] == "csd-python-first-drive.yml"
    assert seen
    flat = " ".join(str(a) for a in seen[0])
    assert "workflow-dispatch" in flat
    assert "csd-python-first-drive.yml" in flat
    assert "grok" not in flat.lower()


def test_cluster_view_three_backends_not_256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Heartbeat cluster is 3090+5080+guest retrieve-index-light, not 256."""
    mod = load_loop(tmp_path, monkeypatch)
    plan = tmp_path / "gpu-plan.json"
    plan.write_text(
        '{"gpu5080-1080ti": {"live": true, "name": "NVIDIA GeForce GTX 1080 Ti"}}',
        encoding="utf-8",
    )
    mod.GPU_PLAN = plan
    view = mod.cluster_view()
    assert view["backends"] == ["akula-prime", "gpu5080", "gpu5080-1080ti"]
    assert view["count"] == 3
    assert view["not_256_specialists"] is True
    guest = view["gpu5080-1080ti"]
    assert guest["live"] is True
    assert guest["role"] == "retrieve-index-light"
    assert guest["host_ip"] == "192.168.1.251"
    assert guest["guest_ip"] == "192.168.1.243"


def test_beat_records_cluster_next_goal_and_need_grok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """beat() writes next_goal, 1080 Ti live role, mailbox, identity autodev."""
    mod = load_loop(tmp_path, monkeypatch)
    steer = tmp_path / "csd-steer.json"
    steer.write_text(
        '{"pause": false, "next_goal": "region-pretrain"}',
        encoding="utf-8",
    )
    heart = tmp_path / "autodev-heartbeat.json"
    need = tmp_path / "csd-need-grok.json"
    need.write_text('{"need": false}', encoding="utf-8")
    plan = tmp_path / "gpu-plan.json"
    plan.write_text(
        '{"gpu5080-1080ti": {"live": true, "name": "NVIDIA GeForce GTX 1080 Ti"}}',
        encoding="utf-8",
    )
    mod.STEER = steer
    mod.HEART = heart
    mod.GROK_NEED = need
    mod.GPU_PLAN = plan
    mod.beat()
    rec = __import__("json").loads(heart.read_text(encoding="utf-8"))
    assert rec["identity"] == "autodev"
    assert rec["next_goal"] == "region-pretrain"
    assert rec["ti"] is True
    assert rec["cluster"]["backends"] == [
        "akula-prime",
        "gpu5080",
        "gpu5080-1080ti",
    ]
    assert rec["cluster"]["gpu5080-1080ti"]["live"] is True
    assert rec["cluster"]["gpu5080-1080ti"]["role"] == "retrieve-index-light"
    assert rec["need_grok"]["need"] is False
    assert rec["need_grok"]["path"] == str(need)
    assert "memory-gate" not in rec["wt"]


def test_ensure_steer_cluster_keeps_region_pretrain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cluster annotation must not steer away from region-pretrain."""
    mod = load_loop(tmp_path, monkeypatch)
    steer = tmp_path / "csd-steer.json"
    steer.write_text(
        '{"pause": false, "next_goal": "region-pretrain", "note": "keep"}',
        encoding="utf-8",
    )
    plan = tmp_path / "gpu-plan.json"
    plan.write_text(
        '{"gpu5080-1080ti": {"live": true, "name": "NVIDIA GeForce GTX 1080 Ti"}}',
        encoding="utf-8",
    )
    need = tmp_path / "csd-need-grok.json"
    mod.STEER = steer
    mod.GPU_PLAN = plan
    mod.GROK_NEED = need
    out = mod.ensure_steer_cluster()
    assert out["next_goal"] == "region-pretrain"
    assert out["cluster"] == ["akula-prime", "gpu5080", "gpu5080-1080ti"]
    assert out["gpu5080-1080ti"]["live"] is True
    assert out["gpu5080-1080ti"]["role"] == "retrieve-index-light"
    assert out["need_grok"] == str(need)
    assert out["note"] == "keep"


def test_ensure_steer_cluster_keeps_p1_09_if_already(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If steer is already P1-09, leave that board row; still attach cluster."""
    mod = load_loop(tmp_path, monkeypatch)
    steer = tmp_path / "csd-steer.json"
    steer.write_text('{"pause": false, "next_goal": "P1-09"}', encoding="utf-8")
    plan = tmp_path / "gpu-plan.json"
    plan.write_text(
        '{"gpu5080-1080ti": {"live": true, "name": "NVIDIA GeForce GTX 1080 Ti"}}',
        encoding="utf-8",
    )
    mod.STEER = steer
    mod.GPU_PLAN = plan
    mod.GROK_NEED = tmp_path / "csd-need-grok.json"
    out = mod.ensure_steer_cluster()
    assert out["next_goal"] == "P1-09"
    assert out["cluster"] == ["akula-prime", "gpu5080", "gpu5080-1080ti"]

