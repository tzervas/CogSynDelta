"""Autodev loop routes region-pretrain onto CogSynDelta, not memory-gate."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
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


def test_ti_ok_requires_live_and_guest_smi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_land_git_skips_when_staged_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrelated dirty files must not force a no-op commit."""
    mod = load_loop(tmp_path, monkeypatch)
    dest = tmp_path / "wt"
    dest.mkdir()
    monkeypatch.setattr(mod, "dest_worktree", lambda _g: dest)

    def fake_git(args: list[str], cwd: Path) -> dict:
        if args[0] in {"commit", "push"}:
            raise AssertionError(f"must not {args[0]} when staged empty")
        stdout = {
            ("rev-parse", "--abbrev-ref"): "feat/agent-harness",
            ("rev-parse", "HEAD"): "abc123",
            ("remote",): "forgejo\norigin",
            ("fetch",): "Already up to date.",
            ("merge",): "Already up to date.",
            ("add",): "",
            ("diff",): "",
        }
        key = tuple(args[:2]) if args[:1] == ["rev-parse"] else (args[0],)
        return {"ok": True, "stdout": stdout.get(key, ""), "rc": 0}

    monkeypatch.setattr(mod, "git_cmd", fake_git)
    rec = mod.land_git(
        "region-pretrain",
        {"ok": True, "path": "docs/program/GOAL-LOOP.md"},
    )
    assert rec["ok"] is True
    assert rec["skipped"] == "clean"
    assert rec["repo"] == "CogSynDelta"


def test_harness_goal_is_csd_autodev_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Loop/lab/forgejo CLI lands in csd-autodev, never CogSynDelta."""
    mod = load_loop(tmp_path, monkeypatch)
    home = tmp_path / "csd-autodev"
    home.mkdir()
    mod.AUTODEV_HOME = home
    assert mod.is_harness_goal("autodev-harness") is True
    assert mod.is_harness_rel("scripts/csd-autodev-loop") is True
    assert mod.is_harness_rel("tests/test_poc_region_pretrain.py") is False
    assert mod.dest_worktree("lab-console") == home
    assert mod.repo_for("autodev-harness") == "csd-autodev"
    assert mod.repo_for("region-pretrain") == "CogSynDelta"


def test_apply_reroutes_harness_path_off_cogsyndelta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """scripts/csd-lab-console proposed during a CSD goal still leaves CSD."""
    mod = load_loop(tmp_path, monkeypatch)
    home = tmp_path / "csd-autodev"
    home.mkdir()
    mod.AUTODEV_HOME = home
    rec = mod.apply_proposal(
        {"path": "scripts/csd-lab-console", "content": "#!/usr/bin/env python3\n"},
        mod.ROOT,
        worktree="region-pretrain",
    )
    assert rec["ok"] is True
    wrote = Path(str(rec.get("wrote") or ""))
    assert wrote.is_file()
    assert str(home) in str(wrote)


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


def test_beat_preserves_inflight_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Start-of-tick beat() must not drop an in-flight PR last record."""
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
    inflight = {
        "ok": False,
        "goal": "region-pretrain",
        "number": 3,
        "sha": "abc123",
        "merged": False,
        "required_ran": False,
    }
    mod.beat({"last": inflight})
    mod.beat()
    rec = __import__("json").loads(heart.read_text(encoding="utf-8"))
    assert rec["next_goal"] == "region-pretrain"
    assert rec["last"]["number"] == 3
    assert rec["last"]["merged"] is False
    assert rec["last"]["sha"] == "abc123"


def test_git_cmd_allows_merge_origin_main_on_feature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch-up merge of origin/main into a feature branch is allowed."""
    mod = load_loop(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "current_branch", lambda _cwd: "feat/agent-harness")
    seen: list[list[str]] = []

    def fake_run(argv: list[str], **_k: object) -> tuple[int, str]:
        seen.append(list(argv))
        return 0, "ok"

    monkeypatch.setattr(mod, "_run", fake_run)
    rec = mod.git_cmd(["merge", "--no-edit", "origin/main"], tmp_path)
    assert rec["ok"] is True
    assert "merge" in seen[0]


def test_git_cmd_refuses_checkout_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never check out protected trunks."""
    mod = load_loop(tmp_path, monkeypatch)
    rec = mod.git_cmd(["checkout", "main"], tmp_path)
    assert rec["ok"] is False
    assert "refusing checkout" in rec["error"]


def test_git_cmd_refuses_merge_while_on_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Merge while HEAD is main is still forbidden."""
    mod = load_loop(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "current_branch", lambda _cwd: "main")
    rec = mod.git_cmd(["merge", "--no-edit", "feat/agent-harness"], tmp_path)
    assert rec["ok"] is False
    assert "refusing merge on main" in rec["error"]


def test_refresh_pr_replaces_stale_heartbeat_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operator merge/rebase moves the PR head; heartbeat SHA must not stick."""
    mod = load_loop(tmp_path, monkeypatch)

    def fj(args: list[str], timeout: int = 90) -> dict:
        if args[:1] == ["pr"]:
            return {
                "http": 200,
                "sha": "c31186ccnew",
                "head": "feat/agent-harness",
                "base": "main",
                "merged": False,
            }
        return {}

    monkeypatch.setattr(mod, "forgejo_cmd", fj)
    live = mod.refresh_pr(
        {
            "number": 3,
            "sha": "b5ed0e75old",
            "repo": "CogSynDelta",
            "goal": "region-pretrain",
        }
    )
    assert live["sha"] == "c31186ccnew"
    assert live["base"] == "main"


def test_follow_pr_statuses_live_head_not_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """status/wait must use the live PR SHA after a merge into the feature."""
    mod = load_loop(tmp_path, monkeypatch)
    seen: list[list[str]] = []

    def fj(args: list[str], timeout: int = 90) -> dict:
        seen.append(list(args))
        if args[:1] == ["pr"]:
            return {
                "http": 200,
                "sha": "c31186ccnew",
                "head": "feat/agent-harness",
                "base": "main",
            }
        if args[:1] == ["status"]:
            return {"http": 200, "state": "failure", "checks": []}
        return {}

    monkeypatch.setattr(mod, "forgejo_cmd", fj)
    rec = mod.follow_pr(
        {"number": 3, "sha": "b5ed0e75old", "repo": "CogSynDelta", "goal": "region-pretrain"}
    )
    assert rec["sha"] == "c31186ccnew"
    assert rec["failed"] is True
    assert ["status", "CogSynDelta", "c31186ccnew"] in seen


def test_handle_inflight_merges_when_behind_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behind merge-target → merge origin/main into feature, never rebase."""
    mod = load_loop(tmp_path, monkeypatch)

    def fj(args: list[str], timeout: int = 90) -> dict:
        if args[:1] == ["pr"]:
            return {
                "http": 200,
                "sha": "oldsha",
                "head": "feat/agent-harness",
                "base": "main",
                "mergeable_state": "behind",
            }
        if args[:1] == ["compare"]:
            return {"http": 200, "behind": 4, "ahead": 2, "ok": True}
        if args[:1] == ["status"]:
            return {"http": 200, "state": "pending"}
        return {}

    monkeypatch.setattr(mod, "forgejo_cmd", fj)
    monkeypatch.setattr(mod, "dest_worktree", lambda _g: tmp_path)
    monkeypatch.setattr(
        mod,
        "sync_feature_with_base",
        lambda _cwd, base="main": {"ok": True, "sha": "mergedsha", "how": "merge", "base": base},
    )
    monkeypatch.setattr(mod, "current_sha", lambda _cwd: "oldsha")
    rec = mod.handle_inflight(
        {
            "number": 3,
            "sha": "oldsha",
            "repo": "CogSynDelta",
            "goal": "region-pretrain",
            "merged": False,
        }
    )
    assert rec["synced"] is True
    assert rec["sha"] == "mergedsha"
    assert rec["sync"]["how"] == "merge"


def test_handle_inflight_steer_note_retries_implement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New steer with CI job URLs must unstick follow-only on a red PR."""
    mod = load_loop(tmp_path, monkeypatch)
    steer = tmp_path / "csd-steer.json"
    note = (
        "updated the branch for the PR off the upstream since it was out of date. "
        "reran checks. https://git.vectorweight.com/tzervas/CogSynDelta/actions/runs/631/jobs/0"
    )
    steer.write_text(
        json.dumps({"pause": False, "next_goal": "region-pretrain", "note": note}),
        encoding="utf-8",
    )
    mod.STEER = steer

    def fj(args: list[str], timeout: int = 90) -> dict:
        if args[:1] == ["pr"]:
            return {
                "http": 200,
                "sha": "c31186ccnew",
                "head": "feat/agent-harness",
                "base": "main",
            }
        if args[:1] == ["compare"]:
            return {"http": 200, "behind": 0, "ahead": 3, "ok": True}
        if args[:1] == ["status"]:
            return {"http": 200, "state": "failure", "checks": []}
        return {}

    monkeypatch.setattr(mod, "forgejo_cmd", fj)
    monkeypatch.setattr(mod, "dest_worktree", lambda _g: tmp_path)
    monkeypatch.setattr(mod, "current_sha", lambda _cwd: "c31186ccnew")
    rec = mod.handle_inflight(
        {
            "number": 3,
            "sha": "b5ed0e75old",
            "repo": "CogSynDelta",
            "goal": "region-pretrain",
            "merged": False,
        }
    )
    assert rec["retry_implement"] is True
    assert rec["sha"] == "c31186ccnew"
    assert rec["consumed_note"]


def test_handle_inflight_same_steer_note_does_not_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Already-consumed steer note must not re-implement forever."""
    mod = load_loop(tmp_path, monkeypatch)
    note = "updated the branch. reran checks. actions/runs/631/jobs/0"
    steer = tmp_path / "csd-steer.json"
    steer.write_text(
        json.dumps({"pause": False, "next_goal": "region-pretrain", "note": note}),
        encoding="utf-8",
    )
    mod.STEER = steer
    fp = mod.note_fingerprint(note)

    def fj(args: list[str], timeout: int = 90) -> dict:
        if args[:1] == ["pr"]:
            return {"http": 200, "sha": "c31186ccnew", "head": "feat/agent-harness", "base": "main"}
        if args[:1] == ["compare"]:
            return {"http": 200, "behind": 0, "ahead": 1, "ok": True}
        if args[:1] == ["status"]:
            return {"http": 200, "state": "failure", "checks": []}
        return {}

    monkeypatch.setattr(mod, "forgejo_cmd", fj)
    monkeypatch.setattr(mod, "dest_worktree", lambda _g: tmp_path)
    monkeypatch.setattr(mod, "current_sha", lambda _cwd: "c31186ccnew")
    rec = mod.handle_inflight(
        {
            "number": 3,
            "sha": "c31186ccnew",
            "repo": "CogSynDelta",
            "goal": "region-pretrain",
            "merged": False,
            "consumed_note": fp,
        }
    )
    assert rec.get("retry_implement") is not True
    assert rec.get("failed") is True


def test_handle_inflight_integrates_when_local_sha_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local worker behind operator merge on the feature ref must merge, not glue."""
    mod = load_loop(tmp_path, monkeypatch)

    def fj(args: list[str], timeout: int = 90) -> dict:
        if args[:1] == ["pr"]:
            return {
                "http": 200,
                "sha": "c31186ccnew",
                "head": "feat/agent-harness",
                "base": "main",
            }
        if args[:1] == ["compare"]:
            return {"http": 200, "behind": 0, "ahead": 5, "ok": True}
        if args[:1] == ["status"]:
            return {"http": 200, "state": "pending"}
        return {}

    monkeypatch.setattr(mod, "forgejo_cmd", fj)
    monkeypatch.setattr(mod, "dest_worktree", lambda _g: tmp_path)
    monkeypatch.setattr(mod, "current_sha", lambda _cwd: "34f63ecalocal")
    monkeypatch.setattr(
        mod,
        "sync_feature_with_base",
        lambda _cwd, base="main": {
            "ok": True,
            "sha": "c31186ccnew",
            "how": "merge",
            "base": base,
        },
    )
    rec = mod.handle_inflight(
        {
            "number": 3,
            "sha": "b5ed0e75old",
            "repo": "CogSynDelta",
            "goal": "region-pretrain",
            "merged": False,
        }
    )
    assert rec["synced"] is True
    assert rec["sha"] == "c31186ccnew"
    assert rec["sync"]["how"] == "merge"


def test_implement_prompt_includes_steer_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """local/code must see the operator steer (job URLs, merge, import fixes)."""
    mod = load_loop(tmp_path, monkeypatch)
    steer = tmp_path / "csd-steer.json"
    steer.write_text(
        json.dumps(
            {
                "pause": False,
                "next_goal": "region-pretrain",
                "note": "updated the branch; jobs/0 import issues",
            }
        ),
        encoding="utf-8",
    )
    mod.STEER = steer
    captured: list[str] = []

    def fake_run(argv: list[str], **_k: object) -> tuple[int, str]:
        if "--prompt" in argv:
            captured.append(argv[argv.index("--prompt") + 1])
        return 0, '{"path":"tests/x.py","content":"x"}'

    monkeypatch.setattr(mod, "_run", fake_run)
    rec = mod.implement("region-pretrain")
    assert rec["rc"] == 0
    assert captured
    assert "updated the branch" in captured[0]
    assert "live PR head" in captured[0]
