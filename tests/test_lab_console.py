"""Unit tests for lab JSON routes wrapping autodev CLIs."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "csd-lab-console"


def load_lab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import csd-lab-console with isolated plan/steer/heartbeat paths."""
    monkeypatch.setenv("CSD_GPU_PLAN", str(tmp_path / "gpu-plan.json"))
    monkeypatch.setenv("CSD_STEER", str(tmp_path / "csd-steer.json"))
    monkeypatch.setenv("CSD_AUTODEV_HEARTBEAT", str(tmp_path / "hb.json"))
    monkeypatch.setenv("CSD_LAB_BIND", "192.168.1.98")
    monkeypatch.delenv("TOKEN", raising=False)
    monkeypatch.delenv("FORGEJO_TOKEN", raising=False)
    loader = importlib.machinery.SourceFileLoader("csd_lab_console", str(SCRIPT))
    spec = importlib.util.spec_from_loader("csd_lab_console", loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _plan() -> dict[str, Any]:
    return {
        "autodev_priority": True,
        "akula-prime": {"name": "RTX 3090 Ti", "free_mib": 6000},
        "gpu5080": {
            "lock": "lock=idle",
            "comfy": "masked",
            "helper_ok": True,
        },
    }


def test_gpu_and_lock_read_plan_without_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = load_lab(tmp_path, monkeypatch)

    def boom(*_a: object, **_k: object) -> tuple[int, str]:
        raise AssertionError("must not spawn CLI when plan file exists")

    monkeypatch.setattr(mod, "_run_cli", boom)
    (tmp_path / "gpu-plan.json").write_text(json.dumps(_plan()), encoding="utf-8")
    code, rec = mod.handle_lab("GET", "/api/gpu", {})
    assert code == 200
    assert rec["akula-prime"]["name"] == "RTX 3090 Ti"
    assert rec["gpu5080"]["comfy"] == "masked"
    code, lock = mod.handle_lab("GET", "/api/gpu/lock", {})
    assert code == 200
    assert lock["lock"] == "lock=idle"
    assert lock["comfy"] == "masked"
    assert lock["helper_ok"] is True


def test_git_refuses_github_and_protected_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    gh = mod.lab_git(["push", "https://github.com/tzervas/CogSynDelta.git", "HEAD"])
    assert gh["ok"] is False
    assert "GitHub" in gh["stdout"]
    main = mod.lab_git(["push", "forgejo", "HEAD:main"])
    assert main["ok"] is False
    assert "main" in main["stdout"]
    code, rec = mod.handle_lab("POST", "/api/git", {"args": ["push", "forgejo", "HEAD:dev"]})
    assert code == 200
    assert rec["ok"] is False


def test_git_wraps_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    called: dict[str, Any] = {}

    def fake(argv: list[str], timeout: int = 90) -> tuple[int, str]:
        called["argv"] = argv
        called["timeout"] = timeout
        return 0, "feat/agent-harness"

    monkeypatch.setattr(mod, "_run_cli", fake)
    rec = mod.lab_git(["status", "-sb"])
    assert rec["ok"] is True
    assert rec["rc"] == 0
    assert rec["stdout"] == "feat/agent-harness"
    assert str(mod.GIT_CLI) == called["argv"][0]


def test_forgejo_prs_and_status_wrap_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    called: list[list[str]] = []

    def fake(argv: list[str], timeout: int = 90) -> tuple[int, str]:
        called.append(argv)
        if "prs" in argv:
            return 0, json.dumps(
                {"http": 200, "repo": "tzervas/CogSynDelta", "pulls": [{"number": 1}]}
            )
        if "status" in argv:
            return 0, json.dumps({"http": 200, "state": "pending", "checks": []})
        return 1, "{}"

    monkeypatch.setattr(mod, "_run_cli", fake)
    code, rec = mod.handle_lab("GET", "/api/forgejo/prs", {"repo": "CogSynDelta", "limit": 5})
    assert code == 200
    assert rec["http"] == 200
    assert rec["pulls"][0]["number"] == 1
    assert "TOKEN=git/autodev" in called[0]
    code, st = mod.handle_lab(
        "POST",
        "/api/forgejo/status",
        {"repo": "tzervas/CogSynDelta", "sha": "abc"},
    )
    assert code == 200
    assert st["state"] == "pending"
    bad = mod.handle_lab("GET", "/api/forgejo/prs", {"repo": "tzervas/evil"})
    assert bad is not None
    assert bad[1].get("error") == "repo not allowlisted"


def test_loop_worker_does_not_spawn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    (tmp_path / "hb.json").write_text(
        json.dumps({"t": time.time(), "last": {"goal": "G-QD", "applied": {"ok": True}}}),
        encoding="utf-8",
    )

    def boom(*_a: object, **_k: object) -> tuple[int, str]:
        raise AssertionError("HTTP must not spawn --worker")

    monkeypatch.setattr(mod, "_run_cli", boom)
    code, rec = mod.handle_lab("POST", "/api/loop", {"mode": "worker"})
    assert code == 200
    assert rec["ok"] is True
    assert rec["goal"] == "G-QD"
    assert "systemd" in rec["notes"]
    assert mod.handle_lab("POST", "/api/provision", {}) is None


def test_steer_get_and_priority_and_bind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    (tmp_path / "csd-steer.json").write_text(
        json.dumps({"pause": False, "note": "hold", "autodev_priority": True}),
        encoding="utf-8",
    )
    code, rec = mod.handle_lab("GET", "/api/steer", {})
    assert code == 200
    assert rec["pause"] is False
    assert rec["autodev_priority"] is True
    pri = mod.lab_priority("nope")
    assert pri["ok"] is False
    assert mod.lab_bind() == "192.168.1.98"
    monkeypatch.setenv("CSD_LAB_BIND", "0.0.0.0")
    with pytest.raises(SystemExit, match="WAN"):
        mod.lab_bind()


def test_apply_allowlist_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    wt = tmp_path / "p1-08"
    (wt / "src").mkdir(parents=True)
    protected = tmp_path / "memory-gate"
    protected.mkdir()
    mod.WORKTREES = {"p1-08": wt}
    mod.PROTECTED_WT = protected
    ok = mod.http_apply("p1-08", "src/foo.py", "x = 1\n")
    assert ok["ok"] is True
    deny = mod.http_apply("p1-08", "README.md", "nope")
    assert deny["ok"] is False
    assert "not allowed" in deny["error"]
    mod.WORKTREES = {"p1-08": protected}
    refuse = mod.http_apply("p1-08", "src/foo.py", "x")
    assert refuse["ok"] is False
    assert "kang-main-wip" in refuse["error"]


def test_comfy_paths_not_captured_by_lab_wrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    assert mod.handle_lab("GET", "/api/comfy/list-workflows", {}) is None
    assert mod.handle_lab("POST", "/api/comfy/queue-prompt", {"prompt": {}}) is None
    assert mod.handle_lab("GET", "/api/status", {}) is None
