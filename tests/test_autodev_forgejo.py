"""Unit tests for Forgejo autodev merge-gate (skip theatre is not green)."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "csd-autodev-forgejo"


def load_fj() -> Any:
    loader = importlib.machinery.SourceFileLoader("csd_autodev_forgejo", str(SCRIPT))
    spec = importlib.util.spec_from_loader("csd_autodev_forgejo", loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_evaluate_gate_empty_is_not_green() -> None:
    mod = load_fj()
    rec = mod.evaluate_gate("success", [])
    assert rec["required_ran"] is False
    assert rec["required_succeeded"] is False
    assert "missing runner" in rec["notes"]


def test_evaluate_gate_skip_theatre() -> None:
    mod = load_fj()
    rec = mod.evaluate_gate(
        "success",
        [
            {
                "context": "quality",
                "state": "success",
                "description": "skipped || true",
            }
        ],
    )
    assert rec["required_ran"] is False
    assert rec["required_succeeded"] is False
    assert "skip theatre" in rec["notes"]


def test_evaluate_gate_pending() -> None:
    mod = load_fj()
    rec = mod.evaluate_gate(
        "pending",
        [{"context": "tests", "state": "pending", "description": "running"}],
    )
    assert rec["required_ran"] is False
    assert rec["required_succeeded"] is False


def test_evaluate_gate_success() -> None:
    mod = load_fj()
    rec = mod.evaluate_gate(
        "success",
        [
            {"context": "quality", "state": "success", "description": "ok"},
            {"context": "tests", "state": "success", "description": "ok"},
        ],
    )
    assert rec["required_ran"] is True
    assert rec["required_succeeded"] is True


def test_evaluate_gate_missing_required_context() -> None:
    mod = load_fj()
    rec = mod.evaluate_gate(
        "success",
        [{"context": "quality", "state": "success"}],
        required=["quality", "tests"],
    )
    assert rec["required_ran"] is False
    assert rec["required_succeeded"] is False
    assert "tests" in rec["notes"]


def _ns(**kwargs: object) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def test_merge_gate_does_not_merge_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = load_fj()
    monkeypatch.setattr(mod, "token", lambda: "x")
    merged = {"n": 0}

    def api(method: str, path: str, body: dict | None = None) -> tuple[int, object]:
        if path.endswith("/pulls/1") and method == "GET":
            return 200, {
                "number": 1,
                "merged": False,
                "mergeable": True,
                "head": {"ref": "feat/agent-harness", "sha": "abc123def"},
                "base": {"ref": "main"},
                "user": {"login": "autodev"},
            }
        if path.endswith("/status"):
            return 200, {
                "state": "success",
                "statuses": [
                    {
                        "context": "quality",
                        "status": "success",
                        "description": "Job skipped",
                    }
                ],
            }
        if path.endswith("/branch_protections"):
            return 200, []
        if path.endswith("/merge"):
            merged["n"] += 1
            return 200, {"merged": True}
        return 404, "no"

    monkeypatch.setattr(mod, "api", api)
    buf = StringIO()
    with patch("sys.stdout", buf):
        rc = mod.cmd_merge_gate(_ns(repo="CogSynDelta", sha="abc123def", number=1))
    rec = json.loads(buf.getvalue())
    assert rc == 1
    assert rec["merged"] is False
    assert rec["required_succeeded"] is False
    assert merged["n"] == 0
    assert "PR 1" in rec["notes"]


def test_merge_gate_merges_when_required_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = load_fj()
    monkeypatch.setattr(mod, "token", lambda: "x")

    def api(method: str, path: str, body: dict | None = None) -> tuple[int, object]:
        if path.endswith("/pulls/2") and method == "GET":
            return 200, {
                "number": 2,
                "merged": False,
                "mergeable": True,
                "head": {"ref": "feat/ok", "sha": "deadbeef"},
                "base": {"ref": "main"},
                "user": {"login": "autodev"},
            }
        if path.endswith("/status"):
            return 200, {
                "state": "success",
                "statuses": [
                    {"context": "quality", "status": "success", "description": "ok"},
                    {"context": "tests", "status": "success", "description": "ok"},
                ],
            }
        if path.endswith("/branch_protections"):
            return 200, []
        if method == "POST" and path.endswith("/merge"):
            assert body is not None
            assert body.get("Do") == "merge"
            return 200, {"merged": True}
        return 404, "no"

    monkeypatch.setattr(mod, "api", api)
    buf = StringIO()
    with patch("sys.stdout", buf):
        rc = mod.cmd_merge_gate(_ns(repo="memory-gate", sha="deadbeef", number=2))
    rec = json.loads(buf.getvalue())
    assert rc == 0
    assert rec["ok"] is True
    assert rec["merged"] is True
    assert rec["required_ran"] is True
    assert rec["required_succeeded"] is True


def test_merge_gate_refuses_develop_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = load_fj()
    monkeypatch.setattr(mod, "token", lambda: "x")

    def api(method: str, path: str, body: dict | None = None) -> tuple[int, object]:
        if "/pulls/" in path and method == "GET":
            return 200, {
                "number": 9,
                "merged": False,
                "mergeable": True,
                "head": {"ref": "develop", "sha": "abc"},
                "base": {"ref": "main"},
                "user": {"login": "jules"},
            }
        raise AssertionError("must not fetch status for denied head")

    monkeypatch.setattr(mod, "api", api)
    buf = StringIO()
    with patch("sys.stdout", buf):
        rc = mod.cmd_merge_gate(_ns(repo="CogSynDelta", sha="abc", number=9))
    rec = json.loads(buf.getvalue())
    assert rc == 1
    assert rec["merged"] is False
    assert rec["state"] == "blocked"
