"""Unit tests for Open WebUI CSD autodev overlay tools."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request

import pytest

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "config/webui/overlay/functions/csd_autodev.py"
INSTALLER = ROOT / "scripts/csd-webui-install-overlay"


def load_mod(path: Path, name: str) -> Any:
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def load_overlay() -> Any:
    return load_mod(OVERLAY, "csd_autodev_overlay")


class _Resp:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self.status = status
        self._raw = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_notes_sandbox_lan_only() -> None:
    tools = load_overlay().Tools()
    rec = tools.akula_notes_sandbox()
    assert rec["ok"] is True
    assert rec["jupyter"].startswith("http://172.32.0.1:")
    assert "openai.com" not in json.dumps(rec)


def test_apply_refuses_kang_main_wip() -> None:
    tools = load_overlay().Tools()
    rec = tools.csd_lab_apply("kang-main-wip", "src/x.py", "print(1)\n")
    assert rec["ok"] is False
    assert "kang-main-wip" in rec["error"]


def test_apply_refuses_bad_prefix() -> None:
    tools = load_overlay().Tools()
    rec = tools.csd_lab_apply("p1-08", "scripts/evil.py", "x")
    assert rec["ok"] is False
    assert "not allowed" in rec["error"]


def test_git_refuses_github(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = load_overlay().Tools()

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("must not call lab for github")

    monkeypatch.setattr(tools, "_call", boom)
    rec = tools.csd_autodev_git(["push", "https://github.com/tzervas/CogSynDelta.git"])
    assert rec["ok"] is False
    assert "GitHub" in rec["notes"]


def test_prs_allowlist() -> None:
    tools = load_overlay().Tools()
    rec = tools.csd_autodev_forgejo_prs("evil/not-ours")
    assert rec["ok"] is False
    assert rec["http"] == 400


def test_steer_posts_lab(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = load_overlay().Tools()
    seen: list[tuple[str, str, dict[str, Any] | None]] = []

    def fake(
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        **_k: object,
    ) -> dict[str, Any]:
        seen.append((method, path, payload))
        return {"ok": True, "http": 200, "pause": False, "note": "hold"}

    monkeypatch.setattr(tools, "_call", fake)
    rec = tools.csd_lab_steer(pause=False, note="hold")
    assert rec["ok"] is True
    assert seen[0][0] == "POST"
    assert seen[0][1] == "/api/steer"
    assert seen[0][2] == {"pause": False, "note": "hold"}


def test_gpu_plan_falls_back_to_status(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = load_overlay().Tools()

    def fake(
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        **_k: object,
    ) -> dict[str, Any]:
        if path == "/api/gpu-plan":
            return {"ok": False, "http": 404, "notes": "missing"}
        assert path == "/api/status"
        return {
            "ok": True,
            "http": 200,
            "prime_smi": "3090",
            "gpu5080_smi": "5080",
            "gpu5080_lock": "idle",
            "gpu5080_comfy": "masked",
            "steer": {"autodev_priority": True},
            "feed": {"prime": {"loaded": []}, "gpu5080": {"loaded": []}},
        }

    monkeypatch.setattr(tools, "_call", fake)
    rec = tools.csd_gpu_plan()
    assert rec["ok"] is True
    assert rec["autodev_priority"] is True
    assert rec["akula-prime"]["smi"] == "3090"
    assert rec["gpu5080"]["comfy"] == "masked"


def test_lock_observe_from_status(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = load_overlay().Tools()

    def fake(
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        **_k: object,
    ) -> dict[str, Any]:
        if path == "/api/gpu5080-lock":
            return {"ok": False, "http": 404}
        return {
            "ok": True,
            "http": 200,
            "gpu5080_lock": "idle",
            "gpu5080_comfy": "masked",
        }

    monkeypatch.setattr(tools, "_call", fake)
    rec = tools.gpu5080_lock_observe()
    assert rec["ok"] is True
    assert rec["lock"] == "idle"
    assert rec["helper_ok"] is True


def test_lock_observe_wrapped_idle_helper_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrap-ready + lock idle is helper_ok; mask is not required."""
    tools = load_overlay().Tools()

    def fake(
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        **_k: object,
    ) -> dict[str, Any]:
        if path == "/api/gpu5080-lock":
            return {"ok": False, "http": 404}
        return {
            "ok": True,
            "http": 200,
            "gpu5080_lock": "idle",
            "gpu5080_comfy": "wrapped",
        }

    monkeypatch.setattr(tools, "_call", fake)
    rec = tools.gpu5080_lock_observe()
    assert rec["ok"] is True
    assert rec["comfy"] == "wrapped"
    assert rec["helper_ok"] is True


def test_merge_gate_missing_is_not_green(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = load_overlay().Tools()

    def fake(
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        **_k: object,
    ) -> dict[str, Any]:
        return {"ok": False, "http": 404, "notes": "no route"}

    monkeypatch.setattr(tools, "_call", fake)
    rec = tools.csd_autodev_merge_gate("CogSynDelta", "abc", 1)
    assert rec["ok"] is False
    assert rec["merged"] is False
    assert rec["required_succeeded"] is False
    assert "skip theatre" in rec["notes"]


def test_priority_status_uses_live_status(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = load_overlay().Tools()

    def fake(
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        **_k: object,
    ) -> dict[str, Any]:
        assert path == "/api/status"
        return {
            "ok": True,
            "steer": {"autodev_priority": True, "pause": False},
            "gpu5080_comfy": "masked",
        }

    monkeypatch.setattr(tools, "_call", fake)
    rec = tools.csd_autodev_priority("status")
    assert rec["ok"] is True
    assert rec["comfy"] == "masked"
    assert rec["steer"]["autodev_priority"] is True


def test_apply_sends_bearer_without_echo(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_overlay()
    tools = mod.Tools()
    dummy = "csd-apply-fixture-not-a-live-key"
    object.__setattr__(tools.valves, "apply_token", dummy)
    captured: dict[str, str] = {}

    def fake_urlopen(req: Request, timeout: float = 0) -> _Resp:
        captured["auth"] = req.get_header("Authorization") or ""
        return _Resp({"ok": True, "wrote": "p1-08/src/foo.py"})

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    rec = tools.csd_lab_apply("p1-08", "src/foo.py", "x = 1\n")
    assert rec["ok"] is True
    assert captured["auth"].startswith("Bearer ")
    assert dummy not in json.dumps(rec)


def test_http_error_404(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_overlay()
    tools = mod.Tools()

    def fake_urlopen(req: object, timeout: float = 0) -> _Resp:
        raise HTTPError(
            "http://192.168.1.98:9118/api/autodev/whoami",
            404,
            "not found",
            None,
            BytesIO(b'{"error":"missing"}'),
        )

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    rec = tools.csd_autodev_forgejo_whoami()
    assert rec["http"] == 404
    assert rec["ok"] is False


def test_loop_rejects_provision_mode() -> None:
    tools = load_overlay().Tools()
    rec = tools.csd_autodev_loop("provision")
    assert rec["ok"] is False


def test_installer_specs_match_tool_methods() -> None:
    overlay = load_overlay()
    installer = load_mod(INSTALLER, "csd_webui_install_overlay")
    names = {s["name"] for s in installer.AUTODEV_SPECS}
    methods = {
        n
        for n in dir(overlay.Tools)
        if not n.startswith("_") and n != "Valves" and callable(getattr(overlay.Tools, n))
    }
    assert names == methods
