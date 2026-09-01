"""Fail-closed Comfy media tools (scripts/csd-comfy)."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "csd-comfy"


def load_comfy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the CLI as a module with isolated plan paths."""
    monkeypatch.setenv("CSD_GPU_PLAN", str(tmp_path / "gpu-plan.json"))
    monkeypatch.setenv("CSD_COMFY_WORKFLOWS", str(ROOT / "config" / "media" / "workflows.json"))
    monkeypatch.setenv(
        "CSD_COMFY_IMAGE_WORKFLOW",
        str(ROOT / "config" / "media" / "comfy-image-workflow.json"),
    )
    monkeypatch.setenv("CSD_COMFY_URL", "http://192.168.1.251:8188")
    loader = importlib.machinery.SourceFileLoader("csd_comfy", str(SCRIPT))
    spec = importlib.util.spec_from_loader("csd_comfy", loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_catalog_has_31_graphs() -> None:
    raw = json.loads((ROOT / "config" / "media" / "workflows.json").read_text())
    assert len(raw["workflows"]) == 31
    nodes = json.loads((ROOT / "config" / "media" / "comfy-image-nodes.json").read_text())
    kinds = {n["type"] for n in nodes}
    assert kinds == {"prompt", "width", "height", "steps", "seed"}


def test_masked_list_queue_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CSD_COMFY_MASKED", "1")
    mod = load_comfy(tmp_path, monkeypatch)

    def boom(*_a, **_k):
        raise AssertionError("http must not run when masked")

    monkeypatch.setattr(mod, "http_json", boom)

    listed = mod.list_workflows("catalog")
    assert listed["ok"] is False
    assert listed["notes"] == "masked"
    assert listed["workflows"] == []

    queued = mod.queue_prompt({"1": {"class_type": "NOP", "inputs": {}}})
    assert queued["ok"] is False
    assert queued["notes"] == "masked"
    assert queued["prompt_id"] == ""

    st = mod.prompt_status("abc")
    assert st["ok"] is False
    assert st["notes"] == "masked"
    assert st["queue"] == {}
    assert st["history"] == {}
    assert st["system_stats"] == {}

    fx = mod.flux_klein_4b(text="a lantern")
    assert fx["ok"] is False
    assert fx["notes"] == "masked"


def test_plan_masked_without_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CSD_COMFY_MASKED", raising=False)
    plan = tmp_path / "gpu-plan.json"
    plan.write_text(
        json.dumps({"gpu5080": {"comfy": "masked", "lock": "lock=idle"}}),
        encoding="utf-8",
    )
    mod = load_comfy(tmp_path, monkeypatch)
    assert mod.is_masked() is True
    rec = mod.list_workflows()
    assert rec == {"ok": False, "notes": "masked", "workflows": []}


def test_plan_wrapped_is_not_masked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrap-ready plan does not fail-closed as masked; HTTP decides."""
    monkeypatch.delenv("CSD_COMFY_MASKED", raising=False)
    plan = tmp_path / "gpu-plan.json"
    plan.write_text(
        json.dumps({"gpu5080": {"comfy": "wrapped", "lock": "lock=idle"}}),
        encoding="utf-8",
    )
    mod = load_comfy(tmp_path, monkeypatch)

    def fake_http(method: str, path: str, body: dict | None = None, timeout: float = 8.0):
        if path == "/system_stats":
            return 200, {"devices": []}
        return 404, {}

    monkeypatch.setattr(mod, "http_json", fake_http)
    assert mod.is_masked() is False


def test_unmasked_catalog_and_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CSD_COMFY_MASKED", "0")
    mod = load_comfy(tmp_path, monkeypatch)

    calls: list[tuple[str, str]] = []
    posts: list[dict[str, Any]] = []

    def fake_http(method: str, path: str, body: dict | None = None, timeout: float = 8.0):
        calls.append((method, path))
        if path == "/object_info":
            return 200, {"CLIPTextEncode": {}, "SaveImage": {}}
        if path == "/prompt":
            assert body is not None and "prompt" in body
            assert "class_type" in next(iter(body["prompt"].values()))
            posts.append(body)
            return 200, {"prompt_id": "p-1"}
        if path == "/queue":
            return 200, {"queue_running": [], "queue_pending": []}
        if path.startswith("/history/"):
            return 200, {"p-1": {"status": {"completed": True}}}
        if path == "/system_stats":
            return 200, {"devices": [{"name": "cuda:0"}]}
        return 404, {}

    monkeypatch.setattr(mod, "http_json", fake_http)

    listed = mod.list_workflows("catalog")
    assert listed["ok"] is True
    assert len(listed["workflows"]) == 31
    assert listed["notes"].startswith("catalog")

    live = mod.list_workflows("live")
    assert live["ok"] is True
    assert {"class_type": "CLIPTextEncode"} in live["workflows"]

    queued = mod.queue_prompt({"5": {"class_type": "CLIPTextEncode", "inputs": {}}})
    assert queued == {
        "ok": True,
        "prompt_id": "p-1",
        "notes": "queued",
        "http": 200,
    }

    st = mod.prompt_status("p-1")
    assert st["ok"] is True
    assert st["notes"] == "live"
    assert "devices" in st["system_stats"]

    fx = mod.flux_klein_4b(text="a red boat", width=768, height=512, steps=6, seed=9)
    assert fx["ok"] is True
    assert fx["prompt_id"] == "p-1"
    assert ("POST", "/prompt") in calls
    graph = posts[-1]["prompt"]
    assert graph["5"]["inputs"]["text"] == "a red boat"
    assert graph["4"]["inputs"]["width"] == 768
    assert graph["4"]["inputs"]["height"] == 512
    assert graph["10"]["inputs"]["steps"] == 6
    assert graph["8"]["inputs"]["noise_seed"] == 9


def test_cli_masked_stdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setenv("CSD_COMFY_MASKED", "1")
    mod = load_comfy(tmp_path, monkeypatch)
    rc = mod.main(["list-workflows"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert out["notes"] == "masked"


def test_native_graph_node_ids() -> None:
    graph = json.loads((ROOT / "config" / "media" / "comfy-image-workflow.json").read_text())
    assert graph["5"]["class_type"] == "CLIPTextEncode"
    assert graph["8"]["class_type"] == "RandomNoise"
    assert graph["4"]["class_type"] == "EmptyFlux2LatentImage"
    assert graph["10"]["class_type"] == "Flux2Scheduler"
