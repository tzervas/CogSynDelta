"""Unit tests for scripts/csd-model-router placement and migrate policy."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "csd-model-router"


def load_router(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the CLI as a module with isolated state/plan paths."""
    monkeypatch.setenv("CSD_ROUTER_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("CSD_GPU_PLAN", str(tmp_path / "gpu-plan.json"))
    monkeypatch.setenv("CSD_ROUTER_CATALOG", str(ROOT / "config" / "model-router.json"))
    loader = importlib.machinery.SourceFileLoader("csd_model_router", str(SCRIPT))
    spec = importlib.util.spec_from_loader("csd_model_router", loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    mod.reset_catalog()
    return mod


def inv_idle(*, gguf: int = 0, helper_ok: bool = True) -> dict[str, Any]:
    return {
        "akula-prime": {
            "used_mib": 16076,
            "free_mib": 6494,
            "total_mib": 23028,
        },
        "gpu5080": {
            "used_mib": 10,
            "free_mib": 15864,
            "total_mib": 16303,
            "lock": "lock=idle",
            "comfy": "masked",
            "helper_ok": helper_ok,
            "gguf_count": gguf,
        },
    }


def test_catalog_has_one_vision() -> None:
    raw = (ROOT / "config" / "model-router.json").read_text()
    assert raw.count('"local/vision"') == 1
    cat = json.loads(raw)
    vis = cat["aliases"]["local/vision"]
    assert vis["fits_5080"] is True
    assert vis["latency"] == "ok-lan"
    assert "sm_86" in cat["hosts"]["akula-prime"]["caps"]
    assert "sm_120" in cat["hosts"]["gpu5080"]["caps"]
    assert "native-fp8" in cat["hosts"]["akula-prime"]["lacks"]


def test_cuda_eval_stays_on_5080(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = load_router(tmp_path, monkeypatch)
    spec = r.catalog()["aliases"]["cuda-eval"]
    assert r.place_host(spec, inv_idle()) == "gpu5080"
    assert r.caps_ok(spec, "akula-prime", inv_idle()) is False


def test_14b_not_placed_on_5080(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = load_router(tmp_path, monkeypatch)
    spec = r.catalog()["aliases"]["local/code"]
    assert r.place_host(spec, inv_idle()) == "akula-prime"
    assert spec["fits_5080"] is False


def test_migrate_8b_off_prime_for_14b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = load_router(tmp_path, monkeypatch)
    inventory = inv_idle(helper_ok=True, gguf=0)
    r.request_alias(
        "local/uncensored-fast",
        "seed",
        180,
        noping=True,
        now=1_000.0,
        inventory=inventory,
    )
    rec = r.request_alias(
        "local/code",
        "autodev",
        3600,
        noping=True,
        now=1_010.0,
        inventory=inventory,
    )
    assert rec["ok"] is True
    assert rec["host"] == "akula-prime"
    moved = rec.get("migrated") or []
    assert any(m.get("alias") == "local/uncensored-fast" for m in moved)
    st = r.load_state()
    parked = st["parked"]["local/uncensored-fast"]
    assert parked["physical"] is False
    assert parked["host"] == "gpu5080"


def test_no_ping_pong_within_dwell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = load_router(tmp_path, monkeypatch)
    inventory = inv_idle()
    r.request_alias(
        "local/uncensored-fast", "seed", 180, noping=True, now=1_000.0, inventory=inventory
    )
    r.request_alias("local/code", "autodev", 3600, noping=True, now=1_010.0, inventory=inventory)
    rest = r.restore_pass(now=1_010.0 + 60.0, inventory=inventory)
    assert rest["actions"] == []
    st = r.load_state()
    assert "local/uncensored-fast" in st["parked"]


def test_restore_after_dwell_when_14b_gone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = load_router(tmp_path, monkeypatch)
    inventory = inv_idle()
    r.request_alias(
        "local/uncensored-fast", "seed", 180, noping=True, now=1_000.0, inventory=inventory
    )
    r.request_alias("local/code", "autodev", 3600, noping=True, now=1_010.0, inventory=inventory)
    st = r.load_state()
    st["resident"].pop("local/code", None)
    r.save_state(st)
    idle = inv_idle()
    idle["akula-prime"] = {"used_mib": 10, "free_mib": 22900, "total_mib": 23028}
    later = 1_010.0 + 700.0
    rest = r.restore_pass(now=later, inventory=idle)
    ops = {a["alias"]: a["op"] for a in rest["actions"]}
    assert ops.get("local/uncensored-fast") == "restore-prime"


def test_drop_parked_when_5080_needs_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = load_router(tmp_path, monkeypatch)
    inventory = inv_idle()
    r.request_alias(
        "local/uncensored-fast", "seed", 180, noping=True, now=1_000.0, inventory=inventory
    )
    r.request_alias("local/code", "autodev", 3600, noping=True, now=1_010.0, inventory=inventory)
    busy = inv_idle(helper_ok=False)
    busy["gpu5080"]["lock"] = "1234"
    busy["akula-prime"]["free_mib"] = 2000
    rest = r.restore_pass(now=1_010.0 + 700.0, inventory=busy)
    ops = {a["alias"]: a["op"] for a in rest["actions"]}
    assert ops.get("local/uncensored-fast") == "drop-for-cuda"


def test_pool_plan_parked_not_executed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = load_router(tmp_path, monkeypatch)
    rec = r.request_alias("pool/large", "trial", 60, noping=True, now=1_000.0, inventory=inv_idle())
    assert rec["ok"] is False
    assert rec["error"] in {"pool-parked", "pool would preempt autodev"}
    plan = rec["plan"]
    assert plan["prime_layers_vram_mib"] + plan["gpu5080_layers_vram_mib"] == 36000
    assert plan["enabled"] is False
    # 24:16 of 36000 = 21600 + 14400
    assert plan["prime_layers_vram_mib"] == 21600


def test_tight_never_splits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = load_router(tmp_path, monkeypatch)
    spec = r.catalog()["aliases"]["local/code"]
    assert spec["latency"] == "tight"
    assert spec["pool_ok"] is False


def test_never_dual_14b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = load_router(tmp_path, monkeypatch)
    inventory = inv_idle()
    r.request_alias("local/fast", "chat", 300, noping=True, now=1_000.0, inventory=inventory)
    rec = r.request_alias(
        "local/code", "autodev", 3600, noping=True, now=1_020.0, inventory=inventory
    )
    assert rec["ok"] is True
    st = r.load_state()
    classes = [
        (r.catalog()["aliases"].get(a) or {}).get("class")
        for a in st["resident"]
        if (st["resident"][a].get("host") == "akula-prime")
    ]
    assert classes.count("14b") == 1


def test_1080ti_is_future_not_live() -> None:
    cat = json.loads((ROOT / "config" / "model-router.json").read_text())
    assert "gpu5080-1080ti" not in cat["hosts"]
    stub = cat["future_hosts"]["gpu5080-1080ti"]
    assert stub["live"] is False
    assert stub["installed"] is False
    assert stub["role"] == "retrieve-index-light"
    assert stub["sm"] == "6.1"
    assert "sm_120" in stub["lacks"]
    assert "fp8" in stub["lacks"]
    assert "fp4" in stub["lacks"]
    assert "live=false" in stub["note"]


def test_1080ti_never_scheduled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = load_router(tmp_path, monkeypatch)
    assert "gpu5080-1080ti" not in r.schedulable_hosts()
    inventory = inv_idle()
    sneak = {
        "host_pref": ["gpu5080-1080ti", "gpu5080"],
        "fits_5080": True,
        "needs_caps": [],
        "prefer_caps": [],
    }
    assert r.place_host(sneak, inventory) == "gpu5080"
    only_stub = {
        "host_pref": ["gpu5080-1080ti"],
        "fits_5080": True,
        "needs_caps": [],
        "prefer_caps": [],
    }
    assert r.place_host(only_stub, inventory) == "reject"
    embed = r.catalog()["aliases"]["embed-qwen3-0.6b"]
    assert r.place_host(embed, inventory) == "gpu5080"
    cuda = r.catalog()["aliases"]["cuda-eval"]
    assert r.place_host(cuda, inventory) == "gpu5080"


def test_embed_prefers_5080_caps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = load_router(tmp_path, monkeypatch)
    spec = r.catalog()["aliases"]["embed-qwen3-0.6b"]
    inventory = inv_idle(helper_ok=True)
    assert r.place_host(spec, inventory) == "gpu5080"
    assert r.prefer_score(spec, "gpu5080", inventory) > r.prefer_score(
        spec, "akula-prime", inventory
    )


def test_helper_aliases_list_multiple_hosts() -> None:
    """Helper/embed/light aliases may land on leftover 3090, share-small 5080,
    and gpu5080-1080ti when live. 14B stays on akula-prime.
    """
    cat = json.loads((ROOT / "config" / "model-router.json").read_text())
    assert cat["never_dual_14b"] is True
    assert cat["autodev_default"] == "local/code"
    code = cat["aliases"]["local/code"]
    assert code["host_pref"] == ["akula-prime"]
    assert code["class"] == "14b"
    assert code["fits_5080"] is False
    assert code["fits_1080ti"] is False
    embed = cat["aliases"]["embed-qwen3-0.6b"]
    assert embed["host_pref"][0] == "gpu5080-1080ti"
    assert "gpu5080" in embed["host_pref"]
    assert "akula-prime" in embed["host_pref"]
    assert "retrieve" in embed["jobs"]
    assert "rag-index" in embed["jobs"]
    eight = cat["aliases"]["local/uncensored-fast"]
    for host in ("akula-prime", "gpu5080", "gpu5080-1080ti"):
        assert host in eight["host_pref"]
        assert host in cat["aliases"]["local/vision"]["host_pref"]
    assert eight["share"]["akula-prime"] == "leftover"
    assert eight["share"]["gpu5080"] == "share-small"
    assert eight["share"]["gpu5080-1080ti"] == "when-live"


def test_rag_index_falls_back_to_5080_when_1080ti_not_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = load_router(tmp_path, monkeypatch)
    assert r.job_to_alias("rag-index") == "embed-qwen3-0.6b"
    rec = r.pick_alias("embed-qwen3-0.6b", inventory=inv_idle())
    assert rec["ok"] is True
    assert rec["host"] == "gpu5080"
    assert rec["host_pref"][0] == "gpu5080-1080ti"


def test_embed_prefers_1080ti_when_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cat = json.loads((ROOT / "config" / "model-router.json").read_text())
    stub = cat["future_hosts"].pop("gpu5080-1080ti")
    stub["live"] = True
    stub["installed"] = True
    cat["hosts"]["gpu5080-1080ti"] = stub
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(cat), encoding="utf-8")
    monkeypatch.setenv("CSD_ROUTER_CATALOG", str(path))
    monkeypatch.setenv("CSD_ROUTER_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("CSD_GPU_PLAN", str(tmp_path / "gpu-plan.json"))
    loader = importlib.machinery.SourceFileLoader("csd_model_router_live", str(SCRIPT))
    spec = importlib.util.spec_from_loader("csd_model_router_live", loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    mod.reset_catalog()
    inventory = inv_idle()
    inventory["gpu5080-1080ti"] = {
        "used_mib": 10,
        "free_mib": 11000,
        "total_mib": 11264,
    }
    rec = mod.pick_alias("embed-qwen3-0.6b", inventory=inventory)
    assert rec["ok"] is True
    assert rec["host"] == "gpu5080-1080ti"
    cuda = mod.catalog()["aliases"]["cuda-eval"]
    assert mod.place_host(cuda, inventory) == "gpu5080"


def test_pick_refuses_oom_and_exclusive_seq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = load_router(tmp_path, monkeypatch)
    busy = inv_idle(helper_ok=False)
    busy["gpu5080"]["lock"] = "1234"
    embed = r.catalog()["aliases"]["embed-qwen3-0.6b"]
    # Leftover on 3090 still FITs embed (~4000 of ~6494).
    assert r.place_host(embed, busy) == "akula-prime"
    busy["akula-prime"]["free_mib"] = 100
    rec = r.pick_alias("embed-qwen3-0.6b", inventory=busy)
    assert rec["ok"] is False
    assert rec["host"] == "reject"
    assert "oom" in rec["error"] or "exclusive-seq" in rec["error"]
    cuda = r.catalog()["aliases"]["cuda-eval"]
    assert r.place_host(cuda, busy) == "reject"
    rec14 = r.pick_alias("local/code", inventory=inv_idle())
    assert rec14["ok"] is True
    assert rec14["host"] == "akula-prime"


def test_cluster_backends_three_hosts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = load_router(tmp_path, monkeypatch)
    rec = r.cluster_backends()
    ids = [b["id"] for b in rec["backends"]]
    assert ids == ["akula-prime", "gpu5080", "gpu5080-1080ti"]
    assert rec["count"] == 3
    assert rec["never_dual_14b"] is True
    assert "csd-autodev" in rec["groups"]
    assert "akula-rag" in rec["groups"]
    live = {b["id"]: b["live"] for b in rec["backends"]}
    assert live["akula-prime"] is True
    assert live["gpu5080"] is True
    assert live["gpu5080-1080ti"] is False
