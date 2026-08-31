"""Tests for interned RO weights and parallel batch scheduling."""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "csd-weight-intern"


def load_mod() -> Any:
    loader = importlib.machinery.SourceFileLoader("csd_weight_intern", str(SCRIPT))
    spec = importlib.util.spec_from_loader("csd_weight_intern", loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _tiny(name: str, shared: bool = True, unique: bool = True) -> dict[str, Any]:
    tensors = []
    if shared:
        tensors.append({"name": "embed", "hash": "emb1", "nbytes": 80_000_000, "writable": False})
    if unique:
        tensors.append(
            {"name": f"head-{name}", "hash": f"h-{name}", "nbytes": 1_000_000, "writable": False}
        )
    tensors.append({"name": "kv", "hash": f"kv-{name}", "nbytes": 500_000, "writable": True})
    return {"name": name, "tensors": tensors}


def test_identical_weights_intern_once() -> None:
    intern = load_mod()
    models = [_tiny(f"m{i}") for i in range(8)]
    plan = intern.intern_plan(models)
    assert plan["pool"]["llama.cpp:emb1"]["refs"] == 8
    # 80M embed once + 8*1M heads + 8*0.5M kv
    assert plan["unique_bytes"] == 80_000_000 + 8 * 1_000_000 + 8 * 500_000
    assert plan["saved_bytes"] == 7 * 80_000_000


def test_writable_never_interned() -> None:
    intern = load_mod()
    plan = intern.intern_plan([_tiny("a"), _tiny("b")])
    assert all(not k.startswith("kv-") for k in plan["pool"])
    assert any("kv" in x for x in plan["writable_private"])


def test_parallel_schedule_batches_shared_gemm() -> None:
    intern = load_mod()
    sched = intern.parallel_schedule([_tiny(f"m{i}") for i in range(4)])
    hashes = {b["hash"] for b in sched["shared_batches"]}
    assert "llama.cpp:emb1" in hashes
    batch = next(b for b in sched["shared_batches"] if b["hash"] == "llama.cpp:emb1")
    assert batch["parallel"] is True
    assert len(batch["models"]) == 4
    assert sched["serial_forwards_avoided"] >= 1


def test_pick_runtime_bitnet_vllm_gguf() -> None:
    intern = load_mod()
    assert intern.pick_runtime({"format": "bitnet-1.58"}) == "bitnet-cpp"
    assert intern.pick_runtime({"format": "safetensors", "kind": "multi-lora"}) == "vllm"
    assert intern.pick_runtime({"format": "gguf"}) == "llama.cpp"
    assert intern.pick_runtime({"kind": "moe-large", "format": "gguf"}) == "llama.cpp"
    assert intern.pick_runtime({"kind": "moe", "format": "safetensors"}) == "vllm"
    plan = intern.intern_plan(
        [
            {
                "name": "a",
                "format": "gguf",
                "tensors": [{"name": "w", "hash": "x", "nbytes": 8, "writable": False}],
            },
            {
                "name": "b",
                "format": "bitnet-1.58",
                "tensors": [{"name": "w", "hash": "y", "nbytes": 8, "writable": False}],
            },
        ]
    )
    assert plan["mix"] == ["bitnet-cpp", "llama.cpp"]
    assert "llama.cpp:x" in plan["pool"]
    assert "bitnet-cpp:y" in plan["pool"]


def test_no_share_means_no_false_batch() -> None:
    intern = load_mod()
    models = [_tiny("solo", shared=False), _tiny("other", shared=False)]
    sched = intern.parallel_schedule(models)
    assert sched["shared_batches"] == []
