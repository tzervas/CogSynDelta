"""Units for gpu5080 Comfy lock wrap, flock serialize, and lab comfy field."""

from __future__ import annotations

import asyncio
import importlib.machinery
import importlib.util
import os
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WRAP = ROOT / "deploy/gpu5080/akula-comfyui-locked"
DROPIN = ROOT / "deploy/gpu5080/akula-comfyui.service.d/gpu5080-lock.conf"
PROXY = ROOT / "deploy/gpu5080/akula-media-proxy.service.d/gpu5080-lan.conf"
HOOK = ROOT / "deploy/gpu5080/comfy-queue-hook.md"
PRIORITY = ROOT / "scripts/csd-autodev-priority"
PLAN = ROOT / "scripts/csd-gpu-plan"
LAB = ROOT / "scripts/csd-lab-console"


def load_script(path: Path, name: str) -> Any:
    """Import a repo script (no .py suffix) as a module."""
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_dropin_execstart_is_lock_wrapper() -> None:
    """systemd drop-in must claim gpu5080-lock the same way CUDA CI does."""
    text = DROPIN.read_text(encoding="utf-8")
    assert "ExecStart=/home/tzervas/akula-harness/scripts/akula-comfyui-locked" in text
    helper = "GPU5080_LOCK_HELPER=/home/tzervas/akula-harness/scripts/gpu5080-lock"
    assert helper in text
    assert "COMFY_LAN_BIND=192.168.1.251" in text
    assert "COMFY_GPU_BIND=127.0.0.1" in text
    assert "Do not mask to schedule" in text
    binds = [
        ln
        for ln in text.splitlines()
        if ln.startswith("Environment=COMFY_") or ln.startswith("ExecStart=")
    ]
    assert binds, "drop-in missing ExecStart/Environment"
    assert not any("0.0.0.0" in ln for ln in binds)
    assert WRAP.is_file()
    assert os.access(WRAP, os.X_OK)
    hook = HOOK.read_text(encoding="utf-8")
    assert "gpu5080-lock" in hook
    assert "192.168.1.251:8188" in hook
    proxy = PROXY.read_text(encoding="utf-8")
    assert "bind=172.32.0.1" in proxy
    assert "TCP:192.168.1.251:8188" in proxy
    execs = [ln for ln in proxy.splitlines() if ln.startswith("ExecStart=")]
    assert execs and not any("0.0.0.0" in ln for ln in execs)


def test_wrapper_refuses_wan_bind() -> None:
    """LAN and GPU publish must not be 0.0.0.0 WAN."""
    mod = load_script(WRAP, "akula_comfyui_locked")
    assert "0.0.0.0" in mod.WAN_BINDS
    mod.LAN_BIND = "0.0.0.0"
    mod.GPU_BIND = "127.0.0.1"
    assert asyncio.run(mod.amain()) == 2
    mod.LAN_BIND = "192.168.1.251"
    mod.GPU_BIND = "0.0.0.0"
    assert asyncio.run(mod.amain()) == 2


def test_wrapper_idle_stubs_have_no_cuda() -> None:
    """Idle GET paths return stub JSON without taking the flock."""
    mod = load_script(WRAP, "akula_comfyui_locked")
    assert "/system_stats" in mod.IDLE_PATHS
    assert "/queue" in mod.IDLE_PATHS
    stats = mod._stub_for("/system_stats")
    assert b"idle HTTP" in stats
    assert b"gpu5080-lock" in stats
    queue = mod._stub_for("/queue")
    assert b"queue_running" in queue
    assert mod.LAN_BIND == "192.168.1.251"
    assert mod.GPU_BIND == "127.0.0.1"
    assert mod.LOCK_HELPER.endswith("gpu5080-lock")


def test_wrapper_missing_lock_helper_exits(tmp_path: Path) -> None:
    """Refuse start if the CUDA CI lock helper is not executable."""
    mod = load_script(WRAP, "akula_comfyui_locked")
    mod.LAN_BIND = "127.0.0.1"
    mod.GPU_BIND = "127.0.0.1"
    mod.LOCK_HELPER = str(tmp_path / "no-such-gpu5080-lock")
    assert asyncio.run(mod.amain()) == 2


def test_two_lock_holders_serialize(tmp_path: Path) -> None:
    """Two gpu5080-lock holders must not overlap (finish-then-free).

    Why: same flock semantics as CUDA CI. Uses a temp lock, never the live
    5080 CUDA mutex.
    """
    lock = tmp_path / "gpu5080.lock"
    helper = tmp_path / "gpu5080-lock"
    helper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'LOCK="{lock}"\n'
        "WAIT=8\n"
        '[[ "${1:-}" == "--" ]] && shift\n'
        'exec 9>"$LOCK"\n'
        'flock -w "$WAIT" 9\n'
        'echo "gpu5080-lock: acquired pid=$$" >&2\n'
        'exec "$@"\n',
        encoding="utf-8",
    )
    helper.chmod(helper.stat().st_mode | stat.S_IXUSR)
    a_out = tmp_path / "a.out"
    b_out = tmp_path / "b.out"
    inner_a = f"echo A_acquired; sleep 1.2; echo A_released > {a_out}"
    inner_b = f"echo B_acquired > {b_out}"
    holder_a = subprocess.Popen(
        [str(helper), "--", "bash", "-lc", inner_a],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.25)
    holder_b = subprocess.Popen(
        [str(helper), "--", "bash", "-lc", inner_b],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rc_a = holder_a.wait(timeout=10)
    rc_b = holder_b.wait(timeout=10)
    assert rc_a == 0 and rc_b == 0
    a_err = holder_a.stderr.read() if holder_a.stderr else ""
    b_err = holder_b.stderr.read() if holder_b.stderr else ""
    assert "acquired" in a_err
    assert "acquired" in b_err
    assert a_out.is_file() and b_out.is_file()
    assert b_out.stat().st_mtime >= a_out.stat().st_mtime


def test_comfy_lab_field_wrap_ready_is_not_masked() -> None:
    """Lab status comfy field is wrapped when the lock drop-in is live."""
    plan = load_script(PLAN, "csd_gpu_plan")
    lab = load_script(LAB, "csd_lab_console")
    assert plan.comfy_lab_field("generated", True) == "wrapped"
    assert plan.comfy_lab_field("enabled", True) == "wrapped"
    assert plan.comfy_lab_field("masked", True) == "masked"
    assert plan.comfy_lab_field("generated", False) == "generated"
    assert plan.comfy_lab_field("", False) == "unknown"
    assert lab.comfy_lab_field("generated", True) == "wrapped"
    assert lab.comfy_lab_field("masked", False) == "masked"
    assert lab.parse_comfy_ssh("generated\nwrap-ready") == "wrapped"
    assert lab.parse_comfy_ssh("masked\nwrap-ready") == "masked"
    assert lab.parse_comfy_ssh("generated\nwrap-missing") == "generated"
    comfy, lock, gguf = plan.parse_5080_aux(["generated", "wrap-ready", "lock=idle", "2"])
    assert comfy == "wrapped"
    assert "idle" in lock
    assert gguf == 2
    old_comfy, old_lock, old_gguf = plan.parse_5080_aux(["masked", "lock=idle", "0"])
    assert old_comfy == "masked"
    assert old_lock == "lock=idle"
    assert old_gguf == 0


def test_helper_ok_wrapped_idle_not_free_running() -> None:
    """Share-small is OK when wrap-ready + lock idle; not when enabled bare."""
    plan = load_script(PLAN, "csd_gpu_plan")
    assert plan.helper_ok_5080(8000, "wrapped", "lock=idle") is True
    assert plan.helper_ok_5080(8000, "masked", "lock=idle") is True
    assert plan.helper_ok_5080(8000, "generated", "lock=idle") is False
    assert plan.helper_ok_5080(8000, "wrapped", "1234") is False
    assert plan.helper_ok_5080(4000, "wrapped", "lock=idle") is False


def test_autodev_priority_does_not_mask_comfy() -> None:
    """Priority on must not systemctl mask Comfy; wrap is the mutex."""
    text = PRIORITY.read_text(encoding="utf-8")
    assert "systemctl mask akula-comfyui" not in text
    assert "lock wrap" in text.lower() or "lock-wrap" in text.lower()
    assert "akula-comfyui-locked" in text


def test_acquire_lock_waits_for_acquired_line(tmp_path: Path) -> None:
    """ComfyLockProxy holds the helper until stderr prints acquired."""
    helper = tmp_path / "gpu5080-lock"
    helper.write_text(
        "#!/usr/bin/env bash\necho 'gpu5080-lock: acquired pid=$$' >&2\nexec sleep 30\n",
        encoding="utf-8",
    )
    helper.chmod(helper.stat().st_mode | stat.S_IXUSR)
    mod = load_script(WRAP, "akula_comfyui_locked")
    mod.LOCK_HELPER = str(helper)

    async def _run() -> None:
        proxy = mod.ComfyLockProxy()
        await proxy._acquire_lock()
        assert proxy._lock_proc is not None
        await proxy._release_lock()
        assert proxy._lock_proc is None

    asyncio.run(_run())
