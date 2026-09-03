"""Tests for `cogsyndelta.util.gpu_budget` -- the gpu-pack env-var adapter.

All CUDA calls are monkeypatched; nothing here touches a real device, so it runs
anywhere (CI, CPU-only hosts, the 1080 Ti's torch build) exactly like
`test_pretrain_resume.py` does for the harness it adapts.
"""

from __future__ import annotations

import torch

from cogsyndelta.util import gpu_budget


class _FakeProps:
    def __init__(self, total_memory: int) -> None:
        self.total_memory = total_memory


def test_apply_budget_sets_fraction_when_set(monkeypatch) -> None:
    calls: list[tuple[float, int]] = []
    monkeypatch.setenv("GPU_PACK_BUDGET_MIB", "4096")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # 8192 MiB total device -> 4096/8192 = 0.5 fraction.
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx: _FakeProps(8192 * 2**20))
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda frac, idx: calls.append((frac, idx)),
    )

    fraction = gpu_budget.apply_budget_from_env()

    assert fraction == 0.5
    assert calls == [(0.5, 0)]


def test_apply_budget_is_noop_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("GPU_PACK_BUDGET_MIB", raising=False)
    called = False

    def boom(*_a, **_k):
        nonlocal called
        called = True

    monkeypatch.setattr(torch.cuda, "is_available", boom)

    assert gpu_budget.apply_budget_from_env() is None
    assert called is False


def test_apply_budget_is_noop_without_cuda(monkeypatch) -> None:
    monkeypatch.setenv("GPU_PACK_BUDGET_MIB", "4096")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert gpu_budget.apply_budget_from_env() is None


def test_apply_budget_rejects_non_integer(monkeypatch) -> None:
    monkeypatch.setenv("GPU_PACK_BUDGET_MIB", "not-a-number")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    try:
        gpu_budget.apply_budget_from_env()
    except ValueError as exc:
        assert "GPU_PACK_BUDGET_MIB" in str(exc)
    else:
        raise AssertionError("expected ValueError for a non-integer budget")


def test_apply_budget_clamps_to_one(monkeypatch) -> None:
    calls: list[tuple[float, int]] = []
    monkeypatch.setenv("GPU_PACK_BUDGET_MIB", "999999")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx: _FakeProps(8192 * 2**20))
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda frac, idx: calls.append((frac, idx)),
    )

    fraction = gpu_budget.apply_budget_from_env()

    assert fraction == 1.0
    assert calls == [(1.0, 0)]


def test_probe_requested(monkeypatch) -> None:
    monkeypatch.delenv("GPU_PACK_PROBE", raising=False)
    assert gpu_budget.probe_requested() is False

    monkeypatch.setenv("GPU_PACK_PROBE", "1")
    assert gpu_budget.probe_requested() is True

    monkeypatch.setenv("GPU_PACK_PROBE", "0")
    assert gpu_budget.probe_requested() is False


def test_report_peak_prints_once_with_int(monkeypatch, capsys) -> None:
    monkeypatch.setenv("GPU_PACK_BUDGET_MIB", "4096")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda idx: 1234 * 2**20 + 7)

    peak = gpu_budget.report_peak()

    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line.startswith("GPU_PACK_PEAK_MIB=")]
    assert lines == [f"GPU_PACK_PEAK_MIB={peak}"]
    assert peak == 1235  # ceil(1234 + 7/2**20)
    assert isinstance(peak, int)


def test_report_peak_prints_once_when_probe_set(monkeypatch, capsys) -> None:
    monkeypatch.delenv("GPU_PACK_BUDGET_MIB", raising=False)
    monkeypatch.setenv("GPU_PACK_PROBE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda idx: 1 * 2**20)

    peak = gpu_budget.report_peak()

    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line.startswith("GPU_PACK_PEAK_MIB=")]
    assert lines == [f"GPU_PACK_PEAK_MIB={peak}"]


def test_report_peak_is_noop_without_cuda(monkeypatch, capsys) -> None:
    monkeypatch.setenv("GPU_PACK_BUDGET_MIB", "4096")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    peak = gpu_budget.report_peak()

    captured = capsys.readouterr()
    assert peak is None
    assert "GPU_PACK_PEAK_MIB" not in captured.err


def test_report_peak_is_silent_when_unset(monkeypatch, capsys) -> None:
    monkeypatch.delenv("GPU_PACK_BUDGET_MIB", raising=False)
    monkeypatch.delenv("GPU_PACK_PROBE", raising=False)
    called = False

    def boom(*_a, **_k):
        nonlocal called
        called = True

    monkeypatch.setattr(torch.cuda, "is_available", boom)

    peak = gpu_budget.report_peak()

    captured = capsys.readouterr()
    assert peak is None
    assert captured.err == ""
    assert called is False  # must not even touch CUDA when not opted in
