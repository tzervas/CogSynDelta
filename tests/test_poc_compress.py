"""Compression bench tests — shared memory substrate."""

from __future__ import annotations

from cogsyndelta.contracts.config import CompressionConfig
from cogsyndelta.contracts.device import DeviceContext
from cogsyndelta.contracts.metrics import MetricsStatus
from cogsyndelta.poc.compress import run_compression_bench


def test_compression_bench_basis_passes() -> None:
    cfg = CompressionConfig(
        embed_dim=128, basis_rank=32, quant_bits=8, min_fidelity=0.85
    )
    ctx = DeviceContext.resolve("cpu")
    records = run_compression_bench(cfg, ctx, batch=8, seed=7)
    by_name = {r.name: r for r in records}
    assert by_name["basis_residual"].status == MetricsStatus.PASS
    assert by_name["basis_residual"].measured["fidelity"] >= 0.99
