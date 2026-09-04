"""Compression bench tests — shared memory substrate."""

from __future__ import annotations

import pytest

from cogsyndelta.contracts.config import CompressionConfig
from cogsyndelta.contracts.device import DeviceContext
from cogsyndelta.contracts.metrics import MetricsStatus
from cogsyndelta.data.stream import SyntheticStream
from cogsyndelta.poc.compress import run_compression_bench


def test_compression_bench_basis_passes() -> None:
    cfg = CompressionConfig(embed_dim=128, basis_rank=32, quant_bits=8, min_fidelity=0.85)
    ctx = DeviceContext.resolve("cpu")
    # SyntheticStream, named explicitly. `stream` is keyword-only and has no default
    # precisely so a bench cannot quietly fall back to noise; a test is no exception,
    # and its provenance marks every record as noise-derived.
    stream = SyntheticStream(cfg.embed_dim, "cpu", seed=7)
    records = run_compression_bench(cfg, ctx, batch=8, seed=7, stream=stream)
    by_name = {r.name: r for r in records}
    assert by_name["basis_residual"].status == MetricsStatus.PASS
    assert by_name["basis_residual"].measured["fidelity"] >= 0.99
    measured = by_name["basis_residual"].measured
    assert "rel_l2_error" in measured
    assert measured["stream"] == stream.provenance


def test_compression_bench_rejects_stream_dim_mismatch() -> None:
    cfg = CompressionConfig(embed_dim=128, basis_rank=32, quant_bits=8, min_fidelity=0.85)
    ctx = DeviceContext.resolve("cpu")
    stream = SyntheticStream(cfg.embed_dim // 2, "cpu", seed=7)
    with pytest.raises(ValueError):
        run_compression_bench(cfg, ctx, batch=8, seed=7, stream=stream)
