"""PoC compression bench — shared memory substrate for all regions."""

from __future__ import annotations

import torch

from cogsyndelta.contracts.compactor import (
    BasisResidualCompactor,
    CalibratedQuantCompactor,
    Compactor,
    measured_fidelity,
)
from cogsyndelta.contracts.config import CompressionConfig
from cogsyndelta.contracts.device import DeviceContext
from cogsyndelta.contracts.metrics import MetricsRecord, MetricsStatus


def _run_one(
    name: str,
    compactor: Compactor,
    x: torch.Tensor,
    min_fidelity: float,
    device_name: str,
) -> MetricsRecord:
    blob = compactor.compact(x)
    recon = compactor.reconstruct(blob)
    fid = measured_fidelity(x.cpu(), recon)
    ratio = compactor.measured_ratio(x, blob)
    status = MetricsStatus.PASS if fid >= min_fidelity else MetricsStatus.GAP
    return MetricsRecord(
        name=name,
        claim=f"fidelity>={min_fidelity} at measured ratio",
        measured={
            "fidelity": round(fid, 6),
            "ratio": round(ratio, 4),
            "stored_bytes": blob.stored_bytes,
            "original_bytes": blob.original_bytes,
            "method": blob.method,
        },
        status=status,
        device=device_name,
        notes="original to bytes to reconstruct",
    )


def run_compression_bench(
    cfg: CompressionConfig,
    device_ctx: DeviceContext,
    batch: int = 16,
    seed: int = 42,
) -> list[MetricsRecord]:
    torch.manual_seed(seed)
    x = torch.randn(batch, cfg.embed_dim, device=device_ctx.device)
    records: list[MetricsRecord] = []
    basis = BasisResidualCompactor(dim=cfg.embed_dim, basis_rank=cfg.basis_rank)
    records.append(_run_one("basis_residual", basis, x, 0.99, device_ctx.name))
    quant = CalibratedQuantCompactor(bits=cfg.quant_bits)
    records.append(
        _run_one(
            f"calibrated_quant_{cfg.quant_bits}bit",
            quant,
            x,
            cfg.min_fidelity,
            device_ctx.name,
        )
    )
    return records
