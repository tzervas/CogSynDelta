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
from cogsyndelta.data.stream import StreamSource


def _run_one(
    name: str,
    compactor: Compactor,
    x: torch.Tensor,
    min_fidelity: float,
    device_name: str,
    provenance: str,
) -> MetricsRecord:
    """Round-trip ``x`` through one compactor and record what came back.

    Args:
        name: Record name.
        compactor: The codec under measurement. Its algorithm is not touched here; only
            what it is measured on has changed.
        x: The batch to compress.
        min_fidelity: Pass threshold on cosine fidelity.
        device_name: Resolved device, for the record.
        provenance: Where ``x`` came from. Recorded because a fidelity number is
            meaningless without it -- a codec measured on isotropic noise is being
            measured near its best case.

    Returns:
        One :class:`MetricsRecord`.
    """
    blob = compactor.compact(x)
    recon = compactor.reconstruct(blob)
    original = x.cpu()
    fid = measured_fidelity(original, recon)
    # Cosine fidelity alone does not discriminate here. At the CLI defaults (embed_dim
    # 512, basis_rank 64, quant_bits 8, batch 16), cosine reads 0.999999-1.000000 across
    # synthetic, corpus:code and corpus:stsb -- six decimal places that all say "fine" --
    # while relative L2 over those same six runs (2 codecs x 3 streams) moves
    # 0.00020-0.00163, an 8x spread; that range is codec-to-codec, not stream-to-stream --
    # basis_residual barely moves across streams (0.00020-0.00021, 1.02x) while
    # calibrated_quant_8bit does (0.00049-0.00163, 3.32x), which is the discrimination
    # rel_l2 actually buys over cosine here. Sweeping quant_bits 4-16 on synthetic at
    # these defaults moves calibrated_quant's rel_l2_error 0.0274-0.0000063, a ~4370x
    # spread. Cosine is dominated by the direction these embeddings share, which the
    # codec preserves for free; relative L2 is not. Reporting both is the cheapest way to
    # stop a 0.99999 from being read as a strong result.
    rel_l2 = float(
        (original - recon).norm() / original.norm().clamp_min(torch.finfo(torch.float32).tiny)
    )
    ratio = compactor.measured_ratio(x, blob)
    status = MetricsStatus.PASS if fid >= min_fidelity else MetricsStatus.GAP
    return MetricsRecord(
        name=name,
        claim=f"fidelity>={min_fidelity} at measured ratio",
        measured={
            "fidelity": round(fid, 6),
            "rel_l2_error": round(rel_l2, 8),
            "ratio": round(ratio, 4),
            "stored_bytes": blob.stored_bytes,
            "original_bytes": blob.original_bytes,
            "method": blob.method,
            "stream": provenance,
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
    *,
    stream: StreamSource,
) -> list[MetricsRecord]:
    """Run basis-residual and calibrated-quant compactors on a batch from ``stream``.

    The compactors are unchanged. What changed is what they are measured on: this was
    ``torch.randn``, and a per-dimension min/max quantiser measured on isotropic
    Gaussians is measured close to its best case. Every dimension has the same spread,
    every sample is independent, and the calibration range is as tight as it can be.
    Real embeddings are not like that, so the number was describing the generator.

    ``stream`` is required and has no default, so no run can quietly go back to noise.

    Args:
        cfg: Compression hyperparameters.
        device_ctx: Resolved CPU/CUDA device.
        batch: Rows to draw from the stream.
        seed: Accepted for call-signature and CLI-config compatibility only. Neither
            compactor draws from the global torch RNG: :class:`BasisResidualCompactor`
            builds its basis from its own fixed ``torch.Generator``, and
            :class:`CalibratedQuantCompactor` has no RNG at all. ``stream`` sampling is
            seeded separately, by the caller, when it is constructed.
        stream: Where the ``[batch, embed_dim]`` embeddings come from.

    Returns:
        One MetricsRecord per compactor, each tagged with the stream provenance.

    Raises:
        ValueError: ``stream.dim`` does not match ``cfg.embed_dim``.
    """
    if stream.dim != cfg.embed_dim:
        raise ValueError(f"stream dim {stream.dim} != cfg.embed_dim {cfg.embed_dim}")
    x = stream.sample(batch).to(device_ctx.device)
    records: list[MetricsRecord] = []
    basis = BasisResidualCompactor(dim=cfg.embed_dim, basis_rank=cfg.basis_rank)
    records.append(_run_one("basis_residual", basis, x, 0.99, device_ctx.name, stream.provenance))
    quant = CalibratedQuantCompactor(bits=cfg.quant_bits)
    records.append(
        _run_one(
            f"calibrated_quant_{cfg.quant_bits}bit",
            quant,
            x,
            cfg.min_fidelity,
            device_ctx.name,
            stream.provenance,
        )
    )
    return records
