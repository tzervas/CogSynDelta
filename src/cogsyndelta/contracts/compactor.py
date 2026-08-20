"""Compactor protocol and honest reference implementations.

Invariant: measured_fidelity always compares original vs reconstruct(compact(original)).
Ratio is original_bytes / stored_bytes of the serialized blob — not theoretical dim counts.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F


@dataclass
class CompactBlob:
    """Serialized compressed representation with measured byte size."""

    payload: bytes
    original_numel: int
    original_dtype: str
    method: str

    @property
    def stored_bytes(self) -> int:
        """Serialized payload size in bytes."""
        return len(self.payload)

    @property
    def original_bytes(self) -> int:
        """Uncompressed tensor size in bytes (float32=4, else 2)."""
        # float32 default; adjust if dtype known
        itemsize = 4 if self.original_dtype in ("float32", "torch.float32") else 2
        return self.original_numel * itemsize


def measured_fidelity(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Cosine similarity between original and reconstructed (batch-mean if 2D)."""
    o = (
        original.detach().float().reshape(original.shape[0], -1)
        if original.dim() > 1
        else original.detach().float().unsqueeze(0)
    )
    r = (
        reconstructed.detach().float().reshape(reconstructed.shape[0], -1)
        if reconstructed.dim() > 1
        else reconstructed.detach().float().unsqueeze(0)
    )
    if o.shape != r.shape:
        raise ValueError(f"Shape mismatch original {o.shape} vs reconstructed {r.shape}")
    sim = F.cosine_similarity(o, r, dim=-1)
    return float(sim.mean().item())


@runtime_checkable
class Compactor(Protocol):
    """Common surface for compression feature lane."""

    name: str

    def compact(self, x: torch.Tensor) -> CompactBlob:
        """Compress tensor to a serializable blob."""
        ...

    def reconstruct(self, blob: CompactBlob) -> torch.Tensor:
        """Reconstruct tensor from blob (must round-trip through bytes)."""
        ...

    def measured_ratio(self, x: torch.Tensor, blob: CompactBlob) -> float:
        """original_bytes / stored_bytes."""
        ...


class BasisResidualCompactor:
    """Orthonormal-ish projection + residual storage.

    Honest about ratio: residual is stored, so compression is modest (~1.1–1.5x)
    when residual is kept at full precision. Fidelity approaches 1.0 by design.
    This is NOT a 10x codec — it is a safe high-fidelity baseline.
    """

    name = "basis_residual"

    def __init__(self, dim: int = 512, basis_rank: int = 64) -> None:
        """Build a fixed random basis of shape ``[dim, rank]``.

        Args:
            dim: Input embedding dimension.
            basis_rank: Number of basis columns (clamped to ``dim``).
        """
        self.dim = dim
        self.basis_rank = min(basis_rank, dim)
        # Fixed random orthonormal-ish basis (deterministic seed for tests)
        g = torch.Generator().manual_seed(42)
        raw = torch.randn(dim, self.basis_rank, generator=g)
        q, _ = torch.linalg.qr(raw)
        self.basis = q  # [dim, rank]

    def compact(self, x: torch.Tensor) -> CompactBlob:
        """Project onto the basis and store coeffs plus residual as float16.

        Args:
            x: Input tensor ``[D]`` or ``[B, D]``.

        Returns:
            Blob whose reconstruct path recovers proj + residual.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_cpu = x.detach().float().cpu()
        # coeffs = x @ basis; proj = coeffs @ basis.T; residual = x - proj
        coeffs = x_cpu @ self.basis  # [B, rank]
        proj = coeffs @ self.basis.T
        residual = x_cpu - proj
        buf = io.BytesIO()
        torch.save(
            {
                "coeffs": coeffs.to(torch.float16),
                "residual": residual.to(torch.float16),
                "shape": list(x_cpu.shape),
            },
            buf,
        )
        payload = buf.getvalue()
        return CompactBlob(
            payload=payload,
            original_numel=int(x_cpu.numel()),
            original_dtype="float32",
            method=self.name,
        )

    def reconstruct(self, blob: CompactBlob) -> torch.Tensor:
        """Reload coeffs/residual from bytes and add them back.

        Args:
            blob: Output of ``compact``.

        Returns:
            Reconstructed tensor on CPU.
        """
        data = torch.load(io.BytesIO(blob.payload), weights_only=True, map_location="cpu")
        coeffs = data["coeffs"].float()
        residual = data["residual"].float()
        proj = coeffs @ self.basis.T
        return proj + residual

    def measured_ratio(self, x: torch.Tensor, blob: CompactBlob) -> float:
        """Return original_bytes / stored_bytes for this blob.

        Args:
            x: Unused; ratio is taken from the blob metadata.
            blob: Compacted payload.

        Returns:
            Byte ratio, or 0.0 if original_bytes is 0.
        """
        orig = blob.original_bytes
        if orig <= 0:
            return 0.0
        return orig / blob.stored_bytes


class CalibratedQuantCompactor:
    """Per-dimension calibrated uniform quantization (ADR-0008 Stage 1).

    Lossy. Ratio depends on bits. Fidelity measured honestly after byte round-trip.
    """

    name = "calibrated_quant"

    def __init__(self, bits: int = 8) -> None:
        """Configure uniform quantizer width.

        Args:
            bits: Quantization bits in ``[2, 16]``.

        Raises:
            ValueError: If ``bits`` is outside that range.
        """
        if bits < 2 or bits > 16:
            raise ValueError("bits must be in [2, 16]")
        self.bits = bits
        self._levels = (1 << bits) - 1

    def compact(self, x: torch.Tensor) -> CompactBlob:
        """Calibrate per-dim min/max on this batch and store uint8 codes.

        Args:
            x: Input tensor ``[D]`` or ``[B, D]``.

        Returns:
            Blob with codes, vmin, and scale.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_cpu = x.detach().float().cpu()
        # Per-dimension calibration on this batch (PoC; production would use fixed stats)
        vmin = x_cpu.min(dim=0).values
        vmax = x_cpu.max(dim=0).values
        scale = (vmax - vmin).clamp_min(1e-8) / self._levels
        q = torch.round((x_cpu - vmin) / scale).clamp(0, self._levels).to(torch.uint8)
        buf = io.BytesIO()
        torch.save(
            {
                "q": q,
                "vmin": vmin.to(torch.float16),
                "scale": scale.to(torch.float16),
                "bits": self.bits,
                "shape": list(x_cpu.shape),
            },
            buf,
        )
        return CompactBlob(
            payload=buf.getvalue(),
            original_numel=int(x_cpu.numel()),
            original_dtype="float32",
            method=self.name,
        )

    def reconstruct(self, blob: CompactBlob) -> torch.Tensor:
        """Dequantize ``q * scale + vmin`` from the serialized blob.

        Args:
            blob: Output of ``compact``.

        Returns:
            Reconstructed tensor on CPU.
        """
        data = torch.load(io.BytesIO(blob.payload), weights_only=True, map_location="cpu")
        q = data["q"].float()
        vmin = data["vmin"].float()
        scale = data["scale"].float()
        return q * scale + vmin

    def measured_ratio(self, x: torch.Tensor, blob: CompactBlob) -> float:
        """Return original_bytes / stored_bytes for this blob.

        Args:
            x: Unused; ratio is taken from the blob metadata.
            blob: Compacted payload.

        Returns:
            Byte ratio, or 0.0 if original_bytes is 0.
        """
        orig = blob.original_bytes
        if orig <= 0:
            return 0.0
        return orig / blob.stored_bytes
