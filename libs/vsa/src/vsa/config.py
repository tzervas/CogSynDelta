"""VSA configuration dataclass."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class VSAConfig:
    """Configuration for Vector Symbolic Architecture operations.

    Attributes:
        dimension: Dimensionality of hypervectors (default: 10000).
                  Higher dimensions increase capacity and robustness.
                  Capacity bound: n ≥ k/(1-S²) × log(M)
        model: VSA model type (default: "FHRR").
              - FHRR: Fourier Holographic Reduced Representations (complex phasors)
              - MAP: Multiply-Add-Permute
              - BSC: Binary Spatter Codes
        device: Compute device ("cuda" or "cpu", default: "cuda")
        dtype: Data type for hypervectors.
              - torch.cfloat (complex64) for FHRR
              - torch.float32 for MAP
              - torch.bool for BSC
        seed: Random seed for reproducibility (default: None)
    """

    dimension: int = 10000
    model: Literal["FHRR", "MAP", "BSC"] = "FHRR"
    device: str = "cuda"
    dtype: str | None = None  # Auto-selected based on model if None
    seed: int | None = None
