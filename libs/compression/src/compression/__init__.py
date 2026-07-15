"""Advanced compression pipeline for CogSynDelta.

This module implements the staged compression strategy (ADR-0008/ADR-0014)
targeting ≥0.95 fidelity at 10x+ compression.

Compression Stages:
1. Matryoshka Representation Learning (MRL): 2-4× via dimension truncation
2. QINCo2 Neural Codebooks: 4-8× with 34-44% MSE reduction vs RVQ
3. Residual Vector Quantization (RVQ): 8-16× with staged refinement
4. BitNet b1.58: 10× weight compression (2 bytes/param → 0.2 bytes/param)

Target: ≥0.95 fidelity at 10-16× compression

References:
    - Kusupati et al. (2022). Matryoshka Representation Learning. NeurIPS.
    - Vallaeys et al. (2025). Qinco2: Vector Compression and Search with
      Improved Implicit Neural Codebooks. ICLR.
    - Ma et al. (2024). The Era of 1-bit LLMs: All Large Language Models
      are in 1.58 Bits.
"""

__version__ = "0.1.0"

from compression.config import CompressionConfig
from compression.mrl.matryoshka import MatryoshkaCompressor, MatryoshkaEncoder
from compression.qinco.codebook import QINCo2Compressor
from compression.rvq.quantizer import ResidualVectorQuantizer
from compression.bitnet.ternary import BitNetb158, TernaryQuantizer
from compression.pipeline import StagedCompressionPipeline

__all__ = [
    "CompressionConfig",
    "MatryoshkaCompressor",
    "MatryoshkaEncoder",
    "QINCo2Compressor",
    "ResidualVectorQuantizer",
    "BitNetb158",
    "TernaryQuantizer",
    "StagedCompressionPipeline",
]
