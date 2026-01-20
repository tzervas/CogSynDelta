"""Vector Symbolic Architecture (VSA) for CogSynDelta.

This module implements VSA operations using torchhd for GPU-accelerated
hyperdimensional computing with FHRR (Fourier Holographic Reduced Representations).

Core components:
- Fractional Power Encoding (FPE) for temporal grounding
- Binding, bundling, and permutation operations
- Modern Hopfield Networks for associative memory
- Resonator networks for compositional factorization

References:
    - Plate, T. A. (2003). Holographic reduced representation.
    - Heddes et al. (2023). Torchhd: An Open Source Python Library to Support
      Research on HDC and VSA. JMLR, 24(255).
    - Ramsauer et al. (2021). Hopfield Networks is All You Need. ICLR.
"""

__version__ = "0.1.0"

from vsa.config import VSAConfig
from vsa.encoders.fpe import FractionalPowerEncoder
from vsa.operations.binding import bind, unbind
from vsa.operations.bundling import bundle
from vsa.operations.permutation import permute
from vsa.memory.hopfield import ModernHopfieldMemory
from vsa.memory.resonator import ResonatorNetwork

__all__ = [
    "VSAConfig",
    "FractionalPowerEncoder",
    "bind",
    "unbind",
    "bundle",
    "permute",
    "ModernHopfieldMemory",
    "ResonatorNetwork",
]
