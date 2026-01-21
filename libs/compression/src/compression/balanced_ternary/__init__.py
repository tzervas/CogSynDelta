"""Balanced Ternary Tryte System for CogSynDelta.

Balanced ternary uses digits {-1, 0, +1} (also written as {T, 0, 1} or {-, 0, +})
for efficient symmetric representation without explicit sign bit.

Key advantages:
- Symmetric around zero (no bias)
- Natural signed arithmetic
- Unique representation (no positive/negative zero)
- Efficient rounding (just truncate)
- Brain-like activation patterns (inhibit/neutral/excite)

Tryte: 9 trits (balanced ternary digits)
- Range: -9841 to +9841 (3^9)
- Precision: ~13 bits equivalent
- Memory: ~1.58 bits per trit (same as BitNet b1.58)

References:
    - Knuth (1981). The Art of Computer Programming Vol 2: Balanced ternary
    - Setun computer (1958): First ternary computer (Soviet Union)
    - Ma et al. (2024). 1.58-bit LLMs: Similar efficiency to balanced ternary
"""

from compression.balanced_ternary.arithmetic import (
    BalancedTernaryArithmetic,
    BalancedTernaryTensor,
    BalancedTernaryTryte,
    tryte_decode,
    tryte_encode,
)
from compression.balanced_ternary.layers import (
    BalancedTernaryConv2d,
    BalancedTernaryEmbedding,
    BalancedTernaryLinear,
)
from compression.balanced_ternary.quantizer import (
    BalancedTernaryCompressor,
    BalancedTernaryQuantizer,
)

__all__ = [
    "BalancedTernaryArithmetic",
    "BalancedTernaryCompressor",
    "BalancedTernaryConv2d",
    "BalancedTernaryEmbedding",
    "BalancedTernaryLinear",
    "BalancedTernaryQuantizer",
    "BalancedTernaryTensor",
    "BalancedTernaryTryte",
    "tryte_decode",
    "tryte_encode",
]
