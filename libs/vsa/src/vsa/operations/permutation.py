"""Permutation operations for VSA.

Permutation provides ordering and role distinction in compositional structures.
Useful for sequences and structured data.

Properties:
- Invertible: ρ^(-1)(ρ(A)) = A
- Can represent order: (A, B) ≠ (B, A) via ρ(A) ⊗ B vs ρ(B) ⊗ A
"""

import torch
import torchhd


def permute(x: torch.Tensor, shifts: int = 1) -> torch.Tensor:
    """Permute a hypervector (cyclic shift).

    For FHRR: element-wise phase rotation
    For others: coordinate permutation

    Args:
        x: Hypervector to permute (shape: [dim] or [batch, dim])
        shifts: Number of positions to shift (default: 1)

    Returns:
        Permuted hypervector (same shape as input)

    Example:
        >>> import torch
        >>> from vsa.operations.permutation import permute
        >>> x = torch.randn(10000, dtype=torch.cfloat)
        >>> y = torch.randn(10000, dtype=torch.cfloat)
        >>> # Ordered pair: (x, y)
        >>> xy = bind(permute(x, 0), permute(y, 1))
        >>> # Different from (y, x)
        >>> yx = bind(permute(y, 0), permute(x, 1))
        >>> # xy and yx are dissimilar
    """
    return torchhd.permute(x, shifts=shifts)


def inverse_permute(x: torch.Tensor, shifts: int = 1) -> torch.Tensor:
    """Inverse permutation (undo permutation).

    Args:
        x: Permuted hypervector (shape: [dim] or [batch, dim])
        shifts: Number of positions to unshift (default: 1)

    Returns:
        Original hypervector (same shape as input)
    """
    # Inverse permutation is permutation by -shifts
    return torchhd.permute(x, shifts=-shifts)


def create_sequence_encoding(vectors: torch.Tensor) -> torch.Tensor:
    """Encode a sequence of hypervectors with position information.

    Uses a shared positional base hypervector that is permuted by position index,
    then bound with each content vector. This ensures position and content are
    properly separable and that permutation order is preserved.

    Args:
        vectors: Sequence of hypervectors (shape: [seq_len, dim])

    Returns:
        Single hypervector encoding the full sequence (shape: [dim])

    Example:
        >>> words = torch.stack([word_a, word_b, word_c])
        >>> sentence = create_sequence_encoding(words)
        >>> # sentence preserves order: "a b c" ≠ "c b a"
    """
    from vsa.operations.binding import bind
    from vsa.operations.bundling import bundle

    seq_len = vectors.shape[0]
    dim = vectors.shape[1]

    # Shared base hypervector for positional encoding
    # Using a deterministic seed based on dimension for reproducibility
    generator = torch.Generator(device=vectors.device)
    generator.manual_seed(42)  # Fixed seed for reproducible position base

    if vectors.dtype in [torch.cfloat, torch.complex64, torch.complex128]:
        position_base = torch.randn(dim, dtype=torch.cfloat, device=vectors.device, generator=generator)
        position_base = position_base / torch.abs(position_base).clamp(min=1e-8)
    else:
        position_base = torch.randn(dim, device=vectors.device, generator=generator)
        position_base = position_base / torch.norm(position_base).clamp(min=1e-8)

    # Bind each vector with its (shared) positional hypervector
    positioned = [bind(permute(position_base, shifts=i), vectors[i]) for i in range(seq_len)]

    # Bundle all positioned vectors
    return bundle(torch.stack(positioned))
