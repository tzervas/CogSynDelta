"""Binding operations for VSA.

Binding creates compositional structures by associating hypervectors.
For FHRR, binding is implemented as element-wise multiplication (circular convolution in Fourier domain).

Properties:
- Associative: (A ⊗ B) ⊗ C = A ⊗ (B ⊗ C)
- Commutative: A ⊗ B = B ⊗ A
- Invertible: unbind(bind(A, B), B) ≈ A
- Dissimilar outputs: sim(A, bind(A, B)) ≈ 0 for random B
"""

import torch
import torchhd


def bind(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Bind two hypervectors.

    For FHRR (complex): binding via element-wise multiplication
    For MAP (real): binding via circular convolution
    For BSC (binary): binding via XOR

    Args:
        x: First hypervector (shape: [dim] or [batch, dim])
        y: Second hypervector (shape: [dim] or [batch, dim])

    Returns:
        Bound hypervector (same shape as inputs)

    Example:
        >>> import torch
        >>> from vsa.operations.binding import bind, unbind
        >>> x = torch.randn(10000, dtype=torch.cfloat)
        >>> y = torch.randn(10000, dtype=torch.cfloat)
        >>> z = bind(x, y)  # Compositional structure
        >>> x_recovered = unbind(z, y)  # Extract x from z
        >>> similarity = (x * x_recovered.conj()).real.mean()  # ≈ 0.9+
    """
    return torchhd.bind(x, y)


def unbind(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Unbind (extract) a hypervector from a bound composition.

    For FHRR: unbind(z, y) = z * conj(y)
    For MAP: unbind via circular correlation
    For BSC: unbind via XOR (same as bind)

    Args:
        z: Bound hypervector (shape: [dim] or [batch, dim])
        y: Known binding partner (shape: [dim] or [batch, dim])

    Returns:
        Extracted hypervector (same shape as inputs)

    Example:
        >>> x_recovered = unbind(bind(x, y), y)
        >>> # x_recovered ≈ x with high similarity
    """
    # For complex (FHRR): unbind is multiplication by conjugate
    if z.dtype == torch.cfloat or z.dtype == torch.complex64 or z.dtype == torch.complex128:
        return z * y.conj()

    # For real/binary: use torchhd's unbind (handles circular correlation)
    return torchhd.bind(z, y)  # XOR is self-inverse for BSC
