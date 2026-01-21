"""Bundling operations for VSA.

Bundling creates superposition of hypervectors, representing sets or probabilistic mixtures.
Implemented as normalized element-wise addition.

Properties:
- Commutative: bundle(A, B) = bundle(B, A)
- Associative: bundle(bundle(A, B), C) = bundle(A, bundle(B, C))
- Similar to inputs: sim(A, bundle(A, B, C, ...)) > 0
- Cleanup: Can extract most similar element from bundle via resonator networks
"""

import torch
import torchhd


def bundle(vectors: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Bundle multiple hypervectors into a superposition.

    Args:
        vectors: Hypervectors to bundle (shape: [n, dim])
        normalize: Whether to normalize the result (default: True)

    Returns:
        Bundled hypervector (shape: [dim])

    Example:
        >>> import torch
        >>> from vsa.operations.bundling import bundle
        >>> x = torch.randn(10000, dtype=torch.cfloat)
        >>> y = torch.randn(10000, dtype=torch.cfloat)
        >>> z = torch.randn(10000, dtype=torch.cfloat)
        >>> superposition = bundle(torch.stack([x, y, z]))
        >>> # superposition is similar to all inputs
        >>> sim_x = (superposition * x.conj()).real.mean()  # > 0
        >>> sim_y = (superposition * y.conj()).real.mean()  # > 0
        >>> sim_z = (superposition * z.conj()).real.mean()  # > 0
    """
    return torchhd.bundle(vectors)


def weighted_bundle(
    vectors: torch.Tensor, weights: torch.Tensor, normalize: bool = True
) -> torch.Tensor:
    """Bundle hypervectors with weights (for importance/confidence).

    Args:
        vectors: Hypervectors to bundle (shape: [n, dim])
        weights: Importance weights (shape: [n])
        normalize: Whether to normalize the result (default: True)

    Returns:
        Weighted bundled hypervector (shape: [dim])

    Example:
        >>> vectors = torch.stack([x, y, z])
        >>> weights = torch.tensor([0.5, 0.3, 0.2])  # x is most important
        >>> superposition = weighted_bundle(vectors, weights)
    """
    # Expand weights for broadcasting: [n, 1]
    weights_expanded = weights.unsqueeze(-1)

    # Weighted sum
    weighted_sum = (vectors * weights_expanded).sum(dim=0)

    if normalize:
        # Normalize: complex vectors get element-wise unit magnitude; real vectors get L2-normalized
        if weighted_sum.dtype in [torch.cfloat, torch.complex64, torch.complex128]:
            # Complex: element-wise magnitude normalization (unit phasors, phase preserved)
            # This maintains the phase information while normalizing magnitudes to 1
            norm = torch.abs(weighted_sum).clamp(min=1e-8)
            return weighted_sum / norm
        else:
            # Real: L2 normalization of the whole vector
            norm = torch.norm(weighted_sum, p=2).clamp(min=1e-8)
            return weighted_sum / norm

    return weighted_sum
