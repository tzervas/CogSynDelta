"""Modern Hopfield Network for VSA associative memory.

Implements continuous modern Hopfield networks with exponential storage capacity.

Mathematical Foundation:
    Energy function: E = -log(Σᵢ exp(βξᵢᵀx))
    Update rule: ξ_new = X·softmax(βXᵀξ)

Attention Equivalence:
    Modern Hopfield update ≡ Transformer attention:
    - Query Q ↔ State ξ
    - Keys K ↔ Stored patterns X
    - Values V ↔ Stored patterns X
    - Temperature 1/β ↔ √d_k

References:
    - Ramsauer et al. (2021). Hopfield Networks is All You Need. ICLR.
    - Demircigil et al. (2017). On a model of associative memory with huge storage capacity.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ModernHopfieldMemory(nn.Module):
    """Modern Hopfield Network with exponential capacity.

    Stores and retrieves hypervectors with one-step convergence and
    exponential storage capacity: ~2^(d/2) patterns for d-dimensional vectors.

    Attributes:
        dimension: Dimension of hypervectors
        beta: Inverse temperature (default: 1.0, higher = sharper retrieval)
        max_patterns: Maximum number of patterns to store
        patterns: Stored memory patterns (shape: [num_patterns, dimension])
    """

    def __init__(
        self,
        dimension: int,
        beta: float = 1.0,
        max_patterns: int | None = None,
        device: str = "cuda",
    ) -> None:
        """Initialize Modern Hopfield memory.

        Args:
            dimension: Dimension of hypervectors
            beta: Inverse temperature for sharpness (default: 1.0)
            max_patterns: Maximum patterns to store (default: None = unlimited)
            device: Compute device ("cuda" or "cpu")
        """
        super().__init__()
        self.dimension = dimension
        self.beta = beta
        self.max_patterns = max_patterns
        self.device = device

        # Initialize empty pattern storage
        self.register_buffer("patterns", torch.empty(0, dimension, device=device))
        self.register_buffer("pattern_count", torch.tensor(0, device=device))

    def store(self, pattern: torch.Tensor) -> None:
        """Store a pattern in memory.

        Args:
            pattern: Hypervector to store (shape: [dimension])
        """
        # Ensure pattern is on correct device
        pattern = pattern.to(self.device)

        # Add pattern
        self.patterns = torch.cat([self.patterns, pattern.unsqueeze(0)], dim=0)
        self.pattern_count += 1

        # Evict oldest if exceeding capacity
        if self.max_patterns is not None and self.pattern_count > self.max_patterns:
            self.patterns = self.patterns[-self.max_patterns :]
            self.pattern_count = torch.tensor(self.max_patterns, device=self.device)

    def store_batch(self, patterns: torch.Tensor) -> None:
        """Store multiple patterns.

        Args:
            patterns: Hypervectors to store (shape: [batch, dimension])
        """
        patterns = patterns.to(self.device)
        self.patterns = torch.cat([self.patterns, patterns], dim=0)
        self.pattern_count += patterns.shape[0]

        if self.max_patterns is not None and self.pattern_count > self.max_patterns:
            self.patterns = self.patterns[-self.max_patterns :]
            self.pattern_count = torch.tensor(self.max_patterns, device=self.device)

    def retrieve(self, query: torch.Tensor, num_iterations: int = 1) -> torch.Tensor:
        """Retrieve (cleanup) a noisy pattern.

        Modern Hopfield networks converge in one step for most cases.

        Args:
            query: Noisy/partial hypervector (shape: [dimension])
            num_iterations: Number of update iterations (default: 1)

        Returns:
            Retrieved clean pattern (shape: [dimension])

        Example:
            >>> memory = ModernHopfieldMemory(10000)
            >>> memory.store(pattern_a)
            >>> memory.store(pattern_b)
            >>> noisy = pattern_a + 0.2 * torch.randn_like(pattern_a)
            >>> retrieved = memory.retrieve(noisy)
            >>> # retrieved ≈ pattern_a with high similarity
        """
        if self.patterns.shape[0] == 0:
            return query

        query = query.to(self.device)
        state = query

        for _ in range(num_iterations):
            # Modern Hopfield update: ξ_new = X·softmax(βXᵀξ)
            # This is equivalent to attention: Attention(Q,K,V) = softmax(QKᵀ/τ)V

            # Handle complex patterns
            if self.patterns.dtype in [torch.cfloat, torch.complex64, torch.complex128]:
                # For complex: use real part of inner product
                similarities = (self.patterns @ state.conj()).real * self.beta
            else:
                # For real: standard dot product
                similarities = (self.patterns @ state) * self.beta

            # Softmax attention weights
            weights = F.softmax(similarities, dim=0)

            # Weighted sum of patterns
            state = (weights.unsqueeze(-1) * self.patterns).sum(dim=0)

        return state

    def retrieve_batch(self, queries: torch.Tensor, num_iterations: int = 1) -> torch.Tensor:
        """Retrieve multiple patterns in parallel.

        Args:
            queries: Noisy/partial hypervectors (shape: [batch, dimension])
            num_iterations: Number of update iterations (default: 1)

        Returns:
            Retrieved clean patterns (shape: [batch, dimension])
        """
        if self.patterns.shape[0] == 0:
            return queries

        queries = queries.to(self.device)
        states = queries

        for _ in range(num_iterations):
            # Batch update: [batch, dimension] @ [dimension, num_patterns]ᵀ
            if self.patterns.dtype in [torch.cfloat, torch.complex64, torch.complex128]:
                similarities = (states @ self.patterns.T.conj()).real * self.beta
            else:
                similarities = (states @ self.patterns.T) * self.beta

            # Softmax: [batch, num_patterns]
            weights = F.softmax(similarities, dim=1)

            # Weighted sum: [batch, num_patterns] @ [num_patterns, dimension]
            states = weights @ self.patterns

        return states

    def capacity(self) -> int:
        """Theoretical exponential capacity: 2^(d/2).

        Returns:
            Approximate capacity (number of patterns)
        """
        return int(2 ** (self.dimension / 2))

    def clear(self) -> None:
        """Clear all stored patterns."""
        self.patterns = torch.empty(0, self.dimension, device=self.device)
        self.pattern_count = torch.tensor(0, device=self.device)

    def __len__(self) -> int:
        """Return number of stored patterns."""
        return int(self.pattern_count.item())
