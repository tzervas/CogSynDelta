"""Resonator Networks for compositional factorization.

Resonator networks solve the binding factorization problem:
Given z = x ⊗ y and x, find y (or vice versa).

This enables compositional reasoning by extracting components from bound structures.

Capacity: Scales quadratically with dimension N (Kent et al., 2020).

References:
    - Kent et al. (2020). Resonator networks for factorizing distributed representations.
      Neural Computation, 32(12), 2379-2427.
"""

import torch
from torch import nn

from vsa.operations.binding import unbind


class ResonatorNetwork(nn.Module):
    """Resonator network for factorizing compositional hypervectors.

    Iteratively refines estimates of unknown factors in binding operations.

    Attributes:
        dimension: Dimension of hypervectors
        max_iterations: Maximum resonance iterations
        convergence_threshold: Stop when change < threshold
        candidates: Set of candidate vectors for discrete factorization
    """

    def __init__(
        self,
        dimension: int,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-4,
        device: str | None = None,
    ) -> None:
        """Initialize resonator network.

        Args:
            dimension: Dimension of hypervectors
            max_iterations: Maximum iterations for resonance
            convergence_threshold: Convergence threshold
            device: Compute device ("cuda" or "cpu", default: auto-detected)
        """
        super().__init__()
        self.dimension = dimension
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.device = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Optional: set of candidate vectors for discrete search
        self.register_buffer("candidates", None)

    def set_candidates(self, candidates: torch.Tensor) -> None:
        """Set candidate vectors for discrete factorization.

        Args:
            candidates: Candidate hypervectors (shape: [num_candidates, dimension])
        """
        self.candidates = candidates.to(self.device)

    def factorize(
        self, bound: torch.Tensor, known: torch.Tensor, initial_guess: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, int]:
        """Factorize a bound hypervector given one component.

        Given: z = x ⊗ y
        Known: x
        Find: y

        Args:
            bound: Bound hypervector z (shape: [dimension])
            known: Known component x (shape: [dimension])
            initial_guess: Initial estimate of y (default: random)

        Returns:
            Tuple of (estimated_y, num_iterations)

        Example:
            >>> from vsa.operations.binding import bind
            >>> x = torch.randn(10000, dtype=torch.cfloat)
            >>> y = torch.randn(10000, dtype=torch.cfloat)
            >>> z = bind(x, y)
            >>> resonator = ResonatorNetwork(10000)
            >>> y_est, iters = resonator.factorize(z, x)
            >>> similarity = (y * y_est.conj()).real.mean()  # > 0.95
        """
        bound = bound.to(self.device)
        known = known.to(self.device)

        # Initialize estimate
        if initial_guess is None:
            if self.candidates is not None:
                # Use best candidate as initial guess
                estimate = self._best_candidate(bound, known)
            else:
                # Random initialization
                if bound.dtype in [torch.cfloat, torch.complex64, torch.complex128]:
                    estimate = torch.randn(self.dimension, dtype=torch.cfloat, device=self.device)
                    estimate = estimate / torch.abs(estimate).clamp(min=1e-8)
                else:
                    estimate = torch.randn(self.dimension, device=self.device)
                    estimate = estimate / torch.norm(estimate).clamp(min=1e-8)
        else:
            estimate = initial_guess.to(self.device)

        # Resonance iterations
        # For iterative refinement, we alternate between:
        # 1. Unbinding: refine estimate using bound and known
        # 2. Projection: snap to nearest candidate (if discrete search)
        # 3. Normalization: maintain unit magnitude
        for iteration in range(self.max_iterations):
            prev_estimate = estimate.clone()

            # Core resonance update: unbind to get new estimate
            # For continuous case: estimate = unbind(bound, known)
            # For discrete/candidate case: this gets refined by projection
            new_estimate = unbind(bound, known)

            # If candidates provided, project onto nearest candidate
            # This is where iterative refinement happens - projection improves estimate
            if self.candidates is not None:
                # Combine direct unbind with current estimate for better convergence
                # (weighted average helps with noisy unbinding)
                combined = 0.5 * new_estimate + 0.5 * estimate
                estimate = self._project_to_candidates(combined)
            else:
                estimate = new_estimate

            # Normalize
            if estimate.dtype in [torch.cfloat, torch.complex64, torch.complex128]:
                estimate = estimate / torch.abs(estimate).clamp(min=1e-8)
            else:
                estimate = estimate / torch.norm(estimate).clamp(min=1e-8)

            # Check convergence
            change = torch.norm(estimate - prev_estimate).item()
            if change < self.convergence_threshold:
                return estimate, iteration + 1

        return estimate, self.max_iterations

    def _best_candidate(self, bound: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        """Find best candidate via similarity search.

        Args:
            bound: Bound hypervector z
            known: Known component x

        Returns:
            Best candidate from self.candidates
        """
        # Unbind with each candidate
        estimate = unbind(bound, known)

        # Find most similar candidate
        if estimate.dtype in [torch.cfloat, torch.complex64, torch.complex128]:
            similarities = (self.candidates @ estimate.conj()).real
        else:
            similarities = self.candidates @ estimate

        best_idx = similarities.argmax()
        return self.candidates[best_idx]

    def _project_to_candidates(self, estimate: torch.Tensor) -> torch.Tensor:
        """Project estimate onto nearest candidate.

        Args:
            estimate: Current estimate

        Returns:
            Nearest candidate vector
        """
        if estimate.dtype in [torch.cfloat, torch.complex64, torch.complex128]:
            similarities = (self.candidates @ estimate.conj()).real
        else:
            similarities = self.candidates @ estimate

        best_idx = similarities.argmax()
        return self.candidates[best_idx]

    def batch_factorize(
        self,
        bounds: torch.Tensor,
        knowns: torch.Tensor,
        initial_guesses: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Factorize multiple bound hypervectors in parallel.

        Args:
            bounds: Bound hypervectors (shape: [batch, dimension])
            knowns: Known components (shape: [batch, dimension])
            initial_guesses: Initial estimates (shape: [batch, dimension])

        Returns:
            Tuple of (estimates, iterations) where estimates is [batch, dimension]
            and iterations is [batch]
        """
        batch_size = bounds.shape[0]
        estimates = torch.zeros_like(bounds)
        iterations = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i in range(batch_size):
            initial = None if initial_guesses is None else initial_guesses[i]
            estimate, iters = self.factorize(bounds[i], knowns[i], initial)
            estimates[i] = estimate
            iterations[i] = iters

        return estimates, iterations
