"""GPU, hardware, and algebraic optimization modules.

This module provides:
1. GPU/CUDA optimization (RTX5080Optimizer, etc.)
2. Algebraic training (predict training outcomes without backprop)
"""

from cogsyndelta.optimization.algebraic_training import (
    AlgebraicOptimizer,
    FisherInformationPredictor,
    InterconnectAlgebraicOptimizer,
    MHCAlgebraicOptimizer,
    NTKPredictor,
    PathwayStrengthPredictor,
    SpectralWeightPredictor,
    UnifiedAlgebraicTrainer,
    WeightDistributionPredictor,
)
from cogsyndelta.optimization.cuda_optimization import (
    COMPUTE_CAPABILITY,
    CUDA_AVAILABLE,
    DEVICE_NAME,
    TOTAL_MEMORY,
    GPUConfig,
    RTX5080Optimizer,
)

__all__ = [  # noqa: RUF022 - Grouped by category for readability
    # GPU optimization
    "COMPUTE_CAPABILITY",
    "CUDA_AVAILABLE",
    "DEVICE_NAME",
    "TOTAL_MEMORY",
    "GPUConfig",
    "RTX5080Optimizer",
    # Algebraic training
    "AlgebraicOptimizer",
    "FisherInformationPredictor",
    "InterconnectAlgebraicOptimizer",
    "MHCAlgebraicOptimizer",
    "NTKPredictor",
    "PathwayStrengthPredictor",
    "SpectralWeightPredictor",
    "UnifiedAlgebraicTrainer",
    "WeightDistributionPredictor",
]
