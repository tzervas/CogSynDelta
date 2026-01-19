"""GPU, hardware, and algebraic optimization modules.

This module provides:
1. GPU/CUDA optimization (RTX5080Optimizer, etc.)
2. Algebraic training (predict training outcomes without backprop)
   - Core library: libs/algebraic-training/
   - CogSynDelta integration: _integration.py
"""
# ruff: noqa: E402

import sys
from pathlib import Path

# Add algebraic-training lib to path for development
_libs_path = Path(__file__).parent.parent.parent.parent / "libs" / "algebraic-training" / "src"
if _libs_path.exists() and str(_libs_path) not in sys.path:
    sys.path.insert(0, str(_libs_path))

# GPU optimization
# Core algebraic training (from libs/)
from algebraic_training import (
    AlgebraicOptimizer,
    FisherInformationPredictor,
    NTKPredictor,
    SpectralWeightPredictor,
    UnifiedAlgebraicTrainer,
    WeightDistributionPredictor,
)
from algebraic_training import (
    functional as algebraic_functional,
)

# CogSynDelta-specific integration
from cogsyndelta.optimization._integration import (
    InterconnectAlgebraicOptimizer,
    MHCAlgebraicOptimizer,
    PathwayStrengthPredictor,
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
    # Core algebraic training
    "AlgebraicOptimizer",
    "FisherInformationPredictor",
    "NTKPredictor",
    "SpectralWeightPredictor",
    "UnifiedAlgebraicTrainer",
    "WeightDistributionPredictor",
    "algebraic_functional",
    # CogSynDelta integration
    "InterconnectAlgebraicOptimizer",
    "MHCAlgebraicOptimizer",
    "PathwayStrengthPredictor",
]
