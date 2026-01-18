"""GPU and hardware optimization modules."""

from cogsyndelta.optimization.cuda_optimization import (
    CUDA_AVAILABLE,
    COMPUTE_CAPABILITY,
    DEVICE_NAME,
    GPUConfig,
    RTX5080Optimizer,
    TOTAL_MEMORY,
)

__all__ = [
    "CUDA_AVAILABLE",
    "DEVICE_NAME",
    "TOTAL_MEMORY",
    "COMPUTE_CAPABILITY",
    "GPUConfig",
    "RTX5080Optimizer",
]
