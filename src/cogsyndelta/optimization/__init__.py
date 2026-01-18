"""GPU and hardware optimization modules."""

from cogsyndelta.optimization.cuda_optimization import (
    COMPUTE_CAPABILITY,
    CUDA_AVAILABLE,
    DEVICE_NAME,
    TOTAL_MEMORY,
    GPUConfig,
    RTX5080Optimizer,
)

__all__ = [
    "COMPUTE_CAPABILITY",
    "CUDA_AVAILABLE",
    "DEVICE_NAME",
    "TOTAL_MEMORY",
    "GPUConfig",
    "RTX5080Optimizer",
]
