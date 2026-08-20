"""Frozen PoC contracts — feature lanes import these, do not redefine them."""

from cogsyndelta.contracts.compactor import (
    BasisResidualCompactor,
    CalibratedQuantCompactor,
    CompactBlob,
    Compactor,
    measured_fidelity,
)
from cogsyndelta.contracts.config import (
    CompressionConfig,
    DevicePrefer,
    PocConfig,
    RouteConfig,
    RouteTrainConfig,
    TrainConfig,
)
from cogsyndelta.contracts.device import DeviceContext
from cogsyndelta.contracts.metrics import MetricsRecord, MetricsStatus, write_metrics_json
from cogsyndelta.contracts.region import CognitiveRegion
from cogsyndelta.contracts.registry import RegionRegistry

__all__ = [
    "BasisResidualCompactor",
    "CalibratedQuantCompactor",
    "CognitiveRegion",
    "CompactBlob",
    "Compactor",
    "CompressionConfig",
    "DeviceContext",
    "DevicePrefer",
    "MetricsRecord",
    "MetricsStatus",
    "PocConfig",
    "RegionRegistry",
    "RouteConfig",
    "RouteTrainConfig",
    "TrainConfig",
    "measured_fidelity",
    "write_metrics_json",
]
