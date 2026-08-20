"""PoC vertical slice: VAE train, measured compression, two-region route."""

from cogsyndelta.poc.regions import ResidualMLPRegion
from cogsyndelta.poc.router import RouteResult, SoftmaxRouter, switch_aux_load_balance
from cogsyndelta.poc.vae import LatentVAE

__all__ = [
    "LatentVAE",
    "ResidualMLPRegion",
    "RouteResult",
    "SoftmaxRouter",
    "switch_aux_load_balance",
]
