"""PoC vertical slice: VAE train, measured compression, two-region route."""

from cogsyndelta.poc.regions import ResidualMLPRegion
from cogsyndelta.poc.router import RouteResult, SoftmaxRouter
from cogsyndelta.poc.vae import LatentVAE

__all__ = ["LatentVAE", "ResidualMLPRegion", "RouteResult", "SoftmaxRouter"]
