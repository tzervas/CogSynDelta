"""Region submodels: the specialists that compose into one mind."""

from cogsyndelta.regions.pretrain import PretrainConfig, evaluate, load_pairs, pretrain_region
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig, info_nce

__all__ = [
    "PretrainConfig",
    "TextEncoder",
    "TextEncoderConfig",
    "evaluate",
    "info_nce",
    "load_pairs",
    "pretrain_region",
]
