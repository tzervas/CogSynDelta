"""Region submodels: the specialists that compose into one mind."""

from cogsyndelta.regions.pretrain import (
    PretrainConfig,
    evaluate,
    evaluate_graded,
    load_graded_pairs,
    load_pairs,
    pretrain_region,
)
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig, info_nce

__all__ = [
    "PretrainConfig",
    "TextEncoder",
    "TextEncoderConfig",
    "evaluate",
    "evaluate_graded",
    "info_nce",
    "load_graded_pairs",
    "load_pairs",
    "pretrain_region",
]
