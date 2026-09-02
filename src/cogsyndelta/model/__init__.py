"""Model backbones.

NOTE ON CURRICULUM ORDER: the causal LM here is step 4 (foundation), not step 1.
CSD-BRAIN-REGIONS.md is explicit that foundation pretraining comes AFTER per-region
pretrain, router training and assembly, and that skipping region pretrain is not a
shortcut -- an unpretrained region becomes dead weight because the gate collapses onto
whichever region moved first. This module exists so step 4 is ready; running it before
steps 1-3 would invert the curriculum.
"""

from cogsyndelta.model.causal_lm import CausalLM, CausalLMConfig
from cogsyndelta.model.vl_jepa import IJEPA, JEPAConfig, ViTEncoder

__all__ = ["IJEPA", "CausalLM", "CausalLMConfig", "JEPAConfig", "ViTEncoder"]
