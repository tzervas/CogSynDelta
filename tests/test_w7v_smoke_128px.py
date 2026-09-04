"""W7v-cfg (6): smoke test — a random 128-px batch through the rebuilt encoder, on CPU.

Not a training run (CPU only, random tensors, no data, no gradient step): this proves
the shapes and the deployed parameter count the rest of W7v-cfg's deliverables assume.
`docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` row W7v; `g8-visual/S02.md` §7
("Config proposal"), §4 (encoder param recount).
"""

from __future__ import annotations

import pytest
import torch

from cogsyndelta.model.vl_jepa import IJEPA, JEPAConfig
from cogsyndelta.vl.composite import FRAME_SIZE, N_PATCHES

pytestmark = pytest.mark.cpu

# g8 sector `S02.md` §4 / `S02-verify.md` survive #34: independently recounted this
# session too (Conv2d 3->384 k=8 with bias: 74,112; 6 x ViTBlock: 10,637,568; final
# LayerNorm(384): 768; total 10,712,448). This is the DEPLOYED half (DEC-34): the EMA
# `target_encoder`, not the full IJEPA pickle (context + target + predictor, which the
# same session measured at 22,905,216 -- matching the A4 receipt's `parameters` field).
_G8_TARGET_ENCODER_PARAMS = 10_712_448


def test_smoke_128px_batch_through_the_default_encoder_on_cpu() -> None:
    """A random 128-px batch produces [B, 256, D] patch latents and pooled [B, D]."""
    device = torch.device("cpu")
    cfg = JEPAConfig()  # default: 128px / patch 8 / grid 16 / 256 patches
    assert cfg.image_size == FRAME_SIZE
    assert cfg.n_patches == N_PATCHES == 256

    model = IJEPA(cfg).to(device)
    model.eval()

    batch = 4
    images = torch.randn(batch, cfg.in_channels, cfg.image_size, cfg.image_size, device=device)

    with torch.no_grad():
        tokens, mask = model.tokens(images)
        pooled = model.pool(tokens, mask)
        encoded = model.encode(images)

    print(f"tokens shape: {tuple(tokens.shape)}")
    print(f"pooled shape: {tuple(pooled.shape)}")

    assert tokens.shape == (batch, 256, cfg.dim)
    assert mask.shape == (batch, 256)
    assert torch.all(mask == 1.0), "images carry no padding; tokens() mask must be all-ones"
    assert pooled.shape == (batch, cfg.dim)
    # DEC-34 / W0's own equivalence (test_token_surface.py): encode() IS pool(tokens()).
    assert torch.equal(encoded, pooled)


def test_deployed_target_encoder_parameter_count() -> None:
    """The DEPLOYED half's parameter count (DEC-34: the EMA target_encoder, not the
    full context+target+predictor pickle) is asserted against the g8 sector's
    independently-recounted figure -- not forced if the code disagrees."""
    cfg = JEPAConfig()
    model = IJEPA(cfg)

    target_params = sum(p.numel() for p in model.target_encoder.parameters())
    full_params = sum(p.numel() for p in model.parameters())

    print(f"target_encoder (deployed) parameters: {target_params:,}")
    print(f"full IJEPA (context + target + predictor) parameters: {full_params:,}")

    # If this ever fails, the correct response is NOT to edit the constant above to
    # match a changed code count -- it is to determine whether the g8 figure or this
    # measurement is right (a real architecture change vs. a regression) and say so in
    # the commit, per this row's own instruction: "if the real count differs, report
    # the measured number and do NOT force it".
    assert target_params == _G8_TARGET_ENCODER_PARAMS, (
        f"measured {target_params:,} target_encoder params, g8 sector recount says "
        f"{_G8_TARGET_ENCODER_PARAMS:,} -- do not force this constant to match; work "
        f"out which number is right and why."
    )

    # context encoder + target encoder (same architecture, deepcopy) + predictor.
    context_params = sum(p.numel() for p in model.encoder.parameters())
    predictor_params = sum(p.numel() for p in model.predictor.parameters())
    assert context_params == target_params
    assert full_params == context_params + target_params + predictor_params
