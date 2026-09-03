"""I-JEPA correctness tests.

Every test here guards a failure mode that a falling loss curve would hide. JEPA's
characteristic failure is representation collapse: the encoder maps every input to the
same vector, the predictor trivially matches it, and the loss looks excellent. So none of
these assert "loss went down" on its own.
"""

from __future__ import annotations

import pytest
import torch

from cogsyndelta.model.vl_jepa import (
    IJEPA,
    JEPAConfig,
    ViTEncoder,
    sample_masks,
)


@pytest.fixture(scope="module")
def cfg() -> JEPAConfig:
    return JEPAConfig(
        image_size=32,
        patch_size=8,
        dim=96,
        depth=2,
        n_heads=4,
        predictor_dim=48,
        predictor_depth=2,
    )


@pytest.mark.cpu
def test_rejects_indivisible_shapes() -> None:
    with pytest.raises(ValueError, match="divisible by patch_size"):
        JEPAConfig(image_size=65, patch_size=8)
    with pytest.raises(ValueError, match="divisible by n_heads"):
        JEPAConfig(dim=100, n_heads=6)


@pytest.mark.cpu
def test_context_and_target_masks_are_disjoint(cfg: JEPAConfig) -> None:
    """Overlap would let the model copy a patch it can already see.

    That drives the loss toward zero while teaching nothing, and it is invisible in any
    metric except this one.
    """
    ctx, tgt = sample_masks(cfg, batch=16)
    for i in range(16):
        assert not (set(ctx[i].tolist()) & set(tgt[i].tolist()))


@pytest.mark.cpu
def test_target_encoder_never_receives_gradients(cfg: JEPAConfig) -> None:
    """Gradients reaching the target encoder make collapse the cheapest solution."""
    model = IJEPA(cfg)
    loss, _ = model(torch.randn(2, 3, cfg.image_size, cfg.image_size))
    loss.backward()

    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.predictor.parameters())


@pytest.mark.cpu
def test_ema_moves_target_toward_context_encoder(cfg: JEPAConfig) -> None:
    """The EMA must actually update, and must lag rather than copy.

    A momentum of exactly 1.0 (a plausible typo) freezes the target at initialisation and
    the model quietly trains against a random network.
    """
    model = IJEPA(cfg)
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    before = [p.clone() for p in model.target_encoder.parameters()]
    model.update_target(momentum=0.9)
    after = list(model.target_encoder.parameters())

    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))
    # Lagging, not copying: still distinguishable from the context encoder.
    assert any(
        not torch.allclose(t, s) for t, s in zip(after, model.encoder.parameters(), strict=True)
    )


@pytest.mark.cpu
def test_predictor_is_narrower_than_encoder(cfg: JEPAConfig) -> None:
    """The asymmetry is one of the three collapse defences, not an efficiency choice."""
    model = IJEPA(cfg)
    enc = sum(p.numel() for p in model.encoder.parameters())
    pred = sum(p.numel() for p in model.predictor.parameters())
    assert pred < enc


@pytest.mark.cpu
def test_masked_encoding_drops_patches_rather_than_padding(cfg: JEPAConfig) -> None:
    """A mask token in the encoder is something the model can learn to exploit; dropping
    the patches entirely means it cannot."""
    encoder = ViTEncoder(cfg)
    x = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    keep = torch.stack([torch.arange(5), torch.arange(5)])
    assert encoder(x, keep=keep).shape == (2, 5, cfg.dim)
    assert encoder(x).shape == (2, cfg.n_patches, cfg.dim)


@pytest.mark.cpu
def test_encode_returns_shared_stream_surface(cfg: JEPAConfig) -> None:
    """Visual latents must land on the same [B, D] surface a CognitiveRegion consumes,
    so a vision region is a peer of a text region rather than a special case."""
    model = IJEPA(cfg)
    latent = model.encode(torch.randn(3, 3, cfg.image_size, cfg.image_size))
    assert latent.shape == (3, cfg.dim)
    assert latent.dtype == torch.float32


@pytest.mark.cpu
def test_does_not_collapse_while_overfitting(cfg: JEPAConfig) -> None:
    """The load-bearing test.

    Overfit one batch and require BOTH that the loss falls AND that representation
    variance survives. A collapsed model passes the first check easily; only the second
    distinguishes learning from degenerate agreement.
    """
    torch.manual_seed(0)
    model = IJEPA(cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    x = torch.randn(8, 3, cfg.image_size, cfg.image_size)

    first = None
    for _ in range(60):
        loss, stats = model(x)
        if first is None:
            first = stats["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()

    assert first is not None
    assert stats["loss"] < first * 0.5, f"failed to learn: {first:.4f} -> {stats['loss']:.4f}"
    assert stats["rep_std"] > 0.01, f"representations collapsed: rep_std={stats['rep_std']:.5f}"
