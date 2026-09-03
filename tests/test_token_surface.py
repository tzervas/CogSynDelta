"""W0 -- the token surface: `tokens()` / `pool()` on `TextEncoder` and `visual`.

`docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` §4.1 (row W0) is explicit that the
``‖pool(tokens(x)) - encode(x)‖∞ < 1e-5`` check cannot audit the pooling itself:
``pool()`` contains the *same* masked-mean expression ``forward`` used to contain
(``regions/text_encoder.py:117-124``), so the smoke test compares an expression to
itself through the same weights, and a pre-existing pooling bug would be present
identically on both sides. It is kept below anyway, as a refactor smoke test on every
checkpoint on disk -- but the real gate is the four properties that can actually FAIL,
each demonstrated here against a deliberately broken pooling in a scratch subclass:

  (i)   pad invariance -- perturbing embeddings at masked positions must not change
        ``pool()``.
  (ii)  independent reference -- ``pool()`` equals a separately-written numpy masked
        mean.
  (iii) row-permutation, BOTH clauses -- permuting the unmasked positions of ``h`` and
        ``mask`` together leaves ``pool()`` unchanged, AND must change ``tokens()``
        (the second clause is what confirms ``tokens()`` is not already the constant
        the central bet in §4.0 feared).
  (iv)  batch-composition invariance -- the same item padded to two different lengths
        pools identically.

Every property test below has a sibling asserting the SAME check catches the specific
defect it exists for, so a property that could never fail is not mistaken for one that
passed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cogsyndelta.model.vl_jepa import IJEPA, JEPAConfig
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig

pytestmark = pytest.mark.cpu

_TOL = 1e-5
_HELD_OUT_N = 512

_RECEIPTS = Path("/akula-data/csd/receipts")
_TEXT_CHECKPOINTS = [
    _RECEIPTS / "code-checkpoints" / "final.pt",
    _RECEIPTS / "compress-checkpoints" / "final.pt",
    _RECEIPTS / "retrieve-checkpoints" / "final.pt",
]
_VL_CHECKPOINT = _RECEIPTS / "vl_latent-checkpoints" / "step-8000.pt"


def _load_text_encoder(path: Path) -> tuple[TextEncoder, TextEncoderConfig]:
    """Load a real checkpoint's weights, unchanged, per W0 ("retrains nothing")."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = TextEncoderConfig(**ckpt["config"])
    model = TextEncoder(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def _load_ijepa(path: Path) -> tuple[IJEPA, JEPAConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = JEPAConfig(**ckpt["config"])
    model = IJEPA(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------------
# Refactor smoke test -- kept per the row's instruction, demoted from "the gate" to
# "a smoke test on the split": every checkpoint on disk, 512 held-out items.
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("path", _TEXT_CHECKPOINTS, ids=lambda p: p.parent.name)
def test_text_pool_tokens_equals_encode_on_checkpoint(path: Path) -> None:
    """``pool(tokens(x)) == encode(x)`` to 1e-5 on 512 items, for a real checkpoint."""
    if not path.is_file():
        pytest.skip(f"{path} not present on this host")
    model, cfg = _load_text_encoder(path)

    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1, cfg.vocab_size, (_HELD_OUT_N, cfg.max_len), generator=g)
    # Varied per-item padding so the smoke test also exercises the mask path used in
    # production, not only the all-real-tokens branch.
    lengths = torch.randint(max(cfg.max_len // 4, 1), cfg.max_len + 1, (_HELD_OUT_N,), generator=g)
    mask = (torch.arange(cfg.max_len).unsqueeze(0) < lengths.unsqueeze(1)).to(torch.float32)

    with torch.no_grad():
        encoded = model(ids, mask)
        h, m = model.tokens(ids, mask)
        pooled = model.pool(h, m)

    diff = (encoded - pooled).abs().max().item()
    assert diff < _TOL, f"{path}: pool(tokens(x)) diverged from encode(x) by {diff}"


def test_visual_pool_tokens_equals_encode_on_checkpoint() -> None:
    """Same equivalence, for `visual`'s deployed half -- the EMA `target_encoder`
    (DEC-34). `tokens()` must expose ITS patch tokens, at ``[B, 64, 384]``."""
    if not _VL_CHECKPOINT.is_file():
        pytest.skip(f"{_VL_CHECKPOINT} not present on this host")
    model, cfg = _load_ijepa(_VL_CHECKPOINT)

    g = torch.Generator().manual_seed(0)
    images = torch.randn(_HELD_OUT_N, cfg.in_channels, cfg.image_size, cfg.image_size, generator=g)

    with torch.no_grad():
        encoded = model.encode(images)
        h, m = model.tokens(images)
        pooled = model.pool(h, m)

    assert h.shape == (_HELD_OUT_N, 64, 384), h.shape
    diff = (encoded - pooled).abs().max().item()
    assert diff < _TOL, f"{_VL_CHECKPOINT}: pool(tokens(x)) diverged from encode(x) by {diff}"


# ---------------------------------------------------------------------------------
# The four failable properties, on a small untrained scratch encoder (no checkpoint
# needed -- these are about the pooling CODE, not any particular trained weights).
# ---------------------------------------------------------------------------------

_SCRATCH_CFG = TextEncoderConfig(
    vocab_size=200, dim=32, depth=2, n_heads=4, max_len=32, out_dim=None
)


def _scratch_model(cls: type[TextEncoder] = TextEncoder) -> TextEncoder:
    torch.manual_seed(0)
    return cls(_SCRATCH_CFG).eval()


def _scratch_batch(
    batch: int = 4, real: int = 6, total: int = 12, seed: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, _SCRATCH_CFG.vocab_size, (batch, total), generator=g)
    mask = torch.zeros(batch, total)
    mask[:, :real] = 1.0
    return ids, mask


# --- (i) Pad invariance ------------------------------------------------------------


class _BrokenPadPool(TextEncoder):
    """The exact historical bug this module's docstring (and the code's) describes:
    averages over ALL positions, padding included."""

    def pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.proj(h.mean(dim=1))


def _pad_perturbation_delta(model: TextEncoder) -> float:
    ids, mask = _scratch_batch()
    with torch.no_grad():
        h, m = model.tokens(ids, mask)
        before = model.pool(h, m)
        h_perturbed = h.clone()
        h_perturbed[:, 6:, :] += torch.randn_like(h_perturbed[:, 6:, :]) * 10.0
        after = model.pool(h_perturbed, m)
    return (before - after).abs().max().item()


def test_pad_invariance_holds() -> None:
    assert _pad_perturbation_delta(_scratch_model()) < _TOL


def test_pad_invariance_is_caught_when_pooling_ignores_the_mask() -> None:
    delta = _pad_perturbation_delta(_scratch_model(_BrokenPadPool))
    assert delta > 1e-2, "an unmasked mean ignores padding perturbation; should be caught"


# --- (ii) Independent numpy reference -----------------------------------------------


def _numpy_masked_mean(h: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Written independently of `TextEncoder.pool`: same idea, different code, no
    shared helper -- so agreement is evidence, not a comparison of an expression with
    itself."""
    m = mask[..., None].astype(h.dtype)
    return (h * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-6, None)


class _BrokenSumPool(TextEncoder):
    """Sums instead of averaging. The numpy reference is a MEAN, so this diverges."""

    def pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.proj((h * m).sum(dim=1))


def _numpy_reference_delta(model: TextEncoder) -> float:
    ids, mask = _scratch_batch()
    with torch.no_grad():
        h, m = model.tokens(ids, mask)
        pooled = model.pool(h, m)
    ref = _numpy_masked_mean(h.numpy(), m.numpy())
    return float(np.abs(pooled.numpy() - ref).max())


def test_independent_numpy_reference_agrees() -> None:
    assert _numpy_reference_delta(_scratch_model()) < 1e-6


def test_independent_numpy_reference_catches_a_broken_pool() -> None:
    delta = _numpy_reference_delta(_scratch_model(_BrokenSumPool))
    assert delta > 1e-2, "sum-not-mean should diverge from the numpy reference"


# --- (iii) Row-permutation, both clauses --------------------------------------------


class _BrokenFirstTokenPool(TextEncoder):
    """Not permutation-invariant: reads position 0 instead of pooling."""

    def pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.proj(h[:, 0, :])


class _BrokenConstantTokens(TextEncoder):
    """Collapses every position to the sequence mean -- exactly what the central bet
    (§4.0) feared `tokens()` might already be: no information beyond what `pool()`
    already carries."""

    def tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, mask = super().tokens(input_ids, attention_mask)
        collapsed = h.mean(dim=1, keepdim=True).expand_as(h).clone()
        return collapsed, mask


def _permutation(total: int, seed: int = 3) -> torch.Tensor:
    """A permutation of the WHOLE T axis -- real and padded positions together. Valid
    because `pool()` must ignore padded slots regardless of where they land, which is
    the point revision 2 got wrong by permuting `h` without permuting `mask`.

    ``seed=3`` is not arbitrary: it is chosen so position 0 actually moves (some seeds
    happen to fix position 0, which would make `_BrokenFirstTokenPool` -- which always
    reads position 0 -- invisible to this permutation by accident rather than because
    the check is sound).
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=g)
    assert perm[0] != 0, (
        "seed must move position 0, or the first-token-pool negative test is vacuous"
    )
    return perm


def test_permutation_leaves_pool_unchanged() -> None:
    model = _scratch_model()
    ids, mask = _scratch_batch()
    perm = _permutation(mask.size(1))
    with torch.no_grad():
        h, m = model.tokens(ids, mask)
        before = model.pool(h, m)
        after = model.pool(h[:, perm, :], m[:, perm])
    assert (before - after).abs().max().item() < _TOL


def test_permutation_invariance_is_caught_when_pool_is_position_sensitive() -> None:
    model = _scratch_model(_BrokenFirstTokenPool)
    ids, mask = _scratch_batch()
    perm = _permutation(mask.size(1))
    with torch.no_grad():
        h, m = model.tokens(ids, mask)
        before = model.pool(h, m)
        after = model.pool(h[:, perm, :], m[:, perm])
    delta = (before - after).abs().max().item()
    assert delta > 1e-2, "reading position 0 is not permutation-invariant; should be caught"


def test_permutation_must_change_tokens() -> None:
    """The second clause A10 asked for and revision 2 dropped: `tokens()` itself must
    NOT be constant across positions, or the invariance check above would be vacuous
    (permuting identical rows changes nothing, correct or not)."""
    model = _scratch_model()
    ids, mask = _scratch_batch()
    perm = _permutation(mask.size(1))
    with torch.no_grad():
        h, _m = model.tokens(ids, mask)
    assert not torch.equal(h[:, perm, :], h)


def test_permutation_change_is_caught_when_tokens_collapse_to_a_constant() -> None:
    model = _scratch_model(_BrokenConstantTokens)
    ids, mask = _scratch_batch()
    perm = _permutation(mask.size(1))
    with torch.no_grad():
        h, _m = model.tokens(ids, mask)
    assert torch.equal(h[:, perm, :], h), (
        "constant tokens() is indistinguishable under permutation; "
        "the invariance check above would have missed this defect entirely"
    )


# --- (iv) Batch-composition invariance -----------------------------------------------

_BATCH_CFG = TextEncoderConfig(vocab_size=200, dim=32, depth=2, n_heads=4, max_len=66, out_dim=None)


class _BrokenNoAttentionMask(TextEncoder):
    """Drops the attention mask before the blocks. Real tokens then attend to padding,
    so the result depends on how much padding the batch happened to carry -- the exact
    bug `TextEncoder`'s own docstring records finding, at cosine 0.958."""

    def tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = input_ids.shape
        h = self.embed(input_ids) + self.pos_embed[:, :t]
        for block in self.blocks:
            h = block(h, None)  # BUG: mask not threaded into attention
        h = self.norm(h)
        mask = torch.ones(b, t, dtype=h.dtype) if attention_mask is None else attention_mask
        return h, mask


def _batch_composition_cosine_and_diff(model: TextEncoder) -> tuple[float, float]:
    torch.manual_seed(3)
    short_len, long_len = 6, 66
    ids_core = torch.randint(1, _BATCH_CFG.vocab_size, (1, short_len))
    ids_short, mask_short = ids_core, torch.ones(1, short_len)
    ids_long = torch.cat([ids_core, torch.zeros(1, long_len - short_len, dtype=torch.long)], dim=1)
    mask_long = torch.cat([torch.ones(1, short_len), torch.zeros(1, long_len - short_len)], dim=1)
    with torch.no_grad():
        h_s, m_s = model.tokens(ids_short, mask_short)
        p_short = model.pool(h_s, m_s)
        h_l, m_l = model.tokens(ids_long, mask_long)
        p_long = model.pool(h_l, m_l)
    cos = torch.nn.functional.cosine_similarity(p_short, p_long).item()
    diff = (p_short - p_long).abs().max().item()
    return cos, diff


def test_batch_composition_invariance_holds() -> None:
    model = TextEncoder(_BATCH_CFG).eval()
    cos, diff = _batch_composition_cosine_and_diff(model)
    assert cos > 1 - 1e-4
    assert diff < 1e-4


def test_batch_composition_invariance_is_caught_when_the_mask_is_dropped() -> None:
    model = _BrokenNoAttentionMask(_BATCH_CFG).eval()
    cos, diff = _batch_composition_cosine_and_diff(model)
    assert cos < 1 - 1e-4 or diff > 1e-3, (
        f"dropping the attention mask should make the result batch-composition "
        f"dependent (cos={cos}, diff={diff}); should be caught"
    )


# ---------------------------------------------------------------------------------
# Bonus: the row-permutation contract, confirmed for `visual` too. W0 wires the visual
# encoder's token surface FIRST (its output shape was already correct), so its
# `tokens()`/`pool()` get the same non-vacuous check TextEncoder's does above.
# ---------------------------------------------------------------------------------

_VL_SCRATCH_CFG = JEPAConfig(
    image_size=32, patch_size=8, dim=48, depth=2, n_heads=4, predictor_dim=24, predictor_depth=2
)


def test_visual_permutation_leaves_pool_unchanged_and_changes_tokens() -> None:
    torch.manual_seed(4)
    model = IJEPA(_VL_SCRATCH_CFG).eval()
    images = torch.randn(3, 3, _VL_SCRATCH_CFG.image_size, _VL_SCRATCH_CFG.image_size)
    perm = _permutation(_VL_SCRATCH_CFG.n_patches, seed=5)

    with torch.no_grad():
        h, mask = model.tokens(images)
        before = model.pool(h, mask)
        after = model.pool(h[:, perm, :], mask[:, perm])

    assert h.shape == (3, _VL_SCRATCH_CFG.n_patches, _VL_SCRATCH_CFG.dim)
    assert (before - after).abs().max().item() < _TOL
    assert not torch.equal(h[:, perm, :], h)
