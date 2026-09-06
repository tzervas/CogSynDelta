"""W0 -- the `Faculty` protocol: conformance, shape/mask invariants, and the DEC-47 guard.

`tests/test_token_surface.py` already proves `pool(tokens(x)) == encode(x)` and the four
failable pooling properties on the RAW `TextEncoder`/`ViTEncoder`/`IJEPA` methods -- that
work is not repeated here. This file is about the layer W0 actually adds on top of it:
`cogsyndelta.faculty.protocol.Faculty` and the adapters in `cogsyndelta.faculty.adapters`
that make the existing encoders satisfy it.

Structure:
  1. A bare encoder does NOT satisfy `Faculty` (missing attributes) -- the adapters are
     not decoration, they close a real gap.
  2. Each adapter DOES satisfy `Faculty` (`@runtime_checkable` `isinstance`).
  3. `tokens()`/`pool()` shape and mask invariants, and the `context_tokens` budget
     check, on both adapters.
  4. `pool()` through the adapter is bit-for-bit the encoder's own existing pooled
     output -- the adapter adds no computation.
  5. The `condition` guard: not-yet-implemented, refuses loudly rather than silently
     ignoring the argument.
  6. THE DEC-47 GUARD: `isinstance(fake, Faculty)` cannot catch a `tokens()` that
     returns discrete vocabulary ids instead of latents -- `Protocol` only checks that
     names resolve, not what they return. `assert_latent_tokens` is what catches it, and
     this file proves it fires against a fake faculty built to violate the invariant.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

# See tests/test_token_surface.py for why this must precede the cogsyndelta imports
# below: cogsyndelta.regions.text_encoder does not itself need `tokenizers`, but the
# train dependency group gates collection in this repo's convention regardless.
pytest.importorskip("tokenizers", reason="train group not installed")

from cogsyndelta.faculty.adapters import (
    TextFacultyAdapter,
    VisualFacultyAdapter,
    kv_bytes_per_token,
)
from cogsyndelta.faculty.protocol import Faculty, assert_latent_tokens, token_mask
from cogsyndelta.model.vl_jepa import IJEPA, JEPAConfig
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig

pytestmark = pytest.mark.cpu

_TEXT_CFG = TextEncoderConfig(vocab_size=200, dim=32, depth=2, n_heads=4, max_len=32)
_VL_CFG = JEPAConfig(
    image_size=32, patch_size=8, dim=48, depth=2, n_heads=4, predictor_dim=24, predictor_depth=2
)


def _text_encoder() -> TextEncoder:
    torch.manual_seed(0)
    return TextEncoder(_TEXT_CFG).eval()


def _ijepa() -> IJEPA:
    torch.manual_seed(0)
    return IJEPA(_VL_CFG).eval()


def _text_batch(batch: int = 3, real: int = 5, total: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(1, _TEXT_CFG.vocab_size, (batch, total), generator=g)
    mask = torch.zeros(batch, total)
    mask[:, :real] = 1.0
    return ids, mask


# ---------------------------------------------------------------------------------
# 1. A bare encoder does not satisfy Faculty -- the adapters close a real gap.
# ---------------------------------------------------------------------------------


def test_bare_text_encoder_does_not_satisfy_faculty() -> None:
    """`TextEncoder` has `tokens()`/`pool()` (landed pre-W0, commit e5e57f6) but not
    `faculty`/`token_dim`/`pooled_dim`/`kv_bytes_per_token`/`accepts_condition`, and its
    `tokens()` does not accept `context_tokens`/`condition` -- `isinstance` under
    `@runtime_checkable` checks attribute NAMES, and these are missing.
    """
    assert not isinstance(_text_encoder(), Faculty)


def test_bare_ijepa_does_not_satisfy_faculty() -> None:
    assert not isinstance(_ijepa(), Faculty)


# ---------------------------------------------------------------------------------
# 2. Each adapter satisfies Faculty.
# ---------------------------------------------------------------------------------


def test_text_adapter_satisfies_faculty() -> None:
    adapter = TextFacultyAdapter(encoder=_text_encoder(), faculty="language")
    assert isinstance(adapter, Faculty)
    assert adapter.name == "language"
    assert adapter.accepts_condition is False


def test_text_adapter_default_name_falls_back_to_faculty() -> None:
    adapter = TextFacultyAdapter(encoder=_text_encoder(), faculty="hippocampus", name="memory")
    assert adapter.name == "memory"
    assert adapter.faculty == "hippocampus"


def test_visual_adapter_satisfies_faculty() -> None:
    adapter = VisualFacultyAdapter(model=_ijepa())
    assert isinstance(adapter, Faculty)
    assert adapter.name == "visual"
    assert adapter.faculty == "visual_cortex"
    assert adapter.accepts_condition is False


# ---------------------------------------------------------------------------------
# 3. Shape / mask invariants and the context_tokens budget.
# ---------------------------------------------------------------------------------


def test_text_tokens_shape_and_mask_invariant() -> None:
    adapter = TextFacultyAdapter(encoder=_text_encoder(), faculty="language")
    ids, mask = _text_batch(batch=3, real=5, total=8)
    h, m = adapter.tokens((ids, mask), context_tokens=8)
    assert h.shape == (3, 8, adapter.token_dim)
    assert m.shape == (3, 8)
    assert torch.equal(m, mask)


def test_text_tokens_refuses_when_input_exceeds_context_tokens_budget() -> None:
    adapter = TextFacultyAdapter(encoder=_text_encoder(), faculty="language")
    ids, mask = _text_batch(batch=2, real=5, total=8)
    with pytest.raises(ValueError, match="context_tokens"):
        adapter.tokens((ids, mask), context_tokens=4)


def test_visual_tokens_shape_and_mask_invariant() -> None:
    adapter = VisualFacultyAdapter(model=_ijepa())
    images = torch.randn(2, 3, _VL_CFG.image_size, _VL_CFG.image_size)
    h, m = adapter.tokens(images, context_tokens=_VL_CFG.n_patches)
    assert h.shape == (2, _VL_CFG.n_patches, adapter.token_dim)
    assert m.shape == (2, _VL_CFG.n_patches)
    assert torch.equal(m, torch.ones_like(m))


def test_visual_tokens_refuses_when_patch_count_exceeds_context_tokens_budget() -> None:
    adapter = VisualFacultyAdapter(model=_ijepa())
    images = torch.randn(2, 3, _VL_CFG.image_size, _VL_CFG.image_size)
    with pytest.raises(ValueError, match="context_tokens"):
        adapter.tokens(images, context_tokens=_VL_CFG.n_patches - 1)


# ---------------------------------------------------------------------------------
# 4. pool() through the adapter is bit-for-bit the encoder's own pooled output.
# ---------------------------------------------------------------------------------


def test_text_adapter_pool_is_bit_exact_with_encoder_forward() -> None:
    encoder = _text_encoder()
    adapter = TextFacultyAdapter(encoder=encoder, faculty="language")
    ids, mask = _text_batch()
    with torch.no_grad():
        expected = encoder(ids, mask)
        h, m = adapter.tokens((ids, mask), context_tokens=ids.size(1))
        actual = adapter.pool(h, m)
    assert torch.equal(actual, expected)


def test_visual_adapter_pool_is_bit_exact_with_ijepa_encode() -> None:
    model = _ijepa()
    adapter = VisualFacultyAdapter(model=model)
    images = torch.randn(2, 3, _VL_CFG.image_size, _VL_CFG.image_size)
    with torch.no_grad():
        expected = model.encode(images)
        h, m = adapter.tokens(images, context_tokens=_VL_CFG.n_patches)
        actual = adapter.pool(h, m)
    assert torch.equal(actual, expected)


# ---------------------------------------------------------------------------------
# 5. The condition guard: refuses loudly, does not silently ignore.
# ---------------------------------------------------------------------------------


def test_text_adapter_refuses_a_condition() -> None:
    adapter = TextFacultyAdapter(encoder=_text_encoder(), faculty="language")
    ids, mask = _text_batch()
    with pytest.raises(NotImplementedError, match="accepts_condition"):
        adapter.tokens((ids, mask), context_tokens=8, condition=torch.zeros(3, 1, 32))


def test_visual_adapter_refuses_a_condition() -> None:
    adapter = VisualFacultyAdapter(model=_ijepa())
    images = torch.randn(2, 3, _VL_CFG.image_size, _VL_CFG.image_size)
    with pytest.raises(NotImplementedError, match="accepts_condition"):
        adapter.tokens(images, context_tokens=_VL_CFG.n_patches, condition=torch.zeros(2, 1, 48))


# ---------------------------------------------------------------------------------
# kv_bytes_per_token (c_r): reproduces the taxonomy's catalogue figures as a formula.
# ---------------------------------------------------------------------------------


def test_kv_bytes_per_token_matches_taxonomy_text_figure() -> None:
    """§1.4's catalogue: `kv_bytes_per_token: 4096` for a 256-dim/depth-4 text region."""
    assert kv_bytes_per_token(dim=256, depth=4) == 4096


def test_kv_bytes_per_token_matches_taxonomy_visual_figure() -> None:
    """§1.4's catalogue: `kv_bytes_per_token: 9216` for `visual` (384-dim/depth-6)."""
    assert kv_bytes_per_token(dim=384, depth=6) == 9216


def test_text_adapter_kv_bytes_per_token_matches_formula() -> None:
    adapter = TextFacultyAdapter(encoder=_text_encoder(), faculty="language")
    assert adapter.kv_bytes_per_token == kv_bytes_per_token(_TEXT_CFG.dim, _TEXT_CFG.depth)


# ---------------------------------------------------------------------------------
# token_mask(): names the mask half of a tokens() result.
# ---------------------------------------------------------------------------------


def test_token_mask_returns_the_mask_element() -> None:
    ids, mask = _text_batch()
    adapter = TextFacultyAdapter(encoder=_text_encoder(), faculty="language")
    result = adapter.tokens((ids, mask), context_tokens=8)
    assert torch.equal(token_mask(result), mask)


# ---------------------------------------------------------------------------------
# 6. THE DEC-47 GUARD, proven to fire against a fake faculty.
# ---------------------------------------------------------------------------------


@dataclass
class _FakeVocabFaculty:
    """Structurally satisfies `Faculty` -- every attribute `isinstance` checks for is
    present -- but its `tokens()` violates DEC-47 by returning discrete vocabulary ids
    (`dtype=torch.long`) instead of position latents. This is exactly the failure mode
    `isinstance(fake, Faculty)` CANNOT catch: `Protocol.__runtime_checkable__` only
    verifies that named attributes resolve via `hasattr`, never what a method returns.
    `assert_latent_tokens` is the check that actually catches it.
    """

    name: str = "fake"
    faculty: str = "fake"
    token_dim: int = 8
    pooled_dim: int = 8
    kv_bytes_per_token: int = 1024
    accepts_condition: bool = False

    def tokens(self, inputs, *, context_tokens, condition=None):
        b = inputs.shape[0] if hasattr(inputs, "shape") else 1
        # THE VIOLATION: vocabulary ids (long), not latents (float).
        h = torch.randint(0, 100, (b, context_tokens, self.token_dim), dtype=torch.long)
        mask = torch.ones(b, context_tokens)
        return h, mask

    def pool(self, h, mask):
        return h.float().mean(dim=1)


def test_fake_vocab_faculty_structurally_satisfies_the_protocol() -> None:
    """The isinstance check alone is blind to the DEC-47 violation -- demonstrated so
    the guard test below is not mistaken for a check `isinstance` already performed."""
    fake = _FakeVocabFaculty()
    assert isinstance(fake, Faculty)


def test_assert_latent_tokens_accepts_a_real_faculty() -> None:
    adapter = TextFacultyAdapter(encoder=_text_encoder(), faculty="language")
    ids, mask = _text_batch()
    h, _m = adapter.tokens((ids, mask), context_tokens=8)
    assert_latent_tokens(h)  # must not raise


def test_assert_latent_tokens_fires_on_the_fake_faculty() -> None:
    """THE GUARD, proven to fire: a fake faculty that passes `isinstance(_, Faculty)`
    but returns `torch.long` vocabulary ids from `tokens()` is refused by
    `assert_latent_tokens`, naming the offending dtype."""
    fake = _FakeVocabFaculty()
    h, _mask = fake.tokens(torch.zeros(2, 4), context_tokens=4)
    assert h.dtype == torch.long, "the fake must actually violate DEC-47, or this test is vacuous"
    with pytest.raises(TypeError, match="DEC-47"):
        assert_latent_tokens(h)
