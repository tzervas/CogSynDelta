"""Adapters that make the existing encoders satisfy `cogsyndelta.faculty.protocol.Faculty`.

WHY ADAPTERS, GIVEN `TextEncoder`/`ViTEncoder`/`IJEPA` ALREADY HAVE `tokens()`/`pool()`
Commit `e5e57f6` (2026-09-02) already split `TextEncoder.forward` and `ViTEncoder.forward`
at the pooling line -- `tokens()`/`pool()` exist on both, and `tests/test_token_surface.py`
already proves `pool(tokens(x)) == encode(x)` and the four failable properties (pad
invariance, independent reference, row-permutation both clauses, batch-composition). That
work landed BEFORE this lane and is not redone here.

What is missing is the `Faculty` PROTOCOL surface (§2.2): `name`, `faculty`, `token_dim`,
`pooled_dim`, `kv_bytes_per_token`, `accepts_condition`, and a `tokens()` call shaped
`(inputs, *, context_tokens, condition=None)`. `TextEncoder.tokens(input_ids,
attention_mask=None)` and `ViTEncoder`/`IJEPA.tokens(images)` do not carry those
attributes and do not accept `context_tokens`/`condition` at all -- `isinstance(encoder,
Faculty)` is `False` for a bare `TextEncoder` or `IJEPA` instance (see
`tests/test_faculty_protocol.py::test_bare_encoder_does_not_satisfy_faculty`). These
adapters bridge that gap by WRAPPING an already-instantiated encoder:

  - They add no parameters and change no weights -- `tokens()`/`pool()` below delegate
    straight through to the wrapped module's own methods, so `pool()` is bit-for-bit the
    module's existing pooled output (tested).
  - `context_tokens` is enforced as a BOUND CHECK against the input actually given, not
    a truncation the region performs -- W0 does not build the controller that decides
    `ctx_r` per iteration (§2.3, DEC-16), so there is nothing here to truncate against
    yet; the adapter can and does refuse an input that already exceeds the declared
    budget, which is the honest subset of "the region never encodes more than its
    budget" (DEC-15) available without that controller.
  - `condition` is accepted (the signature is not renamed) but every adapter here sets
    `accepts_condition = False` and RAISES on a non-`None` value, rather than silently
    ignoring it. The taxonomy's §1.4 catalogue (a proposed diff, headed "REVIEW ONLY, DO
    NOT APPLY" -- `docs/technical/README.md` fact 1) declares `accepts_condition: true`
    for `language`/`memory`/`reasoning`/`visual`, but DEC-17's write-back conditioning
    prefix has no producer: no controller exists to build a `condition` tensor
    (`docs/technical/README.md` fact 2, "the interconnect does not exist"). Declaring
    `True` here would assert a capability nothing implements; raising loudly on a
    non-`None` value is the same "verify guards by making them fail" discipline the rest
    of this repo applies to load-bearing claims.

`kv_bytes_per_token` (`c_r`) is computed, not hand-copied from the catalogue: `2 * dim *
depth * 2` bytes -- two vectors (K and V) of width `dim`, one pair per layer (`depth`
layers), stored at 2 bytes/element (fp16). This reproduces the catalogue's own committed
figures for both region families measured so far -- `4096` for a 256-dim/depth-4 text
region, `9216` for `visual` at 384-dim/depth-6 -- as a formula rather than a second copy
of the same two numbers that could drift from the config that actually produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from cogsyndelta.model.vl_jepa import IJEPA
from cogsyndelta.regions.text_encoder import TextEncoder

__all__ = ["TextFacultyAdapter", "VisualFacultyAdapter", "kv_bytes_per_token"]


def kv_bytes_per_token(
    dim: int, depth: int, *, kv_vectors: int = 2, bytes_per_elem: int = 2
) -> int:
    """`c_r`: region-internal KV/activation cost per input position (DEC-15).

    `kv_vectors=2` (K and V), `bytes_per_elem=2` (fp16) match every figure the taxonomy's
    §1.4 catalogue records; both are keyword-only knobs rather than hardcoded so a
    future int8-KV region (§section on `D_sched`/int8-KV, doc 07) can pass
    `bytes_per_elem=1` without a second formula.

    Args:
        dim: The region's native (pre-`proj`) width -- `token_dim`.
        depth: Number of transformer blocks (`TextEncoderConfig.depth` /
            `JEPAConfig.depth`).

    Returns:
        Bytes per input position, e.g. `4096` for `dim=256, depth=4`.
    """
    return kv_vectors * dim * depth * bytes_per_elem


def _split_inputs(inputs: Any) -> tuple[Tensor, Tensor | None]:
    """`TextEncoder.tokens` takes `(input_ids, attention_mask=None)`; `Faculty.tokens`
    takes one `inputs` argument. Accept either an `input_ids` tensor alone or an
    `(input_ids, attention_mask)` pair, matching `TextEncoder.encode`'s own convention.
    """
    if isinstance(inputs, tuple):
        input_ids, attention_mask = inputs
        return input_ids, attention_mask
    return inputs, None


@dataclass
class TextFacultyAdapter:
    """Wraps a `TextEncoder` (`language`, `memory`, `reasoning`, ... any text region --
    `regions/memory.py` builds a plain `TextEncoder(name="memory")`, so this one adapter
    class covers every text region; there is no separate `MemoryEncoder` class to wrap).

    Args:
        encoder: An already-built and (typically) already-trained `TextEncoder`. Not
            copied, not re-initialized -- this adapter changes no weights.
        faculty: Cognitive-faculty id (taxonomy §1.4), e.g. `"language"`, `"hippocampus"`
            (for `memory`), `"reasoning"`.
        name: Region name; defaults to `faculty` when not given (correct for regions
            whose name and faculty id coincide, e.g. `visual`/`visual_cortex` is the one
            counter-example and gets its own default in `VisualFacultyAdapter`).
    """

    encoder: TextEncoder
    faculty: str
    name: str = ""
    accepts_condition: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Default `name` to `faculty` when the caller did not give one."""
        if not self.name:
            self.name = self.faculty

    @property
    def token_dim(self) -> int:
        """`TextEncoderConfig.dim` -- what `tokens()` emits, pre-`proj`."""
        return self.encoder.cfg.dim

    @property
    def pooled_dim(self) -> int:
        """`TextEncoder.out_dim` -- what `pool()` emits, post-`proj`."""
        return self.encoder.out_dim

    @property
    def kv_bytes_per_token(self) -> int:
        """`c_r`, computed from this encoder's actual `dim`/`depth`."""
        return kv_bytes_per_token(self.encoder.cfg.dim, self.encoder.cfg.depth)

    def tokens(
        self,
        inputs: Any,
        *,
        context_tokens: int,
        condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Delegate to `TextEncoder.tokens`, after a budget check and a condition guard.

        Raises:
            NotImplementedError: `condition` is not `None` -- no controller exists to
                produce one yet (see the module docstring).
            ValueError: The input already carries more positions than `context_tokens`
                allows.
        """
        if condition is not None:
            raise NotImplementedError(
                f"{self.name}: tokens() was called with a non-None condition, but "
                f"accepts_condition=False -- DEC-17's write-back conditioning prefix "
                f"has no producer yet (no controller exists, "
                f"docs/technical/README.md fact 2). Refusing rather than silently "
                f"ignoring the argument."
            )
        input_ids, attention_mask = _split_inputs(inputs)
        t = input_ids.shape[1]
        if t > context_tokens:
            raise ValueError(
                f"{self.name}: input has {t} positions, exceeding context_tokens "
                f"budget {context_tokens} (ctx_r, DEC-15). The region never encodes "
                f"more than its budget."
            )
        return self.encoder.tokens(input_ids, attention_mask)

    def pool(self, h: Tensor, mask: Tensor) -> Tensor:
        """Delegate to `TextEncoder.pool`, unchanged."""
        return self.encoder.pool(h, mask)


@dataclass
class VisualFacultyAdapter:
    """Wraps an `IJEPA` model for the `visual` faculty.

    `IJEPA.tokens()`/`.pool()` already delegate to `target_encoder` (DEC-34's deployed
    half), so this adapter forwards to `IJEPA` directly rather than reaching into
    `.target_encoder` itself -- `pool(tokens(x))` and `IJEPA.encode(x)` then stay
    provably the same call path this adapter did not have to reproduce.

    Args:
        model: An already-built (and typically trained) `IJEPA`.
        faculty: Defaults to `"visual_cortex"` (taxonomy §1.4's `visual` faculty id).
        name: Region name; defaults to `"visual"`.
    """

    model: IJEPA
    faculty: str = "visual_cortex"
    name: str = "visual"
    accepts_condition: bool = field(default=False, init=False)

    @property
    def token_dim(self) -> int:
        """`JEPAConfig.dim` -- what `tokens()` emits. `visual` has no `proj`, so this
        equals `pooled_dim` (measured: `cogsyndelta.faculty.param_table` reports a
        0-parameter pooling head for `visual`, consistent with `ViTEncoder.pool`'s own
        docstring: 'unlike the text regions, ViTEncoder's output width already is the
        shared-stream width the catalogue declares').
        """
        return self.model.cfg.dim

    @property
    def pooled_dim(self) -> int:
        """Equal to `token_dim` -- see that property's docstring."""
        return self.model.cfg.dim

    @property
    def kv_bytes_per_token(self) -> int:
        """`c_r`, computed from this model's actual `dim`/`depth`."""
        return kv_bytes_per_token(self.model.cfg.dim, self.model.cfg.depth)

    def tokens(
        self,
        inputs: Any,
        *,
        context_tokens: int,
        condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Delegate to `IJEPA.tokens`, after a budget check and a condition guard.

        Raises:
            NotImplementedError: `condition` is not `None` (see the module docstring).
            ValueError: The encoder's patch count exceeds `context_tokens`.
        """
        if condition is not None:
            raise NotImplementedError(
                f"{self.name}: tokens() was called with a non-None condition, but "
                f"accepts_condition=False -- see TextFacultyAdapter.tokens's docstring "
                f"for why (identical reasoning, same lane)."
            )
        h, mask = self.model.tokens(inputs)
        t = h.shape[1]
        if t > context_tokens:
            raise ValueError(
                f"{self.name}: encoder emits {t} patch tokens, exceeding "
                f"context_tokens budget {context_tokens} (ctx_r, DEC-15)."
            )
        return h, mask

    def pool(self, h: Tensor, mask: Tensor) -> Tensor:
        """Delegate to `IJEPA.pool`, unchanged."""
        return self.model.pool(h, mask)
