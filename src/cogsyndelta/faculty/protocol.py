"""The `Faculty` protocol -- row W0 of `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md`.

WHAT THIS IS, IN THE TAXONOMY'S OWN WORDS
Row W0 (§4.1): *"Token surface -- read 'position latents': per-position hidden vectors,
not discrete ids `[DEC-47]`. The method name `tokens()` is the API and is NOT renamed."*
§2.2 gives the exact class this module lifts into ``src/``:

    @runtime_checkable
    class Faculty(Protocol):
        name: str
        faculty: str
        token_dim: int                  # what tokens() emits, PRE-proj: 256 text, 384 visual
        pooled_dim: int                 # what pool() emits, POST-proj; every receipt is about this
        kv_bytes_per_token: int         # c_r -- THE REGION DECLARES ITS OWN COST
        accepts_condition: bool

        def tokens(self, inputs, *, context_tokens: int,
                   condition: Tensor | None = None) -> tuple[Tensor, Tensor]: ...
        def pool(self, h: Tensor, mask: Tensor) -> Tensor: ...

DEC-47 -- THE GLOSS THAT GOVERNS EVERY USE OF THE WORD BELOW (taxonomy §2.2, stated once
and cited everywhere else): *"Token surface" and `tokens()` mean position latents --
per-position hidden vectors, not discrete ids. ... nothing in this document ever means
discrete tokens when it says a region's tokens; `pool()` is a latent too. The only
discrete ids anywhere in the system are at the input tokenizer and at the frontal
read-out.* `assert_latent_tokens` below is the local, W0-scale version of the checked
property DEC-47 promises at W9 ("no inter-region path carries discrete token ids, and
the constructed violation is refused") -- this module cannot build W9's runtime
assertion (that needs the controller/workspace, W0 explicitly does not), but it can and
does make the SAME claim checkable against one region's `tokens()` output today.

DEC-14 / DEC-15 (taxonomy §2.2): *"The region contract becomes `tokens(inputs,
context_tokens) -> (h, mask)`; `activate(stream)->[B,D]` dies"* and *"Two budget
currencies, two simplexes: `ctx_r` (pre-computation, region-internal) and `b_r`
(post-computation, workspace KV)."* `context_tokens` below is `ctx_r`'s consumer-side
argument -- the region "never encodes more than its budget" -- and `kv_bytes_per_token`
is `c_r`, "THE REGION DECLARES ITS OWN COST."

WHY THIS MODULE IS SMALL ON PURPOSE
This lane (W0) is explicitly *"Keep it small; no controller, no workspace, no episodic
store"* -- those are DEC-16's module (§2.3) and later rows (E0/E1). Per
`docs/technical/README.md`'s "four facts": *"The interconnect does not exist. Everything
in §2 -- workspace, adapters, `a_r`, `b_r`, write-back, the `Schedule` -- is design
only."* So `condition` (DEC-17's write-back conditioning prefix) is part of the typed
signature below -- the taxonomy's method name is not renamed, and a caller that never
passes it is unaffected -- but no adapter in `cogsyndelta.faculty.adapters` actually
consumes a non-`None` value yet; they raise rather than silently ignore it (see that
module's docstring).

WHERE THIS DEVIATES FROM A LOOSER PARAPHRASE OF THE SAME ROW
An earlier, informal restatement of this row described the surface as `tokens(batch) ->
Tensor[B, T, D]` / `pool(batch) -> Tensor[B, D]` plus a single `latent_dim` and a
`token_mask` accessor. That is not what the taxonomy specifies, and this module follows
the taxonomy, not the paraphrase, because the paraphrase collapses two things the design
deliberately keeps separate:

  - `tokens()` takes the REGION'S NATIVE input (token ids for a text region, an image
    tensor for `visual`), not a generic "batch", and returns a `(h, mask)` PAIR --
    `mask` is not a separate accessor because it is already a return value: see
    `token_mask()` below, which names that element rather than inventing a second call.
  - There are TWO widths, not one `latent_dim`: `token_dim` (pre-`proj`, what `tokens()`
    emits: 256 for every text region today, 384 for `visual`) and `pooled_dim`
    (post-`proj`, what `pool()` emits). Calibration (§2.3): *"a text region is
    16,021,248 params of which 12,865,792 (80.30%) is the token embedding table"* --
    the embedding table lives at `token_dim`, and collapsing the two widths into one
    `latent_dim` would hide exactly the split `proj` (the pooling head) exists to make.

Both widths happen to be equal for every region trained so far (no region's `out_dim`
differs from its `dim` in production), which is why a single-width paraphrase can look
right without being right -- see `cogsyndelta.faculty.param_table` for the measured
per-region pooling-head parameter count (0 for all five, at today's configs), which is
the evidence for "equal so far," not "equal by contract."
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

__all__ = ["Faculty", "assert_latent_tokens", "token_mask"]


@runtime_checkable
class Faculty(Protocol):
    """A region's token-surface contract, taxonomy §2.2, lifted verbatim.

    `token_dim`/`pooled_dim`/`kv_bytes_per_token`/`accepts_condition` are declared as
    plain attributes here (matching the taxonomy's dataclass-shaped sketch); a
    conforming implementation may back them with `@property` -- `isinstance` under
    `@runtime_checkable` only checks that the name resolves via `hasattr`, not how.
    """

    name: str
    """Region name, e.g. `"language"` (legacy `"code"`), `"memory"`, `"visual"`."""

    faculty: str
    """Cognitive-faculty id the region belongs to (taxonomy §1.4), distinct from
    `name`: `language`'s faculty is also `"language"` today, but `memory`'s faculty is
    `"hippocampus"` -- see `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` §1.4."""

    token_dim: int
    """What `tokens()` emits, PRE-`proj`: 256 for every trained text region, 384 for
    `visual` (measured, see `cogsyndelta.faculty.param_table`)."""

    pooled_dim: int
    """What `pool()` emits, POST-`proj`. Every existing receipt is a number at this
    width, not `token_dim` -- W1/W1d's whole point was that the two can differ and the
    receipts were only ever about one of them."""

    kv_bytes_per_token: int
    """`c_r` -- the region's own declared region-internal KV/activation cost per input
    position, bounding `ctx_r` in `Σ_r c_r · ctx_r ≤ B_kv` (DEC-15, §2.2). THE REGION
    DECLARES ITS OWN COST; nothing here computes a budget or enforces the simplex --
    that is the controller's job (§2.3), not built by this lane."""

    accepts_condition: bool
    """Whether this region's `tokens()` will actually apply a non-`None` `condition`
    (DEC-17's write-back conditioning prefix). Every adapter in
    `cogsyndelta.faculty.adapters` sets this `False` today: no controller exists yet to
    produce a `condition`, and DEC-17's write-back circuitry is design-only
    (`docs/technical/README.md` fact 2). Declaring `True` here would assert a
    capability nothing implements."""

    def tokens(
        self,
        inputs: Any,
        *,
        context_tokens: int,
        condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Region-native representation BEFORE pooling AND BEFORE `proj`.

        Computed over AT MOST `context_tokens` input positions -- the region never
        encodes more than its budget (DEC-15's `ctx_r`). `condition`, when not `None`,
        is the workspace latent state projected to `token_dim` and prepended as a
        conditioning prefix (DEC-17); `None` at iteration 0 and, today, always (see the
        module docstring).

        Returns:
            `(h [B, T_r, token_dim], mask [B, T_r])` with `T_r <= context_tokens`.
            `mask` is 1 for a real position, 0 for padding. `h` is always floating
            point -- see `assert_latent_tokens`.
        """
        ...

    def pool(self, h: Tensor, mask: Tensor) -> Tensor:
        """`[B, T_r, token_dim] -> [B, pooled_dim]`: masked mean, THEN `proj`.

        The region's standalone answer, kept so every existing receipt stays
        reproducible and comparable. For `visual`, `pool` is `h.mean(1)` and is exactly
        `IJEPA.encode` (DEC-34) -- there is no `proj` for that region today.
        """
        ...


def token_mask(tokens_result: tuple[Tensor, Tensor]) -> Tensor:
    """Name the `mask` half of a `tokens()` call.

    `Faculty.tokens()` already returns `(h, mask)` -- the taxonomy does not give
    `mask` a separate accessor method, because it would just be a second call
    returning half of what the first call already computed. This function exists so a
    caller that wants "the padded-position mask, given a `tokens()` result" has a name
    for that half rather than a bare `[1]` index scattered through call sites.

    Args:
        tokens_result: The `(h, mask)` pair `Faculty.tokens()` returns.

    Returns:
        `mask`, `[B, T_r]`, 1 for a real position and 0 for padding.
    """
    _h, mask = tokens_result
    return mask


def assert_latent_tokens(h: Tensor) -> None:
    """DEC-47's invariant, checked against one region's `tokens()` output.

    *"The only discrete ids anywhere in the system are at the input tokenizer and at
    the frontal read-out."* A `tokens()` implementation that returns integer
    vocabulary ids instead of position latents would violate that at the first hop --
    exactly the "constructed violation" DEC-47 requires W9 to refuse at the
    interconnect level. W9 needs the controller (not built by this lane); this
    function is the same check applied directly to a region's own output, so the
    property is enforceable today rather than only after the interconnect exists.

    Args:
        h: The first element of a `tokens()` result.

    Raises:
        TypeError: `h` is not a floating-point tensor -- e.g. `torch.long` vocabulary
            ids leaking through the token-surface API instead of latents.
    """
    if not torch.is_floating_point(h):
        raise TypeError(
            f"tokens() returned dtype {h.dtype}, which is not floating point. DEC-47 "
            f"requires the token surface to be position latents, never discrete "
            f"vocabulary ids: 'the only discrete ids anywhere in the system are at the "
            f"input tokenizer and at the frontal read-out' "
            f"(docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md, DEC-47)."
        )
