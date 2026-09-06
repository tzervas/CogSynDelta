"""Frontal read-out and ranking -- lane IC-4 of the interconnect module.

WHAT THIS IS, IN THE SPEC'S OWN WORDS
`docs/design/INTERCONNECT-MODULE-SPEC.md` section 2.1 Table 2 assigns this file four
classes: ``FrontalReadout``, ``RankHead``, ``NullCandidate`` and ``UnifyProbes``, "the
DEC-20 read-out, DEC-41 ranking, the ``NULL`` embedding, the ``L_unify`` probes, the
``z_affect`` seam behind a zero gate." Section 3 step 12 is the forward-pass contract
this module implements: *"``f = FrontalReadout(z_N)``, a learned-query attention pool
over the latents plus its MLP ... ``RankHead`` maps ``f`` and each candidate embedding
to ``[B, D]`` and scores by cosine over ``k = 32`` candidates, with ``NullCandidate``
supplying the learned embedding at index 0."* Section 4's loss definitions name
``L_task`` as "softmax cross-entropy over the ``k = 32`` cosine scores at a recorded
temperature" and ``L_unify`` as ``Σ_r (1 − cos(probe_r(f), pool_r(h_r)))`` over the
admitted regions -- ``RankLoss`` and ``UnifyLoss`` themselves are lane IC-7's
``losses.py``; this module builds only the modules those losses are computed over.
Table 8 row G36, the ``NULL`` gate, fires in ``gates.py`` (lane IC-10) over a phase-A
receipt; this module is named there only because ``NullCandidate`` -- the thing G36
measures -- is built here. Section 5 Table 9 is this file's test-plan row, reproduced
in ``tests/interconnect/test_readout.py``.

WHAT THIS MODULE OWNS AND WHAT IT DOES NOT
This file does not run any region and does not see raw inputs: it consumes ``z_N``,
the final workspace latents ``[B, L, D_w]`` produced by ``workspace.py`` (lane IC-1),
and the frozen candidate embeddings a caller (lane IC-8's ``mind.py``, or a test)
supplies. It does not build the Q7 ``CandidateEncoder`` that produces those candidate
embeddings from raw text/images -- Q7 is explicitly out of scope for every wave-1 lane
and is "conditionally" costed in Table 4 -- so ``RankHead`` here takes pre-embedded
``[B, k-1, D_w]`` content candidates and is agnostic to how they were produced.

SPEC AMBIGUITIES RESOLVED HERE (recorded per the task's instruction to do the smallest
honest thing where the spec is silent, and to record the choice)

1. **What exactly ``RankHead``'s one ``Linear(D_w, D_w)`` maps.** Table 4 costs the rank
   head at exactly ``512·512 + 512 = 262,656`` params, i.e. one linear layer. Step 12's
   prose is "``RankHead`` maps ``f`` AND EACH CANDIDATE EMBEDDING to ``[B, D]``" --
   plural, both sides. A single linear layer that only touched ``f`` would leave the
   candidate side untouched by any of this file's parameters, which does not match "and
   each candidate embedding." This module therefore applies the SAME shared
   ``Linear(D_w, D_w)`` to ``f`` and to every candidate (the null one included) before
   scoring by cosine -- one weight-tied projection, matching the 262,656-parameter
   budget exactly, read as a shared bi-encoder calibration head rather than a
   query-only reweighting.
2. **Where ``NullCandidate`` is spliced into the ``k``-length set.** Table 3's shape row
   for "rank scores" shows ``RankHead`` receiving candidates ``[B, k, D]`` with ``NULL``
   already at index 0, while step 12 says ``RankHead``'s own forward is where
   "``NullCandidate`` suppl[ies] the learned embedding at index 0." Read together: the
   public ``RankHead.forward`` takes the ``k - 1`` content candidates alone and builds
   the full ``k``-length set internally by prepending its own ``NullCandidate``
   submodule -- so from the caller's side the class's own contract already matches
   Table 3's "candidates ``[B, k, D]`` with ``NULL`` at index 0," and the splice point
   is exactly where the docstring says NullCandidate supplies it.
3. **``RankHead``'s ``temperature``.** Neither document gives a numeric default -- "a
   recorded temperature" (section 4) is deliberately a receipt field, not a constant.
   ``temperature`` is therefore a required constructor argument with no default here;
   ``mind.py`` (lane IC-8) is the place a concrete value gets chosen and recorded.
4. **The ``z_affect`` zero gate's shape and behaviour.** The spec says only that
   ``FrontalReadout`` "declares a named ``z_affect`` input behind a zero gate with no
   parameters instantiated at v1" and that "the workspace blocks and the region inputs
   never see it" (section 3, "Affect seam"; TAX's DEC-79 zero-gate text). This module
   reads "no parameters instantiated" as the binding constraint and therefore does not
   multiply ``z_affect`` into the computation at all (not even by a literal ``0.0``
   tensor op, which would still propagate a NaN or Inf were one ever passed) -- the
   argument is accepted and shape-checked, for the seam to exist as a declared API
   surface callers can pass through, and is then provably unused: ``f`` is bitwise
   identical whether ``z_affect`` is omitted or is an arbitrary finite tensor of the
   right shape. This is the "loud disconnect" read literally: disconnect is not a
   value passed through a multiplier, it is the absence of a data path.

Affect is not a v1 workspace participant (OD-20 default: no); admitting it is a later
decision this module does not make.

Every constructor argument named `D_w` matches section 2.1 Table 2's literal signature
(`FrontalReadout(D_w, mlp_ratio)`, etc.) rather than `ruff`'s usual lowercase-argument
convention; each such `__init__` carries a `# noqa: N803` for exactly that reason.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

__all__ = ["FrontalReadout", "NullCandidate", "RankHead", "UnifyProbes"]

_MAX_HEADS = 8


def _select_heads(width: int, want: int = _MAX_HEADS) -> int:
    """Pick the largest head count `<= want` that evenly divides `width`.

    `nn.MultiheadAttention` requires `embed_dim % num_heads == 0`, and the spec's own
    symbol table fixes `H = 8` for `D_w = 512` (`512 % 8 == 0`). This helper exists so
    the same class stays constructible in a unit test at a much smaller `D_w` that is
    not itself a multiple of 8, without ever affecting a real `D_w = 512` instance's
    head count and therefore without affecting the parameter counts of section 2.3's
    Table 4, which do not depend on head count at all.

    Args:
        width: The attention embedding dimension, `D_w`.
        want: The preferred head count; defaults to the spec's `H = 8`.

    Returns:
        A head count in `[1, want]` that evenly divides `width`.
    """
    for heads in range(min(want, width), 0, -1):
        if width % heads == 0:
            return heads
    return 1


class FrontalReadout(nn.Module):
    """DEC-20's frontal read-out: a learned-query attention pool plus its MLP.

    Spec section 2.1 Table 2 (`readout.py` row) and section 3 step 12: `f =
    FrontalReadout(z_N)` is "a learned-query attention pool over the latents plus its
    MLP." At `D_w = 512`, `mlp_ratio = 4` this is exactly the 3,150,848 parameters of
    section 2.3 Table 4 -- the query (512), the attention block
    (`4·(512² + 512) = 1,050,624`), and the MLP (`512·2048 + 2048 + 2048·512 + 512 =
    2,099,712`) -- with no LayerNorm counted or instantiated, matching Table 4's
    silence on one for this component.
    """

    def __init__(self, D_w: int, mlp_ratio: int = 4) -> None:  # noqa: N803
        """Build the learned query, the attention pool and the MLP.

        Args:
            D_w: Workspace width, `512` at v1 (spec section 1, "`D_w = 512`").
            mlp_ratio: MLP hidden-width multiplier; `4` at v1 (spec `InterconnectConfig`
                defaults, section 2.1).
        """
        super().__init__()
        self.D_w = D_w
        self.mlp_ratio = mlp_ratio
        self.query = nn.Parameter(torch.randn(1, 1, D_w) * 0.02)
        heads = _select_heads(D_w)
        self.attn = nn.MultiheadAttention(embed_dim=D_w, num_heads=heads, batch_first=True)
        hidden = D_w * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(D_w, hidden),
            nn.GELU(),
            nn.Linear(hidden, D_w),
        )

    def forward(self, z: Tensor, z_affect: Tensor | None = None) -> Tensor:
        """Pool the workspace latents into one read-out vector `f`.

        Spec section 3 step 12: the learned query cross-attends over `z` (shape `[B,
        L, D_w]`, `L = 64` at v1) to produce a single pooled vector, which the MLP then
        maps to `f`. `z_affect` is the section-3 "Affect seam": a declared but, at v1,
        strictly unused input -- see this module's docstring, ambiguity 4. It is
        shape-checked and then never read, so `f` does not depend on its value.

        Args:
            z: `[B, L, D_w]` float, the final workspace latents `z_N`.
            z_affect: Optional `[B, D_w]` tagged affect stream (DEC-79). When given,
                only its last-dimension width is checked against `D_w`; its values
                never reach the computation of `f`.

        Returns:
            `f`, `[B, D_w]` float.

        Raises:
            ValueError: `z_affect` is given and its last dimension is not `D_w`.
        """
        if z_affect is not None and z_affect.shape[-1] != self.D_w:
            raise ValueError(
                f"z_affect's last dimension must be D_w={self.D_w}, got "
                f"shape {tuple(z_affect.shape)} (spec section 3, 'Affect seam')."
            )
        batch = z.shape[0]
        query = self.query.expand(batch, -1, -1)
        pooled, _ = self.attn(query, z, z, need_weights=False)
        pooled = pooled.squeeze(1)
        return self.mlp(pooled)


class NullCandidate(nn.Module):
    """DEC-41's `NULL` candidate: the learned abstention embedding.

    Spec section 2.3 Table 4: "one learned vector of 512," 512 params, separate from
    `RankHead`'s own 262,656. Taxonomy: "the correct answer on a general-bin item is
    the `NULL` candidate, and it is scored by the identical metric" -- no confidence
    head, no rejection target, no extra machinery beyond one embedding that competes on
    equal footing with the 31 content candidates. Table 8 row G36 (the `NULL` recall /
    false-positive gate) fires over a receipt in `gates.py`, lane IC-10; this class is
    only the embedding G36 measures.
    """

    def __init__(self, D_w: int) -> None:  # noqa: N803
        """Allocate the one learned `NULL` embedding.

        Args:
            D_w: Workspace width, `512` at v1.
        """
        super().__init__()
        self.D_w = D_w
        self.embedding = nn.Parameter(torch.randn(D_w) * 0.02)

    def expand(self, batch: int) -> Tensor:
        """Broadcast the learned embedding to a batch, at index-0 shape.

        Args:
            batch: `B`, the batch size to broadcast to.

        Returns:
            `[B, 1, D_w]` float, the same learned vector repeated over `B`; this is a
            view (via `Tensor.expand`), so gradient from every batch item accumulates
            back onto the single shared parameter.
        """
        return self.embedding.view(1, 1, self.D_w).expand(batch, 1, -1)


class RankHead(nn.Module):
    """DEC-41's ranking head: cosine scoring of `f` against `k` candidates.

    Spec section 3 step 12 and section 4 ("`L_task` is softmax cross-entropy over the
    `k = 32` cosine scores at a recorded temperature"). See this module's docstring,
    ambiguities 1-3, for how the one shared `Linear(D_w, D_w)` and the `NULL` splice
    point are read from the spec's shape table and prose together.
    """

    def __init__(self, D_w: int, k: int, temperature: float) -> None:  # noqa: N803
        """Build the shared projection and the `NULL` candidate.

        Args:
            D_w: Workspace width, `512` at v1.
            k: Candidate-set size, `32` at v1 (1 `NULL` + 30 hard distractors + 1
                gold, per the taxonomy's DEC-41 candidate-set construction).
            temperature: The softmax temperature `L_task` divides the cosine scores
                by; required with no default because the spec calls it "a recorded
                temperature," a receipt field the caller must choose and log, not a
                constant this class should silently supply.

        Raises:
            ValueError: `k < 2` (there is no room for `NULL` plus at least one content
                candidate), or `temperature <= 0`.
        """
        super().__init__()
        if k < 2:
            raise ValueError(f"k must be at least 2 (NULL plus content candidates), got {k}.")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        self.D_w = D_w
        self.k = k
        self.temperature = temperature
        self.proj = nn.Linear(D_w, D_w)
        self.null = NullCandidate(D_w)

    def forward(self, f: Tensor, content_candidates: Tensor) -> Tensor:
        """Score `f` against the `NULL` candidate and `k - 1` content candidates.

        Prepends `self.null`'s embedding at index 0, projects `f` and every candidate
        through the same shared linear layer, and scores by cosine similarity divided
        by `temperature`.

        Args:
            f: `[B, D_w]` float, `FrontalReadout`'s output.
            content_candidates: `[B, k - 1, D_w]` float, the pre-embedded gold and hard
                distractor candidates (Q7's `CandidateEncoder` output, out of this
                lane's scope; any `[B, k - 1, D_w]` tensor is accepted).

        Returns:
            `[B, k]` float cosine-similarity scores divided by `temperature`, `NULL`
            at index 0.

        Raises:
            ValueError: `content_candidates` is not shaped `[B, k - 1, D_w]`.
        """
        batch = f.shape[0]
        expected = (batch, self.k - 1, self.D_w)
        if tuple(content_candidates.shape) != expected:
            raise ValueError(
                f"content_candidates must be shaped {expected} (B, k - 1, D_w), got "
                f"{tuple(content_candidates.shape)}."
            )
        null_embed = self.null.expand(batch)
        candidates = torch.cat([null_embed, content_candidates], dim=1)
        query = self.proj(f)
        keys = self.proj(candidates)
        cosine = functional.cosine_similarity(query.unsqueeze(1), keys, dim=-1)
        return cosine / self.temperature


class UnifyProbes(nn.Module):
    """DEC-20's `L_unify` probes: one frozen-target linear probe per admitted region.

    Spec section 4: `L_unify` is `Σ_r (1 − cos(probe_r(f), pool_r(h_r)))` over the
    admitted regions -- "the unified state must be linearly sufficient for each
    contributing region's own answer as well as for the joint one." Section 2.3 Table
    4 costs this at 590,976 params for the four `R_ctx = 4` encoding participants at
    v1: `3·(512·256 + 256) + (512·384 + 384)` for `language`, `memory`, `reasoning` at
    `pooled_dim = 256` and `visual` at `pooled_dim = 384`. `probe_r(f)` is this class's
    output for region `r`; `pool_r(h_r)` (the region's own `Faculty.pool()` output) and
    the sum over admitted regions are `UnifyLoss`'s job, lane IC-7's `losses.py`, not
    this class's -- `forward` here returns every configured region's probe output
    unconditionally, and the caller selects which are admitted for a given item.
    """

    def __init__(self, D_w: int, pooled_dims: dict[str, int]) -> None:  # noqa: N803
        """Allocate one `Linear(D_w, pooled_dim_r)` per region name.

        Args:
            D_w: Workspace width, `512` at v1.
            pooled_dims: Map from region name to that region's `pooled_dim` (Table 1),
                e.g. `{"language": 256, "memory": 256, "reasoning": 256, "visual":
                384}` at v1. `episodic_store` is not a probe target: Table 4's
                arithmetic sums exactly four terms, not five.

        Raises:
            ValueError: `pooled_dims` is empty.
        """
        super().__init__()
        if not pooled_dims:
            raise ValueError("pooled_dims must name at least one region.")
        self.D_w = D_w
        self.pooled_dims = dict(pooled_dims)
        self.probes = nn.ModuleDict(
            {name: nn.Linear(D_w, dim) for name, dim in pooled_dims.items()}
        )

    def forward(self, f: Tensor) -> dict[str, Tensor]:
        """Apply every region's probe to the same read-out vector `f`.

        Args:
            f: `[B, D_w]` float, `FrontalReadout`'s output.

        Returns:
            One entry per constructor-time region name, `{name: [B, pooled_dim_r]}`.
        """
        return {name: probe(f) for name, probe in self.probes.items()}
