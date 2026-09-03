"""Component protocols — the roles a specialist can play around the shared stream.

WHY THIS EXISTS
``CognitiveRegion`` describes exactly one shape: ``[B, D] -> [B, D]``. That is the right
surface for a region that transforms the shared stream, and the wrong one for most of what
a composed mind needs. A classifier emits logits, not a stream. A ranker consumes a query
and a candidate set. Encoders and decoders cross the modality boundary in opposite
directions. Forcing all of those through ``activate()`` would either lie about their
signature or push the real work into side channels.

THE ORGANISING IDEA
The shared ``[B, D]`` latent stream is the **interchange format**. Every component is
classified by which side of it the component touches:

    Encoder     modality        ->  [B, D]          text tokens, image patches, audio
    Region      [B, D]          ->  [B, D]          the existing CognitiveRegion
    Decoder     [B, D]          ->  modality        stream back out to tokens or pixels
    Classifier  [B, D]          ->  [B, C]          a head; typically a few thousand params
    Ranker      query+candidates->  [B, N] scores   retrieval and reranking

That is what makes "a cohort of specialists acting as one mind" implementable rather than
aspirational: components compose because they agree on the stream, not because they share
a base class. A vision encoder and a text encoder are peers because both produce ``[B, D]``,
and neither needs to know the other exists.

WHY PROTOCOLS RATHER THAN BASE CLASSES
Structural typing means a component satisfies a role by having the right method, with no
inheritance and no registry coupling. A model can hold more than one role -- a LatentVAE is
legitimately both an Encoder and a Decoder -- which a single-inheritance hierarchy makes
awkward and a Protocol makes free. This also matches the existing ``CognitiveRegion``,
which is already a ``runtime_checkable`` Protocol; these are siblings, not a replacement.

ON SIZE
These are deliberately small. The project's thesis is capability per parameter, so a
classifier head or a reranker should cost thousands of parameters, not millions. A utility
model that grows to rival a region has usually absorbed work that belonged in the region.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class StreamEncoder(Protocol):
    """Map some modality into the shared latent stream.

    Named StreamEncoder rather than Encoder to keep it unambiguous next to the many
    ``*Encoder`` nn.Modules already in this codebase, which are implementations rather
    than roles.
    """

    name: str

    def encode(self, inputs: Any) -> torch.Tensor:
        """Return ``[B, D]`` on the shared stream.

        Args:
            inputs: Modality-specific. Token ids for text, ``[B, C, H, W]`` for vision.

        Returns:
            ``[B, D]`` latent.
        """
        ...


@runtime_checkable
class StreamDecoder(Protocol):
    """Map the shared latent stream back out to a modality."""

    name: str

    def decode(self, stream: torch.Tensor) -> Any:
        """Return a modality-specific reconstruction or generation from ``[B, D]``."""
        ...


@runtime_checkable
class Classifier(Protocol):
    """Map the shared stream to class logits.

    Returns logits rather than probabilities so the caller chooses the loss. A component
    that softmaxes internally cannot be used with cross-entropy without silently applying
    it twice -- a mistake that trains, converges, and degrades quality invisibly.
    """

    name: str
    n_classes: int

    def classify(self, stream: torch.Tensor) -> torch.Tensor:
        """Return ``[B, n_classes]`` logits from ``[B, D]``."""
        ...


@runtime_checkable
class Ranker(Protocol):
    """Score candidates against a query, both already on the shared stream.

    Covers rerankers too: a reranker is a Ranker applied to a shortlist, optionally given
    the first-stage scores. It is the same signature, not a separate role, so a component
    can serve either stage without being rewritten.
    """

    name: str

    def rank(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        prior_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score ``candidates`` against ``query``.

        Args:
            query: ``[B, D]``.
            candidates: ``[B, N, D]``.
            prior_scores: Optional ``[B, N]`` from an earlier stage. A reranker may use
                these; a first-stage ranker ignores them.

        Returns:
            ``[B, N]`` scores, higher is better. Unnormalised -- callers that need
            probabilities apply their own softmax, for the same reason Classifier
            returns logits.
        """
        ...


#: Role name -> protocol. Used by specs to validate that a declared component actually
#: implements the role it claims, rather than trusting a string in a config file.
ROLES: dict[str, type] = {
    "encoder": StreamEncoder,
    "region": None,  # filled below to avoid a circular import at module load
    "decoder": StreamDecoder,
    "classifier": Classifier,
    "ranker": Ranker,
}


def _resolve_region_role() -> None:
    """Late-bind CognitiveRegion so contracts.region and contracts.components can import
    each other's names without a cycle."""
    from cogsyndelta.contracts.region import CognitiveRegion

    ROLES["region"] = CognitiveRegion


_resolve_region_role()


def implements(component: object, role: str) -> bool:
    """Return whether ``component`` satisfies ``role``.

    Args:
        component: Any object.
        role: Key of :data:`ROLES`.

    Returns:
        True when the object structurally implements the role's protocol.

    Raises:
        KeyError: If the role is unknown. Failing loudly matters here -- a typo in a spec
            silently matching nothing would let an unimplemented component look valid.
    """
    if role not in ROLES:
        raise KeyError(f"unknown role {role!r}; have {sorted(ROLES)}")
    return isinstance(component, ROLES[role])
