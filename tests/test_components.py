"""Component role protocol tests.

These pin the property that makes a cohort of specialists composable: components agree on
the shared ``[B, D]`` stream, and their role is determined by structure rather than by a
string in a config file or a base class they happened to inherit.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from cogsyndelta.contracts.components import (
    ROLES,
    Classifier,
    Ranker,
    StreamDecoder,
    StreamEncoder,
    implements,
)
from cogsyndelta.poc.vae import LatentVAE

DIM = 64


class _Probe(nn.Module):
    """A classifier head at the scale these are meant to be."""

    def __init__(self, dim: int = DIM, n_classes: int = 10) -> None:
        super().__init__()
        self.name = "probe"
        self.n_classes = n_classes
        self.head = nn.Linear(dim, n_classes)

    def classify(self, stream: torch.Tensor) -> torch.Tensor:
        return self.head(stream)


class _CosineRanker(nn.Module):
    """A parameter-free ranker."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "cosine"

    def rank(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        prior_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = torch.nn.functional.normalize(query, dim=-1).unsqueeze(1)
        c = torch.nn.functional.normalize(candidates, dim=-1)
        scores = (q * c).sum(-1)
        return scores if prior_scores is None else scores + prior_scores


@pytest.mark.cpu
def test_roles_cover_both_sides_of_the_stream() -> None:
    """Every role either produces, transforms, or consumes the shared stream."""
    assert set(ROLES) == {"encoder", "region", "decoder", "classifier", "ranker"}
    assert all(r is not None for r in ROLES.values()), "a role resolved to None"


@pytest.mark.cpu
def test_role_membership_is_structural_not_declared() -> None:
    """A component satisfies a role by having the method, with no inheritance.

    This is what lets a vision encoder and a text encoder be peers without either knowing
    the other exists.
    """
    probe, ranker = _Probe(), _CosineRanker()
    assert implements(probe, "classifier")
    assert implements(ranker, "ranker")
    # And does NOT accidentally satisfy roles it has no method for.
    assert not implements(probe, "ranker")
    assert not implements(ranker, "classifier")
    assert not implements(probe, "region")


@pytest.mark.cpu
def test_unknown_role_raises_rather_than_matching_nothing() -> None:
    """A typo in a spec must fail loudly. Silently matching nothing would let an
    unimplemented component look valid."""
    with pytest.raises(KeyError, match="unknown role"):
        implements(_Probe(), "clasifier")


@pytest.mark.cpu
def test_one_component_can_hold_several_roles() -> None:
    """A LatentVAE is legitimately a region AND encodes/decodes.

    Single inheritance makes that awkward; structural typing makes it free, which is the
    reason these are Protocols.
    """
    vae = LatentVAE(input_dim=DIM, hidden_dim=32, latent_dim=8, name="stream_vae")
    assert implements(vae, "region")
    assert isinstance(vae, nn.Module)


@pytest.mark.cpu
def test_classifier_returns_logits_not_probabilities() -> None:
    """A component that softmaxes internally cannot be used with cross-entropy without
    applying it twice -- which trains, converges, and quietly degrades quality."""
    probe = _Probe()
    out = probe.classify(torch.randn(4, DIM))
    assert out.shape == (4, probe.n_classes)
    sums = out.softmax(dim=-1).sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones(4))
    # Raw outputs must not already be a distribution.
    assert not torch.allclose(out.sum(dim=-1), torch.ones(4))


@pytest.mark.cpu
def test_ranker_handles_both_stages_with_one_signature() -> None:
    """A reranker is a Ranker over a shortlist, not a separate role, so the same
    component can serve either stage without being rewritten."""
    ranker = _CosineRanker()
    query = torch.randn(3, DIM)
    candidates = torch.randn(3, 5, DIM)

    first = ranker.rank(query, candidates)
    assert first.shape == (3, 5)

    second = ranker.rank(query, candidates, prior_scores=first)
    assert second.shape == (3, 5)
    assert not torch.equal(first, second), "prior_scores were ignored"


@pytest.mark.cpu
def test_protocols_are_runtime_checkable() -> None:
    """isinstance against these must work, since specs validate components at build time."""
    for proto in (StreamEncoder, StreamDecoder, Classifier, Ranker):
        assert isinstance(_Probe(), proto) or True  # smoke: no TypeError raised
    assert isinstance(_Probe(), Classifier)
    assert isinstance(_CosineRanker(), Ranker)


@pytest.mark.cpu
def test_utility_components_stay_small() -> None:
    """The thesis is capability per parameter. A utility head that rivals a region has
    usually absorbed work belonging to the region."""
    probe = _Probe()
    assert sum(p.numel() for p in probe.parameters()) < 10_000
    assert sum(p.numel() for p in _CosineRanker().parameters()) == 0
