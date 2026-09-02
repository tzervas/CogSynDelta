"""Region spec tests.

The point of a spec layer is that a mind can be declared as data and rebuilt identically
elsewhere. These assert the invariants that make that true, and the guards that stop a
spec from quietly describing something other than what gets built.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cogsyndelta.contracts.region_spec import (
    MindSpec,
    PretrainSpec,
    QuantSpec,
    RegionSpec,
)
from cogsyndelta.poc.route import build_mind_from_spec

CATALOGUE = Path(__file__).resolve().parents[1] / "config" / "mind" / "csd-regions.json"


def _region(name: str, **kw: object) -> RegionSpec:
    base = {"kind": "residual_mlp", "stream_dim": 64, "hidden_dim": 128}
    base.update(kw)
    return RegionSpec(name=name, **base)  # type: ignore[arg-type]


def test_duplicate_region_names_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        MindSpec(stream_dim=64, regions=[_region("a"), _region("a")])


def test_stream_width_mismatch_rejected() -> None:
    """Every region reads and writes the same stream; a width mismatch is a shape error
    at the first activate() and is far cheaper to catch at declaration."""
    with pytest.raises(ValueError, match="stream_dim"):
        MindSpec(stream_dim=64, regions=[_region("a", stream_dim=128)])


def test_latent_vae_requires_latent_dim() -> None:
    with pytest.raises(ValueError, match="latent_dim"):
        _region("v", kind="latent_vae")


def test_top_k_bounded_by_region_count() -> None:
    with pytest.raises(ValueError, match="top_k"):
        MindSpec(stream_dim=64, regions=[_region("a")], top_k=2)


def test_round_trips_through_json(tmp_path: Path) -> None:
    """A mind must rebuild identically from its serialized form, or a checkpoint cannot
    be tied to the architecture that produced it."""
    spec = MindSpec(
        stream_dim=64,
        regions=[
            _region("residual_mlp", live=True),
            _region(
                "stream_vae",
                kind="latent_vae",
                latent_dim=16,
                live=True,
                pretrain=PretrainSpec(
                    corpus="synthetic", objective="reconstruction", available=True
                ),
                quantization=QuantSpec(bits=4),
            ),
        ],
    )
    path = tmp_path / "mind.json"
    spec.to_json(path)
    assert MindSpec.from_json(path) == spec


def test_builds_only_live_regions() -> None:
    spec = MindSpec(
        stream_dim=64,
        regions=[
            _region("residual_mlp", live=True),
            _region("code", kind="contrastive_encoder", live=False),
        ],
    )
    assert build_mind_from_spec(spec).names() == ("residual_mlp",)


def test_live_region_with_no_implementation_raises() -> None:
    """A mind that quietly comes up with fewer regions than declared would train,
    converge, and be wrong in a way no assertion catches."""
    spec = MindSpec(
        stream_dim=64,
        regions=[_region("code", kind="contrastive_encoder", live=True)],
    )
    with pytest.raises(NotImplementedError, match=r"no .*implementation"):
        build_mind_from_spec(spec)


def test_shipped_catalogue_is_valid_and_honest() -> None:
    """The catalogue must parse, and must not claim capability it does not have.

    This is the guard against the failure the README already exhibits: a nine-region
    brain architecture documented as real while no optimizer can see its parameters.
    """
    spec = MindSpec.from_json(CATALOGUE)
    assert spec.stream_dim > 0
    assert len(spec.regions) >= 5

    live = {r.name for r in spec.live_regions}
    assert live == {"residual_mlp", "stream_vae"}, (
        f"only the two PoC regions are implemented; catalogue claims {live}"
    )

    # Anything marked live must actually build.
    build_mind_from_spec(spec)

    # A region cannot be live while declaring its data is missing.
    for region in spec.live_regions:
        if region.pretrain is not None:
            assert region.pretrain.available, (
                f"{region.name} is live but its pretrain corpus is marked unavailable"
            )


def test_catalogue_records_vl_as_latent_not_text() -> None:
    """VL regions consume visual latents, not tokens. If this flips to 'text' someone has
    misunderstood the architecture -- the whole point of the [B, D] activate surface is
    that a vision region is a peer of a text region, fed by a different encoder."""
    spec = MindSpec.from_json(CATALOGUE)
    vl = [r for r in spec.regions if r.name == "vl_latent"]
    assert vl, "expected a vl_latent region in the catalogue"
    assert vl[0].modality == "latent"
    assert not vl[0].live


def test_catalogue_json_is_sorted_stable() -> None:
    """Serializing the parsed catalogue must not reorder or drop fields."""
    raw = json.loads(CATALOGUE.read_text())
    spec = MindSpec.from_dict(raw)
    assert [r["name"] for r in raw["regions"]] == [r.name for r in spec.regions]
