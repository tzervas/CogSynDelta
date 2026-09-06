"""Region spec tests.

The point of a spec layer is that a mind can be declared as data and rebuilt identically
elsewhere. These assert the invariants that make that true, and the guards that stop a
spec from quietly describing something other than what gets built.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from cogsyndelta.contracts.region_spec import (
    MindSpec,
    PretrainSpec,
    QuantSpec,
    RegionSpec,
)
from cogsyndelta.poc.route import build_mind_from_spec
from cogsyndelta.regions.aliases import canonical_region

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


def test_merged_region_cannot_also_be_live() -> None:
    """DEC-02 (row W4): `compress`/`retrieve` are marked `merged_into: "memory"` and
    stay `live: False` -- a region cannot claim to be both a live implementation and
    folded into another one's training, or a reader cannot tell which receipt a `live`
    flag is even describing."""
    with pytest.raises(ValueError, match="merged"):
        _region("compress", merged_into="memory", live=True)


def test_merged_region_not_live_is_fine() -> None:
    spec = _region("compress", merged_into="memory", live=False)
    assert spec.merged_into == "memory"


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


def test_catalogue_records_the_dec02_merge() -> None:
    """`compress` and `retrieve` stay in the catalogue (old receipts still name them) but
    both point at `memory` as the region their training regime was folded into, and
    `memory` itself exists, is not live (not yet built), and is not marked merged into
    anything else."""
    spec = MindSpec.from_json(CATALOGUE)
    by_name = {r.name: r for r in spec.regions}
    assert by_name["compress"].merged_into == "memory"
    assert by_name["retrieve"].merged_into == "memory"
    assert not by_name["compress"].live
    assert not by_name["retrieve"].live
    assert "memory" in by_name
    assert by_name["memory"].merged_into is None
    assert not by_name["memory"].live


def test_catalogue_records_memory_corpus_as_available_and_on_disk() -> None:
    """`memory.pretrain.available` used to say `false` / "NOT on disk" even though the
    corpus has been on the fleet mount and in production receipts since W4 (2026-09-03)
    -- stale documentation, never read by any training/eval/publish code path
    (`available` is `PretrainSpec` prose, not a gate). Fixed 2026-09-06.

    This pins two things: the static claim (`available` is true, `paths` names the same
    four globs `REGIONS["memory"]` in `scripts/csd-train-all.py` trains on) always, and,
    when the fleet's NFS export is actually mounted, that those exact globs reproduce
    the corpus fingerprint the committed split manifest
    (`config/mind/splits/memory-6fc0cf23-split0.json`) was drawn from -- not merely that
    files exist at those paths, but that they are the SAME files DEC-02's union names.
    """
    import os

    from cogsyndelta.corpus import fingerprint_corpus

    spec = MindSpec.from_json(CATALOGUE)
    memory = {r.name: r for r in spec.regions}["memory"]
    assert memory.pretrain is not None
    assert memory.pretrain.available is True
    assert memory.pretrain.paths == (
        "region/retrieve/fiqa-pairs/train.parquet",
        "region/compress/all-nli/pair/train*.parquet",
        "region/retrieve/natural-questions/**/train*.parquet",
        "region/retrieve/gooaq/**/train*.parquet",
    )

    root = Path(os.environ.get("CSD_MEMORY_ROOT", "/mnt/fleet-datasets/csd"))
    if not root.is_dir():
        pytest.skip("fleet NFS export not mounted here; static claim above still checked")

    def resolve(pattern: str) -> list[str]:
        return sorted(str(p) for p in root.glob(pattern))

    primary_pattern, *extra_patterns = memory.pretrain.paths
    extra_columns = [("anchor", "positive"), ("query", "answer"), ("question", "answer")]
    extra_caps = [0, 0, 400_000]
    extra_sources = [
        {"shards": resolve(pat), "columns": list(cols), "limit": cap}
        for pat, cols, cap in zip(extra_patterns, extra_columns, extra_caps, strict=True)
    ]
    fp = fingerprint_corpus(
        resolve(primary_pattern), columns=("query", "passage"), extra_sources=extra_sources
    )
    assert fp == "6fc0cf23ff8591ff2241278f82c001d2", (
        f"memory's declared paths now fingerprint to {fp}, not the value the committed "
        "split manifest was drawn from -- either the corpus drifted or the paths above "
        "no longer match REGIONS['memory']"
    )


def test_catalogue_records_vl_as_vision_not_text() -> None:
    """The visual faculty consumes RGB images through an I-JEPA EMA target encoder and
    emits [B, D] stream latents -- not tokens. `modality` is `vision` (catalogue
    vocab), never `text`. `kind` is `i-jepa` (the deployed module), not the training
    predictor.

    Looks the region up by canonical id (`visual`, renamed from `vl_latent` per the
    2026-09-04 naming rule) via the alias module rather than a literal name, so this
    keeps passing whether the catalogue entry it finds is spelled either way."""
    spec = MindSpec.from_json(CATALOGUE)
    vl = [r for r in spec.regions if canonical_region(r.name) == "visual"]
    assert vl, "expected a visual (nee vl_latent) region in the catalogue"
    assert vl[0].modality == "vision"
    assert vl[0].kind == "i-jepa"
    assert "latent" not in vl[0].router_trigger.lower()
    assert "RGB" in vl[0].router_trigger or "image" in vl[0].router_trigger.lower()
    assert not vl[0].live


def test_catalogue_json_is_sorted_stable() -> None:
    """Serializing the parsed catalogue must not reorder or drop fields."""
    raw = json.loads(CATALOGUE.read_text())
    spec = MindSpec.from_dict(raw)
    assert [r["name"] for r in raw["regions"]] == [r.name for r in spec.regions]


def test_shipped_catalogue_names_the_language_centre_with_its_code_specialisation() -> None:
    """The region formerly called `code` is the LANGUAGE CENTRE (operator naming rule,
    2026-09-04): id `language`, with `specialisation: "code"` recorded as metadata for
    its current (code-flavoured) corpus and battery -- never the region name itself."""
    spec = MindSpec.from_json(CATALOGUE)
    by_name = {r.name: r for r in spec.regions}
    assert "language" in by_name
    assert "code" not in by_name, "the shipped catalogue must use the canonical name"
    assert by_name["language"].specialisation == "code"
    assert by_name["language"].region_alias_of is None, (
        "the shipped catalogue already spells this canonically -- loading it is not an "
        "alias resolution"
    )


def test_legacy_region_name_loads_through_the_alias_with_a_deprecation_warning() -> None:
    """A config that still says `name: "code"` (an old copy, or a receipt-adjacent
    config nobody has migrated yet) must keep loading -- through the SAME alias module
    every other reader uses -- rather than raising, but it must say so."""
    raw = json.loads(CATALOGUE.read_text())
    data = {
        "stream_dim": raw["stream_dim"],
        "regions": [dict(r) for r in raw["regions"] if r["name"] == "language"],
    }
    data["regions"][0]["name"] = "code"
    with pytest.warns(DeprecationWarning, match="code.*language"):
        spec = MindSpec.from_dict(data)
    region = spec.regions[0]
    assert region.name == "language", "the loaded spec must carry the CANONICAL name"
    assert region.region_alias_of == "code"


def test_legacy_vl_latent_name_loads_through_the_alias_too() -> None:
    raw = json.loads(CATALOGUE.read_text())
    data = {
        "stream_dim": raw["stream_dim"],
        "regions": [dict(r) for r in raw["regions"] if r["name"] == "visual"],
    }
    data["regions"][0]["name"] = "vl_latent"
    with pytest.warns(DeprecationWarning, match="vl_latent.*visual"):
        spec = MindSpec.from_dict(data)
    region = spec.regions[0]
    assert region.name == "visual"
    assert region.region_alias_of == "vl_latent"


def test_canonical_region_name_loads_with_no_warning_and_no_alias_of() -> None:
    """Loading a spec that already names a region canonically is not a deprecated path
    -- it must not warn, and `region_alias_of` stays `None` (nothing to record)."""
    raw = json.loads(CATALOGUE.read_text())
    data = {
        "stream_dim": raw["stream_dim"],
        "regions": [dict(r) for r in raw["regions"] if r["name"] == "language"],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        spec = MindSpec.from_dict(data)
    assert spec.regions[0].name == "language"
    assert spec.regions[0].region_alias_of is None


def test_a_canonical_spec_written_and_reread_stays_canonical() -> None:
    """Round-tripping a spec built directly with the canonical name (not through a
    legacy config) must not spuriously attach a region_alias_of -- the writer emitted
    the canonical id, so there is nothing to alias."""
    spec = MindSpec(
        stream_dim=64,
        regions=[_region("language", kind="contrastive_encoder", specialisation="code")],
    )
    assert canonical_region(spec.regions[0].name) == "language"
    assert spec.regions[0].region_alias_of is None
