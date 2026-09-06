"""Declarative region specs — what a region is, before one is built.

WHY THIS EXISTS
``default_two_region_mind(stream_dim, hidden_dim, latent_dim)`` was the entire
composition API, hardcoded to exactly two regions. There was no way to say that a region
is called ``code``, that its job is docstring-to-function, that it pretrains on
CodeSearchNet, or that it should be quantized to 8 bits after training. So the named
specialists in CSD-BRAIN-REGIONS.md could not be expressed at all, let alone trained
separately — which is step 1 of the mandatory curriculum.

A spec is data, not code, so a mind can be declared in JSON and rebuilt identically on
another host. That is what makes a training run reproducible and a checkpoint comparable.

TWO DESIGN POINTS WORTH STATING

**Modality is explicit, and ``latent`` is a first-class value.** The VL regions do not
consume tokens; they consume visual latents and reason in that space. The routing surface
``activate: [B, D] -> [B, D]`` is already modality-agnostic, so a vision region is a peer
of a text region rather than a special case bolted on the side — but only if the spec can
say which encoder feeds it. ``modality`` is that declaration.

**Quantization is declared, not applied.** The policy travels with the region so a
post-training pass can act on it, but nothing here quantizes anything. Training happens at
full precision; PTQ is a separate, later step. A region trained into low precision cannot
be compared against one that was not.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from cogsyndelta.regions.aliases import canonical_region

Modality = Literal["text", "latent", "vision", "any"]


@dataclass(frozen=True)
class PretrainSpec:
    """Where a region's job-shaped data comes from.

    The curriculum is explicit that a region which never sees its own data before
    assembly becomes dead weight, because the gate collapses onto whichever region moved
    first. So a spec that intends to be trained must say what it trains on.
    """

    corpus: str
    """Key into cogsyndelta.data.corpus.CORPORA, or a dataset id to be fetched."""

    objective: str
    """e.g. 'contrastive-pair', 'reconstruction', 'rank'. Names the loss family, not
    the implementation, so the spec survives a change of optimizer."""

    available: bool = False
    """Whether the corpus is actually on disk. Defaults False deliberately: several
    corpora referenced in the docs exist only as empty stub directories, and treating
    'named' as 'present' is how a training plan turns out to be fiction."""

    paths: tuple[str, ...] = ()
    """Glob patterns, relative to the corpus root, naming the shards `available` claims
    are on disk. Optional (older entries carry the claim only in `notes`) -- when
    present, a reader (or a test) can check the claim instead of trusting the prose."""

    notes: str = ""


@dataclass(frozen=True)
class QuantSpec:
    """Post-training quantization policy for one region. Declarative only."""

    bits: int = 8
    method: str = "calibrated-uniform"
    skip: bool = False
    """Set when a region is known to be quantization-sensitive and should stay at full
    precision while its neighbours are compressed."""


@dataclass(frozen=True)
class RegionSpec:
    """Everything needed to build, train, route to, and later quantize one region."""

    name: str
    kind: str
    """Implementation selector, e.g. 'residual_mlp' or 'latent_vae'."""

    stream_dim: int
    hidden_dim: int
    latent_dim: int | None = None

    modality: Modality = "any"
    role: str = ""
    router_trigger: str = ""
    """Human-readable condition under which the gate should prefer this region. Prose
    for now on purpose -- the routing signal is learned, and writing a machine-readable
    rule here would imply a hand-written gate that does not exist."""

    pretrain: PretrainSpec | None = None
    quantization: QuantSpec | None = None

    live: bool = False
    """False means 'specified but not implemented'. Keeping unbuilt regions in the same
    catalogue as built ones is what stops the docs and the code drifting apart -- the
    README's nine-region brain lineup exists because intent had nowhere honest to live."""

    merged_into: str | None = None
    """DEC-02-style merge: names the region this one's training regime was folded into
    (e.g. `compress`/`retrieve` both carry `merged_into: "memory"` once W4 lands). The
    OLD entry is kept in the catalogue, readable, rather than deleted -- every receipt
    written before the merge still names a region this file has to be able to describe --
    but a merged region can never itself be `live`, since its objective is no longer
    trained on its own (see `__post_init__`)."""

    specialisation: str | None = None
    """Domain flavour of an otherwise domain-agnostic faculty -- e.g. `language`'s
    `specialisation: "code"` records that its CURRENT corpus and battery are
    code-flavoured, without the region itself being named after that domain (operator
    naming rule, 2026-09-04: regions are named by cognitive faculty, never by knowledge
    domain). A domain that is not the faculty's whole identity belongs here, or as a
    memory-gate persona/variant branch -- never as the region `name`."""

    region_alias_of: str | None = None
    """Set by ``MindSpec.from_dict`` when this spec was loaded from a legacy region name
    (``cogsyndelta.regions.aliases.REGION_ALIASES``) -- e.g. a spec built from a config
    that still says ``name: "code"`` carries ``name="language"``,
    ``region_alias_of="code"``. ``None`` for a spec that was already canonical, or one
    built directly (not through the loader) -- this field records how THIS spec was
    resolved, not a general fact about the region."""

    def __post_init__(self) -> None:
        """Reject a spec that cannot build: empty name, non-positive dims, or a
        latent_vae with no latent_dim.
        """
        if not self.name:
            raise ValueError("region name must be non-empty")
        if self.stream_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError(f"{self.name}: stream_dim and hidden_dim must be positive")
        if self.kind == "latent_vae" and self.latent_dim is None:
            raise ValueError(f"{self.name}: kind 'latent_vae' requires latent_dim")
        if self.merged_into and self.live:
            raise ValueError(
                f"{self.name}: merged into {self.merged_into!r} and 'live' at the same "
                f"time -- a merged region's objective is trained through the region it "
                f"was merged into, not on its own; a receipt from before the merge stays "
                f"readable, but this entry cannot claim to be a live implementation too"
            )


@dataclass(frozen=True)
class MindSpec:
    """A full mind: the shared stream width plus its regions."""

    stream_dim: int
    regions: list[RegionSpec] = field(default_factory=list)
    top_k: int = 1
    aux_coef: float = 1.0
    notes: str = ""

    def __post_init__(self) -> None:
        """Reject duplicate region names and out-of-range top_k -- shape errors that
        would otherwise surface much later. Native-width disagreement is warned about,
        not rejected (DEC-14/DEC-15, docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md
        section 2.2): regions keep their own native widths (e.g. text 256, visual 384)
        and adapt into the shared workspace ``stream_dim`` via a per-region
        ``nn.Linear(token_dim, stream_dim)`` adapter, rather than all sharing one global
        width. The old hard uniformity check "enforces a uniformity that has never been
        true and would reject the real trained regions ... against the declared
        catalogue" (taxonomy ~L1194), so a mismatch is expected, not a shape error.
        """
        names = [r.name for r in self.regions]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate region names: {sorted(dupes)}")
        mismatched = [r.name for r in self.regions if r.stream_dim != self.stream_dim]
        if mismatched:
            warnings.warn(
                f"regions {mismatched} use a native stream_dim that differs from mind "
                f"stream_dim {self.stream_dim}; each adapts in via its own per-region "
                "adapter (DEC-14/DEC-15, docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md "
                "section 2.2) rather than sharing one global width",
                UserWarning,
                stacklevel=2,
            )
        if self.top_k < 1 or self.top_k > max(len(self.regions), 1):
            raise ValueError(f"top_k {self.top_k} out of range for {len(self.regions)} regions")

    @property
    def live_regions(self) -> list[RegionSpec]:
        """Regions marked implemented. Declared-but-unbuilt regions are excluded."""
        return [r for r in self.regions if r.live]

    def to_json(self, path: Path | None = None, *, indent: int = 2) -> str:
        """Serialize the mind. Writes to ``path`` when given; always returns the JSON."""
        payload = json.dumps(asdict(self), indent=indent, sort_keys=False)
        if path is not None:
            path.write_text(payload + "\n")
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MindSpec:
        """Rebuild a MindSpec from parsed JSON, restoring nested pretrain/quant specs.

        Resolves a legacy region name (``cogsyndelta.regions.aliases.REGION_ALIASES``,
        e.g. a config that still says ``"code"`` or ``"vl_latent"``) to its canonical
        spelling, warns once per entry, and records where it came from in
        ``RegionSpec.region_alias_of`` -- so an old config keeps loading, but nothing
        downstream ever sees the legacy name as this spec's ``name``.
        """
        regions = []
        for raw in data.get("regions", []):
            raw = dict(raw)
            pre = raw.pop("pretrain", None)
            if pre is not None and "paths" in pre:
                # JSON has no tuple; PretrainSpec is frozen (and therefore hashable),
                # so a list here would make every instance carrying one unhashable.
                pre = {**pre, "paths": tuple(pre["paths"])}
            quant = raw.pop("quantization", None)
            raw_name = raw["name"]
            canonical = canonical_region(raw_name)
            if canonical != raw_name:
                warnings.warn(
                    f"region name {raw_name!r} is deprecated; use {canonical!r} "
                    "(cogsyndelta.regions.aliases.REGION_ALIASES)",
                    DeprecationWarning,
                    stacklevel=2,
                )
                raw["name"] = canonical
                raw.setdefault("region_alias_of", raw_name)
            regions.append(
                RegionSpec(
                    **raw,
                    pretrain=PretrainSpec(**pre) if pre else None,
                    quantization=QuantSpec(**quant) if quant else None,
                )
            )
        return cls(
            stream_dim=data["stream_dim"],
            regions=regions,
            top_k=data.get("top_k", 1),
            aux_coef=data.get("aux_coef", 1.0),
            notes=data.get("notes", ""),
        )

    @classmethod
    def from_json(cls, path: Path) -> MindSpec:
        """Load a MindSpec from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))
