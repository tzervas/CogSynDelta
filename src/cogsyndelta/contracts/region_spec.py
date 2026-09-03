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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

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

    def __post_init__(self) -> None:
        """Reject a spec that cannot build: empty name, non-positive dims, or a
        latent_vae with no latent_dim."""
        if not self.name:
            raise ValueError("region name must be non-empty")
        if self.stream_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError(f"{self.name}: stream_dim and hidden_dim must be positive")
        if self.kind == "latent_vae" and self.latent_dim is None:
            raise ValueError(f"{self.name}: kind 'latent_vae' requires latent_dim")


@dataclass(frozen=True)
class MindSpec:
    """A full mind: the shared stream width plus its regions."""

    stream_dim: int
    regions: list[RegionSpec] = field(default_factory=list)
    top_k: int = 1
    aux_coef: float = 1.0
    notes: str = ""

    def __post_init__(self) -> None:
        """Reject duplicate region names, stream-width disagreement, and out-of-range
        top_k -- all of which are shape errors that would otherwise surface much later."""
        names = [r.name for r in self.regions]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate region names: {sorted(dupes)}")
        mismatched = [r.name for r in self.regions if r.stream_dim != self.stream_dim]
        if mismatched:
            # Every region reads and writes the same shared stream; a width mismatch is a
            # shape error at the first activate() and is far cheaper to catch here.
            raise ValueError(
                f"regions {mismatched} disagree with mind stream_dim {self.stream_dim}"
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
        """Rebuild a MindSpec from parsed JSON, restoring nested pretrain/quant specs."""
        regions = []
        for raw in data.get("regions", []):
            raw = dict(raw)
            pre = raw.pop("pretrain", None)
            quant = raw.pop("quantization", None)
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
