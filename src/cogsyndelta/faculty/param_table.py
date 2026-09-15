"""Per-region parameter table: embedding vs token path vs pooling head.

WHAT THIS IS NOT
Row W0 (§4.1) asks for TWO re-instantiations and this module builds only one of them.
Its own text: *"re-instantiate §2.3's parameter table at the design's actual
configuration (4 controller heads, v1 participant list) so the total stops being
`[I]`"* -- that is the WHITE-MATTER INTERCONNECT table (workspace blocks, frontal
read-out, thalamic controller, conditioning prefixes, region adapters, the
`27,424,039`-param total). Building it needs DEC-16's module (workspace, controller),
which does not exist: *"The interconnect does not exist. Everything in §2 ... is design
only"* (`docs/technical/README.md` fact 2), and this lane's own mandate excludes
building it ("no controller, no workspace, no episodic store"). `27,424,039` and
`86,331,303` (composed) are UNCHANGED by this module and are not recomputed here.

WHAT THIS IS
A smaller, different, and actually buildable table: for each region that has a real
trained checkpoint, how many of its OWN parameters are embedding (the vocabulary /
patch-projection table), token path (the transformer blocks that produce `tokens()`),
or pooling head (`proj`, applied only inside `pool()`). This is the per-region
breakdown the §2.3 "Calibration" paragraph gives for one text region by hand --
*"a text region is 16,021,248 params of which 12,865,792 (80.30%) is the token
embedding table and only 3,155,456 (19.70%) is compute"* -- generalised into a function
and run against the five regions that actually have production checkpoints
(`code`/`compress`/`retrieve`/`reason`/`visual`, under `/akula-data/csd/matrix/`; the
merged `memory` region has no matrix checkpoint yet, only the 50-step smoke run
`regions/memory.py`'s own module docstring records, so it is not one of the five).

WHY "TOKEN PATH", NOT "COMPUTE"
Same quantity the calibration paragraph calls "compute": everything besides the
embedding/patch table and the pooling head -- the transformer blocks plus the final
norm, i.e. what actually produces `tokens()`'s per-position output. "Token path" names
it against this module's own vocabulary (`Faculty.token_dim`) rather than the vaguer
"compute", which could be misread to include the embedding lookup too.

CHECKPOINTS ARE LOADED READ-ONLY, THROUGH THE ONE SANCTIONED LOADER
`cogsyndelta.regions._checkpoint.load_checkpoint` (DEC-40/W0c) -- `weights_only=True`
hardcoded, no `torch.save` anywhere in this module. A checkpoint this module cannot
load (missing on this host, or a real load failure) is reported as a table row saying
so, per the row's own "if a checkpoint cannot be loaded on CPU say so" -- it is not
raised past `build_matrix_param_table`, so one missing/broken cell does not blank the
whole table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from cogsyndelta.model.vl_jepa import IJEPA, JEPAConfig, ViTEncoder
from cogsyndelta.regions._checkpoint import load_checkpoint
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig

__all__ = [
    "CANONICAL_CHECKPOINTS",
    "RegionCheckpoint",
    "RegionParamCount",
    "RegionTableRow",
    "build_matrix_param_table",
    "count_text_encoder_params",
    "count_visual_encoder_params",
    "faculty_param_count",
    "format_param_table",
    "load_region_module",
]

MATRIX_ROOT = Path(os.environ.get("CSD_MATRIX_ROOT", "/akula-data/csd/matrix"))
"""Read-only root for the training-matrix cells (`docs/design/...`'s W-row checkpoints
live under here). Overridable, matching `regions/memory.py`'s `MEMORY_ROOT` convention,
so a test can point it at a fixture tree instead."""


@dataclass(frozen=True)
class RegionCheckpoint:
    """One region's canonical matrix checkpoint, for the re-instantiation table."""

    region: str
    """Canonical region name (`cogsyndelta.regions.aliases.canonical_region`)."""

    legacy: str
    """On-disk name the matrix cell directory and checkpoint were written under --
    identical to `region` when the region has never been renamed."""

    checkpoint: Path
    """Path to a `final.pt` (text) or `step-<n>.pt` (visual) checkpoint."""

    kind: str
    """`"text"` (loads as `TextEncoder`) or `"visual"` (loads as `IJEPA`)."""


CANONICAL_CHECKPOINTS: tuple[RegionCheckpoint, ...] = (
    RegionCheckpoint(
        region="language",
        legacy="code",
        checkpoint=MATRIX_ROOT
        / "code-b512-s0-7bc2699-20260904/receipts/code-checkpoints/e46d74fa/final.pt",
        kind="text",
    ),
    RegionCheckpoint(
        region="compress",
        legacy="compress",
        checkpoint=MATRIX_ROOT
        / "compress-b512-s0-7bc2699-20260904/receipts/compress-checkpoints/116eae1c/final.pt",
        kind="text",
    ),
    RegionCheckpoint(
        region="retrieve",
        legacy="retrieve",
        checkpoint=MATRIX_ROOT
        / "retrieve-b512-s0-7bc2699-20260904/receipts/retrieve-checkpoints/1d2bb223/final.pt",
        kind="text",
    ),
    RegionCheckpoint(
        region="reason",
        legacy="reason",
        checkpoint=MATRIX_ROOT
        / "reason-b256-s0-7bc2699-20260904/receipts/reason-checkpoints/ab5ba4db/final.pt",
        kind="text",
    ),
    RegionCheckpoint(
        region="visual",
        legacy="visual",
        checkpoint=MATRIX_ROOT
        / "visual-b128-s0-3ce18db-20260905/receipts/visual-checkpoints/step-4000.pt",
        kind="visual",
    ),
)
"""The five regions with a real trained checkpoint under `MATRIX_ROOT`, one cell each
(`b512-s0` / `b256-s0` / `b128-s0` -- the smallest-batch, seed-0 cell per region; the
matrix also holds a `b1280`/other-seed cell per region, which this table does not need
to duplicate). `memory` is not here: no matrix cell trains it as of this commit (see
the module docstring)."""


@dataclass(frozen=True)
class RegionParamCount:
    """One region's parameters, split embedding / token path / pooling head."""

    region: str
    kind: str
    embedding: int
    token_path: int
    pooling_head: int
    checkpoint: str | None = None

    @property
    def total(self) -> int:
        """Sum of the three components."""
        return self.embedding + self.token_path + self.pooling_head

    @property
    def embedding_share(self) -> float:
        """Embedding as a fraction of `total`; `0.0` if `total` is `0`."""
        return self.embedding / self.total if self.total else 0.0


@dataclass(frozen=True)
class RegionTableRow:
    """One row of the re-instantiated table: either a measured count, or why not."""

    entry: RegionCheckpoint
    counts: RegionParamCount | None
    error: str | None


def count_text_encoder_params(model: TextEncoder) -> tuple[int, int, int]:
    """Split a `TextEncoder`'s parameters: `(embedding, token_path, pooling_head)`.

    `embedding` is `model.embed`'s table. `pooling_head` is `model.proj` -- `0` when
    `proj` is `nn.Identity` (every production text region today: `out_dim=None`).
    `token_path` is everything else (the transformer blocks and the final norm) --
    computed as `total - embedding - pooling_head` rather than summed over `model.blocks`
    + `model.norm` directly, so a future field added to `TextEncoder` is accounted for
    by construction instead of silently missing from both totals.
    """
    embedding = sum(p.numel() for p in model.embed.parameters())
    pooling_head = (
        0
        if isinstance(model.proj, nn.Identity)
        else sum(p.numel() for p in model.proj.parameters())
    )
    total = sum(p.numel() for p in model.parameters())
    token_path = total - embedding - pooling_head
    return embedding, token_path, pooling_head


def count_visual_encoder_params(encoder: ViTEncoder) -> tuple[int, int, int]:
    """Split a `ViTEncoder`'s parameters: `(embedding, token_path, pooling_head)`.

    `embedding` is `patch_embed`'s `Conv2d` -- the patch-projection analogue of a text
    region's token embedding table. `pooling_head` is always `0`: `ViTEncoder.pool` has
    no `proj` (its own docstring: "unlike the text regions, ViTEncoder's output width
    already is the shared-stream width the catalogue declares"). `token_path` is
    `total - embedding` (the transformer blocks and the final norm).
    """
    embedding = sum(p.numel() for p in encoder.patch_embed.parameters())
    total = sum(p.numel() for p in encoder.parameters())
    token_path = total - embedding
    return embedding, token_path, 0


def faculty_param_count(
    region: str, kind: str, module: nn.Module, *, checkpoint: str | None = None
) -> RegionParamCount:
    """Given a region name/kind and an INSTANTIATED module, return its param split.

    Args:
        region: Region name to label the row with (typically `RegionCheckpoint.region`
            or a `RegionSpec.name`).
        kind: `"text"` or `"visual"` -- selects which counting rule applies. Kept as an
            explicit argument (rather than solely dispatched on `type(module)`) so a
            caller can label a row correctly even when it passes a bare `IJEPA` (visual)
            without unwrapping `.target_encoder` itself -- see below.
        module: The built module to count. Accepts a `TextEncoder`, a `ViTEncoder`, or
            an `IJEPA` (in which case the deployed `target_encoder` half is counted --
            DEC-34 -- not the context encoder or the predictor, since the predictor is a
            training-only component with no `Faculty` role and counting both encoder
            copies would double the real deployed parameter count).
        checkpoint: Optional path string to record on the returned row for provenance.

    Returns:
        A `RegionParamCount` with `total == embedding + token_path + pooling_head` by
        construction.

    Raises:
        TypeError: `module` is not one of the three types this function knows how to
            count, or `kind` does not match `module`'s actual type.
    """
    if isinstance(module, IJEPA):
        module = module.target_encoder
    if kind == "text":
        if not isinstance(module, TextEncoder):
            raise TypeError(f"{region}: kind='text' but module is {type(module).__name__}")
        embedding, token_path, pooling_head = count_text_encoder_params(module)
    elif kind == "visual":
        if not isinstance(module, ViTEncoder):
            raise TypeError(f"{region}: kind='visual' but module is {type(module).__name__}")
        embedding, token_path, pooling_head = count_visual_encoder_params(module)
    else:
        raise TypeError(f"{region}: unknown kind {kind!r} -- expected 'text' or 'visual'")
    return RegionParamCount(
        region=region,
        kind=kind,
        embedding=embedding,
        token_path=token_path,
        pooling_head=pooling_head,
        checkpoint=checkpoint,
    )


def load_region_module(
    entry: RegionCheckpoint, *, map_location: str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Load `entry.checkpoint` read-only, through `load_checkpoint` (DEC-40/W0c).

    Args:
        entry: Which checkpoint to load and what shape to build.
        map_location: Forwarded to `load_checkpoint`; `"cpu"` by default so this never
            touches a GPU another lane may be using.

    Returns:
        `(model, raw_checkpoint_dict)`, `model` in `eval()` mode with the checkpoint's
        weights loaded, unchanged (no retraining, no re-initialization).

    Raises:
        FileNotFoundError: `entry.checkpoint` does not exist on this host.
        Exception: Whatever `load_checkpoint`/`load_state_dict` raises on a genuinely
            unreadable or shape-mismatched checkpoint -- not caught here; callers that
            want a "could not load, say so" row use `build_matrix_param_table` instead.
    """
    ckpt = load_checkpoint(entry.checkpoint, map_location=map_location)
    if entry.kind == "text":
        cfg = TextEncoderConfig(**ckpt["config"])
        text_model: nn.Module = TextEncoder(cfg, name=entry.region)
        text_model.load_state_dict(ckpt["model"])
        text_model.eval()
        return text_model, ckpt
    if entry.kind == "visual":
        jcfg = JEPAConfig(**ckpt["config"])
        visual_model: nn.Module = IJEPA(jcfg)
        visual_model.load_state_dict(ckpt["model"])
        visual_model.eval()
        return visual_model, ckpt
    raise ValueError(f"{entry.region}: unknown checkpoint kind {entry.kind!r}")


def build_matrix_param_table(
    entries: tuple[RegionCheckpoint, ...] = CANONICAL_CHECKPOINTS,
) -> list[RegionTableRow]:
    """Re-instantiate the per-region parameter split for each of `entries`.

    Read-only: never writes a checkpoint, never trains a step. A checkpoint absent from
    this host, or one that fails to load, produces a row with `counts=None` and a
    human-readable `error` instead of raising -- one bad cell does not blank the table.

    Args:
        entries: Which checkpoints to re-instantiate; defaults to
            `CANONICAL_CHECKPOINTS` (the five regions with a real matrix checkpoint).

    Returns:
        One `RegionTableRow` per entry, in the same order.
    """
    rows: list[RegionTableRow] = []
    for entry in entries:
        if not entry.checkpoint.is_file():
            rows.append(
                RegionTableRow(
                    entry=entry,
                    counts=None,
                    error=f"checkpoint not present on this host: {entry.checkpoint}",
                )
            )
            continue
        try:
            module, _ckpt = load_region_module(entry, map_location="cpu")
        except Exception as exc:
            rows.append(
                RegionTableRow(entry=entry, counts=None, error=f"{type(exc).__name__}: {exc}")
            )
            continue
        counts = faculty_param_count(
            entry.region, entry.kind, module, checkpoint=str(entry.checkpoint)
        )
        rows.append(RegionTableRow(entry=entry, counts=counts, error=None))
    return rows


def format_param_table(rows: list[RegionTableRow]) -> str:
    """Render `build_matrix_param_table`'s output as a Markdown table."""
    header = "| region | legacy | kind | embedding | token path | pooling head | total | embedding share |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        if row.counts is None:
            lines.append(
                f"| {row.entry.region} | {row.entry.legacy} | {row.entry.kind} | -- | -- | -- "
                f"| -- | COULD NOT LOAD: {row.error} |"
            )
            continue
        c = row.counts
        lines.append(
            f"| {row.entry.region} | {row.entry.legacy} | {c.kind} | {c.embedding:,} | "
            f"{c.token_path:,} | {c.pooling_head:,} | {c.total:,} | {c.embedding_share:.2%} |"
        )
    return "\n".join(lines)
