#!/usr/bin/env python3
"""Render one CogSynDelta model card, via `cogsyndelta.cards.render_card`.

WHY THIS IS A SEPARATE TOOL, NOT A SWITCH ON `csd-publish-checkpoint.py`
That script's `build_card` is a hand-built, single-format renderer whose exact markdown
shape (`## Metrics`, `### held_out`, ...) dozens of existing tests pin byte-for-byte,
because it is the format every already-published `tzervas/cogsyndelta-region-*` repo
carries today. `cogsyndelta.cards.render_card` is the newer, template-driven library
(CARD SPEC, operator, 2026-09-04) that renders a richer, per-`kind` card
(`region_variant`, `region_main`, `memory`, `placeholder`, `composed`) with grouped
metric tables, numbered footnotes, and MEASURED sizes -- a different card shape by
design, not a drop-in replacement. This CLI is `cogsyndelta.cards`'s own command-line
front end; `csd-publish-checkpoint.py` is untouched by it (see that script's own
docstring for its independent story).

TWO WAYS TO NAME THE RECEIPTS
`--cell <cell dir>` reads `<cell dir>/cell.json` (the shape `model-matrix`'s harness
writes -- see `program/matrix/csd-matrix.yaml`; a real example lives under
`/akula-data/csd/matrix/<cell_id>/cell.json`) and follows its own `stages.*.receipt`
pointers -- the exact paths the harness recorded having written, never a re-derived
glob -- picking up whichever of `train`/`test`/`quantize`/`test-quant` reached
`state: "done"`. `--train-receipt`/`--eval-receipt`/`--eval-quantized-receipt`/
`--quant-receipt` name receipt files directly, for a cell the harness hasn't recorded
(or a receipt that never went through it at all); given alongside `--cell`, an explicit
flag OVERRIDES that one receipt from the cell without disturbing the others.

LICENCE TIER AND REGION CONFIG ARE IMPORTED, NOT DUPLICATED
The audited per-region licence table (`LICENCE_TIER`, `LICENCE_WHY`, `BLOCKING_REGIONS`)
and the `config/mind/csd-regions.json` loader already live in
`scripts/csd-publish-checkpoint.py` -- this tool imports that script as a module (a
hyphenated filename is not importable by name, hence `importlib`) rather than keeping a
second copy that could silently drift from the one publish actually enforces. A region
with no audited tier, or `visual`/`vl_latent` whose training receipt is not the admitted
Mix B pin, refuses exactly as `licence_tier(region, receipt=train)` already does for a
publish -- Mix B (`visual-clean-v1` + the pinned fingerprint) is MIT with the CLEVR
TASL block; tiny-imagenet / missing / fingerprint-mismatched stays BLOCKING. Except
for `--kind placeholder`/`composed`, where `cogsyndelta.cards.render`'s own contract
allows an unresolved tier (a stated-TBD / no-weights line rather than an invented
licence); this tool best-efforts a tier there too, when one happens to be on record,
but never refuses for its absence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(REPO_ROOT / "src")
# Front of sys.path, not append: an editable install of a *different* checkout
# (the operator's shared .venv pointing at main, a worktree running this
# script) would otherwise win and silently use stale METRIC_METHODOLOGY /
# render_card. The script's own tree is the code this CLI is meant to run.
if _SRC in sys.path:
    sys.path.remove(_SRC)
sys.path.insert(0, _SRC)

from cogsyndelta.cards.methodology import CardError  # noqa: E402 -- needs sys.path set above
from cogsyndelta.cards.render import CARD_KINDS, render_card  # noqa: E402
from cogsyndelta.regions.aliases import canonical_region  # noqa: E402

_PUBLISH_SCRIPT = Path(__file__).resolve().parent / "csd-publish-checkpoint.py"


class CardCliError(Exception):
    """Raised for a CLI-level condition this tool refuses to proceed past -- a bad
    argument combination, a missing cell/file, an unreadable comparators table.
    Distinct from `CardError` (`render_card`'s own refusal, e.g. an undocumented
    metric) and from the imported publish script's `PublishAbortError` (its licence
    refusal): `main()` catches all three under one non-zero exit rather than
    conflating "you asked for something invalid" with "the receipts themselves refuse
    to render."
    """


def _load_publish_module() -> Any:
    """Import `scripts/csd-publish-checkpoint.py` as a module -- for its audited
    licence table and `config/mind/csd-regions.json` loader, the SAME source of truth
    a real publish uses (see this file's own module docstring for why this is an
    import, not a second copy)."""
    name = "csd_publish_checkpoint_for_card_cli"
    spec = importlib.util.spec_from_file_location(name, _PUBLISH_SCRIPT)
    if spec is None or spec.loader is None:
        raise CardCliError(f"could not load {_PUBLISH_SCRIPT} as a module")
    mod = importlib.util.module_from_spec(spec)
    # Registered in sys.modules BEFORE exec_module: the script defines a `@dataclass`
    # (`Plan`), and dataclass field-type resolution looks its own module up in
    # sys.modules by `cls.__module__` -- an unregistered module fails that lookup with
    # an opaque AttributeError. tests/test_publish_checkpoint.py's own `load_mod()`
    # does the same, for the same reason.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as e:
        raise CardCliError(f"{path}: not found") from e
    except json.JSONDecodeError as e:
        raise CardCliError(f"{path}: not valid JSON ({e})") from e
    if not isinstance(data, dict):
        raise CardCliError(f"{path}: must be a JSON object, got {type(data).__name__}")
    return data


# `cell.json`'s own stage names (program/matrix/csd-matrix.yaml's `stages:` keys) ->
# the receipts dict key `render_card` expects. `finetune` is deliberately absent: no
# `test-ft`/`finetune` cell has ever been produced (program/matrix/csd-matrix.yaml's
# own comment: the pipeline is declared but unreferenced, A4 unimplemented) -- adding
# a mapping for a stage no cell.json can currently carry would be untestable.
_CELL_STAGE_TO_RECEIPT_KIND: dict[str, str] = {
    "train": "train",
    "test": "eval",
    "quantize": "quant",
    "test-quant": "eval_quantized",
}


def _receipts_from_cell(cell_dir: Path) -> tuple[str, dict[str, dict[str, Any] | None]]:
    """`(region, receipts)` read from `<cell_dir>/cell.json`'s own `stages.*.receipt`
    pointers. A stage that never reached `state: "done"`, or whose pointer no longer
    resolves to a file (pruned, moved), is skipped rather than treated as an error --
    a card for a cell that has only trained so far is still a valid fp32-only card.
    """
    cell_path = cell_dir / "cell.json"
    cell = _load_json(cell_path)
    region = cell.get("region")
    if not region:
        raise CardCliError(f"{cell_path}: no 'region' field")
    stages = cell.get("stages")
    if not isinstance(stages, dict):
        raise CardCliError(f"{cell_path}: 'stages' is missing or not an object")

    receipts: dict[str, dict[str, Any] | None] = {}
    for stage_name, kind in _CELL_STAGE_TO_RECEIPT_KIND.items():
        stage = stages.get(stage_name)
        if not isinstance(stage, dict) or stage.get("state") != "done":
            continue
        receipt_path = stage.get("receipt")
        if not receipt_path:
            continue
        p = Path(str(receipt_path))
        if not p.is_file():
            continue
        receipts[kind] = _load_json(p)
    return str(region), receipts


def _load_comparators(path: Path | None) -> dict[str, dict[str, Any]] | None:
    """`{variant_label: eval_receipt}` from `--comparators <table.json>`: each value is
    either a path string (loaded) or an inline JSON object (used as-is), so a small
    table can name receipts by path without anyone hand-copying their contents.
    """
    if path is None:
        return None
    raw = _load_json(path)
    out: dict[str, dict[str, Any]] = {}
    for label, value in raw.items():
        if isinstance(value, str):
            out[label] = _load_json(Path(value))
        elif isinstance(value, dict):
            out[label] = value
        else:
            raise CardCliError(
                f"{path}: comparator {label!r} must be a path string or a JSON object, "
                f"got {type(value).__name__}"
            )
    return out


def _checkpoint_files_block(
    pub_mod: Any, train_receipt: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    """Best-effort `{"final.pt": {"sha256": ...}}` for the Provenance section, reusing
    the publish script's own containment-checked path resolver and hasher
    (`checkpoint_path_from_receipt` + `verify_checkpoint_sha`) rather than a second,
    unverified `Path(receipt["checkpoint"])`. Never fatal: a receipt with no checkpoint
    path, or one this tool cannot read (moved, permissions), yields an empty block --
    this is a card preview, not an upload, so a missing checkpoint file is not this
    tool's problem to abort over.
    """
    if train_receipt is None:
        return {}
    try:
        checkpoint = pub_mod.checkpoint_path_from_receipt(train_receipt)
        sha256 = pub_mod.verify_checkpoint_sha(checkpoint)
    except pub_mod.PublishAbortError:
        return {}
    return {checkpoint.name: {"sha256": sha256}}


_TIER_REQUIRED_KINDS = frozenset(set(CARD_KINDS) - {"placeholder", "composed"})


def _region_cfg(
    pub_mod: Any,
    region: str,
    kind: str,
    train_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """This region's `config/mind/csd-regions.json` entry, extended with
    `licence_tier`/`licence_why` from the publish script's audited table -- see the
    module docstring for why importing beats duplicating that table.

    `train_receipt` is passed through to `licence_tier()`: visual is BLOCKING by
    default and only resolves to MIT when the receipt names the admitted Mix B
    corpus pin. Calling `licence_tier(region)` with no receipt would abort a Mix B
    visual card the publish path itself would accept.
    """
    if kind == "composed":
        cfg: dict[str, Any] = {}
    else:
        cfg = dict(pub_mod.load_region_config(region))

    if kind in _TIER_REQUIRED_KINDS:
        # raises PublishAbortError: unknown / BLOCKING (including visual whose
        # receipt is not the admitted Mix B pin)
        tier = pub_mod.licence_tier(region, receipt=train_receipt)
    else:
        # placeholder / composed: render_card itself allows an unresolved tier (a
        # stated-TBD / no-weights line) -- best-effort a real one only when the region
        # HAS an audited, non-BLOCKING tier on record, never refuse for its absence.
        try:
            tier = pub_mod.licence_tier(region, receipt=train_receipt)
        except pub_mod.PublishAbortError:
            tier = None
    if tier is not None:
        cfg["licence_tier"] = tier
        cfg["licence_why"] = pub_mod.LICENCE_WHY.get(
            pub_mod.canonical_region(region), "(reason not recorded)"
        )
    return cfg


def _build_receipts_and_region(
    args: argparse.Namespace,
    pub_mod: Any,
) -> tuple[str, dict[str, dict[str, Any] | None]]:
    if args.cell is not None:
        region, receipts = _receipts_from_cell(args.cell)
        # Canonically compared: a cell recorded under the legacy `code` still matches
        # `--region language`, and vice versa -- see cogsyndelta.regions.aliases.
        if args.region and pub_mod.canonical_region(args.region) != pub_mod.canonical_region(
            region
        ):
            raise CardCliError(
                f"--region {args.region!r} does not match {args.cell / 'cell.json'}'s "
                f"own region {region!r} -- refusing to render a card for a region "
                "mismatched with the receipts that back it"
            )
    else:
        if not args.region:
            raise CardCliError("--region is required when --cell is not given")
        region = args.region
        receipts = {}

    if args.train_receipt is not None:
        receipts["train"] = _load_json(args.train_receipt)
    if args.eval_receipt is not None:
        receipts["eval"] = _load_json(args.eval_receipt)
    if args.eval_quantized_receipt is not None:
        receipts["eval_quantized"] = _load_json(args.eval_quantized_receipt)
    if args.quant_receipt is not None:
        receipts["quant"] = _load_json(args.quant_receipt)

    if args.kind not in ("placeholder", "composed") and "train" not in receipts:
        raise CardCliError(
            f"--kind {args.kind!r} needs at least a training receipt "
            "(--cell with a done train stage, or --train-receipt)"
        )
    if args.kind == "placeholder":
        region = canonical_region(region)
    return region, receipts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument(
        "--cell", type=Path, default=None, help="cell directory (reads cell.json + receipts/)"
    )
    ap.add_argument("--train-receipt", type=Path, default=None)
    ap.add_argument("--eval-receipt", type=Path, default=None)
    ap.add_argument("--eval-quantized-receipt", type=Path, default=None)
    ap.add_argument("--quant-receipt", type=Path, default=None)
    ap.add_argument("--kind", choices=CARD_KINDS, default="region_variant")
    ap.add_argument("--region", default=None, help="required unless --cell supplies one")
    ap.add_argument(
        "--comparators",
        type=Path,
        default=None,
        help='JSON {"variant_label": <eval receipt path or object>, ...} -- other '
        "cells of the same region, added as extra table columns",
    )
    ap.add_argument("--out", type=Path, default=Path("README.md"))
    ap.add_argument("--repo", default=None, help="owner/name, printed in the card header")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pub_mod = _load_publish_module()
    try:
        region, receipts = _build_receipts_and_region(args, pub_mod)
        train_receipt = receipts.get("train")
        region_cfg = _region_cfg(pub_mod, region, args.kind, train_receipt=train_receipt)
        attr_lines = pub_mod.visual_attribution_block(train_receipt)
        if attr_lines:
            region_cfg["attribution_md"] = "\n".join(attr_lines).strip()
        comparators = _load_comparators(args.comparators)
        files = _checkpoint_files_block(pub_mod, receipts.get("train"))
        card = render_card(
            args.kind,
            region=region,
            region_cfg=region_cfg,
            receipts=receipts,
            files=files,
            comparators=comparators,
            repo=args.repo,
        )
    except (CardError, CardCliError, pub_mod.PublishAbortError) as e:
        print(f"ABORT: {e}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(card)
    print(f"wrote {args.out} ({len(card)} bytes, kind={args.kind}, region={region})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
