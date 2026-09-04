"""`render_card`: the one public entry point of `cogsyndelta.cards`.

Picks the Jinja2 template for a card `kind`, builds every section (front matter via
`cogsyndelta.cards.metadata`, metric tables via `cogsyndelta.cards.tables`, sizes via
`cogsyndelta.cards.sizes`), and calls `huggingface_hub.ModelCard.from_template`.
Refuses (`cogsyndelta.cards.methodology.CardError`) rather than render on an
undocumented metric or a `metrics_schema` disagreement across the receipts merged into
one card -- see `cogsyndelta.cards.tables.assert_schemas_agree` and
`cogsyndelta.cards.methodology.require_documented`.

THE `receipts` / `region_cfg` CONTRACT
`receipts` is `{"train": ..., "eval": ..., "eval_quantized": ..., "quant": ...}`; every
key is optional (a `placeholder` card supplies none; a `region_variant` fp32-only card
supplies only `"train"`). `region_cfg` is the region's entry from
`config/mind/csd-regions.json`, EXTENDED by the caller with two fields this package
does not derive on its own (deriving a licence tier is `scripts/csd-publish-
checkpoint.py`'s `licence_tier()`/`LICENCE_TIER` job, a policy table this library does
not duplicate -- see that function's own docstring for why guessing one here would be
wrong):

- `region_cfg["licence_tier"]`: the resolved tier string (`"mit"`,
  `"cc-by-nc-sa-4.0"`, ...), or `None` for a `composed` card whose licence the
  operator has not resolved (CARD SPEC §9: "the LICENSE contradiction is the
  operator's to resolve -- render a placeholder line, do not invent a licence").
  Required (not `None`) for every OTHER kind; `render_card` raises `CardError` if
  it is missing.
- `region_cfg["licence_why"]`: one sentence, printed in the Licence section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import ModelCard

from cogsyndelta.cards.metadata import build_card_data
from cogsyndelta.cards.methodology import (
    METHODOLOGY_DOC,
    METRIC_METHODOLOGY,
    CardError,
    MetricMethodology,
    methodology_key,
    normalize_quant_receipt_v1,
)
from cogsyndelta.cards.sizes import (
    DEFAULT_BUDGETS_ROOT,
    MeasuredValue,
    SizeReport,
    build_size_report,
)
from cogsyndelta.cards.tables import (
    V1_FOOTNOTE,
    MetricTable,
    assert_schemas_agree,
    build_eval_tables,
    build_gate_table,
    build_quant_table,
    build_training_table,
    render_table_markdown,
)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

CARD_KINDS: tuple[str, ...] = ("region_variant", "region_main", "memory", "placeholder", "composed")

#: The `memory` card kind's required OD-17 status line (CARD SPEC: "memory (region
#: with the OD-17 status stated)"). Source: `docs/design/REGION-TAXONOMY-AND-
#: INTERCONNECT.md` DEC-68 -- the pre-registered W4 run at batch 1280 passes (a)/(d),
#: fails (c)/(e), and ties exactly on (b) under a strict `>` reading (the receipt's
#: `0.20000000298023224` is `float(numpy.float32(0.2))`, i.e. the margin is float32
#: representation error) -- OD-17 puts the `>`-vs-`>=` reading, and pivot-vs-amend, to
#: the operator; the programme does not proceed past it on `memory`'s own authority.
#: A constant, not re-derived from the design doc at render time, because this
#: library has no dependency on that doc's own prose staying byte-identical --
#: `region_cfg["od17_status"]` overrides this default when a caller has a fresher
#: status to report (OD-17 answered, superseded, etc).
MEMORY_OD17_STATUS_DEFAULT = (
    "**Blocked on OD-17.** The pre-registered W4 gate run (batch 1280, `memory-"
    "20260903T184441Z.json`, code `eb735ab`) passes gates (a) and (d), fails (c) and "
    "(e), and gate (b) is an EXACT TIE under a strict `>` reading: the receipt's "
    "`recall@10` is `0.20000000298023224`, which is `float(numpy.float32(0.2))` -- "
    "the apparent margin over the 0.20 floor is float32 representation error, not a "
    "real pass. OD-17 puts the `>` vs `>=` reading of that gate, and a pivot-vs-amend "
    "decision, to the operator (see `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md`, "
    "DEC-68 and §8 OD-17); nothing downstream of this region proceeds on its own "
    "authority until OD-17 is answered."
)

#: Regions with no design row / no promotable weights, and what their repo name refers
#: to instead -- CARD SPEC's `placeholder` kind: "states weights: none, what the name
#: refers to per the design". Kept here (not in `config/mind/csd-regions.json`)
#: because it is prose about the DESIGN DOCUMENT, not a training-time region property.
PLACEHOLDER_REFERENT: dict[str, str] = {
    "stream_vae": "the DEC-08 tract-codec candidate (see docs/design; not the toy "
    "`stream_vae` PoC region trained today under `residual_mlp`'s sibling config)",
    "residual_mlp": "no design row exists for this name in the current design "
    "revision -- the repo name predates the ratified region catalogue",
    "vl_latent": "the future `visual` slot (see docs/design; today's `zh-plus/tiny-"
    "imagenet` toy checkpoint is BLOCKING per docs/design/LICENCE-FOR-OPEN-WEIGHTS.md "
    "and is not what this repo, once populated, is meant to hold)",
}


def _template_path(kind: str) -> Path:
    if kind not in CARD_KINDS:
        raise CardError(f"unknown card kind {kind!r} -- must be one of {CARD_KINDS}")
    return TEMPLATES_DIR / f"{kind}.md.j2"


def _assign_footnotes(
    tables: list[MetricTable], methodology: dict[str, MetricMethodology]
) -> dict[tuple[str, str, str, str], int]:
    """One footnote number per unique (definition, battery_id, pooling, source) tuple
    referenced by any row of any table, numbered in first-encounter order across every
    table -- so a metric shared by two tables gets one footnote, not two.
    """
    numbers: dict[tuple[str, str, str, str], int] = {}
    for table in tables:
        for row in table.rows:
            m = methodology[methodology_key(row.key)]
            key = (m.definition, m.battery_id, m.pooling, m.source)
            if key not in numbers:
                numbers[key] = len(numbers) + 1
    return numbers


def _footnotes_markdown(numbers: dict[tuple[str, str, str, str], int]) -> str:
    lines = []
    for (definition, battery_id, pooling, source), n in sorted(
        numbers.items(), key=lambda kv: kv[1]
    ):
        bid = f"`{battery_id}`" if battery_id else "_(n/a)_"
        pl = f"`{pooling}`" if pooling else "_(n/a)_"
        lines.append(f"[^{n}]: {definition} -- battery_id={bid}, pooling={pl}, `{source}`.")
    return "\n".join(lines)


def _render_tables_block(
    tables: list[MetricTable],
    footnote_numbers: dict[tuple[str, str, str, str], int],
    methodology: dict[str, MetricMethodology],
) -> str:
    parts = []
    any_v1 = False
    for table in tables:
        if not table.rows:
            continue
        parts.append(f"### {table.heading}\n")
        parts.append(
            render_table_markdown(table, methodology=methodology, footnote_numbers=footnote_numbers)
        )
        parts.append("")
        if any(r.is_v1_mapped for r in table.rows):
            any_v1 = True
    if any_v1:
        parts.append(f"^v1^ {V1_FOOTNOTE}\n")
    return "\n".join(parts)


def _sizes_block(sizes: SizeReport) -> str:
    lines = ["| quantity | value | label |", "|---|---|---|"]
    if sizes.parameters is not None:
        lines.append(f"| parameters | {sizes.parameters / 1e6:.3f} M | measured, `parameters` |")
    if sizes.fp32_bytes is not None:
        lines.append(
            f"| weights on disk, fp32 | {sizes.fp32_bytes / 1e6:.3f} MB | measured, `fp32_bytes` |"
        )
    if sizes.quantized_stored_bytes is not None:
        lines.append(
            f"| weights on disk, csd-ptq-v1 | {sizes.quantized_stored_bytes / 1e6:.3f} MB | "
            "measured, `stored_bytes` |"
        )
    if sizes.compression_ratio is not None:
        lines.append(
            f"| compression ratio (storage, not speed) | {sizes.compression_ratio:.3g}x | measured |"
        )
    if sizes.width_histogram is not None:
        hist = ", ".join(f"{bits}-bit: {n}" for bits, n in sorted(sizes.width_histogram.items()))
        lines.append(f"| bit-width histogram | {hist} | measured, `width_histogram` |")

    def _mv_row(name: str, mv: MeasuredValue) -> str:
        return f"| {name} | {mv.value:.3f} {mv.unit} | MEASURED, {mv.label} |"

    for key, name in (("fp32", "eval peak VRAM, fp32"), ("quantized", "eval peak VRAM, quantized")):
        mv = sizes.eval_peak_vram.get(key)
        if mv is not None:
            lines.append(_mv_row(name, mv))
    if sizes.training_peak is not None:
        lines.append(_mv_row("training peak VRAM", sizes.training_peak))
    if len(lines) == 2:
        return "_No sizes recorded -- no receipts with size/memory fields were supplied._\n"
    return "\n".join(lines)


def _provenance_block(
    *,
    train_receipt: dict[str, Any] | None,
    checkpoint_sha256: str | None,
    code_revision: str | None,
    metrics_schema: str,
    files: dict[str, dict[str, Any]],
) -> str:
    corpus = (train_receipt or {}).get("corpus", {})
    fingerprint = corpus.get("fingerprint") or (train_receipt or {}).get("corpus_fingerprint")
    scheme = corpus.get("fingerprint_scheme", "(scheme not recorded)")
    seed = (train_receipt or {}).get("config", {}).get("seed")
    lines = [
        f"- **Corpus fingerprint:** `{fingerprint or '(none recorded)'}` (scheme `{scheme}`)",
        f"- **Seed:** `{seed if seed is not None else '(none recorded)'}`",
        f"- **Code revision:** `{code_revision or '(none recorded)'}`",
        f"- **Checkpoint sha256:** `{checkpoint_sha256 or '(no checkpoint on this card)'}`",
        f"- **Metrics schema:** `{metrics_schema}`",
    ]
    if files:
        lines.append("- **Files:**")
        for name, meta in sorted(files.items()):
            sha = meta.get("sha256", "(no sha256 recorded)")
            lines.append(f"  - `{name}`: sha256 `{sha}`")
    return "\n".join(lines)


#: Kinds allowed to render with no resolved `licence_tier` -- a `composed` card
#: because CARD SPEC §9 explicitly wants a placeholder line rather than an invented
#: licence, and a `placeholder` card because, by definition (`weights: none`), there
#: is no trained artifact for a data licence to attach to yet. Every OTHER kind names
#: a real trained checkpoint's weights, and `licence_tier()` in `scripts/csd-publish-
#: checkpoint.py` already refuses to publish those without an audited tier -- this
#: card library inherits that refusal rather than softening it.
_TIER_OPTIONAL_KINDS = frozenset({"composed", "placeholder"})


def _licence_block(region_cfg: dict[str, Any], *, kind: str) -> tuple[str | None, str]:
    tier = region_cfg.get("licence_tier")
    why = region_cfg.get("licence_why", "(reason not recorded)")
    if tier is None:
        if kind not in _TIER_OPTIONAL_KINDS:
            raise CardError(
                f"region_cfg has no 'licence_tier' -- refusing to render a {kind!r} "
                "card with a guessed licence. Resolve the tier (see "
                "docs/design/LICENCE-FOR-OPEN-WEIGHTS.md) and set "
                "region_cfg['licence_tier'] before calling render_card."
            )
        if kind == "placeholder":
            return None, (
                "**No licence tier: no weights.** This repo holds no trained "
                "checkpoint (see 'What this name refers to' above), so there is no "
                "trained artifact for a data licence to attach to yet."
            )
        return None, (
            "**Licence: TBD.** This composed model's LICENSE file and its "
            "per-component licence tiers are not reconciled as of this card -- see "
            "`docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`, Rider 1 (a merge inherits its "
            "most restrictive parent). Resolving this is the operator's decision, not "
            "this library's to invent."
        )
    return (
        tier,
        f'`{tier}` -- {why}. See `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`, "Decision 2026-09-02".',
    )


def render_card(
    kind: str,
    *,
    region: str,
    region_cfg: dict[str, Any],
    receipts: dict[str, dict[str, Any] | None],
    files: dict[str, dict[str, Any]],
    comparators: dict[str, dict[str, Any]] | None = None,
    budgets_root: Path = DEFAULT_BUDGETS_ROOT,
    repo: str | None = None,
) -> str:
    """Render one model card as markdown (front matter + body).

    Args:
        kind: one of `CARD_KINDS`.
        region: the region name (or `"cogsyndelta"` for `kind="composed"`).
        region_cfg: this region's `config/mind/csd-regions.json` entry, extended with
            `licence_tier`/`licence_why` -- see the module docstring.
        receipts: `{"train": ..., "eval": ..., "eval_quantized": ..., "quant": ...}`,
            every key optional. A `quant` receipt need not be pre-normalised --
            `render_card` runs it through `normalize_quant_receipt_v1` itself.
        files: `{path_in_repo: {"sha256": ..., ...}}` for every file this card names
            (checkpoint, quantized artifact, receipts) -- printed in Provenance.
        comparators: `{variant_label: eval_receipt}` for other cells of the SAME
            region (CARD SPEC §5: "other variants of the same region when a region
            table is supplied") -- adds one column per label to the eval tables.
        budgets_root: passed through to `cogsyndelta.cards.sizes.build_size_report`;
            override only for a test (a `tmp_path` with no budget files, for a
            deterministic "no training peak recorded" result) or an alternate cluster.
        repo: the Hub `owner/name` this card is destined for, printed in the header
            line beside the licence/code-revision/metrics-schema summary -- `None`
            (the default) omits it, leaving every existing render (and the golden
            snapshot) byte-identical. Never derived here (this library does not decide
            repo naming -- `scripts/csd-publish-checkpoint.py`'s `default_repo()` and
            `scripts/csd-card.py`'s `--repo` are the two callers that resolve one).

    Returns:
        The rendered card as a markdown string (YAML front matter + body).

    Raises:
        CardError: an undocumented metric, a `metrics_schema` disagreement across the
            supplied receipts, or (non-`composed` kinds) a missing `licence_tier`.
    """
    train_receipt = receipts.get("train")
    eval_receipt = receipts.get("eval")
    eval_quantized_receipt = receipts.get("eval_quantized")
    quant_receipt = receipts.get("quant")
    if quant_receipt is not None:
        quant_receipt = normalize_quant_receipt_v1(quant_receipt)

    schema_inputs = {
        "train": train_receipt,
        "eval": eval_receipt,
        "eval_quantized": eval_quantized_receipt,
        "quant": quant_receipt,
    }
    supplied = {k: v for k, v in schema_inputs.items() if v is not None}
    metrics_schema = assert_schemas_agree(supplied) if supplied else "(no receipts supplied)"

    tables: list[MetricTable] = []
    if train_receipt is not None:
        tables.append(build_training_table(train_receipt))
    gate_table = build_gate_table(eval_receipt)
    if gate_table is not None:
        tables.append(gate_table)
    tables.extend(
        build_eval_tables(
            eval_receipt=eval_receipt,
            comparators=comparators,
            heading_suffix=" (fp32)" if eval_quantized_receipt is not None else "",
        )
    )
    if eval_quantized_receipt is not None:
        tables.extend(
            build_eval_tables(
                eval_receipt=eval_quantized_receipt,
                heading_suffix=" (quantized artifact)",
            )
        )
    quant_table = build_quant_table(quant_receipt)
    if quant_table is not None:
        tables.append(quant_table)

    footnote_numbers = _assign_footnotes(tables, METRIC_METHODOLOGY)
    tables_md = _render_tables_block(tables, footnote_numbers, METRIC_METHODOLOGY)
    footnotes_md = _footnotes_markdown(footnote_numbers)

    sizes = build_size_report(
        region=region,
        train_receipt=train_receipt or {},
        quant_receipt=quant_receipt,
        eval_receipt=eval_receipt,
        eval_quantized_receipt=eval_quantized_receipt,
        budgets_root=budgets_root,
    )
    sizes_md = _sizes_block(sizes)

    tier, licence_line = _licence_block(region_cfg, kind=kind)

    checkpoint_sha256 = None
    if train_receipt is not None:
        checkpoint_sha256 = train_receipt.get("checkpoint_sha256") or train_receipt.get(
            "artifacts", {}
        ).get("checkpoint_sha256")
    code_revision = (train_receipt or {}).get("code_revision")

    provenance_md = _provenance_block(
        train_receipt=train_receipt,
        checkpoint_sha256=checkpoint_sha256,
        code_revision=code_revision,
        metrics_schema=metrics_schema,
        files=files,
    )

    card_data = build_card_data(
        kind=kind,
        region=region,
        tier=tier or "other",
        eval_receipt=eval_receipt,
        quantized=quant_receipt is not None,
    )

    placeholder_referent = PLACEHOLDER_REFERENT.get(region, "(no design-doc referent recorded)")

    template_kwargs: dict[str, Any] = {
        "region": region,
        "region_cfg": region_cfg,
        "role": region_cfg.get("role", "(no role recorded)"),
        "router_trigger": region_cfg.get("router_trigger", "(none recorded)"),
        "modality": region_cfg.get("modality", "(not recorded)"),
        "licence_line": licence_line,
        "methodology_doc": METHODOLOGY_DOC,
        "tables_md": tables_md,
        "footnotes_md": footnotes_md,
        "sizes_md": sizes_md,
        "provenance_md": provenance_md,
        "parameters": sizes.parameters,
        "code_revision": code_revision or "(none recorded)",
        "checkpoint_sha256": checkpoint_sha256 or "(no checkpoint on this card)",
        "metrics_schema": metrics_schema,
        "placeholder_referent": placeholder_referent,
        "has_eval": eval_receipt is not None,
        "has_quant": quant_receipt is not None,
        "kind": kind,
        # `region_main` only (CARD SPEC: "same plus release tag and 'how this was
        # chosen'"). Read from `region_cfg` -- the same extension-field pattern as
        # `licence_tier`/`licence_why` -- rather than a dedicated `render_card`
        # parameter, so every kind shares one signature.
        "release_tag": region_cfg.get("release_tag", "(not tagged)"),
        "how_chosen": region_cfg.get(
            "how_chosen",
            "(not recorded -- set region_cfg['how_chosen'] before publishing a region_main card)",
        ),
        # `memory` only (CARD SPEC: "region with the OD-17 status stated").
        "od17_status": region_cfg.get("od17_status", MEMORY_OD17_STATUS_DEFAULT),
        # Optional, printed in the header line when supplied -- see this function's own
        # `repo` parameter doc above. `None` renders nothing (every template guards it
        # with `{% if repo %}`), which is why the golden snapshot (built with no `repo`
        # argument) is unaffected by this field existing.
        "repo": repo,
    }

    card = ModelCard.from_template(
        card_data, template_path=str(_template_path(kind)), **template_kwargs
    )
    return str(card)
