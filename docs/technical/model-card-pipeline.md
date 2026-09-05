# The model-card pipeline

**Not one of the ratified thirteen `technical/` chapters.** This is a standalone
operator/reviewer reference for one concrete piece of tooling — how a CogSynDelta
region's model card gets built, and every point at which that build refuses rather
than print a number it cannot stand behind. It follows `docs/adr/`'s precedent (`docs/
README.md` rule 3: "a separate class... citable as evidence... never part of the
ratified track index") rather than the `plain/`↔`technical/` twin scheme: there is no
`plain/` counterpart planned for it, and it carries no `DEC-nn`/`OD-nn` content of its
own to anchor.

**Two independent tools build a card. They do not share a renderer.**

| | `scripts/csd-publish-checkpoint.py` | `scripts/csd-card.py` |
|---|---|---|
| Card format | hand-built markdown (`build_card`) | Jinja2 templates via `cogsyndelta.cards.render_card` |
| Input | `--receipt`/`--eval-receipt`/`--quant-receipt` (explicit paths) | `--cell <dir>` (reads `cell.json`) or explicit `--*-receipt` flags |
| Side effect | uploads to a private HF repo (`HF_TOKEN` required) | writes a local file (`--out`); no network, no token |
| Card kinds | one shape, always | `region_variant` / `region_main` / `memory` / `placeholder` / `composed` |

They are separate on purpose, not because nobody got around to merging them: `build_card`
(`scripts/csd-publish-checkpoint.py:1082`) is the exact byte-for-byte shape every
already-published `tzervas/cogsyndelta-region-*` repo's `README.md` carries today, and
dozens of tests in `tests/test_publish_checkpoint.py` and `tests/test_metrics_
methodology.py` pin that shape — changing it changes what is already live on the Hub.
`cogsyndelta.cards.render_card` (`src/cogsyndelta/cards/render.py:289`) is the newer,
richer library (grouped metric tables, numbered footnotes, MEASURED sizes, five card
`kind`s) that CARD SPEC (operator, 2026-09-04) actually asked for; `csd-card.py` is its
one command-line front end. The two DO share one thing deliberately: the audited
licence table. `csd-card.py` imports `licence_tier()`/`LICENCE_TIER`/`LICENCE_WHY`
straight out of `csd-publish-checkpoint.py` (`scripts/csd-card.py`'s own
`_load_publish_module`) rather than keeping a second copy that could silently name a
different tier for the same region.

## What renders from what

```
train receipt ──┐
eval receipt ────┼──► cogsyndelta.cards.render_card(kind, region, region_cfg, receipts, files, ...)
eval-quantized ──┤            │
quant receipt ───┘            ├─ cogsyndelta.cards.tables      → grouped rank./eff./repr./quant. tables
                               ├─ cogsyndelta.cards.sizes       → parameters, disk bytes, MEASURED VRAM
                               ├─ cogsyndelta.cards.metadata    → ModelCardData + model-index EvalResults
                               └─ templates/<kind>.md.j2        → the rendered markdown
```

`render_card` (`src/cogsyndelta/cards/render.py:289`) is the one public entry point.
`receipts` is `{"train": ..., "eval": ..., "eval_quantized": ..., "quant": ...}`, every
key optional — a `placeholder` card supplies none, a train-only cell supplies one.
`region_cfg` is the region's `config/mind/csd-regions.json` entry, EXTENDED by the
caller with `licence_tier`/`licence_why` (this library never derives a tier itself —
see `render.py`'s own module docstring for why guessing one here would launder the
audited table). `scripts/csd-card.py --cell <cell dir>` builds that `receipts` dict by
reading `<cell dir>/cell.json`'s own `stages.*.receipt` pointers (the exact paths
`model-matrix` recorded having written) and keeping only a stage whose `state` is
`"done"`; `--train-receipt`/`--eval-receipt`/`--eval-quantized-receipt`/`--quant-
receipt` override or supply a receipt directly.

## The faculty-name alias display rule

`region` (the value `--region` names, and the id under which a cell's receipts were
written) can be a legacy id — `code`, `vl_latent` — resolved to its canonical faculty by
`cogsyndelta.regions.aliases.canonical_region` before `region_cfg` is looked up in
`config/mind/csd-regions.json`. The templates never hide that resolution: when
`region_cfg["name"]` (the canonical id) differs from `region` (what was asked for),
`region_main.md.j2` and `region_variant.md.j2` print a **Faculty** line naming the
canonical id and, if the config carries one, its `specialisation` — *"language
(specialisation: `code`) -- this release's receipts were recorded under the legacy
region id `code`"* — rather than silently rendering the card as if the legacy id were
the whole story. A region whose config id already matches `region` and carries a bare
`specialisation` (no rename involved) gets the shorter **Specialisation** line with no
legacy-id note, because there is no alias to disclose. This is what DEC-01/DEC-78
(`docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` §1) require of anything that reads a
region name: resolve through the one alias module, and record which name was actually on
disk.

## Every refusal, and why

A card build refuses (raises, prints nothing) rather than publish a number it cannot
stand behind. Every refusal below has a mutation-proof test — see each file's own test
suite.

| refusal | raised by | when |
|---|---|---|
| undocumented metric | `CardError` (`require_documented`, `src/cogsyndelta/cards/methodology.py:631`) | a receipt carries a metric key with no entry in `METRIC_METHODOLOGY` — no stated formula/battery/source to print beside it |
| `metrics_schema` disagreement | `CardError` (`assert_schemas_agree`, `src/cogsyndelta/cards/tables.py:79`) | the train/eval/quant receipts merged into one card's tables stamp different `metrics_schema` values — `cogsyndelta.cards` refuses outright (stricter than `csd-publish-checkpoint.py`'s own `_metrics_schema_line`, which prints a `train=... eval=... quant=...` breakdown instead, because that script must keep publishing already-trained checkpoints whose receipts predate a schema migration) |
| unresolved licence tier | `CardError` (`_licence_block`, `src/cogsyndelta/cards/render.py:259`) | any `kind` except `placeholder`/`composed` with no `region_cfg["licence_tier"]` set — a `placeholder` (no weights) or `composed` (licence resolution is the operator's call, CARD SPEC §9) renders a stated placeholder line instead |
| unaudited region | `PublishAbortError` (`licence_tier`, `scripts/csd-publish-checkpoint.py:310`) | `--region` has no entry in `LICENCE_TIER` at all — refuses rather than default to MIT |
| BLOCKING region | `PublishAbortError` (`licence_tier`, same function) | `--region visual` (or its legacy alias `vl_latent` — both resolve to the same config entry) — unreleasable as trained (`docs/design/LICENCE-FOR-OPEN-WEIGHTS.md` §4), checked before the tier table is even consulted |
| region mismatch | `CardCliError` (`scripts/csd-card.py`, `_build_receipts_and_region`) | `--region` disagrees with `--cell`'s own `cell.json` `region` field — the licence tier and repo name are derived from `--region`, so a mismatch would launder a receipt's real tier under a different region's name |
| missing training receipt | `CardCliError` (same function) | `--kind region_variant`/`region_main`/`memory` with no train receipt from `--cell` or `--train-receipt` — `placeholder`/`composed` need none |
| unloadable checkpoint under `--safetensors` | `PublishAbortError` (`build_plan`, `scripts/csd-publish-checkpoint.py`) | the checkpoint does not `torch.load()` as a plain state dict (or a `{"state_dict": ...}`-shaped wrapper) — see `cogsyndelta.cards.export._extract_state_dict` |

## The safetensors export

`cogsyndelta.cards.export.export_safetensors(checkpoint)` (`src/cogsyndelta/cards/
export.py:54`) writes `<checkpoint stem>.safetensors` beside a checkpoint — fp32,
weights-only, CPU, contiguous — so the Hub can render its safetensors badge and
model-size sidebar next to the pickled `final.pt` primary artifact (`.safetensors`
files refuse anything but plain tensors by construction; there is no pickle path
through them). It is opt-in on `csd-publish-checkpoint.py` (`--safetensors`, default
off): every real checkpoint this project trains loads cleanly, but nothing guarantees
an arbitrary `--receipt`-named `.pt` file does, and the flag's absence must not change
any existing publish's behaviour.

## Where the matrix harness fits in

`program/matrix/csd-matrix.yaml`'s top-level `publish:` block names both tools:
`commands.publish` (per-region, `csd-publish-checkpoint.py`, the upload) and
`publish.card_command` (region-generic, `csd-card.py --cell {cell_dir} ... --repo
{hub_repo} --out {out}`, the local card render). `publish.branch_prefix` names the
`variants/` convention `branch`/`base_ref` already follow (DESIGN.v2 §4.2). Each
region also carries a `card:` block (`language`, `tags`, `licence_tier`, `datasets`) —
descriptive metadata for a reader of this config, kept in agreement with
`csd-publish-checkpoint.py`'s own `LICENCE_TIER` by `tests/test_matrix_config.py`
(`test_card_licence_tier_agrees_with_the_publish_script`), never a second source of
truth for what a real publish will actually enforce.

## Executed usage snippets

Every card template's "How to use" section that shows loadable Python (`torch.load`,
`load_packed_artifact`/`unpack_state_dict`) marks that fenced block with a preceding
`<!-- exec -->` HTML comment. `tests/test_cards_usage_snippets.py` extracts every such
block off a card's own RENDERED markdown (never a copy of the template kept in the
test) and actually runs it, on CPU, against a tiny deterministic checkpoint committed
at `tests/fixtures/cards/tiny_checkpoint.pt` (+ its packed `.ptq.pt` sibling) — so a
future API rename that the template's prose was never updated for fails a test, not a
user's first `git clone`.
