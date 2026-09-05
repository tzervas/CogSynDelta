"""`huggingface_hub.ModelCardData` construction, including model-index `EvalResult`s.

CARD SPEC (operator, 2026-09-04): "Front matter via huggingface_hub.ModelCardData ...
tags (cogsyndelta, region:<name>, csd-ptq-v1 when packed), language en, datasets =
corpus catalogue ids when known, model_name, and eval_results=[EvalResult(...)] under
the v2 metric names so the Hub sidebar shows 'Evaluation results'; verified=False."

Only the fp32 EVAL receipt's `metrics` dict feeds `eval_results` -- a training
receipt's `held_out`/`untrained_baseline` numbers are read against a 512-item closed
pool under a different battery (`train_holdout`, not `eval_holdout`; see
`docs/design/METRICS-METHODOLOGY.md` §4) and are not what a Hub visitor comparing
models across the site expects "Evaluation results" to mean.

The model-index is not exempt from either rule `cogsyndelta.cards.tables` already
applies to the human-readable tables: `rank.map`/`rank.precision@10` are RETIRED
(`docs/design/METRICS-METHODOLOGY.md` Sec 13) as independently displayed values --
a model-index entry is exactly that, the Hub renders it in the metrics sidebar and on
leaderboards -- so they are dropped here the same way `build_eval_tables` drops them
from its rank table, never published as `EvalResult`s. And every OTHER key this
function is about to publish must have a `METRIC_METHODOLOGY` entry
(`require_documented`, the same refuse-closed check every table-building function in
`cogsyndelta.cards.tables` already runs) -- the front matter must not carry a number
the card's own body would refuse to print.
"""

from __future__ import annotations

import re
from typing import Any

from huggingface_hub import EvalResult, ModelCardData

from cogsyndelta.cards.methodology import (
    RETIRED_RANK_METRICS,
    MetricMethodology,
    methodology_for_region,
    methodology_key,
    require_documented,
)
from cogsyndelta.regions.aliases import canonical_region

PIPELINE_TAG = "feature-extraction"
"""Every region and the composed mind are text (or text+visual) encoders -- see the
region config's own `role` field for what each one is *for* -- never a text-generation
pipeline, whatever `kind` of card this is."""

LIBRARY_NAME = "cogsyndelta"


# Hub dataset ids are `owner/name`. Text receipts often name parquet files
# (`train-00000-of-00001.parquet`); those must never become `datasets:`.
_HUB_DATASET_ID = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_NOT_A_DATASET_SUFFIX = (".parquet", ".json", ".jsonl", ".csv", ".zip", ".pt", ".arrow")


def _hub_shaped_dataset_id(ds_id: str) -> bool:
    """True only for `owner/name` catalogue ids, never a shard filename."""
    if not _HUB_DATASET_ID.fullmatch(ds_id):
        return False
    return not ds_id.lower().endswith(_NOT_A_DATASET_SUFFIX)


def datasets_from_train_receipt(train_receipt: dict[str, Any] | None) -> list[str] | None:
    """Corpus catalogue ids for the card front matter, derived from the training
    receipt's `corpus.shards` landing names (`nyuuzyou__pxhere/...` ->
    `nyuuzyou/pxhere`). Only Hub-shaped `owner/name` ids are kept -- a parquet
    shard basename is not a dataset id. `None` when nothing Hub-shaped remains.
    """
    if not train_receipt:
        return None
    corpus = train_receipt.get("corpus")
    if not isinstance(corpus, dict):
        return None
    shards = corpus.get("shards")
    if not isinstance(shards, list) or not shards:
        return None
    seen: set[str] = set()
    out: list[str] = []
    for shard in shards:
        landing = str(shard).split("/", 1)[0].strip()
        if not landing:
            continue
        ds_id = landing.replace("__", "/", 1)
        if not _hub_shaped_dataset_id(ds_id):
            continue
        if ds_id not in seen:
            seen.add(ds_id)
            out.append(ds_id)
    return out or None


def region_repo_tags(region: str, *, quantized: bool) -> list[str]:
    """`["cogsyndelta", "region:<region>"]`, plus `"csd-ptq-v1"` when a packed
    quantized artifact is part of this publish (CARD SPEC: "csd-ptq-v1 when packed").
    """
    tags = [LIBRARY_NAME, f"region:{region}"]
    if quantized:
        tags.append("csd-ptq-v1")
    return tags


def _visual_set_identity(train_receipt: dict[str, Any] | None, which: str) -> tuple[str, str]:
    """`(dataset_name, dataset_type)` for a visual probe set, from the train receipt."""
    block: dict[str, Any] = {}
    if isinstance(train_receipt, dict):
        key = "held_out" if which == "probe" else "transfer"
        raw = train_receipt.get(key)
        if isinstance(raw, dict):
            block = raw
    if which == "transfer":
        return (
            str(block.get("name") or "fashion-t10k"),
            str(block.get("source") or "zalando/fashion-mnist"),
        )
    return (
        str(block.get("name") or "eurosat-test"),
        str(block.get("source") or "phelber/eurosat-rgb-128"),
    )


def build_eval_results(
    *,
    region: str,
    eval_receipt: dict[str, Any] | None,
    dataset_name: str | None = None,
    methodology: dict[str, MetricMethodology] | None = None,
    train_receipt: dict[str, Any] | None = None,
) -> list[EvalResult]:
    """One `EvalResult` per numeric key in the fp32 eval receipt's `metrics` dict,
    under whatever name that receipt itself carries (v2 canonical, or a v1 name a
    caller has already normalised -- this function does not rename anything; that is
    `cogsyndelta.cards.tables`'s job for the human-readable tables. The Hub's own
    model-index has no v1/v2 concept, so a legacy-named receipt's numbers still land
    under their own field names here, exactly as `cogsyndelta.cards.render`'s
    methodology footnote explains for the tables).

    Two keys are dropped before anything else runs: `rank.map` and
    `rank.precision@10` are RETIRED as independently displayed values (module
    docstring), so they never become an `EvalResult` here, mirroring
    `cogsyndelta.cards.tables.build_eval_tables`'s own drop of the same two keys from
    its rank table -- one page must not publish a retired number in its machine-
    readable half while suppressing it in its human-readable one. Every remaining key
    is then checked against `METRIC_METHODOLOGY` via `require_documented` (raises
    `CardError` naming any key with no documented definition/battery/source) before a
    single `EvalResult` is built, so the front matter cannot publish a number the
    card's own tables would refuse to print.

    `verified=False` on every result: nothing in this pipeline calls the Hub's
    third-party verification API -- these are self-reported receipts, not
    Hub-verified numbers, and the front matter must not claim otherwise.

    Args:
        region: the region name, used only to build the default `dataset_name`
            (`cogsyndelta-<region>-holdout`).
        eval_receipt: the fp32 eval receipt, or `None` for a card with no eval
            receipt -- returns `[]` in that case.
        dataset_name: overrides the eval-results dataset name/type (default:
            `cogsyndelta-<region>-holdout`).
        methodology: overrides `METRIC_METHODOLOGY` for this call only -- a test's
            mutation-proof hook, the same seam `cogsyndelta.cards.tables`'s
            `build_*` functions expose; production callers leave this `None`.
        train_receipt: the training receipt, used on visual cards to name the
            EuroSAT / Fashion probe sets in the model-index (`held_out.name` /
            `transfer.name` and their `source` fields). Ignored for text regions.
    """
    if eval_receipt is None:
        return []
    metrics = eval_receipt.get("metrics", {})
    ds_name = dataset_name or f"cogsyndelta-{region}-holdout"
    ds_type = ds_name.lower().replace(" ", "-")
    numeric_keys = [
        key
        for key, value in metrics.items()
        if not isinstance(value, bool) and isinstance(value, int | float)
    ]
    published_keys = [
        key
        for key in numeric_keys
        if not (key.startswith("rank.") and key.split(".", 1)[1] in RETIRED_RANK_METRICS)
    ]
    table = methodology if methodology is not None else methodology_for_region(region)
    require_documented(
        (methodology_key(k, methodology=table) for k in published_keys),
        methodology=table,
    )
    results: list[EvalResult] = []
    visual = canonical_region(region) == "visual"
    for key in sorted(published_keys):
        if visual and dataset_name is None:
            which = "transfer" if key.startswith("transfer.") else "probe"
            this_name, this_type = _visual_set_identity(train_receipt, which)
        else:
            this_name, this_type = ds_name, ds_type
        results.append(
            EvalResult(
                task_type=PIPELINE_TAG,
                dataset_type=this_type,
                dataset_name=this_name,
                metric_type=key,
                metric_value=metrics[key],
                metric_name=key,
                verified=False,
            )
        )
    return results


def build_card_data(
    *,
    kind: str,
    region: str,
    tier: str,
    eval_receipt: dict[str, Any] | None = None,
    quantized: bool = False,
    datasets: list[str] | None = None,
    dataset_name: str | None = None,
    model_name: str | None = None,
    train_receipt: dict[str, Any] | None = None,
) -> ModelCardData:
    """The card's YAML front matter, as a `ModelCardData` instance.

    Args:
        kind: one of `region_variant`, `region_main`, `memory`, `placeholder`,
            `composed` -- only affects `model_name`'s default and whether region-scoped
            tags are added (a `composed` card has no single `region:` to tag).
        region: the region name (ignored -- but still required -- for `kind="composed"`,
            to keep the signature uniform across every `render_card` call site).
        tier: the licence tier string (`mit`, `cc-by-sa-4.0`, `cc-by-nc-sa-4.0`, ...),
            already resolved by the caller (this module never derives one).
        eval_receipt: the fp32 eval receipt, or `None` for a card with no eval receipt
            (a `placeholder` card, or a checkpoint published before eval ran).
        quantized: whether a packed artifact is part of this publish -- drives the
            `csd-ptq-v1` tag.
        datasets: corpus catalogue ids, when known; omitted (not guessed) otherwise.
        dataset_name: overrides the eval-results dataset name/type (default:
            `cogsyndelta-<region>-holdout`).
        model_name: overrides the default model name
            (`cogsyndelta-region-<region>` / `cogsyndelta` for a composed card).
        train_receipt: forwarded to `build_eval_results` so a visual card's
            model-index can name eurosat-test / fashion-t10k from the receipt.
    """
    tags = ["cogsyndelta"] if kind == "composed" else region_repo_tags(region, quantized=quantized)
    default_name = "cogsyndelta" if kind == "composed" else f"cogsyndelta-region-{region}"
    name = model_name or default_name
    eval_results = build_eval_results(
        region=region,
        eval_receipt=eval_receipt,
        dataset_name=dataset_name,
        train_receipt=train_receipt,
    )
    return ModelCardData(
        license=tier,
        pipeline_tag=PIPELINE_TAG,
        library_name=LIBRARY_NAME,
        tags=tags,
        language="en",
        datasets=datasets or None,
        model_name=name,
        eval_results=eval_results or None,
    )
