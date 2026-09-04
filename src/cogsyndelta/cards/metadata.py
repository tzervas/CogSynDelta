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
"""

from __future__ import annotations

from typing import Any

from huggingface_hub import EvalResult, ModelCardData

PIPELINE_TAG = "feature-extraction"
"""Every region and the composed mind are text (or text+visual) encoders -- see the
region config's own `role` field for what each one is *for* -- never a text-generation
pipeline, whatever `kind` of card this is."""

LIBRARY_NAME = "cogsyndelta"


def region_repo_tags(region: str, *, quantized: bool) -> list[str]:
    """`["cogsyndelta", "region:<region>"]`, plus `"csd-ptq-v1"` when a packed
    quantized artifact is part of this publish (CARD SPEC: "csd-ptq-v1 when packed").
    """
    tags = [LIBRARY_NAME, f"region:{region}"]
    if quantized:
        tags.append("csd-ptq-v1")
    return tags


def build_eval_results(
    *,
    region: str,
    eval_receipt: dict[str, Any] | None,
    dataset_name: str | None = None,
) -> list[EvalResult]:
    """One `EvalResult` per numeric key in the fp32 eval receipt's `metrics` dict,
    under whatever name that receipt itself carries (v2 canonical, or a v1 name a
    caller has already normalised -- this function does not rename anything; that is
    `cogsyndelta.cards.tables`'s job for the human-readable tables. The Hub's own
    model-index has no v1/v2 concept, so a legacy-named receipt's numbers still land
    under their own field names here, exactly as `cogsyndelta.cards.render`'s
    methodology footnote explains for the tables).

    `verified=False` on every result: nothing in this pipeline calls the Hub's
    third-party verification API -- these are self-reported receipts, not
    Hub-verified numbers, and the front matter must not claim otherwise.
    """
    if eval_receipt is None:
        return []
    metrics = eval_receipt.get("metrics", {})
    ds_name = dataset_name or f"cogsyndelta-{region}-holdout"
    ds_type = ds_name.lower().replace(" ", "-")
    results: list[EvalResult] = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        results.append(
            EvalResult(
                task_type=PIPELINE_TAG,
                dataset_type=ds_type,
                dataset_name=ds_name,
                metric_type=key,
                metric_value=value,
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
    """
    tags = ["cogsyndelta"] if kind == "composed" else region_repo_tags(region, quantized=quantized)
    default_name = "cogsyndelta" if kind == "composed" else f"cogsyndelta-region-{region}"
    name = model_name or default_name
    eval_results = build_eval_results(
        region=region, eval_receipt=eval_receipt, dataset_name=dataset_name
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
