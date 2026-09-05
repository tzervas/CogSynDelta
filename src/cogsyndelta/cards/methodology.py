"""The metrics-methodology reference table and the v1 -> v2 receipt-field alias map.

WHY THIS LIVES HERE, NOT IN THE SCRIPT
`docs/design/METRICS-METHODOLOGY.md` is the full reference (formula, file:line, eval
set, comparison rule, caveats) for every metric this project measures. Both
`scripts/csd-publish-checkpoint.py` (the existing publisher, one card per receipt
trio, hand-built markdown) and `cogsyndelta.cards` (this package: a template-driven
card library used by more than one caller) need the same table -- one metric key means
one formula wherever a card prints it, and a second, drifted copy of this table would
silently let the two disagree about what `recall@1` means. MOVED here rather than
duplicated: `scripts/csd-publish-checkpoint.py` now imports `METRIC_METHODOLOGY`,
`QUANT_METRIC_ALIASES_V1` (as `_QUANT_METRIC_ALIASES_V1`, its existing local name) and
`normalize_quant_receipt_v1` from this module instead of defining them locally: same
values, same behaviour, one source of truth. `tests/test_metrics_methodology.py` and
`tests/test_publish_checkpoint.py` patch `mod.METRIC_METHODOLOGY` /
`mod.normalize_quant_receipt_v1` directly (`mod` being the script, loaded via
`SourceFileLoader`) -- that keeps working unchanged, because an `import` binds a name
in the *importing* module's namespace exactly like a local `def`/assignment would;
`monkeypatch.setattr(mod, "METRIC_METHODOLOGY", {})` patches the script's own bound
name, not this module's, so the script's `_methodology_section` (which reads the
module-global `METRIC_METHODOLOGY` at call time) still sees the patched value.

THE ALIAS MAP'S NAME, AND WHY IT DOES NOT IMPORT `cogsyndelta.pipeline.receipt`'S COPY
`cogsyndelta.pipeline.receipt.QUANT_METRIC_ALIASES_V1` already carries the same three
entries, for a different lane's envelope reader. This module does not import that one
(and vice versa): they cover the same v1 quant-receipt fields today, by coincidence of
scope, not by a shared dependency -- keeping them independent means a future edit to
either one (e.g. this lane's `METRIC_METHODOLOGY` growing a fourth quant alias) never
has to reconcile with an import it does not otherwise need. `QUANT_METRIC_ALIASES_V1`
is exported here (not underscored) because, unlike the script's former local copy,
this module is a public library other callers (`cogsyndelta.cards.tables`, tests) read
directly.

Full reference, source of every prose statement below: `docs/design/METRICS-METHODOLOGY.md`.
"""

from __future__ import annotations

from typing import Any, NamedTuple

METHODOLOGY_DOC = "docs/design/METRICS-METHODOLOGY.md"
"""Repo-relative path to the methodology reference, named in every card. Not a hyperlink
-- a card is read from a private HF repo that does not carry this file, so a relative
link would 404; naming the path matches how the rest of this project's docs point at
each other (e.g. `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`)."""


class CardError(Exception):
    """Raised by `cogsyndelta.cards` whenever a card would otherwise ship a number with
    no stated formula, or numbers from receipts whose `metrics_schema` stamps disagree.
    Refuse-closed, mirroring `scripts/csd-publish-checkpoint.py`'s own
    `PublishAbortError` for the same class of defect -- a different exception type
    because this package has no dependency on that script (or vice versa), not because
    the failure mode differs.
    """


class MetricMethodology(NamedTuple):
    """One line of 'how this number was produced', for one metric-table key."""

    definition: str
    """Short name of the formula -- what METRICS-METHODOLOGY.md's own (b)/(c) sections
    spell out in full."""
    battery: str
    """Which measurement pass produced it, in prose. Two metrics with the same name
    from a different battery are not comparable -- see METRICS-METHODOLOGY.md §4."""
    source: str
    """The file the formula is implemented in, repo-relative."""
    battery_id: str = ""
    """The g7-latent-eval-metrics.md §3.3 canonical battery id (`train_holdout`,
    `eval_holdout`, `eval_quantized_holdout`, `train_graded`, `train_token_rank`,
    `quant_plan`, ...) `compare()` requires two numbers to share before diffing them --
    stricter and machine-checkable where `battery` above is prose for a human reader.
    Empty for a key that is not itself a comparable measurement (a contamination
    guard's report field, a config input like `tolerance`) rather than a battery this
    project runs."""
    pooling: str = ""
    """The g7 §3.3 canonical pooling this metric was computed over (`anchor`,
    `matched`, `pooled_both`, `anchor_pooled`, `anchor_token_global`, `graded_left`,
    `fiqa_corpus`, `fiqa_split`) -- `compare()` requires this to match too. Empty for
    the same reason `battery_id` can be empty above."""


# Every metric-table row key a card (script or library) can print, mapped to where its
# formula and battery are documented in full. Keyed by the BARE row name (e.g.
# "recall@1", not "held_out.recall@1") -- the same key means the same formula wherever
# it appears in a training receipt (held_out / untrained_baseline / beats_untrained all
# read `evaluate()`'s output), so one entry covers all three sections. Extend this
# whenever a caller starts printing a new key, or the refuse-checks below abort.
METRIC_METHODOLOGY: dict[str, MetricMethodology] = {
    "n_pairs": MetricMethodology(
        "size of the closed held-out pool this row's numbers were computed over",
        "training held-out battery",
        "src/cogsyndelta/regions/pretrain.py",
        battery_id="train_holdout",
        pooling="matched",
    ),
    "recall@1": MetricMethodology(
        "recall@k (k=1): fraction of queries whose matched positive is the top-scored "
        "candidate in the closed held-out pool",
        "training held-out battery",
        "src/cogsyndelta/eval/metrics.py",
        battery_id="train_holdout",
        pooling="matched",
    ),
    "recall@10": MetricMethodology(
        "recall@k (k=10): fraction of queries whose matched positive is in the top-10 "
        "of the closed held-out pool",
        "training held-out battery",
        "src/cogsyndelta/eval/metrics.py",
        battery_id="train_holdout",
        pooling="matched",
    ),
    "mrr": MetricMethodology(
        "mean reciprocal rank of the matched positive over the closed held-out pool",
        "training held-out battery",
        "src/cogsyndelta/eval/metrics.py",
        battery_id="train_holdout",
        pooling="matched",
    ),
    "emb_std": MetricMethodology(
        "per-feature embedding std, averaged over features, anchor side only (the collapse signal)",
        "training held-out battery",
        "src/cogsyndelta/regions/pretrain.py",
        battery_id="train_holdout",
        pooling="anchor",
    ),
    "spearman": MetricMethodology(
        "Spearman rank correlation (Pearson over average ranks) between predicted "
        "cosine similarity and the graded corpus's human score",
        "training held-out battery, graded set",
        "src/cogsyndelta/eval/metrics.py",
        battery_id="train_graded",
        pooling="graded_left",
    ),
    # `beats_untrained_eval` (g7 §3.2): the RENAME. `beats_untrained` (below, kept for
    # a receipt on disk before the rename) named the exact same predicate.
    "beats_untrained_eval": MetricMethodology(
        "rank.recall@1 (eval battery) > the training receipt's untrained_baseline "
        "recall@1, unmargined -- a different, simpler predicate than the training "
        "receipt's own beats_untrained_train gate, which is why g7 gives the two "
        "separate names instead of sharing 'beats_untrained' across receipt kinds. "
        "Visual eval receipts reuse this gate name for probe.top1 > "
        "untrained_baseline.top1 (METRICS-METHODOLOGY.md §21), still unmargined; "
        "H1's +0.01 margin is operator-side, not this gate",
        "eval battery",
        "scripts/csd-benchmark.py",
        battery_id="eval_holdout",
        pooling="matched",
    ),
    "beats_untrained": MetricMethodology(
        "LEGACY name for beats_untrained_eval (pre-g7 eval receipts); "
        "rank.recall@1 (eval battery) > the training receipt's untrained_baseline "
        "recall@1, unmargined",
        "eval battery (legacy key)",
        "scripts/csd-benchmark.py",
        battery_id="eval_holdout",
        pooling="matched",
    ),
    "not_anisotropic": MetricMethodology(
        "LEGACY: repr.anisotropy < 0.9, on a receipt written before this was DEMOTED "
        "from a gating admission test to a recorded value only (g7 §3.2 -- no bound "
        "was ever backed by a study; see repr.anisotropy's own entry for the recorded "
        "number). A receipt written after the demotion no longer prints this key.",
        "eval battery (legacy key, no longer a gate)",
        "scripts/csd-benchmark.py",
        battery_id="eval_holdout",
        pooling="pooled_both",
    ),
    "uses_its_dimensions": MetricMethodology(
        "repr.effective_rank_entropy_ratio > 0.05 -- the 0.05 floor is unchanged; only "
        "the metric name changed (g7 §3.2: was repr.effective_rank_ratio, renamed to "
        "disambiguate from the participation-ratio rank ratio a training receipt's "
        "token_aware.final_block_rank reports, METRICS-METHODOLOGY.md §9)",
        "eval battery",
        "scripts/csd-benchmark.py",
        battery_id="eval_holdout",
        pooling="pooled_both",
    ),
    "anisotropy": MetricMethodology(
        "mean cosine similarity between random (off-diagonal) pairs, anchors+positives "
        "pooled -- a representation-geometry diagnostic, NOT a quality score",
        "eval battery",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
        pooling="pooled_both",
    ),
    "alignment": MetricMethodology(
        "mean squared distance between MATCHED pairs (Wang & Isola); read only "
        "together with uniformity, never alone",
        "eval battery",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
        # NOT pooled_both, unlike the rest of this receipt's representation group --
        # matched anchor/positive pairs specifically (MM §3.10(d)).
        pooling="matched",
    ),
    "uniformity": MetricMethodology(
        "log mean Gaussian potential over all pairs (Wang & Isola); read only together "
        "with alignment, never alone",
        "eval battery",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
        pooling="pooled_both",
    ),
    # `effective_rank_entropy` (g7 §3.1): the RENAME (was bare `effective_rank`, kept
    # below for a receipt on disk before the rename -- src/cogsyndelta/eval/benchmark.py
    # writes `repr.effective_rank_entropy` today; card table builders strip the `repr.`
    # prefix so this table is keyed on the bare name either way).
    "effective_rank_entropy": MetricMethodology(
        "Shannon entropy of the normalised singular-value spectrum, exponentiated -- "
        "the ENTROPY definition, not the participation-ratio one training receipts "
        "report under token_aware.final_block_rank (see METRICS-METHODOLOGY.md §9). "
        "Renamed from effective_rank to name which of this project's three 'effective "
        "rank' definitions it is.",
        "eval battery",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
        pooling="pooled_both",
    ),
    "effective_rank": MetricMethodology(
        "LEGACY name for effective_rank_entropy (pre-g7 eval receipts); Shannon entropy "
        "of the normalised singular-value spectrum, exponentiated -- the ENTROPY "
        "definition, not the participation-ratio one training receipts report under "
        "token_aware.final_block_rank (see METRICS-METHODOLOGY.md §9)",
        "eval battery (legacy key)",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
        pooling="pooled_both",
    ),
    "emb_std_anchor": MetricMethodology(
        "mean per-feature embedding std, anchor side only (the collapse signal) -- the "
        "eval-battery counterpart to a training receipt's held_out.emb_std (g7 §3.1: "
        "same formula, ANCHOR-only pooling, never pooled_both like the rest of this "
        "receipt's representation group)",
        "eval battery",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
        pooling="anchor",
    ),
    "dimensions": MetricMethodology(
        "raw embedding width",
        "eval battery",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
        pooling="pooled_both",
    ),
    # `effective_rank_entropy_ratio` (g7 §3.2): the RENAME (was `effective_rank_ratio`,
    # kept below for a receipt on disk before the rename).
    "effective_rank_entropy_ratio": MetricMethodology(
        "effective_rank / dimensions -- how much of the available space is actually "
        "used. Renamed from effective_rank_ratio to name which of this project's three "
        "'effective rank' definitions it is (METRICS-METHODOLOGY.md §9: the entropy "
        "one, never the participation-ratio one).",
        "eval battery",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
        pooling="pooled_both",
    ),
    "effective_rank_ratio": MetricMethodology(
        "LEGACY name for effective_rank_entropy_ratio (pre-g7 eval receipts); "
        "effective_rank / dimensions",
        "eval battery (legacy key)",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
        pooling="pooled_both",
    ),
    # Contamination -- current (multi-channel) receipt shape. Not itself a pooled
    # retrieval quantity `compare()`'s pooling enum has a slot for; `battery_id` is
    # recorded (the guard runs against the training holdout) and `pooling` left empty.
    "train_pairs_seen": MetricMethodology(
        "training pairs streamed past the contamination guard",
        "contamination guard",
        "src/cogsyndelta/eval/metrics.py",
        battery_id="train_holdout",
    ),
    "train_pairs_removed": MetricMethodology(
        "training rows dropped for colliding with the held-out set on a GATED channel "
        "(pair_exact or pair_content) -- a repair; the channel figures below are "
        "measured BEFORE this removal",
        "contamination guard",
        "src/cogsyndelta/eval/metrics.py",
        battery_id="train_holdout",
    ),
    "eval_pairs": MetricMethodology(
        "size of the held-out set the contamination guard indexed",
        "contamination guard",
        "src/cogsyndelta/eval/metrics.py",
        battery_id="train_holdout",
    ),
    "gated_channels": MetricMethodology(
        "which of the six overlap channels cause training-row removal "
        "(pair_exact, pair_content); the rest are reported only",
        "contamination guard",
        "src/cogsyndelta/eval/metrics.py",
        battery_id="train_holdout",
    ),
    "channels": MetricMethodology(
        "per-channel overlap counts and fractions -- see METRICS-METHODOLOGY.md §6.2 "
        "for which channels are gated vs. merely reported",
        "contamination guard",
        "src/cogsyndelta/eval/metrics.py",
        battery_id="train_holdout",
    ),
    "eval_duplicate_positives": MetricMethodology(
        "held-out pairs sharing a positive with another held-out pair -- caps recall@1 "
        "below 1.0 by construction when nonzero",
        "contamination guard",
        "src/cogsyndelta/eval/metrics.py",
        battery_id="train_holdout",
    ),
    # Contamination -- legacy (single-key) receipt shape; still read by this project.
    "train_unique": MetricMethodology(
        "unique training texts under whitespace/case normalisation "
        "(legacy single-key contamination shape)",
        "contamination guard (legacy)",
        "src/cogsyndelta/eval/metrics.py",
        battery_id="train_holdout",
    ),
    "eval_unique": MetricMethodology(
        "unique held-out texts under whitespace/case normalisation "
        "(legacy single-key contamination shape)",
        "contamination guard (legacy)",
        "src/cogsyndelta/eval/metrics.py",
        battery_id="train_holdout",
    ),
    "overlap": MetricMethodology(
        "held-out texts also present in training under the same normalisation "
        "(legacy single-key contamination shape)",
        "contamination guard (legacy)",
        "src/cogsyndelta/eval/metrics.py",
        battery_id="train_holdout",
    ),
    "eval_fraction_contaminated": MetricMethodology(
        "overlap / eval_unique",
        "contamination guard",
        "src/cogsyndelta/eval/metrics.py",
        battery_id="train_holdout",
    ),
    # Quantization. `quant.*` keys are the g7 §3.1 renames this lane owns; the bare
    # legacy keys (`quantized_metric`, `drop`, `compression_ratio`) are kept so a quant
    # receipt written before the rename -- normalised to carry BOTH names by
    # `normalize_quant_receipt_v1` below -- still resolves every key a card might
    # print, old or new.
    "fp32_metric_recomputed": MetricMethodology(
        "task metric measured fresh on the loaded fp32 checkpoint -- recall@1 on the "
        "training held-out battery for text (NOT the eval battery's rank.recall@1; "
        "see METRICS-METHODOLOGY.md §4), or EuroSAT linear-probe top-1 for visual "
        "(§12.8.1)",
        "training held-out battery (quantize stage)",
        "scripts/csd-quantize.py",
        battery_id="train_holdout",
        pooling="matched",
    ),
    "quant.plan_recall@1": MetricMethodology(
        "recall@1 measured on the IN-MEMORY dequantized plan, before the packed "
        "artifact is ever written to disk -- a claim about the plan, not about the "
        "published bytes (see METRICS-METHODOLOGY.md §4). Compare against "
        "quant.artifact_recall@1 ONLY as the plan-vs-artifact sameness guard on the "
        "same checkpoint sha/holdout (g7 §3.3's special case) -- never against rank.* "
        "or repr.* from the eval battery.",
        "quant_plan battery (quantize stage)",
        "scripts/csd-quantize.py",
        battery_id="quant_plan",
        pooling="matched",
    ),
    "quantized_metric": MetricMethodology(
        "LEGACY name for quant.plan_recall@1 (pre-g7 quant receipts); recall@1 "
        "measured on the IN-MEMORY dequantized plan, before the packed artifact is "
        "ever written to disk",
        "quant_plan battery (quantize stage, legacy key)",
        "scripts/csd-quantize.py",
        battery_id="quant_plan",
        pooling="matched",
    ),
    "quant.artifact_recall@1": MetricMethodology(
        "recall@1 measured on the PACKED artifact read back off disk (kind="
        "eval-quantized) -- the byte-verified counterpart to quant.plan_recall@1, and "
        "the only field this receipt shares a comparable formula with across the "
        "quant_plan / eval_quantized_holdout battery boundary (MM §4)",
        "eval_quantized_holdout battery",
        "scripts/csd-benchmark.py",
        battery_id="eval_quantized_holdout",
        pooling="matched",
    ),
    "quant.drop_recall@1": MetricMethodology(
        "fp32_metric_recomputed - quant.plan_recall@1, one named metric on one named "
        "battery (g7 §3.1)",
        "quant_plan battery (quantize stage)",
        "scripts/csd-quantize.py",
        battery_id="quant_plan",
        pooling="matched",
    ),
    "quant.plan_probe_top1": MetricMethodology(
        "EuroSAT official-test linear-probe top-1 measured on the IN-MEMORY "
        "dequantized EMA target encoder, before the packed artifact is written "
        "(METRICS-METHODOLOGY.md §12.8.1). Not closed-pool recall@1.",
        "EuroSAT test linear probe, n_eval=5400, 10-way, chance 0.1 (quantize stage)",
        "scripts/csd-quantize.py",
        battery_id="quant_plan",
        pooling="linear_probe",
    ),
    "quant.artifact_probe_top1": MetricMethodology(
        "EuroSAT official-test linear-probe top-1 measured on the PACKED "
        "target_encoder artifact read back off disk (kind=eval-quantized). The "
        "byte-verified counterpart to quant.plan_probe_top1.",
        "EuroSAT test linear probe, n_eval=5400 (eval-quantized)",
        "scripts/csd-benchmark.py",
        battery_id="eval_quantized_holdout",
        pooling="linear_probe",
    ),
    "quant.drop_probe_top1": MetricMethodology(
        "fp32_metric_recomputed - quant.plan_probe_top1, one named metric on the "
        "EuroSAT linear-probe battery",
        "EuroSAT test linear probe (quantize stage)",
        "scripts/csd-quantize.py",
        battery_id="quant_plan",
        pooling="linear_probe",
    ),
    "drop": MetricMethodology(
        "LEGACY name for quant.drop_recall@1 (pre-g7 quant receipts); "
        "fp32_metric_recomputed - quantized_metric",
        "quant_plan battery (quantize stage, legacy key)",
        "scripts/csd-quantize.py",
        battery_id="quant_plan",
        pooling="matched",
    ),
    "tolerance": MetricMethodology(
        "largest acceptable absolute drop in the task metric -- a configured input, "
        "not a measurement",
        "quantize stage configuration",
        "scripts/csd-quantize.py",
        battery_id="quant_plan",
    ),
    "within_budget": MetricMethodology(
        "quant.drop_recall@1 <= tolerance (text) or quant.drop_probe_top1 <= "
        "tolerance (visual, METRICS-METHODOLOGY.md §12.8.1)",
        "quant_plan battery (quantize stage)",
        "scripts/csd-quantize.py",
        battery_id="quant_plan",
        pooling="matched",
    ),
    "quant.compression_ratio": MetricMethodology(
        "fp32_bytes / stored_bytes -- a PAYLOAD/STORAGE ratio, NOT a speed or "
        "throughput claim (renamed from compression_ratio, g7 §3.1)",
        "quant/ptq.py byte accounting",
        "src/cogsyndelta/quant/ptq.py",
        battery_id="quant_plan",
    ),
    "compression_ratio": MetricMethodology(
        "LEGACY name for quant.compression_ratio (pre-g7 quant receipts); "
        "fp32_bytes / stored_bytes -- a PAYLOAD/STORAGE ratio, NOT a speed or "
        "throughput claim",
        "quant/ptq.py byte accounting (legacy key)",
        "src/cogsyndelta/quant/ptq.py",
        battery_id="quant_plan",
    ),
    "fp32_bytes": MetricMethodology(
        "sum(parameter.numel() * 4) -- weights only, never optimizer or RNG state",
        "quant/ptq.py byte accounting",
        "src/cogsyndelta/quant/ptq.py",
        battery_id="quant_plan",
    ),
    "stored_bytes": MetricMethodology(
        "packed codes + per-channel scale/zero-point for quantized tensors, plus 4 "
        "bytes/element for fp32-kept tensors",
        "quant/ptq.py byte accounting",
        "src/cogsyndelta/quant/ptq.py",
        battery_id="quant_plan",
    ),
    # Eval-battery `rank.*`/`eff.*` families this library's `cards.tables` prints in
    # full (the script above never did -- it only ever read `repr.*` out of an eval
    # receipt's `metrics` dict). `rank.map` and `rank.precision@10` are deliberately
    # ABSENT here: `docs/design/METRICS-METHODOLOGY.md` §13 retires both as
    # independently displayed columns (MAP==MRR and precision@10==recall@10/10 by
    # construction on this single-relevant-item closed pool) -- kept only as an
    # internal sameness guard, never rendered; `cogsyndelta.cards.tables` drops them
    # from a legacy receipt's table rather than documenting them here.
    "candidates": MetricMethodology(
        "size of the closed pool this eval-battery pass ranked against -- the eval "
        "receipt's own count, independent of the training receipt's held_out.n_pairs",
        "eval battery",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
        pooling="matched",
    ),
    "recall@5": MetricMethodology(
        "recall@k (k=5): fraction of queries whose matched positive is in the top-5 "
        "of the closed eval pool",
        "eval battery",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
        pooling="matched",
    ),
    "ndcg@10": MetricMethodology(
        "normalised discounted cumulative gain at 10, single relevant item per query",
        "eval battery",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
        pooling="matched",
    ),
    "parameters": MetricMethodology(
        "parameter count, as passed into benchmark_embeddings() -- the same count a "
        "training receipt's top-level `parameters` field reports, re-stated alongside "
        "the eval battery",
        "eval battery",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
    ),
    "stored_mb": MetricMethodology(
        "stored_bytes / 1e6 -- what the model actually occupies AT THIS EVAL PASS "
        "(post-quantization for an eval-quantized receipt, fp32 for a plain eval "
        "receipt); the disk-footprint half of quant.compression_ratio, not a separate "
        "measurement",
        "eval battery",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
    ),
    "capability_per_param": MetricMethodology(
        "rank.recall@1 / max(1e-9, parameters / 1e6) -- the thesis metric: capability "
        "per million parameters, not capability per byte actually shipped (see "
        "capability_per_mb for that)",
        "eval battery",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
    ),
    "capability_per_mb": MetricMethodology(
        "rank.recall@1 / max(1e-9, stored_bytes / 1e6) -- capability per MB actually "
        "shipped; the more honest sibling of capability_per_param, since parameters "
        "are not what ships",
        "eval battery",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
    ),
    "latency_p50_ms": MetricMethodology(
        "median wall-clock latency per encode call, from profile_latency()",
        "eval battery, latency profile",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
    ),
    "latency_p95_ms": MetricMethodology(
        "95th-percentile wall-clock latency per encode call, from profile_latency()",
        "eval battery, latency profile",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
    ),
    "latency_p99_ms": MetricMethodology(
        "99th-percentile wall-clock latency per encode call, from profile_latency()",
        "eval battery, latency profile",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
    ),
    "throughput_per_s": MetricMethodology(
        "encode calls per wall-clock second, from profile_latency() -- inverse of "
        "mean latency over the profiled runs, not of any single percentile above",
        "eval battery, latency profile",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
    ),
    "peak_vram_mb": MetricMethodology(
        "torch.cuda.max_memory_allocated() / 1e6 over the profiled encode calls; 0.0 "
        "on a CPU-only run (this project's tests and CI run CPU-only -- a 0.0 here is "
        "the profiler never having seen a CUDA device, not a measured zero)",
        "eval battery, latency profile",
        "src/cogsyndelta/eval/benchmark.py",
        battery_id="eval_holdout",
    ),
    # Visual I-JEPA linear-probe battery (METRICS-METHODOLOGY.md §21). Train
    # `held_out` uses the bare names; eval receipts use `probe.*` / `transfer.*`
    # / `repr.rep_std`. `methodology_key` keeps a full `repr.rep_std` key when
    # that exact name is in this table, so the eval battery is not footnoted as
    # the train battery.
    "top1": MetricMethodology(
        "EuroSAT official-test linear-probe top-1 on frozen EMA-target-encoder "
        "latents (10-way, chance 0.1). Not closed-pool recall@1",
        "training EuroSAT linear probe, n_eval=5400 (primary)",
        "src/cogsyndelta/regions/vl_pretrain.py",
        battery_id="train_holdout",
        pooling="linear_probe",
    ),
    "top5": MetricMethodology(
        "EuroSAT official-test linear-probe top-5 on frozen EMA-target-encoder "
        "latents. Not closed-pool recall@5",
        "training EuroSAT linear probe, n_eval=5400 (primary)",
        "src/cogsyndelta/regions/vl_pretrain.py",
        battery_id="train_holdout",
        pooling="linear_probe",
    ),
    "n_eval": MetricMethodology(
        "number of labelled eval images the linear probe was scored on (EuroSAT "
        "test 5400 on the primary set; Fashion t10k 2000 on the transfer set)",
        "training linear-probe battery",
        "src/cogsyndelta/regions/vl_pretrain.py",
        battery_id="train_holdout",
        pooling="linear_probe",
    ),
    "rep_std": MetricMethodology(
        "mean per-feature std of EMA-target-encoder latents on a mixed I-JEPA "
        "train batch -- the visual collapse signal (train-receipt held_out.rep_std)",
        "training collapse diagnostic",
        "src/cogsyndelta/regions/vl_pretrain.py",
        battery_id="train_holdout",
        pooling="anchor",
    ),
    "probe.top1": MetricMethodology(
        "EuroSAT official-test linear-probe top-1 on the loaded checkpoint's "
        "deployed EMA target encoder. Same formula as train held_out.top1; eval "
        "battery, not train_holdout. Do not compare to rank.recall@1",
        "eval EuroSAT linear probe, n_eval=5400, 10-way, chance 0.1",
        "scripts/csd-benchmark.py",
        battery_id="eval_holdout",
        pooling="linear_probe",
    ),
    "probe.top5": MetricMethodology(
        "EuroSAT official-test linear-probe top-5 on the loaded checkpoint's "
        "deployed EMA target encoder. Same formula as train held_out.top5",
        "eval EuroSAT linear probe, n_eval=5400",
        "scripts/csd-benchmark.py",
        battery_id="eval_holdout",
        pooling="linear_probe",
    ),
    "transfer.top1": MetricMethodology(
        "Fashion-MNIST t10k linear-probe top-1 (probe role transfer, n_eval=2000). "
        "Same probe protocol as EuroSAT primary; a different labelled set. Not H1",
        "eval Fashion t10k transfer probe, n_eval=2000",
        "scripts/csd-benchmark.py",
        battery_id="eval_holdout",
        pooling="linear_probe",
    ),
    "transfer.top5": MetricMethodology(
        "Fashion-MNIST t10k linear-probe top-5 (probe role transfer, n_eval=2000)",
        "eval Fashion t10k transfer probe, n_eval=2000",
        "scripts/csd-benchmark.py",
        battery_id="eval_holdout",
        pooling="linear_probe",
    ),
    "repr.rep_std": MetricMethodology(
        "mean per-feature std of EMA-target-encoder latents on a mixed I-JEPA "
        "train batch -- the visual collapse signal, eval-battery counterpart of "
        "train held_out.rep_std",
        "eval collapse diagnostic",
        "src/cogsyndelta/regions/vl_pretrain.py",
        battery_id="eval_holdout",
        pooling="anchor",
    ),
    "not_collapsed": MetricMethodology(
        "rep_std / untrained_baseline.rep_std >= 0.1 -- visual collapse gate; "
        "the same predicate the training receipt records as collapsed=false",
        "eval battery",
        "scripts/csd-benchmark.py",
        battery_id="eval_holdout",
        pooling="anchor",
    ),
}

#: `rank.*` keys `docs/design/METRICS-METHODOLOGY.md` §13 retires as independently
#: displayed columns (kept only as an internal sameness guard: MAP==MRR and
#: precision@10==recall@10/10 by construction on the single-relevant-item closed
#: pool). `cogsyndelta.cards.tables` drops these from a legacy receipt's table
#: rather than raising `CardError` for them -- they are RETIRED, not undocumented.
RETIRED_RANK_METRICS: frozenset[str] = frozenset({"map", "precision@10"})


# v1 -> v2 field-name aliases for the quant-receipt fields this table owns
# (g7-latent-eval-metrics.md §3.1: `quant.plan_recall@1` WAS `quantized_metric`,
# `quant.compression_ratio` WAS `compression_ratio`, `quant.drop_recall@1` WAS the bare
# `drop`). A quant receipt this project writes NOW (`scripts/csd-quantize.py`) uses the
# v2 names directly; this table is for a v1-shaped quant receipt already on disk (or a
# v1-shaped raw dict a caller still hands this module) so it can be normalised to v2
# field names before anything reads it by name. The project does not yet have one
# shared, all-battery alias map covering the train-receipt gate/battery renames another
# lane owns (`beats_untrained` -> `beats_untrained_train`, `train_holdout`/
# `train_graded`/`train_token_rank` battery ids) -- see `EVAL_METRIC_ALIASES_V1` below
# for the eval-receipt counterpart this package (not the script) owns.
QUANT_METRIC_ALIASES_V1: dict[str, str] = {
    "quantized_metric": "quant.plan_recall@1",
    "compression_ratio": "quant.compression_ratio",
    "drop": "quant.drop_recall@1",
}


def normalize_quant_receipt_v1(receipt: dict[str, Any]) -> dict[str, Any]:
    """Return `receipt` with every `QUANT_METRIC_ALIASES_V1` v1 key ALSO present under
    its v2 name, so every reader downstream of this call (`METRIC_METHODOLOGY`
    lookups, a card's quantization table) can assume v2 names unconditionally, whether
    the receipt on disk was written before or after `scripts/csd-quantize.py` started
    emitting v2 names directly.

    Copies rather than renames in place: a v1-shaped receipt keeps its v1 keys too
    (harmless, and other readers touch other fields on this same dict by name and must
    keep working unmodified). A receipt already carrying the v2 name for a given field
    is left alone -- this never overwrites a value the producer actually wrote.
    """
    out = dict(receipt)
    for old, new in QUANT_METRIC_ALIASES_V1.items():
        if old in out and new not in out:
            out[new] = out[old]
    return out


# v1 -> v2 aliases for the metric/gate keys this package's `cards.tables` reads out of
# an EVAL (or eval-quantized) receipt's `metrics` / `gates` dicts -- the eval-receipt
# counterpart of `QUANT_METRIC_ALIASES_V1` above, owned by `cogsyndelta.cards` (the
# script above never reads an eval receipt's `metrics` dict by anything other than its
# `repr.` family, which it strips and re-keys itself; it has no equivalent table to
# move). Keyed on the FULL `metrics`/`gates` key (`repr.effective_rank`, not the bare
# `effective_rank` `METRIC_METHODOLOGY` above is keyed on) because that is the shape
# `cogsyndelta.cards.tables` reads receipts in.
#
# `not_anisotropic` has no v2 counterpart at all (DEMOTED, g7 §3.2 -- see its
# `METRIC_METHODOLOGY` entry) and so is deliberately absent here: a v1 receipt that
# carries it keeps carrying it, rendered under its own LEGACY methodology entry, never
# aliased to a gate that no longer exists.
EVAL_METRIC_ALIASES_V1: dict[str, str] = {
    "repr.effective_rank": "repr.effective_rank_entropy",
    "repr.effective_rank_ratio": "repr.effective_rank_entropy_ratio",
}

#: The `gates` dict counterpart of `EVAL_METRIC_ALIASES_V1` -- a separate table because
#: `gates` and `metrics` are different dicts on the same eval receipt and a key
#: colliding between them (there are none today) must not be aliased by the wrong table.
EVAL_GATE_ALIASES_V1: dict[str, str] = {
    "beats_untrained": "beats_untrained_eval",
}


#: Category prefixes `METRIC_METHODOLOGY` keys on the BARE name (strip the prefix
#: before looking a key up) -- the convention `scripts/csd-publish-checkpoint.py`'s
#: `build_card` already used for `repr.*` (its `{k[len("repr."):]: v ...}` dict
#: comprehension), extended here to the `rank.*`/`eff.*` families this library also
#: prints. `quant.*` is deliberately ABSENT: those keys (`quant.plan_recall@1`,
#: `quant.compression_ratio`, `quant.drop_recall@1`, `quant.artifact_recall@1`) are
#: stored in `METRIC_METHODOLOGY` WITH their prefix, because g7 §3.1 minted them as
#: first-class v2 names, not as a family-prefixed shorthand for a bare concept shared
#: with another family.
_BARE_KEY_PREFIXES = frozenset({"rank", "eff", "repr"})


def methodology_key(key: str, *, methodology: dict[str, MetricMethodology] | None = None) -> str:
    """The `METRIC_METHODOLOGY` dict key a receipt field `key` resolves to: the bare
    name for `rank.*`/`eff.*`/`repr.*`, or `key` unchanged for everything else
    (`quant.*`, `probe.*`, `transfer.*`, and every prefix-free key -- gates, held_out,
    contamination, quant plain fields).

    An exact match in the table wins before prefix-stripping: visual eval writes
    `repr.rep_std` as its own eval-battery metric, while the train receipt's
    `held_out.rep_std` is the train-battery counterpart -- stripping first would
    footnote both as the same battery.
    """
    table = METRIC_METHODOLOGY if methodology is None else methodology
    if key in table:
        return key
    prefix, sep, rest = key.partition(".")
    return rest if sep and prefix in _BARE_KEY_PREFIXES else key


def require_documented(
    keys: Any, *, methodology: dict[str, MetricMethodology] | None = None
) -> None:
    """Raise `CardError` naming every key in `keys` that has no `METRIC_METHODOLOGY`
    entry. `methodology` defaults to the module-level table; a caller's test passes a
    stubbed/trimmed dict to prove this refusal is load-bearing (mutation proof).
    """
    table = METRIC_METHODOLOGY if methodology is None else methodology
    missing = sorted({k for k in keys if k not in table})
    if missing:
        raise CardError(
            f"card would print metric key(s) {missing} with no entry in "
            "METRIC_METHODOLOGY -- refusing to render a number with no stated "
            f"definition/battery/source. Add an entry (and, if it names a new formula, "
            f"a section to {METHODOLOGY_DOC}) before rendering."
        )
