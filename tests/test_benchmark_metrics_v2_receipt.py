"""`scripts/csd-benchmark.py`'s eval / eval-quantized receipts under the metrics-v2
unification (g7-latent-eval-metrics.md §3.1/§3.2/§3.3), driven against REAL receipts
from a tiny CPU pretrain + quantize + benchmark run (same fixture shape as
`tests/test_benchmark_checkpoint_sha_receipt.py`), not hand-built stand-ins:

- `gates.beats_untrained` -> `gates.beats_untrained_eval` (two predicates, two names --
  MM §1's training-receipt gate is a DIFFERENT rule under the OLD shared name).
- `gates.not_anisotropic` DEMOTED entirely -- no longer printed as a gate; `repr.anisotropy`
  stays a recorded metric.
- `metrics["repr.effective_rank_ratio"]` -> `metrics["repr.effective_rank_entropy_ratio"]`,
  read by the (unrenamed) `gates.uses_its_dimensions` predicate.
- `kind="eval-quantized"` receipts additionally carry `metrics["quant.artifact_recall@1"]`,
  the plan-vs-artifact sameness-guard counterpart to a quant receipt's
  `quant.plan_recall@1` (MM §4's special case, g7 §3.3).
- `provenance.metric_groups`: one entry per `rank`/`eff`/`repr` family, each carrying
  `battery_id` (`eval_holdout` for `kind="eval"`, `eval_quantized_holdout` for
  `kind="eval-quantized"` -- never the same value, even though both run the identical
  code path), `pooling`, and `seed`. Two `repr.*` fields are pooled differently than
  the rest of that family (MM §3.10(d)/§12.5) and so get their own entries keyed by
  their full dotted name -- `repr.emb_std_anchor` (`pooling="anchor"`) and
  `repr.alignment` (`pooling="matched"`) -- rather than inheriting the bare `repr`
  entry's `pooled_both`. A `kind="eval-quantized"` receipt carries a further `quant`
  entry for `quant.artifact_recall@1`, identical to its `rank` entry since the field
  is `rank.recall@1` verbatim, not an independent measurement.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import ClassVar

import pytest

pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from cogsyndelta.eval.metrics import MetricGroup, MetricIdentity, compare
from cogsyndelta.regions.pretrain import PretrainConfig, pretrain_region
from cogsyndelta.regions.text_encoder import TextEncoderConfig

pytestmark = pytest.mark.cpu

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str, filename: str):
    """Import a hyphenated `scripts/*.py` module -- see
    tests/test_benchmark_checkpoint_sha_receipt.py for why this indirection exists."""
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bench = _load("csd_benchmark_metrics_v2_test", "csd-benchmark.py")
quant = _load("csd_quantize_metrics_v2_bench_test", "csd-quantize.py")


def _build_tokenizer(path: Path, n_pairs: int) -> None:
    texts = [f"anchor number {i} word" for i in range(n_pairs)] + [
        f"anchor number {i} match" for i in range(n_pairs)
    ]
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106 -- a token id name, not a secret
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(path))


def _build_pairs_parquet(path: Path, n_pairs: int) -> None:
    anchors = [f"anchor number {i} word" for i in range(n_pairs)]
    positives = [f"anchor number {i} match" for i in range(n_pairs)]
    pq.write_table(pa.table({"anchor": anchors, "positive": positives}), path)


@pytest.fixture
def trained_receipt(tmp_path: Path) -> dict:
    tok_path = tmp_path / "tokenizer.json"
    shard_path = tmp_path / "pairs.parquet"
    _build_tokenizer(tok_path, 40)
    _build_pairs_parquet(shard_path, 40)

    cfg = PretrainConfig(
        region="benchv2-test",
        pair_columns=("anchor", "positive"),
        shards=[str(shard_path)],
        steps=2,
        batch_size=4,
        holdout_pairs=4,
        eval_every=2,
        checkpoint_every=2,
        max_len=16,
        seed=3,
        device="cpu",
        encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=16),
        tokenizer_path=str(tok_path),
        out_dir=str(tmp_path / "receipts"),
    )
    receipt = pretrain_region(cfg)

    def fake_regions_spec() -> dict:
        def _shards(glob_pat: str, root: Path | None = None) -> list[str]:
            return [str(shard_path)]

        class _Entry:
            sources: ClassVar = [("pairs.parquet", ("anchor", "positive"), 0)]
            root = tmp_path

        return {"REGIONS": {}, "_shards": _shards, "region_spec": lambda name: _Entry()}

    bench._regions_spec = fake_regions_spec
    quant._load_regions_spec = fake_regions_spec
    return receipt


@pytest.fixture(autouse=True)
def _restore_regions_spec():
    orig_bench, orig_quant = bench._regions_spec, quant._load_regions_spec
    yield
    bench._regions_spec = orig_bench
    quant._load_regions_spec = orig_quant


# ============================================================ fp32 eval receipt (kind="eval")


def test_fp32_receipt_gates_are_renamed_and_not_anisotropic_is_gone(
    tmp_path: Path, trained_receipt: dict
) -> None:
    rec = bench.benchmark_region("benchv2-test", tmp_path)
    assert rec is not None

    assert set(rec.gates) == {"beats_untrained_eval", "uses_its_dimensions"}
    assert "beats_untrained" not in rec.gates
    assert "not_anisotropic" not in rec.gates
    # Demoted, not deleted: still a recorded value.
    assert "repr.anisotropy" in rec.metrics


def test_fp32_receipt_effective_rank_ratio_is_renamed(
    tmp_path: Path, trained_receipt: dict
) -> None:
    rec = bench.benchmark_region("benchv2-test", tmp_path)
    assert rec is not None

    assert "repr.effective_rank_entropy_ratio" in rec.metrics
    assert "repr.effective_rank_ratio" not in rec.metrics
    # The gate reads the renamed field at the same 0.05 floor -- "kept as recorded".
    assert rec.gates["uses_its_dimensions"] == (
        rec.metrics["repr.effective_rank_entropy_ratio"] > 0.05
    )


def test_fp32_receipt_metric_groups(tmp_path: Path, trained_receipt: dict) -> None:
    rec = bench.benchmark_region("benchv2-test", tmp_path)
    assert rec is not None

    groups = rec.provenance["metric_groups"]
    assert set(groups) == {"rank", "eff", "repr", "repr.emb_std_anchor", "repr.alignment"}
    for family in ("rank", "eff"):
        assert groups[family]["battery_id"] == "eval_holdout"
        assert groups[family]["pooling"] == "matched"
        assert groups[family]["seed"] == 3
    assert groups["repr"]["battery_id"] == "eval_holdout"
    assert groups["repr"]["pooling"] == "pooled_both"
    assert groups["repr"]["seed"] == 0
    # `repr.emb_std_anchor` (MM §12.5) and `repr.alignment` (MM §3.10(d)/§12.4) are NOT
    # `pooled_both` like the rest of the `repr` family -- each needs its OWN entry, or a
    # `MetricIdentity` (MM §14) built off this receipt reads the wrong pool for them.
    assert groups["repr.emb_std_anchor"]["battery_id"] == "eval_holdout"
    assert groups["repr.emb_std_anchor"]["pooling"] == "anchor"
    assert groups["repr.emb_std_anchor"]["seed"] == 0
    assert groups["repr.alignment"]["battery_id"] == "eval_holdout"
    assert groups["repr.alignment"]["pooling"] == "matched"
    assert groups["repr.alignment"]["seed"] == 0


def _identity_from_group(rec, group_key: str) -> MetricIdentity:
    """Build the `MetricIdentity` (MM §14) a caller would read for a metric filed under
    `provenance.metric_groups[group_key]` on `rec` -- every field taken straight off the
    real receipt except `corpus_fingerprint`/`fingerprint_scheme`, which this eval
    receipt does not itself carry (a separate, disclosed gap: reachable one hop away via
    `artifacts.source_training_receipt`, not needed here since both `MetricGroup`s built
    from the SAME `rec` always agree on it)."""
    g = rec.provenance["metric_groups"][group_key]
    return MetricIdentity(
        metrics_schema=rec.metrics_schema,
        corpus_fingerprint="shared-fixture-fingerprint",
        fingerprint_scheme="csd-corpus-fp/v2",
        battery_id=g["battery_id"],
        k=None,
        pooling=g["pooling"],
        checkpoint_sha256=rec.artifacts["checkpoint_sha256"],
        region=rec.producer.component,
        git_sha=rec.code_revision.get("git_sha", ""),
        seed=g["seed"],
    )


@pytest.mark.parametrize(
    ("anchor_group", "anchor_metric"),
    [
        ("repr.emb_std_anchor", "repr.emb_std_anchor"),
        ("repr.alignment", "repr.alignment"),
    ],
)
def test_compare_refuses_repr_anchor_or_matched_field_against_repr_pooled_both(
    tmp_path: Path, trained_receipt: dict, anchor_group: str, anchor_metric: str
) -> None:
    """MM §14's closing paragraph names this exact class of pair illegal: it "may NOT
    compare held_out.emb_std to repr.anisotropy" (different pools). `repr.alignment` is
    the same story (`matched` vs `pooled_both`). `compare()` (g7 §3.3's refuse
    predicate) must refuse both even though the two sides come from the SAME receipt and
    so agree on every OTHER identity field -- built straight off
    `provenance.metric_groups`, the shape a real caller has to use (no production caller
    exists yet, see `tests/test_eval_metrics.py`/`tests/test_guards_can_fail.py`, so this
    is the shape one WOULD use)."""
    rec = bench.benchmark_region("benchv2-test", tmp_path)
    assert rec is not None

    odd_pool_side = MetricGroup(
        identity=_identity_from_group(rec, anchor_group),
        values={anchor_metric: rec.metrics[anchor_metric]},
    )
    pooled_both_side = MetricGroup(
        identity=_identity_from_group(rec, "repr"),
        values={"repr.anisotropy": rec.metrics["repr.anisotropy"]},
    )

    result = compare(odd_pool_side, pooled_both_side, lower_is_better=set())

    assert result["refused"] is True
    assert result["mismatched_key"] == "pooling"


@pytest.mark.parametrize("anchor_group", ["repr.emb_std_anchor", "repr.alignment"])
def test_pre_fix_single_repr_group_would_not_have_refused_the_illegal_pair(
    tmp_path: Path, trained_receipt: dict, anchor_group: str
) -> None:
    """MUTATION PROOF: before this fix, neither `repr.emb_std_anchor` nor
    `repr.alignment` had a `metric_groups` entry of its own -- a caller building either
    one's `MetricIdentity` had only `groups["repr"]` to read, which declares
    `pooling="pooled_both"`. Reproducing that exact defect (stub the lookup to always
    use the bare `repr` entry, the way `scripts/csd-benchmark.py`'s `_metric_groups` did
    before this branch's fix) shows `compare()` does NOT refuse the illegal pair --
    confirming the refusal test above is not vacuous."""
    rec = bench.benchmark_region("benchv2-test", tmp_path)
    assert rec is not None
    assert anchor_group in rec.provenance["metric_groups"], (
        "the fix under test has regressed -- this receipt no longer carries its own "
        f"{anchor_group!r} entry, so this proof no longer applies"
    )

    pre_fix_identity = _identity_from_group(rec, "repr")  # both sides read "repr" only
    metric_name = anchor_group  # the group key doubles as the metric's own field name
    odd_pool_side = MetricGroup(
        identity=pre_fix_identity, values={metric_name: rec.metrics[metric_name]}
    )
    pooled_both_side = MetricGroup(
        identity=pre_fix_identity, values={"repr.anisotropy": rec.metrics["repr.anisotropy"]}
    )

    result = compare(odd_pool_side, pooled_both_side, lower_is_better=set())

    assert result["refused"] is False, (
        "reproducing the pre-fix single-'repr'-group shape must NOT refuse -- this is "
        "the exact defect this branch's fix closes"
    )


# =================================================== quantized eval receipt (kind="eval-quantized")


def test_quantized_receipt_battery_id_differs_from_fp32(
    tmp_path: Path, trained_receipt: dict
) -> None:
    quant_rec = quant.quantize_text_region(
        "benchv2-test", tmp_path, tolerance=1.0, aggressive=3, max_bits=8
    )
    rec = bench.benchmark_region_quantized(
        "benchv2-test", tmp_path, Path(quant_rec["artifacts"]["quantized_path"])
    )

    groups = rec.provenance["metric_groups"]
    assert groups["rank"]["battery_id"] == "eval_quantized_holdout"
    assert groups["rank"]["battery_id"] != "eval_holdout"
    assert set(rec.gates) == {"beats_untrained_eval", "uses_its_dimensions"}


def test_quantized_receipt_carries_quant_artifact_recall_alias(
    tmp_path: Path, trained_receipt: dict
) -> None:
    """`quant.artifact_recall@1` (g7 §3.1) must be the SAME number as this receipt's
    own `rank.recall@1` -- the plan-vs-artifact sameness guard reads these two field
    names, not `rank.recall@1` directly, so this receipt has to carry both."""
    quant_rec = quant.quantize_text_region(
        "benchv2-test", tmp_path, tolerance=1.0, aggressive=3, max_bits=8
    )
    rec = bench.benchmark_region_quantized(
        "benchv2-test", tmp_path, Path(quant_rec["artifacts"]["quantized_path"])
    )

    assert "quant.artifact_recall@1" in rec.metrics
    assert rec.metrics["quant.artifact_recall@1"] == rec.metrics["rank.recall@1"]


def test_quantized_receipt_declares_provenance_for_the_quant_artifact_alias(
    tmp_path: Path, trained_receipt: dict
) -> None:
    """`quant.artifact_recall@1` is the one v2 metric this receipt writes that crosses
    battery families by design (it IS `rank.recall@1`, MM §12.8's sameness special
    case) -- it must still carry its own `provenance.metric_groups` entry, not be a
    `metrics` key with no declared `battery_id`/`pooling`/`seed` a `MetricIdentity`
    (MM §14) could be built from. Same three values as the `rank` group it mirrors,
    since it is not an independent measurement."""
    quant_rec = quant.quantize_text_region(
        "benchv2-test", tmp_path, tolerance=1.0, aggressive=3, max_bits=8
    )
    rec = bench.benchmark_region_quantized(
        "benchv2-test", tmp_path, Path(quant_rec["artifacts"]["quantized_path"])
    )

    groups = rec.provenance["metric_groups"]
    assert "quant" in groups, "quant.artifact_recall@1 has no provenance.metric_groups entry"
    assert groups["quant"] == groups["rank"]


# ============================================================================= mutation proof


def test_pre_rename_gate_shape_would_fail_this_files_own_assertions(
    tmp_path: Path, trained_receipt: dict
) -> None:
    """Reproduce the pre-rename gates shape (`beats_untrained` + `not_anisotropic` +
    `uses_its_dimensions`, what `scripts/csd-benchmark.py` wrote before this change)
    and confirm `test_fp32_receipt_gates_are_renamed_and_not_anisotropic_is_gone`'s own
    assertion would have caught it -- proving that test is not vacuous."""
    rec = bench.benchmark_region("benchv2-test", tmp_path)
    assert rec is not None
    legacy_shaped_gates = {
        "beats_untrained": rec.gates["beats_untrained_eval"],
        "not_anisotropic": rec.metrics["repr.anisotropy"] < 0.9,
        "uses_its_dimensions": rec.gates["uses_its_dimensions"],
    }

    assert set(legacy_shaped_gates) != {"beats_untrained_eval", "uses_its_dimensions"}
