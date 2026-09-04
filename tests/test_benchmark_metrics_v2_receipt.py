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
  code path), `pooling`, and `seed`.
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
    assert set(groups) == {"rank", "eff", "repr"}
    for family in ("rank", "eff"):
        assert groups[family]["battery_id"] == "eval_holdout"
        assert groups[family]["pooling"] == "matched"
        assert groups[family]["seed"] == 3
    assert groups["repr"]["battery_id"] == "eval_holdout"
    assert groups["repr"]["pooling"] == "pooled_both"
    assert groups["repr"]["seed"] == 0


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
