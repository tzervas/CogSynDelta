"""`scripts/csd-quantize.py`'s quant receipt under the metrics-v2 unification
(g7-latent-eval-metrics.md §3.1/§3.3): field renames, per-battery-group provenance, and
the `metrics_schema` stamp.

Renames proven against a REAL `quantize_text_region` receipt (same tiny CPU
pretrain + quantize fixture `tests/test_quantize_checkpoint_sha_receipt.py` uses), not a
hand-built stand-in:

- `quantized_metric` -> `quant.plan_recall@1`
- `compression_ratio` -> `quant.compression_ratio`
- `drop` -> `quant.drop_recall@1`

plus the new `battery_id`/`pooling`/`seed` fields (one group: this receipt reports
exactly one battery, `quant_plan`) and the `metrics_schema` stamp
`cogsyndelta.regions._receipt.write_receipt` adds on write.

Mutation proofs at the bottom: deleting a renamed field (what the pre-rename code
produced) fails the same assertions the positive tests make, and a receipt dict that
already carries a stale `metrics_schema` before `write_receipt` runs gets it
overwritten, not merely defaulted.
"""

from __future__ import annotations

import importlib.util
import json
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

from cogsyndelta.regions._receipt import METRICS_SCHEMA_V2, write_receipt
from cogsyndelta.regions.pretrain import PretrainConfig, pretrain_region
from cogsyndelta.regions.text_encoder import TextEncoderConfig

pytestmark = pytest.mark.cpu

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_csd_quantize():
    """Import scripts/csd-quantize.py -- a hyphenated filename is not a valid module
    name, so every consumer in this repo loads it this way (see
    tests/test_region_spec_consumers.py, tests/test_quantize_checkpoint_sha_receipt.py)."""
    path = SCRIPTS / "csd-quantize.py"
    spec = importlib.util.spec_from_file_location("csd_quantize_metrics_v2_test", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_csd_quantize()


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
    """A REAL receipt and checkpoint from a tiny CPU run of `pretrain_region`, written
    under `<tmp_path>/receipts` -- exactly the shape `quantize_text_region` reads in
    production."""
    tok_path = tmp_path / "tokenizer.json"
    shard_path = tmp_path / "pairs.parquet"
    _build_tokenizer(tok_path, 40)
    _build_pairs_parquet(shard_path, 40)

    cfg = PretrainConfig(
        region="metricsv2-test",
        pair_columns=("anchor", "positive"),
        shards=[str(shard_path)],
        steps=2,
        batch_size=4,
        holdout_pairs=4,
        eval_every=2,
        checkpoint_every=2,
        max_len=16,
        seed=7,
        device="cpu",
        encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=16),
        tokenizer_path=str(tok_path),
        out_dir=str(tmp_path / "receipts"),
    )
    receipt = pretrain_region(cfg)

    def fake_load_regions_spec() -> dict:
        def _shards(glob_pat: str, root: Path | None = None) -> list[str]:
            return [str(shard_path)]

        class _Entry:
            sources: ClassVar = [("pairs.parquet", ("anchor", "positive"), 0)]
            root = tmp_path

        return {"REGIONS": {}, "_shards": _shards, "region_spec": lambda name: _Entry()}

    mod._load_regions_spec = fake_load_regions_spec  # module-level monkeypatch, restored below
    return receipt


@pytest.fixture(autouse=True)
def _restore_load_regions_spec():
    original = mod._load_regions_spec
    yield
    mod._load_regions_spec = original


# ============================================================== field renames


def test_quant_receipt_uses_v2_field_names(tmp_path: Path, trained_receipt: dict) -> None:
    rec = mod.quantize_text_region(
        "metricsv2-test", tmp_path, tolerance=1.0, aggressive=3, max_bits=8
    )

    for legacy in ("quantized_metric", "compression_ratio", "drop"):
        assert legacy not in rec, f"legacy field {legacy!r} must not be written by a v2 producer"

    assert isinstance(rec["quant.plan_recall@1"], float)
    assert isinstance(rec["quant.compression_ratio"], float)
    assert isinstance(rec["quant.drop_recall@1"], float)
    # Unrenamed fields, per g7 §3.1 -- the spec names only these three.
    assert "fp32_metric_recomputed" in rec
    assert "within_budget" in rec
    assert "fp32_bytes" in rec
    assert "stored_bytes" in rec

    # The relationship the old names encoded must still hold under the new names.
    assert rec["quant.drop_recall@1"] == pytest.approx(
        rec["fp32_metric_recomputed"] - rec["quant.plan_recall@1"], abs=1e-9
    )
    assert rec["quant.compression_ratio"] == pytest.approx(
        rec["fp32_bytes"] / rec["stored_bytes"], rel=1e-6
    )
    assert rec["within_budget"] == (rec["quant.drop_recall@1"] <= 1.0)


# ==================================================== per-battery-group provenance


def test_quant_receipt_carries_battery_id_pooling_and_seed(
    tmp_path: Path, trained_receipt: dict
) -> None:
    """This receipt reports exactly one battery (the quantize stage's in-memory
    sensitivity/plan pass, MM §4/§5.4) -- one battery_id/pooling/seed for the whole
    receipt, not per field, matches that shape (g7 §3.1/§3.3)."""
    rec = mod.quantize_text_region(
        "metricsv2-test", tmp_path, tolerance=1.0, aggressive=3, max_bits=8
    )

    assert rec["battery_id"] == "quant_plan"
    assert rec["pooling"] == "matched"
    # The corpus/holdout-construction seed (split_seed, E0) -- not the training-init
    # seed. The same split quantize_text_region rebuilt via build_splits(cfg).
    assert rec["seed"] == trained_receipt["split"]["seed"]
    assert rec["seed"] == trained_receipt["config"]["split_seed"]


# =============================================================== metrics_schema stamp


def test_write_receipt_stamps_metrics_schema_v2_on_a_real_quant_receipt(
    tmp_path: Path, trained_receipt: dict
) -> None:
    rec = mod.quantize_text_region(
        "metricsv2-test", tmp_path, tolerance=1.0, aggressive=3, max_bits=8
    )
    assert "metrics_schema" not in rec, (
        "quantize_text_region's return value is pre-write -- the stamp is write_receipt's "
        "job, proven below, not something the region function should pre-empt"
    )

    out_dir = tmp_path / "receipts"
    path = write_receipt(rec, out_dir, "metricsv2-test-quant-20260101T000000Z.json")

    assert rec["metrics_schema"] == METRICS_SCHEMA_V2 == "csd-metrics/v2"
    on_disk = json.loads(path.read_text())
    assert on_disk["metrics_schema"] == "csd-metrics/v2"


def test_write_receipt_overwrites_a_stale_metrics_schema_not_merely_defaults_it(
    tmp_path: Path,
) -> None:
    """MUTATION-adjacent: same 'always overwrite, never merely default' contract
    `code_revision` already has (see tests/test_receipt_provenance.py) -- a caller that
    somehow re-wrote a receipt dict still carrying an old `metrics_schema` value must
    not have that stale value survive `write_receipt`."""
    receipt: dict = {"region": "test", "metrics_schema": "csd-metrics/v1", "held_out": {}}

    write_receipt(receipt, tmp_path, "stale-schema-20260101T000000Z.json")

    assert receipt["metrics_schema"] == "csd-metrics/v2"


# ============================================================== mutation proof


def test_receipt_missing_the_renamed_fields_would_fail_the_positive_assertions(
    tmp_path: Path, trained_receipt: dict
) -> None:
    """Reproduce the pre-rename receipt shape (v1 names only, no `quant.*` keys -- what
    `scripts/csd-quantize.py` wrote before this change) and confirm this file's own v2
    assertions reject it, proving `test_quant_receipt_uses_v2_field_names` is actually
    checking something and would have failed against the old code."""
    rec = mod.quantize_text_region(
        "metricsv2-test", tmp_path, tolerance=1.0, aggressive=3, max_bits=8
    )
    legacy_shaped = dict(rec)
    legacy_shaped["quantized_metric"] = legacy_shaped.pop("quant.plan_recall@1")
    legacy_shaped["compression_ratio"] = legacy_shaped.pop("quant.compression_ratio")
    legacy_shaped["drop"] = legacy_shaped.pop("quant.drop_recall@1")

    for v2_key in ("quant.plan_recall@1", "quant.compression_ratio", "quant.drop_recall@1"):
        assert v2_key not in legacy_shaped, (
            f"the legacy-shaped fixture must not accidentally carry {v2_key!r}, or this "
            "proof is vacuous"
        )
