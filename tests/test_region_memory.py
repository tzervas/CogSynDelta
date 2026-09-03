"""`memory`-region wiring tests (DEC-02, row W4).

Three things this file proves, matching the task's own checklist:

  1. BOTH HEADS PRODUCE RECEIPTS -- a full tiny run through the real entry point
     (`pretrain_region(memory_config(...))`) on synthetic shards yields a receipt with
     the consolidation head's graded Spearman gate AND the retrieval head's diagonal
     recall/MRR, with §4.0's token-aware terms on by default.
  2. THE GRADED GATE FIRES WHEN DECLARED-BUT-UNRESOLVED -- through `run_region`
     (`scripts/csd-train-all.py`), the same real entry point every other region's
     graded-gate regression test uses (see `tests/test_guards_can_fail.py`'s
     `test_run_region_raises_when_the_declared_graded_source_does_not_resolve`).
  3. A RESERVED SOURCE IS REFUSED -- through `run_region`, the same real entry point
     `tests/test_reserved_corpus_guard.py` exercises for `code`.

(2) and (3) need no `memory`-specific code: `REGIONS["memory"]` (added alongside `code`/
`compress`/`retrieve`/`reason`) gets `_refuse_reserved_shards`/`GradedSourceMissingError`
for free, because both guards live in `run_region` itself and fire for every region that
goes through it. These tests exist to prove that is actually true for `memory`, not
merely plausible.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from cogsyndelta.regions import memory as memory_mod
from cogsyndelta.regions.pretrain import pretrain_region
from cogsyndelta.regions.text_encoder import TextEncoderConfig

pytestmark = pytest.mark.cpu


def _load_csd_train_all():
    """Same convention every consumer of the hyphenated script uses (see
    `tests/test_reserved_corpus_guard.py`)."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "csd-train-all.py"
    spec = importlib.util.spec_from_file_location("csd_train_all_memory_test", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


train_all = _load_csd_train_all()


# ---------------------------------------------------------------------------------------
# region_spec / REGIONS wiring.
# ---------------------------------------------------------------------------------------


def test_memory_is_declared_in_regions_and_resolves_through_region_spec() -> None:
    entry = train_all.region_spec("memory")
    cols = [c for _glob, c, _cap in entry.sources]
    assert ("query", "passage") in cols, "retrieve's FiQA pairs must be a declared source"
    assert ("anchor", "positive") in cols, "compress's AllNLI pairs must be a declared source"
    assert entry.graded is not None
    _glob, graded_cols, graded_name = entry.graded
    assert graded_name == "stsb-validation"
    assert graded_cols == ("sentence1", "sentence2", "score")


def test_memory_run_refuses_to_start_on_a_reserved_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same shape as `tests/test_reserved_corpus_guard.py`'s `code` test: a resolved
    shard under a reserved directory must stop the run before anything else happens,
    for `memory` exactly as for every other region `run_region` drives."""
    monkeypatch.setattr(
        train_all,
        "_shards",
        lambda pattern, root=train_all.CORPUS: [
            "/mnt/fleet-datasets/csd/region/code/apps/train.parquet"
        ],
    )
    with pytest.raises(train_all.ReservedSourceError):
        train_all.run_region(
            name="memory", state=tmp_path, steps=1, batch=1, shard_limit=0, dry=True
        )


def test_memory_run_refuses_to_start_when_the_graded_source_does_not_resolve(
    tmp_path: Path,
) -> None:
    """Reproduces `test_guards_can_fail.py`'s compress regression for `memory`: the
    training shards resolve, the STS-B graded shard does not."""

    def fake_shards(pattern: str, root: Path = train_all.CORPUS) -> list[str]:
        if "stsb" in pattern:
            return []  # declared (memory has a GradedSpec), unresolved
        return [str(tmp_path / "shard.parquet")]

    with pytest.MonkeyPatch.context() as m:
        m.setattr(train_all, "_shards", fake_shards)
        m.setattr(train_all, "_schema_mismatch", lambda shards, cols: None)
        with pytest.raises(train_all.GradedSourceMissingError, match="stsb-validation"):
            train_all.run_region(
                name="memory", state=tmp_path, steps=1, batch=1, shard_limit=0, dry=True
            )


# ---------------------------------------------------------------------------------------
# `memory_config` -- the standalone module, same shape as `regions/compress.py`.
# ---------------------------------------------------------------------------------------


def test_missing_corpus_raises_instead_of_training_on_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_mod, "MEMORY_ROOT", tmp_path / "definitely-absent")
    with pytest.raises(FileNotFoundError, match="memory corpus shard missing"):
        memory_mod.memory_config()


def test_consolidation_gate_needs_a_measured_spearman() -> None:
    with pytest.raises(KeyError, match="no graded_held_out"):
        memory_mod.consolidation_gate_report({"held_out": {"recall@1": 0.9}})


def test_consolidation_gate_reports_a_pass_that_did_not_beat_untrained() -> None:
    receipt = {
        "untrained_graded_baseline": {"spearman": 0.55, "emb_std": 0.02},
        "graded_held_out": {"spearman": 0.50, "emb_std": 0.02},
    }
    report = memory_mod.consolidation_gate_report(receipt)
    assert report["passed"] is True
    assert report["beats_untrained"] is False
    assert report["verdict"] == "PASS (but does not beat untrained)"


def test_token_loss_and_decorr_weight_default_on() -> None:
    """Row W4 is designed as the first token-aware retrain -- unlike every other
    region's `*_config`, this one must not need an override to get §4.0's terms."""
    # Constructing memory_config() itself needs real shards; check the module constants
    # directly, which is what memory_config()'s PretrainConfig(...) call reads from.
    assert memory_mod.TOKEN_LOSS_WEIGHT > 0.0
    assert memory_mod.DECORR_WEIGHT > 0.0


# ---------------------------------------------------------------------------------------
# End to end: a real tiny run, both heads.
# ---------------------------------------------------------------------------------------


def _build_synthetic_root(tmp_path: Path) -> Path:
    """A `MEMORY_ROOT`-shaped directory tree with just enough real parquet/tokenizer data
    for `pretrain_region` to run a few real steps -- fiqa (retrieval, primary) and
    all-nli + stsb (consolidation)."""
    root = tmp_path / "corpus-root"
    fiqa_dir = root / "region" / "retrieve" / "fiqa-pairs"
    allnli_dir = root / "region" / "compress" / "all-nli" / "pair"
    stsb_dir = root / "region" / "compress" / "stsb" / "data"
    for d in (fiqa_dir, allnli_dir, stsb_dir):
        d.mkdir(parents=True)

    words = [f"w{i}" for i in range(50)]

    def sentence(seed: int, n: int = 6) -> str:
        return " ".join(words[(seed + i) % len(words)] for i in range(n))

    n = 80
    queries = [sentence(i) for i in range(n)]
    passages = [sentence(i) + " " + sentence(i + 7, 2) for i in range(n)]
    pq.write_table(pa.table({"query": queries, "passage": passages}), fiqa_dir / "train.parquet")

    anchors = [sentence(i + 100) for i in range(n)]
    positives = [sentence(i + 100) + " " + sentence(i + 200, 2) for i in range(n)]
    pq.write_table(
        pa.table({"anchor": anchors, "positive": positives}),
        allnli_dir / "train-00000-of-00001.parquet",
    )

    s1 = [sentence(i + 300) for i in range(60)]
    s2 = [sentence(i + 300) for i in range(60)]  # near-identical -> high similarity
    scores = [[0.9, 0.5, 0.1][i % 3] for i in range(60)]  # varied, not a degenerate constant
    pq.write_table(
        pa.table({"sentence1": s1, "sentence2": s2, "score": scores}),
        stsb_dir / "validation-00000-of-00001.parquet",
    )

    tok_path = tmp_path / "tokenizer.json"
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tok.train_from_iterator(queries + passages + anchors + positives + s1 + s2, trainer=trainer)
    tok.save(str(tok_path))
    return root, tok_path


def test_short_run_produces_a_receipt_with_both_heads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end on synthetic data: two steps train nothing meaningful, but the receipt
    shape must hold -- consolidation's graded Spearman AND the retrieval-shaped diagonal
    recall/MRR, with the token-aware terms on (row W4's own choice, not an override)."""
    root, tok_path = _build_synthetic_root(tmp_path)
    monkeypatch.setattr(memory_mod, "MEMORY_ROOT", root)

    receipt = pretrain_region(
        memory_mod.memory_config(
            steps=6,
            batch_size=16,
            holdout_pairs=16,
            eval_every=2,
            checkpoint_every=0,
            max_len=24,
            device="cpu",
            bf16=False,
            out_dir=str(tmp_path / "receipts"),
            tokenizer_path=str(tok_path),
            encoder=TextEncoderConfig(dim=16, depth=1, n_heads=2, max_len=24),
        )
    )

    assert receipt["region"] == "memory"
    # Consolidation head.
    assert receipt["graded_corpus"]["name"] == "stsb-validation"
    assert -1.0 <= receipt["graded_held_out"]["spearman"] <= 1.0
    consolidation = memory_mod.consolidation_gate_report(receipt)
    assert consolidation["gate"] == "consolidation/memory"
    # Retrieval head (diagonal today; the BEIR full-pool gate layers on top separately).
    assert receipt["held_out"]["n_pairs"] == 16.0
    assert 0.0 <= receipt["held_out"]["recall@1"] <= 1.0
    assert 0.0 <= receipt["held_out"]["mrr"] <= 1.0
    # Row W4's own choice: token-aware terms on by default for this region.
    assert receipt["token_aware"]["enabled"] is True
    assert receipt["token_aware"]["token_loss_weight"] == memory_mod.TOKEN_LOSS_WEIGHT
    assert receipt["token_aware"]["decorr_weight"] == memory_mod.DECORR_WEIGHT
    # The union corpus: AllNLI landed as an extra source alongside FiQA.
    assert any(k == "anchor->positive" for k in receipt["corpus"]["sources"])
