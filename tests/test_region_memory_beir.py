"""`regions/memory.py`'s `run_memory_pretrain` -- the retrieval head's real gate, layered
over `pretrain_region` via `cogsyndelta.eval.beir_fiqa`, and DEC-24's shared-embedding-
table wiring (`embedding_table_divergence`, `init_embedding_from`).

Synthetic end to end: a tiny FiQA-shaped BEIR pool (not the real 57,638-passage corpus)
so the plumbing is proven -- `retrieval` and `gates` blocks land in the receipt with the
right shape -- without the fleet's NFS export.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from cogsyndelta.regions import memory as memory_mod
from cogsyndelta.regions.text_encoder import TextEncoderConfig

pytestmark = pytest.mark.cpu

_WORDS = [f"w{i}" for i in range(60)]


def _sentence(seed: int, n: int = 6) -> str:
    return " ".join(_WORDS[(seed + i) % len(_WORDS)] for i in range(n))


def _build_beir_pool(root: Path) -> None:
    """`beir_fiqa.resolve_paths`-shaped tree: `fiqa-pairs/{train,dev}.parquet` and
    `fiqa/corpus/*.parquet`, with a small pool and one judged-relevant doc per query."""
    pairs_dir = root / "fiqa-pairs"
    corpus_dir = root / "fiqa" / "corpus"
    pairs_dir.mkdir(parents=True)
    corpus_dir.mkdir(parents=True)

    n_docs = 24
    doc_ids = [f"doc{i}" for i in range(n_docs)]
    doc_texts = [_sentence(i + 1000, 10) for i in range(n_docs)]
    pq.write_table(
        pa.table({"_id": doc_ids, "title": [""] * n_docs, "text": doc_texts}),
        corpus_dir / "corpus.parquet",
    )

    def write_split(name: str, n_queries: int, offset: int) -> list[str]:
        query_ids = [f"q{name}{i}" for i in range(n_queries)]
        gold_doc_idx = [(i + offset) % n_docs for i in range(n_queries)]
        queries = [_sentence(i + offset, 6) for i in range(n_queries)]
        passages = [doc_texts[gold_doc_idx[i]] for i in range(n_queries)]
        scores = [1.0] * n_queries
        gold_doc_ids = [doc_ids[gold_doc_idx[i]] for i in range(n_queries)]
        pq.write_table(
            pa.table(
                {
                    "query_id": query_ids,
                    "doc_id": gold_doc_ids,
                    "score": scores,
                    "query": queries,
                    "passage": passages,
                }
            ),
            pairs_dir / f"{name}.parquet",
        )
        return queries

    write_split("train", n_queries=30, offset=0)
    write_split("dev", n_queries=10, offset=500)


def _build_memory_root(tmp_path: Path) -> tuple[Path, Path]:
    """A `MEMORY_ROOT`-shaped tree for TRAINING (fiqa-pairs/train + all-nli + stsb),
    reusing the BEIR pool's own `fiqa-pairs/train.parquet` as the retrieval training
    source (a real FiQA deployment does this too: the BEIR eval's `train` split IS the
    training corpus, `dev`/`test` are held out)."""
    root = tmp_path / "memory-root"
    allnli_dir = root / "region" / "compress" / "all-nli" / "pair"
    stsb_dir = root / "region" / "compress" / "stsb" / "data"
    allnli_dir.mkdir(parents=True)
    stsb_dir.mkdir(parents=True)

    anchors = [_sentence(i + 2000) for i in range(40)]
    positives = [_sentence(i + 2000) + " " + _sentence(i + 3000, 2) for i in range(40)]
    pq.write_table(
        pa.table({"anchor": anchors, "positive": positives}),
        allnli_dir / "train-00000-of-00001.parquet",
    )
    s1 = [_sentence(i + 4000) for i in range(30)]
    s2 = [_sentence(i + 4000) for i in range(30)]
    scores = [[0.9, 0.5, 0.1][i % 3] for i in range(30)]
    pq.write_table(
        pa.table({"sentence1": s1, "sentence2": s2, "score": scores}),
        stsb_dir / "validation-00000-of-00001.parquet",
    )

    tok_path = tmp_path / "tokenizer.json"
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tok.train_from_iterator(_WORDS, trainer=trainer)
    tok.save(str(tok_path))
    return root, tok_path


@pytest.fixture
def beir_and_memory_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    beir_root = tmp_path / "beir-root"
    beir_root.mkdir()
    _build_beir_pool(beir_root)

    memory_root, tok_path = _build_memory_root(tmp_path)
    # memory's own primary training source IS the BEIR pool's train split -- copy it into
    # MEMORY_ROOT/region/retrieve/fiqa-pairs, where memory_config() resolves it from,
    # leaving beir_root's own copy untouched for the eval side.
    fiqa_pairs_dir = memory_root / "region" / "retrieve" / "fiqa-pairs"
    fiqa_pairs_dir.mkdir(parents=True)
    pq.write_table(
        pq.read_table(beir_root / "fiqa-pairs" / "train.parquet"),
        fiqa_pairs_dir / "train.parquet",
    )

    monkeypatch.setattr(memory_mod, "MEMORY_ROOT", memory_root)
    return beir_root, tok_path


def test_run_memory_pretrain_produces_retrieval_and_gates_blocks(
    beir_and_memory_roots: tuple[Path, Path], tmp_path: Path
) -> None:
    beir_root, tok_path = beir_and_memory_roots

    receipt = memory_mod.run_memory_pretrain(
        steps=6,
        batch_size=8,
        holdout_pairs=8,
        eval_every=3,
        checkpoint_every=0,
        max_len=24,
        device="cpu",
        encoder=TextEncoderConfig(dim=16, depth=1, n_heads=2, max_len=24),
        tokenizer_path=str(tok_path),
        out_dir=str(tmp_path / "receipts"),
        eval_root=beir_root,
        eval_split="dev",
    )

    assert receipt["region"] == "memory"
    retrieval = receipt["retrieval"]
    assert retrieval["eval_split"] == "dev"
    full = retrieval["full_pool"]
    assert full["task"]["pool_size"] == 24.0
    for arm in ("trained", "untrained", "lexical_bm25"):
        assert "recall@10" in full[arm]
        assert "mrr" in full[arm]

    gates = receipt["gates"]
    assert set(gates) == {
        "a_beats_both_parents",
        "b_full_pool_thresholds",
        "c_beats_bm25",
        "d_beats_random_init",
        "e_retrain_gate",
        "passed",
    }
    assert isinstance(gates["passed"], bool)
    # e's PR-rank clause is readable straight from the receipt commit 1 wrote.
    assert gates["e_retrain_gate"]["pr_rank_clause"]["token_global_pr_rank"] == pytest.approx(
        receipt["token_aware"]["final_block_rank"]["token_global_pr_rank"]
    )


def test_run_memory_pretrain_skip_lexical_leaves_gates_unevaluable(
    beir_and_memory_roots: tuple[Path, Path], tmp_path: Path
) -> None:
    beir_root, tok_path = beir_and_memory_roots
    receipt = memory_mod.run_memory_pretrain(
        steps=4,
        batch_size=8,
        holdout_pairs=8,
        eval_every=2,
        checkpoint_every=0,
        max_len=24,
        device="cpu",
        encoder=TextEncoderConfig(dim=16, depth=1, n_heads=2, max_len=24),
        tokenizer_path=str(tok_path),
        out_dir=str(tmp_path / "receipts"),
        eval_root=beir_root,
        skip_lexical=True,
    )
    assert receipt["gates"]["passed"] is None
    assert receipt["retrieval"]["full_pool"]["lexical_bm25"] == {}


# ---------------------------------------------------------------------------------------
# DEC-24: embedding_table_divergence + init_embedding_from through run_memory_pretrain.
# ---------------------------------------------------------------------------------------


def test_embedding_table_divergence_of_a_table_against_itself_is_near_perfect() -> None:
    table = torch.randn(30, 8)
    ckpt_a = "table_a.pt"
    ckpt_b = "table_b.pt"
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        pa_ = Path(d) / ckpt_a
        pb_ = Path(d) / ckpt_b
        torch.save({"model": {"embed.weight": table}}, pa_)
        torch.save({"model": {"embed.weight": table.clone()}}, pb_)
        result = memory_mod.embedding_table_divergence(str(pa_), str(pb_))
    assert result["mean_cosine_similarity"] == pytest.approx(1.0, abs=1e-5)
    assert result["mean_l2_distance"] == pytest.approx(0.0, abs=1e-5)
    assert result["n_tokens"] == 30


def test_embedding_table_divergence_refuses_a_shape_mismatch(tmp_path: Path) -> None:
    a_path = tmp_path / "a.pt"
    b_path = tmp_path / "b.pt"
    torch.save({"model": {"embed.weight": torch.randn(30, 8)}}, a_path)
    torch.save({"model": {"embed.weight": torch.randn(40, 8)}}, b_path)
    with pytest.raises(ValueError, match="different shape"):
        memory_mod.embedding_table_divergence(str(a_path), str(b_path))


def test_run_memory_pretrain_records_shared_embedding_table_when_given_parent_checkpoints(
    beir_and_memory_roots: tuple[Path, Path], tmp_path: Path
) -> None:
    beir_root, tok_path = beir_and_memory_roots
    tok = Tokenizer.from_file(str(tok_path))
    dim, depth, heads = 16, 1, 2
    encoder_cfg = TextEncoderConfig(
        vocab_size=tok.get_vocab_size(), dim=dim, depth=depth, n_heads=heads, max_len=24
    )

    retrieve_ckpt = tmp_path / "retrieve-final.pt"
    compress_ckpt = tmp_path / "compress-final.pt"
    retrieve_table = torch.randn(encoder_cfg.vocab_size, dim)
    compress_table = torch.randn(encoder_cfg.vocab_size, dim)
    torch.save(
        {"model": {"embed.weight": retrieve_table}, "config": vars(encoder_cfg)}, retrieve_ckpt
    )
    torch.save(
        {"model": {"embed.weight": compress_table}, "config": vars(encoder_cfg)}, compress_ckpt
    )

    receipt = memory_mod.run_memory_pretrain(
        steps=4,
        batch_size=8,
        holdout_pairs=8,
        eval_every=2,
        checkpoint_every=0,
        max_len=24,
        device="cpu",
        encoder=TextEncoderConfig(dim=dim, depth=depth, n_heads=heads, max_len=24),
        tokenizer_path=str(tok_path),
        out_dir=str(tmp_path / "receipts"),
        eval_root=beir_root,
        skip_lexical=True,
        retrieve_checkpoint=str(retrieve_ckpt),
        compress_checkpoint=str(compress_ckpt),
    )
    block = receipt["shared_embedding_table"]
    assert block["applied"] is True
    assert block["source"] == str(retrieve_ckpt)
    assert "divergence_from_compress" in block
    assert block["divergence_from_compress"]["n_tokens"] == encoder_cfg.vocab_size
