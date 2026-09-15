"""`scripts/csd-train-all.py --seed` must actually reach every region runner it
dispatches to, not merely exist on the parser.

WHY THIS EXISTS
`program/matrix/csd-matrix.yaml`'s default `regions.defaults.commands.train` template
(used by `code`/`compress`/`retrieve`/`reason`) already passes `--seed {seed}` to this
script, and `regions.memory.commands.train` passes `--seed {seed}` to
`python -m cogsyndelta.regions.memory` directly. Before this fix, `csd-train-all.py`'s
`ArgumentParser` had no `--seed` flag at all, so the FIRST real matrix cell died with
`error: unrecognized arguments: --seed 0` (exit 2) before training ever started. Every
underlying config (`PretrainConfig.seed`, `VLPretrainConfig.seed`,
`ClassifyPretrainConfig.seed`, `run_memory_pretrain(seed=...)`) already honoured a seed
correctly -- the gap was purely in this script's CLI surface and the four `run_*_region`
functions that build those configs.

This file proves three things per the task's own checklist:
  1. `--seed 7` on the command line lands in the WRITTEN receipt for the generic
     `run_region` path (code/compress/retrieve/reason) -- the exact path the first
     failing matrix cell (`code`) uses.
  2. `--seed` is threaded into `run_vl_region`/`run_classify_region`/`run_memory_region`
     too (see `scripts/csd-train-all.py`'s docstrings for why each needs its own wire --
     none of the four share a config type).
  3. The seed genuinely governs the run, not just a receipt field nobody reads: two
     toy-scale CPU runs of `pretrain_region` at the same seed produce a bit-identical
     final checkpoint; two different seeds do not.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")
pytest.importorskip("pyarrow", reason="train group not installed")

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from cogsyndelta.regions.pretrain import PretrainConfig, pretrain_region
from cogsyndelta.regions.text_encoder import TextEncoderConfig

pytestmark = pytest.mark.cpu

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "csd-train-all.py"


def _load_csd_train_all(name: str):
    """Same loading convention every other consumer of this hyphenated script uses --
    see `tests/test_reserved_corpus_guard.py`'s `_load_csd_train_all` docstring."""
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {SCRIPT_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_csd_train_all("csd_train_all_seed_flag_test")


# ---------------------------------------------------------------------------------------
# (1) Receipt-level: `--seed 7` through `main()` -> `run_region` -> the written receipt.
# ---------------------------------------------------------------------------------------


def _fake_pretrain_region_receipt(cfg: Any, receipt_path: Path) -> dict[str, Any]:
    """Shape `pretrain_region` returns, trimmed to what `run_region`'s tail print and
    `main()`'s gate bookkeeping read out of a receipt (same trimming
    `tests/test_csd_train_all_memory_dispatch.py`'s `_fake_generic_receipt` uses) --
    PLUS the two fields the real `pretrain_region` actually carries the seed in:
    `receipt["config"]["seed"]` (via `asdict(cfg)`, folded into `receipt["config"]` --
    `"seed"` is a plain `PretrainConfig` field and is not one of the two keys that block
    excludes, `("shards", "encoder")`) and `receipt["untrained_baseline_seed"]` (a
    dedicated top-level field the same module also writes, "what makes
    `untrained_baseline` reproducible by anyone re-running this exact config" -- see
    that field's own docstring in `regions/pretrain.py`). These two are the one thing
    the fake must reproduce faithfully: everything downstream of it (`run_region`'s own
    `Path(receipt["receipt_path"]).write_text(...)` call) is REAL code, not faked, so a
    passing test here proves that real line writes what `main()` actually threaded
    through argparse.
    """
    return {
        "region": cfg.region,
        "config": {"seed": cfg.seed},
        "untrained_baseline_seed": cfg.seed,
        "untrained_baseline": {"recall@1": 0.10, "recall@10": 0.30},
        "held_out": {"recall@1": 0.50, "recall@10": 0.80, "mrr": 0.60},
        "beats_untrained": {"recall@1": True, "recall@10": True},
        "parameters": 16_021_248,
        "receipt_path": str(receipt_path),
        "resumed": False,
    }


def _run_code_region_via_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, extra_argv: list[str]
) -> dict[str, Any]:
    """Drive `mod.main()` for `--regions code`, corpus resolution stubbed (same pattern
    `tests/test_token_aware_dry_run_plan.py` uses), `pretrain_region` faked so no real
    training happens. Returns the receipt JSON actually written to disk by `run_region`.
    """
    monkeypatch.setattr(mod, "CORPUS", tmp_path)
    monkeypatch.setattr(
        mod, "_shards", lambda pattern, root=mod.CORPUS: [str(tmp_path / "shard.parquet")]
    )
    monkeypatch.setattr(mod, "_schema_mismatch", lambda shards, cols: None)

    receipt_path = tmp_path / "code-receipt.json"

    def fake_pretrain_region(cfg: Any) -> dict[str, Any]:
        return _fake_pretrain_region_receipt(cfg, receipt_path)

    monkeypatch.setattr("cogsyndelta.regions.pretrain_region", fake_pretrain_region)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "csd-train-all.py",
            "--regions",
            "code",
            "--steps",
            "5",
            "--batch",
            "4",
            "--state",
            str(tmp_path),
            *extra_argv,
        ],
    )
    rc = mod.main()
    assert rc == 0, "the fake receipt beats its untrained baseline; rc must be 0"
    assert receipt_path.is_file(), "run_region never wrote the receipt to receipt_path"
    return json.loads(receipt_path.read_text())


def test_seed_flag_flows_from_cli_through_run_region_into_the_written_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _run_code_region_via_main(tmp_path, monkeypatch, extra_argv=["--seed", "7"])
    assert data["config"]["seed"] == 7
    assert data["untrained_baseline_seed"] == 7


def test_omitting_seed_flag_still_writes_the_pre_existing_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard for the "unchanged when omitted" half of the requirement:
    `PretrainConfig.seed`'s own default is 0, so a run with no `--seed` at all must
    still write `config.seed: 0`, byte-identical to every receipt written before this
    flag existed."""
    data = _run_code_region_via_main(tmp_path, monkeypatch, extra_argv=[])
    assert data["config"]["seed"] == 0
    assert data["untrained_baseline_seed"] == 0


# ---------------------------------------------------------------------------------------
# (2) `--seed` reaches run_vl_region / run_classify_region's own Config construction.
#
# Dispatched directly (not through main()) -- these two runners' shard resolution and
# receipt shape are exercised in detail elsewhere (tests/test_reserved_corpus_guard.py);
# the only new thing here is that `seed` reaches `VLPretrainConfig`/`ClassifyPretrainConfig`
# construction, which a dry run (returns before either is built) cannot observe.
# ---------------------------------------------------------------------------------------


def test_seed_flag_flows_into_vl_pretrain_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # W7v-cfg's OD-4 gate (VisualCorpusUnsetError) fires before shard resolution unless
    # a corpus_source is admitted -- irrelevant to what this test actually checks (that
    # --seed flows into VLPretrainConfig), so name a placeholder to get past it.
    monkeypatch.setitem(mod.VL_REGIONS["vl_latent"], "corpus_source", "test-corpus")
    monkeypatch.setitem(mod.VL_REGIONS["vl_latent"], "manifest", None)
    monkeypatch.setattr(
        mod, "_shards", lambda pattern, root=mod.CORPUS: [str(tmp_path / "shard.parquet")]
    )

    captured: list[Any] = []

    def fake_pretrain_vl_region(cfg: Any) -> dict[str, Any]:
        captured.append(cfg)
        return {
            "region": cfg.region,
            "untrained_baseline": {"top1": 0.10, "top5": 0.30, "rep_std": 0.02},
            "held_out": {"top1": 0.50, "top5": 0.80, "rep_std": 0.05},
            "collapse_ratio": 2.5,
            "collapsed": False,
            "beats_untrained": {"top1": True},
            "parameters": 1_000_000,
        }

    monkeypatch.setattr(
        "cogsyndelta.regions.vl_pretrain.pretrain_vl_region", fake_pretrain_vl_region
    )

    receipt = mod.run_vl_region(
        name="vl_latent", state=tmp_path, steps=2, batch=2, dry=False, seed=7
    )
    assert receipt is not None
    assert len(captured) == 1
    assert captured[0].seed == 7


def test_seed_flag_flows_into_classify_pretrain_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        mod, "_shards", lambda pattern, root=mod.LOCAL_CORPUS: [str(tmp_path / "shard.parquet")]
    )

    captured: list[Any] = []

    def fake_pretrain_classify_region(cfg: Any) -> dict[str, Any]:
        captured.append(cfg)
        return {
            "region": cfg.region,
            "untrained_baseline": {"top1": 0.05, "top1_chance": 0.013, "macro_f1": 0.01},
            "held_out": {"top1": 0.50, "top1_chance": 0.013, "macro_f1": 0.40},
            "beats_untrained": {"top1": True},
            "parameters": 500_000,
        }

    monkeypatch.setattr(
        "cogsyndelta.regions.classify_pretrain.pretrain_classify_region",
        fake_pretrain_classify_region,
    )

    receipt = mod.run_classify_region(
        name="classify_banking77", state=tmp_path, steps=2, batch=2, dry=False, seed=7
    )
    assert receipt is not None
    assert len(captured) == 1
    assert captured[0].seed == 7


# ---------------------------------------------------------------------------------------
# (3) Determinism: the seed governs the actual RNG stream, not just a receipt field.
# ---------------------------------------------------------------------------------------


def _shared_tiny_fixture(tmp_path: Path) -> tuple[str, str]:
    """One tokenizer + one corpus shard, shared read-only across every run this test
    file trains -- same fixture shape as `tests/test_bf16_precision.py`'s `_tiny_cfg`,
    factored out so multiple `PretrainConfig`s (different seeds, different `out_dir`s)
    can point at the identical corpus without retraining the tokenizer per call.
    """
    n_pairs = 40
    texts = [f"anchor number {i} word" for i in range(n_pairs)] + [
        f"anchor number {i} match" for i in range(n_pairs)
    ]
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106 -- a token id name
    tok.pre_tokenizer = Whitespace()
    tok.train_from_iterator(texts, trainer=WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"]))
    tok_path = tmp_path / "tokenizer.json"
    tok.save(str(tok_path))
    shard_path = tmp_path / "pairs.parquet"
    pq.write_table(pa.table({"anchor": texts[:n_pairs], "positive": texts[n_pairs:]}), shard_path)
    return str(tok_path), str(shard_path)


def _seeded_cfg(
    tok_path: str, shard_path: str, tmp_path: Path, *, seed: int, run_id: str
) -> PretrainConfig:
    """A config small enough to train on CPU in a fraction of a second (same dims as
    `tests/test_bf16_precision.py`'s `_tiny_cfg`), with its OWN `out_dir` per `run_id`
    so two calls never see each other's checkpoint and silently resume instead of
    training fresh -- see `pretrain_region`'s `ckpt_dir = Path(cfg.out_dir) /
    f"{cfg.region}-checkpoints" / ...` (regions/pretrain.py), which is exactly why a
    shared `out_dir` across the "same seed twice" and "different seed" calls below
    would defeat this test rather than prove anything.
    """
    return PretrainConfig(
        region="tiny-seed-determinism",
        pair_columns=("anchor", "positive"),
        shards=[shard_path],
        steps=4,
        batch_size=4,
        holdout_pairs=4,
        max_len=8,
        eval_every=100,
        warmup_steps=1,
        checkpoint_every=0,
        seed=seed,
        bf16=False,
        device="cpu",
        encoder=TextEncoderConfig(dim=16, depth=1, n_heads=2, max_len=8),
        tokenizer_path=tok_path,
        out_dir=str(tmp_path / f"out-{run_id}"),
    )


def _final_state_dict(receipt: dict[str, Any]) -> dict[str, torch.Tensor]:
    payload = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    return payload["model"]


def test_same_seed_produces_a_bit_identical_checkpoint_different_seed_diverges(
    tmp_path: Path,
) -> None:
    tok_path, shard_path = _shared_tiny_fixture(tmp_path)

    receipt_a1 = pretrain_region(_seeded_cfg(tok_path, shard_path, tmp_path, seed=3, run_id="a1"))
    receipt_a2 = pretrain_region(_seeded_cfg(tok_path, shard_path, tmp_path, seed=3, run_id="a2"))
    receipt_b = pretrain_region(_seeded_cfg(tok_path, shard_path, tmp_path, seed=9, run_id="b"))

    # The receipt itself must record the seed that was actually used -- the other half
    # of the "RECORDED in the training receipt" requirement, proven here at the library
    # level, with the REAL (unfaked) pretrain_region, against its two actual seed
    # fields (the CLI-level proof, against a lighter fake, is
    # test_seed_flag_flows_from_cli_through_run_region_into_the_written_receipt above).
    assert receipt_a1["config"]["seed"] == 3
    assert receipt_a2["config"]["seed"] == 3
    assert receipt_b["config"]["seed"] == 9
    assert receipt_a1["untrained_baseline_seed"] == 3
    assert receipt_b["untrained_baseline_seed"] == 9

    state_a1 = _final_state_dict(receipt_a1)
    state_a2 = _final_state_dict(receipt_a2)
    state_b = _final_state_dict(receipt_b)

    assert set(state_a1) == set(state_a2) == set(state_b) and state_a1, (
        "empty or mismatched state dicts"
    )

    for name in state_a1:
        assert torch.equal(state_a1[name], state_a2[name]), (
            f"two seed=3 runs diverged at parameter {name!r} -- the seed is not fully "
            f"governing initial weights, cap sampling and pair shuffle"
        )

    diverged = any(not torch.equal(state_a1[name], state_b[name]) for name in state_a1)
    assert diverged, "seed=3 and seed=9 produced a bit-identical checkpoint"
