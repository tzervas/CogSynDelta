"""Regression guard: `scripts/csd-train-all.py --regions memory` must dispatch to
`cogsyndelta.regions.memory.run_memory_pretrain` -- the ONLY entry point that trains
`memory`, runs the BEIR full-57,638-passage FiQA eval, the BM25 reference and the five
pre-registered W4 gates (`cogsyndelta.eval.beir_fiqa.w4_gates`), and writes all three
into the receipt -- rather than the generic `run_region` -> `pretrain_region` path every
other text region uses, which silently trains `memory` with none of that: only
`held_out`/`graded_held_out` land in the receipt, and the five W4 gates never run.

Before this fix, `main()`'s dispatch loop had no branch for `memory` at all, so it fell
into the `else: receipt = run_region(...)` clause exactly like `code`/`compress`/
`retrieve`/`reason` -- a `csd-train@memory.service` unit (which invokes this script, not
`python -m cogsyndelta.regions.memory`) silently skipped its own acceptance gates. See
`run_memory_region`'s and `REGION_RUNNERS`'s docstrings in `scripts/csd-train-all.py` for
the full account.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

pytestmark = pytest.mark.cpu

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "csd-train-all.py"

# The exact `main()` dispatch clause the fix adds -- see `scripts/csd-train-all.py`'s
# `REGION_RUNNERS` table and its use in `main()`'s per-region `try:` block. Used verbatim
# by `test_reverting_the_dispatch_hook_makes_the_regression_test_fail` to construct a
# scratch copy of the script with the fix reverted -- i.e. the exact pre-fix bug.
DISPATCH_HOOK = (
    "            elif name in REGION_RUNNERS:\n"
    "                receipt = REGION_RUNNERS[name](\n"
    "                    name,\n"
    "                    state,\n"
    "                    args.steps,\n"
    "                    args.batch,\n"
    "                    args.dry_run,\n"
    "                    args.max_len,\n"
    "                    args.init_embedding_from,\n"
    "                    args.seed,\n"
    "                )\n"
)


def _load_module(name: str, *, source: str | None = None, tmp_path: Path | None = None):
    """Load `csd-train-all.py` as a fresh module under a unique name -- the convention
    every other consumer of this hyphenated script's tests uses (see
    `tests/test_token_aware_dry_run_plan.py`, `tests/test_region_memory.py`).

    `source`/`tmp_path` (both required together) load a MUTATED copy of the script's
    text from a scratch file instead of the real one on disk -- see
    `test_reverting_the_dispatch_hook_makes_the_regression_test_fail`.
    """
    if source is None:
        path = SCRIPT_PATH
    else:
        assert tmp_path is not None
        path = tmp_path / f"{name}.py"
        path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_module("csd_train_all_memory_dispatch_test")


def _dry_run_memory_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> str:
    """`run_memory_region(..., dry=True)`'s printed plan, with I/O bypassed (same pattern
    `tests/test_token_aware_dry_run_plan.py` uses for `run_region`)."""
    result = mod.run_memory_region(name="memory", state=tmp_path, steps=10, batch=8, dry=True)
    assert result is None  # dry runs never return a receipt
    return capsys.readouterr().out


def test_memory_dry_run_plan_names_the_memory_entry_point_and_the_w4_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """(a) A reader scanning `--regions memory --dry-run` output must see that this row
    runs something OTHER than the generic pretrain path -- the exact phrase this fix
    prints, the real entry point's dotted name, and the W4 gates module."""
    out = _dry_run_memory_plan(tmp_path, monkeypatch, capsys)
    assert "pretrain + BEIR eval + w4 gates" in out
    assert "cogsyndelta.regions.memory.run_memory_pretrain" in out
    assert "cogsyndelta.eval.beir_fiqa.w4_gates" in out
    # And NOT the generic path's own marker -- this plan is not `run_region`'s.
    assert "resolved PretrainConfig (dry run, no training started):" not in out


def test_memory_dry_run_plan_reflects_max_len_override_and_init_embedding_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    fake_ckpt = str(tmp_path / "retrieve-final.pt")
    result = mod.run_memory_region(
        name="memory",
        state=tmp_path,
        steps=10,
        batch=8,
        dry=True,
        max_len=256,
        init_embedding_from=fake_ckpt,
    )
    assert result is None
    out = capsys.readouterr().out
    assert '"max_len": 256' in out
    assert fake_ckpt in out
    assert '"applied": true' in out.lower()


# ---------------------------------------------------------------------------------------
# (b) CLI-level dispatch: `--regions memory` calls `run_memory_pretrain`, never
# `pretrain_region`; `--regions compress` (an ordinary text region) is unaffected and
# still calls `pretrain_region`.
# ---------------------------------------------------------------------------------------


def _fake_generic_receipt(*, region: str) -> dict[str, Any]:
    """Shape `pretrain_region` returns -- everything `run_region`'s tail print and
    `main()`'s own gate bookkeeping read out of a receipt."""
    return {
        "region": region,
        "untrained_baseline": {"recall@1": 0.10, "recall@10": 0.30},
        "held_out": {"recall@1": 0.50, "recall@10": 0.80, "mrr": 0.60},
        "beats_untrained": {"recall@1": True, "recall@10": True},
        "parameters": 16_021_248,
        "receipt_path": None,
        "resumed": False,
    }


def _fake_memory_receipt() -> dict[str, Any]:
    """Shape `run_memory_pretrain` returns: `pretrain_region`'s receipt PLUS the
    `retrieval`/`gates` blocks `run_memory_region`'s own tail print reads via `.get`."""
    receipt = _fake_generic_receipt(region="memory")
    receipt["gates"] = {"passed": True}
    return receipt


def _assert_memory_and_compress_dispatch_correctly(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive `module.main()` with `--regions memory,compress`, `run_memory_pretrain` and
    `pretrain_region` both monkeypatched to recorders instead of real training, and
    assert: `memory` calls `run_memory_pretrain` exactly once, with the same CLI knobs
    `run_region` derives for every other region (steps, batch, derived lr, max_len, state
    root); `compress` calls `pretrain_region` exactly once, unaffected; and
    `run_memory_pretrain` is NEVER called for `compress`, nor `pretrain_region` for
    `memory`.

    Factored out of the test body (rather than inlined) so
    `test_reverting_the_dispatch_hook_makes_the_regression_test_fail` can run the
    IDENTICAL assertions against a scratch copy of the script with the fix reverted, and
    show they actually fail -- the mutation proof for this file's own regression
    coverage.
    """
    monkeypatch.setattr(module, "CORPUS", tmp_path)
    # compress's single source + its STS-B graded source are stubbed the same way
    # `tests/test_token_aware_dry_run_plan.py`'s own dry-run helper stubs them: any glob
    # resolves to one fake shard, and no shard is ever schema-mismatched. `memory`'s own
    # dispatch (via REGION_RUNNERS, when the fix is present) never calls either of these
    # -- it delegates corpus resolution entirely to the (monkeypatched)
    # `run_memory_pretrain` -- so this stubbing is here only for `compress`'s sake.
    monkeypatch.setattr(
        module, "_shards", lambda pattern, root=module.CORPUS: [str(tmp_path / "shard.parquet")]
    )
    monkeypatch.setattr(module, "_schema_mismatch", lambda shards, cols: None)

    memory_calls: list[dict[str, Any]] = []

    def fake_run_memory_pretrain(**kwargs: Any) -> dict[str, Any]:
        memory_calls.append(kwargs)
        return _fake_memory_receipt()

    pretrain_region_calls: list[Any] = []

    def fake_pretrain_region(cfg: Any) -> dict[str, Any]:
        pretrain_region_calls.append(cfg)
        return _fake_generic_receipt(region=getattr(cfg, "region", "?"))

    monkeypatch.setattr("cogsyndelta.regions.memory.run_memory_pretrain", fake_run_memory_pretrain)
    monkeypatch.setattr("cogsyndelta.regions.pretrain_region", fake_pretrain_region)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "csd-train-all.py",
            "--regions",
            "memory,compress",
            "--steps",
            "5",
            "--batch",
            "4",
            "--state",
            str(tmp_path),
        ],
    )
    rc = module.main()
    assert rc == 0, "both fake receipts beat their untrained baseline; rc must be 0"

    assert len(memory_calls) == 1, (
        f"expected exactly one run_memory_pretrain call (for memory), got {memory_calls}"
    )
    assert len(pretrain_region_calls) == 1, (
        f"expected exactly one pretrain_region call (for compress), got "
        f"{[getattr(c, 'region', c) for c in pretrain_region_calls]}"
    )
    assert pretrain_region_calls[0].region == "compress"

    kwargs = memory_calls[0]
    assert kwargs["steps"] == 5
    assert kwargs["batch_size"] == 4
    assert kwargs["lr"] == module.lr_for_batch(4)
    assert kwargs["max_len"] == module.region_spec("memory").default_max_len
    assert kwargs["out_dir"] == str(tmp_path / "receipts")
    assert kwargs["retrieve_checkpoint"] is None
    # --seed was not passed on argv; must still default to 0 (PretrainConfig.seed's own
    # default), not be silently dropped.
    assert kwargs["seed"] == 0


def test_regions_memory_dispatches_to_run_memory_pretrain_not_pretrain_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_memory_and_compress_dispatch_correctly(mod, tmp_path, monkeypatch)


def test_regions_memory_forwards_init_embedding_from_as_retrieve_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "CORPUS", tmp_path)

    memory_calls: list[dict[str, Any]] = []

    def fake_run_memory_pretrain(**kwargs: Any) -> dict[str, Any]:
        memory_calls.append(kwargs)
        return _fake_memory_receipt()

    monkeypatch.setattr("cogsyndelta.regions.memory.run_memory_pretrain", fake_run_memory_pretrain)

    fake_ckpt = str(tmp_path / "retrieve-final.pt")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "csd-train-all.py",
            "--regions",
            "memory",
            "--steps",
            "5",
            "--batch",
            "4",
            "--state",
            str(tmp_path),
            "--init-embedding-from",
            fake_ckpt,
        ],
    )
    rc = mod.main()
    assert rc == 0
    assert memory_calls[0]["retrieve_checkpoint"] == fake_ckpt


def test_regions_memory_forwards_seed_via_region_runners_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--seed` must reach `run_memory_pretrain(seed=...)` through the `REGION_RUNNERS`
    dispatch path exactly as it reaches `PretrainConfig.seed` through the generic
    `run_region` path -- `memory` trains through its own entry point (see
    `run_memory_region`'s docstring), so it needs its own forwarding wire, not a free
    ride off `run_region`'s."""
    monkeypatch.setattr(mod, "CORPUS", tmp_path)

    memory_calls: list[dict[str, Any]] = []

    def fake_run_memory_pretrain(**kwargs: Any) -> dict[str, Any]:
        memory_calls.append(kwargs)
        return _fake_memory_receipt()

    monkeypatch.setattr("cogsyndelta.regions.memory.run_memory_pretrain", fake_run_memory_pretrain)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "csd-train-all.py",
            "--regions",
            "memory",
            "--steps",
            "5",
            "--batch",
            "4",
            "--state",
            str(tmp_path),
            "--seed",
            "7",
        ],
    )
    rc = mod.main()
    assert rc == 0
    assert memory_calls[0]["seed"] == 7


# ---------------------------------------------------------------------------------------
# (c) Mutation proof: revert the dispatch fix in a scratch copy of the script and show
# the SAME assertions above fail against it -- proof this test suite would have caught
# the original bug, not merely that it is plausible.
# ---------------------------------------------------------------------------------------


def test_reverting_the_dispatch_hook_makes_the_regression_test_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete the `elif name in REGION_RUNNERS:` branch from a scratch copy of the
    script -- reverting `main()`'s dispatch to exactly the pre-fix bug, where `memory`
    silently falls through to the generic `run_region` -> `pretrain_region` path -- and
    show `_assert_memory_and_compress_dispatch_correctly` (the body of this file's real
    regression test) raises `AssertionError` against it, naming `run_memory_pretrain` as
    what it expected and did not get."""
    source = SCRIPT_PATH.read_text()
    assert DISPATCH_HOOK in source, (
        "this test's mutation string no longer matches scripts/csd-train-all.py -- "
        "update DISPATCH_HOOK to match the current dispatch clause"
    )
    mutated_source = source.replace(DISPATCH_HOOK, "", 1)
    assert DISPATCH_HOOK not in mutated_source  # the mutation actually took

    mutated = _load_module(
        "csd_train_all_memory_dispatch_test_mutated", source=mutated_source, tmp_path=tmp_path
    )

    with pytest.raises(AssertionError, match="run_memory_pretrain"):
        _assert_memory_and_compress_dispatch_correctly(mutated, tmp_path, monkeypatch)
