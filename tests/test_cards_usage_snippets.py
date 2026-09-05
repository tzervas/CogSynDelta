"""Tests for the `<!-- exec -->`-marked "How to use" python snippets in
`cogsyndelta.cards`'s templates (`region_variant`, `region_main`, `memory`).

A model card's usage instructions rot silently the moment the code they show stops
matching the real API -- nobody runs a README's code block by hand. `<!-- exec -->`
marks exactly the snippets this file extracts (via regex, off the CARD'S OWN rendered
markdown, never a hand-copied duplicate of the template source) and actually executes,
on CPU, against a tiny deterministic checkpoint committed at
`tests/fixtures/cards/tiny_checkpoint.pt` (+ its packed `.ptq.pt` sibling) -- so a
future edit that breaks the import, the variable name, or the loader call fails a test,
not a user's first `git clone`.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from cogsyndelta.cards.render import render_card
from tests.test_publish_checkpoint import (
    make_checkpoint,
    make_eval_receipt,
    make_quant_receipt,
    make_training_receipt,
    mod,
)

pytestmark = pytest.mark.cpu

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CHECKPOINT = REPO_ROOT / "tests" / "fixtures" / "cards" / "tiny_checkpoint.pt"
FIXTURE_PTQ = REPO_ROOT / "tests" / "fixtures" / "cards" / "tiny_checkpoint.ptq.pt"

_EXEC_BLOCK_RE = re.compile(r"<!-- exec -->\n```python\n(.*?)```", re.DOTALL)


def _exec_blocks(card: str) -> list[str]:
    """Every `<!-- exec -->`-marked python fenced block's source, read off the
    RENDERED card -- not a copy of the template kept in this file, which could drift
    from what the template actually produces without either copy noticing.
    """
    return [m.group(1) for m in _EXEC_BLOCK_RE.finditer(card)]


@pytest.fixture(autouse=True)
def _allow_tmp_path_as_checkpoint_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same allow-list extension every other file that reuses `tests.test_publish_
    checkpoint`'s fixture builders applies to itself -- required again here, see those
    other files' own copies of this fixture for why."""
    monkeypatch.setattr(mod, "ALLOWED_CHECKPOINT_ROOTS", [*mod.ALLOWED_CHECKPOINT_ROOTS, tmp_path])


_REGION_CFG: dict[str, object] = {
    "kind": "contrastive_encoder",
    "modality": "text",
    "role": "test role",
    "router_trigger": "test trigger",
    "licence_tier": "mit",
    "licence_why": "no NC or share-alike input in the catalogue",
}


def _full_receipts(tmp_path: Path, region: str = "compress") -> dict[str, dict[str, Any]]:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region=region)
    eval_path = make_eval_receipt(tmp_path, checkpoint, region=region)
    quant_path = make_quant_receipt(tmp_path, checkpoint, region=region)
    return {
        "train": json.loads(train_path.read_text()),
        "eval": json.loads(eval_path.read_text()),
        "quant": json.loads(quant_path.read_text()),
    }


def _render(kind: str, receipts: dict[str, object], tmp_path: Path, **cfg_overrides: object) -> str:
    cfg = dict(_REGION_CFG)
    cfg.update(cfg_overrides)
    return render_card(
        kind, region="compress", region_cfg=cfg, receipts=receipts, files={}, budgets_root=tmp_path
    )


# =====================================================================================
# Fixture sanity.
# =====================================================================================


def test_fixture_files_exist_and_are_readable_checkpoints() -> None:
    assert FIXTURE_CHECKPOINT.is_file()
    assert FIXTURE_PTQ.is_file()
    state = torch.load(FIXTURE_CHECKPOINT, map_location="cpu", weights_only=True)
    assert set(state) == {"weight", "bias"}


# =====================================================================================
# The markers survive rendering, in the right count for has_quant / not.
# =====================================================================================


@pytest.mark.parametrize("kind", ["region_variant", "region_main", "memory"])
def test_two_exec_blocks_when_quantized(tmp_path: Path, kind: str) -> None:
    receipts = _full_receipts(tmp_path, region="memory" if kind == "memory" else "compress")
    region = "memory" if kind == "memory" else "compress"
    cfg = dict(_REGION_CFG)
    if kind == "region_main":
        cfg.update(release_tag="v0.0.1-test", how_chosen="test fixture")
    card = render_card(
        kind, region=region, region_cfg=cfg, receipts=receipts, files={}, budgets_root=tmp_path
    )
    assert len(_exec_blocks(card)) == 2


def test_one_exec_block_fp32_only(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    receipts = {"train": json.loads(train_path.read_text())}
    card = _render("region_variant", receipts, tmp_path)
    assert len(_exec_blocks(card)) == 1


# =====================================================================================
# The snippets actually run, against the tiny committed fixture.
# =====================================================================================


def _run_in(work_dir: Path, source: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(compile(source, "<card-usage-snippet>", "exec"), namespace)  # noqa: S102 -- the point of this file
    return namespace


def test_load_checkpoint_snippet_actually_loads_the_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    receipts = {"train": json.loads(train_path.read_text())}
    card = _render("region_variant", receipts, tmp_path)
    blocks = _exec_blocks(card)
    assert len(blocks) == 1

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    shutil.copy(FIXTURE_CHECKPOINT, work_dir / "final.pt")
    monkeypatch.chdir(work_dir)

    namespace = _run_in(work_dir, blocks[0])
    assert set(namespace["state"]) == {"weight", "bias"}
    fixture_state = torch.load(FIXTURE_CHECKPOINT, map_location="cpu", weights_only=True)
    for name, tensor in fixture_state.items():
        assert torch.equal(namespace["state"][name], tensor)


def test_load_quantized_snippet_actually_unpacks_the_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipts = _full_receipts(tmp_path)
    card = _render("region_variant", receipts, tmp_path)
    blocks = _exec_blocks(card)
    assert len(blocks) == 2

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    shutil.copy(FIXTURE_CHECKPOINT, work_dir / "final.pt")
    shutil.copy(FIXTURE_PTQ, work_dir / "final.ptq.pt")
    monkeypatch.chdir(work_dir)

    namespace = _run_in(work_dir, blocks[1])
    assert set(namespace["state_dict"]) == {"weight", "bias"}
    # unpacked back to plain fp32 tensors of the right shape -- not merely "a dict
    # with the right keys" (which a no-op stub could satisfy too).
    assert namespace["state_dict"]["weight"].shape == (4, 8)
    assert namespace["state_dict"]["bias"].shape == (4,)


# =====================================================================================
# Mutation proof: this file's own exec harness catches a real breakage, not just
# whatever text happens to be in the template today.
# =====================================================================================


def test_exec_harness_catches_a_broken_snippet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snippet that imports a module which does not exist -- the shape of the actual
    defect class this file exists to catch (an API rename the template's prose was
    never updated for) -- must raise, proving `_run_in` is not a silent no-op."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    broken = "from cogsyndelta.quant.ptq import load_packed_artifact_TYPO_DOES_NOT_EXIST\n"
    with pytest.raises(ImportError):
        _run_in(work_dir, broken)
