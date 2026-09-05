"""Visual region card: Mix B fixtures render through cogsyndelta.cards + csd-card.

Covers the g44 card rules against the 24-step smoke receipts (H1 FAILS honestly;
this is not a real H1 pass):
- Mix B train receipt + eval/quant trio renders `region_variant`
- deployed `parameters` is eval `provenance.parameters` (10,712,448), training
  I-JEPA count stated separately
- H1 table (untrained vs trained vs threshold) from the receipts, result FAIL
- transfer set named Fashion t10k, n_eval 2000
- `visual_pairs: NOT MEASURED`
- CLEVR TASL attribution when the CLI wires `visual_attribution_block`
- datasets list = Mix B shard landings
- hub id `tzervas/cogsyndelta-vl-jepa` when `--repo` is passed
- publish guard: Mix B PASS, altered fingerprint BLOCK
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from cogsyndelta.cards.render import render_card
from tests.test_csd_card_cli import cli

pytestmark = pytest.mark.cpu

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cards" / "visual"
TRAIN = FIXTURES / "visual-20260905T202258Z.json"
EVAL = FIXTURES / "cogsyndelta-visual-eval-20260905T202332Z.json"
QUANT = FIXTURES / "visual-quant-20260905T205025Z.json"
EVAL_Q = FIXTURES / "cogsyndelta-visual-eval-quantized-20260905T205058Z.json"

HUB = "tzervas/cogsyndelta-vl-jepa"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _visual_cfg(**overrides: object) -> dict[str, object]:
    cfg: dict[str, object] = {
        "kind": "i-jepa",
        "modality": "vision",
        "role": "I-JEPA EMA target encoder over RGB images; maps an image to a [B, D] stream vector.",
        "router_trigger": "RGB image input",
        "licence_tier": "mit",
        "licence_why": "Mix B includes CLEVR (CC BY 4.0, ATTRIBUTION)",
        "stream_dim": 512,
        "hidden_dim": 1024,
    }
    cfg.update(overrides)
    return cfg


def _fixture_receipts() -> dict[str, dict[str, Any]]:
    return {
        "train": _load(TRAIN),
        "eval": _load(EVAL),
        "quant": _load(QUANT),
        "eval_quantized": _load(EVAL_Q),
    }


def test_visual_fixture_receipts_are_mix_b() -> None:
    train = _load(TRAIN)
    assert train["corpus"]["corpus_source"] == "visual-clean-v1"
    assert train["corpus"]["fingerprint"] == "ab5761b65714e4ba4d7c36df095f3599"
    pub = cli._load_publish_module()
    assert pub._visual_receipt_is_admitted(train) is True
    assert pub.licence_tier("visual", receipt=train) == "mit"
    attr = pub.visual_attribution_block(train)
    assert any("CLEVR" in line for line in attr)
    assert any("creativecommons.org/licenses/by/4.0" in line for line in attr)
    assert len(attr) >= 7


def test_visual_publish_guard_blocks_altered_fingerprint() -> None:
    pub = cli._load_publish_module()
    rec = _load(TRAIN)
    rec["corpus"] = dict(rec["corpus"])
    rec["corpus"]["fingerprint"] = "0" * 32
    with pytest.raises(pub.PublishAbortError, match="BLOCKING"):
        pub.licence_tier("visual", receipt=rec)
    assert pub.visual_attribution_block(rec) == []


def test_render_visual_fixture_card_states_h1_fail_and_deployed_params(tmp_path: Path) -> None:
    receipts = _fixture_receipts()
    card = render_card(
        "region_variant",
        region="visual",
        region_cfg=_visual_cfg(),
        receipts=receipts,
        files={},
        budgets_root=tmp_path,
        repo=HUB,
    )
    assert "repo `tzervas/cogsyndelta-vl-jepa`" in card
    assert "visual_pairs: NOT MEASURED" in card
    assert "| H1 | FAIL |" in card
    assert "threshold" in card
    assert "eurosat-test" in card
    assert "phelber/eurosat-rgb-128" in card
    assert "fashion-t10k" in card
    assert "zalando/fashion-mnist" in card
    assert "| n_eval | 2000 |" in card
    assert "parameters (deployed EMA target encoder)" in card
    assert "10.712 M" in card
    assert "parameters (training I-JEPA module)" in card
    assert "22.905 M" in card
    assert "compression ratio (storage, not speed)" in card
    assert "nyuuzyou/pxhere" in card  # datasets front matter
    assert "facebookresearch/clevr" in card
    assert "Anisotropy is a representation-geometry diagnostic" in card
    # smoke 24-step: trained 0.6246 < threshold ~0.635; do not invent a pass
    assert "| H1 | PASS |" not in card


def test_visual_overview_comes_from_receipt_jepa_not_stale_config(tmp_path: Path) -> None:
    """(1) type/geometry/modality from receipt config.jepa + corrected catalogue
    fields -- not stream_dim 512 / hidden_dim 1024 / jepa_predictor / latent."""
    card = render_card(
        "region_variant",
        region="visual",
        region_cfg=_visual_cfg(),
        receipts=_fixture_receipts(),
        files={},
        budgets_root=tmp_path,
        repo=HUB,
    )
    assert "| type | i-jepa |" in card
    assert "| image_size | 128 |" in card
    assert "| patch_size | 8 |" in card
    assert "| dim | 384 |" in card
    assert "| depth | 6 |" in card
    assert "| n_heads | 6 |" in card
    assert "| pos_kind | sincos2d |" in card
    assert "| modality | image |" in card
    assert "jepa_predictor" not in card
    assert "stream_dim" not in card
    assert "hidden_dim" not in card
    assert "Latent visual reasoning" not in card
    assert "Stream carrying visual latents" not in card
    assert "Triggered by: RGB image input." in card
    assert "latents.." not in card
    assert "| n_eval | **5400** |" not in card
    assert "| n_eval | 5400 |" in card


def test_visual_how_to_is_ijepa_not_text_encoder_and_is_not_exec(tmp_path: Path) -> None:
    """(2) Never a TextEncoder snippet on a visual card. Snippets are not
    `<!-- exec -->` because the card-snippet harness has no tiny I-JEPA checkpoint
    (tests/fixtures/cards/tiny_checkpoint.pt is a 4x8 linear layer)."""
    card = render_card(
        "region_variant",
        region="visual",
        region_cfg=_visual_cfg(),
        receipts=_fixture_receipts(),
        files={},
        budgets_root=tmp_path,
    )
    assert "TextEncoder" not in card
    assert "from cogsyndelta.model.vl_jepa import IJEPA" in card
    assert "wrap_deployed_visual_encoder" in card
    assert "DeployedVisualEncoder" in card
    assert "[B, D]" in card
    assert "model.encode(images)" in card
    assert "target_encoder.*" in card
    assert "--regions visual" in card
    assert "--train-receipt" in card
    assert "--quantized" in card
    assert "csd-quantize.py" in card
    assert "--checkpoint final.pt" not in card
    assert "<!-- exec -->\n```python" not in card


def test_visual_footnotes_print_only_the_probe_branch(tmp_path: Path) -> None:
    """(3) beats_untrained_eval / fp32_metric_recomputed / within_budget footnotes
    are the visual formula, not the text rank.recall@1 slash."""
    card = render_card(
        "region_variant",
        region="visual",
        region_cfg=_visual_cfg(),
        receipts=_fixture_receipts(),
        files={},
        budgets_root=tmp_path,
    )
    assert "probe.top1 > untrained_baseline.top1" in card
    assert "rank.recall@1 (eval battery) > the training receipt's untrained_baseline" not in card
    assert "EuroSAT linear-probe top-1 measured fresh" in card
    assert "quant.drop_probe_top1 <= tolerance" in card
    assert "quant.drop_recall@1 <= tolerance" not in card


def test_visual_model_index_splits_eurosat_and_fashion(tmp_path: Path) -> None:
    """(4) one model-index dataset per probe set, named from the receipt."""
    from huggingface_hub import ModelCard

    card = render_card(
        "region_variant",
        region="visual",
        region_cfg=_visual_cfg(),
        receipts=_fixture_receipts(),
        files={},
        budgets_root=tmp_path,
    )
    data = ModelCard(card).data
    results = list(data.eval_results or [])
    by_metric = {r.metric_type: r for r in results}
    assert by_metric["probe.top1"].dataset_name == "eurosat-test"
    assert by_metric["probe.top1"].dataset_type == "phelber/eurosat-rgb-128"
    assert by_metric["transfer.top1"].dataset_name == "fashion-t10k"
    assert by_metric["transfer.top1"].dataset_type == "zalando/fashion-mnist"
    assert "cogsyndelta-visual-holdout" not in card


def test_csd_card_cli_renders_mix_b_cell_with_attribution_and_hub(
    tmp_path: Path,
) -> None:
    cell_dir = tmp_path / "visual-b128-s0-fixture"
    cell_dir.mkdir()
    receipts_dir = cell_dir / "receipts"
    receipts_dir.mkdir()
    train_path = receipts_dir / TRAIN.name
    eval_path = receipts_dir / EVAL.name
    quant_path = receipts_dir / QUANT.name
    eval_q_path = receipts_dir / EVAL_Q.name
    train_path.write_text(TRAIN.read_text())
    eval_path.write_text(EVAL.read_text())
    quant_path.write_text(QUANT.read_text())
    eval_q_path.write_text(EVAL_Q.read_text())
    (cell_dir / "cell.json").write_text(
        json.dumps(
            {
                "cell_id": cell_dir.name,
                "region": "visual",
                "stages": {
                    "train": {"state": "done", "receipt": str(train_path)},
                    "test": {"state": "done", "receipt": str(eval_path)},
                    "quantize": {"state": "done", "receipt": str(quant_path)},
                    "test-quant": {"state": "done", "receipt": str(eval_q_path)},
                },
            }
        )
    )
    out = tmp_path / "README.md"
    rc = cli.main(
        [
            "--cell",
            str(cell_dir),
            "--kind",
            "region_variant",
            "--region",
            "visual",
            "--repo",
            HUB,
            "--out",
            str(out),
        ]
    )
    assert rc == 0, out.exists() and out.read_text()[:500]
    card = out.read_text()
    assert "license: mit" in card or "mit" in card.split("\n", 20)[0:20]
    assert "CLEVR" in card
    assert "CC BY 4.0" in card
    assert "repo `tzervas/cogsyndelta-vl-jepa`" in card
    assert "| H1 | FAIL |" in card
    assert "visual_pairs: NOT MEASURED" in card


def test_csd_card_cli_mix_b_explicit_receipts_mutation_restored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same Mix B train file, fingerprint flipped, CLI must BLOCK and write nothing.
    Restore the pin and the same flags succeed (mutation proof).
    """
    good = _load(TRAIN)
    bad = json.loads(json.dumps(good))
    bad["corpus"]["fingerprint"] = "ff" * 16
    good_path = tmp_path / "good.json"
    bad_path = tmp_path / "bad.json"
    good_path.write_text(json.dumps(good))
    bad_path.write_text(json.dumps(bad))
    out = tmp_path / "README.md"

    rc_bad = cli.main(
        [
            "--region",
            "visual",
            "--kind",
            "region_variant",
            "--train-receipt",
            str(bad_path),
            "--eval-receipt",
            str(EVAL),
            "--out",
            str(out),
        ]
    )
    assert rc_bad == 2
    assert not out.exists()
    assert "BLOCKING" in capsys.readouterr().err

    rc_good = cli.main(
        [
            "--region",
            "visual",
            "--kind",
            "region_variant",
            "--train-receipt",
            str(good_path),
            "--eval-receipt",
            str(EVAL),
            "--quant-receipt",
            str(QUANT),
            "--eval-quantized-receipt",
            str(EVAL_Q),
            "--repo",
            HUB,
            "--out",
            str(out),
        ]
    )
    assert rc_good == 0
    assert out.is_file()
    assert "CLEVR" in out.read_text()


def test_csd_card_subprocess_uses_this_tree_not_an_installed_package(
    tmp_path: Path,
) -> None:
    """model-matrix publish runs `{python} scripts/csd-card.py` as a subprocess.
    The script must load THIS checkout's `src/` even when a different
    `cogsyndelta` is already installed in `python`'s site-packages."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "csd-card.py"
    out = tmp_path / "README.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--region",
            "visual",
            "--kind",
            "region_variant",
            "--train-receipt",
            str(TRAIN),
            "--eval-receipt",
            str(EVAL),
            "--quant-receipt",
            str(QUANT),
            "--eval-quantized-receipt",
            str(EVAL_Q),
            "--repo",
            HUB,
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(script.parent.parent),
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert proc.returncode == 0, proc.stderr
    card = out.read_text()
    assert "| H1 | FAIL |" in card
    assert "CLEVR" in card
    assert "10.712 M" in card
