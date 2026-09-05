"""Tests for `cogsyndelta.cards.render.render_card` -- the library's one public entry
point.

Covers, per the lane's TESTS requirement:

- Renders every receipt-fixture shape the repo already has (`make_training_receipt` /
  `make_eval_receipt` / `make_quant_receipt`, reused from `tests.test_publish_checkpoint`
  rather than re-invented) across all five card kinds.
- Renders the real receipts under
  `/akula-data/csd/matrix/code-b1280-s1-7bc2699-20260904/receipts/` -- ONLY if that
  path exists on this machine, skipped (not failed) otherwise.
- Front-matter round-trips through `ModelCard(content).data`.
- Model-index is present, under v2 metric names.
- A `metrics_schema` disagreement across the receipts merged into one card raises
  `CardError` (mutation proof: same receipts, stamps forced to agree, renders fine).
- An undocumented metric raises `CardError` (mutation proof: stub
  `METRIC_METHODOLOGY` empty and show a normally-fine render now fails).
- A v1-shaped (unstamped) receipt renders through the alias map, marked, with the v1
  footnote present in the body.
- A golden snapshot of one `region_variant` card, byte-for-byte, updated only via the
  explicit `CSD_CARDS_UPDATE_GOLDEN=1` environment flag.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from huggingface_hub import ModelCard

from cogsyndelta.cards.methodology import METRIC_METHODOLOGY, CardError
from cogsyndelta.cards.render import CARD_KINDS, render_card
from tests.test_publish_checkpoint import (
    make_checkpoint,
    make_eval_receipt,
    make_quant_receipt,
    make_training_receipt,
    mod,
)

pytestmark = pytest.mark.cpu

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "cards" / "region_variant.golden.md"
REAL_RECEIPT_DIR = Path("/akula-data/csd/matrix/code-b1280-s1-7bc2699-20260904/receipts")

_BASE_REGION_CFG: dict[str, object] = {
    "kind": "contrastive_encoder",
    "stream_dim": 512,
    "hidden_dim": 1024,
    "modality": "text",
    "role": "Docstring <-> function, search and generation.",
    "router_trigger": "language=python and a docstring is present.",
    "licence_tier": "mit",
    "licence_why": "no NC or share-alike input in the catalogue",
}


def region_cfg(**overrides: object) -> dict[str, object]:
    cfg = dict(_BASE_REGION_CFG)
    cfg.update(overrides)
    return cfg


@pytest.fixture(autouse=True)
def _allow_tmp_path_as_checkpoint_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same allow-list extension `tests/test_publish_checkpoint.py` applies to itself
    -- required again here since pytest does not apply another module's autouse
    fixture just because this one imports its helper functions."""
    monkeypatch.setattr(mod, "ALLOWED_CHECKPOINT_ROOTS", [*mod.ALLOWED_CHECKPOINT_ROOTS, tmp_path])


def _full_fixture_receipts(
    tmp_path: Path, region: str = "compress"
) -> dict[str, dict[str, object]]:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region=region)
    eval_path = make_eval_receipt(tmp_path, checkpoint, region=region)
    quant_path = make_quant_receipt(tmp_path, checkpoint, region=region)
    return {
        "train": json.loads(train_path.read_text()),
        "eval": json.loads(eval_path.read_text()),
        "quant": json.loads(quant_path.read_text()),
    }


# =====================================================================================
# Every fixture receipt shape the repo already has, across every card kind that takes
# receipts (the two receipt-free kinds -- placeholder, and composed with none supplied
# -- are covered separately below).
# =====================================================================================


@pytest.mark.parametrize("kind", ["region_variant", "region_main"])
def test_renders_the_full_fixture_trio(tmp_path: Path, kind: str) -> None:
    receipts = _full_fixture_receipts(tmp_path)
    cfg = (
        region_cfg(release_tag="v0.0.1-test", how_chosen="test fixture")
        if kind == "region_main"
        else region_cfg()
    )
    card = render_card(
        kind, region="compress", region_cfg=cfg, receipts=receipts, files={}, budgets_root=tmp_path
    )
    assert card.startswith("---\n")
    assert "## Evaluation results" in card
    assert "## Sizes" in card


def test_renders_train_only_fp32_card(tmp_path: Path) -> None:
    receipts = {"train": _full_fixture_receipts(tmp_path)["train"]}
    card = render_card(
        "region_variant",
        region="compress",
        region_cfg=region_cfg(),
        receipts=receipts,
        files={},
        budgets_root=tmp_path,
    )
    assert "### Training held-out battery" in card
    assert "No quant receipt" not in card  # this card never claims that -- own wording


# =====================================================================================
# `code_revision` normalization -- a real receipt's `code_revision` is a dict
# (`{"git_sha", "dirty", "branch", "describe"}`, see `cogsyndelta.pipeline.receipt.
# Receipt.code_revision`'s docstring), not a bare string. `render_card` must reduce it
# to a single checkout-able SHA at every site that prints it, never a Python dict repr
# -- pasting `git checkout {'git_sha': ...}` is not a command, and `{code revision
# {'git_sha': ...}}` unbalances BibTeX's braces.
# =====================================================================================

_REAL_CODE_REVISION: dict[str, object] = {
    "git_sha": "a7694090903664bc256b4b96d998b37cacd316cf",
    "dirty": False,
    "branch": "HEAD",
    "describe": "a769409",
}


def test_code_revision_dict_shape_normalized_to_bare_sha(tmp_path: Path) -> None:
    receipts = _full_fixture_receipts(tmp_path)
    receipts["train"]["code_revision"] = _REAL_CODE_REVISION
    card = render_card(
        "region_variant",
        region="compress",
        region_cfg=region_cfg(),
        receipts=receipts,
        files={},
        budgets_root=tmp_path,
    )
    sha = _REAL_CODE_REVISION["git_sha"]
    # never a raw dict repr anywhere on the card (header, provenance, checkout snippet,
    # bibtex note all interpolate the same `code_revision` value)
    assert "{'git_sha'" not in card
    assert f"git checkout {sha}" in card  # a command a reader can actually paste
    assert f"code revision `{sha}`" in card  # header line
    assert f"**Code revision:** `{sha}`" in card  # Provenance bullet
    assert f"code revision {sha}" in card  # bibtex note, unbroken braces
    # the bibtex block's own braces stay balanced now that the value has none of its own
    bibtex_start = card.index("```bibtex")
    bibtex_end = card.index("```", bibtex_start + 1)
    bibtex_block = card[bibtex_start:bibtex_end]
    assert bibtex_block.count("{") == bibtex_block.count("}")


def test_code_revision_legacy_string_shape_passes_through(tmp_path: Path) -> None:
    """A receipt written before the dict shape existed (or a test fixture) may still
    carry a bare string -- that must keep working unchanged."""
    receipts = _full_fixture_receipts(tmp_path)
    receipts["train"]["code_revision"] = "deadbeef"
    card = render_card(
        "region_variant",
        region="compress",
        region_cfg=region_cfg(),
        receipts=receipts,
        files={},
        budgets_root=tmp_path,
    )
    assert "git checkout deadbeef" in card
    assert "{'git_sha'" not in card


def test_code_revision_dict_shape_with_no_git_sha_falls_back_to_none_recorded(
    tmp_path: Path,
) -> None:
    """A malformed/legacy dict with no `git_sha` key must not crash or leak a dict
    repr -- it degrades to the same '(none recorded)' a missing field already gets."""
    receipts = _full_fixture_receipts(tmp_path)
    receipts["train"]["code_revision"] = {"dirty": True, "branch": "HEAD"}
    card = render_card(
        "region_variant",
        region="compress",
        region_cfg=region_cfg(),
        receipts=receipts,
        files={},
        budgets_root=tmp_path,
    )
    assert "{'dirty'" not in card
    assert "(none recorded)" in card


def test_renders_memory_kind_with_od17_status(tmp_path: Path) -> None:
    receipts = _full_fixture_receipts(tmp_path, region="memory")
    cfg = region_cfg(licence_tier="cc-by-nc-sa-4.0", licence_why="merge of compress+retrieve")
    card = render_card(
        "memory",
        region="memory",
        region_cfg=cfg,
        receipts=receipts,
        files={},
        budgets_root=tmp_path,
    )
    assert "OD-17" in card


def test_renders_placeholder_kind_with_no_receipts() -> None:
    card = render_card(
        "placeholder",
        region="stream_vae",
        region_cfg={"kind": "latent_vae", "modality": "any"},
        receipts={},
        files={},
    )
    assert "weights: none" in card


def test_placeholder_vl_latent_alias_renders_as_visual() -> None:
    card = render_card(
        "placeholder",
        region="vl_latent",
        region_cfg={"kind": "jepa", "modality": "image"},
        receipts={},
        files={},
    )
    assert "cogsyndelta-region-visual" in card
    assert "region:visual" in card
    assert "# CogSynDelta -- visual (placeholder)" in card
    assert "formerly `vl_latent`" in card


def test_renders_composed_kind_with_no_receipts() -> None:
    card = render_card("composed", region="cogsyndelta", region_cfg={}, receipts={}, files={})
    assert "TBD" in card


def test_every_card_kind_is_reachable_by_name() -> None:
    """`CARD_KINDS` must actually be every kind CARD SPEC names -- a drift guard for
    the constant the parametrized tests above assume is complete."""
    assert set(CARD_KINDS) == {"region_variant", "region_main", "memory", "placeholder", "composed"}


def test_repo_prints_in_header_when_supplied(tmp_path: Path) -> None:
    receipts = _full_fixture_receipts(tmp_path)
    card = render_card(
        "region_variant",
        region="compress",
        region_cfg=region_cfg(),
        receipts=receipts,
        files={},
        budgets_root=tmp_path,
        repo="tzervas/cogsyndelta-region-compress",
    )
    assert "repo `tzervas/cogsyndelta-region-compress`" in card


def test_repo_omitted_by_default_is_byte_identical_to_golden(tmp_path: Path) -> None:
    """Mutation-proof companion to the golden snapshot test: `repo=None` (the
    default) must render nothing extra -- proven here by comparing the SAME fixture's
    output with and without an explicit `repo=None`, not merely by omitting the
    argument (which the golden test already does)."""
    receipts = _full_fixture_receipts(tmp_path)
    kwargs = {
        "region": "compress",
        "region_cfg": region_cfg(),
        "receipts": receipts,
        "files": {},
        "budgets_root": tmp_path,
    }
    default_card = render_card("region_variant", **kwargs)
    explicit_none_card = render_card("region_variant", repo=None, **kwargs)
    assert default_card == explicit_none_card


# --------------------------------------------------------- region rename (2026-09-04)


def test_faculty_line_shown_when_region_cfg_carries_a_different_canonical_name(
    tmp_path: Path,
) -> None:
    """A cell recorded under the legacy `code` id, rendered with `region_cfg` loaded
    from the (now-renamed) catalogue -- which carries `name: "language"` and
    `specialisation: "code"` -- must show BOTH the faculty name and the specialisation,
    and say which legacy id the receipts were recorded under."""
    receipts = _full_fixture_receipts(tmp_path)
    card = render_card(
        "region_variant",
        region="code",
        region_cfg=region_cfg(name="language", specialisation="code"),
        receipts=receipts,
        files={},
        budgets_root=tmp_path,
    )
    assert "**Faculty:** `language`" in card
    assert "specialisation: `code`" in card
    assert "legacy region id `code`" in card


def test_specialisation_shown_with_no_alias_note_when_name_matches(tmp_path: Path) -> None:
    """When `region_cfg["name"]` already equals `region` (a canonical receipt, no
    rename involved), the specialisation still shows but there is nothing to call a
    legacy alias -- no 'legacy region id' text."""
    receipts = _full_fixture_receipts(tmp_path)
    card = render_card(
        "region_variant",
        region="language",
        region_cfg=region_cfg(name="language", specialisation="code"),
        receipts=receipts,
        files={},
        budgets_root=tmp_path,
    )
    assert "**Specialisation:** `code`" in card
    assert "legacy region id" not in card


def test_no_faculty_or_specialisation_line_when_region_cfg_carries_neither(
    tmp_path: Path,
) -> None:
    """The base fixture `region_cfg()` (no `name`, no `specialisation` -- what every
    other test in this file, including the golden snapshot, renders with) must add
    NEITHER line -- this is the regression the golden snapshot itself already pins,
    stated explicitly here so its intent survives a future golden-file update."""
    receipts = _full_fixture_receipts(tmp_path)
    card = render_card(
        "region_variant",
        region="compress",
        region_cfg=region_cfg(),
        receipts=receipts,
        files={},
        budgets_root=tmp_path,
    )
    assert "**Faculty:**" not in card
    assert "**Specialisation:**" not in card


def test_unknown_kind_raises_card_error() -> None:
    with pytest.raises(CardError, match="unknown card kind"):
        render_card(
            "not_a_real_kind", region="code", region_cfg=region_cfg(), receipts={}, files={}
        )


# =====================================================================================
# The real receipts, ONLY if present on this machine.
# =====================================================================================


@pytest.mark.skipif(
    not REAL_RECEIPT_DIR.is_dir(), reason=f"{REAL_RECEIPT_DIR} not present on this machine"
)
def test_renders_the_real_code_receipts() -> None:
    def load(name: str) -> dict[str, object]:
        return json.loads((REAL_RECEIPT_DIR / name).read_text())

    train = load("code-20260904T140242Z.json")
    quant = load("code-quant-20260904T140334Z.json")
    eval_receipt = load("cogsyndelta-code-eval-20260904T140306Z.json")
    eval_quantized = load("cogsyndelta-code-eval-quantized-20260904T140358Z.json")

    card = render_card(
        "region_variant",
        region="code",
        region_cfg=region_cfg(),
        receipts={
            "train": train,
            "eval": eval_receipt,
            "eval_quantized": eval_quantized,
            "quant": quant,
        },
        files={"final.pt": {"sha256": train.get("checkpoint_sha256")}},
        budgets_root=Path("/akula-data/csd/matrix/budgets"),
    )
    assert "## Evaluation results" in card
    # this real receipt trio is entirely unstamped -- every table should carry the v1 mark
    assert "v1 receipt; names mapped to csd-metrics/v2" in card
    # the real training-peak budget for this exact cell is on disk -- MEASURED, not estimated
    assert "training peak" in card
    # this real train receipt's `code_revision` IS the dict shape
    # (`{"git_sha", "dirty", "branch", "describe"}`) -- must render as the bare SHA
    # everywhere, never the Python dict repr (rejected review round 1, criterion 3).
    sha = train["code_revision"]["git_sha"]
    assert "{'git_sha'" not in card
    assert f"git checkout {sha}" in card
    # RETIRED as independently displayed values (METRICS-METHODOLOGY.md Sec 13) --
    # must not reach the model-index front matter even though the receipt carries them
    # (rejected review round 1, criterion 3).
    card_data = ModelCard(card).data
    model_index_metric_types = {r.metric_type for r in (card_data.eval_results or [])}
    assert "rank.map" not in model_index_metric_types
    assert "rank.precision@10" not in model_index_metric_types


# =====================================================================================
# Front matter round-trip + model-index under v2 names.
# =====================================================================================


def test_front_matter_round_trips_through_modelcard(tmp_path: Path) -> None:
    receipts = _full_fixture_receipts(tmp_path)
    card_md = render_card(
        "region_variant",
        region="compress",
        region_cfg=region_cfg(),
        receipts=receipts,
        files={},
        budgets_root=tmp_path,
    )
    card = ModelCard(card_md)
    assert card.data.license == "mit"
    assert card.data.pipeline_tag == "feature-extraction"
    assert "region:compress" in card.data.tags


def test_model_index_present_under_v2_names(tmp_path: Path) -> None:
    receipts = _full_fixture_receipts(tmp_path)
    card_md = render_card(
        "region_variant",
        region="compress",
        region_cfg=region_cfg(),
        receipts=receipts,
        files={},
        budgets_root=tmp_path,
    )
    card = ModelCard(card_md)
    assert card.data.eval_results
    metric_types = {r.metric_type for r in card.data.eval_results}
    # the fixture eval receipt carries rank.recall@1 and repr.anisotropy verbatim
    assert "rank.recall@1" in metric_types
    assert "repr.anisotropy" in metric_types


# =====================================================================================
# metrics_schema disagreement -> CardError, with a mutation proof (same receipts,
# stamps forced to agree, render succeeds).
# =====================================================================================


def test_schema_disagreement_raises_card_error(tmp_path: Path) -> None:
    receipts = _full_fixture_receipts(tmp_path)
    receipts["eval"] = dict(receipts["eval"], metrics_schema="csd-metrics/v2")
    with pytest.raises(CardError, match="disagree on metrics_schema"):
        render_card(
            "region_variant",
            region="compress",
            region_cfg=region_cfg(),
            receipts=receipts,
            files={},
            budgets_root=tmp_path,
        )


def test_schema_disagreement_mutation_proof_agreeing_stamps_render_fine(tmp_path: Path) -> None:
    """Same fixture, same shape -- only the stamps changed from disagreeing to
    agreeing -- proves the refusal above is about the disagreement, not merely about
    the presence of a `metrics_schema` field at all."""
    receipts = _full_fixture_receipts(tmp_path)
    receipts["train"] = dict(receipts["train"], metrics_schema="csd-metrics/v2")
    receipts["eval"] = dict(receipts["eval"], metrics_schema="csd-metrics/v2")
    receipts["quant"] = dict(receipts["quant"], metrics_schema="csd-metrics/v2")
    card = render_card(
        "region_variant",
        region="compress",
        region_cfg=region_cfg(),
        receipts=receipts,
        files={},
        budgets_root=tmp_path,
    )
    assert "**Metrics schema:** `csd-metrics/v2`" in card


# =====================================================================================
# Undocumented metric -> CardError, mutation proof.
# =====================================================================================


def test_undocumented_metric_raises_card_error(tmp_path: Path) -> None:
    receipts = _full_fixture_receipts(tmp_path)
    receipts["eval"]["metrics"]["rank.brand_new_undocumented_metric"] = 0.5
    with pytest.raises(CardError, match="brand_new_undocumented_metric"):
        render_card(
            "region_variant",
            region="compress",
            region_cfg=region_cfg(),
            receipts=receipts,
            files={},
            budgets_root=tmp_path,
        )


def test_undocumented_metric_mutation_proof_stubbed_methodology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A render that succeeds today must fail once `METRIC_METHODOLOGY` is emptied --
    proving the refusal is load-bearing on the render path, not merely unit-tested in
    isolation on `tables.py`."""
    receipts = _full_fixture_receipts(tmp_path)
    render_card(
        "region_variant",
        region="compress",
        region_cfg=region_cfg(),
        receipts=receipts,
        files={},
        budgets_root=tmp_path,
    )  # sanity: succeeds before the mutation
    # `require_documented` (called from `cogsyndelta.cards.tables`'s build_* functions,
    # every one of which `render_card` calls with `methodology=None`, i.e. "use the
    # module default") resolves `METRIC_METHODOLOGY` as `cogsyndelta.cards.methodology`'s
    # OWN global at call time -- patch that module's attribute, not a name some other
    # module already imported (an `import`-time copy a later `setattr` elsewhere can't
    # reach).
    monkeypatch.setattr("cogsyndelta.cards.methodology.METRIC_METHODOLOGY", {})
    with pytest.raises(CardError):
        render_card(
            "region_variant",
            region="compress",
            region_cfg=region_cfg(),
            receipts=receipts,
            files={},
            budgets_root=tmp_path,
        )
    assert METRIC_METHODOLOGY  # the real, un-monkeypatched table is untouched


# =====================================================================================
# v1 receipt -> mapped + footnote.
# =====================================================================================


def test_v1_receipt_renders_through_the_alias_map_with_footnote(tmp_path: Path) -> None:
    receipts = _full_fixture_receipts(tmp_path)
    card = render_card(
        "region_variant",
        region="compress",
        region_cfg=region_cfg(),
        receipts=receipts,
        files={},
        budgets_root=tmp_path,
    )
    # make_eval_receipt's repr.effective_rank_ratio is a v1 (pre-g7) name
    assert "repr.effective_rank_entropy_ratio" in card
    assert "v1 receipt; names mapped to csd-metrics/v2" in card
    # make_quant_receipt's quantized_metric/drop/compression_ratio are v1 names too
    assert "quant.plan_recall@1" in card
    assert "quant.compression_ratio" in card


# =====================================================================================
# Golden snapshot.
# =====================================================================================


def test_golden_snapshot_region_variant(tmp_path: Path) -> None:
    """Byte-for-byte snapshot of one `region_variant` card, built from a fully
    deterministic fixture (fixed checkpoint bytes -> fixed sha256; `torch.manual_seed(0)`
    in `write_packed_artifact`; no `code_revision` field in the fixture receipts, so
    that line reads the same fixed '(none recorded)' every run; `budgets_root=tmp_path`
    with no budget file, so 'training peak' is deterministically absent rather than
    reading whatever this machine's real `/akula-data/csd/matrix/budgets` currently
    holds).

    Update the golden file (after confirming a diff is intentional) with:
        CSD_CARDS_UPDATE_GOLDEN=1 pytest tests/test_cards_render.py -k golden_snapshot
    """
    receipts = _full_fixture_receipts(tmp_path)
    card = render_card(
        "region_variant",
        region="compress",
        region_cfg=region_cfg(),
        receipts=receipts,
        files={"final.pt": {"sha256": receipts["train"]["artifacts"]["checkpoint_sha256"]}},
        budgets_root=tmp_path,
    )

    if os.environ.get("CSD_CARDS_UPDATE_GOLDEN") == "1":
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(card)
        pytest.skip(
            "golden file updated (CSD_CARDS_UPDATE_GOLDEN=1) -- re-run without the flag to verify"
        )

    assert GOLDEN_PATH.is_file(), (
        f"{GOLDEN_PATH} missing -- run with CSD_CARDS_UPDATE_GOLDEN=1 once to create it"
    )
    golden = GOLDEN_PATH.read_text()
    assert card == golden, (
        "rendered card no longer matches the golden snapshot -- if this change is "
        "intentional, re-run with CSD_CARDS_UPDATE_GOLDEN=1 to update it"
    )
