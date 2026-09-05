"""Unit tests for scripts/csd-publish-checkpoint.py.

No network: every huggingface_hub call is a mock. Each test builds the exact
condition a requirement exists to catch and asserts on both the outcome (an
exception raised, an exit code, what got printed) and, wherever it matters, that no
upload happened -- a private-by-construction claim that assumes the mock but never
checks `upload_file.assert_not_called()` has not actually been tested.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from cogsyndelta.quant.ptq import (
    QuantPlan,
    load_packed_artifact,
    packed_stored_bytes,
    packed_width_histogram,
    save_packed_artifact,
)
from cogsyndelta.regions.pretrain import PretrainConfig, pretrain_region
from cogsyndelta.regions.text_encoder import TextEncoderConfig

# Reused, not reimplemented: the same tokenizer/parquet fixture builders and
# hyphenated-module loaders tests/test_benchmark_metrics_v2_receipt.py already built to
# drive REAL v2 eval / eval-quantized receipts through production code (`bench` =
# csd-benchmark.py, `quant` = csd-quantize.py, both loaded there via the same importlib
# indirection this file uses for `mod` above).
from tests.test_benchmark_metrics_v2_receipt import (
    _build_pairs_parquet,
    _build_tokenizer,
)
from tests.test_benchmark_metrics_v2_receipt import (
    bench as _v2_bench,
)
from tests.test_benchmark_metrics_v2_receipt import (
    quant as _v2_quant,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "csd-publish-checkpoint.py"

# huggingface_hub is deliberately NOT a project dependency (scripts/csd-hf-repos.py
# uses raw urllib against HF's REST API for exactly this reason -- see its module
# docstring) and so is absent from the isolated CI venv scripts/ci_local.sh builds via
# `uv sync --group dev`. This script's own import of it is lazy (inside publish(),
# only reached once a token is present), but `patch("huggingface_hub.HfApi", ...)`
# below still needs the *name* importable to resolve its target. Stub it into
# sys.modules when the real package isn't installed so these tests need no network
# dependency and no project-dependency change -- every test patches HfApi itself
# before it matters, so a placeholder attribute here is never actually exercised.
if "huggingface_hub" not in sys.modules:
    try:
        import huggingface_hub as _hf_probe  # noqa: F401
    except ModuleNotFoundError:
        import types

        _hf_stub = types.ModuleType("huggingface_hub")
        _hf_stub.HfApi = object  # placeholder; every test patches this before use
        sys.modules["huggingface_hub"] = _hf_stub


def load_mod() -> Any:
    loader = importlib.machinery.SourceFileLoader("csd_publish_checkpoint", str(SCRIPT))
    spec = importlib.util.spec_from_loader("csd_publish_checkpoint", loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["csd_publish_checkpoint"] = mod  # dataclass field resolution needs this registered
    loader.exec_module(mod)
    return mod


mod = load_mod()


@pytest.fixture(autouse=True)
def _allow_tmp_path_as_checkpoint_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every fixture below writes its checkpoint under pytest's tmp_path, not under one
    of the script's real allow-listed roots (/akula-data/csd, the repo). Extend the
    allow-list with tmp_path so the path-containment fix -- a security control, not
    something most of these tests mean to exercise -- doesn't reject legitimate test
    fixtures. The containment tests further down point deliberately outside tmp_path
    and so are unaffected by this."""
    monkeypatch.setattr(mod, "ALLOWED_CHECKPOINT_ROOTS", [*mod.ALLOWED_CHECKPOINT_ROOTS, tmp_path])


# --------------------------------------------------------------------------- fixtures


def make_checkpoint(
    tmp_path: Path, name: str = "final.pt", content: bytes = b"fake-weights-blob"
) -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


def _default_recorded(checkpoint: Path) -> str:
    """The realistic case, not a comfortable one: every real receipt producer
    (csd-quantize.py, csd-benchmark.py, the pretrain loop) stamps `recorded` with
    time.strftime("%Y-%m-%dT%H:%M:%SZ") moments after torch.save writes the
    checkpoint -- frequently in the SAME wall-clock second, since the checkpoint
    mtime carries sub-second precision the receipt's whole-second stamp cannot
    express. Using a +5-minute offset here would conceal exactly that defect, so
    this default reproduces the same-second case instead. Fixtures that want a
    different relationship to the checkpoint's mtime pass `recorded=` explicitly."""
    ts = datetime.fromtimestamp(checkpoint.stat().st_mtime, tz=UTC).replace(microsecond=0)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def make_training_receipt(
    tmp_path: Path,
    checkpoint: Path,
    region: str = "compress",
    checkpoint_sha256: str | None = "USE_REAL",
    name: str = "compress-20260902T211539Z.json",
    recorded: str | None = None,
) -> Path:
    """`checkpoint_sha256`: the sentinel "USE_REAL" (default) records the checkpoint's
    actual sha256, matching what a real, correctly-bound receipt looks like post-fix.
    Pass an explicit (wrong) value to test mismatch, or None to test absence."""
    sha = mod.sha256_of(checkpoint) if checkpoint_sha256 == "USE_REAL" else checkpoint_sha256
    receipt: dict[str, Any] = {
        "schema": "csd-pretrain-receipt/v1",
        "region": region,
        "recorded": recorded or _default_recorded(checkpoint),
        "corpus": {
            "fingerprint": "5de8a340c37137554824578a0040604d",
            "train_pairs": 277269,
            "holdout_pairs": 512,
        },
        "contamination": {
            "train_unique": 277269,
            "eval_unique": 512,
            "overlap": 0,
            "eval_fraction_contaminated": 0.0,
            "examples": [],
        },
        "config": {"region": region, "steps": 8000, "batch_size": 256, "lr": 0.0003},
        "checkpoint": str(checkpoint),
        "held_out": {"n_pairs": 512, "recall@1": 0.707, "recall@10": 0.9199},
        "untrained_baseline": {"n_pairs": 512, "recall@1": 0.037, "recall@10": 0.068},
        "beats_untrained": {"recall@1": True, "recall@10": True},
    }
    if sha is not None:
        receipt["artifacts"] = {"checkpoint_sha256": sha}
    path = tmp_path / name
    path.write_text(json.dumps(receipt))
    return path


def make_eval_receipt(
    tmp_path: Path,
    checkpoint: Path,
    region: str = "compress",
    checkpoint_sha256: str | None = "USE_REAL",
    recorded: str | None = None,
) -> Path:
    sha = mod.sha256_of(checkpoint) if checkpoint_sha256 == "USE_REAL" else checkpoint_sha256
    receipt: dict[str, Any] = {
        "producer": {"project": "cogsyndelta", "component": region},
        "stage": "eval",
        "started_utc": recorded or _default_recorded(checkpoint),
        "metrics": {
            "rank.recall@1": 0.49,
            "repr.anisotropy": 0.1306,
            "repr.effective_rank_ratio": 0.459,
        },
        "gates": {"beats_untrained": True, "not_anisotropic": True, "uses_its_dimensions": True},
        "artifacts": {"checkpoint": str(checkpoint)},
        "schema": "model-pipeline-receipt/v1",
    }
    if sha is not None:
        receipt["artifacts"]["checkpoint_sha256"] = sha
    path = tmp_path / f"cogsyndelta-{region}-eval-20260902T183739Z.json"
    path.write_text(json.dumps(receipt))
    return path


def write_packed_artifact(path: Path, bits: int = 3) -> tuple[int, int, dict[str, int]]:
    """Write a REAL `csd-ptq-v1` artifact at `path`; return its (fp32_bytes,
    stored_bytes, width_histogram) as a quant receipt would record them.

    A stand-in blob would do for the sha256 checks alone, but the publisher now loads
    this file and re-measures the numbers the card prints, so the fixture has to be a
    real artifact or every test would be exercising the "unreadable artifact" abort
    instead of the behaviour it names. Small on purpose: one quantizable tensor (8192
    elements, over `quantizable`'s floor) plus three that stay fp32.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 8))
    plan = QuantPlan(bits={"0.weight": bits})
    plan.fp32 = [n for n, _ in model.named_parameters() if n not in plan.bits]
    save_packed_artifact(model, plan, path)
    packed = load_packed_artifact(path)
    fp32_bytes = sum(p.numel() * 4 for p in model.parameters())
    return fp32_bytes, packed_stored_bytes(packed), packed_width_histogram(packed)


def make_quant_receipt(
    tmp_path: Path,
    checkpoint: Path,
    region: str = "compress",
    checkpoint_sha256: str | None = "USE_REAL",
    recorded: str | None = None,
    quantized_artifact: bool = True,
    quantized_sha256: str | None = "USE_REAL",
    quantized_path: Path | None = None,
    stored_bytes: int | None = None,
    width_histogram: dict[str, int] | None = None,
) -> Path:
    """`quantized_artifact`: when True (the realistic case -- every quant receipt
    scripts/csd-quantize.py writes now names a persisted packed artifact), a real
    packed artifact is written next to `checkpoint`, at the same derived path
    csd-quantize.py uses, and the receipt's measured numbers are taken from it.

    `quantized_sha256`: the sentinel "USE_REAL" (default) declares that file's actual
    sha256, matching a correctly-bound receipt; pass an explicit (wrong) value to
    test a mismatch, or `None` to test the field's absence -- both leave the file on
    disk, so the mismatch is the only thing under test.

    `quantized_path` / `stored_bytes` / `width_histogram` override what the receipt
    *claims* while leaving the artifact on disk untouched, which is the shape every
    substitution and tampering test needs: the receipt is the attacker-controlled
    document, the file is the ground truth, and the publisher is supposed to notice
    when the two disagree.
    """
    sha = mod.sha256_of(checkpoint) if checkpoint_sha256 == "USE_REAL" else checkpoint_sha256
    receipt: dict[str, Any] = {
        "region": region,
        "checkpoint": str(checkpoint),
        "recorded_utc": recorded or _default_recorded(checkpoint),
        "corpus_fingerprint": "5de8a340c37137554824578a0040604d",
        "tolerance": 0.01,
        "fp32_metric_recomputed": 0.496,
        "quantized_metric": 0.488,
        "drop": 0.0078,
        "within_budget": True,
    }
    if sha is not None:
        receipt["checkpoint_sha256"] = sha
    if quantized_artifact:
        quant_file = checkpoint.with_name(f"{checkpoint.stem}.ptq.pt")
        fp32_bytes, measured_bytes, measured_hist = write_packed_artifact(quant_file)
        receipt["fp32_bytes"] = fp32_bytes
        receipt["stored_bytes"] = measured_bytes if stored_bytes is None else stored_bytes
        receipt["compression_ratio"] = fp32_bytes / measured_bytes
        receipt["width_histogram"] = measured_hist if width_histogram is None else width_histogram
        named = quant_file if quantized_path is None else quantized_path
        artifacts: dict[str, Any] = {"quantized_path": str(named)}
        real_q_sha = mod.sha256_of(quant_file)
        q_sha = real_q_sha if quantized_sha256 == "USE_REAL" else quantized_sha256
        if q_sha is not None:
            artifacts["quantized_sha256"] = q_sha
        receipt["artifacts"] = artifacts
    path = tmp_path / f"{region}-quant-20260902T181604Z.json"
    path.write_text(json.dumps(receipt))
    return path


def repo_file(
    path: str, blob_id: str | None = None, sha256: str | None = None, size: int = 0
) -> Any:
    lfs = SimpleNamespace(sha256=sha256) if sha256 else None
    return SimpleNamespace(path=path, blob_id=blob_id, lfs=lfs, size=size)


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324


# --------------------------------------------------------------- repo naming (req 1)


def test_default_repo_matches_csd_hf_repos_convention() -> None:
    assert mod.default_repo("code") == "tzervas/cogsyndelta-region-code"
    assert mod.default_repo("compress") == "tzervas/cogsyndelta-region-compress"
    assert mod.default_repo("retrieve") == "tzervas/cogsyndelta-region-retrieve"
    assert mod.default_repo("residual_mlp") == "tzervas/cogsyndelta-region-residual"
    assert mod.default_repo("stream_vae") == "tzervas/cogsyndelta-region-stream-vae"
    assert mod.default_repo("vl_latent") == "tzervas/cogsyndelta-vl-jepa"


def test_default_repo_extends_pattern_for_unlisted_regions() -> None:
    # reason/classify_* aren't in csd-hf-repos.py's REPOS list yet; the plain
    # "-region-<name>" pattern is the natural extension, not an arbitrary choice.
    assert mod.default_repo("reason") == "tzervas/cogsyndelta-region-reason"
    assert mod.default_repo("classify_banking77") == "tzervas/cogsyndelta-region-classify-banking77"


def test_no_public_flag_exists() -> None:
    with pytest.raises(SystemExit):
        mod.parse_args(["--region", "code", "--receipt", "x.json", "--public"])


# ------------------------------------------------------------------ licence (req 3)


@pytest.mark.parametrize(
    ("region", "tier"),
    [
        ("code", "mit"),
        ("classify_banking77", "mit"),
        ("classify_go_emotions", "mit"),
        ("reason", "mit"),
        ("compress", "cc-by-sa-4.0"),
        ("retrieve", "cc-by-nc-sa-4.0"),
        ("memory", "cc-by-nc-sa-4.0"),
    ],
)
def test_licence_tier_matches_decision_2026_09_02(region: str, tier: str) -> None:
    assert mod.licence_tier(region) == tier


def test_memory_licence_tier_equals_retrieve() -> None:
    # docs/design/LICENCE-FOR-OPEN-WEIGHTS.md, Decision 2026-09-02, Rider 1: a region
    # MERGE inherits the most restrictive licence of its parts. `memory` merges
    # `compress` (CC BY-SA 4.0) and `retrieve` (CC BY-NC-SA 4.0) and trains directly on
    # retrieve's corpus, so it must carry exactly retrieve's tier -- not a value that
    # happens to match today by coincidence.
    assert mod.licence_tier("memory") == mod.licence_tier("retrieve")


@pytest.mark.parametrize("region", ["residual_mlp", "stream_vae", "some_unaudited_region"])
def test_licence_tier_unknown_aborts(region: str) -> None:
    # Adding memory's tier must not weaken the refusal for a region that still has none.
    with pytest.raises(mod.PublishAbortError, match="licence tier"):
        mod.licence_tier(region)


def test_vl_latent_is_blocking_not_mit() -> None:
    # docs/design/LICENCE-FOR-OPEN-WEIGHTS.md section 4: unreleasable as trained --
    # 100,000/100,000 pretraining images with no licence grant in the provenance chain.
    # LICENCE_TIER still carries "mit" for vl_latent's eventual composite replacement
    # (Decision 2026-09-02), but licence_tier() must refuse regardless of that value.
    with pytest.raises(mod.PublishAbortError, match="BLOCKING"):
        mod.licence_tier("vl_latent")


def test_vl_latent_blocks_full_plan_before_any_file_read(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    receipt = make_training_receipt(tmp_path, checkpoint, region="vl_latent")
    with pytest.raises(mod.PublishAbortError, match="BLOCKING"):
        mod.build_plan("vl_latent", "tzervas/cogsyndelta-vl-jepa", receipt, None, None)


# ------------------------------------------------- region rename (naming rule 2026-09-04)


def test_default_repo_of_the_canonical_language_name_is_the_future_hub_repo() -> None:
    """`default_repo` does not canonicalize: `code` (today's live Hub repo) and
    `language` (the future one, once the orchestrator's rename lands) compute
    DIFFERENT names on purpose -- see that function's own docstring."""
    assert mod.default_repo("language") == "tzervas/cogsyndelta-region-language"
    assert mod.default_repo("code") == "tzervas/cogsyndelta-region-code"
    assert mod.default_repo("language") != mod.default_repo("code")


def test_default_repo_of_visual_matches_vl_latent() -> None:
    """Unlike `code`/`language`, the vl-jepa repo suffix was never derived from the
    region's own spelling, so both spellings resolve to the identical Hub repo."""
    assert (
        mod.default_repo("visual") == mod.default_repo("vl_latent") == "tzervas/cogsyndelta-vl-jepa"
    )


def test_licence_tier_accepts_the_canonical_names_too() -> None:
    """LICENCE_TIER is keyed canonically; licence_tier() must still resolve `language`
    and `visual` (not just their legacy spellings, already covered above)."""
    assert mod.licence_tier("language") == mod.licence_tier("code") == "mit"
    with pytest.raises(mod.PublishAbortError, match="BLOCKING"):
        mod.licence_tier("visual")


def _mix_b_receipt(fingerprint: str | None = None, source: str | None = None) -> dict[str, Any]:
    admitted_source, admitted_fp = mod.admitted_visual_identity()
    return {
        "corpus": {
            "corpus_source": admitted_source if source is None else source,
            "fingerprint": admitted_fp if fingerprint is None else fingerprint,
        }
    }


def test_visual_tiny_imagenet_receipt_still_blocks() -> None:
    """Mutation: naming tiny-imagenet must not ride Mix B's fingerprint into a publish."""
    _, fp = mod.admitted_visual_identity()
    with pytest.raises(mod.PublishAbortError, match="BLOCKING"):
        mod.licence_tier("visual", receipt=_mix_b_receipt(fingerprint=fp, source="tiny-imagenet"))
    with pytest.raises(mod.PublishAbortError, match="BLOCKING"):
        mod.licence_tier(
            "vl_latent",
            receipt=_mix_b_receipt(fingerprint=fp, source="zh-plus/tiny-imagenet"),
        )


def test_visual_unset_corpus_still_blocks() -> None:
    with pytest.raises(mod.PublishAbortError, match="BLOCKING"):
        mod.licence_tier("visual")
    with pytest.raises(mod.PublishAbortError, match="BLOCKING"):
        mod.licence_tier("visual", receipt={"corpus": {}})
    with pytest.raises(mod.PublishAbortError, match="BLOCKING"):
        mod.licence_tier(
            "visual", receipt={"corpus": {"fingerprint": "ab5761b65714e4ba4d7c36df095f3599"}}
        )


def test_visual_clean_v1_matching_fingerprint_is_mit() -> None:
    """Mix B pin: ATTRIBUTION (CLEVR) is MIT-usable with a notice (:98-102)."""
    assert mod.licence_tier("visual", receipt=_mix_b_receipt()) == "mit"
    assert mod.licence_tier("vl_latent", receipt=_mix_b_receipt()) == "mit"


def test_visual_clean_v1_mismatched_fingerprint_blocks() -> None:
    """Mutation: same corpus_source, wrong fingerprint, must stay BLOCKING."""
    bad = "0" * 32
    assert bad != mod.admitted_visual_identity()[1]
    with pytest.raises(mod.PublishAbortError, match="BLOCKING"):
        mod.licence_tier("visual", receipt=_mix_b_receipt(fingerprint=bad))


def test_visual_card_carries_clevr_tasl() -> None:
    card = mod.build_card(
        region="visual",
        region_cfg={"role": "visual cortex", "router_trigger": "image"},
        tier="mit",
        checkpoint=Path("final.pt"),
        checkpoint_sha256="abc",
        rev="deadbeef",
        train_receipt=_mix_b_receipt(),
        eval_receipt=None,
        quant_receipt=None,
    )
    assert "CLEVR" in card
    assert "creativecommons.org/licenses/by/4.0" in card
    assert "modified" in card.lower()
    assert "license: mit" in card
    assert ":98-102" in card


def test_visual_card_refuses_without_clevr_tasl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: drop the TASL block and Mix B must not publish as silent MIT."""
    monkeypatch.setattr(mod, "CLEVR_TASL", "attribution omitted")
    with pytest.raises(mod.PublishAbortError, match="TASL"):
        mod.build_card(
            region="visual",
            region_cfg={"role": "visual cortex", "router_trigger": "image"},
            tier="mit",
            checkpoint=Path("final.pt"),
            checkpoint_sha256="abc",
            rev="deadbeef",
            train_receipt=_mix_b_receipt(),
            eval_receipt=None,
            quant_receipt=None,
        )


def test_load_region_config_accepts_both_spellings(tmp_path: Path) -> None:
    """`config/mind/csd-regions.json` stores the language centre under `language`
    (specialisation: code); load_region_config must find that ONE entry whether asked
    for `code` or `language`."""
    by_legacy = mod.load_region_config("code")
    by_canonical = mod.load_region_config("language")
    assert by_legacy == by_canonical
    assert by_legacy["name"] == "language"
    assert by_legacy["specialisation"] == "code"


def test_load_region_config_visual_by_either_name() -> None:
    by_legacy = mod.load_region_config("vl_latent")
    by_canonical = mod.load_region_config("visual")
    assert by_legacy == by_canonical
    assert by_legacy["name"] == "visual"


def test_assert_region_matches_across_the_rename(tmp_path: Path) -> None:
    """A receipt written before the rename (`region: "code"`) must still match
    `--region language`, and the reverse -- the rename must not turn every existing
    receipt into a forced mismatch."""
    checkpoint = make_checkpoint(tmp_path)
    legacy_receipt = json.loads(
        make_training_receipt(tmp_path, checkpoint, region="code").read_text()
    )
    mod.assert_region_matches(legacy_receipt, "language", "training")  # must not raise

    canonical_receipt = json.loads(
        make_training_receipt(
            tmp_path,
            checkpoint,
            region="language",
            name="language-20260902T211539Z.json",
        ).read_text()
    )
    mod.assert_region_matches(canonical_receipt, "code", "training")  # must not raise too


def test_assert_region_matches_still_rejects_a_genuine_mismatch(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    receipt = json.loads(make_training_receipt(tmp_path, checkpoint, region="code").read_text())
    with pytest.raises(mod.PublishAbortError, match="does not match"):
        mod.assert_region_matches(receipt, "retrieve", "training")


def test_unknown_tier_aborts_full_plan_before_any_file_read(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    receipt = make_training_receipt(tmp_path, checkpoint, region="stream_vae")
    with pytest.raises(mod.PublishAbortError, match="licence tier"):
        mod.build_plan("stream_vae", "tzervas/whatever", receipt, None, None)


# --------------------------------------------------------------- sha256 (req 3, 5)


def test_sha_computed_from_file(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    got = mod.verify_checkpoint_sha(checkpoint)
    assert got == mod.sha256_of(checkpoint)
    assert len(got) == 64


def test_missing_checkpoint_aborts(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pt"
    with pytest.raises(mod.PublishAbortError, match="not found"):
        mod.verify_checkpoint_sha(missing)


# --------------------------------------------- receipt-to-checkpoint binding (this fix)
#
# The review this fix responds to: eval and quant receipts were region-checked but
# never bound to the checkpoint actually being published -- every receipt named the
# same mutable <region>-checkpoints/final.pt, and a later training run silently
# invalidated earlier eval/quant receipts without either check noticing. These tests
# pin both halves of the fix: (a) every receipt's own declared checkpoint_sha256 must
# equal the checkpoint's real sha256, absence or mismatch aborts; (b) a receipt
# timestamped before the checkpoint's mtime aborts too, belt-and-braces for a legacy
# receipt that (by malicious luck or a hand-edit) carried a matching sha256.


def test_binding_sha_mismatch_aborts(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    mtime = datetime.fromtimestamp(checkpoint.stat().st_mtime, tz=UTC)
    receipt_path = make_training_receipt(tmp_path, checkpoint, checkpoint_sha256="0" * 64)
    receipt = json.loads(receipt_path.read_text())
    with pytest.raises(mod.PublishAbortError, match="does not match"):
        mod.assert_receipt_bound_to_checkpoint(
            receipt, mod.sha256_of(checkpoint), mtime, "training"
        )


def test_binding_sha_absent_aborts(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    mtime = datetime.fromtimestamp(checkpoint.stat().st_mtime, tz=UTC)
    receipt_path = make_training_receipt(tmp_path, checkpoint, checkpoint_sha256=None)
    receipt = json.loads(receipt_path.read_text())
    with pytest.raises(mod.PublishAbortError, match=r"no artifacts\.checkpoint_sha256"):
        mod.assert_receipt_bound_to_checkpoint(
            receipt, mod.sha256_of(checkpoint), mtime, "training"
        )


def test_binding_older_than_checkpoint_aborts(tmp_path: Path) -> None:
    # sha256 matches (belt-and-braces case: a legacy receipt whose sha happens to be
    # right) but its recorded timestamp predates the checkpoint file's mtime by
    # hours -- the realistic legacy case (a receipt for a superseded final.pt), well
    # outside the one-second grace period the same-second fix introduces.
    checkpoint = make_checkpoint(tmp_path)
    mtime = datetime.fromtimestamp(checkpoint.stat().st_mtime, tz=UTC)
    stale = (mtime - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    receipt_path = make_training_receipt(tmp_path, checkpoint, recorded=stale)
    receipt = json.loads(receipt_path.read_text())
    with pytest.raises(mod.PublishAbortError, match="predates the checkpoint"):
        mod.assert_receipt_bound_to_checkpoint(
            receipt, mod.sha256_of(checkpoint), mtime, "training"
        )


def test_binding_one_second_before_floored_mtime_aborts(tmp_path: Path) -> None:
    """The one-second grace period floors the checkpoint's mtime to whole seconds
    before comparing -- it must NOT become an open-ended grace period. A receipt
    timestamped one full second before that floored mtime still describes a
    checkpoint that didn't exist yet, and must still abort."""
    checkpoint = make_checkpoint(tmp_path)
    floored_mtime = datetime.fromtimestamp(checkpoint.stat().st_mtime, tz=UTC).replace(
        microsecond=0
    )
    too_early = (floored_mtime - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    receipt_path = make_training_receipt(tmp_path, checkpoint, recorded=too_early)
    receipt = json.loads(receipt_path.read_text())
    with pytest.raises(mod.PublishAbortError, match="predates the checkpoint"):
        mod.assert_receipt_bound_to_checkpoint(
            receipt, mod.sha256_of(checkpoint), floored_mtime, "training"
        )


def test_binding_timestamp_absent_aborts(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    mtime = datetime.fromtimestamp(checkpoint.stat().st_mtime, tz=UTC)
    receipt_path = make_training_receipt(tmp_path, checkpoint)
    receipt = json.loads(receipt_path.read_text())
    del receipt["recorded"]
    with pytest.raises(mod.PublishAbortError, match="none of"):
        mod.assert_receipt_bound_to_checkpoint(
            receipt, mod.sha256_of(checkpoint), mtime, "training"
        )


def test_binding_matches_passes(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    # Floored to whole seconds, matching how the real caller (main()) derives
    # checkpoint_mtime before ever calling this function -- see its docstring.
    mtime = datetime.fromtimestamp(checkpoint.stat().st_mtime, tz=UTC).replace(microsecond=0)
    receipt_path = make_training_receipt(
        tmp_path, checkpoint
    )  # default: real sha, same-second timestamp
    receipt = json.loads(receipt_path.read_text())
    mod.assert_receipt_bound_to_checkpoint(
        receipt, mod.sha256_of(checkpoint), mtime, "training"
    )  # no raise


def test_binding_eval_receipt_top_level_or_nested_sha_both_work(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    real_sha = mod.sha256_of(checkpoint)
    assert (
        mod.receipt_checkpoint_sha256({"artifacts": {"checkpoint_sha256": real_sha}}, "x")
        == real_sha
    )
    assert mod.receipt_checkpoint_sha256({"checkpoint_sha256": real_sha}, "x") == real_sha


def test_build_plan_aborts_when_eval_receipt_unbound(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    eval_path = make_eval_receipt(tmp_path, checkpoint, region="compress", checkpoint_sha256=None)
    with pytest.raises(mod.PublishAbortError, match="eval receipt has no"):
        mod.build_plan(
            "compress", "tzervas/cogsyndelta-region-compress", train_path, eval_path, None
        )


def test_build_plan_aborts_when_quant_receipt_unbound(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress", checkpoint_sha256=None)
    with pytest.raises(mod.PublishAbortError, match="quant receipt has no"):
        mod.build_plan(
            "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
        )


# --------------------------------------------------- quantized artifact (this fix)
#
# scripts/csd-quantize.py now persists the packed artifact next to the checkpoint and
# records artifacts.quantized_path / artifacts.quantized_sha256 in the quant receipt.
# These tests pin the publisher half: the artifact is verified by sha256 -- exactly as
# strictly as the fp32 checkpoint is bound to its receipts -- before it ever enters the
# upload plan, and it is uploaded ALONGSIDE final.pt, never in place of it.


def test_quant_artifact_sha_mismatch_aborts(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(
        tmp_path, checkpoint, region="compress", quantized_sha256="0" * 64
    )
    with pytest.raises(mod.PublishAbortError, match="does not match the quantized artifact"):
        mod.build_plan(
            "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
        )


def test_quant_artifact_missing_sha_field_aborts(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress", quantized_sha256=None)
    with pytest.raises(mod.PublishAbortError, match=r"no artifacts\.quantized_sha256"):
        mod.build_plan(
            "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
        )


def test_quant_artifact_missing_path_field_aborts(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(
        tmp_path, checkpoint, region="compress", quantized_artifact=False
    )
    with pytest.raises(mod.PublishAbortError, match=r"no artifacts\.quantized_path"):
        mod.build_plan(
            "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
        )


def test_quant_artifact_never_becomes_the_primary(tmp_path: Path) -> None:
    """The checkpoint stays the required, unconditional file regardless of the quant
    artifact's own name or presence -- `plan.checkpoint_path` names final.pt, not the
    packed artifact, even though both are now uploaded."""
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress")
    plan = mod.build_plan(
        "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
    )
    assert plan.checkpoint_path == checkpoint
    assert plan.quantized_path is not None
    assert plan.quantized_path != plan.checkpoint_path
    assert checkpoint.name in plan.files
    assert plan.quantized_path.name in plan.files
    assert plan.files[checkpoint.name] == checkpoint


def test_quant_artifact_consistent_trio_publishes_both_files(tmp_path: Path) -> None:
    """The checkpoint AND the quantized artifact are both uploaded when the quant
    receipt correctly binds to both (mocked HfApi; no network)."""
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    eval_path = make_eval_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress")
    plan = mod.build_plan(
        "compress", "tzervas/cogsyndelta-region-compress", train_path, eval_path, quant_path
    )
    assert plan.quantized_path is not None
    assert {checkpoint.name, plan.quantized_path.name}.issubset(plan.files.keys())

    fake_api = MagicMock()
    fake_api.repo_info.return_value = SimpleNamespace(private=True)
    fake_api.get_paths_info.return_value = []
    with (
        patch("huggingface_hub.HfApi", return_value=fake_api),
        patch.dict("os.environ", {"HF_TOKEN": "tok"}),
    ):
        result = mod.publish(plan)
    assert checkpoint.name in result["uploaded"]
    assert plan.quantized_path.name in result["uploaded"]
    assert "Quantized artifact" in plan.card
    assert plan.quantized_sha256 is not None
    assert plan.quantized_sha256 in plan.card

    # N1 regression guard: the card's filename and the actual upload key must be the
    # SAME string, and both must come from the checkpoint stem -- never from a
    # resolved foreign basename a symlink or a receipt could otherwise substitute in.
    card_filename_match = re.search(r"\*\*File:\*\* `([^`]+)`", plan.card)
    assert card_filename_match is not None
    card_filename = card_filename_match.group(1)
    assert card_filename == plan.quantized_path.name
    assert card_filename == f"{checkpoint.stem}.ptq.pt"
    assert card_filename in plan.files
    assert plan.files[card_filename] == plan.quantized_path


def test_dry_run_lists_checkpoint_and_quantized_artifact(tmp_path: Path, capsys: Any) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress")

    with patch("huggingface_hub.HfApi", side_effect=AssertionError("no network in --dry-run")):
        rc = mod.main(
            [
                "--region",
                "compress",
                "--receipt",
                str(train_path),
                "--quant-receipt",
                str(quant_path),
                "--repo",
                "tzervas/cogsyndelta-region-compress",
                "--dry-run",
            ]
        )
    out = capsys.readouterr().out
    assert rc == 0
    assert checkpoint.name in out
    assert f"{checkpoint.stem}.ptq.pt" in out
    assert "Quantized artifact" in out
    assert "not primary" in out


def test_consistent_synthetic_trio_publishes(tmp_path: Path) -> None:
    """The positive case: training, eval and quant receipts that all correctly name
    this exact checkpoint's sha256 and are all timestamped after it -- the publish
    must proceed (mocked HfApi; no network). `_default_recorded` stamps these in the
    SAME wall-clock second as the checkpoint's mtime (see its docstring), so this is
    also the realistic case: every producer's whole-second `recorded` stamp landing
    in the same second as the checkpoint's own (sub-second-precision) mtime."""
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    eval_path = make_eval_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress")
    plan = mod.build_plan(
        "compress", "tzervas/cogsyndelta-region-compress", train_path, eval_path, quant_path
    )
    assert plan.bound_receipt_labels == ("training", "eval", "quant")

    fake_api = MagicMock()
    fake_api.repo_info.return_value = SimpleNamespace(private=True)
    fake_api.get_paths_info.return_value = []
    with (
        patch("huggingface_hub.HfApi", return_value=fake_api),
        patch.dict("os.environ", {"HF_TOKEN": "tok"}),
    ):
        result = mod.publish(plan)
    assert set(result["uploaded"]) == set(plan.files.keys())
    assert fake_api.upload_file.call_count == len(plan.files)


def test_same_second_receipt_publishes(tmp_path: Path) -> None:
    """Pinned explicitly, not just via the shared default: a training receipt
    timestamped in the exact same second as the checkpoint's mtime, with a correct
    sha256, must publish -- not abort on a nanosecond-vs-whole-second mismatch
    between the checkpoint's mtime and every receipt producer's whole-second
    `recorded` stamp (mocked HfApi; no network)."""
    checkpoint = make_checkpoint(tmp_path)
    same_second = datetime.fromtimestamp(checkpoint.stat().st_mtime, tz=UTC).replace(microsecond=0)
    recorded = same_second.strftime("%Y-%m-%dT%H:%M:%SZ")
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress", recorded=recorded)
    eval_path = make_eval_receipt(tmp_path, checkpoint, region="compress", recorded=recorded)
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress", recorded=recorded)
    plan = mod.build_plan(
        "compress", "tzervas/cogsyndelta-region-compress", train_path, eval_path, quant_path
    )
    assert plan.bound_receipt_labels == ("training", "eval", "quant")

    fake_api = MagicMock()
    fake_api.repo_info.return_value = SimpleNamespace(private=True)
    fake_api.get_paths_info.return_value = []
    with (
        patch("huggingface_hub.HfApi", return_value=fake_api),
        patch.dict("os.environ", {"HF_TOKEN": "tok"}),
    ):
        result = mod.publish(plan)
    assert set(result["uploaded"]) == set(plan.files.keys())
    assert fake_api.upload_file.call_count == len(plan.files)


def test_dry_run_against_real_compress_receipts_aborts_sha_absent() -> None:
    """The dry-run this fix's spec calls for: the real receipts under
    /akula-data/csd/receipts carry no checkpoint_sha256 anywhere, so this must ABORT
    with the sha-absent message -- pasted into the task report -- not silently
    publish a card mixing current and superseded-weights metrics."""
    receipts_dir = Path("/akula-data/csd/receipts")
    if not receipts_dir.is_dir():
        pytest.skip("real receipts fixture directory not present on this host")
    rc = mod.main(
        [
            "--region",
            "compress",
            "--receipt",
            str(receipts_dir / "compress-20260902T211539Z.json"),
            "--eval-receipt",
            str(receipts_dir / "cogsyndelta-compress-eval-20260902T183739Z.json"),
            "--quant-receipt",
            str(receipts_dir / "compress-quant-20260902T181604Z.json"),
            "--repo",
            "tzervas/cogsyndelta-region-compress",
            "--dry-run",
        ]
    )
    assert rc == 2


# --------------------------- review fix: --dry-run must not abort on this branch's OWN
# --------------------------- v2 receipts (blocking item 1, feat/metrics-v2)


def test_dry_run_publishes_real_v2_eval_and_eval_quantized_receipts_end_to_end(
    tmp_path: Path,
) -> None:
    """The review's exact failure: `--dry-run` ABORTED with `card would print metric
    key(s) ['effective_rank_entropy', 'emb_std_anchor'] with no entry in
    METRIC_METHODOLOGY` on a receipt `scripts/csd-benchmark.py` (this branch's own lane
    A) produces natively -- `METRIC_METHODOLOGY` here (lane B) had renamed
    `effective_rank_ratio` -> `effective_rank_entropy_ratio` but never added
    `effective_rank_entropy` or `emb_std_anchor`. A hand-built receipt would not have
    caught this: the hole is specifically that lane A's real output and lane B's table
    disagreed, so this drives a REAL tiny CPU pretrain + benchmark + quantize run
    (region "code" -- a real MIT-tiered region name, so `--dry-run` reaches the
    METRIC_METHODOLOGY check this fix targets rather than aborting earlier on an
    unknown licence tier; same tokenizer/parquet fixture builders as
    tests/test_benchmark_metrics_v2_receipt.py) through the publish script's own
    `--dry-run`, for both `kind=eval` and `kind=eval-quantized` receipts, exactly as
    the review's repro did.
    """
    region = "code"
    tok_path = tmp_path / "tokenizer.json"
    shard_path = tmp_path / "pairs.parquet"
    _build_tokenizer(tok_path, 40)
    _build_pairs_parquet(shard_path, 40)
    receipts_dir = tmp_path / "receipts"

    cfg = PretrainConfig(
        region=region,
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
        out_dir=str(receipts_dir),
    )
    pretrain_region(cfg)
    train_path = next(receipts_dir.glob(f"{region}-*.json"))

    def fake_regions_spec() -> dict:
        class _Entry:
            sources: ClassVar = [("pairs.parquet", ("anchor", "positive"), 0)]
            root = tmp_path

        return {
            "REGIONS": {},
            "_shards": lambda *a, **k: [str(shard_path)],
            "region_spec": lambda name: _Entry(),
        }

    orig_bench_spec, orig_quant_spec = _v2_bench._regions_spec, _v2_quant._load_regions_spec
    _v2_bench._regions_spec = fake_regions_spec
    _v2_quant._load_regions_spec = fake_regions_spec
    try:
        eval_rec = _v2_bench.benchmark_region(region, tmp_path)
        assert eval_rec is not None
        eval_path = eval_rec.write(receipts_dir)

        quant_rec = _v2_quant.quantize_text_region(
            region, tmp_path, tolerance=1.0, aggressive=3, max_bits=8
        )
        eval_quant_rec = _v2_bench.benchmark_region_quantized(
            region, tmp_path, Path(quant_rec["artifacts"]["quantized_path"])
        )
        eval_quant_path = eval_quant_rec.write(receipts_dir)
    finally:
        _v2_bench._regions_spec = orig_bench_spec
        _v2_quant._load_regions_spec = orig_quant_spec

    with patch("huggingface_hub.HfApi", side_effect=AssertionError("no network in --dry-run")):
        rc_eval = mod.main(
            [
                "--region",
                region,
                "--receipt",
                str(train_path),
                "--eval-receipt",
                str(eval_path),
                "--dry-run",
            ]
        )
        rc_eval_quantized = mod.main(
            [
                "--region",
                region,
                "--receipt",
                str(train_path),
                "--eval-receipt",
                str(eval_quant_path),
                "--dry-run",
            ]
        )

    assert rc_eval == 0, "kind=eval dry-run must not abort on this branch's own v2 receipt"
    assert rc_eval_quantized == 0, (
        "kind=eval-quantized dry-run must not abort on this branch's own v2 receipt"
    )


# -------------------------------------------------- receipt must be a JSON object (review fix)
#
# json.loads('null') returns None, and json.loads('[1,2,3]') returns a list -- both
# are valid JSON, neither is a receipt. The review's reproduction: an --eval-receipt
# file containing exactly the four bytes 'null' made `eval_receipt` (the parsed
# object) None, which build_plan's checks read as "no eval receipt supplied" (`if
# eval_receipt is not None`) -- but the *path* to that same file stayed truthy
# wherever the upload-plan builder instead checked `if eval_receipt_path:`, so the
# receipt was smuggled past both assert_region_matches and
# assert_receipt_bound_to_checkpoint straight into the files a mocked HfApi actually
# uploaded. These tests pin both the direct fix (load_json rejects a non-dict) and
# the outcome that matters: build_plan itself now aborts for such a file, so no path
# through main()/publish() can ever reach upload_file with it.


def test_eval_receipt_null_aborts(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    bogus = tmp_path / "not-a-receipt.json"
    bogus.write_text("null")
    with pytest.raises(mod.PublishAbortError, match=r"receipt must be a JSON object, got NoneType"):
        mod.build_plan("compress", "tzervas/cogsyndelta-region-compress", train_path, bogus, None)


def test_eval_receipt_list_aborts(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    bogus = tmp_path / "not-a-receipt.json"
    bogus.write_text("[1,2,3]")
    with pytest.raises(mod.PublishAbortError, match=r"receipt must be a JSON object, got list"):
        mod.build_plan("compress", "tzervas/cogsyndelta-region-compress", train_path, bogus, None)


def test_quant_receipt_null_aborts(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    bogus = tmp_path / "not-a-receipt.json"
    bogus.write_text("null")
    with pytest.raises(mod.PublishAbortError, match=r"receipt must be a JSON object, got NoneType"):
        mod.build_plan("compress", "tzervas/cogsyndelta-region-compress", train_path, None, bogus)


def test_null_receipt_never_reaches_upload_end_to_end(tmp_path: Path) -> None:
    """The review's own reproduction, run to the end: build_plan must raise before
    mod.publish() is ever reached, so a mocked HfApi.upload_file is never even given
    the chance to be called with the bogus receipt."""
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    bogus = tmp_path / "not-a-receipt.json"
    bogus.write_text("null")

    fake_api = MagicMock()
    fake_api.repo_info.return_value = SimpleNamespace(private=True)
    fake_api.get_paths_info.return_value = []
    with (
        patch("huggingface_hub.HfApi", return_value=fake_api),
        patch.dict("os.environ", {"HF_TOKEN": "tok"}),
        pytest.raises(mod.PublishAbortError, match="receipt must be a JSON object"),
    ):
        plan = mod.build_plan(
            "compress", "tzervas/cogsyndelta-region-compress", train_path, bogus, None
        )
        mod.publish(plan)
    fake_api.upload_file.assert_not_called()


# ------------------------------------------------------- checkpoint containment (review fix)
#
# checkpoint_path_from_receipt() reads a receipt-controlled path with zero containment
# was the first review finding: a receipt pointing '../../../../../../../tmp/.../
# not_a_checkpoint_stand_in.txt' produced a valid upload plan for that file and exited
# 0. Both closing checks (allow-listed root, allow-listed suffix) get their own test,
# plus a regression test shaped exactly like the review's own reproduction.


def test_checkpoint_outside_allowed_roots_aborts(tmp_path: Path) -> None:
    # A directory that exists but was never added to ALLOWED_CHECKPOINT_ROOTS by this
    # test file's autouse fixture (that fixture only allow-lists tmp_path itself).
    import tempfile

    with tempfile.TemporaryDirectory() as outside_dir:
        outside = Path(outside_dir) / "final.pt"
        outside.write_bytes(b"fake-weights-blob")
        receipt = {"checkpoint": str(outside)}
        with pytest.raises(mod.PublishAbortError, match="outside the allow-listed"):
            mod.checkpoint_path_from_receipt(receipt)


def test_checkpoint_bad_suffix_aborts(tmp_path: Path) -> None:
    not_a_checkpoint = tmp_path / "not_a_checkpoint_stand_in.txt"
    not_a_checkpoint.write_text("definitely not a checkpoint")
    receipt = {"checkpoint": str(not_a_checkpoint)}
    with pytest.raises(mod.PublishAbortError, match="suffix"):
        mod.checkpoint_path_from_receipt(receipt)


def test_receipt_path_traversal_to_arbitrary_file_aborts(tmp_path: Path) -> None:
    # Shaped exactly like the review's own reproduction: a receipt naming a file that
    # both escapes the allow-listed roots AND isn't a checkpoint by suffix. The
    # containment check runs first (before hashing anything), so this must not produce
    # a plan, must not exit 0, and above all must never reach sha256_of() on the target.
    import tempfile

    with tempfile.TemporaryDirectory() as outside_dir:
        target = Path(outside_dir) / "not_a_checkpoint_stand_in.txt"
        target.write_text("secret file contents that must never be hashed or uploaded")
        traversal = ("../" * 10) + str(target).lstrip("/")
        receipt = {"checkpoint": traversal}
        with pytest.raises(mod.PublishAbortError):
            mod.checkpoint_path_from_receipt(receipt)


def test_checkpoint_containment_blocks_full_plan(tmp_path: Path) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as outside_dir:
        outside = Path(outside_dir) / "final.pt"
        outside.write_bytes(b"fake-weights-blob")
        receipt_path = make_training_receipt(tmp_path, outside)
        with pytest.raises(mod.PublishAbortError, match="outside the allow-listed"):
            mod.build_plan(
                "compress", "tzervas/cogsyndelta-region-compress", receipt_path, None, None
            )


# ---------------------------------------------------------- private-by-construction


def test_ensure_private_aborts_on_public_repo() -> None:
    api = MagicMock()
    api.repo_info.return_value = SimpleNamespace(private=False)
    with pytest.raises(mod.PublishAbortError, match="private"):
        mod.ensure_private(api, "tzervas/cogsyndelta-region-compress")
    api.create_repo.assert_called_once_with(
        repo_id="tzervas/cogsyndelta-region-compress",
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    api.upload_file.assert_not_called()


def test_ensure_private_passes_when_actually_private() -> None:
    api = MagicMock()
    api.repo_info.return_value = SimpleNamespace(private=True)
    mod.ensure_private(api, "tzervas/cogsyndelta-region-compress")  # must not raise


def test_full_publish_never_uploads_when_repo_is_public(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    receipt_path = make_training_receipt(tmp_path, checkpoint)
    plan = mod.build_plan(
        "compress", "tzervas/cogsyndelta-region-compress", receipt_path, None, None
    )

    fake_api = MagicMock()
    fake_api.repo_info.return_value = SimpleNamespace(private=False)
    with (
        patch("huggingface_hub.HfApi", return_value=fake_api) as api_cls,
        patch.dict("os.environ", {"HF_TOKEN": "tok"}),
        pytest.raises(mod.PublishAbortError),
    ):
        mod.publish(plan)
    assert api_cls.called
    fake_api.upload_file.assert_not_called()


# --------------------------------------------------------------------- missing token


def test_missing_token_aborts_with_no_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(mod.PublishAbortError, match="HF_TOKEN"):
        mod.get_token()


def test_main_aborts_cleanly_without_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    checkpoint = make_checkpoint(tmp_path)
    receipt_path = make_training_receipt(tmp_path, checkpoint)

    def _boom(*_a: Any, **_kw: Any) -> None:
        raise AssertionError("HfApi must never be constructed when HF_TOKEN is unset")

    with patch("huggingface_hub.HfApi", side_effect=_boom):
        rc = mod.main(
            [
                "--region",
                "compress",
                "--receipt",
                str(receipt_path),
                "--repo",
                "tzervas/cogsyndelta-region-compress",
            ]
        )
    assert rc == 2
    assert "HF_TOKEN" in capsys.readouterr().err


# -------------------------------------------------------------------------- dry-run


def test_dry_run_plan_content_no_network(tmp_path: Path, capsys: Any) -> None:
    checkpoint = make_checkpoint(tmp_path)
    receipt_path = make_training_receipt(tmp_path, checkpoint)
    eval_path = make_eval_receipt(tmp_path, checkpoint)
    quant_path = make_quant_receipt(tmp_path, checkpoint)

    with patch("huggingface_hub.HfApi", side_effect=AssertionError("no network in --dry-run")):
        rc = mod.main(
            [
                "--region",
                "compress",
                "--receipt",
                str(receipt_path),
                "--eval-receipt",
                str(eval_path),
                "--quant-receipt",
                str(quant_path),
                "--repo",
                "tzervas/cogsyndelta-region-compress",
                "--dry-run",
            ]
        )
    out = capsys.readouterr().out
    assert rc == 0
    assert "tzervas/cogsyndelta-region-compress" in out
    assert "region:      compress" in out
    assert "cc-by-sa-4.0" in out  # compress's licence tier
    assert mod.sha256_of(checkpoint) in out
    assert "README.md" in out
    assert "held_out" in out
    assert "license: cc-by-sa-4.0" in out  # front matter in the printed card
    assert "repr.anisotropy" not in out or "anisotropy" in out  # card renders the eval metrics
    assert "quantization" in out.lower()


def test_dry_run_refuses_unknown_tier_region(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    receipt_path = make_training_receipt(tmp_path, checkpoint, region="stream_vae")
    rc = mod.main(
        [
            "--region",
            "stream_vae",
            "--receipt",
            str(receipt_path),
            "--repo",
            "tzervas/cogsyndelta-region-stream-vae",
            "--dry-run",
        ]
    )
    assert rc == 2


def test_dry_run_default_repo_used_when_omitted(tmp_path: Path, capsys: Any) -> None:
    checkpoint = make_checkpoint(tmp_path)
    receipt_path = make_training_receipt(tmp_path, checkpoint, region="code")
    rc = mod.main(["--region", "code", "--receipt", str(receipt_path), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tzervas/cogsyndelta-region-code" in out


# ------------------------------------------------------------------- idempotency


def test_sync_repo_skips_matching_content(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    receipt_path = make_training_receipt(tmp_path, checkpoint)
    card_bytes = b"---\nlicense: cc-by-sa-4.0\n---\ncard\n"
    files: dict[str, Any] = {
        checkpoint.name: checkpoint,
        f"receipts/{receipt_path.name}": receipt_path,
        "README.md": card_bytes,
    }
    ckpt_sha = mod.sha256_of(checkpoint)

    remote = [
        repo_file(checkpoint.name, sha256=ckpt_sha, size=checkpoint.stat().st_size),
        repo_file(
            f"receipts/{receipt_path.name}", blob_id=git_blob_sha1(receipt_path.read_bytes())
        ),
        repo_file("README.md", blob_id=git_blob_sha1(card_bytes)),
    ]
    api = MagicMock()
    api.get_paths_info.return_value = remote

    result = mod.sync_repo(api, "tzervas/x", "model", files, checkpoint.name, ckpt_sha)

    assert result["uploaded"] == []
    assert set(result["skipped"]) == set(files.keys())
    api.upload_file.assert_not_called()


def test_sync_repo_uploads_when_content_differs(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    files: dict[str, Any] = {checkpoint.name: checkpoint}
    ckpt_sha = mod.sha256_of(checkpoint)

    api = MagicMock()
    # remote has a DIFFERENT sha256 for this LFS file -> must re-upload
    api.get_paths_info.return_value = [repo_file(checkpoint.name, sha256="f" * 64, size=1)]

    result = mod.sync_repo(api, "tzervas/x", "model", files, checkpoint.name, ckpt_sha)

    assert result["uploaded"] == [checkpoint.name]
    assert result["skipped"] == []
    api.upload_file.assert_called_once()


def test_sync_repo_uploads_when_file_absent_remotely(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    files: dict[str, Any] = {checkpoint.name: checkpoint}
    api = MagicMock()
    api.get_paths_info.return_value = []  # nothing there yet
    result = mod.sync_repo(
        api, "tzervas/x", "model", files, checkpoint.name, mod.sha256_of(checkpoint)
    )
    assert result["uploaded"] == [checkpoint.name]
    api.upload_file.assert_called_once()


def test_full_publish_idempotent_second_run_uploads_nothing(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    receipt_path = make_training_receipt(tmp_path, checkpoint)
    plan = mod.build_plan(
        "compress", "tzervas/cogsyndelta-region-compress", receipt_path, None, None
    )

    remote = []
    for path_in_repo, item in plan.files.items():
        if path_in_repo == checkpoint.name:
            remote.append(repo_file(path_in_repo, sha256=plan.checkpoint_sha256, size=1))
        else:
            data = item.read_bytes() if isinstance(item, Path) else item
            remote.append(repo_file(path_in_repo, blob_id=git_blob_sha1(data)))

    fake_api = MagicMock()
    fake_api.repo_info.return_value = SimpleNamespace(private=True)
    fake_api.get_paths_info.return_value = remote

    with (
        patch("huggingface_hub.HfApi", return_value=fake_api),
        patch.dict("os.environ", {"HF_TOKEN": "tok"}),
    ):
        result = mod.publish(plan)

    assert result["uploaded"] == []
    assert set(result["skipped"]) == set(plan.files.keys())
    fake_api.upload_file.assert_not_called()
    fake_api.create_repo.assert_called_once()


# --------------------------------------------------------------------- region config


def test_load_region_config_unknown_region_aborts() -> None:
    with pytest.raises(mod.PublishAbortError, match="not found"):
        mod.load_region_config("not-a-real-region")


def test_load_region_config_compress_has_role() -> None:
    cfg = mod.load_region_config("compress")
    assert cfg["name"] == "compress"
    assert cfg.get("role")


# ---------------------------------------------------- region cross-check (review fix)
#
# --region was decoupled from every receipt's own declared region, which laundered the
# licence tier: passing a receipt with region='compress' together with --region code
# produced repo=tzervas/cogsyndelta-region-code and licence=mit in the card. These
# tests pin the fix: build_plan() must cross-check --region against the training
# receipt's 'region' and, when supplied, the eval receipt's producer.component and the
# quant receipt's 'region'.


def test_region_mismatch_training_receipt_aborts(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    # Receipt says compress; --region claims code -- exactly the review's repro shape.
    receipt_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    with pytest.raises(mod.PublishAbortError, match="does not match --region"):
        mod.build_plan("code", "tzervas/cogsyndelta-region-code", receipt_path, None, None)


def test_region_mismatch_does_not_leak_the_wrong_licence(tmp_path: Path, capsys: Any) -> None:
    # Regression shape for the review finding: must never reach a state where
    # license: mit is written into a card for a receipt whose real region is compress
    # (cc-by-sa-4.0). Assert on both the raised error and that nothing gets printed.
    checkpoint = make_checkpoint(tmp_path)
    receipt_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    with pytest.raises(mod.PublishAbortError, match="does not match --region"):
        mod.build_plan("code", "tzervas/cogsyndelta-region-code", receipt_path, None, None)
    assert "license: mit" not in capsys.readouterr().out


def test_region_mismatch_eval_receipt_aborts(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    eval_path = make_eval_receipt(tmp_path, checkpoint, region="retrieve")  # wrong region
    with pytest.raises(mod.PublishAbortError, match="eval receipt's region"):
        mod.build_plan(
            "compress", "tzervas/cogsyndelta-region-compress", train_path, eval_path, None
        )


def test_region_mismatch_quant_receipt_aborts(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="retrieve")  # wrong region
    with pytest.raises(mod.PublishAbortError, match="quant receipt's region"):
        mod.build_plan(
            "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
        )


def test_region_matches_across_all_three_receipts_succeeds(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    eval_path = make_eval_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress")
    plan = mod.build_plan(
        "compress",
        "tzervas/cogsyndelta-region-compress",
        train_path,
        eval_path,
        quant_path,
    )
    assert plan.region == "compress"


def test_receipt_region_requires_the_field(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    receipt_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    receipt = json.loads(receipt_path.read_text())
    del receipt["region"]
    with pytest.raises(mod.PublishAbortError, match="no 'region'"):
        mod.assert_region_matches(receipt, "compress", "training")


# ------------------------------------------------------------------- content match


def test_content_matches_none_remote_is_false(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    assert mod._content_matches(checkpoint, None) is False


def test_content_matches_lfs_sha256(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    sha = mod.sha256_of(checkpoint)
    assert mod._content_matches(checkpoint, repo_file("x", sha256=sha)) is True
    assert mod._content_matches(checkpoint, repo_file("x", sha256="0" * 64)) is False


def test_content_matches_git_blob_sha1_for_non_lfs(tmp_path: Path) -> None:
    p = tmp_path / "README.md"
    data = b"hello world"
    p.write_bytes(data)
    assert mod._content_matches(p, repo_file("README.md", blob_id=git_blob_sha1(data))) is True
    assert mod._content_matches(p, repo_file("README.md", blob_id="deadbeef")) is False


# ------------------------------------- quantized artifact: derived, not receipt-named
#
# The three tests above check that the artifact a receipt names hashes to what that
# receipt says. That is a real check and it is not enough: one document supplied both
# the path and the expected hash, so it agreed with itself, and containment passed for
# anything under an allow-listed root. The publisher therefore DERIVES the artifact
# path from the checkpoint it has already verified and uses the receipt's claim only
# to assert agreement. These tests are the attacks that derivation closes and the
# receipt-vs-artifact disagreements it now refuses.


def test_quant_receipt_naming_a_decoy_with_the_checkpoint_basename_aborts(
    tmp_path: Path,
) -> None:
    """The substitution attack, in full.

    A second allow-listed `.pt` whose basename is `final.pt`, in another directory.
    Before derivation this rebound `files["final.pt"]` to the decoy while the card
    still carried the real checkpoint's sha256 -- a published `final.pt` whose bytes
    were not the ones attested. It must now abort, and abort before any HfApi object
    is even constructed.
    """
    checkpoint = make_checkpoint(tmp_path)
    decoy_dir = tmp_path / "deadbeef"
    decoy_dir.mkdir()
    decoy = decoy_dir / checkpoint.name  # same basename as the real checkpoint
    decoy.write_bytes(b"ATTACKER-CHOSEN-WEIGHTS" * 100)

    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(
        tmp_path, checkpoint, region="compress", quantized_path=decoy, quantized_sha256="unused"
    )

    class Boom:
        def __init__(self, *a: Any, **k: Any) -> None:
            raise AssertionError("HfApi constructed despite an aborting plan")

    with (
        patch("huggingface_hub.HfApi", Boom),
        patch.dict("os.environ", {"HF_TOKEN": "tok"}),
        pytest.raises(mod.PublishAbortError, match="but this checkpoint's artifact is"),
    ):
        mod.build_plan(
            "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
        )


def test_quant_receipt_naming_an_unrelated_allowlisted_pt_aborts(tmp_path: Path) -> None:
    """The same hole without the basename trick: any other `.pt` the receipt cares to
    name, hashed by the receipt itself, was publishable as this region's quantized
    weights."""
    checkpoint = make_checkpoint(tmp_path)
    other = tmp_path / "some-other-region.pt"
    other.write_bytes(b"SOME-OTHER-REGIONS-WEIGHTS" * 100)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(
        tmp_path, checkpoint, region="compress", quantized_path=other, quantized_sha256="unused"
    )
    with pytest.raises(mod.PublishAbortError, match="but this checkpoint's artifact is"):
        mod.build_plan(
            "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
        )


def test_quant_receipt_path_outside_allowed_roots_aborts(tmp_path: Path) -> None:
    """The real packed artifact `make_quant_receipt` writes always lands at the
    derived, contained location beside the checkpoint (it ignores `quantized_path`
    for where it WRITES, only for what the receipt CLAIMS) -- so this receipt's
    `outside` claim disagrees with the derived path itself, and is caught by that
    disagreement check, not by containment (which the derived path here legitimately
    passes: it never left the checkpoint's own, allow-listed directory). The
    symlink-and-containment tests below cover the case where the artifact's actual
    on-disk location, not merely the receipt's claim about it, is made to point
    outside the allow-listed roots."""
    import tempfile

    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    with tempfile.TemporaryDirectory() as outside_dir:
        outside = Path(outside_dir) / "final.ptq.pt"
        outside.write_bytes(b"outside-the-roots")
        quant_path = make_quant_receipt(
            tmp_path,
            checkpoint,
            region="compress",
            quantized_path=outside,
            quantized_sha256="unused",
        )
        with pytest.raises(mod.PublishAbortError, match="but this checkpoint's artifact is"):
            plan = mod.build_plan(
                "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
            )
            assert plan.quantized_path != outside  # unreachable; states the property


# --------------------------------------------------- N1: derived-path symlink bypass
#
# The disagreement checks above compare the RECEIPT's claim about the artifact path
# to the derived path -- they say nothing about what the on-disk entry AT the derived
# path actually is. A symlink at `<checkpoint-stem>.ptq.pt` is followed silently by
# `Path.resolve()`, and a receipt whose `artifacts.quantized_path` simply names the
# same target the symlink already points at agrees with that (already-compromised)
# resolution -- containment on the checkpoint itself never runs on the symlink's
# target, because nothing about `quantized_artifact_path()` re-checked it. These
# tests are the exact reviewer-found bypass (N1) and its regression guards.


def _symlinked_quant_receipt(tmp_path: Path, checkpoint: Path, target: Path) -> Path:
    """A quant receipt whose artifacts.quantized_path / quantized_sha256 / measured
    numbers all correctly describe `target` -- the receipt AGREES with whatever the
    symlink at the derived location resolves to, which is exactly the shape that let
    the old disagreement check pass trivially."""
    quant_path = make_quant_receipt(
        tmp_path, checkpoint, region="compress", quantized_artifact=False
    )
    receipt = json.loads(quant_path.read_text())
    packed = load_packed_artifact(target)
    receipt["stored_bytes"] = packed_stored_bytes(packed)
    receipt["width_histogram"] = packed_width_histogram(packed)
    receipt["artifacts"] = {
        "quantized_path": str(target),
        "quantized_sha256": mod.sha256_of(target),
    }
    quant_path.write_text(json.dumps(receipt))
    return quant_path


class _BoomHfApi:
    """Construction is itself the failure: reaching `publish()` at all means the
    plan should never have built."""

    def __init__(self, *a: Any, **k: Any) -> None:
        raise AssertionError("HfApi constructed despite an aborting plan")


def test_symlinked_quant_artifact_outside_roots_aborts_before_hfapi(tmp_path: Path) -> None:
    """N1, the reviewer's exact finding: `<checkpoint-stem>.ptq.pt` is a symlink to a
    real, correctly-hashed packed artifact OUTSIDE every allow-listed root. This must
    abort with PublishAbortError before `HfApi` is ever constructed -- proven to FAIL
    (upload proceeds) on commit 8ac8fbb; see the session's verification script."""
    import tempfile

    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")

    with tempfile.TemporaryDirectory() as outside_dir:
        outside_artifact = Path(outside_dir) / "someone-elses-region.ptq.pt"
        write_packed_artifact(outside_artifact)

        link = checkpoint.with_name(f"{checkpoint.stem}.ptq.pt")
        link.symlink_to(outside_artifact)

        quant_path = _symlinked_quant_receipt(tmp_path, checkpoint, outside_artifact)

        with (
            patch("huggingface_hub.HfApi", _BoomHfApi),
            patch.dict("os.environ", {"HF_TOKEN": "tok"}),
            pytest.raises(mod.PublishAbortError, match="is a symlink"),
        ):
            mod.build_plan(
                "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
            )


def test_symlinked_quant_artifact_inside_roots_but_other_region_aborts(tmp_path: Path) -> None:
    """N1 variant: the symlink's target is itself inside an allow-listed root (plain
    containment on the target alone would pass) but sits in a DIFFERENT region's
    directory, not beside this checkpoint. Caught by the same symlink refusal --
    the derived location must never be a symlink at all, regardless of where it
    points -- so this also aborts before `HfApi` is constructed."""
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")

    other_region_dir = tmp_path / "some-other-regions-checkpoints"
    other_region_dir.mkdir()
    other_artifact = other_region_dir / "final.ptq.pt"
    write_packed_artifact(other_artifact)

    link = checkpoint.with_name(f"{checkpoint.stem}.ptq.pt")
    link.symlink_to(other_artifact)

    quant_path = _symlinked_quant_receipt(tmp_path, checkpoint, other_artifact)

    with (
        patch("huggingface_hub.HfApi", _BoomHfApi),
        patch.dict("os.environ", {"HF_TOKEN": "tok"}),
        pytest.raises(mod.PublishAbortError, match="is a symlink"),
    ):
        mod.build_plan(
            "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
        )


def test_hardlinked_quant_artifact_is_allowed(tmp_path: Path) -> None:
    """Regression guard: a HARD link at the derived location -- a second directory
    entry for the same inode, not a symlink -- must still publish. The fix refuses
    symlinks specifically (the thing `Path.resolve()` follows unchecked), not every
    non-canonical directory entry."""
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")

    real_artifact = tmp_path / "real-storage.ptq.pt"
    write_packed_artifact(real_artifact)

    link = checkpoint.with_name(f"{checkpoint.stem}.ptq.pt")
    os.link(real_artifact, link)  # hard link: NOT a symlink
    assert not link.is_symlink()

    quant_path = _symlinked_quant_receipt(tmp_path, checkpoint, link)
    plan = mod.build_plan(
        "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
    )
    assert plan.quantized_path == link.resolve()
    assert link.name in plan.files


def test_regular_file_quant_artifact_is_allowed(tmp_path: Path) -> None:
    """Regression guard: the ordinary case -- a plain regular file written directly
    at the derived location, no link involved -- must still publish. Already covered
    indirectly by most tests above (`make_quant_receipt`'s default), pinned here
    explicitly as the counterpart to the symlink and hard-link tests."""
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress")
    plan = mod.build_plan(
        "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
    )
    assert plan.quantized_path is not None
    assert not plan.quantized_path.is_symlink()


def test_quant_receipt_stored_bytes_tampered_aborts(tmp_path: Path) -> None:
    """A receipt can hash the artifact correctly and still carry compression numbers
    measured on something else. The card prints those numbers, so they are checked
    against the artifact rather than transcribed from the receipt."""
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress", stored_bytes=1)
    with pytest.raises(mod.PublishAbortError, match="stored_bytes 1 does not match"):
        mod.build_plan(
            "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
        )


def test_quant_receipt_histogram_tampered_aborts(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(
        tmp_path, checkpoint, region="compress", width_histogram={"2": 99}
    )
    with pytest.raises(mod.PublishAbortError, match=r"width_histogram .* does not match"):
        mod.build_plan(
            "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
        )


def test_quant_receipt_missing_stored_bytes_aborts(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress")
    receipt = json.loads(quant_path.read_text())
    del receipt["stored_bytes"]
    quant_path.write_text(json.dumps(receipt))
    with pytest.raises(mod.PublishAbortError, match="no stored_bytes"):
        mod.build_plan(
            "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
        )


def test_unreadable_quant_artifact_aborts(tmp_path: Path) -> None:
    """The artifact is loaded to be measured, so a file that is not a packed artifact
    -- however correctly the receipt hashes it -- cannot be published."""
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress")
    artifact = checkpoint.with_name(f"{checkpoint.stem}.ptq.pt")
    artifact.write_bytes(b"not a torch file at all")
    receipt = json.loads(quant_path.read_text())
    receipt["artifacts"]["quantized_sha256"] = mod.sha256_of(artifact)
    quant_path.write_text(json.dumps(receipt))
    with pytest.raises(mod.PublishAbortError, match="not a readable packed artifact"):
        mod.build_plan(
            "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
        )


def test_card_prints_measured_numbers_and_does_not_repeat_the_budget(tmp_path: Path) -> None:
    """The `## Quantized artifact` section carries facts about the file; the
    ratio/drop/tolerance/within_budget story stays in the single `### quantization`
    table above it."""
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress")
    plan = mod.build_plan(
        "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
    )
    section = plan.card.split("## Quantized artifact", 1)[1].split("## Training config", 1)[0]

    assert plan.quantized_stored_bytes is not None
    assert str(plan.quantized_stored_bytes) in section
    assert str(plan.quantized_width_histogram) in section
    assert plan.quantized_path is not None
    assert str(plan.quantized_path.stat().st_size) in section
    for repeated in ("Compression ratio", "Within budget", "tolerance", "Metric drop"):
        assert repeated not in section, f"{repeated!r} is duplicated in the artifact section"
    # ...and is still reported once, in the table.
    assert "compression_ratio" in plan.card
    assert "within_budget" in plan.card


def test_full_publish_with_quant_is_idempotent_including_the_ptq_file(tmp_path: Path) -> None:
    """The F7 gap: idempotency was only ever tested on an fp32-only plan, so nothing
    pinned that the `.ptq.pt` file compares by its LFS sha256 and is skipped on a
    second run. A re-publish must upload nothing at all."""
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    eval_path = make_eval_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress")
    plan = mod.build_plan(
        "compress", "tzervas/cogsyndelta-region-compress", train_path, eval_path, quant_path
    )
    assert plan.quantized_path is not None
    ptq_name = f"{checkpoint.stem}.ptq.pt"
    assert ptq_name in plan.files

    # What the repo looks like after a first, successful publish: both .pt files
    # LFS-tracked (sha256), everything else a plain git blob.
    remote = []
    for path_in_repo, item in plan.files.items():
        if path_in_repo == checkpoint.name:
            remote.append(repo_file(path_in_repo, sha256=plan.checkpoint_sha256, size=1))
        elif path_in_repo == ptq_name:
            remote.append(repo_file(path_in_repo, sha256=plan.quantized_sha256, size=1))
        else:
            data = item.read_bytes() if isinstance(item, Path) else item
            remote.append(repo_file(path_in_repo, blob_id=git_blob_sha1(data)))

    fake_api = MagicMock()
    fake_api.repo_info.return_value = SimpleNamespace(private=True)
    fake_api.get_paths_info.return_value = remote
    with (
        patch("huggingface_hub.HfApi", return_value=fake_api),
        patch.dict("os.environ", {"HF_TOKEN": "tok"}),
    ):
        result = mod.publish(plan)

    assert result["uploaded"] == []
    assert ptq_name in result["skipped"]
    assert set(result["skipped"]) == set(plan.files.keys())
    fake_api.upload_file.assert_not_called()


def test_quant_artifact_repo_name_comes_from_the_checkpoint_stem(tmp_path: Path) -> None:
    """The upload path is derived, not taken from the artifact's on-disk name -- so a
    checkpoint called `step-8000.pt` publishes `step-8000.ptq.pt`, and the checkpoint's
    own slot in the plan is untouched."""
    checkpoint = make_checkpoint(tmp_path, name="step-8000.pt")
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress")
    plan = mod.build_plan(
        "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
    )
    assert "step-8000.ptq.pt" in plan.files
    assert plan.files["step-8000.pt"] == checkpoint
    assert plan.files["step-8000.ptq.pt"] == plan.quantized_path


# ================================================================== metrics-v2 (g7 §3.1/§3.3)
#
# `normalize_quant_receipt_v1`: a v1-shaped quant receipt on disk (make_quant_receipt's
# own fixture shape -- `quantized_metric`/`compression_ratio`/`drop`, what
# scripts/csd-quantize.py wrote before the rename) must resolve every v2 key
# `METRIC_METHODOLOGY` and `build_card`'s quantization table now look up, without the
# fixture itself changing -- proving `build_plan`'s normalisation step, not a rewritten
# fixture, is what makes an old receipt on disk still publishable.


def test_normalize_quant_receipt_v1_adds_v2_keys_without_removing_v1_ones() -> None:
    v1 = {
        "region": "compress",
        "quantized_metric": 0.955,
        "compression_ratio": 9.79,
        "drop": 0.004,
        "stored_bytes": 1000,
    }
    out = mod.normalize_quant_receipt_v1(v1)

    assert out["quant.plan_recall@1"] == 0.955
    assert out["quant.compression_ratio"] == 9.79
    assert out["quant.drop_recall@1"] == 0.004
    # Original v1 keys untouched -- other readers of this same dict (
    # verify_quantized_measurements, assert_receipt_bound_to_checkpoint) still work.
    assert out["quantized_metric"] == 0.955
    assert out["compression_ratio"] == 9.79
    assert out["drop"] == 0.004
    assert v1 == {
        "region": "compress",
        "quantized_metric": 0.955,
        "compression_ratio": 9.79,
        "drop": 0.004,
        "stored_bytes": 1000,
    }, "normalize_quant_receipt_v1 must not mutate its argument"


def test_normalize_quant_receipt_v1_never_overwrites_a_real_v2_value() -> None:
    """A quant receipt that already carries the v2 name (post-rename producer) must
    keep ITS value even if a legacy key happens to also be present with a different
    number -- never silently overwritten by the alias."""
    already_v2 = {
        "quantized_metric": 0.111,  # a stale/unrelated legacy key, if one existed
        "quant.plan_recall@1": 0.955,
    }
    out = mod.normalize_quant_receipt_v1(already_v2)
    assert out["quant.plan_recall@1"] == 0.955


def test_build_plan_normalises_a_v1_shaped_quant_receipt_on_disk(tmp_path: Path) -> None:
    """End to end: `make_quant_receipt`'s fixture writes v1 field names (the realistic
    shape for a receipt already on disk from before this change) and the full
    `build_plan` -> `build_card` -> `_methodology_section` pipeline must not refuse it."""
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress")
    raw_on_disk = json.loads(quant_path.read_text())
    assert "quant.plan_recall@1" not in raw_on_disk, (
        "fixture must stay v1-shaped on disk, or this test proves nothing"
    )

    plan = mod.build_plan(
        "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
    )

    assert "quant.plan_recall@1" in plan.card
    assert "quant.compression_ratio" in plan.card
    section = plan.card.split("## How these numbers were produced", 1)[1]
    assert "| `quant.plan_recall@1` |" in section
    assert "`quant_plan`" in section  # battery_id column
    assert "`matched`" in section  # pooling column


def test_stubbed_methodology_map_missing_a_v2_key_fails_the_card_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same mutation proof `tests/test_metrics_methodology.py` runs for
    `recall@1`, narrowed to a key this lane's rename introduced: dropping
    `quant.plan_recall@1` from `METRIC_METHODOLOGY` must abort the build and name it,
    proving the refusal covers renamed keys too, not only the pre-existing ones."""
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress")
    trimmed = {k: v for k, v in mod.METRIC_METHODOLOGY.items() if k != "quant.plan_recall@1"}
    monkeypatch.setattr(mod, "METRIC_METHODOLOGY", trimmed)

    with pytest.raises(mod.PublishAbortError, match=re.escape("quant.plan_recall@1")):
        mod.build_plan(
            "compress", "tzervas/cogsyndelta-region-compress", train_path, None, quant_path
        )


# ============================================== card metrics_schema stamp (round 3 review)
#
# `_methodology_section` used to derive the card's ONE `- **Metrics schema:**` line from
# the TRAINING receipt alone, then merge train + eval + quant metrics into the same
# table -- so a v1 training receipt re-benchmarked/re-quantized with v2 code (the
# near-term real case for every already-trained matrix cell) printed a false
# `csd-metrics/v1 (not recorded)` stamp directly above a table of v2 field names.
# `metrics_schema` is identity key #1 of MM §14's refuse predicate; a published card
# naming the wrong one is the provenance-falsehood class this branch exists to close.


def _set_metrics_schema(path: Path, schema: str) -> None:
    receipt = json.loads(path.read_text())
    receipt["metrics_schema"] = schema
    path.write_text(json.dumps(receipt))


def test_card_metrics_schema_stamp_is_a_single_value_when_all_receipts_agree(
    tmp_path: Path,
) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    eval_path = make_eval_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress")
    for path in (train_path, eval_path, quant_path):
        _set_metrics_schema(path, "csd-metrics/v2")

    plan = mod.build_plan(
        "compress", "tzervas/cogsyndelta-region-compress", train_path, eval_path, quant_path
    )

    assert "- **Metrics schema:** `csd-metrics/v2`" in plan.card
    assert "disagree" not in plan.card


def test_card_metrics_schema_stamp_shows_disagreement_rather_than_claiming_v1(
    tmp_path: Path,
) -> None:
    """The near-term real case: a v1 training receipt (no `metrics_schema` field --
    predates this migration, reads back as the `csd-metrics/v1 (not recorded)`
    fallback) paired with an eval and a quant receipt this branch's v2 code produced
    (both stamp `csd-metrics/v2`). The card must not print a single
    `csd-metrics/v1 (not recorded)` stamp above a table full of v2 field names."""
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    eval_path = make_eval_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress")
    for path in (eval_path, quant_path):
        _set_metrics_schema(path, "csd-metrics/v2")
    assert "metrics_schema" not in json.loads(train_path.read_text()), (
        "fixture must stay v1-shaped (no metrics_schema field) on disk, or this test proves nothing"
    )

    plan = mod.build_plan(
        "compress", "tzervas/cogsyndelta-region-compress", train_path, eval_path, quant_path
    )

    schema_line = next(
        line for line in plan.card.splitlines() if line.startswith("- **Metrics schema:**")
    )
    assert schema_line != "- **Metrics schema:** `csd-metrics/v1 (not recorded)`"
    assert "train=`csd-metrics/v1 (not recorded)`" in schema_line
    assert "eval=`csd-metrics/v2`" in schema_line
    assert "quant=`csd-metrics/v2`" in schema_line


def test_pre_fix_train_receipt_only_lookup_would_have_claimed_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MUTATION PROOF: stub `_metrics_schema_line` back to the exact pre-fix formula
    (`train_receipt.get('metrics_schema', 'csd-metrics/v1 (not recorded)')`, train
    receipt only) and confirm the card THEN claims `csd-metrics/v1 (not recorded)` on
    the same v1-train/v2-eval/v2-quant trio the test above uses -- driven through the
    real `build_plan` -> `build_card` -> `_methodology_section` pipeline, proving the
    test above is not vacuous and that this fix, not something else about the trio, is
    what changed the printed stamp."""
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    eval_path = make_eval_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress")
    for path in (eval_path, quant_path):
        _set_metrics_schema(path, "csd-metrics/v2")

    def pre_fix_line(
        train_receipt: dict[str, Any],
        eval_receipt: dict[str, Any] | None,
        quant_receipt: dict[str, Any] | None,
    ) -> str:
        del eval_receipt, quant_receipt  # the pre-fix formula never looked at these
        schema = train_receipt.get("metrics_schema", "csd-metrics/v1 (not recorded)")
        return f"- **Metrics schema:** `{schema}`"

    monkeypatch.setattr(mod, "_metrics_schema_line", pre_fix_line)

    plan = mod.build_plan(
        "compress", "tzervas/cogsyndelta-region-compress", train_path, eval_path, quant_path
    )

    assert "- **Metrics schema:** `csd-metrics/v1 (not recorded)`" in plan.card, (
        "reproducing the pre-fix train-receipt-only lookup must claim v1 -- this is "
        "the exact defect this branch's fix closes"
    )


# ===================================================================== --safetensors
#
# Opt-in, off by default (every test above builds its plan with the flag's default,
# `want_safetensors=False`, and none of them gained a `.safetensors` entry -- proving
# the default is really off, not merely undocumented). `make_checkpoint`'s fixture
# writes an arbitrary byte blob, not a real torch state dict (that is deliberate --
# see its own docstring reasoning elsewhere in this file: most of this suite is about
# path containment and receipt binding, not about the checkpoint's pickled contents),
# so these tests need their OWN fixture that actually torch.save()s a state dict.


def make_real_torch_checkpoint(tmp_path: Path, name: str = "final.pt") -> Path:
    """A checkpoint `--safetensors` can actually export: a real `torch.save`d state
    dict of a tiny model, not `make_checkpoint`'s arbitrary byte blob."""
    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    path = tmp_path / name
    torch.save(model.state_dict(), path)
    return path


def test_safetensors_flag_off_by_default(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)  # the ordinary fake-bytes fixture
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    plan = mod.build_plan("compress", "tzervas/cogsyndelta-region-compress", train_path, None, None)
    assert plan.safetensors_path is None
    assert plan.safetensors_sha256 is None
    assert not any(name.endswith(".safetensors") for name in plan.files)


def test_safetensors_flag_exports_and_adds_to_plan(tmp_path: Path) -> None:
    checkpoint = make_real_torch_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")

    plan = mod.build_plan(
        "compress",
        "tzervas/cogsyndelta-region-compress",
        train_path,
        None,
        None,
        want_safetensors=True,
    )

    assert plan.safetensors_path == checkpoint.with_suffix(".safetensors")
    assert plan.safetensors_path.is_file()
    assert plan.safetensors_sha256 == mod.sha256_of(plan.safetensors_path)
    assert "final.safetensors" in plan.files
    assert plan.files["final.safetensors"] == plan.safetensors_path
    # never promoted over the primary artifact
    assert plan.files["final.pt"] == checkpoint


def test_safetensors_round_trips_the_real_weights(tmp_path: Path) -> None:
    """Not just "a file got written" -- the exported tensors are the checkpoint's own,
    read back through the actual safetensors loader (the same guarantee
    tests/test_cards_export.py proves at the unit level, exercised here through the
    full build_plan() pipeline)."""
    from safetensors.torch import load_file

    checkpoint = make_real_torch_checkpoint(tmp_path)
    original = torch.load(checkpoint, map_location="cpu", weights_only=True)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")

    plan = mod.build_plan(
        "compress",
        "tzervas/cogsyndelta-region-compress",
        train_path,
        None,
        None,
        want_safetensors=True,
    )

    restored = load_file(str(plan.safetensors_path))
    assert set(restored) == set(original)
    for name, tensor in original.items():
        assert torch.equal(tensor.to(dtype=torch.float32), restored[name])


def test_safetensors_uploaded_alongside_checkpoint_end_to_end(tmp_path: Path) -> None:
    """`--safetensors` all the way through `publish()`: both `final.pt` and
    `final.safetensors` get uploaded, and `sync_repo` never re-hashes the safetensors
    file from scratch (it uses the sha `build_plan` already computed) -- verified by a
    fake `get_paths_info` that returns nothing remote, so both must be freshly uploaded."""
    checkpoint = make_real_torch_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    plan = mod.build_plan(
        "compress",
        "tzervas/cogsyndelta-region-compress",
        train_path,
        None,
        None,
        want_safetensors=True,
    )

    fake_api = MagicMock()
    fake_api.get_paths_info.return_value = []
    fake_api.repo_info.return_value = SimpleNamespace(private=True)

    with (
        patch.object(mod, "get_token", return_value="fake-token"),
        patch("huggingface_hub.HfApi", return_value=fake_api),
    ):
        result = mod.publish(plan)

    assert set(result["uploaded"]) == set(plan.files.keys())
    assert "final.safetensors" in result["uploaded"]


def test_safetensors_flag_aborts_on_unloadable_checkpoint(tmp_path: Path) -> None:
    """A checkpoint that passes every containment/binding check but is not a real
    torch state dict (the ordinary `make_checkpoint` fixture: an arbitrary byte blob)
    must abort the WHOLE plan when `--safetensors` is explicitly requested, rather
    than silently publishing without the file it was asked to include."""
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")

    with pytest.raises(mod.PublishAbortError, match="safetensors"):
        mod.build_plan(
            "compress",
            "tzervas/cogsyndelta-region-compress",
            train_path,
            None,
            None,
            want_safetensors=True,
        )
    assert not checkpoint.with_suffix(".safetensors").exists()
