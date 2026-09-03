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
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

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


def make_training_receipt(
    tmp_path: Path,
    checkpoint: Path,
    region: str = "compress",
    checkpoint_sha256: str | None = None,
    name: str = "compress-20260902T211539Z.json",
) -> Path:
    receipt: dict[str, Any] = {
        "schema": "csd-pretrain-receipt/v1",
        "region": region,
        "recorded": "2026-09-02T21:15:39Z",
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
    if checkpoint_sha256 is not None:
        receipt["artifacts"] = {"checkpoint_sha256": checkpoint_sha256}
    path = tmp_path / name
    path.write_text(json.dumps(receipt))
    return path


def make_eval_receipt(tmp_path: Path, checkpoint: Path, region: str = "compress") -> Path:
    receipt = {
        "producer": {"project": "cogsyndelta", "component": region},
        "stage": "eval",
        "metrics": {
            "rank.recall@1": 0.49,
            "repr.anisotropy": 0.1306,
            "repr.effective_rank_ratio": 0.459,
        },
        "gates": {"beats_untrained": True, "not_anisotropic": True, "uses_its_dimensions": True},
        "artifacts": {"checkpoint": str(checkpoint)},
        "schema": "model-pipeline-receipt/v1",
    }
    path = tmp_path / f"cogsyndelta-{region}-eval-20260902T183739Z.json"
    path.write_text(json.dumps(receipt))
    return path


def make_quant_receipt(tmp_path: Path, checkpoint: Path, region: str = "compress") -> Path:
    receipt = {
        "region": region,
        "checkpoint": str(checkpoint),
        "corpus_fingerprint": "5de8a340c37137554824578a0040604d",
        "tolerance": 0.01,
        "fp32_metric_recomputed": 0.496,
        "quantized_metric": 0.488,
        "drop": 0.0078,
        "within_budget": True,
        "fp32_bytes": 64084992,
        "stored_bytes": 6519016,
        "compression_ratio": 9.83,
    }
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
    ],
)
def test_licence_tier_matches_decision_2026_09_02(region: str, tier: str) -> None:
    assert mod.licence_tier(region) == tier


@pytest.mark.parametrize("region", ["residual_mlp", "stream_vae", "some_unaudited_region"])
def test_licence_tier_unknown_aborts(region: str) -> None:
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


def test_unknown_tier_aborts_full_plan_before_any_file_read(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    receipt = make_training_receipt(tmp_path, checkpoint, region="stream_vae")
    with pytest.raises(mod.PublishAbortError, match="licence tier"):
        mod.build_plan("stream_vae", "tzervas/whatever", receipt, None, None)


# --------------------------------------------------------------- sha256 (req 3, 5)


def test_sha_mismatch_aborts(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    receipt_path = make_training_receipt(tmp_path, checkpoint, checkpoint_sha256="0" * 64)
    receipt = json.loads(receipt_path.read_text())
    with pytest.raises(mod.PublishAbortError, match="sha256 mismatch"):
        mod.verify_checkpoint_sha(checkpoint, receipt)


def test_sha_computed_and_recorded_when_absent(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    receipt_path = make_training_receipt(tmp_path, checkpoint)  # no checkpoint_sha256
    receipt = json.loads(receipt_path.read_text())
    got = mod.verify_checkpoint_sha(checkpoint, receipt)
    assert got == mod.sha256_of(checkpoint)
    assert len(got) == 64


def test_sha_match_passes_through(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    real_sha = mod.sha256_of(checkpoint)
    receipt_path = make_training_receipt(tmp_path, checkpoint, checkpoint_sha256=real_sha)
    receipt = json.loads(receipt_path.read_text())
    assert mod.verify_checkpoint_sha(checkpoint, receipt) == real_sha


def test_missing_checkpoint_aborts(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pt"
    receipt = {"checkpoint": str(missing)}
    with pytest.raises(mod.PublishAbortError, match="not found"):
        mod.verify_checkpoint_sha(missing, receipt)


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
