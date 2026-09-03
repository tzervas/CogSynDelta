#!/usr/bin/env python3
"""Publish one region's checkpoint (+ receipts + model card) to a private HF model repo.

THE GAP THIS FILLS
`scripts/csd-hf-repos.py` provisions the 8 private repos under `tzervas`; nothing
uploads a checkpoint into one of them. `scripts/csd-corpus-publish.py` is the closest
analog in design -- private=True on every create, per-file verification against what
HF actually reports rather than trusting that a repo existing means its contents
landed -- but it publishes bulk *corpora* to dataset repos, not trained checkpoints to
model repos. This script is that missing step, for models.

PRIVATE BY CONSTRUCTION
`create_repo(private=True, exist_ok=True)` is followed immediately by `repo_info()`,
and if that does not report `private=True` the tool aborts before a single byte is
uploaded. There is deliberately no `--public` flag anywhere in this file -- the
operator authorised private publication only, and the way to keep that authorisation
from eroding one convenience flag at a time is to never add the flag.

LICENCE TIER IS DERIVED, NOT ASKED FOR
`docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`, "Decision 2026-09-02", sets a per-region
licence table: a region whose training data carries a non-commercial or share-alike
term releases under the matching restrictive licence; a region with no such input
stays MIT. `LICENCE_TIER` below is that table transcribed. Two regions --
`residual_mlp` and `stream_vae` -- were never run through that audit at all, so their
tier is UNKNOWN. This script refuses to publish a region whose tier is unknown rather
than guess a licence for it (the operator's rule, recorded in the
`csd-release-licence-decision` memory: "the overall model ... is going to have
whatever the strictest license is between its data sets and sub models" -- silently
defaulting an unaudited region to MIT would violate that rule the moment its real
input turns out to need one). `vl_latent` is BLOCKING, not merely a tier: section 4
states it is unreleasable as trained, so `licence_tier()` refuses it outright before
ever consulting `LICENCE_TIER`'s "mit" entry for it (that entry is for the region's
eventual composite replacement, per Decision 2026-09-02, not today's checkpoint).
`--region` is also cross-checked against every receipt's own declared region before
anything else is read from them -- the licence tier is derived from `--region`, and an
unchecked mismatch would let a receipt's real licence tier be laundered under whatever
`--region` claims.

CHECKPOINT PATH IS CONTAINED, NOT TRUSTED
A receipt's 'checkpoint' value is attacker-reachable -- anyone who can write a receipt
JSON, or misdirect an operator/agent into pointing `--receipt` at one, otherwise
controls what file this script reads. `checkpoint_path_from_receipt()` resolves that
path and requires it to sit under an allow-listed root (`/akula-data/csd`, this repo)
and carry an allow-listed suffix (`.pt`, `.safetensors`) before anything is hashed --
closing off "point a receipt at any file readable by this process and get it uploaded,
with its sha256 published in the model card."

IDEMPOTENCY
Every file this script would upload is hashed first and compared against what the
repo already has (`HfApi.get_paths_info(..., expand=True)`): an LFS-tracked file
(the checkpoint) compares by its recorded `lfs.sha256`; a small text file (a receipt,
the README) compares by git's own blob hash, since HF does not compute a sha256 for
non-LFS blobs. A file whose content already matches is skipped, not re-uploaded, so a
re-run with the same inputs uploads nothing and still exits 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OWNER = "tzervas"
DEFAULT_BASE = "cogsyndelta"
DEFAULT_REGIONS_CONFIG = REPO_ROOT / "config" / "mind" / "csd-regions.json"

# Mirrors the suffix table in scripts/csd-hf-repos.py exactly, for the three regions
# whose repo name does not follow the plain "-region-<name>" pattern. Every other
# region (reason, classify_banking77, classify_go_emotions, ...) gets the plain
# pattern -- those repos are not in csd-hf-repos.py's REPOS list yet, so this is an
# extension of its convention, not a transcription of it; create_repo(exist_ok=True)
# provisions them the first time this script runs against one.
REGION_REPO_SUFFIX = {
    "residual_mlp": "-region-residual",
    "stream_vae": "-region-stream-vae",
    "vl_latent": "-vl-jepa",
}

# Source: docs/design/LICENCE-FOR-OPEN-WEIGHTS.md, "Decision 2026-09-02" and the
# per-region table immediately above it (search "Updated per-region table"). The
# composed-mind entries are recorded here for completeness (the strictest-input rule
# in the csd-release-licence-decision memory) but this CLI only ever resolves a
# per-region tier -- a composed-model publish is not this script's job.
#
# residual_mlp and stream_vae are deliberately absent: no licence audit exists for
# either, so `licence_tier()` raises for them rather than defaulting to MIT.
LICENCE_TIER: dict[str, str] = {
    "code": "mit",
    "classify_banking77": "mit",
    "classify_go_emotions": "mit",
    "reason": "mit",
    "vl_latent": "mit",
    "compress": "cc-by-sa-4.0",
    "retrieve": "cc-by-nc-sa-4.0",
    "composed": "cc-by-nc-sa-4.0",
    "cogsyndelta": "cc-by-nc-sa-4.0",
}

# A region's checkpoint value is 100% receipt-controlled -- see checkpoint_path_from_receipt
# below. Anyone who can write (or misdirect an operator/agent into pointing --receipt at) a
# receipt JSON must not thereby gain "upload this file, whatever it is, to a remote repo,
# with its sha256 published in the model card". Two independent fail-closed checks apply
# before any hashing happens: the resolved path must sit under one of these roots, and it
# must carry one of these suffixes.
ALLOWED_CHECKPOINT_ROOTS: list[Path] = [Path("/akula-data/csd"), REPO_ROOT]
ALLOWED_CHECKPOINT_SUFFIXES: frozenset[str] = frozenset({".pt", ".safetensors"})

# Regions whose training data has no licence grant anywhere in the provenance chain at
# all -- BLOCKING per docs/design/LICENCE-FOR-OPEN-WEIGHTS.md section 4, strictly worse
# than "unknown". Checked inside licence_tier() before the tier lookup below, so a
# BLOCKING region refuses regardless of what LICENCE_TIER says for it (vl_latent's entry
# there is for its eventual composite replacement -- Decision 2026-09-02 -- not for the
# tiny-imagenet checkpoint that exists today, which is why the tier value alone is not
# a safe gate).
BLOCKING_REGIONS: frozenset[str] = frozenset({"vl_latent"})

LICENCE_WHY: dict[str, str] = {
    "code": "no NC or share-alike input in the catalogue, once GitHub-licence-filtered",
    "classify_banking77": "no NC or share-alike input in the catalogue",
    "classify_go_emotions": "no NC or share-alike input in the catalogue",
    "reason": "no NC or share-alike input in the catalogue",
    "vl_latent": "no NC or share-alike input in the catalogue (corpus's own BLOCKING "
    "grant problem is separate from licence family and is not resolved by this tag)",
    "compress": "SNLI (+ government + fiction) repaired corpus is share-alike; "
    "no NC-tagged input identified",
    "retrieve": "GooAQ (NC, accepted 2026-09-02) and Natural Questions / FiQA "
    "(CC BY-SA) are both present in the corpus as trained",
    "composed": "carries the single strictest term across every dataset, submodel, "
    "and the composed model itself",
    "cogsyndelta": "carries the single strictest term across every dataset, submodel, "
    "and the composed model itself",
}


class PublishAbortError(Exception):
    """Raised for any condition this script refuses to proceed past."""


# --------------------------------------------------------------------------- hashing


def sha256_of(item: Path | bytes) -> str:
    """sha256 of a file's bytes (streamed) or of an in-memory blob."""
    h = hashlib.sha256()
    if isinstance(item, Path):
        with item.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    else:
        h.update(item)
    return h.hexdigest()


def git_blob_sha1_of(item: Path | bytes) -> str:
    """The git blob hash HF reports as `blob_id` for a non-LFS file."""
    data = item.read_bytes() if isinstance(item, Path) else item
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 -- matching git's own algorithm


def size_of(item: Path | bytes) -> int:
    return item.stat().st_size if isinstance(item, Path) else len(item)


def _content_matches(
    local: Path | bytes, remote: Any, precomputed_sha256: str | None = None
) -> bool:
    """Does `local`'s content already match what `remote` (a get_paths_info entry) reports?"""
    if remote is None:
        return False
    lfs = getattr(remote, "lfs", None)
    remote_sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
    if remote_sha256:
        local_sha256 = precomputed_sha256 or sha256_of(local)
        return local_sha256 == remote_sha256
    return git_blob_sha1_of(local) == getattr(remote, "blob_id", None)


# ------------------------------------------------------------------------- resolving


def default_repo(region: str, owner: str = DEFAULT_OWNER, base: str = DEFAULT_BASE) -> str:
    suffix = REGION_REPO_SUFFIX.get(region, f"-region-{region.replace('_', '-')}")
    return f"{owner}/{base}{suffix}"


def licence_tier(region: str) -> str:
    if region in BLOCKING_REGIONS:
        raise PublishAbortError(
            f"region {region!r} is BLOCKING per docs/design/LICENCE-FOR-OPEN-WEIGHTS.md "
            "section 4 -- unreleasable as trained (no licence grant anywhere in the "
            "provenance chain). Refusing regardless of any licence tier value on record "
            "for it; BLOCKING is strictly worse than unknown."
        )
    tier = LICENCE_TIER.get(region)
    if tier is None:
        raise PublishAbortError(
            f"region {region!r} has no licence tier in docs/design/LICENCE-FOR-OPEN-WEIGHTS.md "
            "'Decision 2026-09-02' -- refusing to publish it under a guessed licence. "
            "Run the licence audit for this region first."
        )
    return tier


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as e:
        raise PublishAbortError(f"{path}: not found") from e
    except json.JSONDecodeError as e:
        raise PublishAbortError(f"{path}: not valid JSON ({e})") from e


def load_region_config(region: str, regions_path: Path = DEFAULT_REGIONS_CONFIG) -> dict[str, Any]:
    data = load_json(regions_path)
    for r in data.get("regions", []):
        if r.get("name") == region:
            return r
    raise PublishAbortError(f"region {region!r} not found in {regions_path}")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def checkpoint_path_from_receipt(receipt: dict[str, Any]) -> Path:
    """The checkpoint a receipt names -- resolved and contained, never trusted verbatim.

    A receipt's 'checkpoint' value is attacker-reachable (anyone who can write a receipt
    JSON, or misdirect --receipt at one). Without containment this becomes arbitrary-file
    upload: whatever the path names gets hashed, uploaded, and its sha256 published in the
    model card. So: resolve symlinks/`..` first, then require the resolved path to sit
    under an allow-listed root AND carry an allow-listed suffix, and abort before touching
    the filesystem again (no hashing, no upload) if either check fails.
    """
    ckpt = receipt.get("checkpoint") or receipt.get("artifacts", {}).get("checkpoint")
    if not ckpt:
        raise PublishAbortError(
            "training receipt has no 'checkpoint' (or artifacts.checkpoint) path"
        )
    raw = Path(ckpt)
    resolved = raw.resolve()

    if resolved.suffix not in ALLOWED_CHECKPOINT_SUFFIXES:
        raise PublishAbortError(
            f"receipt checkpoint {raw} has suffix {resolved.suffix!r}, not one of "
            f"{sorted(ALLOWED_CHECKPOINT_SUFFIXES)} -- refusing to treat an arbitrary "
            "receipt-named file as a checkpoint"
        )

    allowed_roots = [r.resolve() for r in ALLOWED_CHECKPOINT_ROOTS]
    if not any(_is_relative_to(resolved, root) for root in allowed_roots):
        raise PublishAbortError(
            f"receipt checkpoint {raw} resolves to {resolved}, outside the allow-listed "
            f"checkpoint roots {[str(r) for r in ALLOWED_CHECKPOINT_ROOTS]} -- refusing "
            "to upload a file a receipt points at outside those roots"
        )
    return resolved


def receipt_region(receipt: dict[str, Any], label: str) -> str:
    """The region a receipt itself claims, however that receipt shape spells it.

    Training/quant receipts carry a top-level 'region'; eval receipts (schema
    model-pipeline-receipt/v1) carry it as producer.component instead. Either way the
    field is required, not merely consulted if present -- a receipt that omits it entirely
    would otherwise be a way to dodge the cross-check below.
    """
    if "region" in receipt:
        val = receipt["region"]
    else:
        producer = receipt.get("producer")
        val = producer.get("component") if isinstance(producer, dict) else None
    if not val:
        raise PublishAbortError(
            f"{label} receipt has no 'region' (or producer.component) field -- refusing: "
            "cannot verify it matches --region"
        )
    return str(val)


def assert_region_matches(receipt: dict[str, Any], region: str, label: str) -> None:
    """--region is what licence_tier() and the repo name are derived from. Nothing else
    ties it to the receipt actually being published, so a mismatch -- a CLI typo, or a
    misdirected agent -- would launder that receipt's real licence tier under whatever
    --region claims. One comparison per receipt closes the class."""
    claimed = receipt_region(receipt, label)
    if claimed != region:
        raise PublishAbortError(
            f"{label} receipt's region {claimed!r} does not match --region {region!r} -- "
            "refusing: the licence tier and repo name are derived from --region, and a "
            "mismatch would publish this receipt's data under a different licence than "
            "the one it was actually trained/evaluated under"
        )


def verify_checkpoint_sha(checkpoint: Path, receipt: dict[str, Any]) -> str:
    """Compute the checkpoint's sha256; verify it against the receipt's recorded value if any."""
    if not checkpoint.is_file():
        raise PublishAbortError(f"checkpoint not found: {checkpoint}")
    computed = sha256_of(checkpoint)
    recorded = receipt.get("artifacts", {}).get("checkpoint_sha256")
    if recorded and recorded != computed:
        raise PublishAbortError(
            f"checkpoint sha256 mismatch for {checkpoint}: "
            f"receipt records {recorded}, computed {computed} -- refusing to publish "
            "a checkpoint that does not match the receipt naming it"
        )
    return computed


def code_revision(receipt: dict[str, Any], repo_dir: Path = REPO_ROOT) -> str:
    for key in ("code_revision", "git_rev", "git_sha", "commit", "revision"):
        v = receipt.get(key)
        if v:
            return str(v)
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# --------------------------------------------------------------------------- the card


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _dict_table(d: dict[str, Any]) -> str:
    if not d:
        return "_(none)_\n"
    lines = ["| key | value |", "|---|---|"]
    lines += [f"| `{k}` | {_fmt(v)} |" for k, v in sorted(d.items())]
    return "\n".join(lines) + "\n"


def build_card(
    region: str,
    region_cfg: dict[str, Any],
    tier: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    rev: str,
    train_receipt: dict[str, Any],
    eval_receipt: dict[str, Any] | None,
    quant_receipt: dict[str, Any] | None,
) -> str:
    corpus = train_receipt.get("corpus", {})
    contamination = train_receipt.get("contamination", {})
    parts = [
        "---",
        f"license: {tier}",
        "tags:",
        "- cogsyndelta",
        f"- csd-region-{region}",
        "---",
        "",
        f"# CogSynDelta -- {region}",
        "",
        f"**Faculty / role.** {region_cfg.get('role', '(no role recorded)')}",
        "",
        f"**Router trigger.** {region_cfg.get('router_trigger', '(none recorded)')}",
        "",
        f"**Licence tier.** `{tier}` -- {LICENCE_WHY.get(region, '(reason not recorded)')}. "
        'See `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`, "Decision 2026-09-02".',
        "",
        "## Metrics",
        "",
        "### held_out",
        _dict_table(train_receipt.get("held_out", {})),
        "### untrained_baseline",
        _dict_table(train_receipt.get("untrained_baseline", {})),
        "### gates (training receipt: beats_untrained)",
        _dict_table(train_receipt.get("beats_untrained", {})),
    ]
    if eval_receipt is not None:
        parts += [
            "### gates (eval receipt)",
            _dict_table(eval_receipt.get("gates", {})),
            "### representation (eval receipt, includes anisotropy)",
            _dict_table(
                {
                    k[len("repr.") :]: v
                    for k, v in eval_receipt.get("metrics", {}).items()
                    if k.startswith("repr.")
                }
            ),
        ]
    else:
        parts.append("_No eval receipt supplied -- no independently re-measured anisotropy._\n")
    parts += [
        "### contamination",
        _dict_table({k: v for k, v in contamination.items() if k != "examples"}),
    ]
    if quant_receipt is not None:
        parts += [
            "### quantization",
            _dict_table(
                {
                    k: v
                    for k, v in quant_receipt.items()
                    if k
                    in (
                        "fp32_metric_recomputed",
                        "quantized_metric",
                        "drop",
                        "tolerance",
                        "within_budget",
                        "compression_ratio",
                        "fp32_bytes",
                        "stored_bytes",
                    )
                }
            ),
        ]
    else:
        parts.append("_No quant receipt supplied -- this checkpoint is fp32._\n")
    parts += [
        "## Training config",
        "",
        "```json",
        json.dumps(train_receipt.get("config", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Provenance",
        "",
        f"- **Corpus fingerprint:** `{corpus.get('fingerprint', '(none recorded)')}`",
        f"- **Checkpoint:** `{checkpoint.name}`",
        f"- **Checkpoint sha256:** `{checkpoint_sha256}`",
        f"- **Code revision:** `{rev}`",
        f"- **Region:** `{region}`",
        "",
        "Published by `scripts/csd-publish-checkpoint.py`. Repo is private; the operator's "
        "publishing authorisation covers private repos only -- see that script's module "
        "docstring before ever adding a `--public` flag here.",
        "",
    ]
    return "\n".join(parts)


# ----------------------------------------------------------------------------- token


def get_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise PublishAbortError(
            "HF_TOKEN not set in the environment -- run under: "
            "secret exec HF_TOKEN=gpu/huggingface-token -- python3 scripts/csd-publish-checkpoint.py ..."
        )
    return token


# ------------------------------------------------------------------------- the plan


@dataclass
class Plan:
    region: str
    repo: str
    repo_type: str
    tier: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_sha256_was_recorded: bool
    code_rev: str
    card: str
    files: dict[str, Path | bytes] = field(default_factory=dict)
    """path_in_repo -> local Path or in-memory bytes (the README)."""


def build_plan(
    region: str,
    repo: str,
    train_receipt_path: Path,
    eval_receipt_path: Path | None,
    quant_receipt_path: Path | None,
    regions_config: Path = DEFAULT_REGIONS_CONFIG,
) -> Plan:
    tier = licence_tier(region)  # fail fast, before touching any file we don't need to
    region_cfg = load_region_config(region, regions_config)
    train_receipt = load_json(train_receipt_path)
    eval_receipt = load_json(eval_receipt_path) if eval_receipt_path else None
    quant_receipt = load_json(quant_receipt_path) if quant_receipt_path else None

    # --region drives the licence tier and repo name; verify every receipt actually says
    # it's for this region before reading anything else out of them.
    assert_region_matches(train_receipt, region, "training")
    if eval_receipt is not None:
        assert_region_matches(eval_receipt, region, "eval")
    if quant_receipt is not None:
        assert_region_matches(quant_receipt, region, "quant")

    checkpoint = checkpoint_path_from_receipt(train_receipt)
    checkpoint_sha256 = verify_checkpoint_sha(checkpoint, train_receipt)
    was_recorded = bool(train_receipt.get("artifacts", {}).get("checkpoint_sha256"))
    rev = code_revision(train_receipt)

    card = build_card(
        region,
        region_cfg,
        tier,
        checkpoint,
        checkpoint_sha256,
        rev,
        train_receipt,
        eval_receipt,
        quant_receipt,
    )

    files: dict[str, Path | bytes] = {
        checkpoint.name: checkpoint,
        f"receipts/{train_receipt_path.name}": train_receipt_path,
        "README.md": card.encode("utf-8"),
    }
    if eval_receipt_path:
        files[f"receipts/{eval_receipt_path.name}"] = eval_receipt_path
    if quant_receipt_path:
        files[f"receipts/{quant_receipt_path.name}"] = quant_receipt_path

    return Plan(
        region=region,
        repo=repo,
        repo_type="model",
        tier=tier,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_sha256_was_recorded=was_recorded,
        code_rev=rev,
        card=card,
        files=files,
    )


def print_plan(plan: Plan, dry_run: bool) -> None:
    tag = "DRY-RUN" if dry_run else "APPLY"
    print(f"csd-publish-checkpoint [{tag}]")
    print(f"  region:      {plan.region}")
    print(f"  repo:        {plan.repo} ({plan.repo_type}, private)")
    print(f"  licence:     {plan.tier}")
    print(f"  checkpoint:  {plan.checkpoint_path}")
    recorded = "recorded in receipt" if plan.checkpoint_sha256_was_recorded else "computed here"
    print(f"               sha256={plan.checkpoint_sha256} ({recorded})")
    print(f"  code_rev:    {plan.code_rev}")
    print("  files:")
    for path_in_repo, item in sorted(plan.files.items()):
        print(f"    {path_in_repo:<50} {size_of(item):>12,} bytes")
    print("\n--- README.md ---")
    print(plan.card)


# ---------------------------------------------------------------------------- upload


def ensure_private(api: Any, repo_id: str, repo_type: str = "model") -> None:
    """create_repo(private=True) then verify -- abort before any upload if it isn't."""
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
    if not getattr(info, "private", False):
        raise PublishAbortError(
            f"{repo_id} reports private={getattr(info, 'private', None)!r} after "
            "create_repo(private=True, exist_ok=True) -- aborting before uploading a single byte"
        )


def sync_repo(
    api: Any,
    repo_id: str,
    repo_type: str,
    files: dict[str, Path | bytes],
    checkpoint_path_in_repo: str,
    checkpoint_sha256: str,
) -> dict[str, list[str]]:
    """Upload only what has changed. Returns {'uploaded': [...], 'skipped': [...]}."""
    try:
        remote_list = api.get_paths_info(
            repo_id, list(files.keys()), expand=True, repo_type=repo_type
        )
    except Exception:  # network hiccup, or HF's own error shape -- treat as "unknown", upload
        remote_list = []
    remote = {getattr(r, "path", None): r for r in remote_list}

    uploaded: list[str] = []
    skipped: list[str] = []
    for path_in_repo, item in files.items():
        precomputed = checkpoint_sha256 if path_in_repo == checkpoint_path_in_repo else None
        if _content_matches(item, remote.get(path_in_repo), precomputed):
            skipped.append(path_in_repo)
            continue
        api.upload_file(
            path_or_fileobj=str(item) if isinstance(item, Path) else item,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"csd-publish-checkpoint: {path_in_repo}",
        )
        uploaded.append(path_in_repo)
    return {"uploaded": uploaded, "skipped": skipped}


def publish(plan: Plan) -> dict[str, list[str]]:
    token = get_token()  # before the import: a missing token must touch nothing, not even this
    from huggingface_hub import HfApi  # lazy: --dry-run must not require the package importable

    api = HfApi(token=token)
    ensure_private(api, plan.repo, plan.repo_type)
    result = sync_repo(
        api,
        plan.repo,
        plan.repo_type,
        plan.files,
        plan.checkpoint_path.name,
        plan.checkpoint_sha256,
    )
    info = api.repo_info(repo_id=plan.repo, repo_type=plan.repo_type)
    print(f"  private={getattr(info, 'private', None)} after publish  ({plan.repo})")
    return result


# ------------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--region", required=True, help="region name, e.g. compress")
    ap.add_argument("--receipt", required=True, type=Path, help="training receipt JSON")
    ap.add_argument("--eval-receipt", type=Path, default=None)
    ap.add_argument("--quant-receipt", type=Path, default=None)
    ap.add_argument(
        "--repo",
        default=None,
        help="owner/name; default follows scripts/csd-hf-repos.py's naming convention",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo or default_repo(args.region)
    try:
        plan = build_plan(args.region, repo, args.receipt, args.eval_receipt, args.quant_receipt)
    except PublishAbortError as e:
        print(f"ABORT: {e}", file=sys.stderr)
        return 2

    print_plan(plan, dry_run=args.dry_run)
    if args.dry_run:
        return 0

    try:
        result = publish(plan)
    except PublishAbortError as e:
        print(f"ABORT: {e}", file=sys.stderr)
        return 2

    print(f"  uploaded={len(result['uploaded'])} skipped={len(result['skipped'])}")
    for p in result["uploaded"]:
        print(f"    uploaded {p}")
    for p in result["skipped"]:
        print(f"    skipped  {p} (already matches)")
    return 0


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"  ({time.time() - t0:.1f}s)", file=sys.stderr)
    sys.exit(rc)
