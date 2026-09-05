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
`visual` (nee `vl_latent`) is BLOCKING while its receipt names tiny-imagenet, an
unset corpus, or a fingerprint that is not the admitted Mix B pin in
`config/mind/visual-clean-v1.json`. A receipt whose `corpus.corpus_source` and
`corpus.fingerprint` both match that manifest is publishable under **MIT**, with a
mandatory attribution notice: Mix B includes CLEVR (CC BY 4.0, ATTRIBUTION), and
LICENCE-FOR-OPEN-WEIGHTS.md §"Verdict categories" (`:98-102`) says ATTRIBUTION is
usable for an MIT weights release if the specific notice is carried. The card
prints the CLEVR TASL block; the region tag stays `mit`. `--region` is still
cross-checked against every receipt's own declared region.

CHECKPOINT PATH IS CONTAINED, NOT TRUSTED
A receipt's 'checkpoint' value is attacker-reachable -- anyone who can write a receipt
JSON, or misdirect an operator/agent into pointing `--receipt` at one, otherwise
controls what file this script reads. `checkpoint_path_from_receipt()` resolves that
path and requires it to sit under an allow-listed root (`/akula-data/csd`, this repo)
and carry an allow-listed suffix (`.pt`, `.safetensors`) before anything is hashed --
closing off "point a receipt at any file readable by this process and get it uploaded,
with its sha256 published in the model card."

RECEIPTS ARE BOUND TO THE CHECKPOINT, NOT MERELY THE REGION
Region-matching alone (the check above) does not mean a receipt is *for* the
checkpoint being published -- every region's checkpoint lives at one mutable
`<region>-checkpoints/final.pt` path, so a later training run silently invalidates
every earlier eval/quant receipt that named the same path, while that stale receipt
still passes the region cross-check untouched. `assert_receipt_bound_to_checkpoint()`
closes that gap for every receipt this script is given (training, eval, quant): the
receipt's own `artifacts.checkpoint_sha256` (or top-level `checkpoint_sha256`, for a
receipt shape with no `artifacts` wrapper -- today's quant receipts) must equal the
sha256 this script just computed of the checkpoint file. Absent or mismatched, it
aborts before any upload, naming which receipt failed. Belt-and-braces for a receipt
written before that field existed: the receipt's own recorded timestamp (`recorded` /
`recorded_utc` / `started_utc`, whichever key that receipt shape uses) must not be
older than the checkpoint file's mtime -- a receipt cannot describe a checkpoint that
did not yet exist when it was recorded.

Consequence: none of the 16 real receipts under `/akula-data/csd/receipts` carry
`checkpoint_sha256` today, so every one of them is unpublishable by construction until
the training/eval/quant tools that produce them start writing that field. That is
correct, not a bug to route around -- those receipts were measured against a `final.pt`
that a later training run went on to overwrite at that same mutable path, so a card
built from them would mix the *current* held-out score with an eval/quant score
measured on different, superseded weights, under one published sha256. That is exactly
what the review that prompted this fix found (held_out 0.707 on current weights next to
eval 0.492 / quant 0.496 on older weights, all under one "the checkpoint" heading).

THE QUANTIZED ARTIFACT IS DERIVED, NOT NAMED BY THE RECEIPT
`scripts/csd-quantize.py` writes the packed artifact at exactly
`checkpoint.with_name(f"{checkpoint.stem}.ptq.pt")`. This script recomputes that
from the checkpoint path it has ALREADY resolved, contained and hashed, and uses the
receipt's `artifacts.quantized_path` only to assert the two agree. The receipt never
supplies a path this script will read, and never supplies the basename it uploads
under. Both halves matter and the second is the one that bit: an earlier version did
`files[quantized_path.name] = quantized_path` straight from the receipt, so a receipt
naming any other allow-listed `.pt` whose basename happened to be `final.pt` rebound
`files["final.pt"]` to that file -- and the card, built from the real checkpoint's
sha256, then attested a hash the uploaded bytes did not have. Containment alone did
not stop it (the decoy sat inside an allow-listed root) and neither did the sha256
check (the same receipt supplied both the path and the expected hash, so it agreed
with itself). Derivation stops both: there is exactly one path this script will read
as "the quantized artifact", and it is a function of the checkpoint.

Derivation alone was later found to be incomplete (N1, a second review finding):
"a function of the checkpoint" said nothing about what happens when the on-disk
entry AT that derived path is a symlink -- `Path.resolve()` follows it wherever it
points, unchecked, and a receipt whose `artifacts.quantized_path` simply named the
same foreign target agreed with that already-compromised resolution. Derivation
narrows *which path* gets read; it was never a substitute for checking that the path
is safe to read. `quantized_artifact_path()` now refuses a symlink at the derived
location outright and, for a non-symlink, re-runs it through the exact containment
`_contained_path()` gives the checkpoint (allow-listed root, allow-listed suffix,
same parent directory as the checkpoint) -- see that function's docstring.

THE CARD'S QUANTIZATION NUMBERS ARE MEASURED, NOT TRANSCRIBED
`stored_bytes` and the width histogram are recomputed from the artifact itself
(`cogsyndelta.quant.ptq.load_packed_artifact` + `packed_stored_bytes` /
`packed_width_histogram`, the same accounting `apply_plan` uses to produce the
number a receipt records) and the publish aborts if either disagrees with the
receipt. Without that, a verified-by-sha artifact could still be published under a
compression claim measured from something else -- the sha binds the bytes, but
nothing bound the bytes to the numbers printed beside them.

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    # cogsyndelta.cards.methodology has no heavy dependency (no torch, no
    # tokenizers/pyarrow -- see that module's own docstring), so importing it here
    # costs this script nothing on the --help / licence-refusal / fp32-only paths that
    # motivate every OTHER lazy import in this file. Appended (not prepended), same as
    # `_ptq()` below: an installed `cogsyndelta` still wins over a checkout's `src/`.
    sys.path.append(_SRC)

from cogsyndelta.cards.methodology import (  # noqa: E402 -- needs sys.path set above
    METHODOLOGY_DOC,
    METRIC_METHODOLOGY,
    CardError,
    methodology_key,
    normalize_quant_receipt_v1,
    require_documented,
)
from cogsyndelta.regions.aliases import canonical_region  # noqa: E402

DEFAULT_OWNER = "tzervas"
DEFAULT_BASE = "cogsyndelta"
DEFAULT_REGIONS_CONFIG = REPO_ROOT / "config" / "mind" / "csd-regions.json"

# Mirrors the suffix table in scripts/csd-hf-repos.py exactly, for the three regions
# whose repo name does not follow the plain "-region-<name>" pattern. Every other
# region (reason, classify_banking77, classify_go_emotions, ...) gets the plain
# pattern -- those repos are not in csd-hf-repos.py's REPOS list yet, so this is an
# extension of its convention, not a transcription of it; create_repo(exist_ok=True)
# provisions them the first time this script runs against one.
#
# Keyed CANONICALLY -- `default_repo()` resolves its input through `canonical_region()`
# before this lookup, so a legacy spelling (`code`, `vl_latent`) needs no entry of its
# own: it resolves to its canonical id first. `visual`'s suffix (`-vl-jepa`) was never
# derived from the region id at all, so it stays a table entry; `language`'s Hub repo
# was renamed 2026-09-05 (tzervas/cogsyndelta-region-code ->
# tzervas/cogsyndelta-region-language, via `move_repo`, old id redirects), so the plain
# "-region-<name>" pattern now resolves it correctly and it needs no entry either.
REGION_REPO_SUFFIX = {
    "residual_mlp": "-region-residual",
    "stream_vae": "-region-stream-vae",
    "visual": "-vl-jepa",
}

# Source: docs/design/LICENCE-FOR-OPEN-WEIGHTS.md, "Decision 2026-09-02" and the
# per-region table immediately above it (search "Updated per-region table"). The
# composed-mind entries are recorded here for completeness (the strictest-input rule
# in the csd-release-licence-decision memory) but this CLI only ever resolves a
# per-region tier -- a composed-model publish is not this script's job.
#
# residual_mlp and stream_vae are deliberately absent: no licence audit exists for
# either, so `licence_tier()` raises for them rather than defaulting to MIT.
#
# Keyed CANONICALLY (`language`, not `code`; `visual`, not `vl_latent`) -- unlike
# REGION_REPO_SUFFIX above, this table is an internal lookup with no Hub-visible name
# baked into its keys, so there is nothing gained by keeping it legacy-spelled. Every
# reader (`licence_tier()`, `LICENCE_WHY.get()` in build_card) resolves its input
# through `canonical_region()` first, so a caller still spelling either region the old
# way keeps working.
LICENCE_TIER: dict[str, str] = {
    "language": "mit",
    "classify_banking77": "mit",
    "classify_go_emotions": "mit",
    "reason": "mit",
    "visual": "mit",
    "compress": "cc-by-sa-4.0",
    "retrieve": "cc-by-nc-sa-4.0",
    "memory": "cc-by-nc-sa-4.0",
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
# than "unknown". `visual` stays in this set as the DEFAULT: licence_tier() only takes
# it out when the training receipt names the admitted Mix B corpus (see
# `admitted_visual_identity` / `_visual_receipt_is_admitted`). A tiny-imagenet,
# unset, or fingerprint-mismatched receipt is still BLOCKING. Canonical key;
# licence_tier() canonicalizes its input before this check.
BLOCKING_REGIONS: frozenset[str] = frozenset({"visual"})

ADMITTED_VISUAL_MANIFEST = REPO_ROOT / "config" / "mind" / "visual-clean-v1.json"
TINY_IMAGENET_MARKERS = ("tiny-imagenet", "zh-plus/tiny-imagenet")

CLEVR_TASL = (
    "- **CLEVR** — Johnson et al. / Facebook, Inc. (c) 2017. "
    "Source: https://cs.stanford.edu/people/jcjohns/clevr/ "
    "Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). "
    "Images were resized to 128×128 PNG for this corpus (modified)."
)

LICENCE_WHY: dict[str, str] = {
    "language": "no NC or share-alike input in the catalogue, once GitHub-licence-filtered",
    "classify_banking77": "no NC or share-alike input in the catalogue",
    "classify_go_emotions": "no NC or share-alike input in the catalogue",
    "reason": "no NC or share-alike input in the catalogue",
    "visual": "Mix B includes CLEVR (CC BY 4.0, ATTRIBUTION); MIT-usable with the "
    "mandatory TASL notice (LICENCE-FOR-OPEN-WEIGHTS.md :98-102). tiny-imagenet "
    "remains BLOCKING and is not this corpus",
    "compress": "SNLI (+ government + fiction) repaired corpus is share-alike; "
    "no NC-tagged input identified",
    "retrieve": "GooAQ (NC, accepted 2026-09-02) and Natural Questions / FiQA "
    "(CC BY-SA) are both present in the corpus as trained",
    "memory": "region MERGE of compress (CC BY-SA 4.0) and retrieve (CC BY-NC-SA "
    "4.0) per Rider 1 (LICENCE-FOR-OPEN-WEIGHTS.md, Decision 2026-09-02): a merge "
    "inherits the most restrictive parent licence, and memory also trains directly "
    "on retrieve's GooAQ/NQ/FiQA corpus",
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
    """The repo a region's checkpoints publish to. Canonicalizes `region` first (the
    same `canonical_region` alias layer `licence_tier` uses), so a legacy spelling and
    its canonical replacement always agree: `default_repo("code") == default_repo(
    "language") == "tzervas/cogsyndelta-region-language"` (Hub repo renamed 2026-09-05
    via `move_repo`; the old id redirects). Likewise `vl_latent`/`visual` both resolve
    to `tzervas/cogsyndelta-vl-jepa`.
    """
    region = canonical_region(region)
    suffix = REGION_REPO_SUFFIX.get(region, f"-region-{region.replace('_', '-')}")
    return f"{owner}/{base}{suffix}"


def admitted_visual_identity(manifest_path: Path | None = None) -> tuple[str, str]:
    """`(corpus_source, fingerprint)` pinned in `config/mind/visual-clean-v1.json`."""
    path = manifest_path if manifest_path is not None else ADMITTED_VISUAL_MANIFEST
    data = json.loads(path.read_text(encoding="utf-8"))
    source = data.get("id")
    fingerprint = data.get("fingerprint")
    if not source or not fingerprint:
        raise PublishAbortError(
            f"{path}: admitted visual manifest missing id or fingerprint -- refusing "
            "rather than treating an incomplete pin as Mix B"
        )
    return str(source), str(fingerprint)


def _visual_receipt_source_and_fingerprint(
    receipt: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    if not isinstance(receipt, dict):
        return None, None
    corpus = receipt.get("corpus")
    corpus_d = corpus if isinstance(corpus, dict) else {}
    raw_source = (
        corpus_d.get("corpus_source") or corpus_d.get("source") or receipt.get("corpus_source")
    )
    raw_fp = corpus_d.get("fingerprint")
    source = str(raw_source) if raw_source else None
    fingerprint = str(raw_fp) if raw_fp else None
    return source, fingerprint


def _visual_receipt_is_admitted(receipt: dict[str, Any] | None) -> bool:
    """True only when the receipt names Mix B id AND the pinned fingerprint."""
    source, fingerprint = _visual_receipt_source_and_fingerprint(receipt)
    if source is None or fingerprint is None:
        return False
    if any(marker in source for marker in TINY_IMAGENET_MARKERS):
        return False
    admitted_source, admitted_fp = admitted_visual_identity()
    return source == admitted_source and fingerprint == admitted_fp


def visual_attribution_block(receipt: dict[str, Any] | None) -> list[str]:
    """CLEVR TASL + the ATTRIBUTION notice LICENCE-FOR-OPEN-WEIGHTS.md :98-102 requires.

    Raises:
        PublishAbortError: Mix B is admitted but `CLEVR_TASL` no longer names CLEVR
            and CC BY 4.0 -- the card would then be MIT without the mandatory notice.
    """
    if not _visual_receipt_is_admitted(receipt):
        return []
    if "CLEVR" not in CLEVR_TASL or "creativecommons.org/licenses/by/4.0" not in CLEVR_TASL:
        raise PublishAbortError(
            "Mix B visual card requires the CLEVR CC BY 4.0 TASL block "
            "(LICENCE-FOR-OPEN-WEIGHTS.md :98-102 ATTRIBUTION notice)"
        )
    return [
        "## Training data attribution",
        "",
        "This model was trained on Mix B (`visual-clean-v1`). ATTRIBUTION corpora "
        "are MIT-usable if the specific notice is carried "
        "(`docs/design/LICENCE-FOR-OPEN-WEIGHTS.md` :98-102). CLEVR is CC BY 4.0; "
        "TASL (Title, Author, Source, Licence) follows.",
        "",
        "### CC BY 4.0",
        CLEVR_TASL,
        "",
    ]


def licence_tier(region: str, receipt: dict[str, Any] | None = None) -> str:
    """Return the region's standalone licence tag, or abort if BLOCKING/unknown.

    Mix B visual is MIT (LICENCE-FOR-OPEN-WEIGHTS.md :98-102, :1592, :1685-1691,
    :1867): ATTRIBUTION training data is MIT-usable when the card carries the
    specific notice. The TASL block is `visual_attribution_block`, not this tag.
    """
    region = canonical_region(region)
    if region in BLOCKING_REGIONS:
        if region == "visual" and _visual_receipt_is_admitted(receipt):
            return LICENCE_TIER["visual"]
        raise PublishAbortError(
            f"region {region!r} is BLOCKING per docs/design/LICENCE-FOR-OPEN-WEIGHTS.md "
            "section 4 -- unreleasable as trained (no licence grant anywhere in the "
            "provenance chain, or visual receipt is not the admitted Mix B pin). "
            "Refusing regardless of any licence tier value on record for it; BLOCKING "
            "is strictly worse than unknown."
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
    """Parse `path` as a receipt -- and require the result to actually be a JSON
    *object*.

    `json.loads` happily returns `None` for the four bytes `'null'`, or a `list` for
    `'[1,2,3]'`, or a bare `str`/`int`/etc for other scalars -- every one of those is
    valid JSON and none of them is a receipt. Left unchecked, a `null` receipt used to
    read as "no receipt supplied" everywhere a caller wrote `if eval_receipt is not
    None`, while the *path* to that same file stayed truthy wherever a caller instead
    checked `if eval_receipt_path:` -- so the two conditions disagreed for exactly a
    `null` receipt, and a receipt containing the four bytes `null` sailed past both
    `assert_region_matches` and `assert_receipt_bound_to_checkpoint` and reached
    `upload_file` anyway. A list or scalar receipt hit the same gap one step earlier,
    as an uncaught `AttributeError` from `.get()` on a non-dict. One check here closes
    the whole class for every caller at once: a receipt that isn't a JSON object
    aborts before any field is ever read out of it.
    """
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as e:
        raise PublishAbortError(f"{path}: not found") from e
    except json.JSONDecodeError as e:
        raise PublishAbortError(f"{path}: not valid JSON ({e})") from e
    if not isinstance(data, dict):
        raise PublishAbortError(f"{path}: receipt must be a JSON object, got {type(data).__name__}")
    return data


# The v1 -> v2 quant-receipt alias table and `normalize_quant_receipt_v1` (imported
# above) now live in `cogsyndelta.cards.methodology` -- see that module's docstring for
# why this script does not instead import `cogsyndelta.pipeline.receipt`'s copy of the
# same table, and why the reasoning that used to keep this table local to this script
# (avoiding `cogsyndelta.regions`'s import cost) does not apply to a plain, dependency-
# free module like `cogsyndelta.cards.methodology`.


def load_region_config(region: str, regions_path: Path = DEFAULT_REGIONS_CONFIG) -> dict[str, Any]:
    """Find `region`'s entry in the catalogue, matching on canonical name so a caller
    still spelling it the legacy way (`code`, `vl_latent`) finds the same entry the
    catalogue now stores under its canonical name (`language`, `visual`)."""
    target = canonical_region(region)
    data = load_json(regions_path)
    for r in data.get("regions", []):
        name = r.get("name")
        if isinstance(name, str) and canonical_region(name) == target:
            return r
    raise PublishAbortError(f"region {region!r} not found in {regions_path}")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _contained_path(raw_value: str, *, what: str) -> Path:
    """Resolve a path string and require it to be contained.

    Used by `checkpoint_path_from_receipt`, which reads a path out of a receipt that
    anyone able to write (or misdirect an operator/agent into pointing `--receipt`
    at) controls, AND by `quantized_artifact_path`, on the derived `<stem>.ptq.pt`
    location -- THE REAL RULE, not the premise an earlier version of this docstring
    asserted: deriving that path from an already-contained checkpoint narrows *which*
    path gets read, but narrowing which path is not the same as that path being safe
    to read. Nothing about derivation stops the on-disk entry at the derived location
    from being a symlink to anywhere the process can read (see the reviewer-found
    bypass, N1, in `quantized_artifact_path`'s own docstring) -- so containment is
    checked there too, on the resolved target, exactly as it is checked here for the
    checkpoint. Without containment this becomes arbitrary-file upload: whatever the
    path names gets hashed, uploaded, and its sha256 published in the model card. So:
    resolve symlinks/`..` first, then require the resolved path to sit under an
    allow-listed root AND carry an allow-listed suffix, and abort before touching the
    filesystem again (no hashing, no upload) if either check fails.
    """
    raw = Path(raw_value)
    resolved = raw.resolve()

    if resolved.suffix not in ALLOWED_CHECKPOINT_SUFFIXES:
        raise PublishAbortError(
            f"receipt {what} {raw} has suffix {resolved.suffix!r}, not one of "
            f"{sorted(ALLOWED_CHECKPOINT_SUFFIXES)} -- refusing to treat an arbitrary "
            f"receipt-named file as a {what}"
        )

    allowed_roots = [r.resolve() for r in ALLOWED_CHECKPOINT_ROOTS]
    if not any(_is_relative_to(resolved, root) for root in allowed_roots):
        raise PublishAbortError(
            f"receipt {what} {raw} resolves to {resolved}, outside the allow-listed "
            f"checkpoint roots {[str(r) for r in ALLOWED_CHECKPOINT_ROOTS]} -- refusing "
            "to upload a file a receipt points at outside those roots"
        )
    return resolved


def checkpoint_path_from_receipt(receipt: dict[str, Any]) -> Path:
    """The checkpoint a receipt names -- resolved and contained, never trusted verbatim.

    See `_contained_path` for the containment rules this enforces.
    """
    ckpt = receipt.get("checkpoint") or receipt.get("artifacts", {}).get("checkpoint")
    if not ckpt:
        raise PublishAbortError(
            "training receipt has no 'checkpoint' (or artifacts.checkpoint) path"
        )
    return _contained_path(ckpt, what="checkpoint")


def quantized_artifact_path(checkpoint: Path, receipt: dict[str, Any]) -> Path:
    """Where this checkpoint's packed artifact must be -- DERIVED from the verified
    checkpoint path, with the receipt's own claim used only to confirm agreement, and
    the derived location itself put through the SAME containment the checkpoint gets.

    `scripts/csd-quantize.py` writes the artifact at
    `checkpoint.with_name(f"{checkpoint.stem}.ptq.pt")` and nowhere else, so that
    expression -- applied to the path this script has already resolved, contained and
    hashed -- names which file is this region's quantized artifact.
    `artifacts.quantized_path` is then a cross-check: if the receipt names a
    different file, the receipt and this checkpoint are not describing the same
    quantization run and the publish stops.

    Deriving rather than reading closes the substitution hole this function was
    written for: containment alone passes for any file under an allow-listed root,
    including a decoy the attacker put there, and the sha256 check passes trivially
    when the same document supplies both the path and the expected hash. Derivation
    makes the set of files this script will even *consider* "the quantized artifact"
    a function of the checkpoint rather than of the receipt.

    THE RULE, NOT THE PREMISE (N1). An earlier version of this function treated
    derivation as sufficient on its own -- as if narrowing which path gets read also
    made that path safe to read. It does not: nothing stopped the on-disk entry at
    `<checkpoint-stem>.ptq.pt` from being a symlink to any file this process can
    read, anywhere on the filesystem. `Path.resolve()` follows a symlink silently, so
    the OLD code's `checkpoint.with_name(...).resolve()` happily resolved straight
    through a planted symlink to a file outside every allow-listed root -- and a
    receipt whose `artifacts.quantized_path` simply named that same foreign file
    agreed with the (already-compromised) derivation, so the cross-check above passed
    too. Two things fix that, applied to the UNRESOLVED derived path before anything
    follows it anywhere:

    1. The on-disk entry must not itself be a symlink at all (`Path.is_symlink()`,
       checked strictly before any `.resolve()` call -- resolving is exactly the
       operation that would silently follow one). A hard link or a plain regular
       file at this location is unaffected: neither is a symlink, and containment
       still applies to both via step 2.
    2. Once confirmed not to be a symlink, the path is run through `_contained_path`
       -- the identical allow-listed-root-and-suffix check the checkpoint itself
       gets -- and its resolved parent directory must equal the checkpoint's own
       resolved parent directory, so a non-symlink path that somehow named a
       sibling-of-a-symlink or a different region's directory is caught too.

    Returns:
        The resolved artifact path (existence is checked by `verify_quantized_sha`).
    """
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    declared = artifacts.get("quantized_path")
    if not declared:
        raise PublishAbortError(
            "quant receipt has no artifacts.quantized_path -- refusing: a quant "
            "receipt without a persisted packed artifact describes nothing this tool "
            "can upload alongside the fp32 checkpoint. Regenerate it with "
            "scripts/csd-quantize.py (which now writes this field)."
        )

    unresolved = checkpoint.with_name(f"{checkpoint.stem}.ptq.pt")
    if unresolved.is_symlink():
        raise PublishAbortError(
            f"the derived quantized artifact path {unresolved} is a symlink -- "
            "refusing: a symlink at this location can be made to resolve to any "
            "file this process can read, anywhere on the filesystem, regardless of "
            "what allow-listed root the symlink itself sits under. The quantized "
            "artifact must be a regular file (or hard link) physically present "
            "beside the checkpoint, never a link."
        )
    derived = _contained_path(str(unresolved), what="quantized artifact")

    checkpoint_dir = checkpoint.resolve().parent
    if derived.parent != checkpoint_dir:
        raise PublishAbortError(
            f"the derived quantized artifact path {derived} does not sit in the "
            f"checkpoint's own directory ({checkpoint_dir}) -- refusing: the "
            "quantized artifact must live beside the checkpoint it was produced "
            "from, never in another region's directory"
        )

    if derived == checkpoint.resolve():
        # Unreachable while the suffix table forbids a checkpoint already named
        # `*.ptq.pt`; asserted anyway because the one thing that must never happen is
        # the artifact and the checkpoint resolving to the same file, and a future
        # edit to either the suffix list or this derivation could make it happen
        # quietly.
        raise PublishAbortError(
            f"the derived quantized artifact path {derived} is the checkpoint itself "
            "-- refusing: the quantized artifact is never the primary weights file"
        )
    if Path(str(declared)).resolve() != derived:
        raise PublishAbortError(
            f"quant receipt's artifacts.quantized_path {declared!r} resolves to "
            f"{Path(str(declared)).resolve()}, but this checkpoint's artifact is "
            f"{derived} -- refusing. The artifact is derived from the verified "
            f"checkpoint {checkpoint}, exactly as scripts/csd-quantize.py derives it; "
            "a receipt naming a different file is either stale (measured against "
            "another checkpoint) or an attempt to have some other file published as "
            "this region's quantized weights."
        )
    return derived


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
    --region claims. One comparison per receipt closes the class.

    Compared CANONICALLY: an old receipt naming the legacy spelling (`region: "code"`)
    still matches `--region language`, and vice versa -- the rename does not turn every
    receipt written before it into a forced mismatch.
    """
    claimed = receipt_region(receipt, label)
    if canonical_region(claimed) != canonical_region(region):
        raise PublishAbortError(
            f"{label} receipt's region {claimed!r} does not match --region {region!r} -- "
            "refusing: the licence tier and repo name are derived from --region, and a "
            "mismatch would publish this receipt's data under a different licence than "
            "the one it was actually trained/evaluated under"
        )


def verify_checkpoint_sha(checkpoint: Path) -> str:
    """Existence-check the checkpoint file and return its computed sha256.

    This is the ground truth every receipt's own declared checkpoint_sha256 is
    checked against by `assert_receipt_bound_to_checkpoint()` below -- it is computed
    from the file itself, never taken from (or reconciled with) any single receipt,
    so that a training receipt cannot certify its own binding.
    """
    if not checkpoint.is_file():
        raise PublishAbortError(f"checkpoint not found: {checkpoint}")
    return sha256_of(checkpoint)


# Key precedence for "when was this receipt recorded", by receipt shape observed on
# disk: training receipts use 'recorded', quant receipts 'recorded_utc', eval receipts
# (schema model-pipeline-receipt/v1) 'started_utc'. Tried in this order; the first key
# present wins.
_RECEIPT_TIMESTAMP_KEYS: tuple[str, ...] = ("recorded", "recorded_utc", "started_utc", "timestamp")


def receipt_checkpoint_sha256(receipt: dict[str, Any], label: str) -> str:
    """The checkpoint sha256 a receipt itself declares -- required, not merely
    consulted if present.

    Looked up at `artifacts.checkpoint_sha256` (the shape the fingerprint branch
    writes into training receipts, and the natural extension for any receipt schema
    that already nests `artifacts.checkpoint`, e.g. eval) or, for a receipt shape with
    no `artifacts` wrapper at all (today's quant receipts), a top-level
    `checkpoint_sha256`. Absence aborts: none of the 16 real receipts on disk carry
    either field today, and that is the exact gap this function exists to close --
    see the module docstring's "RECEIPTS ARE BOUND TO THE CHECKPOINT" section.
    """
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    val = artifacts.get("checkpoint_sha256") or receipt.get("checkpoint_sha256")
    if not val:
        raise PublishAbortError(
            f"{label} receipt has no artifacts.checkpoint_sha256 (or top-level "
            "checkpoint_sha256) -- refusing: every receipt (training, eval, quant) "
            "must name the sha256 of the exact checkpoint it measured before it can "
            "be published alongside that checkpoint. A receipt written before this "
            "field existed may describe weights since overwritten at the same "
            "mutable <region>-checkpoints/final.pt path -- it is unpublishable by "
            "construction, and that is correct, not a defect to route around."
        )
    return str(val)


def receipt_quantized_sha256(receipt: dict[str, Any]) -> str:
    """The quantized artifact's sha256 a quant receipt itself declares -- required,
    not merely consulted if present.

    Looked up ONLY at `artifacts.quantized_sha256` (the field
    `cogsyndelta.quant.ptq.save_packed_artifact` writes into the quant receipt) --
    unlike `receipt_checkpoint_sha256` there is no top-level fallback, because no
    quant receipt ever wrote this field anywhere else. Absence aborts: a quant
    receipt with no declared sha256 for its own packed artifact names nothing this
    tool can verify before upload, and publishing an unverified file next to the
    fp32 checkpoint under one card would be worse than not publishing it at all.
    """
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    val = artifacts.get("quantized_sha256")
    if not val:
        raise PublishAbortError(
            "quant receipt has no artifacts.quantized_sha256 -- refusing: the "
            "quantized artifact this receipt names cannot be verified before "
            "upload. Regenerate the quant receipt with scripts/csd-quantize.py "
            "(which now writes this field)."
        )
    return str(val)


def verify_quantized_sha(quantized: Path, receipt: dict[str, Any]) -> str:
    """Existence-check the quantized artifact and require its real sha256 to equal
    what the quant receipt itself declares, before any upload.

    Mirrors `verify_checkpoint_sha` + the sha half of
    `assert_receipt_bound_to_checkpoint`, but for the packed artifact rather than
    the fp32 checkpoint: computed from the file itself (never trusted from the
    receipt), and checked against the receipt's own claim rather than merely
    returned, because unlike the checkpoint there is no separate binding pass that
    would otherwise catch a mismatch here.
    """
    if not quantized.is_file():
        raise PublishAbortError(f"quantized artifact not found: {quantized}")
    actual = sha256_of(quantized)
    declared = receipt_quantized_sha256(receipt)
    if actual != declared:
        raise PublishAbortError(
            f"quant receipt's artifacts.quantized_sha256 {declared!r} does not "
            f"match the quantized artifact's actual sha256 {actual!r} -- refusing: "
            f"{quantized} is not the exact bytes this receipt measured"
        )
    return actual


def _ptq() -> Any:
    """`cogsyndelta.quant.ptq`, imported lazily and without requiring an install.

    Lazy for the same reason `publish()`'s `huggingface_hub` import is: importing
    torch costs seconds, and every code path that never reaches a quant receipt --
    `--help`, a licence refusal, an fp32-only publish -- must not pay it. `src` is
    appended (not prepended) to `sys.path` as a fallback for running this script
    straight out of a checkout, so an installed `cogsyndelta` still wins.
    """
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.append(src)
    from cogsyndelta.quant import ptq

    return ptq


def _export_safetensors() -> Any:
    """`cogsyndelta.cards.export.export_safetensors`, imported lazily -- same reasoning
    as `_ptq()` above: it pulls in torch, and only the `--safetensors` path (and a
    future publish once this becomes non-opt-in) should pay that cost."""
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.append(src)
    from cogsyndelta.cards.export import export_safetensors

    return export_safetensors


def verify_quantized_measurements(
    quantized: Path, receipt: dict[str, Any]
) -> tuple[int, dict[str, int]]:
    """Recompute the artifact's stored size and width histogram FROM THE ARTIFACT, and
    require the quant receipt to agree.

    The sha256 check above binds the bytes; this binds the numbers printed beside
    them. They are different claims: an artifact can hash exactly as its receipt says
    while that receipt's `stored_bytes` and `width_histogram` were measured on
    something else entirely, and the card -- which is what a reader of the published
    repo actually sees -- would carry the wrong compression story under a correct
    hash. Transcribing a receipt number into a card asserts nothing; recomputing it
    and refusing to publish on a mismatch does.

    Both numbers come from `cogsyndelta.quant.ptq`, which is also where
    `apply_plan` produces the figure the receipt recorded, so the two are comparable
    by construction: packed codes plus per-channel scale and zero-point for every
    quantized tensor, four bytes an element for everything kept in fp32.

    Returns:
        The measured (stored_bytes, width_histogram) for the card to print.
    """
    ptq = _ptq()
    try:
        packed = ptq.load_packed_artifact(quantized)
    except Exception as exc:
        raise PublishAbortError(
            f"quantized artifact {quantized} is not a readable packed artifact "
            f"({type(exc).__name__}: {exc}) -- refusing to publish a file this tool "
            "cannot itself load and measure"
        ) from exc

    measured_bytes = ptq.packed_stored_bytes(packed)
    measured_hist = ptq.packed_width_histogram(packed)

    declared_bytes = receipt.get("stored_bytes")
    if declared_bytes is None:
        raise PublishAbortError(
            "quant receipt has no stored_bytes -- refusing: the card would print a "
            "compression claim with nothing to check it against"
        )
    if int(declared_bytes) != measured_bytes:
        raise PublishAbortError(
            f"quant receipt's stored_bytes {declared_bytes!r} does not match the "
            f"{measured_bytes} bytes actually stored in {quantized.name} -- refusing: "
            "the receipt's compression numbers were not measured on this artifact"
        )

    declared_hist = receipt.get("width_histogram")
    if declared_hist is None:
        raise PublishAbortError(
            "quant receipt has no width_histogram -- refusing: the card would print a "
            "bit-width breakdown with nothing to check it against"
        )
    normalised = {str(k): int(v) for k, v in dict(declared_hist).items()}
    if normalised != measured_hist:
        raise PublishAbortError(
            f"quant receipt's width_histogram {normalised} does not match the widths "
            f"actually stored in {quantized.name} ({measured_hist}) -- refusing: the "
            "receipt's bit assignment was not measured on this artifact"
        )
    return measured_bytes, measured_hist


def receipt_timestamp(receipt: dict[str, Any], label: str) -> datetime:
    """When a receipt says it was recorded -- required, not merely consulted if
    present. See `_RECEIPT_TIMESTAMP_KEYS` for the key precedence."""
    for key in _RECEIPT_TIMESTAMP_KEYS:
        raw = receipt.get(key)
        if not raw:
            continue
        try:
            ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError as e:
            raise PublishAbortError(
                f"{label} receipt's {key!r} value {raw!r} is not a parseable "
                "ISO-8601 timestamp -- refusing: cannot verify it postdates the "
                "checkpoint it claims to describe"
            ) from e
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)
    raise PublishAbortError(
        f"{label} receipt has none of {_RECEIPT_TIMESTAMP_KEYS} -- refusing: cannot "
        "verify it was recorded after the checkpoint file it claims to describe"
    )


def assert_receipt_bound_to_checkpoint(
    receipt: dict[str, Any],
    checkpoint_sha256: str,
    checkpoint_mtime: datetime,
    label: str,
) -> None:
    """Bind a receipt to the exact checkpoint bytes being published, not merely to
    the mutable path both happen to name (region-matching alone does not do this --
    see the module docstring). Two independent, fail-closed checks; either aborts
    before any upload, naming the receipt that failed:

    (a) the receipt's own declared checkpoint_sha256 must equal the sha256 this tool
        just computed of the checkpoint file -- absence or mismatch aborts.
    (b) belt-and-braces for a receipt that predates this field: the receipt's own
        recorded timestamp must not be older than the checkpoint file's mtime -- a
        receipt cannot describe a checkpoint that did not yet exist when it was
        recorded. The mtime is floored to whole seconds before this comparison (a
        one-second grace period), since every receipt producer stamps recorded_at
        with whole-second resolution (time.strftime("%Y-%m-%dT%H:%M:%SZ")) while the
        checkpoint file's mtime carries sub-second precision -- without the floor, a
        receipt genuinely written in the same second as the checkpoint's torch.save
        would be spuriously rejected.
    """
    recorded_sha = receipt_checkpoint_sha256(receipt, label)
    if recorded_sha != checkpoint_sha256:
        raise PublishAbortError(
            f"{label} receipt's checkpoint_sha256 {recorded_sha!r} does not match "
            f"the checkpoint's actual sha256 {checkpoint_sha256!r} -- refusing: this "
            f"{label} receipt was measured on different weights than the checkpoint "
            "being published now"
        )
    recorded_ts = receipt_timestamp(receipt, label)
    if recorded_ts < checkpoint_mtime:
        raise PublishAbortError(
            f"{label} receipt is timestamped {recorded_ts.isoformat()}, before the "
            f"checkpoint file's mtime {checkpoint_mtime.isoformat()} (floored to "
            "whole seconds, a one-second grace period) -- refusing: this "
            f"{label} receipt predates the checkpoint it claims to describe "
            "(a legacy receipt for a superseded final.pt)"
        )


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


#  ------------------------------------------------------------- metrics methodology
#
# `METHODOLOGY_DOC`, `MetricMethodology` and `METRIC_METHODOLOGY` now live in
# `cogsyndelta.cards.methodology` (imported above) -- see that module's docstring
# for why. `_methodology_section` below is refused-closed exactly as before: a
# metric key the card prints with no entry in `METRIC_METHODOLOGY` aborts the
# publish.


def _card_metric_keys(
    train_receipt: dict[str, Any],
    eval_receipt: dict[str, Any] | None,
    quant_receipt: dict[str, Any] | None,
) -> list[str]:
    """Every metric-table row key `build_card` actually prints, de-duplicated, in
    encounter order -- the exact set `_methodology_section` must cover. Mirrors
    `build_card`'s own `_dict_table` calls one-for-one; keep the two in sync if either
    changes which sections print a table (a test pins this -- see
    `tests/test_metrics_methodology.py`).

    ASSUMES `quant_receipt` (when not `None`) has already been passed through
    `normalize_quant_receipt_v1` -- every production caller (`build_plan`, which loads
    it via `load_json` and normalises it before ever calling `build_card` /
    `_methodology_section` / this function) does. A caller that hands this a raw,
    un-normalised v1-shaped quant receipt directly will find only its unrenamed keys
    (`fp32_metric_recomputed`, `tolerance`, `within_budget`, `fp32_bytes`,
    `stored_bytes`) here, not `quant.plan_recall@1` / `quant.drop_recall@1` /
    `quant.compression_ratio` -- those three exist in `quant_receipt` only once
    normalisation has added them.
    """
    keys: dict[str, None] = {}
    for section in (
        train_receipt.get("held_out", {}),
        train_receipt.get("untrained_baseline", {}),
        # `beats_untrained_train` is the g7 §3.2 rename of this same gate/context dict
        # (see `beats_untrained_eval`'s own docstring above); read defensively, new
        # name first, without importing the module that writes it.
        train_receipt.get("beats_untrained_train") or train_receipt.get("beats_untrained", {}),
    ):
        for k in section:
            keys[k] = None
    if eval_receipt is not None:
        for k in eval_receipt.get("gates", {}):
            keys[k] = None
        for k in eval_receipt.get("metrics", {}):
            # `methodology_key` (cogsyndelta.cards.methodology) is the SAME repr./rank./
            # eff. prefix-strip `cogsyndelta.cards.tables` uses to key into
            # METRIC_METHODOLOGY -- this table only ever prints the repr.* family (never
            # rank./eff.*, which the newer cogsyndelta.cards library prints instead), so
            # only that prefix is selected here; `quant.artifact_recall@1` is passed
            # through unstripped (methodology_key leaves quant.* alone -- it is already
            # the canonical g7 §3.1 name, not a family-prefixed shorthand).
            if k.startswith("repr.") or k.startswith("quant."):
                keys[methodology_key(k)] = None
    for k in train_receipt.get("contamination", {}):
        if k != "examples":
            keys[k] = None
    if quant_receipt is not None:
        for k in (
            "fp32_metric_recomputed",
            "quant.plan_recall@1",
            "quant.drop_recall@1",
            "tolerance",
            "within_budget",
            "quant.compression_ratio",
            "fp32_bytes",
            "stored_bytes",
        ):
            if k in quant_receipt:
                keys[k] = None
    return list(keys)


def _metrics_schema_line(
    train_receipt: dict[str, Any],
    eval_receipt: dict[str, Any] | None,
    quant_receipt: dict[str, Any] | None,
) -> str:
    """The card's single `- **Metrics schema:**` line -- across every receipt
    `build_card` was given, not only the training one.

    `metrics_schema` is identity key #1 of MM §14's refuse predicate: a card that
    stamps `csd-metrics/v1` while rendering a table of `csd-metrics/v2` field names
    (`effective_rank_entropy`, `quant.plan_recall@1`, ...) right below it is the same
    class of provenance falsehood the refuse predicate exists to catch -- and the
    near-term real case for every already-trained matrix cell is exactly that: a
    training receipt from before this migration (no `metrics_schema` field at all,
    reads back as the `csd-metrics/v1 (not recorded)` fallback) re-benchmarked and
    re-quantized with this branch's v2 code (both stamp `csd-metrics/v2`). Refusing to
    publish that combination outright would block re-publishing every already-trained
    checkpoint until it is retrained from scratch, which is worse than the falsehood
    this fixes -- so when the receipts disagree, this renders each one's own stamp
    instead of picking one and hiding the disagreement.

    Args:
        train_receipt: The training receipt `build_card` was given.
        eval_receipt: The eval receipt, or `None`.
        quant_receipt: The quant receipt, or `None`.

    Returns:
        A single markdown bullet line: one stamp when every supplied receipt agrees,
        or a `train=... eval=... quant=...` breakdown (only the receipts actually
        supplied) when they do not.
    """
    labeled = [
        (name, receipt.get("metrics_schema", "csd-metrics/v1 (not recorded)"))
        for name, receipt in (
            ("train", train_receipt),
            ("eval", eval_receipt),
            ("quant", quant_receipt),
        )
        if receipt is not None
    ]
    schemas = {schema for _, schema in labeled}
    if len(schemas) == 1:
        return f"- **Metrics schema:** `{next(iter(schemas))}`"
    per_receipt = ", ".join(f"{name}=`{schema}`" for name, schema in labeled)
    return (
        "- **Metrics schema:** receipts disagree -- "
        f"{per_receipt}. The tables above merge metrics from more than one schema; "
        "read each metric's own row provenance above, not this line alone, before "
        "comparing numbers across them."
    )


def _methodology_section(
    train_receipt: dict[str, Any],
    eval_receipt: dict[str, Any] | None,
    quant_receipt: dict[str, Any] | None,
    rev: str,
    train_receipt_filename: str,
    eval_receipt_filename: str | None,
    quant_receipt_filename: str | None,
) -> str:
    """'How these numbers were produced': one line per metric key this card prints,
    naming its definition, battery and source file, plus the receipt-level provenance
    (corpus fingerprint, seed, code revision, receipt filenames) every number above was
    read from.

    Args:
        train_receipt: The training receipt `build_card` was given.
        eval_receipt: The eval receipt, or `None`.
        quant_receipt: The quant receipt, or `None`.
        rev: This checkpoint's code revision, exactly as `build_card`'s own Provenance
            section prints it.
        train_receipt_filename: Basename of the training receipt file, as uploaded.
        eval_receipt_filename: Basename of the eval receipt file, or `None`.
        quant_receipt_filename: Basename of the quant receipt file, or `None`.

    Returns:
        The section as markdown.

    Raises:
        PublishAbortError: A metric key this card prints has no entry in
            `METRIC_METHODOLOGY` -- refusing to publish a number with no stated formula
            rather than silently shipping one undocumented. This is the enforcement
            `tests/test_metrics_methodology.py` proves is load-bearing by stubbing
            `METRIC_METHODOLOGY` empty and asserting the card build then fails.
    """
    keys = _card_metric_keys(train_receipt, eval_receipt, quant_receipt)
    # Delegates the "is every key documented" check itself to
    # `cogsyndelta.cards.methodology.require_documented` -- the SAME missing-key
    # computation `cogsyndelta.cards.tables`'s build_* functions call, rather than a
    # second, hand-rolled `[k for k in keys if k not in METRIC_METHODOLOGY]` that could
    # silently drift from it. `methodology=METRIC_METHODOLOGY` passes THIS MODULE's own
    # bound name (imported above, not re-imported here), so a test's
    # `monkeypatch.setattr(mod, "METRIC_METHODOLOGY", {...})` -- which reassigns that
    # name in this module's namespace -- is exactly what this lookup sees; passing
    # `methodology=None` instead would read `cogsyndelta.cards.methodology`'s own
    # (unpatched) global, defeating that test's monkeypatch entirely (see
    # `cogsyndelta.cards.methodology`'s own module docstring for why this distinction
    # matters). `CardError` is translated to this script's own `PublishAbortError`
    # rather than propagated, so every caller of `build_card`/`build_plan` keeps seeing
    # one exception type for every refusal this script makes.
    try:
        require_documented(keys, methodology=METRIC_METHODOLOGY)
    except CardError as e:
        raise PublishAbortError(f"{e} (publishing, not rendering)") from e
    lines = [
        "## How these numbers were produced",
        "",
        f"Full definitions, formulas, `file:line` anchors and comparison rules for every "
        f"metric below: `{METHODOLOGY_DOC}` in this repository.",
        "",
        "| metric | definition | battery | battery_id | pooling | source |",
        "|---|---|---|---|---|---|",
    ]
    for key in sorted(keys):
        m = METRIC_METHODOLOGY[key]
        bid = f"`{m.battery_id}`" if m.battery_id else "_(n/a)_"
        pooling = f"`{m.pooling}`" if m.pooling else "_(n/a)_"
        lines.append(
            f"| `{key}` | {m.definition} | {m.battery} | {bid} | {pooling} | `{m.source}` |"
        )
    lines += [
        "",
        _metrics_schema_line(train_receipt, eval_receipt, quant_receipt),
        f"- **Corpus fingerprint:** `{train_receipt.get('corpus', {}).get('fingerprint', '(none recorded)')}`",
        f"- **Seed:** `{train_receipt.get('config', {}).get('seed', '(none recorded)')}`",
        f"- **Code revision:** `{rev}`",
        f"- **Training receipt:** `{train_receipt_filename}`",
    ]
    if eval_receipt_filename is not None:
        lines.append(f"- **Eval receipt:** `{eval_receipt_filename}`")
    if quant_receipt_filename is not None:
        lines.append(f"- **Quant receipt:** `{quant_receipt_filename}`")
    lines.append("")
    return "\n".join(lines)


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
    quantized_filename: str | None = None,
    quantized_sha256: str | None = None,
    quantized_file_bytes: int | None = None,
    quantized_stored_bytes: int | None = None,
    quantized_width_histogram: dict[str, int] | None = None,
    train_receipt_filename: str = "",
    eval_receipt_filename: str | None = None,
    quant_receipt_filename: str | None = None,
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
        f"**Licence tier.** `{tier}` -- "
        f"{LICENCE_WHY.get(canonical_region(region), '(reason not recorded)')}. "
        'See `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`, "Decision 2026-09-02".',
        "",
    ]
    parts += visual_attribution_block(train_receipt)
    parts += [
        "## Metrics",
        "",
        "### held_out",
        _dict_table(train_receipt.get("held_out", {})),
        "### untrained_baseline",
        _dict_table(train_receipt.get("untrained_baseline", {})),
        "### gates (training receipt: beats_untrained_train)",
        _dict_table(
            train_receipt.get("beats_untrained_train") or train_receipt.get("beats_untrained", {})
        ),
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
                        "quant.plan_recall@1",
                        "quant.drop_recall@1",
                        "tolerance",
                        "within_budget",
                        "quant.compression_ratio",
                        "fp32_bytes",
                        "stored_bytes",
                    )
                }
            ),
        ]
    else:
        parts.append("_No quant receipt supplied -- this checkpoint is fp32._\n")
    parts += [
        _methodology_section(
            train_receipt,
            eval_receipt,
            quant_receipt,
            rev,
            train_receipt_filename,
            eval_receipt_filename,
            quant_receipt_filename,
        ),
    ]
    if quant_receipt is not None and quantized_filename is not None:
        # Facts about the FILE only. The measured-vs-budget story (compression ratio,
        # metric drop, tolerance, within_budget, stored_bytes) lives in the
        # `### quantization` table above and is not repeated here: two copies of the
        # same number in one card is one copy too many to keep honest, and the table
        # is where a reader already goes for what the quantization cost.
        parts += [
            "## Quantized artifact",
            "",
            "A packed, sub-byte-quantized copy of this checkpoint's weights -- NOT the "
            "primary artifact; `final.pt` above (fp32) remains the required weights for "
            "this repo. See `cogsyndelta.quant.ptq` for the packing format; load it with "
            "`load_packed_artifact` + `unpack_state_dict`.",
            "",
            f"- **File:** `{quantized_filename}`",
            f"- **sha256:** `{quantized_sha256}`",
            f"- **File size:** {_fmt(quantized_file_bytes)} bytes",
            f"- **Stored tensor bytes (measured from this file):** {_fmt(quantized_stored_bytes)}",
            f"- **Width histogram, bits -> tensor count (measured from this file):** "
            f"{_fmt(quantized_width_histogram)}",
            "",
            "The last two were recomputed from the artifact at publish time and had to "
            "equal the quant receipt's `stored_bytes` and `width_histogram` -- which is "
            "what lets the `### quantization` table above be read as a claim about "
            "*this* file rather than a transcription from a document that merely names "
            "it.",
            "",
        ]
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
        _receipt_binding_note(eval_receipt is not None, quant_receipt is not None),
        "",
        "Published by `scripts/csd-publish-checkpoint.py`. Repo is private; the operator's "
        "publishing authorisation covers private repos only -- see that script's module "
        "docstring before ever adding a `--public` flag here.",
        "",
    ]
    return "\n".join(parts)


def _receipt_binding_note(has_eval: bool, has_quant: bool) -> str:
    receipts = ["training", *(["eval"] if has_eval else []), *(["quant"] if has_quant else [])]
    named = ", ".join(receipts)
    return (
        f"**Receipt binding.** Every receipt above ({named}) was verified, before "
        "publish, to declare *this exact checkpoint's* sha256 in "
        "`artifacts.checkpoint_sha256` (or a top-level `checkpoint_sha256`) and to be "
        "timestamped no earlier than the checkpoint file itself. A receipt that names "
        "a different checkpoint's measurements -- e.g. one recorded against an "
        "earlier `final.pt` before a later training run overwrote that same mutable "
        "path -- aborts the publish rather than being merged into this card under "
        "the current weights' sha256. If you expected a metric here and it is "
        "missing, the receipt that would have supplied it predates this checkpoint "
        "and needs to be regenerated against it."
    )


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
    bound_receipt_labels: tuple[str, ...]
    code_rev: str
    card: str
    files: dict[str, Path | bytes] = field(default_factory=dict)
    """path_in_repo -> local Path or in-memory bytes (the README)."""
    quantized_path: Path | None = None
    quantized_sha256: str | None = None
    quantized_stored_bytes: int | None = None
    quantized_width_histogram: dict[str, int] | None = None
    """Set only when a --quant-receipt was supplied, the artifact DERIVED from
    `checkpoint_path` (never a path the receipt supplied) verified against that
    receipt's own artifacts.quantized_sha256, its stored bytes and width histogram
    re-measured from the file and found to agree with the receipt, and the file added
    to `files` -- never the primary artifact; `checkpoint_path` above stays the one
    required weights file."""
    safetensors_path: Path | None = None
    safetensors_sha256: str | None = None
    """Set only when `--safetensors` was passed: `cogsyndelta.cards.export.
    export_safetensors(checkpoint_path)`'s own output, hashed the same way
    `checkpoint_path` is, and added to `files` under the checkpoint's basename with a
    `.safetensors` suffix -- never the primary artifact; `checkpoint_path` (fp32 `.pt`)
    stays the one required weights file, exactly as `quantized_path` above is never
    promoted over it."""


def build_plan(
    region: str,
    repo: str,
    train_receipt_path: Path,
    eval_receipt_path: Path | None,
    quant_receipt_path: Path | None,
    regions_config: Path = DEFAULT_REGIONS_CONFIG,
    want_safetensors: bool = False,
) -> Plan:
    # Visual's BLOCKING status depends on the training receipt's corpus pin, so the
    # receipt is loaded before licence_tier(). Other regions still fail closed on
    # region name alone; a visual receipt that is not Mix B stays BLOCKING.
    train_receipt = load_json(train_receipt_path)
    tier = licence_tier(region, receipt=train_receipt)
    region_cfg = load_region_config(region, regions_config)
    eval_receipt = load_json(eval_receipt_path) if eval_receipt_path else None
    quant_receipt = load_json(quant_receipt_path) if quant_receipt_path else None
    if quant_receipt is not None:
        quant_receipt = normalize_quant_receipt_v1(quant_receipt)

    # --region drives the licence tier and repo name; verify every receipt actually says
    # it's for this region before reading anything else out of them.
    assert_region_matches(train_receipt, region, "training")
    if eval_receipt is not None:
        assert_region_matches(eval_receipt, region, "eval")
    if quant_receipt is not None:
        assert_region_matches(quant_receipt, region, "quant")

    checkpoint = checkpoint_path_from_receipt(train_receipt)
    checkpoint_sha256 = verify_checkpoint_sha(checkpoint)
    # Floored to whole seconds: every receipt producer stamps recorded_at with
    # time.strftime("%Y-%m-%dT%H:%M:%SZ"), which has no sub-second resolution, while
    # the checkpoint file's mtime does -- comparing at nanosecond precision would
    # spuriously reject a legitimate receipt written in the same second as the
    # torch.save that produced the checkpoint. See assert_receipt_bound_to_checkpoint.
    checkpoint_mtime = datetime.fromtimestamp(checkpoint.stat().st_mtime, tz=UTC).replace(
        microsecond=0
    )

    # Region-matching (above) is not enough to say a receipt is *for* this checkpoint --
    # see the module docstring's "RECEIPTS ARE BOUND TO THE CHECKPOINT" section. Every
    # receipt actually supplied must additionally bind to these exact bytes, by sha256
    # and by time, or the publish aborts before any upload.
    bound_receipt_labels = ["training"]
    assert_receipt_bound_to_checkpoint(
        train_receipt, checkpoint_sha256, checkpoint_mtime, "training"
    )
    if eval_receipt is not None:
        assert_receipt_bound_to_checkpoint(
            eval_receipt, checkpoint_sha256, checkpoint_mtime, "eval"
        )
        bound_receipt_labels.append("eval")
    if quant_receipt is not None:
        assert_receipt_bound_to_checkpoint(
            quant_receipt, checkpoint_sha256, checkpoint_mtime, "quant"
        )
        bound_receipt_labels.append("quant")

    # The quant receipt binds to the fp32 checkpoint above (region + sha256 + time),
    # same as every other receipt -- but that says nothing about the SEPARATE quantized
    # artifact. Three separate things are established here, in order, before the file
    # can enter the upload plan: WHICH file it is (derived from the checkpoint, not
    # read from the receipt), that its bytes are the ones the receipt measured (sha256,
    # checked before anything unpickles it), and that the receipt's compression numbers
    # were measured on those bytes (re-measured from the file). Never promoted to the
    # primary artifact: `checkpoint.name` above stays required and unconditional
    # regardless of whether any of this succeeds.
    quantized_path: Path | None = None
    quantized_sha256: str | None = None
    quantized_stored_bytes: int | None = None
    quantized_width_histogram: dict[str, int] | None = None
    if quant_receipt is not None:
        quantized_path = quantized_artifact_path(checkpoint, quant_receipt)
        quantized_sha256 = verify_quantized_sha(quantized_path, quant_receipt)
        quantized_stored_bytes, quantized_width_histogram = verify_quantized_measurements(
            quantized_path, quant_receipt
        )

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
        quantized_filename=quantized_path.name if quantized_path is not None else None,
        quantized_sha256=quantized_sha256,
        quantized_file_bytes=quantized_path.stat().st_size if quantized_path else None,
        quantized_stored_bytes=quantized_stored_bytes,
        quantized_width_histogram=quantized_width_histogram,
        train_receipt_filename=train_receipt_path.name,
        eval_receipt_filename=eval_receipt_path.name if eval_receipt is not None else None,
        quant_receipt_filename=quant_receipt_path.name if quant_receipt is not None else None,
    )

    files: dict[str, Path | bytes] = {
        checkpoint.name: checkpoint,
        f"receipts/{train_receipt_path.name}": train_receipt_path,
        "README.md": card.encode("utf-8"),
    }
    # Deliberately gated on the same `is not None` condition already used above to
    # decide whether to run assert_region_matches / assert_receipt_bound_to_checkpoint
    # on this receipt -- not on the path's truthiness, which was the divergence a
    # `null` receipt exploited (see load_json's docstring). A receipt path can only
    # enter the upload plan once its parsed object has actually passed both checks.
    if eval_receipt is not None:
        files[f"receipts/{eval_receipt_path.name}"] = eval_receipt_path
    if quant_receipt is not None:
        files[f"receipts/{quant_receipt_path.name}"] = quant_receipt_path
    if quantized_path is not None:
        # The repo path is derived from the CHECKPOINT's basename, the same expression
        # `quantized_artifact_path` derives the local path from -- never a basename a
        # receipt supplied. Assigning `files[<receipt-supplied name>]` was the actual
        # substitution bug: a receipt naming another allow-listed `.pt` called
        # `final.pt` silently rebound `files["final.pt"]` away from the checkpoint, and
        # the card went on attesting the real checkpoint's sha256 for bytes that were
        # not it. The collision check below is belt-and-braces on the same property:
        # this name is a fresh key in the plan, or the plan is not built.
        name_in_repo = f"{checkpoint.stem}.ptq.pt"
        if name_in_repo in files:
            raise PublishAbortError(
                f"quantized artifact would be uploaded as {name_in_repo!r}, which the "
                f"plan already maps to {files[name_in_repo]!r} -- refusing to overwrite "
                "another file's slot in the upload plan"
            )
        files[name_in_repo] = quantized_path

    # OPT-IN (--safetensors), not part of the default plan: `export_safetensors`
    # requires the checkpoint to actually torch.load() as a state dict, which every
    # REAL checkpoint this project trains is -- but is not guaranteed of an arbitrary
    # `--receipt`-named `.pt` file (see checkpoint_path_from_receipt's own containment
    # docstring: the *path* is verified, never the pickled contents). A caller that
    # asked for this explicitly gets a fail-closed abort naming why, rather than a
    # publish that silently omits the file it was told to include.
    safetensors_path: Path | None = None
    safetensors_sha256: str | None = None
    if want_safetensors:
        export_safetensors_fn = _export_safetensors()
        try:
            safetensors_path = export_safetensors_fn(checkpoint)
        except Exception as e:
            raise PublishAbortError(
                f"--safetensors requested but {checkpoint} could not be exported to "
                f"safetensors ({type(e).__name__}: {e}) -- refusing to publish a plan "
                "that silently omits the file it was asked to include"
            ) from e
        safetensors_sha256 = sha256_of(safetensors_path)
        name_in_repo = f"{checkpoint.stem}.safetensors"
        if name_in_repo in files:
            raise PublishAbortError(
                f"safetensors export would be uploaded as {name_in_repo!r}, which the "
                f"plan already maps to {files[name_in_repo]!r} -- refusing to overwrite "
                "another file's slot in the upload plan"
            )
        files[name_in_repo] = safetensors_path

    return Plan(
        region=region,
        repo=repo,
        repo_type="model",
        tier=tier,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        bound_receipt_labels=tuple(bound_receipt_labels),
        code_rev=rev,
        card=card,
        files=files,
        quantized_path=quantized_path,
        quantized_sha256=quantized_sha256,
        quantized_stored_bytes=quantized_stored_bytes,
        quantized_width_histogram=quantized_width_histogram,
        safetensors_path=safetensors_path,
        safetensors_sha256=safetensors_sha256,
    )


def print_plan(plan: Plan, dry_run: bool) -> None:
    tag = "DRY-RUN" if dry_run else "APPLY"
    print(f"csd-publish-checkpoint [{tag}]")
    print(f"  region:      {plan.region}")
    print(f"  repo:        {plan.repo} ({plan.repo_type}, private)")
    print(f"  licence:     {plan.tier}")
    print(f"  checkpoint:  {plan.checkpoint_path}")
    bound = ", ".join(plan.bound_receipt_labels)
    print(f"               sha256={plan.checkpoint_sha256} (bound to: {bound})")
    if plan.quantized_path is not None:
        print(f"  quantized:   {plan.quantized_path} (not primary; {plan.checkpoint_path.name} is)")
        print(f"               sha256={plan.quantized_sha256}")
        print(
            f"               measured stored_bytes={plan.quantized_stored_bytes} "
            f"widths={plan.quantized_width_histogram}"
        )
    if plan.safetensors_path is not None:
        print(
            f"  safetensors: {plan.safetensors_path} (not primary; {plan.checkpoint_path.name} is)"
        )
        print(f"               sha256={plan.safetensors_sha256}")
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
    precomputed_sha256: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Upload only what has changed. Returns {'uploaded': [...], 'skipped': [...]}.

    `precomputed_sha256` supplies the already-verified hash for any other LFS-tracked
    file besides the checkpoint (e.g. the quantized artifact) -- so its content-match
    check compares against the exact bytes `build_plan` already hashed and verified,
    the same reasoning `checkpoint_sha256` follows for the checkpoint itself.
    """
    try:
        remote_list = api.get_paths_info(
            repo_id, list(files.keys()), expand=True, repo_type=repo_type
        )
    except Exception:  # network hiccup, or HF's own error shape -- treat as "unknown", upload
        remote_list = []
    remote = {getattr(r, "path", None): r for r in remote_list}
    extra_shas = precomputed_sha256 or {}

    uploaded: list[str] = []
    skipped: list[str] = []
    for path_in_repo, item in files.items():
        if path_in_repo == checkpoint_path_in_repo:
            precomputed = checkpoint_sha256
        else:
            precomputed = extra_shas.get(path_in_repo)
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
    extra_shas: dict[str, str] = {}
    if plan.quantized_path is not None and plan.quantized_sha256 is not None:
        extra_shas[plan.quantized_path.name] = plan.quantized_sha256
    if plan.safetensors_path is not None and plan.safetensors_sha256 is not None:
        extra_shas[plan.safetensors_path.name] = plan.safetensors_sha256
    result = sync_repo(
        api,
        plan.repo,
        plan.repo_type,
        plan.files,
        plan.checkpoint_path.name,
        plan.checkpoint_sha256,
        precomputed_sha256=extra_shas,
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
    ap.add_argument(
        "--safetensors",
        action="store_true",
        help="also export and upload final.safetensors (cogsyndelta.cards.export) "
        "beside the primary final.pt, so the Hub shows the safetensors badge/model-size "
        "sidebar. Opt-in: refuses (rather than silently skipping) if the checkpoint "
        "does not torch.load() as a plain state dict.",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo or default_repo(args.region)
    try:
        plan = build_plan(
            args.region,
            repo,
            args.receipt,
            args.eval_receipt,
            args.quant_receipt,
            want_safetensors=args.safetensors,
        )
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
