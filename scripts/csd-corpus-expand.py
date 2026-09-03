#!/usr/bin/env python3
"""Fetch training corpora onto the mechanical bulk array, license-gated and idempotent.

WHERE THIS WRITES AND WHY
/bulk is 5.95 TB of spinning RAID0 at 6.7% used. Corpora are large, they grow, and they
are read sequentially during training -- which is exactly what mechanical storage is good
at and exactly what should not sit on the homelab SSD the whole fleet shares. So datasets
land in /bulk/csd-corpus, and the SSD keeps only what a running job needs right now.

THE LICENSE GATE IS STRUCTURAL, NOT ADVISORY
Every entry must carry a license string that was actually observed on the dataset card or
upstream repository, and a verdict. `fetch` refuses any entry whose verdict is not
TRAIN_OK, and there is deliberately no flag to override that. This project has already
turned down MS MARCO -- whose official terms are non-commercial research only -- and ELI5,
whose license could not be established. Those decisions are worth exactly as much as the
mechanism that keeps them enforced.

A dataset card is not authoritative about its own upstream. Where the card and the source
disagree, the source wins and the entry is rejected.

IDEMPOTENT
A dataset already on disk with a manifest recording the same revision is skipped. Re-runs
are cheap and safe, which is the point: this is meant to be run by an unattended agent.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


# /bulk is local on gpu5080 and /mnt/bulk over NFS everywhere else, so the root is
# resolved rather than hardcoded -- a script that only runs on one host is not a
# fleet tool.
def _default_root() -> Path:
    for candidate in (Path("/bulk"), Path("/mnt/bulk")):
        if candidate.is_dir():
            return candidate / "csd-corpus"
    return Path("/bulk/csd-corpus")


BULK_CORPUS = _default_root()

TRAIN_OK = "TRAIN_OK"
REJECTED = "REJECTED"
REFUSE = "REFUSE"
"""A BLOCKING source (audio-licence-audit.md's vocabulary): no licence grant exists at all
(distributor disclaims ownership), an active dispute, a paywalled/membership corpus, or a
use-restriction the NC-tolerant policy was not written to absorb (ND, academic-only,
no-redistribution EULA). Distinct from REJECTED: REJECTED entries are skipped with a status
line; REFUSE entries make `fetch()` raise (`BlockingSourceError`) the instant they are
called, so refusal is structural rather than a branch that has to remember to fire. Kept in
the catalogue on purpose -- see REJECTED's own docstring reasoning above."""


class BlockingSourceError(PermissionError):
    """Raised by `fetch()` for any entry whose verdict is REFUSE.

    This is the mechanism, not the policy: the policy is that BLOCKING entries exist in
    CATALOGUE at all (a rejection nobody can see is a rejection that gets made again) with
    verdict=REFUSE recorded and why. `main()`'s per-entry loop catches this one exception
    type and turns it back into a normal status line so one BLOCKING entry does not abort a
    whole-catalogue run; calling `fetch()` on a REFUSE entry directly -- e.g. from a test,
    or from any future caller that does not go through that loop -- gets the raise with no
    opt-out. There is deliberately no flag that suppresses it.
    """


@dataclass
class Dataset:
    """One corpus, with the provenance needed to defend using it."""

    repo_id: str
    region: str
    license: str
    """The license string actually observed on the dataset card."""
    verdict: str
    why: str
    upstream: str = ""
    """What the ORIGINAL source says, when it is not the same repo as the card.

    A mirror's tag is not evidence about its upstream. Measured on this fleet:
    BeIR/scifact is tagged cc-by-sa-4.0 while allenai/scifact, the dataset it mirrors,
    is tagged cc-by-nc-2.0. Anything re-hosting someone else's corpus must record what
    the original says here, or it stays REJECTED.
    """
    config: str = ""
    splits: tuple[str, ...] = ("train",)
    caveat: str = ""
    columns: tuple[str, ...] = ()
    revision: str = ""
    """Pin a non-default git revision on the Hub.

    Exists for repos that still carry a legacy Python loading script: `datasets>=4`
    refuses to execute those ("Dataset scripts are no longer supported"), but HF's own
    conversion bot has already re-published many of them as Parquet under the
    `refs/convert/parquet` ref. Measured on this fleet: codeparrot/apps needs this;
    PolyAI/banking77 has no such ref (see `data_files`).
    """
    data_files: dict[str, str] = field(default_factory=dict)
    """Escape hatch keyed by split name, for a repo whose ONLY content is a retired
    loading script with no Parquet conversion available, where the script itself does
    nothing but download plain files from a fixed public URL. Set this to those URLs and
    `fetch` reads them through `builder` (a packaged loader shipped with `datasets`
    itself, e.g. "csv") instead of through `repo_id` -- the packaged loaders are not
    remote code and are unaffected by the script ban. Measured on this fleet:
    PolyAI/banking77's script does nothing but fetch two CSVs from
    github.com/PolyAI-LDN/task-specific-datasets.
    """
    builder: str = "csv"
    """Which packaged `datasets` loader to use with `data_files`. Ignored otherwise."""

    # --- fields added for the auditory / speech_output rows (audio-licence-audit.md) ---
    upstream_url: str = ""
    """Where the upstream terms this entry's `upstream` field quotes were fetched from."""
    upstream_license_verbatim: str = ""
    """The upstream's own words, quoted -- not paraphrased. `upstream` (above) already
    exists for a one-line disagree/agree note; this field is for entries where the exact
    wording is the whole finding (e.g. WavCaps: 'Only academic uses are allowed', against a
    cc-by-4.0 mirror tag) and paraphrasing it would understate the mismatch."""
    redistribute_verdict: str = ""
    """Separate from `verdict` (which gates TRAINING): whether re-hosting the raw bytes is
    clean, restricted, or BLOCKING. A corpus can be training-clean but redistribution-
    restricted (VoxPopuli: data is PERMISSIVE_OK to train on; Meta's OWN models built from
    it are CC BY-NC -- that restriction does not attach to the data itself, so it is
    recorded here rather than moving `verdict`)."""
    flags: tuple[str, ...] = ()
    """Zero or more of NC / SA / ND / attribution -- the audit's own flag vocabulary,
    kept as machine-checkable tags rather than only prose in `why`/`caveat`."""
    provenance_group: str = ""
    """The audit's load-bearing correction: LibriSpeech, LibriTTS/-R, MLS, Libri-Light,
    LJSpeech, CSS10, M-AILABS and Hi-Fi TTS/HiFiTTS-2 are ten differently-named datasets
    over ONE volunteer public-domain-audiobook pool (LibriVox) -- balance caps (B1/B2) must
    apply to the group, not to each name, or a v1 corpus is a LibriVox monoculture wearing
    ten labels. Every entry that draws on that pool MUST set provenance_group="librivox",
    whichever of auditory/speech_output region it lands in -- `audio_balance_report()`
    groups across both regions for exactly this reason.
    """
    approx_hours: float = 0.0
    """Best-available hour figure from audio-licence-audit.md, used ONLY as the weight for
    `audio_balance_report()`'s B1/B2 arithmetic. 0.0 means "not measured this session" --
    the audit says so explicitly for some entries (CSS10; HiFiTTS-2's 36,700h is mirror-tag-
    verified only and is deliberately excluded rather than risk anchoring a balance number
    on an unconfirmed figure that would dominate any mix it joined) -- and such entries are
    excluded from the balance sum rather than assumed negligible.
    """
    fetch_kind: str = "hf_dataset"
    """Which download path `fetch()` takes:
    - "hf_dataset": the existing `datasets.load_dataset(...).to_parquet()` path (unchanged).
    - "hf_files_bounded": walk the Hub repo's file tree under `file_prefix`, taking files in
      a seeded-random (not prefix) order until `max_fetch_bytes` is reached -- for corpora
      whose full size is TB-scale and only a bounded slice is wanted (VoxPopuli, People's
      Speech, FSD50K).
    - "http_archive": stream one fixed URL (`archive_url`) straight to disk -- for corpora
      whose real distribution point is not the Hub at all (MUSAN, VCTK: both ship as a
      single archive from their academic host, and both HF mirrors found for them turned
      out to carry no actual audio -- see caveat on each entry).
    - "unavailable": no packaged mirror exists (Libri-Light: audit found none; CSS10: no
      official data-licence source, only a code-repo badge). `fetch()` writes a manifest-
      only receipt and does not attempt a download.
    """
    file_prefix: str = ""
    """hf_files_bounded only: restrict the repo's file tree to paths starting with this
    (e.g. "en/" for VoxPopuli's English config, "clean/train-" for People's Speech)."""
    archive_url: str = ""
    """http_archive only: the single URL to stream to disk."""
    max_fetch_bytes: int = 0
    """hf_files_bounded only: stop once accumulated file size would exceed this. 0 means
    unbounded (http_archive entries ignore this -- they always fetch the one whole file)."""
    bounded_slice: bool = False
    """True for the handful of entries actually selected for the first bounded audio fetch
    (audio-licence-audit.md's 150 GB slice). False entries still get a real manifest via
    `fetch()` -- provenance, licence, verdict, flags, all recorded -- just with no audio
    bytes pulled in this pass; `manifest["audio_fetched"]` records which case applied so a
    manifest never silently implies bytes that were never pulled.
    """
    name: str = ""
    """Explicit local directory name, for entries with no `repo_id` at all (no packaged
    mirror -- Libri-Light, CSS10; or fetched from a direct archive URL rather than the Hub
    -- MUSAN, and REFUSE entries with no mirror -- TED-LIUM 3, Switchboard). Ignored (in
    favour of the repo_id-derived name) when `repo_id` is set."""

    @property
    def local(self) -> Path:
        base = self.name or (self.repo_id.split("/")[-1] if self.repo_id else "unnamed")
        name = base + (f"-{self.config}" if self.config else "")
        return BULK_CORPUS / self.region / name


# Seeded with entries already verified in this repo's fetch scripts. Everything added here
# must carry an observed license string and a verdict; unverified entries are REJECTED
# rather than fetched, so an unattended run cannot quietly widen the licence surface.
CATALOGUE: list[Dataset] = [
    # --- code -------------------------------------------------------------------------
    Dataset(
        repo_id="codeparrot/apps",
        region="code",
        license="mit (verified via HF dataset_info tags)",
        verdict=TRAIN_OK,
        why="10k Python problem/solution pairs with test cases; clean MIT",
        columns=("question", "solutions"),
        revision="refs/convert/parquet",
        caveat=(
            "repo root is a legacy apps.py loading script (train.jsonl/test.jsonl are "
            "gated behind it) -- datasets 5.0.1 refuses to run it: 'Dataset scripts are no "
            "longer supported, but found apps.py'. Fetch pins revision=refs/convert/parquet, "
            "HF's own auto-converted Parquet mirror of the same data (config='default', "
            "verified to carry the same 7 columns: problem_id/question/solutions/"
            "input_output/difficulty/url/starter_code). solutions and input_output are "
            "JSON-encoded strings; ~1235 test rows have no solution"
        ),
    ),
    Dataset(
        repo_id="deepmind/code_contests",
        region="code",
        license="cc-by-4.0 (verified via HF dataset_info tags)",
        verdict=TRAIN_OK,
        why="competitive programming descriptions with multi-language solutions",
        columns=("description", "solutions"),
        caveat="carries incorrect_solutions too; exclude them or the region learns wrong code. CC-BY requires attribution",
    ),
    # --- retrieval --------------------------------------------------------------------
    Dataset(
        repo_id="rajpurkar/squad",
        region="retrieve",
        license="cc-by-sa-4.0 (verified via HF dataset_info tags)",
        verdict=TRAIN_OK,
        why="single-hop Wikipedia QA; broadens beyond finance and web answers",
        columns=("question", "context"),
        caveat="share-alike obligation attaches to derivatives of the data itself",
    ),
    # BeIR/hotpotqa has no "train" split at all -- it is published as two SEPARATE
    # configs, "corpus" (each split literally named after its config) and "queries", with
    # relevance judgements in a THIRD, companion repo (BeIR/hotpotqa-qrels: config
    # "default", splits train/validation/test, columns corpus-id/query-id/score, also
    # cc-by-sa-4.0 and also agreeing with hotpotqa.github.io). The original entry's
    # splits=("train",) default and columns=("query","passage") described the joined
    # form nobody had built, not anything the repo actually serves -- that is why fetch
    # raised `ValueError: Config name is missing. Please pick one among the available
    # configs: ['corpus', 'queries']`. Fetching corpus and queries is a clean fit for the
    # existing one-config-per-entry fetch(); the corpus+queries+qrels JOIN into
    # query-passage training pairs is not, and is left to a runner (fetch
    # BeIR/hotpotqa-qrels separately, then join on query-id/corpus-id).
    Dataset(
        repo_id="BeIR/hotpotqa",
        region="retrieve",
        license="cc-by-sa-4.0 (HF card)",
        upstream="hotpotqa.github.io states CC BY-SA 4.0 — mirror and source AGREE",
        verdict=TRAIN_OK,
        config="corpus",
        splits=("corpus",),
        why="multi-hop retrieval, a reasoning shape absent from the current mix",
        columns=("_id", "title", "text"),
        caveat=(
            "this is the passage pool only. queries ship as the sibling 'queries' config "
            "below; relevance judgements ship in the separate BeIR/hotpotqa-qrels repo and "
            "are NOT fetched by this entry -- join corpus-id/query-id from qrels against "
            "these two before using this as query-passage training pairs"
        ),
    ),
    Dataset(
        repo_id="BeIR/hotpotqa",
        region="retrieve",
        license="cc-by-sa-4.0 (HF card)",
        upstream="hotpotqa.github.io states CC BY-SA 4.0 — mirror and source AGREE",
        verdict=TRAIN_OK,
        config="queries",
        splits=("queries",),
        why="query side of the multi-hop retrieval pair; see the 'corpus' config above",
        columns=("_id", "text"),
        caveat="join against BeIR/hotpotqa-qrels and the 'corpus' config entry before use",
    ),
    # --- semantic similarity ----------------------------------------------------------
    Dataset(
        repo_id="stanfordnlp/snli",
        region="compress",
        license="cc-by-sa-4.0 (verified via HF dataset_info tags)",
        verdict=TRAIN_OK,
        why="entailment pairs; the same objective all-nli's pair config serves",
        columns=("premise", "hypothesis"),
        caveat=(
            "NOT pre-filtered: label 0=entailment 1=neutral 2=contradiction, -1=no consensus. "
            "MUST filter to label==0. Training on it unfiltered is exactly the mistake that "
            "held compress at 0.26"
        ),
    ),
    # --- classification / utility models ----------------------------------------------
    Dataset(
        repo_id="PolyAI/banking77",
        region="classify",
        license="cc-by-4.0 (verified via HF dataset_info tags)",
        upstream="github.com/PolyAI-LDN/task-specific-datasets LICENSE file is Creative "
        "Commons Attribution 4.0 International — mirror and source AGREE",
        verdict=TRAIN_OK,
        why="77 fine-grained intents; sized for the tiny classifiers CSD wants",
        splits=("train", "test"),
        columns=("text", "category"),
        data_files={
            "train": "https://raw.githubusercontent.com/PolyAI-LDN/"
            "task-specific-datasets/master/banking_data/train.csv",
            "test": "https://raw.githubusercontent.com/PolyAI-LDN/"
            "task-specific-datasets/master/banking_data/test.csv",
        },
        caveat=(
            "repo root is a legacy banking77.py loading script with NO "
            "refs/convert/parquet mirror available -- datasets 5.0.1 refuses to run it: "
            "'Dataset scripts are no longer supported, but found banking77.py'. Read the "
            "script: it does nothing but download these same two CSVs from "
            "PolyAI-LDN/task-specific-datasets on GitHub, so fetch reads them directly via "
            "the built-in 'csv' loader instead (verified: 10003 train rows, columns "
            "text/category). Column is the raw 'category' string label ('card_arrival' "
            "etc), not the retired script's int-encoded ClassLabel 'label'"
        ),
    ),
    Dataset(
        repo_id="google-research-datasets/go_emotions",
        region="classify",
        license="apache-2.0 (verified via HF dataset_info tags)",
        verdict=TRAIN_OK,
        config="simplified",
        why="27 emotions plus neutral; a second classifier task with a clean licence",
        columns=("text", "labels"),
        caveat="multi-label, not single-label; example_very_unclear rows are annotator-flagged noise",
    ),
    # --- math / reasoning -------------------------------------------------------------
    Dataset(
        repo_id="openai/gsm8k",
        region="reason",
        config="main",
        license="mit (verified via HF dataset_info tags)",
        verdict=TRAIN_OK,
        why="human-authored step-by-step arithmetic; no third-party competition provenance",
        columns=("question", "answer"),
    ),
    Dataset(
        repo_id="deepmind/aqua_rat",
        region="reason",
        config="raw",
        license="apache-2.0 (verified via HF dataset_info tags)",
        verdict=TRAIN_OK,
        why="algebraic reasoning with natural-language rationales; a different shape to GSM8K",
        columns=("question", "rationale"),
    ),
    # --- vision -----------------------------------------------------------------------
    Dataset(
        repo_id="timm/oxford-iiit-pet",
        region="vl",
        license="cc-by-sa-4.0 (verified via HF dataset_info tags)",
        verdict=TRAIN_OK,
        why="37-class fine-grained classification with parquet-embedded images",
        columns=("image", "label"),
        caveat="native resolution varies 114-3260px; the loader must resize",
    ),
    Dataset(
        repo_id="zalando-datasets/fashion_mnist",
        region="vl",
        license="mit (verified via HF dataset_info tags)",
        verdict=TRAIN_OK,
        why="clean-licence 28x28 classification; a cheap probe target",
        columns=("image", "label"),
    ),
    # --- auditory (REGION-TAXONOMY-AND-INTERCONNECT.md §1.3/§4.1 row A0; source of truth
    #     is audio-licence-audit.md's "Recommended v1 auditory corpus mix", not re-derived
    #     here). Spectrogram-frame patch tokens, interface byte-identical to `vl`'s. -------
    Dataset(
        repo_id="openslr/librispeech_asr",
        region="auditory",
        license="cc-by-4.0 (verified via HF dataset_info tags)",
        upstream_url="https://www.openslr.org/12",
        upstream_license_verbatim="CC BY 4.0 (name-level match with the mirror tag)",
        verdict=TRAIN_OK,
        redistribute_verdict="clean",
        flags=("attribution",),
        provenance_group="librivox",
        approx_hours=1000.0,
        why="ASR-original LibriVox re-segmentation; the anchor of the librivox group",
    ),
    Dataset(
        repo_id="facebook/multilingual_librispeech",
        region="auditory",
        license="cc-by-4.0 (INFERRED via WebSearch snippet; huggingface.co API returned "
        "401 to this audit's fork all session -- see audio-licence-audit.md's session-wide "
        "tooling caveat)",
        upstream_url="https://www.openslr.org/94/",
        upstream_license_verbatim="Public domain (LibriVox source audio)",
        verdict=TRAIN_OK,
        redistribute_verdict="clean",
        flags=(),
        provenance_group="librivox",
        approx_hours=50500.0,
        why="8-language LibriVox re-segmentation; the largest clean non-English audio in the mix",
    ),
    Dataset(
        repo_id="",
        name="libri-light",
        region="auditory",
        license="no HF mirror found (audit searched)",
        upstream_url="https://github.com/facebookresearch/libri-light",
        upstream_license_verbatim="code=MIT (verified); the AUDIO data licence was not "
        "located this session -- inferred PERMISSIVE_OK from LibriVox lineage, not itself "
        "confirmed",
        verdict=TRAIN_OK,
        redistribute_verdict="inferred clean, not independently confirmed",
        flags=(),
        provenance_group="librivox",
        approx_hours=60000.0,
        fetch_kind="unavailable",
        why="~99% unlabeled; USE for self-supervised pretrain only, EVAL-ONLY for the "
        "small transcribed slice",
        caveat="no packaged mirror exists to fetch from -- this entry gets a manifest-only "
        "receipt recording the finding, not audio bytes. A future fetch needs the "
        "project's own download scripts (github.com/facebookresearch/libri-light) run by "
        "hand, licence-reviewed again before use",
    ),
    Dataset(
        repo_id="facebook/voxpopuli",
        region="auditory",
        config="en",
        license="cc0-1.0 + other (verified via HF dataset_info tags)",
        upstream_url="https://github.com/facebookresearch/voxpopuli",
        upstream_license_verbatim="CC0 (the data itself); CC BY-NC (Meta's OWN trained "
        "models/code only -- does not attach to the data)",
        verdict=TRAIN_OK,
        redistribute_verdict="clean (data)",
        flags=(),
        provenance_group="voxpopuli",
        approx_hours=401800.0,
        fetch_kind="hf_files_bounded",
        file_prefix="en/",
        max_fetch_bytes=32 * 1024**3,
        bounded_slice=True,
        why="European Parliament speech, public official proceedings; a genuinely "
        "independent-lineage provenance group at a scale that can offset the LibriVox "
        "group instead of compounding it",
        caveat="400,000h unlabelled + 1,800h transcribed at full scale -- the English "
        "config alone (30 train shards, ~3.2GB each) is ~97GB; this entry's bounded fetch "
        "takes a seeded-random subset of shards up to max_fetch_bytes, not a prefix",
    ),
    Dataset(
        repo_id="MLCommons/peoples_speech",
        region="auditory",
        config="clean",
        license="cc-by-2.0/2.5/3.0/4.0 + cc-by-sa-3.0/4.0, per-config (verified via HF "
        "dataset_info tags)",
        upstream_url="https://huggingface.co/datasets/MLCommons/peoples_speech",
        upstream_license_verbatim="mixed CC-BY / CC-BY-SA, self-declared per Archive.org "
        "uploader; the 'clean'/'clean_sa' configs are the pre-filtered attribution-only "
        "and share-alike splits",
        verdict=TRAIN_OK,
        redistribute_verdict="clean on the *-clean configs fetched here",
        flags=("attribution",),
        provenance_group="peoples_speech",
        approx_hours=30000.0,
        fetch_kind="hf_files_bounded",
        file_prefix="clean/train-",
        max_fetch_bytes=32 * 1024**3,
        bounded_slice=True,
        why="Archive.org crowd speech, filtered to the clean (non-SA) config; a third "
        "independent provenance group",
        caveat="'clean-config share not separately measured' per the audit -- fetch the "
        "'clean' config only, never the unfiltered 'raw' superset",
    ),
    Dataset(
        repo_id="edinburghcstr/ami",
        region="auditory",
        license="cc-by-4.0 (verified via HF dataset_info tags)",
        upstream_url="https://groups.inf.ed.ac.uk/ami/corpus/license.shtml",
        upstream_license_verbatim="CC BY 4.0 (full text, name-level match)",
        verdict=TRAIN_OK,
        redistribute_verdict="clean",
        flags=("attribution",),
        provenance_group="purpose_recorded",
        approx_hours=100.0,
        why="purpose-recorded meeting speech; small but independent-lineage",
    ),
    Dataset(
        repo_id="Fhrozen/FSD50k",
        region="auditory",
        license="cc-by-4.0 (this mirror's blanket card tag)",
        upstream_url="https://zenodo.org/records/4060432",
        upstream_license_verbatim="per-clip CC0 / CC-BY / CC-BY-NC mix under a CC-BY "
        "umbrella; each clip's own Freesound licence ships in "
        "metadata/dev_clips_info_FSD50K.json and metadata/eval_clips_info_FSD50K.json",
        verdict=TRAIN_OK,
        redistribute_verdict="clean IF filtered to CC0+CC-BY clips; NC clips are USE-NC, not clean",
        flags=("attribution",),
        provenance_group="freesound",
        approx_hours=108.0,
        fetch_kind="hf_files_bounded",
        max_fetch_bytes=6 * 1024**3,
        bounded_slice=True,
        why="general sound events, not speech; the mix's non-speech auditory diversity",
        caveat="the blanket cc-by-4.0 mirror tag is NOT the per-clip truth -- a consumer "
        "MUST join clip id against the licence column in the metadata json before training "
        "on anything beyond CC0/CC-BY, exactly the 'repo'-column filter pattern already "
        "used to rescue 71.3% of `code`. This entry's bounded fetch pulls metadata+labels "
        "in full plus a seeded-random sample of clips/dev and clips/eval; it does NOT "
        "pre-filter by licence -- that filter is a training-time loader responsibility "
        "(B3/B4 discipline: declare it in code, not a comment)",
    ),
    Dataset(
        repo_id="",
        name="musan",
        region="auditory",
        license="not confirmed on any HF mirror found (audit's own conclusion)",
        upstream_url="https://www.openslr.org/17/",
        upstream_license_verbatim="CC BY 4.0 (name-level match)",
        verdict=TRAIN_OK,
        redistribute_verdict="clean",
        flags=("attribution",),
        provenance_group="musan",
        approx_hours=109.0,
        fetch_kind="http_archive",
        archive_url="https://www.openslr.org/resources/17/musan.tar.gz",
        bounded_slice=True,
        why="CC-licensed music + public-domain speech + noise; used for augmentation as "
        "much as content diversity",
        caveat="every HF 'musan' mirror this session checked (e.g. csukuangfj/musan) "
        "carries no actual audio content, only a .gitattributes stub -- fetched directly "
        "from OpenSLR instead, the same host the mirror's own README points back to",
    ),
    # --- speech_output (Bucket C: TTS/speech-output training material). Shares
    #     provenance_group="librivox" with several `auditory` rows above where the
    #     lineage is the same LibriVox pool repurposed for TTS -- audio_balance_report()
    #     groups both regions together for exactly this reason. ------------------------
    Dataset(
        repo_id="keithito/lj_speech",
        region="speech_output",
        license="unlicense (verified via HF dataset_info tags)",
        upstream_url="https://keithito.com/LJ-Speech-Dataset/",
        upstream_license_verbatim="public domain",
        verdict=TRAIN_OK,
        redistribute_verdict="clean",
        flags=(),
        provenance_group="librivox",
        approx_hours=24.0,
        why="single-speaker neutral-narration TTS anchor, fully public domain",
    ),
    Dataset(
        repo_id="mythicinfinity/libritts_r",
        region="speech_output",
        license="cc-by-4.0 (verified via HF dataset_info tags)",
        upstream_url="https://www.openslr.org/141/",
        upstream_license_verbatim="CC BY 4.0 (name-level match)",
        verdict=TRAIN_OK,
        redistribute_verdict="clean",
        flags=("attribution",),
        provenance_group="librivox",
        approx_hours=585.0,
        why="multi-speaker LibriVox re-recording, quality-passed specifically for TTS "
        "(preferred over plain LibriTTS per the audit)",
    ),
    Dataset(
        repo_id="",
        name="css10",
        region="speech_output",
        license="apache-2.0 badge on the github repo (verified) -- this is a CODE licence, "
        "not a data licence; no HF mirror carries an explicit data-licence tag",
        upstream_url="https://github.com/Kyubyong/css10",
        upstream_license_verbatim="no explicit data licence located; texts are LibriVox-"
        "sourced public domain (inferred, not independently confirmed)",
        verdict=TRAIN_OK,
        redistribute_verdict="likely clean (inferred)",
        flags=(),
        provenance_group="librivox",
        approx_hours=0.0,
        fetch_kind="unavailable",
        why="10-language single-speaker TTS; the mix's only non-English/non-Mandarin "
        "breadth from the librivox lineage",
        caveat="approx_hours deliberately left at 0.0 (excluded from the balance sum) -- "
        "size was not measured this session, and the apache-2.0 GitHub badge is almost "
        "certainly the CodeSearchNet-Python shape (a code licence read as a data licence). "
        "EVAL-ONLY, not TRAIN_OK, if an explicit per-language data licence is later "
        "required rather than the inferred LibriVox-PD reading",
    ),
    Dataset(
        repo_id="gigant/m-ailabs_speech_dataset_fr",
        region="speech_output",
        config="8-language subset, EXCLUDING Ukrainian",
        license="cc (this mirror's tag -- meaningless generic string, verified mismatch)",
        upstream_url="https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/",
        upstream_license_verbatim="BSD-style, explicit commercial use permitted, notice-"
        "retention required (NOT a CC licence at all)",
        verdict=TRAIN_OK,
        redistribute_verdict="clean, notice-retention (8 langs only)",
        flags=(),
        provenance_group="librivox",
        approx_hours=900.0,
        why="9-language LibriVox-PD TTS corpus; 8 of 9 languages are BSD-style clean",
        caveat="the Ukrainian subset is separately marked 'ML purposes only' upstream -- "
        "EVAL-ONLY/REFUSE, excluded from this entry and from approx_hours",
    ),
    Dataset(
        repo_id="MikhailT/hifi-tts",
        region="speech_output",
        license="cc-by-4.0 (verified via HF dataset_info tags)",
        upstream_url="https://www.openslr.org/109/",
        upstream_license_verbatim="CC BY 4.0 (name-level match)",
        verdict=TRAIN_OK,
        redistribute_verdict="clean",
        flags=("attribution",),
        provenance_group="librivox",
        approx_hours=292.0,
        why="10-speaker high-fidelity LibriVox re-recording for TTS",
    ),
    Dataset(
        repo_id="nvidia/hifitts-2",
        region="speech_output",
        license="cc-by-4.0 (verified via HF dataset_info tags, mirror only)",
        upstream_url="https://huggingface.co/datasets/nvidia/hifitts-2",
        upstream_license_verbatim="not independently re-checked off-HF this session",
        verdict=TRAIN_OK,
        redistribute_verdict="clean, verify upstream before anchoring anything on the "
        "36,700h figure",
        flags=("attribution",),
        provenance_group="librivox",
        approx_hours=0.0,
        why="~5,000-speaker LibriVox re-recording; the mix's largest single TTS source by "
        "a wide margin, if the hour figure holds up",
        caveat="approx_hours deliberately left at 0.0 (excluded from the balance sum): the "
        "36,700h figure is mirror-tag-verified only, not confirmed at an independent "
        "upstream page this session, and per the audit's own open question it would "
        "dominate any mix it joined uncapped -- do not fetch or size this at scale until "
        "that figure is independently confirmed",
    ),
    Dataset(
        repo_id="CSTR-Edinburgh/vctk",
        region="speech_output",
        license="cc-by-4.0 (verified via HF dataset_info tags) -- BUT this mirror is a "
        "legacy loading script (vctk.py) with no data files on the Hub at all",
        upstream_url="https://datashare.ed.ac.uk/handle/10283/3443",
        upstream_license_verbatim="CC BY 4.0 (full text, name-level match). 'Moral rights "
        "and personality rights are not licensed' -- CC BY covers the recording copyright, "
        "not a blanket right to synthesize any of the 110 speakers' voices",
        verdict=TRAIN_OK,
        redistribute_verdict="clean + attribution",
        flags=("attribution",),
        provenance_group="purpose_recorded",
        approx_hours=44.0,
        fetch_kind="http_archive",
        archive_url="https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip",
        bounded_slice=True,
        why="110 speakers, purpose-recorded for voice-cloning research -- the strongest "
        "consent story in the whole audit and an independent (non-librivox) lineage",
        caveat="the HF mirror's own loading script (vctk.py) downloads from this exact "
        "URL -- fetched directly rather than through `datasets`, since the script itself "
        "would be refused by datasets>=4 ('Dataset scripts are no longer supported') and "
        "carries no repo files to fall back to",
    ),
    Dataset(
        repo_id="AISHELL/AISHELL-3",
        region="speech_output",
        license="apache-2.0 (verified via HF dataset_info tags)",
        upstream_url="https://www.openslr.org/93/",
        upstream_license_verbatim="Apache-2.0 (name-level match)",
        verdict=TRAIN_OK,
        redistribute_verdict="clean",
        flags=(),
        provenance_group="purpose_recorded",
        approx_hours=85.0,
        why="218-speaker Mandarin, purpose-recorded professionally for TTS -- the "
        "strongest non-English, non-LibriVox candidate in the whole audit",
    ),
    Dataset(
        repo_id="ylacombe/expresso",
        region="speech_output",
        license="cc-by-nc-4.0 (verified via HF dataset_info tags)",
        upstream_url="https://speechbot.github.io/expresso/",
        upstream_license_verbatim="CC BY-NC 4.0 (confirmed both ends, no contradiction)",
        verdict=TRAIN_OK,
        redistribute_verdict="NC-tier redistribution",
        flags=("NC", "attribution"),
        provenance_group="professional_expressive",
        approx_hours=40.0,
        why="the ONLY expressive/paralinguistic-labelled TTS source in the audit (26 "
        "styles, 4 professional performers). Costs nothing extra at the composed-model "
        "level (already CC BY-NC-SA 4.0 via `memory`/GooAQ, DEC-31); a real, separate cost "
        "only if speech_output ever ships as a standalone checkpoint (CC BY-NC 4.0 instead "
        "of MIT/CC BY)",
    ),
    # --- REFUSE. BLOCKING audio sources -- present on purpose (per REFUSE's own docstring
    #     above), verdict=REFUSE so `fetch()` raises BlockingSourceError by construction
    #     rather than relying on a status check nobody re-verifies. -----------------------
    Dataset(
        repo_id="speechcolab/gigaspeech",
        region="auditory",
        license="apache-2.0 (verified via HF dataset_info tags)",
        upstream_url="TERMS_OF_ACCESS.md (404 on both expected GitHub paths; corroborated "
        "via two independent secondary sources)",
        upstream_license_verbatim="'Researcher shall use the Database only for non-"
        "commercial research and educational purposes.' 'SpeechColab does not own the "
        "copyright of the audio files.'",
        verdict=REFUSE,
        redistribute_verdict="BLOCKING",
        flags=("NC",),
        provenance_group="gigaspeech",
        why="the second sentence is the tiny-imagenet shape: restrictive scope PLUS a "
        "non-ownership disclaimer, not a real rights holder's NC grant. A live, unresolved "
        "maintainer dispute over the apache-2.0 tag exists on the HF discussion thread",
    ),
    Dataset(
        repo_id="",
        name="ted-lium-3",
        region="auditory",
        license="not resolvable via API (401 all session)",
        upstream_url="openslr.org's TED-LIUM 3 page, reproduced by TFDS's catalog entry "
        "and independently corroborated by coqui-ai's open-speech-corpora table",
        upstream_license_verbatim="'The TED-LIUM corpus is licensed under Creative Commons "
        "BY-NC-ND 3.0... All talks and text are property of TED Conferences LLC.'",
        verdict=REFUSE,
        redistribute_verdict="BLOCKING",
        flags=("NC", "ND"),
        provenance_group="ted_lium",
        why="ND, not merely NC -- CC's own ND clause forbids distributing modified "
        "material at all, commercial or not. Outside the scope of the NC-tolerant policy, "
        "which was written to absorb NonCommercial terms, not NoDerivatives ones. Needs "
        "its own operator ruling before this could ever move off REFUSE",
    ),
    Dataset(
        repo_id="amphion/Emilia-Dataset",
        region="auditory",
        license="cc-by-nc-4.0 core / cc-by-4.0 YODAS subset per the audit (401 all session; "
        "this mirror's live card now reads cc-by-4.0, recorded for completeness -- it does "
        "not change the verdict, see why)",
        upstream_url="github.com raw README (fetched live)",
        upstream_license_verbatim="'Emilia does not own the copyright to the audio files; "
        "the copyright remains with the original owners of the videos or audio.'",
        verdict=REFUSE,
        redistribute_verdict="BLOCKING",
        flags=("NC",),
        provenance_group="emilia",
        why="101,000 hours -- the single largest tempting corpus in the whole audit and "
        "the clearest REFUSE. Whatever CC tag the curators apply, they say themselves they "
        "had no ownership standing to grant it. In-the-wild scrape, no consent",
    ),
    Dataset(
        repo_id="kensho/spgispeech",
        region="auditory",
        license="gated EULA (structurally -- access requires accepting terms on HF)",
        upstream_url="S&P Global's own SPGISpeech terms (snippet only, not a direct fetch)",
        upstream_license_verbatim="academic/internal-only use, no re-identification, no "
        "database-building (INFERRED from a WebSearch snippet, not independently re-"
        "fetched in full this session)",
        verdict=REFUSE,
        redistribute_verdict="BLOCKING",
        flags=("NC",),
        provenance_group="spgispeech",
        why="a real named owner (S&P Global) granting a stricter-than-NC bespoke academic "
        "EULA -- no re-hosting, no derivative database. 5,000h, ~50K speakers",
    ),
    Dataset(
        repo_id="",
        name="switchboard-1",
        region="auditory",
        license="none -- LDC, no HF or other public mirror",
        upstream_url="https://catalog.ldc.upenn.edu/LDC97S62",
        upstream_license_verbatim="all-rights-reserved, paywalled LDC membership/fee corpus",
        verdict=REFUSE,
        redistribute_verdict="BLOCKING",
        flags=(),
        provenance_group="switchboard",
        why="paywalled membership corpus; no licence grant this project could ever act on",
    ),
    Dataset(
        repo_id="agkphysics/AudioSet",
        region="auditory",
        license="cc-by-4.0 (verified via HF dataset_info tags -- applied to the actual "
        "re-hosted FLAC audio, not just metadata)",
        upstream_url="Google's own AudioSet download page (fetched live)",
        upstream_license_verbatim="'The dataset is made available by Google Inc. under a "
        "Creative Commons Attribution 4.0 International (CC BY 4.0) license, while the "
        "ontology is available under [CC BY-SA 4.0].' Google's own release ships ONLY "
        "YouTube video IDs, timestamps and labels -- never audio",
        verdict=REFUSE,
        redistribute_verdict="BLOCKING",
        flags=("SA",),
        provenance_group="audioset",
        why="the CC BY 4.0 grant covers metadata Google authored, not the copyrighted "
        "third-party YouTube audio it never touched. This mirror downloaded the real audio "
        "in 2023 and tags the whole FLAC re-host cc-by-4.0 anyway -- the tiny-imagenet "
        "shape, now with real audio bytes in circulation. AudioCaps and WavCaps inherit "
        "this defect by lineage (below)",
    ),
    Dataset(
        repo_id="jp1924/AudioCaps",
        region="auditory",
        license="no explicit tag on this mirror's card (audit's own AudioCaps finding was "
        "'mit', INFERRED, on a different mirror -- not independently re-verified here)",
        upstream_url="AudioCaps' own upstream terms (fetched live)",
        upstream_license_verbatim="'academic purposes only'",
        verdict=REFUSE,
        redistribute_verdict="BLOCKING",
        flags=(),
        provenance_group="audioset",
        why="inherits AudioSet's non-ownership defect by lineage AND carries its own "
        "academic-only restriction on top -- two independent BLOCKING reasons",
    ),
    Dataset(
        repo_id="cvssp/WavCaps",
        region="auditory",
        license="cc-by-4.0 (verified via HF dataset_info tags)",
        upstream_url="WavCaps' own upstream README (fetched live)",
        upstream_license_verbatim="'Only academic uses are allowed for WavCaps dataset. By "
        "downloading audio clips through the links provided in the json files, you agree "
        "that you will use the audios for research purposes only.'",
        verdict=REFUSE,
        redistribute_verdict="BLOCKING",
        flags=(),
        provenance_group="audioset",
        why="the cleanest contradiction in the whole audit: no ambiguity to resolve the "
        "way GooAQ's self-contradictory README-vs-LICENSE needed -- the upstream says one "
        "thing, the mirror tags another, flatly. ~27% of clips also inherit AudioSet's "
        "non-ownership defect by lineage",
    ),
    # --- REFUSED. Kept in the catalogue on purpose: a rejection nobody can see is a
    #     rejection that gets made again. ------------------------------------------------
    Dataset(
        repo_id="code-search-net/code_search_net",
        region="code",
        config="go",
        license="other (verified via HF dataset_info tags — NOT mit)",
        upstream="CodeSearchNet filtered only for repos permitting REDISTRIBUTION, which is "
        "not a commercial-use grant, and the dataset carries no per-row licence column",
        verdict=REJECTED,
        why="an earlier revision of this file claimed 'mit' here. That string was never "
        "observed; HF reports 'other'. The Python config already training the `code` "
        "region has the same unresolved status",
    ),
    Dataset(
        repo_id="BeIR/scifact",
        region="retrieve",
        license="cc-by-sa-4.0 (HF card)",
        upstream="allenai/scifact, the dataset this mirrors, is tagged cc-by-nc-2.0 — "
        "NON-COMMERCIAL. The mirror's tag contradicts its own source",
        verdict=REJECTED,
        why="proof that a mirror's licence tag is not evidence about the corpus it mirrors",
    ),
    Dataset(
        repo_id="hendrycks/competition_math",
        region="reason",
        license="mit (card) — but access is disabled on HF",
        upstream="under an active DMCA takedown tied to its AoPS competition-problem "
        "provenance; the MIT tag covers the repo's scripts, not the problem text",
        verdict=REJECTED,
        why="a live copyright dispute is not cured by a permissive tag on a repackaging. "
        "The same applies to MATH-derived subsets of otherwise-clean datasets",
    ),
]


def _disk_free_fraction(path: Path) -> tuple[float, float]:
    usage = shutil.disk_usage(path)
    return usage.used / usage.total, usage.free / 1e12


def _manifest_path(ds: Dataset) -> Path:
    return ds.local / "MANIFEST.json"


def is_present(ds: Dataset) -> bool:
    """True when this exact dataset+config is already on disk with a manifest."""
    man = _manifest_path(ds)
    if not man.is_file():
        return False
    try:
        rec = json.loads(man.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return rec.get("repo_id") == ds.repo_id and rec.get("config", "") == ds.config


def _manifest_sha256(manifest: dict) -> str:
    """Hash the manifest's own canonical JSON (sorted keys), excluding this field itself.

    This is a receipt over the manifest record, not over multi-GB audio payloads --
    hashing an 11GB tar.gz or a 32GB shard slice on every run would make `fetch()`
    minutes slower for no integrity benefit `bytes` doesn't already give; the manifest
    JSON is what downstream tooling reads and is what needs a tamper-evident checksum.
    """
    import hashlib

    canon = json.dumps(manifest, indent=2, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()


def _write_manifest(ds: Dataset, extra: dict) -> dict:
    ds.local.mkdir(parents=True, exist_ok=True)
    manifest = {
        **asdict(ds),
        **extra,
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    _manifest_path(ds).write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


def _fetch_hf_dataset(ds: Dataset, token: str | None) -> tuple[int, int]:
    """The pre-existing path: `datasets.load_dataset(...).to_parquet()`. Returns
    (rows, bytes)."""
    from datasets import load_dataset

    rows = 0
    for split in ds.splits:
        if ds.data_files:
            data = load_dataset(
                ds.builder,
                data_files={split: ds.data_files[split]},
                split=split,
                token=token,
                streaming=False,
            )
        else:
            data = load_dataset(
                ds.repo_id,
                ds.config or None,
                split=split,
                revision=ds.revision or None,
                token=token,
                streaming=False,
            )
        out = ds.local / f"{split}.parquet"
        data.to_parquet(str(out))
        rows += data.num_rows
    total_bytes = sum(f.stat().st_size for f in ds.local.glob("*.parquet"))
    return rows, total_bytes


def _fetch_hf_files_bounded(ds: Dataset, token: str | None) -> tuple[int, int]:
    """Walk a Hub repo's file tree under `file_prefix`, downloading files in a SEEDED-
    RANDOM order (not file-listing order) until `max_fetch_bytes` is reached.

    Seeded-random, not a prefix: CORPUS-CONTRACT.md's B4 names the exact anti-pattern this
    avoids -- `retrieve`'s `load_pairs` capped GooAQ at the first 400,000 rows in file
    order, which was the first 13.3% of the corpus, not a sample. Hub file listings for
    sharded datasets are often ordered by upload/session/recording time, so a raw prefix
    here risks the same shape of bias (e.g. VoxPopuli shards from only the earliest
    sessions). random.Random(seed) with a fixed seed keeps this reproducible across runs
    without keeping the bias.
    """
    import random

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.dataset_info(ds.repo_id, files_metadata=True, token=token)
    candidates = [
        (s.rfilename, s.size or 0)
        for s in info.siblings
        if s.rfilename.startswith(ds.file_prefix) and (s.size or 0) > 0
    ]
    random.Random(1337).shuffle(candidates)  # noqa: S311 -- sampling, not cryptographic

    budget = ds.max_fetch_bytes or float("inf")
    taken: list[str] = []
    total_bytes = 0
    for rfilename, size in candidates:
        if total_bytes + size > budget:
            continue  # keep scanning -- a later, smaller file may still fit the budget
        hf_hub_download(
            repo_id=ds.repo_id,
            repo_type="dataset",
            filename=rfilename,
            local_dir=str(ds.local),
            token=token,
        )
        taken.append(rfilename)
        total_bytes += size

    return len(taken), total_bytes


def _fetch_http_archive(ds: Dataset) -> tuple[int, int]:
    """Stream one fixed URL straight to disk (MUSAN, VCTK -- see fetch_kind docstring on
    Dataset). No datasets/huggingface_hub involved; this is not a Hub repo at all."""
    import urllib.request

    out = ds.local / ds.archive_url.rsplit("/", 1)[-1]
    with urllib.request.urlopen(ds.archive_url, timeout=120) as resp, open(out, "wb") as f:
        shutil.copyfileobj(resp, f, length=1024 * 1024)
    return 1, out.stat().st_size


def _fetch_gate(ds: Dataset, apply: bool, max_used_fraction: float) -> dict[str, object] | None:
    """The short-circuiting half of `fetch()`: licence/presence/disk/apply/manifest-only
    checks that each resolve the call without a real download. Returns a finished result
    dict when one of those applies, or None when `fetch()` should proceed to an actual
    network fetch. Split out from `fetch()` purely to keep both functions' return-statement
    count reasonable -- no behavioural difference from having it all in one function.
    """
    res: dict[str, object] = {"repo_id": ds.repo_id, "config": ds.config, "region": ds.region}

    if ds.verdict != TRAIN_OK:
        res["status"] = "REFUSED"
        res["reason"] = f"verdict={ds.verdict}; licence observed: {ds.license}. {ds.why}"
        return res

    if is_present(ds):
        res["status"] = "present"
        res["path"] = str(ds.local)
        return res

    used, _free_tb = _disk_free_fraction(BULK_CORPUS.parent)
    if used >= max_used_fraction:
        res["status"] = "DEFERRED"
        res["reason"] = (
            f"bulk at {used:.1%} used, at or above the {max_used_fraction:.0%} ceiling. "
            f"Publish and evict cold datasets before fetching more."
        )
        return res

    if not apply:
        res["status"] = "would fetch"
        res["path"] = str(ds.local)
        return res

    if ds.fetch_kind == "unavailable" or not ds.bounded_slice:
        # Either no packaged mirror exists at all (Libri-Light, CSS10), or this entry
        # simply isn't one of the handful selected for this pass's real audio fetch --
        # either way, write a manifest recording the catalogue's provenance/licence
        # findings with no audio bytes pulled. `audio_fetched: False` keeps this from ever
        # being read back as a real fetch by code that inspects the manifest contents.
        _write_manifest(ds, {"audio_fetched": False, "rows": 0, "bytes": 0, "seconds": 0.0})
        res["status"] = (
            "manifest-only (no mirror available)"
            if ds.fetch_kind == "unavailable"
            else "manifest-only (not in this pass's bounded slice)"
        )
        res["bytes"] = 0
        return res

    return None


def fetch(
    ds: Dataset, token: str | None, apply: bool, max_used_fraction: float
) -> dict[str, object]:
    """Fetch one dataset, refusing anything not licence-cleared.

    A REFUSE verdict raises `BlockingSourceError` immediately -- see that class's
    docstring for why this is a raise and not a returned status, and why `main()` catches
    it rather than letting it propagate.
    """
    if ds.verdict == REFUSE:
        raise BlockingSourceError(
            f"{ds.name or ds.repo_id or '(unnamed)'}: verdict=REFUSE -- BLOCKING per "
            f"audio-licence-audit.md. {ds.why}"
        )

    gated = _fetch_gate(ds, apply, max_used_fraction)
    if gated is not None:
        return gated

    res: dict[str, object] = {"repo_id": ds.repo_id, "config": ds.config, "region": ds.region}
    ds.local.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        if ds.fetch_kind == "hf_files_bounded":
            rows, total_bytes = _fetch_hf_files_bounded(ds, token)
        elif ds.fetch_kind == "http_archive":
            rows, total_bytes = _fetch_http_archive(ds)
        else:
            rows, total_bytes = _fetch_hf_dataset(ds, token)
    except Exception as exc:
        # A dataset already on disk with no MANIFEST.json still reads as "not present"
        # (see is_present()), so a retry after this is fixed re-fetches cleanly -- this
        # does not touch the licence gate, it only stops one entry's failure from
        # aborting every entry after it in the catalogue.
        res["status"] = "error"
        res["reason"] = f"{type(exc).__name__}: {exc}"
        return res

    _write_manifest(
        ds,
        {
            "audio_fetched": True,
            "rows": rows,
            "bytes": total_bytes,
            "seconds": round(time.time() - started, 1),
        },
    )
    res["status"] = "fetched"
    res["rows"] = rows
    res["bytes"] = total_bytes
    return res


def audio_balance_report(regions: tuple[str, ...] = ("auditory", "speech_output")) -> dict:
    """B1 (max single-source share <= 0.40) and B2 (N_eff = 1/Sum(p^2) >= 3), grouped by
    `provenance_group` -- LibriVox counted as ONE group across BOTH regions, per
    audio-licence-audit.md's central finding: LibriSpeech, LibriTTS-R, MLS, Libri-Light,
    LJSpeech, CSS10, M-AILABS and Hi-Fi TTS/HiFiTTS-2 are ten differently-named datasets
    over one volunteer-read public-domain-audiobook pool.

    Weighted by `approx_hours` (0.0 = not measured, excluded from the sum rather than
    assumed negligible -- CSS10 and HiFiTTS-2 are the current examples). This computes
    over the catalogue's NATURAL, uncapped sizes, exactly like CORPUS-CONTRACT.md's own
    "uncapped" comparison rows -- it is deliberately NOT the post-cap number a training
    sampler would draw, because a guard that can only ever report success is not a guard
    (verify-guards-by-making-them-fail). Callers building an actual training mix apply
    their own cap and recompute.
    """
    groups: dict[str, float] = {}
    for ds in CATALOGUE:
        if (
            ds.region in regions
            and ds.verdict == TRAIN_OK
            and ds.provenance_group
            and ds.approx_hours > 0
        ):
            groups[ds.provenance_group] = groups.get(ds.provenance_group, 0.0) + ds.approx_hours

    total = sum(groups.values())
    if total <= 0:
        return {
            "total_hours": 0.0,
            "groups": {},
            "shares": {},
            "b1_max_share": 0.0,
            "b2_n_eff": 0.0,
        }

    shares = {g: h / total for g, h in groups.items()}
    b1_max_share = max(shares.values())
    b2_n_eff = 1.0 / sum(s * s for s in shares.values())
    return {
        "total_hours": total,
        "groups": groups,
        "shares": shares,
        "b1_max_share": b1_max_share,
        "b2_n_eff": b2_n_eff,
    }


def print_balance_report(regions: tuple[str, ...] = ("auditory", "speech_output")) -> None:
    report = audio_balance_report(regions)
    print(f"\n  audio balance (uncapped, natural sizes) over {'+'.join(regions)}:")
    if not report["groups"]:
        print("    no TRAIN_OK entries with approx_hours>0 in these regions")
        return
    for group, hours in sorted(report["groups"].items(), key=lambda kv: -kv[1]):
        share = report["shares"][group]
        print(f"    {group:<24} {hours:>10,.0f} h  {share:>6.1%}")
    b1 = report["b1_max_share"]
    b2 = report["b2_n_eff"]
    print(f"    total {report['total_hours']:>10,.0f} h")
    print(
        f"    B1 max single-source share = {b1:.1%}  (threshold <=40%)  {'OK' if b1 <= 0.40 else 'FAILS'}"
    )
    print(
        f"    B2 N_eff (inverse Simpson) = {b2:.2f}  (threshold >=3)    {'OK' if b2 >= 3.0 else 'FAILS'}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="", help="restrict to these regions")
    ap.add_argument("--bulk-root", default="", help="override the corpus root")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument(
        "--max-used-fraction",
        type=float,
        default=0.5,
        help="stop fetching once bulk passes this; the operator's eviction threshold",
    )
    ap.add_argument(
        "--balance",
        action="store_true",
        help="print the auditory/speech_output B1/B2 provenance-group balance report and exit",
    )
    args = ap.parse_args()

    if args.balance:
        print_balance_report()
        return 0

    global BULK_CORPUS
    if args.bulk_root:
        BULK_CORPUS = Path(args.bulk_root)
    if not BULK_CORPUS.parent.is_dir():
        print(f"bulk not mounted at {BULK_CORPUS.parent}", file=sys.stderr)
        return 2
    BULK_CORPUS.mkdir(parents=True, exist_ok=True)

    used, free_tb = _disk_free_fraction(BULK_CORPUS.parent)
    print(f"csd-corpus-expand [{'APPLY' if args.apply else 'DRY-RUN'}]")
    print(f"  bulk {used:.1%} used, {free_tb:.2f} TB free, ceiling {args.max_used_fraction:.0%}")

    token = os.environ.get("HF_TOKEN")
    wanted = {r.strip() for r in args.regions.split(",") if r.strip()}
    ok = refused = errored = 0
    for ds in CATALOGUE:
        if wanted and ds.region not in wanted:
            continue
        try:
            res = fetch(ds, token, args.apply, args.max_used_fraction)
        except BlockingSourceError as exc:
            # verdict=REFUSE makes fetch() raise by construction (see BlockingSourceError's
            # docstring) -- caught here so one BLOCKING entry does not abort the whole
            # catalogue's status listing; the raise itself is still what a direct caller
            # (e.g. a test) sees.
            res = {"status": "REFUSED", "reason": str(exc)}
        status = str(res["status"])
        mark = {
            "fetched": "+",
            "present": "=",
            "would fetch": ".",
            "REFUSED": "!",
            "error": "x",
        }.get(status, "~" if str(status).startswith("manifest-only") else "?")
        label = (ds.name or ds.repo_id or "(unnamed)") + (f":{ds.config}" if ds.config else "")
        print(f"  {mark} {ds.region:<14} {label:<46} {status}")
        if "reason" in res:
            print(f"      {res['reason']}")
        if status in ("fetched", "present", "would fetch") or str(status).startswith(
            "manifest-only"
        ):
            ok += 1
        elif status == "REFUSED":
            refused += 1
        elif status == "error":
            errored += 1

    print(f"\n  {ok} usable, {refused} refused on licence grounds, {errored} errored")
    if wanted & {"auditory", "speech_output"} or not wanted:
        print_balance_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
