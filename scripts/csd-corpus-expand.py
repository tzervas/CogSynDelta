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

    @property
    def local(self) -> Path:
        name = self.repo_id.split("/")[-1] + (f"-{self.config}" if self.config else "")
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
        upstream="github.com/openai/grade-school-math LICENSE is the MIT License, "
        "'Copyright (c) 2021 OpenAI' — read at the primary source 2026-09-06, not "
        "inherited from the card. The HF repo is under OpenAI's own namespace (it is the "
        "publisher, not a re-host) and its card repeats 'The GSM8K dataset is licensed "
        "under the MIT License'. Mirror and source AGREE; the same MIT terms cover the "
        "`test` split as the `train` split",
        verdict=TRAIN_OK,
        why="human-authored step-by-step arithmetic; no third-party competition provenance",
        splits=("train", "test"),
        columns=("question", "answer"),
        caveat=(
            "`test` (1,319 rows) is a HOLDOUT and is not training material for any region. "
            "It is landed for the pre-registered reasoning battery only, reserved by shard "
            "path in csd-train-all.py's HELD_OUT_SHARDS and by item id in "
            "config/mind/splits/reason-gsm8k-test-holdout.json. Only `train` (7,473) is "
            "declared as a shard in REGIONS['reason']"
        ),
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
    """True when this exact dataset+config+splits is already on disk with a manifest.

    The splits clause is load-bearing, not defensive. Until 2026-09-06 this compared
    `repo_id` and `config` only, so WIDENING an already-fetched entry's `splits` -- the
    exact shape of landing `openai/gsm8k`'s `test` holdout beside the `train` split it
    already had -- read as "present" and the new split was never fetched. The run printed
    `= reason openai/gsm8k:main present` and exited 0, indistinguishable from success. A
    declared split with no parquet on disk is not present.
    """
    man = _manifest_path(ds)
    if not man.is_file():
        return False
    try:
        rec = json.loads(man.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if rec.get("repo_id") != ds.repo_id or rec.get("config", "") != ds.config:
        return False
    return all((ds.local / f"{split}.parquet").is_file() for split in ds.splits)


def fetch(ds: Dataset, token: str | None, apply: bool, max_used_fraction: float) -> dict:
    """Fetch one dataset, refusing anything not licence-cleared."""
    res: dict[str, object] = {"repo_id": ds.repo_id, "config": ds.config, "region": ds.region}

    if ds.verdict != TRAIN_OK:
        res["status"] = "REFUSED"
        res["reason"] = f"verdict={ds.verdict}; licence observed: {ds.license}. {ds.why}"
        return res

    if is_present(ds):
        res["status"] = "present"
        res["path"] = str(ds.local)
        return res

    used, free_tb = _disk_free_fraction(BULK_CORPUS.parent)
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

    from datasets import load_dataset

    ds.local.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = 0
    rows_by_split: dict[str, int] = {}
    try:
        for split in ds.splits:
            if ds.data_files:
                # Escape hatch: repo_id has no usable loading path on the Hub (retired
                # script, no parquet conversion), but the script's own only job was
                # downloading a plain file from a fixed URL -- so read it directly
                # through a packaged loader instead of executing remote code.
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
            rows_by_split[split] = int(data.num_rows)
    except Exception as exc:
        # A dataset already on disk with no MANIFEST.json still reads as "not present"
        # (see is_present()), so a retry after this is fixed re-fetches cleanly -- this
        # does not touch the licence gate, it only stops one entry's failure from
        # aborting every entry after it in the catalogue.
        res["status"] = "error"
        res["reason"] = f"{type(exc).__name__}: {exc}"
        return res
    manifest = {
        **asdict(ds),
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": rows,
        # Per split, not only the total: a multi-split entry's `rows` is a sum, and a sum
        # cannot say how many rows are training material and how many are holdout. The
        # balance accounting in CORPUS-CONTRACT.md Part 3 is per provenance group over the
        # REALISED training set, so it needs the split breakdown to be recorded, not
        # re-derived by opening parquet files later.
        "rows_by_split": rows_by_split,
        "seconds": round(time.time() - started, 1),
        "bytes": sum(f.stat().st_size for f in ds.local.glob("*.parquet")),
    }
    _manifest_path(ds).write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    res["status"] = "fetched"
    res["rows"] = rows
    res["bytes"] = manifest["bytes"]
    return res


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
    args = ap.parse_args()

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
        res = fetch(ds, token, args.apply, args.max_used_fraction)
        status = res["status"]
        mark = {
            "fetched": "+",
            "present": "=",
            "would fetch": ".",
            "REFUSED": "!",
            "error": "x",
        }.get(status, "?")
        label = f"{ds.repo_id}" + (f":{ds.config}" if ds.config else "")
        print(f"  {mark} {ds.region:<10} {label:<46} {status}")
        if "reason" in res:
            print(f"      {res['reason']}")
        if status in ("fetched", "present", "would fetch"):
            ok += 1
        elif status == "REFUSED":
            refused += 1
        elif status == "error":
            errored += 1

    print(f"\n  {ok} usable, {refused} refused on licence grounds, {errored} errored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
