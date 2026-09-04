"""Shared receipt-writing mechanics: code provenance and the training-shape defaults.

WHY THIS EXISTS
Every training, eval and quantization receipt in this project ends with the same three
lines: `out.mkdir(...)`, `path.write_text(json.dumps(receipt, ...))`,
`receipt["receipt_path"] = str(path)` -- three copies (`regions/pretrain.py`,
`regions/classify_pretrain.py`, `regions/vl_pretrain.py`) plus `scripts/csd-quantize.py`,
none of which recorded WHAT CODE produced the numbers in the receipt. A receipt with a
suspicious metric is unreadable without knowing whether the run that wrote it matches the
commit under review, or was training on an uncommitted change nobody else can reproduce.

`write_receipt` is now the one place that decides. It refuses outright -- raises, writes
nothing -- rather than let a receipt reach disk with no `code_revision` block, because a
receipt silently missing its own provenance is worse than one that fails loudly: nothing
downstream (`pipeline/receipt.py`'s dashboard reader included) checks for that gap, so the
first sign of it would be an operator staring at a green run they can no longer place
against a commit.

`trainer_defaults` closes the companion gap the PR #5 merge review found: ten default
values (steps, batch_size, lr, bf16, max_len across the three text/classify/vl configs)
changed across commits without any receipt recording what was actually used. Each
region's `config` block already stamps its own dataclass verbatim, which technically
carries these -- but only for the fields that dataclass happens to define, under whatever
name it chose, and only for a reader who already knows which five fields matter and where
to look for them across three different config shapes. `trainer_defaults` makes the five
load-bearing ones a fixed, named contract: present in every receipt, `None` (not a
silently-omitted key) for a field this run's config genuinely has no concept of --
`VLPretrainConfig` trains images, not tokenized text, so it has no `max_len`, and it
always runs images through the encoder in fp32 rather than switching on a `bf16` flag.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

# The five fields the PR #5 merge review flagged as having drifted, across three
# training-region config dataclasses, with no receipt ever recording the change.
TRAINER_DEFAULT_FIELDS = ("steps", "batch_size", "lr", "bf16", "max_len")

#: The metric-naming/battery/pooling unification stamp every receipt this project
#: writes now carries, BESIDE its existing envelope schema (`model-pipeline-receipt/v1`
#: for eval receipts) or native schema (`csd-pretrain-receipt/v1` for training
#: receipts) -- envelope/native schema versions are unchanged by this. `v1` (the
#: absence of this key, on a receipt written before it existed) named metrics with the
#: mixed, sometimes-ambiguous names this project used to ship (`effective_rank`,
#: `map`, `precision@10`, an unnamed `drop`); `v2` is the unified table this repo's
#: `g7-latent-eval-metrics.md` §3.1 specifies -- every renamed field, plus `battery_id`
#: and `pooling` recorded per metric group so two numbers are never diffed across
#: batteries or pools by accident (see `g7-latent-eval-metrics.md` §3.3's refuse
#: predicate). A reader must never infer `v1` vs `v2` from field *names* alone --
#: `metrics_schema` is the one place that says which table applied.
METRICS_SCHEMA_V2 = "csd-metrics/v2"


def trainer_defaults(cfg: Any) -> dict[str, Any]:
    """Pull :data:`TRAINER_DEFAULT_FIELDS` off `cfg` by name, `None` where absent.

    `getattr(cfg, name, None)` rather than `asdict(cfg)[name]`: `cfg` is any of
    `PretrainConfig`, `ClassifyPretrainConfig` or `VLPretrainConfig`, and the last of
    those defines only three of the five (no `bf16`, no `max_len`). `None` for those two
    says "this architecture has no such setting" -- distinguishable, by a reader of the
    receipt, from a value that was simply forgotten.
    """
    return {name: getattr(cfg, name, None) for name in TRAINER_DEFAULT_FIELDS}


def capture_code_revision(repo_root: Path | None = None) -> dict[str, Any]:
    """The git SHA, branch and dirty state of the code that is about to write a receipt.

    Runs `git` as a subprocess rather than importing a git library, because the only
    thing this needs is three plumbing commands and every dev/CI environment already has
    a working `git` on PATH. `repo_root` is passed as `cwd`; git resolves the repository
    root by walking up from there itself (this also does the right thing inside a
    worktree, where `git rev-parse HEAD` reports the WORKTREE's own HEAD, not the main
    checkout's), so this does not need to locate `.git` itself.

    Strips every `GIT_*` variable from the subprocess environment before running.
    `cwd`-based discovery is only what actually happens when NOTHING already tells git
    which repository to use -- but `GIT_DIR` (and `GIT_WORK_TREE`, `GIT_INDEX_FILE`, ...)
    override that discovery outright, and git sets them in ITS OWN environment while
    running a hook so the hook's own nested `git` calls resolve unambiguously. That
    leaks to every subprocess the hook's script spawns, this one included: this
    project's OWN `.githooks/pre-push` -> `scripts/ci_local.sh` -> `pytest` -> this
    function, called with a `repo_root` that is deliberately NOT a git repository (the
    "outside a git checkout" test case) -- silently reported THIS repo's real SHA and
    branch instead of the honest "unknown" fallback, because `GIT_DIR` set by the outer
    hook process was still in the environment `subprocess.run` inherited by default.
    Passing an explicit, `GIT_*`-free `env` closes that: `cwd` is then the ONLY thing
    telling git which repository to look at, exactly as this function's docstring above
    already claims.

    Never raises. If `git` is missing, this is not a git checkout, or any command fails
    or times out, the fallback below is returned instead -- explicit "unknown" values
    with `dirty` forced `True` (an unverifiable revision must never be recorded as
    verified-clean), so a receipt written outside a git checkout still carries a
    `code_revision` block, just an honestly uninformative one, rather than this function
    raising and the caller having to decide whether that is fatal.

    `describe` (`git describe --tags --always --dirty`) is captured SEPARATELY from the
    three commands above and can degrade on its own: a repository with no tags reachable
    from HEAD makes `git describe` exit non-zero even though `rev-parse` and `status`
    both succeeded, and treating that as a total failure would replace a perfectly good
    sha with "unknown". It is recorded because a sha alone does not tell a reader WHERE
    in the history a receipt sits -- "v0.2.0-14-gf48fd8a-dirty" does, at a glance, which
    is the whole job of a provenance block someone reads months later.
    """
    root = repo_root or Path(__file__).resolve().parent
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    try:
        # S607: `git` is resolved from PATH deliberately -- every dev/CI environment on
        # this fleet has one, its location varies host to host, and this reads local
        # repository state rather than crossing a trust boundary.
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=root,
            env=clean_env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607
            cwd=root,
            env=clean_env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            cwd=root,
            env=clean_env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        describe = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],  # noqa: S607
            cwd=root,
            env=clean_env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": "unknown", "dirty": True, "branch": "unknown", "describe": "unknown"}

    if sha.returncode != 0 or branch.returncode != 0 or status.returncode != 0:
        return {"git_sha": "unknown", "dirty": True, "branch": "unknown", "describe": "unknown"}

    return {
        "git_sha": sha.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
        "branch": branch.stdout.strip(),
        # Degrades alone: no reachable tag means a non-zero exit here while every other
        # command succeeded. See this function's docstring.
        "describe": describe.stdout.strip() if describe.returncode == 0 else "unknown",
    }


def write_receipt(
    receipt: dict[str, Any],
    out_dir: Path,
    filename: str,
    *,
    repo_root: Path | None = None,
    capture: Callable[[Path | None], dict[str, Any] | None] = capture_code_revision,
) -> Path:
    """Stamp `code_revision` onto `receipt` and write it to `out_dir/filename`.

    Mutates and returns `receipt` unchanged in every other respect -- callers keep using
    the same dict afterward, as every existing call site already does for
    `receipt["receipt_path"]`.

    Raises:
        RuntimeError: If `capture` returns a falsy value (`None` or `{}`). The real
            `capture_code_revision` never does this -- it falls back to an "unknown"
            block rather than returning `None`, see its own docstring -- so a falsy
            return here means the CAPTURE MECHANISM ITSELF is broken (the shape a test
            proves by monkeypatching `capture`), not merely that git was unavailable.
            Either way, writing a receipt with no `code_revision` block at all is refused
            rather than silently produced.
    """
    revision = capture(repo_root)
    if not revision:
        raise RuntimeError(
            "write_receipt: code_revision capture returned nothing -- refusing to write "
            f"{filename!r} with no code_revision block"
        )
    receipt["code_revision"] = revision
    # Every receipt this helper writes gets the metrics-schema stamp unconditionally --
    # same "always overwrite, never merely default" treatment as code_revision above,
    # so a caller cannot accidentally ship a receipt still claiming the v1 table by
    # passing in a stale dict. `pretrain_region` and `quantize_text_region` both route
    # through this one function, so this is the single place that guarantees it rather
    # than three call sites each remembering to stamp it themselves.
    receipt["metrics_schema"] = METRICS_SCHEMA_V2

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt["receipt_path"] = str(path)
    return path
