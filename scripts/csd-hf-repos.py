#!/usr/bin/env python3
"""Provision the private HuggingFace repos for CSD and its region submodels.

Idempotent: an existing repo is left alone, never recreated or overwritten.

NAMING
One repo per region, versioned by git revision inside it -- NOT one repo per
(region, scale-rung). The scale ladder's original convention was
``cogsyndelta-region-stream_vae-tiny``, which mints a new repo every time a region is
retrained at a different size and leaves no way to diff two rungs of the same region.
HF repos are git; tags and revisions are the versioning mechanism, and using them keeps a
region's history in one place.

    tzervas/cogsyndelta                  the composed mind
    tzervas/cogsyndelta-region-<name>    one per brain region
    tzervas/cogsyndelta-vl-jepa          the latent visual encoder
    tzervas/cogsyndelta-eval             held-out eval sets (dataset repo)

PRIVACY
private=True on every create, without exception. The operator authorised publishing to
his personal account **as private repos**; public is not in scope. `private` is passed
explicitly rather than relying on an account default, because a default can change under
you and the failure mode is a permanently public model.

ORGS
`POST /api/organizations` returns 403 with this token -- org creation is a web-UI action.
Once an org exists, these repos can be moved with the transfer endpoint and the naming
survives the move unchanged. Run with --owner <org> to provision there instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://huggingface.co/api"

# (repo_type, suffix, description). Suffix is appended to the base name.
REPOS: list[tuple[str, str, str]] = [
    ("model", "", "CogSynDelta — composed multi-region mind"),
    ("model", "-region-code", "CSD region: docstring<->function (CodeSearchNet Python)"),
    (
        "model",
        "-region-compress",
        "CSD region: neighbours stay neighbours in a short latent (STS-B + AllNLI)",
    ),
    ("model", "-region-retrieve", "CSD region: query -> passage rank (FiQA)"),
    ("model", "-region-residual", "CSD region: residual update on the shared [B, D] stream"),
    ("model", "-region-stream-vae", "CSD region: encode / reparameterize / decode"),
    ("model", "-vl-jepa", "CSD latent visual encoder (I-JEPA); emits [B, D] on the shared stream"),
    ("dataset", "-eval", "CSD held-out eval sets — never trained on"),
]


def request(path: str, token: str, method: str = "GET", body: dict | None = None):
    req = urllib.request.Request(
        f"{API}{path}",
        method=method,
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]


def exists(owner: str, name: str, repo_type: str, token: str) -> bool | None:
    """True/False, or None when the check itself failed (do not treat as absent)."""
    kind = "models" if repo_type == "model" else "datasets"
    code, _ = request(f"/{kind}/{owner}/{name}", token)
    if code == 200:
        return True
    if code == 404:
        return False
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", default="tzervas", help="user or org namespace")
    ap.add_argument("--base", default="cogsyndelta")
    ap.add_argument("--apply", action="store_true", help="create; otherwise dry-run")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set", file=sys.stderr)
        return 2

    mode = "APPLY" if args.apply else "DRY-RUN (pass --apply to create)"
    print(f"  owner={args.owner}  base={args.base}  {mode}\n")

    created = skipped = failed = 0
    for repo_type, suffix, description in REPOS:
        name = f"{args.base}{suffix}"
        full = f"{args.owner}/{name}"
        present = exists(args.owner, name, repo_type, token)

        if present is None:
            print(f"  ?      {full:<44} could not verify — skipping rather than risk a duplicate")
            failed += 1
            continue
        if present:
            print(f"  exists {full:<44} ({repo_type})")
            skipped += 1
            continue
        if not args.apply:
            print(f"  WOULD  {full:<44} ({repo_type}, private)")
            continue

        code, body = request(
            "/repos/create",
            token,
            "POST",
            {"name": name, "type": repo_type, "private": True, "organization": None},
        )
        if code in (200, 201):
            print(f"  created {full:<43} ({repo_type}, private)")
            created += 1
        else:
            print(f"  FAILED {full:<44} HTTP {code}: {str(body)[:110]}")
            failed += 1

    print(f"\n  created={created} existing={skipped} failed={failed}")

    if args.apply:
        print("\n  verifying privacy of every repo (a public model repo is not recoverable):")
        for repo_type, suffix, _ in REPOS:
            name = f"{args.base}{suffix}"
            kind = "models" if repo_type == "model" else "datasets"
            code, d = request(f"/{kind}/{args.owner}/{name}", token)
            if code == 200 and isinstance(d, dict):
                flag = "PRIVATE" if d.get("private") else "!! PUBLIC !!"
                print(f"    {flag:<12} {args.owner}/{name}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
