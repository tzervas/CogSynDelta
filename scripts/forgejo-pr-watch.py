#!/usr/bin/env python3
"""Watch Forgejo PR combined status. Print one line per transition.

Monitor-safe: stdout is only DONE / FAILED / CANCELLED. No progress spam.
Uses TOKEN from the environment (secret exec). Never argv.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("CSD_FORGEJO_URL", "https://git.vectorweight.com").rstrip("/")
REPOS = os.environ.get("CSD_WATCH_REPOS", "tzervas/memory-gate,tzervas/CogSynDelta")
STATE = Path(
    os.environ.get(
        "CSD_WATCH_STATE",
        str(Path.home() / ".cache" / "csd-forgejo-pr-watch.json"),
    )
)
SLEEP = int(os.environ.get("CSD_WATCH_SLEEP_S", "30"))


def get(path: str, token: str) -> object:
    req = urllib.request.Request(
        f"{BASE}/api/v1{path}",
        headers={"Authorization": f"token {token}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def snapshot(token: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for repo in [r.strip() for r in REPOS.split(",") if r.strip()]:
        prs = get(f"/repos/{repo}/pulls?state=open&limit=30", token)
        if not isinstance(prs, list):
            continue
        for pr in prs:
            n = pr.get("number")
            sha = str((pr.get("head") or {}).get("sha") or "")[:12]
            try:
                st = get(f"/repos/{repo}/commits/{pr['head']['sha']}/status", token)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError):
                continue
            combined = str(st.get("state") or "unknown")
            mergeable = pr.get("mergeable")
            key = f"{repo}#{n}"
            out[key] = f"{combined} sha={sha} mergeable={mergeable}"
    return out


def emit(kind: str, key: str, detail: str) -> None:
    # Exactly one token the monitor will wake on.
    print(f"{kind} {key} {detail}", flush=True)


def main() -> int:
    token = os.environ.get("TOKEN") or ""
    if not token:
        print("FAILED missing TOKEN", flush=True)
        return 2
    prev: dict[str, str] = {}
    if STATE.is_file():
        try:
            prev = json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}
    primed = bool(prev)
    while True:
        try:
            now = snapshot(token)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            # Stay up; do not wake on transient API blips.
            time.sleep(SLEEP)
            _ = exc
            continue
        if not primed:
            STATE.parent.mkdir(parents=True, exist_ok=True)
            STATE.write_text(json.dumps(now, indent=0) + "\n", encoding="utf-8")
            prev = now
            primed = True
            time.sleep(SLEEP)
            continue
        for key, detail in now.items():
            old = prev.get(key)
            if old == detail:
                continue
            combined = detail.split(" ", 1)[0]
            if combined == "success":
                emit("DONE", key, detail)
            elif combined in {"failure", "error"}:
                emit("FAILED", key, detail)
            # pending/unknown: record but do not wake
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(now, indent=0) + "\n", encoding="utf-8")
        prev = now
        time.sleep(SLEEP)


if __name__ == "__main__":
    raise SystemExit(main())
