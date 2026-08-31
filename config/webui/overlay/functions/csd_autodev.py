"""
title: CSD autodev lab tools
author: CogSynDelta
version: 0.1.0
license: MIT
description: OWUI Tools wrapping lab :9118 (git/forgejo/steer/apply/gpu). No cloud. Identity autodev.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from pydantic import BaseModel, Field

ALLOW_REPOS = frozenset({"tzervas/CogSynDelta", "tzervas/memory-gate"})
ALLOW_PREFIXES = ("src/", "tests/", "docs/")
PROTECTED_WT = frozenset({"kang-main-wip", "memory-gate"})


def _headers(token: str, auth: bool) -> dict[str, str]:
    h = {
        "User-Agent": "csd-webui-overlay",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if auth and token:
        h["Authorization"] = "Bearer " + token
    return h


def _scrub(text: str) -> str:
    """Drop bearer-looking tokens from error strings. Why: never print keys."""
    low = text.lower()
    if "bearer " in low or "token=" in low:
        return "[redacted error]"
    return text[:300]


def _as_args(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x).strip()]
        return [p for p in text.split() if p]
    return []


def _repo(raw: str) -> str | None:
    name = (raw or "").strip()
    if not name or "github.com" in name.lower():
        return None
    if "/" not in name:
        name = f"tzervas/{name}"
    if name not in ALLOW_REPOS:
        return None
    return name


class Tools:
    class Valves(BaseModel):
        lab_base: str = Field(
            default="http://192.168.1.98:9118",
            description="CSD lab HTTP (prime :9118; homelab /lab proxies /api). LAN only.",
        )
        apply_token: str = Field(
            default="",
            description="Bearer matching csd-apply.token. Never log this value.",
        )
        timeout_s: float = Field(default=12.0, ge=1.0, le=120.0)

    def __init__(self) -> None:
        self.valves = self.Valves()

    def _base(self) -> str:
        return (self.valves.lab_base or "").rstrip("/")

    def _call(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        query: dict[str, Any] | None = None,
        auth: bool = False,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """JSON GET/POST to lab. Why: OWUI cannot exec prime scripts; lab wraps them."""
        url = self._base() + path
        if query:
            q = {k: str(v) for k, v in query.items() if v is not None and v != ""}
            if q:
                url += ("&" if "?" in url else "?") + urllib.parse.urlencode(q)
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers=_headers(self.valves.apply_token, auth),
        )
        wait = float(timeout if timeout is not None else self.valves.timeout_s)
        try:
            with urllib.request.urlopen(req, timeout=wait) as resp:
                raw = resp.read().decode("utf-8", errors="replace") or "{}"
                try:
                    body: Any = json.loads(raw)
                except json.JSONDecodeError:
                    body = {"raw": raw[:400]}
                if not isinstance(body, dict):
                    return {"ok": False, "http": resp.status, "notes": "non-object"}
                body.setdefault("http", resp.status)
                body.setdefault("ok", resp.status in {200, 201, 202, 204})
                return body
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")[:400]
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                body = {}
            if not isinstance(body, dict):
                body = {}
            body["http"] = exc.code
            body.setdefault("ok", False)
            body.setdefault("notes", _scrub(raw or str(exc)))
            return body
        except Exception as exc:
            # Fail closed; chat must not stall if lab is down.
            return {"ok": False, "http": 0, "notes": _scrub(str(exc))}

    def _get_or_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        rec = self._call("GET", path, query=payload)
        if rec.get("http") == 404:
            rec = self._call("POST", path, payload)
        return rec

    def _forbid_github(self, *chunks: str) -> dict[str, Any] | None:
        blob = " ".join(chunks).lower()
        if "github.com" in blob or "api.github.com" in blob:
            return {
                "ok": False,
                "notes": "GitHub refused; Forgejo git.vectorweight.com only",
            }
        return None

    def csd_autodev_git(self, args: list[str] | str) -> dict[str, Any]:
        """Lab JSON wrap of scripts/csd-autodev-git. TOKEN=git/autodev; never operator admin."""
        argv = _as_args(args)
        blocked = self._forbid_github(*argv)
        if blocked:
            return blocked
        joined = " ".join(argv).lower()
        if "csd-autodev-provision" in joined or "cabal-forgejo-admin" in joined:
            return {"ok": False, "notes": "refusing provision/admin wrap"}
        rec = self._call("POST", "/api/autodev/git", {"args": argv})
        rec.setdefault("rc", 1 if not rec.get("ok") else 0)
        rec.setdefault("notes", "lab wrap of ./scripts/csd-autodev-git")
        rec.setdefault("evidence", "csd_autodev_git")
        return rec

    def csd_autodev_forgejo_whoami(self) -> dict[str, Any]:
        """Forgejo whoami as autodev. 403 /user is expected; memory_gate 200 required."""
        rec = self._get_or_post("/api/autodev/whoami", {})
        rec.setdefault("notes", "identity autodev only")
        rec.setdefault("evidence", "csd_autodev_forgejo_whoami")
        return rec

    def csd_autodev_forgejo_prs(
        self, repo: str, state: str = "open", limit: int = 20
    ) -> dict[str, Any]:
        """List Forgejo PRs. Allowlist tzervas/CogSynDelta and tzervas/memory-gate."""
        blocked = self._forbid_github(repo)
        if blocked:
            return blocked
        name = _repo(repo)
        if name is None:
            return {"ok": False, "http": 400, "notes": "repo not allowlisted"}
        rec = self._get_or_post(
            "/api/autodev/prs",
            {"repo": name, "state": state or "open", "limit": int(limit or 20)},
        )
        rec.setdefault("repo", name)
        rec.setdefault("pulls", rec.get("pulls") or [])
        rec.setdefault("evidence", "csd_autodev_forgejo_prs")
        return rec

    def csd_autodev_forgejo_pr_create(
        self,
        repo: str,
        title: str,
        head: str,
        base: str = "main",
        body: str = "",
    ) -> dict[str, Any]:
        """Open a Forgejo PR as autodev. No GitHub. No Grok git tool."""
        blocked = self._forbid_github(repo, head, base)
        if blocked:
            return blocked
        name = _repo(repo)
        if name is None:
            return {"ok": False, "http": 400, "notes": "repo not allowlisted"}
        rec = self._call(
            "POST",
            "/api/autodev/pr-create",
            {
                "repo": name,
                "title": title,
                "head": head,
                "base": base or "main",
                "body": body or "",
            },
        )
        rec.setdefault("evidence", "csd_autodev_forgejo_pr_create")
        return rec

    def csd_autodev_forgejo_status(self, repo: str, sha: str) -> dict[str, Any]:
        """Combined Forgejo commit status. CLI exists; this is the lab HTTP wrap."""
        blocked = self._forbid_github(repo, sha)
        if blocked:
            return blocked
        name = _repo(repo)
        if name is None:
            return {"ok": False, "http": 400, "notes": "repo not allowlisted"}
        rec = self._get_or_post("/api/autodev/status", {"repo": name, "sha": sha})
        rec.setdefault("checks", rec.get("checks") or [])
        rec.setdefault("evidence", "csd_autodev_forgejo_status")
        return rec

    def csd_autodev_forgejo_wait(
        self, repo: str, sha: str, timeout: int = 90, poll: int = 5
    ) -> dict[str, Any]:
        """Poll combined status. Does not merge. Honest red stays red."""
        blocked = self._forbid_github(repo, sha)
        if blocked:
            return blocked
        name = _repo(repo)
        if name is None:
            return {"ok": False, "http": 400, "notes": "repo not allowlisted"}
        wait = max(5, min(int(timeout or 90), 120))
        rec = self._call(
            "POST",
            "/api/autodev/wait",
            {
                "repo": name,
                "sha": sha,
                "timeout": wait,
                "poll": max(1, min(int(poll or 5), 30)),
            },
            timeout=float(wait + 5),
        )
        rec.setdefault("notes", "does not merge; skip theatre is not green")
        rec.setdefault("evidence", "csd_autodev_forgejo_wait")
        return rec

    def csd_autodev_forgejo_comment(
        self, repo: str, number: int, body: str
    ) -> dict[str, Any]:
        """PR comment as autodev. Lab JSON wrap. No OWUI overlay of GitHub."""
        blocked = self._forbid_github(repo, body)
        if blocked:
            return blocked
        name = _repo(repo)
        if name is None:
            return {"ok": False, "http": 400, "notes": "repo not allowlisted"}
        rec = self._call(
            "POST",
            "/api/autodev/comment",
            {"repo": name, "number": int(number), "body": body},
        )
        rec.setdefault("evidence", "csd_autodev_forgejo_comment")
        return rec

    def csd_autodev_merge_gate(
        self, repo: str, sha: str, number: int
    ) -> dict[str, Any]:
        """Merge-when-green gate. Skip or missing runner is not green. Does not merge PR 1."""
        blocked = self._forbid_github(repo, sha)
        if blocked:
            return blocked
        name = _repo(repo)
        if name is None:
            return {"ok": False, "http": 400, "notes": "repo not allowlisted"}
        rec = self._call(
            "POST",
            "/api/autodev/merge-gate",
            {"repo": name, "sha": sha, "number": int(number)},
        )
        if rec.get("http") == 404:
            rec = {
                "ok": False,
                "http": 404,
                "state": "missing",
                "merged": False,
                "required_ran": False,
                "required_succeeded": False,
                "notes": (
                    "merge-gate cmd missing; skip theatre is not green; "
                    "CogSynDelta PR 1 waits for required checks ran and succeeded"
                ),
            }
        rec.setdefault("merged", False)
        rec.setdefault("notes", "does not merge; required checks must have run")
        rec.setdefault("evidence", "csd_autodev_merge_gate")
        return rec

    def csd_autodev_loop(self, mode: str) -> dict[str, Any]:
        """HTTP JSON wrap of csd-autodev-loop --once/--worker. Does not wrap provision."""
        kind = (mode or "").strip().lower()
        if kind not in {"once", "worker"}:
            return {"ok": False, "notes": "mode must be once|worker"}
        rec = self._call("POST", "/api/autodev/loop", {"mode": kind})
        rec.setdefault(
            "notes",
            "worker belongs on systemd csd-autodev-loop.service; overlay does not daemonize",
        )
        rec.setdefault("evidence", "csd_autodev_loop")
        return rec

    def csd_autodev_priority(self, action: str) -> dict[str, Any]:
        """CSD_STEER autodev_priority on/off/status. Mask Comfy. Never pause 3090 LocalAI."""
        act = (action or "").strip().lower()
        if act not in {"on", "off", "status"}:
            return {"ok": False, "notes": "action must be on|off|status"}
        if act == "status":
            snap = self._call("GET", "/api/status")
            if not snap.get("ok") and "steer" not in snap:
                snap.setdefault("ok", False)
                snap.setdefault("evidence", "csd_autodev_priority")
                return snap
            return {
                "ok": True,
                "steer": snap.get("steer") or {},
                "comfy": snap.get("gpu5080_comfy"),
                "webui": "homelab",
                "notes": "never pause 3090 LocalAI; Comfy stays masked",
                "evidence": "csd_autodev_priority",
            }
        rec = self._call("POST", "/api/autodev/priority", {"action": act})
        if rec.get("http") == 404:
            rec = self._call(
                "POST", "/api/steer", {"autodev_priority": act == "on"}
            )
            rec.setdefault("notes", "priority via POST /api/steer fallback")
        rec.setdefault("evidence", "csd_autodev_priority")
        return rec

    def csd_gpu_plan(self) -> dict[str, Any]:
        """GPU plan from lab. Nested in GET /api/status when /api/gpu-plan is absent."""
        rec = self._call("GET", "/api/gpu-plan")
        if "akula-prime" in rec and "gpu5080" in rec:
            rec.setdefault("evidence", "csd_gpu_plan")
            return rec
        snap = self._call("GET", "/api/status")
        if "prime_smi" not in snap and not snap.get("ok"):
            rec.setdefault("evidence", "csd_gpu_plan")
            rec.setdefault("notes", rec.get("notes") or "lab gpu-plan missing")
            return rec
        feed = snap.get("feed") or {}
        steer = snap.get("steer") or {}
        rec = {
            "ok": True,
            "http": snap.get("http", 200),
            "autodev_priority": bool(steer.get("autodev_priority")),
            "akula-prime": {
                "smi": snap.get("prime_smi"),
                "loaded": (feed.get("prime") or {}).get("loaded"),
            },
            "gpu5080": {
                "smi": snap.get("gpu5080_smi"),
                "lock": snap.get("gpu5080_lock"),
                "comfy": snap.get("gpu5080_comfy"),
                "loaded": (feed.get("gpu5080") or {}).get("loaded"),
            },
            "notes": "synthesized from GET /api/status; fuser-observe lock",
            "evidence": "csd_gpu_plan",
        }
        return rec

    def gpu5080_lock_observe(self) -> dict[str, Any]:
        """Observe gpu5080.lock only. Acquire/release lives in akula with-gpu-5080."""
        rec = self._call("GET", "/api/gpu5080-lock")
        if rec.get("lock") is not None and rec.get("comfy") is not None:
            rec.setdefault("evidence", "gpu5080_lock_observe")
            return rec
        snap = self._call("GET", "/api/status")
        if "gpu5080_lock" not in snap and not snap.get("ok"):
            rec.setdefault("evidence", "gpu5080_lock_observe")
            rec.setdefault("notes", rec.get("notes") or "lab lock route missing")
            return rec
        lock = str(snap.get("gpu5080_lock") or "")
        comfy = str(snap.get("gpu5080_comfy") or "")
        return {
            "ok": True,
            "http": snap.get("http", 200),
            "lock": lock,
            "comfy": comfy,
            "helper_ok": "idle" in lock.lower() and "masked" in comfy.lower(),
            "notes": "observe only; do not acquire/release from OWUI",
            "evidence": "gpu5080_lock_observe",
        }

    def csd_lab_steer(
        self,
        pause: bool | None = None,
        note: str = "",
        next_goal: str = "",
        autodev_priority: bool | None = None,
    ) -> dict[str, Any]:
        """POST /api/steer (CSD_STEER). Does not pause 3090 LocalAI."""
        payload: dict[str, Any] = {}
        if pause is not None:
            payload["pause"] = bool(pause)
        if note:
            payload["note"] = note
        if next_goal:
            payload["next_goal"] = next_goal
        if autodev_priority is not None:
            payload["autodev_priority"] = bool(autodev_priority)
        rec = self._call("POST", "/api/steer", payload)
        rec.setdefault("evidence", "csd_lab_steer")
        rec.setdefault("notes", "CSD_STEER; never docker-stop akula-localai")
        return rec

    def csd_lab_apply(self, worktree: str, path: str, content: str) -> dict[str, Any]:
        """POST /api/apply Bearer csd-apply.token. Prefixes src/tests/docs. No kang-main-wip."""
        wt = (worktree or "").strip()
        rel = (path or "").lstrip("/")
        if wt in PROTECTED_WT or "kang-main-wip" in wt or "kang-main-wip" in rel:
            return {
                "ok": False,
                "error": "refusing kang-main-wip",
                "notes": "protected worktree",
                "evidence": "csd_lab_apply",
            }
        if ".." in rel.split("/") or not rel.startswith(ALLOW_PREFIXES):
            return {
                "ok": False,
                "error": f"path not allowed: {rel}",
                "notes": "allow prefixes src/ tests/ docs/",
                "evidence": "csd_lab_apply",
            }
        rec = self._call(
            "POST",
            "/api/apply",
            {"worktree": wt, "path": rel, "content": content or ""},
            auth=True,
        )
        rec.setdefault("evidence", "csd_lab_apply")
        rec.setdefault("notes", "HTTP apply; no SSH")
        return rec

    def akula_notes_sandbox(self) -> dict[str, Any]:
        """Self-hosted JupyterLab REPL. LAN only. Not in akula MEDIA_SPECS sqlite."""
        return {
            "ok": True,
            "ui": "https://notes.vectorweight.com/lab",
            "jupyter": "http://172.32.0.1:8086",
            "kernels": ["python3", "typescript", "bash"],
            "notes": (
                "Open WebUI interpreter engine jupyter. Field Notes stay at "
                "pfn.vectorweight.com. Do not send code to cloud kernels."
            ),
            "evidence": "akula_notes_sandbox",
        }
