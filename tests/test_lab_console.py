"""Unit tests for lab JSON routes wrapping autodev CLIs."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "csd-lab-console"


def load_lab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import csd-lab-console with isolated plan/steer/heartbeat paths."""
    monkeypatch.setenv("CSD_GPU_PLAN", str(tmp_path / "gpu-plan.json"))
    monkeypatch.setenv("CSD_STEER", str(tmp_path / "csd-steer.json"))
    monkeypatch.setenv("CSD_AUTODEV_HEARTBEAT", str(tmp_path / "hb.json"))
    monkeypatch.setenv("CSD_LAB_BIND", "192.168.1.98")
    monkeypatch.delenv("TOKEN", raising=False)
    monkeypatch.delenv("FORGEJO_TOKEN", raising=False)
    loader = importlib.machinery.SourceFileLoader("csd_lab_console", str(SCRIPT))
    spec = importlib.util.spec_from_loader("csd_lab_console", loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _plan() -> dict[str, Any]:
    return {
        "autodev_priority": True,
        "akula-prime": {"name": "RTX 3090 Ti", "free_mib": 6000},
        "gpu5080": {
            "lock": "lock=idle",
            "comfy": "masked",
            "helper_ok": True,
        },
    }


def test_gpu_and_lock_read_plan_without_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = load_lab(tmp_path, monkeypatch)

    def boom(*_a: object, **_k: object) -> tuple[int, str]:
        raise AssertionError("must not spawn CLI when plan file exists")

    monkeypatch.setattr(mod, "_run_cli", boom)
    (tmp_path / "gpu-plan.json").write_text(json.dumps(_plan()), encoding="utf-8")
    code, rec = mod.handle_lab("GET", "/api/gpu", {})
    assert code == 200
    assert rec["akula-prime"]["name"] == "RTX 3090 Ti"
    assert rec["gpu5080"]["comfy"] == "masked"
    code, lock = mod.handle_lab("GET", "/api/gpu/lock", {})
    assert code == 200
    assert lock["lock"] == "lock=idle"
    assert lock["comfy"] == "masked"
    assert lock["helper_ok"] is True


def test_git_refuses_github_and_protected_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    gh = mod.lab_git(["push", "https://github.com/tzervas/CogSynDelta.git", "HEAD"])
    assert gh["ok"] is False
    assert "GitHub" in gh["stdout"]
    main = mod.lab_git(["push", "forgejo", "HEAD:main"])
    assert main["ok"] is False
    assert "main" in main["stdout"]
    code, rec = mod.handle_lab("POST", "/api/git", {"args": ["push", "forgejo", "HEAD:dev"]})
    assert code == 200
    assert rec["ok"] is False


def test_git_wraps_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    called: dict[str, Any] = {}

    def fake(argv: list[str], timeout: int = 90) -> tuple[int, str]:
        called["argv"] = argv
        called["timeout"] = timeout
        return 0, "feat/agent-harness"

    monkeypatch.setattr(mod, "_run_cli", fake)
    rec = mod.lab_git(["status", "-sb"])
    assert rec["ok"] is True
    assert rec["rc"] == 0
    assert rec["stdout"] == "feat/agent-harness"
    assert str(mod.GIT_CLI) == called["argv"][0]


def test_forgejo_prs_and_status_wrap_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    called: list[list[str]] = []

    def fake(argv: list[str], timeout: int = 90) -> tuple[int, str]:
        called.append(argv)
        if "prs" in argv:
            return 0, json.dumps(
                {"http": 200, "repo": "tzervas/CogSynDelta", "pulls": [{"number": 1}]}
            )
        if "status" in argv:
            return 0, json.dumps({"http": 200, "state": "pending", "checks": []})
        return 1, "{}"

    monkeypatch.setattr(mod, "_run_cli", fake)
    code, rec = mod.handle_lab("GET", "/api/forgejo/prs", {"repo": "CogSynDelta", "limit": 5})
    assert code == 200
    assert rec["http"] == 200
    assert rec["pulls"][0]["number"] == 1
    assert "TOKEN=git/autodev" in called[0]
    code, st = mod.handle_lab(
        "POST",
        "/api/forgejo/status",
        {"repo": "tzervas/CogSynDelta", "sha": "abc"},
    )
    assert code == 200
    assert st["state"] == "pending"
    bad = mod.handle_lab("GET", "/api/forgejo/prs", {"repo": "tzervas/evil"})
    assert bad is not None
    assert bad[1].get("error") == "repo not allowlisted"


def test_loop_worker_does_not_spawn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    (tmp_path / "hb.json").write_text(
        json.dumps({"t": time.time(), "last": {"goal": "G-QD", "applied": {"ok": True}}}),
        encoding="utf-8",
    )

    def boom(*_a: object, **_k: object) -> tuple[int, str]:
        raise AssertionError("HTTP must not spawn --worker")

    monkeypatch.setattr(mod, "_run_cli", boom)
    code, rec = mod.handle_lab("POST", "/api/loop", {"mode": "worker"})
    assert code == 200
    assert rec["ok"] is True
    assert rec["goal"] == "G-QD"
    assert "systemd" in rec["notes"]
    assert mod.handle_lab("POST", "/api/provision", {}) is None


def test_steer_get_and_priority_and_bind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    (tmp_path / "csd-steer.json").write_text(
        json.dumps({"pause": False, "note": "hold", "autodev_priority": True}),
        encoding="utf-8",
    )
    code, rec = mod.handle_lab("GET", "/api/steer", {})
    assert code == 200
    assert rec["pause"] is False
    assert rec["autodev_priority"] is True
    pri = mod.lab_priority("nope")
    assert pri["ok"] is False
    assert mod.lab_bind() == "192.168.1.98"
    monkeypatch.setenv("CSD_LAB_BIND", "0.0.0.0")
    with pytest.raises(SystemExit, match="WAN"):
        mod.lab_bind()


def test_apply_allowlist_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    wt = tmp_path / "p1-08"
    (wt / "src").mkdir(parents=True)
    protected = tmp_path / "memory-gate"
    protected.mkdir()
    mod.WORKTREES = {"p1-08": wt}
    mod.PROTECTED_WT = protected
    ok = mod.http_apply("p1-08", "src/foo.py", "x = 1\n")
    assert ok["ok"] is True
    deny = mod.http_apply("p1-08", "README.md", "nope")
    assert deny["ok"] is False
    assert "not allowed" in deny["error"]
    mod.WORKTREES = {"p1-08": protected}
    refuse = mod.http_apply("p1-08", "src/foo.py", "x")
    assert refuse["ok"] is False
    assert "kang-main-wip" in refuse["error"]


def test_comfy_paths_not_captured_by_lab_wrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    assert mod.handle_lab("GET", "/api/comfy/list-workflows", {}) is None
    assert mod.handle_lab("POST", "/api/comfy/queue-prompt", {"prompt": {}}) is None
    assert mod.handle_lab("GET", "/api/status", {}) is None


# --- OD-2: default-deny proxy, never relays client credentials ---------------


def test_apply_refused_before_any_upstream_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/api/apply must never reach the proxy, whether or not it is auth'd.

    This is the exact OD-2 bug: previously a proxying node forwarded a POST
    to /api/apply upstream, client Authorization header verbatim, before
    check_apply_auth ever ran. Prove refusal happens with zero upstream calls.
    Now this is one instance of the general rule (check_api_auth gates every
    /api route, apply included) rather than an apply-specific check.
    """
    mod = load_lab(tmp_path, monkeypatch)
    mod.UPSTREAM = "http://prime.internal:9118"

    def boom(*_a: object, **_k: object) -> tuple[int, bytes]:
        raise AssertionError("must not call upstream for /api/apply")

    monkeypatch.setattr(mod, "proxy_upstream", boom)

    # No apply token configured at all -> the console itself is
    # misconfigured, fail closed with 503 (not 401 -- see check_api_auth).
    code, raw = mod.dispatch_api_post("/api/apply", "Bearer whatever", b'{"path": "src/x.py"}')
    assert code == 503
    assert "CSD_APPLY_TOKEN" in json.loads(raw)["error"]

    # Configure a real token; a wrong bearer must still be refused locally.
    monkeypatch.setenv("CSD_APPLY_TOKEN", "s3cr3t")
    code, raw = mod.dispatch_api_post("/api/apply", "Bearer nope", b'{"path": "src/x.py"}')
    assert code == 401
    assert json.loads(raw) == {"error": "unauthorized"}

    # And /api/apply is excluded from the allowlist as an independent guard.
    assert mod.proxy_eligible("/api/apply") is False


def test_proxy_default_deny_and_no_credential_relay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-allowlisted route is denied; an allowlisted one mints its own token."""
    mod = load_lab(tmp_path, monkeypatch)
    mod.UPSTREAM = "http://prime.internal:9118"
    monkeypatch.setenv("CSD_APPLY_TOKEN", "s3cr3t")
    auth = "Bearer s3cr3t"

    # Non-allowlisted GET and POST routes never reach the network: assert
    # this with monkeypatch's own context manager so the block is undone
    # before the allowlisted-route checks below need the real function.
    # Authenticated (the auth gate runs first regardless -- these calls are
    # about proxy default-deny, not about auth, so they carry a valid token).
    with monkeypatch.context() as denied:

        def boom(*_a: object, **_k: object) -> tuple[int, bytes]:
            raise AssertionError("must not proxy a non-allowlisted route")

        denied.setattr(mod, "proxy_upstream", boom)
        code, _raw = mod.dispatch_api_get("/api/git", "", auth)
        assert code == 404
        code, _raw = mod.dispatch_api_post("/api/git", auth, b"{}")
        assert code == 404
        code, _raw = mod.dispatch_api_post("/api/forgejo/pr-create", auth, b"{}")
        assert code == 404

    captured: dict[str, object] = {}

    class FakeResp:
        status = 200

        def read(self) -> bytes:
            return b"{}"

        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def fake_urlopen(req: object, timeout: int = 45) -> FakeResp:
        captured["headers"] = dict(req.headers)  # type: ignore[attr-defined]
        return FakeResp()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    fake_token = "downstream-scoped-token"  # noqa: S105 -- test fixture, not a real secret
    monkeypatch.setenv("CSD_UPSTREAM_TOKEN", fake_token)
    mod.UPSTREAM_TOKEN = fake_token

    # An allowlisted route proxies -- with a server-minted token, never the
    # caller's own bearer (proxy_upstream() takes no header argument at all --
    # it structurally cannot relay whatever the caller authenticated with to
    # the console with; the caller's Authorization header is consumed by the
    # auth gate and never forwarded).
    code, _raw = mod.dispatch_api_get("/api/status", "", auth)
    assert code == 200
    assert captured["headers"].get("Authorization") == f"Bearer {fake_token}"

    # No upstream token configured -> no Authorization header at all, and
    # crucially never one lifted from a client request.
    captured.clear()
    mod.UPSTREAM_TOKEN = ""
    code, _raw = mod.dispatch_api_get("/api/status", "", auth)
    assert code == 200
    assert "Authorization" not in captured["headers"]


# --- OD-1: server-side protected paths and no-assert refusal in http_apply --


def test_http_apply_refuses_protected_guard_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    wt = tmp_path / "p1-08"
    (wt / "src").mkdir(parents=True)
    mod.WORKTREES = {"p1-08": wt}
    mod.PROTECTED_WT = tmp_path / "memory-gate"

    for rel in (
        "src/cogsyndelta/eval/metrics.py",
        "src/cogsyndelta/regions/pretrain.py",
        "src/cogsyndelta/regions/_checkpoint.py",
        "scripts/csd-train-all.py",
        "tests/test_guards_can_fail.py",
        "tests/test_reserved_corpus_guard.py",
        "tests/test_checkpoint_load_security.py",
        ".github/workflows/ci.yml",
        ".githooks/pre-push",
    ):
        rec = mod.http_apply("p1-08", rel, "poisoned")
        assert rec["ok"] is False, rel
        assert "protected" in rec["error"], rel
        assert not (wt / rel).exists(), rel

    # A neighbouring, non-guarded src/ file is untouched by the guard.
    ok = mod.http_apply("p1-08", "src/cogsyndelta/regions/whatever.py", "x = 1\n")
    assert ok["ok"] is True


def test_http_apply_protected_check_survives_path_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The protected-path check must run on the RESOLVED write target, not
    the client-supplied string. A raw-string check is bypassable: PurePath
    silently collapses "//" and "." segments and a trailing "/", and
    resolve() follows symlinks, so a spelling that looks unprotected (or a
    symlink alias) can still land on a protected file's real path."""
    mod = load_lab(tmp_path, monkeypatch)
    wt = tmp_path / "p1-08"
    guard = wt / "src/cogsyndelta/eval/metrics.py"
    guard.parent.mkdir(parents=True)
    guard.write_text("original\n", encoding="utf-8")
    (wt / "tests").mkdir()
    guard_test = wt / "tests/test_guards_can_fail.py"
    guard_test.write_text("original\n", encoding="utf-8")
    alias = wt / "src/cogsyndelta/eval/alias.py"
    alias.symlink_to(guard)
    mod.WORKTREES = {"p1-08": wt}
    mod.PROTECTED_WT = tmp_path / "memory-gate"

    vectors = [
        "src//cogsyndelta/eval/metrics.py",
        "src/./cogsyndelta/eval/metrics.py",
        "src/cogsyndelta//eval//metrics.py",
        "src/cogsyndelta/eval/metrics.py/",
        "src/cogsyndelta/eval/./metrics.py",
        "tests//test_guards_can_fail.py",
        "src/cogsyndelta/eval/alias.py",  # symlink -> metrics.py
    ]
    for rel in vectors:
        rec = mod.http_apply("p1-08", rel, "canary\n")
        assert rec["ok"] is False, rel
        assert "protected" in rec["error"], rel

    assert guard.read_text(encoding="utf-8") == "original\n"
    assert guard_test.read_text(encoding="utf-8") == "original\n"


def test_http_apply_escaped_worktree_via_dotdot_and_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A "../" escape and a symlink pointing outside the worktree must both
    be refused, and refused before ALLOW_PREFIXES/protected-path checks run
    on a string that no longer describes where the write would land."""
    mod = load_lab(tmp_path, monkeypatch)
    wt = tmp_path / "p1-08"
    (wt / "src").mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("original\n", encoding="utf-8")
    escape_link = wt / "src" / "escape.py"
    escape_link.symlink_to(outside)
    mod.WORKTREES = {"p1-08": wt}
    mod.PROTECTED_WT = tmp_path / "memory-gate"

    dotdot = mod.http_apply("p1-08", "src/../../outside.py", "canary\n")
    assert dotdot["ok"] is False
    assert "escaped worktree" in dotdot["error"]

    symlink_escape = mod.http_apply("p1-08", "src/escape.py", "canary\n")
    assert symlink_escape["ok"] is False
    assert "escaped worktree" in symlink_escape["error"]
    assert outside.read_text(encoding="utf-8") == "original\n"


def test_http_apply_refuses_assertless_test_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    wt = tmp_path / "p1-08"
    (wt / "tests").mkdir(parents=True)
    mod.WORKTREES = {"p1-08": wt}
    mod.PROTECTED_WT = tmp_path / "memory-gate"

    stub = "def test_always_passes():\n    pass\n"
    rec = mod.http_apply("p1-08", "tests/test_stub.py", stub)
    assert rec["ok"] is False
    assert "no assert" in rec["error"]
    assert not (wt / "tests/test_stub.py").exists()

    real = "def test_real():\n    assert 1 == 1\n"
    ok = mod.http_apply("p1-08", "tests/test_stub.py", real)
    assert ok["ok"] is True
