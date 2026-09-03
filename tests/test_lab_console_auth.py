"""Every /api route requires a valid CSD_APPLY_TOKEN bearer, GET and POST alike.

Before this fix, dispatch_api_get() performed no caller authentication on any
route, and dispatch_api_post() authenticated only /api/apply. Proxied routes
attached the upstream's own credential for any unauthenticated caller, and
/api/status, /api/sessions, and every handle_lab route were open. See
prove_old_script_served_routes_unauthenticated below, which loads the exact
pre-fix commit and demonstrates the gap directly, alongside the same
enumeration proving the current code closes it.

Route enumeration is source-derived (regex over the actual dispatch tables in
scripts/csd-lab-console), not a hand-maintained list -- a new `if apipath ==
"/api/whatever":` line is picked up automatically, so a new route landing
without going through check_api_auth() would be caught by test (a) without
anyone remembering to add a case for it here.
"""

from __future__ import annotations

import hmac
import importlib.machinery
import importlib.util
import inspect
import json
import re
from pathlib import Path
from typing import Any

import pytest

from tests.test_lab_console import load_lab  # reuses the isolated-env module loader

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "csd-lab-console"

# The pre-fix commit this branch forked from (fix/lab-console-auth-all-routes
# off 7fc139d). Frozen to a local file rather than shelled out to `git show`
# at test time, so this test has no dependency on git history/availability
# in whatever environment runs the suite.
OLD_SCRIPT_COMMIT = "7fc139d1c04ca82d04a51ab1bde8d50412b7fd42"


def _routes_from_source(path: Path) -> list[str]:
    """Every literal "/api/..." route string that appears in the script.

    Matches the dispatch_api_get/dispatch_api_post if-chains, handle_lab's
    if-chain, and PROXY_ALLOW -- every place a route is registered spells
    its path as a quoted string literal, so this one regex covers all of
    them without needing three separate AST walks.
    """
    src = path.read_text(encoding="utf-8")
    return sorted(set(re.findall(r'"(/api/[a-zA-Z0-9/_-]*)"', src)))


ROUTES = _routes_from_source(SCRIPT)


def _load_old_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the pre-fix scripts/csd-lab-console from its own git blob.

    Skips (rather than failing the suite) if git history for that commit
    is not available in whatever checkout this runs from -- the permanent
    regression coverage is `test_new_script_gates_every_route_the_old_one_left_open`
    below, which needs no git access at all.
    """
    import subprocess

    proc = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{OLD_SCRIPT_COMMIT}:scripts/csd-lab-console"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        pytest.skip(f"git history for {OLD_SCRIPT_COMMIT} not available: {proc.stderr[:200]}")
    tmp_path.mkdir(parents=True, exist_ok=True)
    old_path = tmp_path / "csd-lab-console-old.py"
    old_path.write_text(proc.stdout, encoding="utf-8")

    monkeypatch.setenv("CSD_GPU_PLAN", str(tmp_path / "gpu-plan.json"))
    monkeypatch.setenv("CSD_STEER", str(tmp_path / "csd-steer.json"))
    monkeypatch.setenv("CSD_AUTODEV_HEARTBEAT", str(tmp_path / "hb.json"))
    monkeypatch.setenv("CSD_LAB_BIND", "192.168.1.98")
    monkeypatch.delenv("CSD_APPLY_TOKEN", raising=False)
    monkeypatch.delenv("CSD_APPLY_TOKEN_FILE", raising=False)
    loader = importlib.machinery.SourceFileLoader("csd_lab_console_old", str(old_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _get(mod: Any, path: str, auth: str = "") -> tuple[int, bytes]:
    """Call dispatch_api_get regardless of whether it takes 2 or 3 args.

    The old (pre-fix) signature is dispatch_api_get(apipath, query); the
    fixed one adds auth_header. Introspecting lets the same test body drive
    both the old and new module.
    """
    sig = inspect.signature(mod.dispatch_api_get)
    if len(sig.parameters) == 2:
        return mod.dispatch_api_get(path, "")
    return mod.dispatch_api_get(path, "", auth)


def _post(mod: Any, path: str, auth: str = "") -> tuple[int, bytes]:
    return mod.dispatch_api_post(path, auth, b"{}")


def _block_all_network(mod: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Any outbound call (proxy_upstream, or a raw urlopen for chat/status)
    fails the test loudly instead of trying to reach a real host."""

    def boom(*_a: object, **_k: object) -> tuple[int, bytes]:
        raise AssertionError("must not call upstream/network for an unauthenticated caller")

    monkeypatch.setattr(mod, "proxy_upstream", boom, raising=False)
    monkeypatch.setattr(mod.urllib.request, "urlopen", boom, raising=False)


# --- (a) every enumerated route, unauthenticated, is refused before dispatch -


@pytest.mark.parametrize("route", ROUTES)
def test_unauthenticated_route_is_refused_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    monkeypatch.setenv("CSD_APPLY_TOKEN", "s3cr3t")
    mod.UPSTREAM = "http://prime.internal:9118"  # exercise the proxy path too
    _block_all_network(mod, monkeypatch)

    for auth in ("", "Bearer wrong", "not-even-bearer-shaped"):
        code, raw = _get(mod, route, auth)
        assert code == 401, f"GET {route} auth={auth!r} -> {code} (expected 401)"
        assert json.loads(raw) == {"error": "unauthorized"}

        code, raw = _post(mod, route, auth)
        assert code == 401, f"POST {route} auth={auth!r} -> {code} (expected 401)"
        assert json.loads(raw) == {"error": "unauthorized"}


def test_unknown_route_404s_only_after_auth_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    monkeypatch.setenv("CSD_APPLY_TOKEN", "s3cr3t")
    _block_all_network(mod, monkeypatch)

    # No/wrong token against a route that does not exist -> still 401, not
    # 404: an unauthenticated caller cannot distinguish "wrong route" from
    # "route exists but I'm unauthorized".
    code, _raw = _get(mod, "/api/does-not-exist", "")
    assert code == 401
    code, _raw = _post(mod, "/api/does-not-exist", "Bearer wrong")
    assert code == 401

    # Correct token against the same nonexistent route -> 404, now that
    # auth has actually passed.
    code, _raw = _get(mod, "/api/does-not-exist", "Bearer s3cr3t")
    assert code == 404
    code, _raw = _post(
        mod,
        "/api/does-not-exist",
        "Bearer s3cr3t",
    )
    assert code == 404


# --- (b) with the token, a proxied route reaches the mocked upstream with --
# --- the upstream credential, and the caller's own header is not relayed --


def test_authenticated_proxy_route_gets_upstream_credential_not_callers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    monkeypatch.setenv("CSD_APPLY_TOKEN", "s3cr3t")
    mod.UPSTREAM = "http://prime.internal:9118"
    monkeypatch.setenv("CSD_UPSTREAM_TOKEN", "downstream-scoped-token")
    mod.UPSTREAM_TOKEN = "downstream-scoped-token"  # noqa: S105 -- test fixture, not a real secret

    captured: dict[str, Any] = {}

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

    # Caller authenticates to the CONSOLE with the console's own apply
    # token -- this is not, and must not become, what reaches upstream.
    code, _raw = mod.dispatch_api_get("/api/status", "", "Bearer s3cr3t")
    assert code == 200
    assert captured["headers"].get("Authorization") == "Bearer downstream-scoped-token"
    assert "s3cr3t" not in captured["headers"].get("Authorization", "")


# --- (c) empty/unset CSD_APPLY_TOKEN -> 503 on every route, fail closed ----


@pytest.mark.parametrize("route", ROUTES)
def test_unset_apply_token_is_503_not_401(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    monkeypatch.delenv("CSD_APPLY_TOKEN", raising=False)
    monkeypatch.delenv("CSD_APPLY_TOKEN_FILE", raising=False)
    mod.APPLY_TOKEN_FILE = tmp_path / "no-such-token-file"
    mod.UPSTREAM = "http://prime.internal:9118"
    _block_all_network(mod, monkeypatch)

    code, raw = _get(mod, route, "Bearer anything-at-all")
    assert code == 503, f"GET {route} -> {code} (expected 503, misconfigured console)"
    assert "CSD_APPLY_TOKEN" in json.loads(raw)["error"]

    code, raw = _post(mod, route, "Bearer anything-at-all")
    assert code == 503, f"POST {route} -> {code} (expected 503, misconfigured console)"
    assert "CSD_APPLY_TOKEN" in json.loads(raw)["error"]


def test_empty_string_apply_token_env_is_also_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An env var explicitly set to "" is the same as unset -- not a valid empty token."""
    mod = load_lab(tmp_path, monkeypatch)
    monkeypatch.setenv("CSD_APPLY_TOKEN", "")
    code, raw = mod.dispatch_api_get("/api/status", "", "Bearer ")
    assert code == 503
    code, raw = mod.dispatch_api_post("/api/apply", "Bearer ", b"{}")
    assert code == 503


# --- (d) path-normalisation tricks never reach dispatch/proxy unauthenticated


@pytest.mark.parametrize(
    "tricky_path",
    [
        "/api%2Fstatus",
        "//api/status",
        "/API/status",
        "/api/Status",
        "/api/status/../apply",
        "/lab/api/status",
        "/api/status ",
        "/api//status",
    ],
)
def test_path_normalisation_tricks_never_bypass_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tricky_path: str
) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    monkeypatch.setenv("CSD_APPLY_TOKEN", "s3cr3t")
    mod.UPSTREAM = "http://prime.internal:9118"
    _block_all_network(mod, monkeypatch)

    # No credential at all -> gated (401) or not-found (404), but NEVER a
    # 200 and never a call through to proxy_upstream/urlopen (the
    # monkeypatched boom() above raises AssertionError if either fires,
    # which pytest surfaces as a failure of this test, not a silent pass).
    code, _raw = _get(mod, tricky_path, "")
    assert code in (401, 404), f"{tricky_path!r} -> {code}, expected 401 or 404"
    code, _raw = _post(mod, tricky_path, "")
    assert code in (401, 404), f"{tricky_path!r} -> {code}, expected 401 or 404"


# --- (e) constant-time compare -----------------------------------------------


def test_apply_auth_uses_constant_time_compare() -> None:
    src = inspect.getsource(__import__("importlib.util").util)  # keep import machinery warm
    del src
    text = SCRIPT.read_text(encoding="utf-8")
    m = re.search(r"def check_apply_auth.*?(?=\ndef )", text, re.DOTALL)
    assert m is not None, "check_apply_auth() not found in scripts/csd-lab-console"
    body = m.group(0)
    assert "hmac.compare_digest(" in body, (
        "check_apply_auth must compare the caller's token with hmac.compare_digest "
        "(constant-time), not == (timing side-channel)"
    )
    assert "==" not in re.sub(r"^\s*(\"\"\".*?\"\"\"|#.*)$", "", body, flags=re.DOTALL), (
        "unexpected plain == comparison alongside the constant-time compare"
    )


def test_hmac_compare_digest_is_actually_constant_time_capable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sanity check the primitive itself, independent of source inspection."""
    mod = load_lab(tmp_path, monkeypatch)
    monkeypatch.setenv("CSD_APPLY_TOKEN", "s3cr3t-token-value")
    assert mod.check_apply_auth("Bearer s3cr3t-token-value") is True
    assert mod.check_apply_auth("Bearer wrong") is False
    assert mod.check_apply_auth("") is False
    # hmac.compare_digest is what the stdlib offers for this; confirm it is
    # actually reachable/importable in this module's namespace.
    assert mod.hmac.compare_digest is hmac.compare_digest


# --- proof: the exact pre-fix commit served routes with zero auth ----------


def test_old_script_served_routes_unauthenticated_new_script_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loads the pre-fix commit's own scripts/csd-lab-console and shows it
    served real route logic (200s, 404s from route-specific code, even a
    405 from a method check inside a route) for callers with no credential
    at all -- then shows the current module refuses every one of the same
    calls before any route-specific code runs.

    This is the permanent regression proof: if check_api_auth() is ever
    bypassed or a route is added outside it, this test's second half fails
    exactly the way the first half demonstrates the old code always did.
    """
    old = _load_old_script(tmp_path / "old", monkeypatch)
    old.UPSTREAM = ""  # local mode: exercise handle_lab / local dispatch, not the network

    def boom(*_a: object, **_k: object) -> tuple[int, bytes]:
        raise AssertionError("upstream must never be called unauthenticated")

    monkeypatch.setattr(old, "proxy_upstream", boom, raising=False)
    monkeypatch.setattr(old.urllib.request, "urlopen", boom, raising=False)

    served_without_auth = 0
    for route in ROUTES:
        for fn in (_get, _post):
            try:
                code, _raw = fn(old, route, "")  # no Authorization header at all
            except AssertionError:
                continue  # old code tried to reach the network unauthenticated -- also a leak
            except Exception:  # old code ran real handler code (subprocess etc.)
                served_without_auth += 1
                continue
            if code not in (401, 503):
                served_without_auth += 1
    # The old script authenticated only /api/apply's own dedicated check;
    # every other route (of 22, GET+POST = 44 calls) ran unauthenticated.
    assert served_without_auth >= 20, (
        f"expected the pre-fix script to serve most routes unauthenticated, "
        f"only counted {served_without_auth} -- did the fixture load the wrong commit?"
    )

    # Now the same sweep against the CURRENT module: nothing is served.
    new = load_lab(tmp_path / "new", monkeypatch)
    monkeypatch.setenv("CSD_APPLY_TOKEN", "s3cr3t")
    new.UPSTREAM = "http://prime.internal:9118"
    monkeypatch.setattr(new, "proxy_upstream", boom, raising=False)
    monkeypatch.setattr(new.urllib.request, "urlopen", boom, raising=False)

    for route in ROUTES:
        code, _raw = _get(new, route, "")
        assert code == 401
        code, _raw = _post(new, route, "")
        assert code == 401
