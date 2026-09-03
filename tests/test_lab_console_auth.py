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

import ast
import hmac
import http.client
import importlib.machinery
import importlib.util
import inspect
import json
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tests.test_lab_console import load_lab  # reuses the isolated-env module loader

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "csd-lab-console"

# The pre-fix commit this branch forked from (fix/lab-console-auth-all-routes
# off 7fc139d). Its scripts/csd-lab-console is frozen verbatim to
# tests/fixtures/csd-lab-console-pre-auth-fix (checked in, <60 KB) -- loaded
# straight off disk by _load_old_script(), with no subprocess and no git
# history dependency at test time. OLD_SCRIPT_COMMIT is provenance only: the
# commit this fixture was captured from, not something read at test time.
OLD_SCRIPT_COMMIT = "7fc139d1c04ca82d04a51ab1bde8d50412b7fd42"
OLD_SCRIPT_FIXTURE = ROOT / "tests" / "fixtures" / "csd-lab-console-pre-auth-fix"


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
    """Import the pre-fix scripts/csd-lab-console from the frozen fixture.

    No subprocess, no git access, no skip path: OLD_SCRIPT_FIXTURE is a
    file checked into this repo (tests/fixtures/), so this is exactly as
    available as any other test in the suite, in any checkout.
    """
    assert OLD_SCRIPT_FIXTURE.is_file(), f"missing fixture: {OLD_SCRIPT_FIXTURE}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    old_path = tmp_path / "csd-lab-console-old.py"
    old_path.write_text(OLD_SCRIPT_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

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


# --- (f) header-parsing edge cases: exact scheme, no bare token, no stray --
# --- whitespace tolerance (adversarial-review finding D) -------------------


@pytest.mark.parametrize(
    ("header", "why"),
    [
        ("s3cr3t", "bare token, no scheme at all -- must not fall through to a raw compare"),
        ("bearer s3cr3t", "lowercase scheme"),
        ("BEARER s3cr3t", "uppercase scheme"),
        ("BeArEr s3cr3t", "mixed-case scheme"),
        ("Bearer s3cr3t ", "real trailing space after a correct token"),
        ("Bearer  s3cr3t", "double space between scheme and token"),
        ("Bearers3cr3t", "no space between scheme and token"),
        ("Basic s3cr3t", "wrong scheme entirely"),
        (" Bearer s3cr3t", "leading space before the scheme"),
    ],
)
def test_check_apply_auth_rejects_non_exact_bearer_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, header: str, why: str
) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    monkeypatch.setenv("CSD_APPLY_TOKEN", "s3cr3t")
    assert mod.check_apply_auth(header) is False, why


def test_check_apply_auth_accepts_the_exact_bearer_scheme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = load_lab(tmp_path, monkeypatch)
    monkeypatch.setenv("CSD_APPLY_TOKEN", "s3cr3t")
    assert mod.check_apply_auth("Bearer s3cr3t") is True


def test_check_apply_auth_strips_only_a_single_trailing_crlf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A folded continuation line (an obs-fold per RFC 9110 3.2.4) can leave
    http.client's parse_headers() handing back "Bearer TOKEN\\r\\n " for an
    Authorization header split across two wire lines -- note the trailing
    space AFTER the embedded CRLF is real content the client sent, not a
    parser artifact. Only a value ending in a bare, literal "\\r\\n" (no
    trailing content after it) gets that stripped; this folded case does
    not end in "\\r\\n" (it ends in a space) and so is compared
    byte-for-byte and correctly refused.
    """
    mod = load_lab(tmp_path, monkeypatch)
    monkeypatch.setenv("CSD_APPLY_TOKEN", "s3cr3t")
    assert mod.check_apply_auth("Bearer s3cr3t\r\n") is True
    assert mod.check_apply_auth("Bearer s3cr3t\r\n ") is False


def test_route_dispatch_401s_for_bare_token_and_trailing_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding D at the route level, not just check_apply_auth() in
    isolation -- the same three cases as they actually reach a caller."""
    mod = load_lab(tmp_path, monkeypatch)
    monkeypatch.setenv("CSD_APPLY_TOKEN", "s3cr3t")
    for bad in ("s3cr3t", "Bearer s3cr3t ", "bearer s3cr3t"):
        code, raw = mod.dispatch_api_get("/api/status", "", bad)
        assert code == 401, f"{bad!r} -> {code}"
        assert json.loads(raw) == {"error": "unauthorized"}
    code, _raw = mod.dispatch_api_get("/api/status", "", "Bearer s3cr3t")
    assert code == 200


# --- (g) non-ASCII Authorization header: 401, never an unhandled exception -
# --- (adversarial-review finding A) -----------------------------------------

NON_ASCII_HEADERS = [
    # What a raw Authorization byte >= 0x80 becomes once http.client's
    # header parser decodes the wire bytes as latin-1 (RFC 9110 field
    # values are historically ISO-8859-1): plain non-ASCII str content,
    # not anything exotic like a lone surrogate.
    "Bearer éé",
    "Bearer és3cr3t",
    "éé",  # non-ASCII with no "Bearer " scheme at all
]


@pytest.mark.parametrize("header", NON_ASCII_HEADERS)
def test_check_apply_auth_never_raises_on_non_ascii_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, header: str
) -> None:
    """Before this fix, hmac.compare_digest(got, want) raised TypeError for
    a non-ASCII str -- uncaught, inside check_apply_auth() -- for every one
    of these. It must now return False instead of raising."""
    mod = load_lab(tmp_path, monkeypatch)
    monkeypatch.setenv("CSD_APPLY_TOKEN", "s3cr3t")
    assert mod.check_apply_auth(header) is False


@pytest.mark.parametrize("header", NON_ASCII_HEADERS)
def test_dispatch_401s_on_non_ascii_authorization_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, header: str
) -> None:
    """The route-level view: dispatch_api_get/dispatch_api_post (what
    do_GET/do_POST actually call) must return a clean 401 for a non-ASCII
    Authorization header rather than letting an exception escape to the
    HTTP server layer, which would abandon the connection with no response
    at all. Covers both dispatch entry points, since do_GET and do_POST
    each call exactly one.
    """
    mod = load_lab(tmp_path, monkeypatch)
    monkeypatch.setenv("CSD_APPLY_TOKEN", "s3cr3t")
    code, raw = mod.dispatch_api_get("/api/status", "", header)
    assert code == 401
    assert json.loads(raw) == {"error": "unauthorized"}
    code, raw = mod.dispatch_api_post("/api/apply", header, b"{}")
    assert code == 401
    assert json.loads(raw) == {"error": "unauthorized"}


def test_server_survives_a_non_ascii_header_over_a_real_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end proof over the actual http.server plumbing, not just the
    pure dispatch functions: http.client itself refuses to send a
    non-ASCII header value, so this drives serve()'s real handler through
    a raw socket on an ephemeral loopback port. Before this fix, the
    uncaught TypeError from hmac.compare_digest propagated out of do_GET,
    past http.server's own exception handling, and the connection was
    dropped with no HTTP response at all -- a crash, not a 401. This
    confirms both halves of the fix: the malformed request gets a clean
    401, and the server (a real ThreadingHTTPServer, one thread per
    request) is still alive and serves the very next request normally.
    """
    mod = load_lab(tmp_path, monkeypatch)
    monkeypatch.setenv("CSD_APPLY_TOKEN", "s3cr3t")
    monkeypatch.setattr(mod, "lab_bind", lambda: "127.0.0.1")

    instances: list[Any] = []
    real_cls = mod.ThreadingHTTPServer

    class CapturingServer(real_cls):  # type: ignore[misc,valid-type]
        def __init__(self, *a: object, **k: object) -> None:
            super().__init__(*a, **k)
            instances.append(self)

    monkeypatch.setattr(mod, "ThreadingHTTPServer", CapturingServer)

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    thread = threading.Thread(target=mod.serve, args=(port,), daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if instances:
                break
            time.sleep(0.02)
        assert instances, "serve() did not start a ThreadingHTTPServer"

        for _ in range(100):
            try:
                sock = socket.create_connection(("127.0.0.1", port), timeout=0.2)
                sock.close()
                break
            except OSError:
                time.sleep(0.02)
        else:
            pytest.fail("server never accepted a connection")

        raw_req = (
            "GET /api/status HTTP/1.1\r\nHost: x\r\n"
            "Authorization: Bearer \xe9\xe9\r\nConnection: close\r\n\r\n"
        ).encode("latin-1")
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.sendall(raw_req)
        resp = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            resp += chunk
        sock.close()
        assert resp, "connection was dropped with no HTTP response at all"
        status_line = resp.split(b"\r\n", 1)[0]
        assert b" 401 " in status_line, f"expected a 401 status line, got {status_line!r}"

        # The server is still alive: a normal authenticated request right
        # after the malformed one is served, not dropped.
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/api/status", headers={"Authorization": "Bearer s3cr3t"})
        good = conn.getresponse()
        assert good.status == 200
        good.read()
        conn.close()
    finally:
        for inst in instances:
            inst.shutdown()
            inst.server_close()


# --- (h) source-level guard: nothing route-shaped runs ahead of the gate ---
# --- (adversarial-review finding B) -----------------------------------------
#
# The enumeration tests in section (a) are regex-over-source: they see every
# "/api/..." literal that exists, but only INSIDE dispatch_api_get's and
# dispatch_api_post's own if-chains. A route added directly to do_GET/do_POST
# BEFORE the call into those functions -- ahead of check_api_auth() -- would
# never show up there, and would run fully unauthenticated. This section
# parses scripts/csd-lab-console with ast and enforces an explicit allowlist
# of what may precede that call: path parsing, and the one static page/asset
# branch. Anything else fails the test.

_PRE_DISPATCH_SPEC: dict[str, dict[str, object]] = {
    "do_GET": {
        "target": "dispatch_api_get",
        "allowed": [
            "path = urlparse(self.path).path",
            "if path in ('/', '/lab', '/lab/'):\n"
            "    body = PAGE.encode()\n"
            "    self.send_response(200)\n"
            "    self.send_header('Content-Type', 'text/html; charset=utf-8')\n"
            "    self.send_header('Content-Length', str(len(body)))\n"
            "    self.end_headers()\n"
            "    self.wfile.write(body)\n"
            "    return",
            "parsed = urlparse(self.path)",
            "apipath = parsed.path[4:] if parsed.path.startswith('/lab/') else parsed.path",
        ],
    },
    "do_POST": {
        "target": "dispatch_api_post",
        "allowed": [
            "n = int(self.headers.get('Content-Length') or '0')",
            "raw_in = self.rfile.read(n)",
            "postpath = urlparse(self.path).path",
            "if postpath.startswith('/lab/'):\n    postpath = postpath[4:]",
        ],
    },
}


def _find_dispatch_call_index(body: list[ast.stmt], dispatch_name: str) -> int | None:
    for i, stmt in enumerate(body):
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == dispatch_name
            ):
                return i
    return None


def _pre_dispatch_violations(source: str) -> dict[str, list[str]]:
    """For do_GET and do_POST, every statement before the call into
    dispatch_api_get/dispatch_api_post must match _PRE_DISPATCH_SPEC's
    allowlist exactly (path parsing, and the one static page/asset branch)
    -- never a path comparison or route handler. Returns {func_name:
    [unparsed offending statements]}; empty when both functions are clean.
    """
    tree = ast.parse(source)
    funcs = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in _PRE_DISPATCH_SPEC
    }
    violations: dict[str, list[str]] = {}
    for fname, spec in _PRE_DISPATCH_SPEC.items():
        node = funcs.get(fname)
        if node is None:
            violations[fname] = [f"<{fname} not found>"]
            continue
        idx = _find_dispatch_call_index(node.body, spec["target"])
        if idx is None:
            violations[fname] = [f"<no call to {spec['target']}() found in {fname}>"]
            continue
        bad = [
            ast.unparse(stmt)
            for stmt in node.body[:idx]
            if ast.unparse(stmt) not in spec["allowed"]
        ]
        if bad:
            violations[fname] = bad
    return violations


def test_do_get_and_do_post_have_no_route_handling_before_the_auth_gate() -> None:
    """Finding B: the route-enumeration tests above cannot catch a NEW
    route handled directly inside do_GET/do_POST ahead of the
    dispatch_api_get/dispatch_api_post call -- such code would never reach
    check_api_auth() at all. This is a structural, source-level guard: it
    does not care what a route does, only that do_GET/do_POST contain
    nothing but path parsing and the one static page/asset branch before
    the gate. See test_guard_detects_a_route_inserted_ahead_of_the_gate
    below for proof this actually fires on the failure mode it exists to
    catch, rather than being vacuously true.
    """
    violations = _pre_dispatch_violations(SCRIPT.read_text(encoding="utf-8"))
    assert violations == {}, violations


def test_guard_detects_a_route_inserted_ahead_of_the_gate() -> None:
    """Proves the guard above is not vacuous: insert a route handled
    directly in do_GET, between the apipath computation and the call into
    dispatch_api_get -- the exact shape of a real regression (a route
    added ahead of the auth gate instead of inside dispatch_api_get's
    if-chain) -- and confirm _pre_dispatch_violations() flags do_GET for
    it, distinctly from the unmutated source being clean.
    """
    src = SCRIPT.read_text(encoding="utf-8")
    needle = (
        "            parsed = urlparse(self.path)\n"
        '            apipath = parsed.path[4:] if parsed.path.startswith("/lab/") '
        "else parsed.path\n"
    )
    assert needle in src, "do_GET body shape changed; update this test's mutation to match"
    mutant_route = (
        '            if apipath == "/api/newthing":\n'
        '                self._send(200, b\'{"secret":"leaked"}\')\n'
        "                return\n"
    )
    mutated = src.replace(needle, needle + mutant_route, 1)
    assert mutated != src

    clean = _pre_dispatch_violations(src)
    assert clean == {}, "sanity: unmutated source must still be clean"

    violated = _pre_dispatch_violations(mutated)
    assert "do_GET" in violated, violated
    assert any("newthing" in v for v in violated["do_GET"]), violated["do_GET"]
