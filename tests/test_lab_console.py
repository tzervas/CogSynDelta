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
    monkeypatch.setenv("CSD_GROK_NEED", str(tmp_path / "csd-need-grok.json"))
    monkeypatch.setenv("CSD_LAB_BIND", "192.168.1.98")
    vault = tmp_path / "csd-vault"
    (vault / "hf").mkdir(parents=True)
    monkeypatch.setenv("CSD_VAULT", str(vault))
    monkeypatch.delenv("SECRET_VAULT", raising=False)
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


def test_cluster_snapshot_includes_1080ti_guest_ip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cluster snapshot always exposes gpu5080 `.251` and 1080 Ti guest_ip."""
    mod = load_lab(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "ssh_5080", lambda _cmd: "")
    rec = mod.cluster_snapshot()
    assert rec["gpu5080"]["ip"] == "192.168.1.251"
    ids = [b["id"] for b in rec["backends"]]
    assert ids == ["akula-prime", "gpu5080", "gpu5080-1080ti"]
    assert rec["count"] == 3
    assert rec["groups"] == ["csd-autodev", "akula-rag"]
    g = rec["gpu5080_1080ti"]
    assert g["host"] == "gpu5080"
    assert g["host_ip"] == "192.168.1.251"
    assert "guest_ip" in g
    assert g["guest_ip"] == "192.168.1.243"
    assert g["live"] is True
    assert g["group"] == "akula-rag"
    assert g["path"] == "lab.gpu5080.index.1080ti"
    assert g["rag"] == "retrieve-index-light"


def test_cluster_snapshot_parses_guest_ip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse a real virsh domifaddr IPv4; never treat `.251` as the guest."""
    mod = load_lab(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "ssh_5080", lambda _cmd: " vnet0  ipv4  192.168.1.187/24")
    rec = mod.cluster_snapshot()
    assert rec["gpu5080_1080ti"]["guest_ip"] == "192.168.1.187"
    assert rec["gpu5080_1080ti"]["live"] is True


def test_steer_post_next_goal_not_p1_08(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /api/steer writes next_goal P1-09 (remaining closeable after P1-08)."""
    mod = load_lab(tmp_path, monkeypatch)
    code, rec = mod.handle_lab("POST", "/api/steer", {"next_goal": "P1-09"})
    assert code == 200
    assert rec["next_goal"] == "P1-09"
    got = json.loads((tmp_path / "csd-steer.json").read_text(encoding="utf-8"))
    assert got["next_goal"] == "P1-09"
    code, rec = mod.handle_lab("GET", "/api/steer", {})
    assert code == 200
    assert rec["next_goal"] == "P1-09"


def test_steer_post_region_pretrain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /api/steer writes next_goal region-pretrain (one live PoC region)."""
    mod = load_lab(tmp_path, monkeypatch)
    code, rec = mod.handle_lab(
        "POST",
        "/api/steer",
        {
            "next_goal": "region-pretrain",
            "pause": False,
            "note": "LatentVAE WikiText-2 train; failing test first",
        },
    )
    assert code == 200
    assert rec["next_goal"] == "region-pretrain"
    got = json.loads((tmp_path / "csd-steer.json").read_text(encoding="utf-8"))
    assert got["next_goal"] == "region-pretrain"
    assert "region-pretrain" in got["next_goal"]
    code, rec = mod.handle_lab("GET", "/api/steer", {})
    assert code == 200
    assert rec["next_goal"] == "region-pretrain"
    code, goals = mod.handle_lab("GET", "/api/goals", {})
    assert code == 200
    assert goals["next_goal"] == "region-pretrain"
    assert any(row["id"] == "region-pretrain" for row in goals["todos"])
    assert mod.WORKTREES["region-pretrain"] == mod.ROOT


def test_api_goals_phase1_steer_heartbeat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /api/goals returns PHASE-1 board, next_goal, heartbeat, notes."""
    mod = load_lab(tmp_path, monkeypatch)
    (tmp_path / "csd-steer.json").write_text(
        json.dumps(
            {
                "pause": False,
                "next_goal": "P1-09",
                "note": "retrieve domain isolation",
                "reasoning": "P1-08 merged; next closeable is P1-09",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "hb.json").write_text(
        json.dumps(
            {
                "t": time.time(),
                "pid": 1,
                "wt": str(tmp_path / "p1-09"),
                "last": {"ok": True, "goal": "P1-09", "applied": {"ok": True}},
            }
        ),
        encoding="utf-8",
    )
    code, rec = mod.handle_lab("GET", "/api/goals", {})
    assert code == 200
    assert rec["ok"] is True
    assert rec["next_goal"] == "P1-09"
    assert rec["notes"] == "retrieve domain isolation"
    assert "P1-09" in rec["reasoning"]
    assert rec["heartbeat"]["live"] is True
    assert rec["heartbeat"]["last"]["goal"] == "P1-09"
    ids = [row["id"] for row in rec["phase1"]]
    assert "P1-08" in ids
    assert "P1-09" in ids
    by_id = {row["id"]: row for row in rec["phase1"]}
    assert by_id["P1-09"]["status"] == "next"
    assert by_id["P1-08"]["status"] == "done"
    assert by_id["P1-05"]["status"] == "blocked"
    assert any(t["id"] == "P1-09" for t in rec["todos"])
    assert rec["hf"]["autodev"] is False
    assert rec["hf"]["gap"] == "mint HF"
    assert rec["hf"]["never_copy"] == "gpu/huggingface-token"
    assert "tzervas/cogsyndelta-tiny" in rec["hf"]["repos"]
    assert "tzervas/cogsyndelta-region-stream_vae-tiny" in rec["hf"]["repos"]
    assert rec["scale_ladder"]["ok"] is True
    assert rec["scale_ladder"]["any_green"] is False
    sizes = [row["size"] for row in rec["scale_ladder"]["rungs"]]
    assert sizes[0] == "region_pretrain"
    assert sizes == ["region_pretrain", "router", "tiny_mind", "small", "medium"]
    assert all(row.get("green") is False for row in rec["scale_ladder"]["rungs"])


def test_goals_tab_renders_from_api_goals_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Goals/Todos tab fetches /api/goals and does not iframe Open WebUI."""
    mod = load_lab(tmp_path, monkeypatch)
    page = mod.PAGE
    assert "data-tab=todos" in page
    assert "/api/goals" in page
    assert "function loadGoals" in page or "async function loadGoals" in page
    assert "renderGoals" in page
    todos_start = page.index("id=todos")
    chat_start = page.index("id=chat")
    todos_html = page[todos_start:chat_start]
    assert "ai.vectorweight.com" not in todos_html
    assert "iframe" not in todos_html
    assert "id=gladder" in todos_html
    assert "id=gping" in todos_html
    assert "scale_ladder" in page
    assert "grok_need" in page or "gping" in page


def test_metrics_scale_ladder_gauge_stays_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /metrics gauge is 0 while scale_ladder.json green is false."""
    mod = load_lab(tmp_path, monkeypatch)
    code, rec = mod.handle_lab("GET", "/metrics", {})
    assert code == 200
    text = rec["exposition"]
    assert "csd_scale_ladder_rung_green" in text
    assert 'rung="0",size="region_pretrain"} 0' in text
    assert 'rung="1",size="router"} 0' in text
    assert 'rung="2",size="tiny_mind"} 0' in text
    assert 'rung="3",size="small"} 0' in text
    assert 'rung="4",size="medium"} 0' in text
    assert 'csd_need_grok{identity="autodev"} 0' in text
    assert 'csd_need_grok{identity="autodev"} 1' not in text
    assert "csd_need_grok_mtime_seconds" in text


def test_hf_autodev_present_when_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hf/autodev file in CSD vault clears the mint-HF gap."""
    mod = load_lab(tmp_path, monkeypatch)
    vault = tmp_path / "csd-vault"
    (vault / "hf" / "autodev").write_text("placeholder-not-a-token\n", encoding="utf-8")
    rec = mod.hf_autodev_gap()
    assert rec["autodev"] is True
    assert rec["gap"] is None


def test_hf_autodev_refuses_operator_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never treat ~/.secrets as the CSD vault."""
    mod = load_lab(tmp_path, monkeypatch)
    monkeypatch.setenv("CSD_VAULT", str(Path.home() / ".secrets"))
    rec = mod.hf_autodev_gap()
    assert rec["autodev"] is False
    assert rec["gap"] == "mint HF"
    assert rec["vault"] == "refused-operator-vault"



def test_grok_need_get_idle_and_post_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET idle; POST need=true writes mailbox; transcripts dropped; 2 KiB cap."""
    mod = load_lab(tmp_path, monkeypatch)
    code, rec = mod.handle_lab("GET", "/api/grok-need", {})
    assert code == 200
    assert rec["need"] is False
    assert rec["hosted_grok"] is False
    assert rec["identity"] == "autodev"
    code, rec = mod.handle_lab(
        "POST",
        "/api/grok-need",
        {
            "need": True,
            "question": "Forgejo required pytest failed twice?",
            "evidence": "blocker: CI red; paths: tests/test_x.py; tried: wait x2",
            "goal": "P1-09",
            "transcript": "never dump a chat",
            "stdout": "pytest spew",
        },
    )
    assert code == 200
    assert rec["need"] is True
    assert rec["identity"] == "autodev"
    assert rec["hosted_grok"] is False
    assert "transcript" not in rec
    assert "stdout" not in rec
    raw = (tmp_path / "csd-need-grok.json").read_bytes()
    assert len(raw) <= 2048
    code, rec = mod.handle_lab("GET", "/api/goals", {})
    assert rec["grok_need"]["need"] is True
    code, rec = mod.handle_lab("GET", "/metrics", {})
    text = rec["exposition"]
    assert 'csd_need_grok{identity="autodev"} 1' in text
    mtime_line = [
        ln for ln in text.splitlines() if ln.startswith("csd_need_grok_mtime_seconds")
    ]
    assert mtime_line
    assert float(mtime_line[0].rsplit(" ", 1)[1]) > 0
    code, rec = mod.handle_lab("POST", "/api/grok-need", {"need": False})
    assert rec["need"] is False
    code, rec = mod.handle_lab("GET", "/metrics", {})
    assert 'csd_need_grok{identity="autodev"} 0' in rec["exposition"]


def test_grok_need_post_does_not_spawn_grok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mailbox write must not call hosted grok or GitHub."""
    mod = load_lab(tmp_path, monkeypatch)

    def boom(*_a: object, **_k: object) -> tuple[int, str]:
        raise AssertionError("grok-need must not spawn CLI")

    monkeypatch.setattr(mod, "_run_cli", boom)
    monkeypatch.setattr(mod, "sh", boom)
    code, rec = mod.handle_lab(
        "POST",
        "/api/grok-need",
        {"need": True, "question": "HF mint?", "evidence": "empty", "goal": "P1-15"},
    )
    assert code == 200
    assert rec["need"] is True
    assert rec["hosted_grok"] is False
