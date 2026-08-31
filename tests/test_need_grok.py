"""Ping protocol, Grafana one-shot, and no hosted-Grok scheduler."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PING_DOC = ROOT / "docs" / "program" / "CSD-GROK-PING.md"
RULES = ROOT / "deploy" / "grafana" / "provisioning" / "alerting" / "rules.yaml"
CONTACTS = ROOT / "deploy" / "grafana" / "provisioning" / "alerting" / "contact-points.yaml"
POLICIES = ROOT / "deploy" / "grafana" / "provisioning" / "alerting" / "policies.yaml"
SCRAPE = ROOT / "deploy" / "o11y" / "vm-scrape.yml"
SYSTEMD = ROOT / "deploy" / "systemd"
LOOP = ROOT / "scripts" / "csd-autodev-loop"
ESCALATE = ROOT / "scripts" / "csd-escalate"


def test_ping_protocol_doc_is_operator_wake_not_scheduler() -> None:
    """CSD-GROK-PING.md is the mailbox contract; hosted Grok is not polled."""
    text = PING_DOC.read_text(encoding="utf-8")
    assert "csd-need-grok.json" in text
    assert "/api/grok-need" in text
    assert "≤ ~2 KiB" in text or "2 KiB" in text
    assert "Never" in text and "GitHub" in text
    assert "Do **not** poll hosted Grok" in text or "not a scheduler" in text.lower()
    assert "maintainers@" in text
    assert "CSDNeedGrok" in text
    assert "csd_need_grok_mtime_seconds" in text
    assert "identity" in text and "autodev" in text


def test_grafana_need_grok_pages_maintainers_once() -> None:
    """One Grafana rule; noData OK; contact is maintainers-email; not per tick."""
    rules = RULES.read_text(encoding="utf-8")
    assert "uid: csd-need-grok" in rules
    assert "title: CSDNeedGrok" in rules
    assert "csd_need_grok == 1" in rules
    assert "noDataState: OK" in rules
    assert "for: 2m" in rules
    assert "Do not poll hosted Grok" in rules
    assert "Do not fire on autodev loop ticks" in rules
    contacts = CONTACTS.read_text(encoding="utf-8")
    assert "name: maintainers-email" in contacts
    assert "maintainers@vectorweight.com" in contacts
    policies = POLICIES.read_text(encoding="utf-8")
    assert "receiver: maintainers-email" in policies
    assert "repeat_interval: 4h" in policies


def test_lab_metrics_scrape_job_is_prime_lan_only() -> None:
    """csd-lab scrape is prime :9118; never WAN; never hosted Grok."""
    text = SCRAPE.read_text(encoding="utf-8")
    assert "job_name: csd-lab" in text
    assert "192.168.1.98:9118" in text
    assert "0.0.0.0" not in text
    assert "Do not poll hosted Grok" in text


def test_no_grok_scheduler_unit() -> None:
    """deploy/systemd has autodev loop, not a hosted Grok timer."""
    names = {p.name for p in SYSTEMD.iterdir() if p.is_file()}
    assert "csd-autodev-loop.service" in names
    assert not any("grok" in n.lower() for n in names)
    loop = LOOP.read_text(encoding="utf-8")
    assert "refusing hosted grok" in loop
    assert "never GitHub" in loop
    escalate = ESCALATE.read_text(encoding="utf-8")
    assert "dry-run" in escalate
    assert "CSD_ESCALATE" in escalate



def test_local_python_first_drive_workflow_exists() -> None:
    """Fallback job is a Forgejo workflow, not hosted /csd-python-first-drive."""
    wf = ROOT / ".github" / "workflows" / "csd-python-first-drive.yml"
    text = wf.read_text(encoding="utf-8")
    assert "workflow_dispatch" in text
    assert "csd-autodev-loop --once" in text
    assert "compute-cpu" in text
    assert "host-homelab" in text
    assert "CSD_IN_WORKFLOW" in text
    assert "grok --print" not in text
    assert "ubuntu-latest" not in text
