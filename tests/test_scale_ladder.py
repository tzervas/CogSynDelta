"""Scale ladder receipt, Grafana sat rule, and honest HF gap."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / "benchmark_results" / "scale_ladder.json"
RULES = ROOT / "deploy" / "grafana" / "provisioning" / "alerting" / "rules.yaml"
CONTACTS = ROOT / "deploy" / "grafana" / "provisioning" / "alerting" / "contact-points.yaml"
POLICIES = ROOT / "deploy" / "grafana" / "provisioning" / "alerting" / "policies.yaml"
DOC = ROOT / "docs" / "program" / "CSD-SCALE-LADDER.md"


def test_scale_ladder_json_starts_with_region_pretrain() -> None:
    """Receipt file exists; first rung is region-pretrain; none Grafana-green."""
    rec = json.loads(LADDER.read_text(encoding="utf-8"))
    assert rec["schema"] == "csd-scale-ladder/v2"
    rungs = rec["rungs"]
    assert rungs[0]["n"] == 0
    assert rungs[0]["size"] == "region_pretrain"
    assert "region-pretrain" in str(rungs[0]["stage"])
    assert rungs[0]["hf_repo"].startswith("tzervas/cogsyndelta-region-")
    assert [row["size"] for row in rungs] == [
        "region_pretrain",
        "router",
        "tiny_mind",
        "small",
        "medium",
    ]
    assert [row["n"] for row in rungs] == [0, 1, 2, 3, 4]
    assert [row["hf_repo"] for row in rungs] == [
        "tzervas/cogsyndelta-region-stream_vae-tiny",
        "tzervas/cogsyndelta-region-route-tiny",
        "tzervas/cogsyndelta-tiny",
        "tzervas/cogsyndelta-small",
        "tzervas/cogsyndelta-medium",
    ]
    assert all(row["green"] is False for row in rungs)
    assert rec["hf"]["gap"] == "mint HF"
    assert rec["hf"]["never_copy"] == "gpu/huggingface-token"
    assert rec["hf"]["region_pattern"] == "tzervas/cogsyndelta-region-<name>-<size>"
    assert rec["grafana_sat"]["contact"] == "maintainers-email"


def test_scale_ladder_doc_starts_with_region_pretrain() -> None:
    """Program doc names R0 region-pretrain first, then router, tiny mind, scale."""
    text = DOC.read_text(encoding="utf-8")
    assert "region-pretrain" in text
    assert text.find("region-pretrain") < text.find("Rung 1")
    assert "Rung 0 — tiny region-pretrain" in text
    assert "Rung 1 — router / interconnect" in text
    assert "Rung 2 — assembled tiny whole-mind" in text
    assert "Rung 3+ — larger param counts" in text
    assert "train --device cpu" in text
    assert "train-route --device cpu" in text
    assert "train-route --device cuda" in text
    assert "--gate-only" in text
    assert "192.168.1.251" in text
    assert "tzervas/cogsyndelta-region-<name>-tiny" in text
    assert "tzervas/cogsyndelta-tiny" in text
    assert "tzervas/cogsyndelta-small" in text
    assert "tzervas/cogsyndelta-medium" in text
    assert "mint HF" in text
    assert "gpu/huggingface-token" in text
    assert "Never dual 14B" in text
    assert "maintainers-email" in text


def test_grafana_rung_green_rule_pages_maintainers_only_on_one() -> None:
    """One alert rule; noData OK; contact is maintainers-email."""
    rules = RULES.read_text(encoding="utf-8")
    assert "uid: csd-scale-ladder-rung-green" in rules
    assert "title: CSDScaleLadderRungGreen" in rules
    assert "csd_scale_ladder_rung_green == 1" in rules
    assert "noDataState: OK" in rules
    assert "isPaused: false" in rules
    contacts = CONTACTS.read_text(encoding="utf-8")
    assert "name: maintainers-email" in contacts
    assert "maintainers@vectorweight.com" in contacts
    policies = POLICIES.read_text(encoding="utf-8")
    assert "receiver: maintainers-email" in policies
