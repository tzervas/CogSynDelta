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


def test_scale_ladder_json_has_five_rungs_none_green() -> None:
    """Receipt file exists; no rung is Grafana-green yet.

    v2 extended the ladder from three assembled sizes to five rungs, adding the two
    region-level stages (region_pretrain, router) ahead of tiny_mind. The receipt is
    generated, so it is authoritative and this test follows it.
    """
    rec = json.loads(LADDER.read_text(encoding="utf-8"))
    assert rec["schema"] == "csd-scale-ladder/v2"
    rungs = rec["rungs"]
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
    # v2 additions: region repos are named separately from assembled ones.
    assert rec["hf"]["region_pattern"] == "tzervas/cogsyndelta-region-<name>-<size>"
    assert rec["hf"]["assembled_pattern"] == "tzervas/cogsyndelta-<size>"
    assert rec["grafana_sat"]["contact"] == "maintainers-email"


def test_scale_ladder_doc_lists_recipes() -> None:
    """Program doc names train/eval/HF per size and mint steps."""
    text = DOC.read_text(encoding="utf-8")
    assert "tiny" in text
    assert "small" in text
    assert "medium" in text
    assert "train-route --device cpu" in text
    assert "train-route --device cuda" in text
    assert "192.168.1.251" in text
    assert "tzervas/cogsyndelta-tiny" in text
    assert "tzervas/cogsyndelta-small" in text
    assert "tzervas/cogsyndelta-medium" in text
    assert "mint HF" in text
    assert "gpu/huggingface-token" in text
    assert "Never dual 14B" in text


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
