"""Grafana taxonomy catalog path: group=akula-rag host=gpu5080."""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "grafana" / "apply.py"


def load_apply() -> Any:
    """Import deploy/grafana/apply.py as a module."""
    loader = importlib.machinery.SourceFileLoader("csd_grafana_apply", str(SCRIPT))
    spec = importlib.util.spec_from_loader("csd_grafana_apply", loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_taxonomy_vars_include_akula_rag_gpu5080_path() -> None:
    """Catalog group/host/path exist as options + constants; no invented PromQL."""
    mod = load_apply()
    variables = mod.taxonomy_vars("prometheus", mod.VM_UID)
    by_name = {v["name"]: v for v in variables}
    group = by_name["group"]
    host = by_name["host"]
    path = by_name["path"]
    assert any(opt.get("value") == "akula-rag" for opt in group.get("options") or [])
    assert any(opt.get("value") == "gpu5080" for opt in host.get("options") or [])
    assert any(opt.get("value") == "lab.gpu5080.index.1080ti" for opt in path.get("options") or [])
    assert by_name["catalog_group_akula_rag"]["query"] == "akula-rag"
    assert by_name["catalog_host_gpu5080"]["query"] == "gpu5080"
    assert by_name["catalog_path_1080ti"]["query"] == "lab.gpu5080.index.1080ti"
    assert "akula-rag" in mod.AKULA_RAG_DASH_URL
    assert "var-host=gpu5080" in mod.AKULA_RAG_DASH_URL
    assert "lab.gpu5080.index.1080ti" in mod.AKULA_RAG_DASH_URL
    gpus = mod.dash_gpus()
    note = gpus["panels"][0]["options"]["content"]
    assert "group=akula-rag" in note
    assert "path=lab.gpu5080.index.1080ti" in note
    assert "Do not invent" in note
    exprs = [t.get("expr", "") for panel in gpus["panels"] for t in panel.get("targets") or []]
    joined = "\n".join(exprs)
    assert "1080" not in joined
    assert "gtx1080ti" not in joined.lower()
    assert any(link.get("url") == mod.AKULA_RAG_DASH_URL for link in gpus.get("links") or [])
