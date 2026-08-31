#!/usr/bin/env python3
"""Provision CSD lab Grafana dashboards, library panels, playlist, correlations.

Reads GRAFANA_URL (default https://grafana.vectorweight.com) and TOKEN
(service account). Never prints the token. Never creates a Tempo
datasource. Jaeger is LAN 172.30.0.15:16686 only.

Why: Grafana 11.4 was dashboard-first UniFi; CSD needs taxonomy variables
(env, host, ns, group, service, path) queried from labels, drilldown to Loki
and Jaeger, and reusable library panels. PromQL/LogQL only use series
that were live at authoring time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

GRAFANA_URL = os.environ.get("GRAFANA_URL", "https://grafana.vectorweight.com").rstrip(
    "/"
)
VM_UID = "P4169E866C3094E38"
LOKI_UID = "P8E80F9AEF21F6940"
JAEGER_UID = "csd-jaeger"
JAEGER_URL = "http://172.30.0.15:16686"
FOLDER_UID = "csd-lab"
FOLDER_TITLE = "CSD lab"
SCHEMA = 39
PLUGIN = "11.4.0"
OUT_DIR = Path(__file__).resolve().parent / "dashboards"
REQUIRED_VARS = ("env", "host", "ns", "group", "service", "path")

TAX_SEL = (
    'env=~"$env", host=~"$host", ns=~"$ns", '
    'group=~"$group", service=~"$service"'
)
# Health exporter still emits host=akula-prime; taxonomy uses prime.
GPU_HOST = 'host=~"$host|akula-$host"'
LOKI_SEL = (
    '{env=~"$env", host=~"$host", ns=~"$ns", '
    'group=~"$group", service=~"$service"}'
)


def _token() -> str:
    """Return the Grafana SA token from the environment.

    Args:
        None.

    Returns:
        Bearer token string.

    Raises:
        SystemExit: TOKEN is missing.
    """
    tok = os.environ.get("TOKEN", "").strip()
    if not tok:
        raise SystemExit("TOKEN is required (secret exec TOKEN=homelab/grafana-sa-token)")
    return tok


def gf(
    path: str,
    method: str = "GET",
    body: dict[str, Any] | list[Any] | None = None,
) -> tuple[int, Any]:
    """Call the Grafana HTTP API.

    Args:
        path: Path beginning with /api/.
        method: HTTP method.
        body: Optional JSON object.

    Returns:
        Tuple of status code and parsed JSON (or raw text on parse failure).

    Raises:
        URLError: Transport failure.
    """
    url = GRAFANA_URL + path
    headers = {
        "Authorization": f"Bearer {_token()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            if not raw:
                return resp.status, None
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw.decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            parsed: Any = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            parsed = raw.decode("utf-8", "replace")[:800]
        return exc.code, parsed


def ds_ref(kind: str, uid: str) -> dict[str, str]:
    """Return a Grafana datasource ref.

    Args:
        kind: prometheus, loki, or jaeger.
        uid: Datasource UID.

    Returns:
        Grafana datasource object.
    """
    return {"type": kind, "uid": uid}


def taxonomy_vars(ds_type: str, ds_uid: str) -> list[dict[str, Any]]:
    """Build env/host/ns/group/service/path variables from label_values.

    Args:
        ds_type: prometheus or loki.
        ds_uid: Datasource UID to query.

    Returns:
        Templating list. includeAll uses .* so Loki/VM both match when All
        is selected. Values come from labels, never from IP lists.
    """
    ds = ds_ref(ds_type, ds_uid)
    out: list[dict[str, Any]] = [
        {
            "current": {"selected": True, "text": "VictoriaMetrics", "value": VM_UID},
            "hide": 2,
            "includeAll": False,
            "multi": False,
            "name": "datasource",
            "query": "prometheus",
            "refresh": 1,
            "type": "datasource",
        },
        {
            "current": {"selected": True, "text": "Loki", "value": LOKI_UID},
            "hide": 2,
            "includeAll": False,
            "multi": False,
            "name": "loki_ds",
            "query": "loki",
            "refresh": 1,
            "type": "datasource",
        },
    ]
    for name in REQUIRED_VARS:
        if ds_type == "prometheus":
            query = f'label_values({{{name}!=""}}, {name})'
        else:
            query = f'label_values({{{name}=~".+"}}, {name})'
        out.append(
            {
                "allValue": ".*",
                "current": {"selected": True, "text": "All", "value": "$__all"},
                "datasource": ds,
                "definition": query,
                "includeAll": True,
                "label": name,
                "multi": True,
                "name": name,
                "query": query,
                "refresh": 2,
                "regex": "",
                "skipUrlSync": False,
                "sort": 1,
                "type": "query",
            }
        )
    return out


def _field_config(
    unit: str | None = None,
    links: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a timeseries/stat fieldConfig block.

    Args:
        unit: Optional Grafana unit.
        links: Optional data links on the field.

    Returns:
        fieldConfig object.
    """
    defaults: dict[str, Any] = {
        "color": {"mode": "palette-classic"},
        "custom": {
            "drawStyle": "line",
            "fillOpacity": 10,
            "lineWidth": 1,
            "showPoints": "never",
            "spanNulls": True,
        },
        "mappings": [],
        "thresholds": {
            "mode": "absolute",
            "steps": [{"color": "green", "value": None}],
        },
    }
    if unit:
        defaults["unit"] = unit
    if links:
        defaults["links"] = links
    return {"defaults": defaults, "overrides": []}


def loki_explore_link() -> dict[str, Any]:
    """Data link: click series -> Loki {group,service,host}.

    Returns:
        Grafana internal data link object.
    """
    expr = (
        '{group="${__field.labels.group}",'
        'service="${__field.labels.service}",'
        'host="${__field.labels.host}"}'
    )
    return {
        "title": "Loki {group,service,host}",
        "url": "",
        "internal": {
            "datasourceName": "Loki",
            "datasourceUid": LOKI_UID,
            "query": {"expr": expr, "queryType": "range", "refId": "A"},
        },
    }


def logs_dash_link() -> dict[str, Any]:
    """Dashboard link into CSD logs with taxonomy vars from labels.

    Returns:
        Grafana URL data link.
    """
    return {
        "title": "CSD logs drilldown",
        "url": (
            "/d/csd-logs?var-group=${__field.labels.group}"
            "&var-service=${__field.labels.service}"
            "&var-host=${__field.labels.host}"
            "&var-env=${__field.labels.env}"
            "&var-ns=${__field.labels.ns}"
        ),
    }


def jaeger_trace_link() -> dict[str, Any]:
    """Data link: trace_id field -> Jaeger Explore.

    Returns:
        Grafana internal data link object.
    """
    return {
        "title": "Jaeger trace_id",
        "url": "",
        "internal": {
            "datasourceName": "Jaeger",
            "datasourceUid": JAEGER_UID,
            "query": {"query": "${__value.raw}", "queryType": "traceId"},
        },
    }


def timeseries(
    pid: int,
    title: str,
    expr: str,
    *,
    ds: dict[str, str],
    x: int,
    y: int,
    w: int = 12,
    h: int = 8,
    unit: str | None = None,
    legend: str = "",
    links: list[dict[str, Any]] | None = None,
    ds_kind_expr: str = "prometheus",
) -> dict[str, Any]:
    """Build a timeseries panel.

    Args:
        pid: Panel id.
        title: Title.
        expr: PromQL or LogQL.
        ds: Datasource ref.
        x: Grid x.
        y: Grid y.
        w: Width.
        h: Height.
        unit: Optional unit.
        legend: legendFormat.
        links: Optional data links.
        ds_kind_expr: prometheus or loki (target shape).

    Returns:
        Panel JSON.
    """
    target: dict[str, Any] = {
        "datasource": ds,
        "editorMode": "code",
        "expr": expr,
        "legendFormat": legend,
        "range": True,
        "refId": "A",
    }
    if ds_kind_expr == "prometheus":
        target["instant"] = False
    else:
        target["queryType"] = "range"
    return {
        "datasource": ds,
        "fieldConfig": _field_config(unit=unit, links=links),
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": pid,
        "options": {
            "legend": {
                "displayMode": "list",
                "placement": "bottom",
                "showLegend": True,
            },
            "tooltip": {"mode": "multi", "sort": "none"},
        },
        "pluginVersion": PLUGIN,
        "targets": [target],
        "title": title,
        "type": "timeseries",
    }


def stat(
    pid: int,
    title: str,
    expr: str,
    *,
    ds: dict[str, str],
    x: int,
    y: int,
    w: int = 4,
    h: int = 4,
    unit: str | None = None,
    legend: str = "",
    instant: bool = True,
) -> dict[str, Any]:
    """Build a stat panel.

    Args:
        pid: Panel id.
        title: Title.
        expr: PromQL.
        ds: Datasource ref.
        x: Grid x.
        y: Grid y.
        w: Width.
        h: Height.
        unit: Optional unit.
        legend: legendFormat.
        instant: Instant query.

    Returns:
        Panel JSON.
    """
    return {
        "datasource": ds,
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "mappings": [],
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "red", "value": None},
                        {"color": "green", "value": 1},
                    ],
                },
                **({"unit": unit} if unit else {}),
            },
            "overrides": [],
        },
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": pid,
        "options": {
            "colorMode": "value",
            "graphMode": "none",
            "justifyMode": "auto",
            "orientation": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "textMode": "auto",
        },
        "pluginVersion": PLUGIN,
        "targets": [
            {
                "datasource": ds,
                "editorMode": "code",
                "expr": expr,
                "instant": instant,
                "legendFormat": legend,
                "range": not instant,
                "refId": "A",
            }
        ],
        "title": title,
        "type": "stat",
    }


def table_panel(
    pid: int,
    title: str,
    expr: str,
    *,
    ds: dict[str, str],
    x: int,
    y: int,
    w: int = 24,
    h: int = 8,
    instant: bool = True,
    links: list[dict[str, Any]] | None = None,
    ds_kind: str = "prometheus",
) -> dict[str, Any]:
    """Build a table panel for group/service/instance maps.

    Args:
        pid: Panel id.
        title: Title.
        expr: Query.
        ds: Datasource ref.
        x: Grid x.
        y: Grid y.
        w: Width.
        h: Height.
        instant: Instant query (Prom).
        links: Data links.
        ds_kind: prometheus or loki.

    Returns:
        Panel JSON.
    """
    target: dict[str, Any] = {
        "datasource": ds,
        "editorMode": "code",
        "expr": expr,
        "legendFormat": "",
        "refId": "A",
    }
    if ds_kind == "prometheus":
        target["instant"] = instant
        target["range"] = not instant
        target["format"] = "table"
    else:
        target["queryType"] = "instant"
        target["instant"] = True
    fc = {
        "defaults": {
            "custom": {"align": "auto", "inspect": False},
            "mappings": [],
            "thresholds": {
                "mode": "absolute",
                "steps": [{"color": "green", "value": None}],
            },
            **({"links": links} if links else {}),
        },
        "overrides": [],
    }
    return {
        "datasource": ds,
        "fieldConfig": fc,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": pid,
        "options": {
            "cellHeight": "sm",
            "footer": {"show": False},
            "showHeader": True,
        },
        "pluginVersion": PLUGIN,
        "targets": [target],
        "title": title,
        "transformations": [
            {"id": "labelsToFields", "options": {"mode": "columns"}},
            {
                "id": "organize",
                "options": {
                    "excludeByName": {"Time": True, "__name__": True},
                    "indexByName": {
                        "group": 0,
                        "service": 1,
                        "host": 2,
                        "instance": 3,
                        "env": 4,
                        "ns": 5,
                        "Value": 6,
                    },
                },
            },
        ],
        "type": "table",
    }


def logs_panel(
    pid: int,
    title: str,
    expr: str,
    *,
    x: int,
    y: int,
    w: int = 24,
    h: int = 10,
    links: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a Loki logs panel.

    Args:
        pid: Panel id.
        title: Title.
        expr: LogQL selector.
        x: Grid x.
        y: Grid y.
        w: Width.
        h: Height.
        links: Optional data links (trace_id).

    Returns:
        Panel JSON.
    """
    ds = ds_ref("loki", "${loki_ds}")
    fc: dict[str, Any] = {"defaults": {}, "overrides": []}
    if links:
        fc["defaults"]["links"] = links
    return {
        "datasource": ds,
        "fieldConfig": fc,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": pid,
        "options": {
            "dedupStrategy": "none",
            "enableLogDetails": True,
            "prettifyLogMessage": False,
            "showCommonLabels": False,
            "showLabels": True,
            "showTime": True,
            "sortOrder": "Descending",
            "wrapLogMessage": True,
        },
        "pluginVersion": PLUGIN,
        "targets": [
            {
                "datasource": ds,
                "editorMode": "code",
                "expr": expr,
                "maxLines": 200,
                "queryType": "range",
                "refId": "A",
            }
        ],
        "title": title,
        "type": "logs",
    }


def row(pid: int, title: str, y: int, repeat: str | None = None) -> dict[str, Any]:
    """Build a dashboard row, optionally repeating on a template var.

    Args:
        pid: Panel id.
        title: Row title.
        y: Grid y.
        repeat: Optional variable name (group or service).

    Returns:
        Row panel JSON.
    """
    panel: dict[str, Any] = {
        "collapsed": False,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "id": pid,
        "panels": [],
        "title": title,
        "type": "row",
    }
    if repeat:
        panel["repeat"] = repeat
    return panel


def text_panel(pid: int, title: str, content: str, y: int, h: int = 4) -> dict[str, Any]:
    """Markdown note panel (honest gaps, no invented RED).

    Args:
        pid: Panel id.
        title: Title.
        content: Markdown.
        y: Grid y.
        h: Height.

    Returns:
        Panel JSON.
    """
    return {
        "gridPos": {"h": h, "w": 24, "x": 0, "y": y},
        "id": pid,
        "options": {"content": content, "mode": "markdown"},
        "pluginVersion": PLUGIN,
        "title": title,
        "type": "text",
    }


def lib_wrap(panel: dict[str, Any], uid: str, name: str) -> dict[str, Any]:
    """Attach a libraryPanel pointer to a dashboard panel stub.

    Args:
        panel: Panel with gridPos/id.
        uid: Library UID.
        name: Library name.

    Returns:
        Panel JSON referencing the library element.
    """
    return {
        "gridPos": panel["gridPos"],
        "id": panel["id"],
        "libraryPanel": {"name": name, "uid": uid},
        "title": panel.get("title", name),
        "type": panel.get("type", "timeseries"),
    }


def library_models() -> list[dict[str, Any]]:
    """Return library panel definitions (kind 1).

    Returns:
        List of POST /api/library-elements bodies.
    """
    vm = ds_ref("prometheus", "${datasource}")
    loki = ds_ref("loki", "${loki_ds}")
    vram_3090 = timeseries(
        1,
        "GPU VRAM 3090",
        f'akula_gpu_memory_used_mib{{{GPU_HOST},name=~".*3090.*"}}',
        ds=vm,
        x=0,
        y=0,
        w=12,
        h=8,
        unit="suffix:MiB",
        legend="{{host}} {{name}}",
        links=[loki_explore_link(), logs_dash_link()],
    )
    vram_5080 = timeseries(
        1,
        "GPU VRAM 5080",
        f'akula_gpu_memory_used_mib{{{GPU_HOST},name=~".*5080.*"}}',
        ds=vm,
        x=0,
        y=0,
        w=12,
        h=8,
        unit="suffix:MiB",
        legend="{{host}} {{name}}",
        links=[loki_explore_link(), logs_dash_link()],
    )
    loki_err = timeseries(
        1,
        "Loki errors",
        f'sum by (group, service, host) (count_over_time({LOKI_SEL} |~ "(?i)error|fail|panic|fatal" [5m]))',
        ds=loki,
        x=0,
        y=0,
        w=12,
        h=8,
        legend="{{group}}/{{service}}/{{host}}",
        links=[loki_explore_link(), logs_dash_link()],
        ds_kind_expr="loki",
    )
    vm_up = timeseries(
        1,
        "VM scrape up",
        f"up{{{TAX_SEL}}}",
        ds=vm,
        x=0,
        y=0,
        w=12,
        h=8,
        legend="{{group}}/{{service}}/{{instance}}",
        links=[loki_explore_link(), logs_dash_link()],
    )
    specs = [
        ("csd-lib-gpu-vram-3090", "CSD GPU VRAM 3090", vram_3090),
        ("csd-lib-gpu-vram-5080", "CSD GPU VRAM 5080", vram_5080),
        ("csd-lib-loki-errors", "CSD Loki errors", loki_err),
        ("csd-lib-vm-up", "CSD VM scrape up", vm_up),
    ]
    out = []
    for uid, name, model in specs:
        model = dict(model)
        model.pop("gridPos", None)
        model.pop("id", None)
        model["title"] = name
        out.append(
            {
                "uid": uid,
                "folderUid": FOLDER_UID,
                "name": name,
                "model": model,
                "kind": 1,
            }
        )
    return out


def dashboard_shell(
    uid: str,
    title: str,
    vars_ds: tuple[str, str],
    panels: list[dict[str, Any]],
    description: str,
) -> dict[str, Any]:
    """Wrap panels into a Grafana 11.4 dashboard.

    Args:
        uid: Dashboard UID.
        title: Title.
        vars_ds: (type, uid) for taxonomy label_values.
        panels: Panel list.
        description: Markdown description.

    Returns:
        Dashboard JSON (no meta).
    """
    return {
        "annotations": {"list": []},
        "description": description,
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [
            {
                "asDropdown": True,
                "icon": "dashboard",
                "includeVars": True,
                "keepTime": True,
                "tags": ["csd-investigate"],
                "title": "CSD lab",
                "type": "dashboards",
            }
        ],
        "liveNow": False,
        "panels": panels,
        "refresh": "30s",
        "schemaVersion": SCHEMA,
        "tags": ["csd", "lab", "csd-investigate"],
        "templating": {"list": taxonomy_vars(vars_ds[0], vars_ds[1])},
        "time": {"from": "now-6h", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "title": title,
        "uid": uid,
        "weekStart": "",
    }


def dash_service_map() -> dict[str, Any]:
    """Service map: group -> service -> instance, plus Loki host streams.

    Returns:
        Dashboard JSON.
    """
    vm = ds_ref("prometheus", "${datasource}")
    loki = ds_ref("loki", "${loki_ds}")
    links = [loki_explore_link(), logs_dash_link()]
    note = (
        "VM `up` is the overlay scrape (taxonomy labels). Health-exporter "
        "GPU series use host=akula-prime|gpu5080 without group/service. "
        "node_exporter is often down. Click a series for Loki "
        "{group,service,host}. No Tempo."
    )
    panels = [
        text_panel(1, "How to read", note, 0, h=3),
        row(2, "Overview", 3),
        stat(
            3,
            "up==1",
            f"sum(up{{{TAX_SEL}}} == 1) or vector(0)",
            ds=vm,
            x=0,
            y=4,
            unit="short",
        ),
        stat(
            4,
            "up==0",
            f"sum(up{{{TAX_SEL}}} == 0) or vector(0)",
            ds=vm,
            x=4,
            y=4,
            unit="short",
        ),
        stat(
            5,
            "backends up",
            f'sum(akula_backend_up{{{GPU_HOST}}}) or vector(0)',
            ds=vm,
            x=8,
            y=4,
            unit="short",
        ),
        lib_wrap(
            timeseries(6, "VM scrape up", "", ds=vm, x=12, y=4, w=12, h=8),
            "csd-lib-vm-up",
            "CSD VM scrape up",
        ),
        row(10, "group → service → instance (VM up)", 12),
        table_panel(
            11,
            "up by group / service / instance",
            f"sum by (group, service, instance, host, env, ns) (up{{{TAX_SEL}}})",
            ds=vm,
            x=0,
            y=13,
            h=8,
            links=links,
        ),
        timeseries(
            12,
            "up over time by group, service, instance",
            f"up{{{TAX_SEL}}}",
            ds=vm,
            x=0,
            y=21,
            w=24,
            h=8,
            legend="{{group}} / {{service}} / {{instance}}",
            links=links,
        ),
        row(20, "group $group", 29, repeat="group"),
        table_panel(
            21,
            "service × instance in $group",
            (
                'sum by (service, instance, host) '
                '(up{group=~"$group", env=~"$env", host=~"$host", '
                'ns=~"$ns", service=~"$service"})'
            ),
            ds=vm,
            x=0,
            y=30,
            w=12,
            h=8,
            links=links,
        ),
        timeseries(
            22,
            "up $group",
            (
                'up{group=~"$group", env=~"$env", host=~"$host", '
                'ns=~"$ns", service=~"$service"}'
            ),
            ds=vm,
            x=12,
            y=30,
            w=12,
            h=8,
            legend="{{service}} {{instance}}",
            links=links,
        ),
        row(30, "Loki streams (homelab journal taxonomy)", 38),
        table_panel(
            31,
            "log streams by group / service / host",
            f"sum by (group, service, host, env, ns) (count_over_time({LOKI_SEL}[5m]))",
            ds=loki,
            x=0,
            y=39,
            h=8,
            instant=True,
            links=links,
            ds_kind="loki",
        ),
        row(40, "Akula backends (exporter host/backend; not taxonomy group)", 47),
        timeseries(
            41,
            "akula_backend_up",
            f"akula_backend_up{{{GPU_HOST}}}",
            ds=vm,
            x=0,
            y=48,
            w=24,
            h=8,
            legend="{{host}} {{backend}}",
            links=links,
        ),
    ]
    return dashboard_shell(
        "csd-service-map",
        "CSD service map",
        ("prometheus", VM_UID),
        panels,
        "Grouped rows group → service → instance from live `up` + Loki streams.",
    )


def dash_gpus() -> dict[str, Any]:
    """3090 vs 5080 GPU dashboard with host filter from labels.

    Returns:
        Dashboard JSON.
    """
    vm = ds_ref("prometheus", "${datasource}")
    links = [loki_explore_link(), logs_dash_link()]
    note = (
        "Series are `akula_gpu_*` (not nvidia_* / DCGM). Host label is "
        "akula-prime or gpu5080 on the exporter; taxonomy host=prime also "
        "matches via akula-$host. 3090 LocalAI must stay loaded. Comfy is "
        "masked — `akula_backend_up{backend=\"comfy\"}` is shown only as "
        "the live 0, not an SLO to unmask."
    )
    panels = [
        text_panel(1, "How to read", note, 0, h=3),
        row(2, "GPU presence (host filter)", 3),
        stat(
            3,
            "3090 present",
            f'sum(akula_gpu_present{{{GPU_HOST},name=~".*3090.*"}}) or vector(0)',
            ds=vm,
            x=0,
            y=4,
        ),
        stat(
            4,
            "5080 present",
            f'sum(akula_gpu_present{{{GPU_HOST},name=~".*5080.*"}}) or vector(0)',
            ds=vm,
            x=4,
            y=4,
        ),
        stat(
            5,
            "LocalAI (stay loaded)",
            f'sum(akula_backend_up{{{GPU_HOST},backend="localai"}}) or vector(0)',
            ds=vm,
            x=8,
            y=4,
        ),
        stat(
            6,
            "timeshare queued",
            f"sum(akula_timeshare_queued{{{GPU_HOST}}}) or vector(0)",
            ds=vm,
            x=12,
            y=4,
            unit="short",
        ),
        stat(
            7,
            "timeshare busy",
            f"sum(akula_timeshare_busy{{{GPU_HOST}}}) or vector(0)",
            ds=vm,
            x=16,
            y=4,
            unit="short",
        ),
        row(10, "VRAM 3090 vs 5080", 8),
        lib_wrap(
            timeseries(11, "GPU VRAM 3090", "", ds=vm, x=0, y=9, w=12, h=8),
            "csd-lib-gpu-vram-3090",
            "CSD GPU VRAM 3090",
        ),
        lib_wrap(
            timeseries(12, "GPU VRAM 5080", "", ds=vm, x=12, y=9, w=12, h=8),
            "csd-lib-gpu-vram-5080",
            "CSD GPU VRAM 5080",
        ),
        timeseries(
            13,
            "VRAM fraction",
            (
                f"akula_gpu_memory_used_mib{{{GPU_HOST}}} "
                f"/ clamp_min(akula_gpu_memory_total_mib{{{GPU_HOST}}}, 1)"
            ),
            ds=vm,
            x=0,
            y=17,
            w=12,
            h=8,
            unit="percentunit",
            legend="{{host}} {{name}}",
            links=links,
        ),
        timeseries(
            14,
            "SM util (ratio)",
            f"akula_gpu_utilization_ratio{{{GPU_HOST}}}",
            ds=vm,
            x=12,
            y=17,
            w=12,
            h=8,
            unit="percentunit",
            legend="{{host}} {{name}}",
            links=links,
        ),
        timeseries(
            15,
            "GPU temp C",
            f"akula_gpu_temperature_celsius{{{GPU_HOST}}}",
            ds=vm,
            x=0,
            y=25,
            w=12,
            h=8,
            unit="celsius",
            legend="{{host}} {{name}}",
            links=links,
        ),
        timeseries(
            16,
            "timeshare queue / busy",
            f"akula_timeshare_queued{{{GPU_HOST}}}",
            ds=vm,
            x=12,
            y=25,
            w=12,
            h=8,
            legend="queued {{host}}",
            links=links,
        ),
        row(20, "repeat by host", 33, repeat="host"),
        timeseries(
            21,
            "VRAM on $host",
            'akula_gpu_memory_used_mib{host=~"$host|akula-$host"}',
            ds=vm,
            x=0,
            y=34,
            w=24,
            h=8,
            unit="suffix:MiB",
            legend="{{host}} {{name}}",
            links=links,
        ),
    ]
    return dashboard_shell(
        "csd-gpus",
        "CSD GPUs 3090 vs 5080",
        ("prometheus", VM_UID),
        panels,
        "Host filter from labels. 3090 vs 5080 split by GPU name, not IPs.",
    )


def dash_logs() -> dict[str, Any]:
    """Loki logs with taxonomy filters and error rate.

    Returns:
        Dashboard JSON.
    """
    loki = ds_ref("loki", "${loki_ds}")
    links = [loki_explore_link(), logs_dash_link(), jaeger_trace_link()]
    note = (
        "Selector `{env,host,ns,group,service}` from Loki labels (homelab "
        "journal). Suricata has no env and is excluded. Click a series to "
        "Explore Loki. If a line contains trace_id, the derived field opens "
        "Jaeger (LAN). No Tempo."
    )
    err = f'{LOKI_SEL} |~ "(?i)error|fail|panic|fatal"'
    panels = [
        text_panel(1, "How to read", note, 0, h=3),
        row(2, "Volume and errors", 3),
        lib_wrap(
            timeseries(3, "Loki errors", "", ds=loki, x=0, y=4, w=12, h=8),
            "csd-lib-loki-errors",
            "CSD Loki errors",
        ),
        timeseries(
            4,
            "log lines / 5m by group, service, host",
            f"sum by (group, service, host) (count_over_time({LOKI_SEL}[5m]))",
            ds=loki,
            x=12,
            y=4,
            w=12,
            h=8,
            legend="{{group}}/{{service}}/{{host}}",
            links=[loki_explore_link()],
            ds_kind_expr="loki",
        ),
        row(10, "Logs $group / $service / $host", 12),
        logs_panel(11, "CSD journal", LOKI_SEL, x=0, y=13, h=12, links=links),
        logs_panel(12, "error|fail|panic|fatal", err, x=0, y=25, h=10, links=links),
    ]
    return dashboard_shell(
        "csd-logs",
        "CSD logs",
        ("loki", LOKI_UID),
        panels,
        "Click series → Loki {group,service,host}. Taxonomy vars from Loki labels.",
    )


def dash_traces() -> dict[str, Any]:
    """Loki lines that look like traces, plus Jaeger search.

    Returns:
        Dashboard JSON.
    """
    loki = ds_ref("loki", "${loki_ds}")
    jaeger = ds_ref("jaeger", JAEGER_UID)
    links = [jaeger_trace_link(), loki_explore_link()]
    note = (
        "CSD OTLP is not enabled (`intern_parallel`). Journal has no "
        "trace_id today; this dashboard + Loki derived field + correlation "
        "are ready for when lines appear. Jaeger datasource is LAN "
        "`172.30.0.15:16686` (Apache-2.0). No Tempo. Jaeger UI is not "
        "published on WAN."
    )
    trace_logql = (
        f'{LOKI_SEL} |~ "(?i)trace[_-]?id|traceparent|traceId" '
        r'| regexp "(?P<trace_id>[a-fA-F0-9]{16,32})"'
    )
    panels = [
        text_panel(1, "How to read", note, 0, h=4),
        row(2, "Loki trace_id → Jaeger", 4),
        logs_panel(
            3,
            "lines with trace_id / traceparent",
            trace_logql,
            x=0,
            y=5,
            h=10,
            links=links,
        ),
        timeseries(
            4,
            "trace-like lines / 5m",
            (
                f"sum by (group, service, host) "
                f'(count_over_time({LOKI_SEL} |~ "(?i)trace[_-]?id|traceparent" [5m]))'
            ),
            ds=loki,
            x=0,
            y=15,
            w=24,
            h=8,
            legend="{{group}}/{{service}}/{{host}}",
            links=[loki_explore_link(), logs_dash_link()],
            ds_kind_expr="loki",
        ),
        row(10, "Jaeger search (LAN)", 23),
        {
            "datasource": jaeger,
            "gridPos": {"h": 10, "w": 24, "x": 0, "y": 24},
            "id": 11,
            "targets": [
                {
                    "datasource": jaeger,
                    "queryType": "search",
                    "service": "$service",
                    "limit": 20,
                    "refId": "A",
                }
            ],
            "title": "Jaeger search by $service (empty until OTLP)",
            "type": "traces",
        },
        logs_panel(
            12,
            "akula-jaeger.service journal",
            '{service=~"akula-jaeger.*", env=~"$env", host=~"$host", group=~"$group"}',
            x=0,
            y=34,
            h=8,
            links=links,
        ),
    ]
    return dashboard_shell(
        "csd-traces",
        "CSD traces (Jaeger)",
        ("loki", LOKI_UID),
        panels,
        "Loki trace_id → Jaeger LAN. No Tempo.",
    )


def dash_investigate() -> dict[str, Any]:
    """Error / saturation / latency from series that actually exist.

    Returns:
        Dashboard JSON.
    """
    vm = ds_ref("prometheus", "${datasource}")
    loki = ds_ref("loki", "${loki_ds}")
    links = [loki_explore_link(), logs_dash_link()]
    note = (
        "**Not classic RED.** There is no `http_requests_total` or app "
        "latency histogram. Used live series only: Loki error rate, `up` "
        "(availability), `scrape_duration_seconds` (scrape latency, often "
        "~8s timeout), GPU util/VRAM (saturation). node_exporter "
        "`node_load1` exists unlabeled and is omitted here so taxonomy "
        "filters stay honest. 3090 LocalAI stay loaded; Comfy masked."
    )
    panels = [
        text_panel(1, "RED/USE — real metrics only", note, 0, h=5),
        row(2, "Errors / availability", 5),
        lib_wrap(
            timeseries(3, "Loki errors", "", ds=loki, x=0, y=6, w=12, h=8),
            "csd-lib-loki-errors",
            "CSD Loki errors",
        ),
        lib_wrap(
            timeseries(4, "VM scrape up", "", ds=vm, x=12, y=6, w=12, h=8),
            "csd-lib-vm-up",
            "CSD VM scrape up",
        ),
        row(10, "Saturation (GPUs)", 14),
        timeseries(
            11,
            "GPU util (saturation)",
            f"akula_gpu_utilization_ratio{{{GPU_HOST}}}",
            ds=vm,
            x=0,
            y=15,
            w=12,
            h=8,
            unit="percentunit",
            legend="{{host}} {{name}}",
            links=links,
        ),
        timeseries(
            12,
            "VRAM fraction (saturation)",
            (
                f"akula_gpu_memory_used_mib{{{GPU_HOST}}} "
                f"/ clamp_min(akula_gpu_memory_total_mib{{{GPU_HOST}}}, 1)"
            ),
            ds=vm,
            x=12,
            y=15,
            w=12,
            h=8,
            unit="percentunit",
            legend="{{host}} {{name}}",
            links=links,
        ),
        row(20, "Latency (scrape, not app RED)", 23),
        timeseries(
            21,
            "scrape_duration_seconds",
            f"scrape_duration_seconds{{{TAX_SEL}}}",
            ds=vm,
            x=0,
            y=24,
            w=24,
            h=8,
            unit="s",
            legend="{{group}}/{{service}}/{{instance}}",
            links=links,
        ),
        row(30, "Logs for the selected group/service/host", 32),
        logs_panel(
            31,
            "Loki {env,host,ns,group,service}",
            LOKI_SEL,
            x=0,
            y=33,
            h=10,
            links=[loki_explore_link(), jaeger_trace_link()],
        ),
    ]
    return dashboard_shell(
        "csd-investigate",
        "CSD investigate",
        ("prometheus", VM_UID),
        panels,
        "Error rate (Loki), saturation (GPU), latency (scrape). No invented RED.",
    )


def all_dashboards() -> list[dict[str, Any]]:
    """Return the five CSD lab dashboards.

    Returns:
        Dashboard JSON list.
    """
    return [
        dash_service_map(),
        dash_gpus(),
        dash_logs(),
        dash_traces(),
        dash_investigate(),
    ]


def write_json(dashboards: list[dict[str, Any]]) -> None:
    """Write dashboard JSON snapshots next to this script.

    Args:
        dashboards: Dashboard objects.

    Returns:
        None.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for dash in dashboards:
        path = OUT_DIR / f"{dash['uid']}.json"
        path.write_text(json.dumps(dash, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")


def ensure_folder() -> str:
    """Return CSD lab folder UID, creating the folder if missing.

    Reuses a same-title folder (Grafana 11 forbids duplicate titles even
    with a different UID).

    Returns:
        Folder UID.

    Raises:
        SystemExit: API rejected the folder.
    """
    global FOLDER_UID
    code, data = gf(f"/api/folders/{FOLDER_UID}")
    if code == 200:
        print(f"folder exists uid={FOLDER_UID}")
        return FOLDER_UID
    code, listing = gf("/api/folders")
    if code == 200 and isinstance(listing, list):
        for folder in listing:
            if folder.get("title") == FOLDER_TITLE:
                FOLDER_UID = str(folder["uid"])
                print(f"folder exists by title uid={FOLDER_UID}")
                return FOLDER_UID
    code, data = gf(
        "/api/folders",
        "POST",
        {"uid": FOLDER_UID, "title": FOLDER_TITLE},
    )
    if code not in (200, 201):
        raise SystemExit(f"folder create failed {code} {data}")
    if isinstance(data, dict) and data.get("uid"):
        FOLDER_UID = str(data["uid"])
    print(f"folder created uid={FOLDER_UID}")
    return FOLDER_UID


def ensure_jaeger() -> None:
    """Create or update the Jaeger LAN datasource. Never Tempo.

    Returns:
        None.

    Raises:
        SystemExit: Datasource API failed.
    """
    json_data = {
        "tracesToLogsV2": {
            "datasourceUid": LOKI_UID,
            "spanStartTimeShift": "-1h",
            "spanEndTimeShift": "1h",
            "filterByTraceID": True,
            "customQuery": True,
            "query": (
                '{group=~"${__span.tags.group:regex}",'
                'service=~"${__span.tags.service:regex}",'
                'host=~"${__span.tags.host:regex}"} |~ "${__span.traceId}"'
            ),
        },
        "tracesToMetrics": {
            "datasourceUid": VM_UID,
            "spanStartTimeShift": "-1h",
            "spanEndTimeShift": "1h",
            "queries": [
                {
                    "name": "up",
                    "query": 'up{service=~"${__span.tags.service:regex}"}',
                }
            ],
        },
        "nodeGraph": {"enabled": True},
    }
    payload = {
        "name": "Jaeger",
        "type": "jaeger",
        "uid": JAEGER_UID,
        "access": "proxy",
        "url": JAEGER_URL,
        "isDefault": False,
        "jsonData": json_data,
    }
    code, existing = gf(f"/api/datasources/uid/{JAEGER_UID}")
    if code == 200 and isinstance(existing, dict):
        payload["id"] = existing.get("id")
        code, data = gf(f"/api/datasources/{existing['id']}", "PUT", payload)
    else:
        code, data = gf("/api/datasources", "POST", payload)
    if code not in (200, 201):
        raise SystemExit(f"jaeger datasource failed {code} {data}")
    print(f"jaeger datasource uid={JAEGER_UID} url={JAEGER_URL}")


def patch_loki_derived() -> None:
    """Add Loki derived field trace_id -> Jaeger; keep OpenSearch event_id.

    Returns:
        None.
    """
    code, ds = gf(f"/api/datasources/uid/{LOKI_UID}")
    if code != 200 or not isinstance(ds, dict):
        print(f"skip loki derived: get {code}")
        return
    json_data = dict(ds.get("jsonData") or {})
    fields = list(json_data.get("derivedFields") or [])
    if any(f.get("name") == "trace_id" for f in fields if isinstance(f, dict)):
        print("loki derived field trace_id already present")
        return
    fields.append(
        {
            "datasourceUid": JAEGER_UID,
            "matcherRegex": (
                r"(?:trace_id|traceId|trace-id)[=:\"\s]+([A-Fa-f0-9]{16,32})"
            ),
            "name": "trace_id",
            "url": "",
            "urlDisplayLabel": "Jaeger",
        }
    )
    json_data["derivedFields"] = fields
    body = {
        "id": ds["id"],
        "uid": ds["uid"],
        "name": ds["name"],
        "type": ds["type"],
        "access": ds.get("access", "proxy"),
        "url": ds.get("url"),
        "isDefault": ds.get("isDefault", False),
        "jsonData": json_data,
    }
    code, data = gf(f"/api/datasources/{ds['id']}", "PUT", body)
    if code not in (200, 201):
        print(f"loki derived field update failed {code} {data}")
        return
    print("loki derived field trace_id -> Jaeger")


def upsert_library(items: list[dict[str, Any]]) -> None:
    """Create or update library panels in CSD lab.

    Args:
        items: Library element payloads.

    Returns:
        None.
    """
    for item in items:
        uid = item["uid"]
        code, existing = gf(f"/api/library-elements/{uid}")
        if code == 200 and isinstance(existing, dict):
            result = existing.get("result") or existing
            version = result.get("version", 1)
            patch = {
                "folderUid": FOLDER_UID,
                "name": item["name"],
                "model": item["model"],
                "kind": 1,
                "uid": uid,
                "version": version,
            }
            code, data = gf(f"/api/library-elements/{uid}", "PATCH", patch)
        else:
            code, data = gf("/api/library-elements", "POST", item)
        if code not in (200, 201):
            print(f"library {uid} failed {code} {data}")
        else:
            print(f"library {uid}")


def upsert_dashboards(dashboards: list[dict[str, Any]]) -> None:
    """POST dashboards into folder CSD lab with overwrite.

    Args:
        dashboards: Dashboard JSON list.

    Returns:
        None.

    Raises:
        SystemExit: A dashboard failed to save.
    """
    for dash in dashboards:
        body = {
            "dashboard": dash,
            "folderUid": FOLDER_UID,
            "overwrite": True,
            "message": "CSD lab o11y (taxonomy vars, no Tempo)",
        }
        code, data = gf("/api/dashboards/db", "POST", body)
        if code not in (200, 201) or not isinstance(data, dict):
            raise SystemExit(f"dashboard {dash['uid']} failed {code} {data}")
        print(f"dashboard uid={dash['uid']} url={data.get('url')}")


def ensure_playlist(uids: list[str]) -> None:
    """Create or update playlist CSD-investigate (60s).

    Args:
        uids: Dashboard UIDs in rotation order.

    Returns:
        None.
    """
    items = [
        {"type": "dashboard_by_uid", "value": uid, "order": i + 1}
        for i, uid in enumerate(uids)
    ]
    payload = {"name": "CSD-investigate", "interval": "1m", "items": items, "uid": "csd-investigate-pl"}
    code, listing = gf("/api/playlists")
    existing_uid = None
    if code == 200 and isinstance(listing, list):
        for pl in listing:
            if pl.get("name") == "CSD-investigate" or pl.get("uid") == "csd-investigate-pl":
                existing_uid = pl.get("uid")
                break
    if existing_uid:
        code, data = gf(f"/api/playlists/{existing_uid}", "PUT", payload)
    else:
        code, data = gf("/api/playlists", "POST", payload)
    if code not in (200, 201):
        print(f"playlist failed {code} {data}")
        return
    uid = (data or {}).get("uid") if isinstance(data, dict) else existing_uid
    print(f"playlist CSD-investigate uid={uid}")


def ensure_correlations() -> None:
    """Wire Loki <-> VM <-> Jaeger correlations.

    Returns:
        None.
    """
    code, listing = gf("/api/datasources/correlations")
    existing: list[dict[str, Any]] = []
    if code == 200 and isinstance(listing, dict):
        existing = list(listing.get("correlations") or [])

    def has(label: str) -> bool:
        return any(c.get("label") == label for c in existing)

    specs = [
        (
            LOKI_UID,
            {
                "targetUID": JAEGER_UID,
                "label": "Loki trace_id -> Jaeger",
                "description": "Open Jaeger by trace_id extracted from a Loki line",
                "type": "query",
                "config": {
                    "field": "trace_id",
                    "target": {"query": "${__value.raw}", "queryType": "traceId"},
                    "transformations": [
                        {
                            "type": "regex",
                            "expression": (
                                r"(?:trace_id|traceId|trace-id)"
                                r"[=:\"\s]+([A-Fa-f0-9]{16,32})"
                            ),
                            "field": "trace_id",
                        }
                    ],
                },
            },
        ),
        (
            LOKI_UID,
            {
                "targetUID": VM_UID,
                "label": "Loki service -> VM up",
                "description": "From a Loki service label to VictoriaMetrics up",
                "type": "query",
                "config": {
                    "field": "service",
                    "target": {
                        "expr": 'up{service="${__value.raw}"}',
                        "refId": "A",
                    },
                },
            },
        ),
        (
            VM_UID,
            {
                "targetUID": LOKI_UID,
                "label": "VM service -> Loki",
                "description": "From a VM series service label to Loki {service}",
                "type": "query",
                "config": {
                    "field": "service",
                    "target": {
                        "expr": '{service="${__value.raw}"}',
                        "queryType": "range",
                        "refId": "A",
                    },
                },
            },
        ),
        (
            VM_UID,
            {
                "targetUID": JAEGER_UID,
                "label": "VM service -> Jaeger",
                "description": "Jaeger search by taxonomy service (LAN, no Tempo)",
                "type": "query",
                "config": {
                    "field": "service",
                    "target": {
                        "queryType": "search",
                        "service": "${__value.raw}",
                    },
                },
            },
        ),
    ]
    for source, body in specs:
        if has(body["label"]):
            print(f"correlation exists {body['label']}")
            continue
        code, data = gf(f"/api/datasources/uid/{source}/correlations", "POST", body)
        if code not in (200, 201):
            print(f"correlation {body['label']} failed {code} {data}")
        else:
            uid = (data or {}).get("result", {}).get("uid") if isinstance(data, dict) else None
            print(f"correlation {body['label']} uid={uid}")


def verify() -> dict[str, Any]:
    """Confirm >=3 dashboards in CSD lab have the required template vars.

    Returns:
        Dict with ok, uids, missing, playlist, library, correlations.
    """
    code, search = gf("/api/search?type=dash-db&limit=100")
    uids: list[str] = []
    missing: list[str] = []
    if code == 200 and isinstance(search, list):
        for item in search:
            if item.get("folderUid") != FOLDER_UID and item.get("folderTitle") != FOLDER_TITLE:
                if not str(item.get("uid", "")).startswith("csd-"):
                    continue
            uid = item.get("uid")
            if not uid:
                continue
            c2, dash = gf(f"/api/dashboards/uid/{uid}")
            if c2 != 200 or not isinstance(dash, dict):
                continue
            names = [
                t.get("name")
                for t in (dash.get("dashboard") or {}).get("templating", {}).get("list", [])
            ]
            if all(v in names for v in REQUIRED_VARS):
                uids.append(uid)
            else:
                missing.append(f"{uid} vars={names}")
    code, libs = gf("/api/library-elements?perPage=50")
    lib_uids: list[str] = []
    if code == 200 and isinstance(libs, dict):
        for el in (libs.get("result") or {}).get("elements") or []:
            if str(el.get("uid", "")).startswith("csd-lib-"):
                lib_uids.append(el["uid"])
    code, plays = gf("/api/playlists")
    play_ok = False
    play_uid = ""
    if code == 200 and isinstance(plays, list):
        for pl in plays:
            if pl.get("name") == "CSD-investigate":
                play_ok = True
                play_uid = pl.get("uid") or ""
    code, corrs = gf("/api/datasources/correlations")
    corr_labels: list[str] = []
    if code == 200 and isinstance(corrs, dict):
        corr_labels = [c.get("label") for c in corrs.get("correlations") or []]
    code, dss = gf("/api/datasources")
    types = []
    if code == 200 and isinstance(dss, list):
        types = [d.get("type") for d in dss]
    return {
        "ok": len(uids) >= 3 and "tempo" not in types,
        "dashboard_uids": uids,
        "missing": missing,
        "library": lib_uids,
        "playlist": play_uid if play_ok else "",
        "correlations": corr_labels,
        "datasource_types": types,
    }


def main() -> int:
    """CLI entry: dry-run writes JSON; default applies via API.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write JSON only; do not call Grafana",
    )
    args = parser.parse_args()
    dashboards = all_dashboards()
    write_json(dashboards)
    if args.dry_run:
        print("dry-run: skipped Grafana API")
        return 0
    if not os.environ.get("TOKEN"):
        raise SystemExit("TOKEN is required unless --dry-run")
    ensure_folder()
    ensure_jaeger()
    patch_loki_derived()
    upsert_library(library_models())
    upsert_dashboards(dashboards)
    ensure_playlist([d["uid"] for d in dashboards])
    ensure_correlations()
    result = verify()
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
