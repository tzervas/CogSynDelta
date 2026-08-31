let meta = #{
    name: "csd-grafana-o11y",
    description: "Flesh out Grafana OSS: dashboards, playlists, library panels, correlations, recording and alert rules to maintainers@. No Tempo. No WAN SMTP.",
    when_to_use: "Grafana Explore works but playlists/library panels/alerts/recording rules are thin. After send-only Postfix.",
    phases: [
        #{ title: "Audit", detail: "datasources, dashboards, plugins, contact points" },
        #{ title: "Provision", detail: "library panels, playlist, correlations, VM recording" },
        #{ title: "Alerts", detail: "unified alert rules + maintainers-email" },
        #{ title: "Verify", detail: "API lists + SMTP still send-only" },
    ],
};

let ok_schema = #{
    "type": "object",
    "required": ["ok", "notes", "evidence"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
        "evidence": #{ "type": "string" },
        "gaps": #{ "type": "string" },
    },
};

fn as_str(x) {
    if type_of(x) == "string" { return x; }
    "";
}

let do_apply = false;
if args != () {
    if args.apply == true { do_apply = true; }
}

phase("Audit");
let ap = "";
ap += "Homelab Grafana 11.4 at https://grafana.vectorweight.com (unifi-grafana 172.30.0.13:3000).\n";
ap += "Do not print admin password. Use /etc/ap-secrets/unifi-grafana.env on homelab via sudo, or tyler Editor if enough.\n";
ap += "Inventory via API: datasources (Loki, VictoriaMetrics, OpenSearch uid), dashboards, folders, playlists, library elements, plugins, correlations, alert rules, contact points, notification policies.\n";
ap += "Invariants: no Tempo (AGPL). Traces stay Jaeger LAN. SMTP 172.30.0.1:25 send-only. Contact point maintainers-email.\n";
ap += "gaps = missing playlists, library panels, recording rules, alert rules, correlations between Loki and VM.\n";
ap += "ok=true if datasources exist and SMTP contact point exists. evidence=counts. Never merge.\n";
let audit = agent(ap, #{
    label: "grafana-audit",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Provision");
    let pp = "";
    pp += "Repo CogSynDelta deploy/grafana if present, else write provision YAML under /data/unifi-security/deploy/config/grafana/provisioning/{dashboards,alerting,alerting/rules,plugins}.\n";
    pp += "Create folder CSD lab. Library panels: GPU VRAM 3090, GPU VRAM 5080, Loki errors, VM scrape up.\n";
    pp += "Dashboards: CSD GPUs (nvidia_smi if scraped), CSD services (lab :9118, gateway :9120, postfix queue), CSD logs (Loki {unit=~\"csd-.*|postfix|unifi-grafana\"}).\n";
    pp += "Playlist CSD-oncall rotating those dashboards 60s.\n";
    pp += "Correlations: Loki trace_id -> Jaeger if field present; else Loki job -> VM up{job}.\n";
    pp += "Recording rules in Grafana or VM: job:up:avg5m, instance:log_error_rate:5m if log metrics exist. Do not invent PromQL that has no series.\n";
    pp += "Do not install AGPL plugins. OSS only. Do not bind Grafana 0.0.0.0 extra ports.\n";
    pp += "ok=true if playlist + at least 2 dashboards exist. evidence=uids. Never print secrets.\n";
    let prov = agent(pp, #{
        label: "grafana-provision",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Alerts");
    let al = "";
    al += "Unified alerting. Contact point maintainers-email -> maintainers@vectorweight.com already.\n";
    al += "Alert rules (only if series exist): instance down (up==0 for 5m) for unifi-grafana, loki, victoria, postfix, csd-lab if exported; GPU 5080 lock held > 2h optional; disk homelab > 90% if node_exporter.\n";
    al += "Do not alert on grafana@ bounce. notify_classes already resource,software.\n";
    al += "Fire a test via contact-point test API once, not a flood.\n";
    al += "ok=true if >=1 alert rule is live and policy receiver is maintainers-email. evidence=rule names.\n";
    let alerts = agent(al, #{
        label: "grafana-alerts",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Verify");
    let vp = "";
    vp += "GET playlists, library-elements, alert-rules, contact-points. Confirm postfix still 127.0.0.1+172.30.0.1 only, not 0.0.0.0.\n";
    vp += "ok=true if all three APIs 200 and SMTP bind unchanged. Never merge.\n";
    let ver = agent(vp, #{
        label: "grafana-verify",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if audit != () && audit.success && audit.output.ok == true {
        if prov != () && prov.success && prov.output.ok == true {
            if alerts != () && alerts.success && alerts.output.ok == true {
                if ver != () && ver.success && ver.output.ok == true {
                    sat = true;
                }
            }
        }
    }
    complete(#{
        summary: if sat { "Grafana o11y fleshed out" } else { "Grafana o11y gaps remain" },
        sat: sat,
        audit: audit,
        provision: prov,
        alerts: alerts,
        verify: ver,
    });
}

phase("Verify");
let vp2 = "";
vp2 += "Read-only: list Grafana datasources and alert contact points without changing them. evidence=names. ok=true if Loki+VM exist and maintainers-email exists.\n";
let ver2 = agent(vp2, #{
    label: "grafana-readonly",
    capability_mode: "execute",
    output_schema: ok_schema,
});
let sat2 = audit != () && audit.success && audit.output.ok == true;
complete(#{
    summary: if sat2 { "Grafana audit ok; re-run args.apply=true to provision" } else { "Grafana audit gaps" },
    sat: sat2,
    audit: audit,
    verify: ver2,
});
