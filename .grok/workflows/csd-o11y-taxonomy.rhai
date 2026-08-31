let meta = #{
    name: "csd-o11y-taxonomy",
    description: "Label/tag/annotate env, host, namespace, service, group, path; isolation environments; Grafana drilldown with metrics, Loki, Jaeger spans. Follows csd-grafana-o11y. No Tempo.",
    when_to_use: "After Grafana Explore/alerts exist; need filterable service paths, grouped dashes, traces, isolation envs.",
    phases: [
        #{ title: "Taxonomy", detail: "label keys + isolation env matrix" },
        #{ title: "Instrument", detail: "scrape/journal/Jaeger tags; no WAN" },
        #{ title: "Views", detail: "variable dashes, groups, drilldown, alerts" },
        #{ title: "Verify", detail: "filter by host/ns/group; SMTP bind; no Tempo" },
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

let hard = "";
hard += "HARD: never 0.0.0.0 WAN. No Tempo (AGPL); traces = Jaeger Apache-2.0 LAN 172.30.0.15:16686. ";
hard += "SMTP send-only 127.0.0.1+172.30.0.1:25, contact maintainers-email. Never pause 3090 LocalAI. ";
hard += "Never GitHub bot push. Never main/staging/develop/dev. Never write operator/model vaults. ";
hard += "Never mix 384-d Qdrant. Do not unmask Comfy. Use tools; do not invent series.\n";

phase("Taxonomy");
let tp = "";
tp += hard;
tp += "Repo /home/kang/code/personal/tzervas/CogSynDelta feat/agent-harness. Also homelab Grafana/Loki/VM/Jaeger.\n";
tp += "Read docs/CODEX-OPS.md, docs/program/WHO-RUNS-WHAT.md, docs/program/HARNESS.md, existing deploy/grafana if any.\n";
tp += "Propose a label taxonomy (stable keys, low cardinality):\n";
tp += "  env=lab|ci|prod-intent  host=prime|gpu5080|homelab  ns=csd|webui|media|mail|git|o11y|runner\n";
tp += "  service=full systemd or compose name  kind=api|gpu|ui|mta|runner|index|gateway\n";
tp += "  group=logical set that works together (csd-autodev, akula-chat, forgejo-ci, grafana-stack)\n";
tp += "  path=logical path for filters e.g. lab.csd.gateway or ci.gpu5080.runner\n";
tp += "Isolation environments (long-term, do not invent new VLANs this run unless files already exist):\n";
tp += "  1 process/user unit  2 host  3 LAN ns/quadlet  4 vault (CSD vs operator secrets)  5 GPU lock\n";
tp += "Document in docs/program/CSD-O11Y-TAXONOMY.md. ok=true if the doc lists keys+allowed values+group map.\n";
tp += "evidence=path. Do not provision yet unless args.apply handled later.\n";
let tax = agent(tp, #{
    label: "taxonomy-doc",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Instrument");
    let ip = "";
    ip += hard;
    ip += "Apply taxonomy without exploding cardinality. Prefer relabel in scrape configs and Loki pipeline stages.\n";
    ip += "Hosts: prime 192.168.1.98, gpu5080 192.168.1.252, homelab 192.168.1.170.\n";
    ip += "Find node_exporter/nvidia/journald/promtail/alloy/victoriametrics scrape YAML on homelab (unifi-grafana stack).\n";
    ip += "Add labels env, host, ns, service, kind, group, path where the series already exist. Do not invent PromQL.\n";
    ip += "Journald/Loki: extra labels from systemd unit names mapped to group=csd-autodev|akula-chat|forgejo-ci|grafana-stack.\n";
    ip += "Jaeger: service names must match taxonomy service; process tags host+env+ns. Do not replace Jaeger with Tempo.\n";
    ip += "Commit YAML under CogSynDelta deploy/o11y/ or the existing grafana provisioning tree. Push autodev feat/agent-harness.\n";
    ip += "ok=true if at least one scrape or Loki config gained the new labels. evidence=file paths.\n";
    let inst = agent(ip, #{
        label: "instrument-labels",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Views");
    let vjobs = [];
    let vd = "";
    vd += hard;
    vd += "Grafana 11.4 https://grafana.vectorweight.com (172.30.0.13:3000). Folder CSD lab.\n";
    vd += "Dashboards with template vars: env, host, ns, group, service (query from labels, not hardcoded IP lists).\n";
    vd += "1) Service map / grouped rows: group -> service -> instance. 2) GPU 3090 vs 5080 with host filter.\n";
    vd += "3) Logs drilldown: click series -> Loki {group,service,host}. 4) Trace drilldown: Loki trace_id -> Jaeger.\n";
    vd += "5) Investigation: error rate, saturation, latency if series exist; RED/USE only on real metrics.\n";
    vd += "Library panels reusable. Playlist CSD-investigate. Correlations Loki<->VM<->Jaeger.\n";
    vd += "ok=true if >=3 dashboards with those variables exist. evidence=uids.\n";
    vjobs.push(#{
        prompt: vd,
        label: "views-dashboards",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let va = "";
    va += hard;
    va += "Alert rules labeled with group/ns/host so routing and filters work. Receiver maintainers-email only.\n";
    va += "Rules (only if series exist): service down by group, disk, GPU lock stale, postfix deferred, runner offline.\n";
    va += "One contact-point test, not a flood. Notification policy can group by group then host.\n";
    va += "ok=true if rules have the taxonomy labels. evidence=rule names.\n";
    vjobs.push(#{
        prompt: va,
        label: "views-alerts",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let views = parallel(vjobs);

    phase("Verify");
    let vp = "";
    vp += hard;
    vp += "Prove filterability: PromQL or Grafana variable query that lists group= or host= labels (paste result counts).\n";
    vp += "GET dashboards/playlists/alert-rules. Confirm no Tempo datasource. Postfix bind still loopback+172.30.0.1.\n";
    vp += "docs/program/CSD-O11Y-TAXONOMY.md exists. Do not merge PR #1. evidence=label values + uids.\n";
    vp += "ok=true only with those API/query results.\n";
    let ver = agent(vp, #{
        label: "verify-taxonomy",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    let d0 = ();
    let d1 = ();
    if views != () {
        if views.len() > 0 { d0 = views[0]; }
        if views.len() > 1 { d1 = views[1]; }
    }
    let sat = false;
    if tax != () && tax.success && tax.output.ok == true {
        if inst != () && inst.success && inst.output.ok == true {
            if d0 != () && d0.success && d0.output.ok == true {
                if d1 != () && d1.success && d1.output.ok == true {
                    if ver != () && ver.success && ver.output.ok == true {
                        sat = true;
                    }
                }
            }
        }
    }
    complete(#{
        summary: if sat { "o11y taxonomy and drilldown live" } else { "o11y taxonomy incomplete" },
        sat: sat,
        apply: true,
        taxonomy: tax,
        instrument: inst,
        dashboards: d0,
        alerts: d1,
        verify: ver,
    });
}

phase("Verify");
let vp2 = "";
vp2 += hard;
vp2 += "Read-only: taxonomy agent wrote docs/program/CSD-O11Y-TAXONOMY.md or said gaps. ok=true if file exists.\n";
vp2 += "Re-run args.apply=true to instrument labels and Grafana drilldown.\n";
let ver2 = agent(vp2, #{
    label: "verify-doc",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "taxonomy doc path; re-run args.apply=true to instrument and dash",
    apply: false,
    taxonomy: tax,
    verify: ver2,
});
