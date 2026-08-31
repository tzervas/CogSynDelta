let meta = #{
    name: "csd-grok-idle-ping",
    description: "Grok sits idle until autodev files a short need_grok ping. No Grok scheduler. Autodev pick-implement-PR-wait-merge-next. Quota-protect.",
    when_to_use: "Weekly hosted Grok quota low; autodev must run unsupervised and only wake Grok on a real blocker.",
    phases: [
        #{ title: "Protocol", detail: "need_grok JSON + lab API; never dump a session" },
        #{ title: "Loop", detail: "autodev-loop: pick, PR, wait-ci, merge-when-green, next" },
        #{ title: "Alert", detail: "Grafana/email when need_grok set; no 60s Grok poll" },
    ],
};

let ok_schema = #{
    "type": "object",
    "required": ["ok", "notes", "evidence"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
        "evidence": #{ "type": "string" },
    },
};

let do_apply = false;
if args != () {
    if args.apply == true { do_apply = true; }
}

let hard = "";
hard += "HARD: Do NOT create a Grok scheduler or poll hosted Grok. That burns quota. ";
hard += "Autodev is local (3090 local/code + Forgejo). Grok wakes only when the operator opens a session after ping. ";
hard += "Ping payload max ~2KB: question, blocker, paths, what was tried. No transcripts. ";
hard += "Identity autodev. Never GitHub. Never dual 14B. Never pause 3090 LocalAI. Merge only when required checks ran.\n";
hard += "1080 Ti: use only if router live=true and guest smi. Else 3090+5080.\n";

phase("Protocol");
let pp = "";
pp += hard;
pp += "Write docs/program/CSD-GROK-PING.md: file /akula-data/cabal/csd-need-grok.json {need:true, question, evidence, goal}. ";
pp += "Lab POST /api/grok-need and GET /api/grok-need. Autodev sets need=true only when stuck after N failed CI or missing capability (web research, HF mint, hardware). ";
pp += "Grok reads that file when the human opens chat — do not auto-call the hosted API.\n";
pp += "ok=true if the doc exists.\n";
let proto = agent(pp, #{
    label: "ping-doc",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Loop");
    let lp = "";
    lp += hard;
    lp += "Ensure csd-autodev-loop --worker: pick next_goal from CSD_STEER/PHASE-1, implement, csd-autodev-git commit/push, ";
    lp += "csd-autodev-forgejo pr-create/status/wait, merge-gate only when checks ran, then next_goal. ";
    lp += "If it cannot run a job, launch a *local* workflow name (csd-python-first-drive) — not hosted Grok. ";
    lp += "On real blocker write csd-need-grok.json. Wire /api/grok-need on csd-lab-console. Goals tab shows the ping.\n";
    lp += "ok=true if loop is active and ping file/API exist.\n";
    let autoloop = agent(lp, #{
        label: "wire-loop-ping",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Alert");
    let ap = "";
    ap += hard;
    ap += "Grafana or send-only postfix: one alert to maintainers@ when need_grok.json need=true (file mtime or a tiny exporter). ";
    ap += "Do not fire on every loop tick. Do not schedule Grok. Commit as autodev; do not merge skip-theatre.\n";
    ap += "ok=true if ping protocol is in git and loop still running.\n";
    let alert = agent(ap, #{
        label: "ping-alert",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if proto != () && proto.success && proto.output.ok == true {
        if autoloop != () && autoloop.success && autoloop.output.ok == true {
            if alert != () && alert.success && alert.output.ok == true {
                sat = true;
            }
        }
    }
    complete(#{
        summary: if sat { "Grok idle; autodev pings via need_grok only" } else { "idle-ping incomplete" },
        sat: sat,
        apply: true,
        protocol: proto,
        autoloop: autoloop,
        alert: alert,
    });
}

phase("Alert");
let vp2 = "";
vp2 += "Read-only. ok=true if CSD-GROK-PING.md exists. Re-run args.apply=true to wire API+loop.\n";
let ver2 = agent(vp2, #{
    label: "doc-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "ping doc; re-run args.apply=true",
    apply: false,
    protocol: proto,
    verify: ver2,
});
