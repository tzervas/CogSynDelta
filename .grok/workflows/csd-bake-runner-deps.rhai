let meta = #{
    name: "csd-bake-runner-deps",
    description: "Bake Node 24 LTS onto Forgejo runners and pin GitHub Actions to latest majors (checkout@v7 not v7.0.1). Deps on the host, not at clone.",
    when_to_use: "gpu-5080 dies with node missing in PATH, or workflow uses pin patches like @v7.0.1, or runner tools are not baked.",
    phases: [
        #{ title: "Audit", detail: "node on hosts; uses: pins; Forgejo action mirrors" },
        #{ title: "Bake", detail: "official Node 24 tarball to /usr/local; restart runners" },
        #{ title: "Pin", detail: "restore checkout@v7; major tags only; push as autodev" },
        #{ title: "Verify", detail: "node -v on runner PATH; no patch pins; no git-init workaround" },
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
ap += "CogSynDelta repo /home/kang/code/personal/tzervas/CogSynDelta on feat/agent-harness.\n";
ap += "Use tools. Do not invent. Never pause 3090 LocalAI. Never GitHub bot push. Never merge PRs.\n";
ap += "1. Grep .github/workflows for uses: lines. List every action pin. Flag any patch pin like @v7.0.1 (must become @v7). Flag gpu-5080.yml git-init workaround.\n";
ap += "2. Probe which majors Forgejo can fetch: curl -sI https://code.forgejo.org/actions/checkout/archive/v7.tar.gz and data.forgejo.org equivalents. checkout@v7 is current. Do not recommend @v7.0.1.\n";
ap += "3. For astral-sh/setup-uv, actions/upload-artifact, actions/download-artifact: find the newest major that actually returns HTTP 200 from the Forgejo action mirror. If v10/v7 404, keep the highest mirrored major. Never re-add codecov/*.\n";
ap += "4. SSH BatchMode: gpu5080 (192.168.1.252) and homelab (192.168.1.170). Report command -v node, node -v, PATH, id. Runner units: forgejo-runner-gpu-tzervas on gpu5080; forgejo-runner-cpu on homelab. gpu5080 currently has no node.\n";
ap += "ok=true if inventory is complete with host evidence. evidence=uses pins + node versions + HTTP codes.\n";
let audit = agent(ap, #{
    label: "audit-pins-node",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Bake");
    let bake_common = "";
    bake_common += "Bake Node latest 24 LTS (stable for CI; checkout@v7 needs Node 24+) via official tarball.\n";
    bake_common += "Script: /home/kang/code/personal/tzervas/CogSynDelta/scripts/csd-bake-runner-node\n";
    bake_common += "Do NOT use NodeSource apt (SHA1 mirrors). Do NOT nvm. Install to /usr/local so runner PATH /usr/local/bin wins.\n";
    bake_common += "Resolve version: curl https://nodejs.org/dist/index.json and pick the newest lts with major 24 (example 24.20.0). Set NODE_LTS_VERSION.\n";
    bake_common += "chmod +x the bake script. Copy it to the host, run with sudo for tar to /usr/local, verify node -v and npm -v.\n";
    bake_common += "Then restart ONLY the Forgejo runner unit so jobs inherit PATH. Do not stop LocalAI, Comfy, Caddy, or Grafana.\n";
    bake_common += "ok=true only if `node -v` on that host as the runner user prints v24.x and `command -v node` is /usr/local/bin/node.\n";
    bake_common += "Never pause 3090. Never bind 0.0.0.0 WAN. Never write operator vaults.\n";

    let bg = bake_common;
    bg += "HOST=gpu5080 (192.168.1.252), user tzervas. ssh -o BatchMode=yes gpu5080.\n";
    bg += "Unit: forgejo-runner-gpu-tzervas.service (system or user — inspect before restart).\n";
    bg += "PATH expected /home/tzervas/.local/bin:/usr/local/bin:/usr/bin:/bin.\n";
    let bh = bake_common;
    bh += "HOST=homelab (192.168.1.170). ssh -o BatchMode=yes homelab or 192.168.1.170.\n";
    bh += "Unit: forgejo-runner-cpu. Currently node v22 in /usr/bin — bake 24 into /usr/local so PATH prefers it.\n";

    let baked = parallel([
        #{
            prompt: bg,
            label: "bake-gpu5080",
            capability_mode: "execute",
            output_schema: ok_schema,
        },
        #{
            prompt: bh,
            label: "bake-homelab",
            capability_mode: "execute",
            output_schema: ok_schema,
        },
    ]);

    phase("Pin");
    let pp = "";
    pp += "CogSynDelta /home/kang/code/personal/tzervas/CogSynDelta branch feat/agent-harness. Isolated? No — edit this tree.\n";
    pp += "Never main/staging/develop/dev. Never GitHub bot push. Never merge PR #1 or memory-gate #11.\n";
    pp += "1. Restore .github/workflows/gpu-5080.yml first steps to:\n";
    pp += "   - uses: actions/checkout@v7\n";
    pp += "   Remove the git init/fetch workaround and the comment that checkout dies without node.\n";
    pp += "2. Pin majors that actually run on Forgejo/act, not GitHub-latest. checkout@v7 and setup-uv@v7. upload-artifact and download-artifact MUST stay @v3 — v4+ raise GHESNotSupportedError on this runner. Never @v7.0.1 patch pins. Do not re-add codecov. Leave @master third-party (snyk/sonar) unless a versioned major exists on the mirror.\n";
    pp += "3. scripts/csd-bake-runner-node must be executable and committed if not already.\n";
    pp += "4. Commit conventional: ci(ci): bake Node 24 on runners and pin actions majors. Push with ./scripts/csd-autodev-git to Forgejo origin feat/agent-harness. Never GitHub.\n";
    pp += "ok=true if gpu-5080.yml uses checkout@v7, no patch pins remain, push succeeded. evidence=sha + grep of uses: lines.\n";
    let pinned = agent(pp, #{
        label: "pin-majors",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Verify");
    let vp = "";
    vp += "Independent check. Use tools.\n";
    vp += "1. ssh gpu5080: command -v node; node -v — must be /usr/local/bin/node and v24.x\n";
    vp += "2. ssh homelab: same, v24.x preferred via /usr/local/bin\n";
    vp += "3. grep -n 'uses:' .github/workflows/*.yml — no @vN.N.N patch pins; gpu-5080.yml has uses: actions/checkout@v7; no git-init workaround.\n";
    vp += "4. Do not merge. Do not claim CI green unless you listed actual Forgejo check runs for the new sha.\n";
    vp += "ok=true only with host command output and grep evidence.\n";
    let ver = agent(vp, #{
        label: "verify-baked",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    let bake0 = ();
    let bake1 = ();
    if baked != () {
        if baked.len() > 0 { bake0 = baked[0]; }
        if baked.len() > 1 { bake1 = baked[1]; }
    }
    let sat = false;
    if audit != () && audit.success && audit.output.ok == true {
        if bake0 != () && bake0.success && bake0.output.ok == true {
            if bake1 != () && bake1.success && bake1.output.ok == true {
                if pinned != () && pinned.success && pinned.output.ok == true {
                    if ver != () && ver.success && ver.output.ok == true {
                        sat = true;
                    }
                }
            }
        }
    }
    complete(#{
        summary: if sat { "runners have Node 24; actions pinned to majors" } else { "bake or pin incomplete" },
        sat: sat,
        apply: true,
        audit: audit,
        bake_gpu5080: bake0,
        bake_homelab: bake1,
        pin: pinned,
        verify: ver,
    });
}

phase("Verify");
let vp2 = "";
vp2 += "Read-only: confirm scripts/csd-bake-runner-node exists and .github/workflows/gpu-5080.yml still documents missing node or checkout@v7. ok=true if bake script exists.\n";
vp2 += "Re-run with args.apply=true to install Node 24 on gpu5080+homelab and restore uses: actions/checkout@v7.\n";
let ver2 = agent(vp2, #{
    label: "verify-script",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "audit-only; re-run args.apply=true to bake Node and pin majors",
    apply: false,
    audit: audit,
    verify: ver2,
});
