let meta = #{
    name: "csd-autodev-forgejo-tools",
    description: "Expose Forgejo git to autodev via csd-autodev-git + csd-autodev-forgejo. Identity autodev only. Never operator admin. Never GitHub.",
    when_to_use: "Autodev needs PR/issue/status/wait-ci tools, or git push is still using cabal-forgejo-admin.",
    phases: [
        #{ title: "Identity", detail: "whoami autodev; allowlist repos" },
        #{ title: "Wire", detail: "unit TOKEN=git/autodev; CLI tools executable" },
        #{ title: "Verify", detail: "prs list + git ls-remote as autodev" },
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

phase("Identity");
let ip = "";
ip += "CogSynDelta. SECRET_VAULT=/akula-data/cabal/csd-vault. secret ls must include git/autodev and must NOT include cabal-forgejo-admin.\n";
ip += "secret exec TOKEN=git/autodev -- ./scripts/csd-autodev-forgejo whoami — login must be autodev.\n";
ip += "Allowlist only tzervas/CogSynDelta and tzervas/memory-gate. evidence=login. Never print token.\n";
let ident = agent(ip, #{
    label: "whoami",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Wire");
    let wp = "";
    wp += "Ensure deploy/systemd/csd-autodev-loop.service has TOKEN=git/autodev, GIT_USERNAME=autodev, GIT_ASKPASS git-askpass-token.\n";
    wp += "chmod +x scripts/csd-autodev-git scripts/csd-autodev-forgejo. systemctl --user daemon-reload && restart csd-autodev-loop if unit changed.\n";
    wp += "Do not put operator admin in CSD vault. ok=true if process environ has GIT_USERNAME=autodev and TOKEN set (name only). evidence=env names.\n";
    let wired = agent(wp, #{
        label: "wire-unit",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    phase("Verify");
    let vp = "";
    vp += "secret exec TOKEN=git/autodev -- ./scripts/csd-autodev-forgejo prs memory-gate --limit 5\n";
    vp += "./scripts/csd-autodev-git ls-remote https://git.vectorweight.com/tzervas/memory-gate.git HEAD\n";
    vp += "ok=true if both succeed. evidence=PR numbers and HEAD sha. Never GitHub.\n";
    let ver = agent(vp, #{
        label: "verify-tools",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if ident != () && ident.success && ident.output.ok == true {
        if wired != () && wired.success && wired.output.ok == true {
            if ver != () && ver.success && ver.output.ok == true {
                sat = true;
            }
        }
    }
    complete(#{
        summary: if sat { "autodev Forgejo tools live" } else { "autodev Forgejo tools incomplete" },
        sat: sat,
        identity: ident,
        wire: wired,
        verify: ver,
    });
}

phase("Verify");
let vp2 = "";
vp2 += "Read-only: confirm scripts/csd-autodev-git and scripts/csd-autodev-forgejo exist and skill akula-lab documents them. ok=true if both files exist.\n";
let ver2 = agent(vp2, #{
    label: "verify-files",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "autodev Forgejo tools (read-only); re-run args.apply=true to wire units",
    identity: ident,
    verify: ver2,
});
