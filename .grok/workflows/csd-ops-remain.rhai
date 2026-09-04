let meta = #{
    name: "csd-ops-remain",
    description: "Close leftover lab ops after autodev identity: HF token gap, P1-08 stall, CogSynDelta PR CI. Never GitHub push.",
    when_to_use: "After autodev Forgejo bot is live; HF still missing or P1-08/PR#1 is red.",
    phases: [
        #{ title: "Pick", detail: "which leftover is closeable" },
        #{ title: "Close", detail: "one track" },
        #{ title: "Verify", detail: "no operator secrets in CSD vault" },
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

let track = "p1-08";
if args != () {
    let t = as_str(args.track);
    if t != "" { track = t; }
}

phase("Pick");
let pk = "";
pk += "CogSynDelta repo. Read docs/program/CSD-SECRETS.md, HANDOFF-NEXT.md, CONTEXT-PACK-P1-08.md.\n";
pk += "Requested track: " + track + " (hf | p1-08 | pr-ci).\n";
pk += "hf: do NOT copy gpu/huggingface-token. Operator must mint fine-grained hf/autodev for tzervas/cogsyndelta* then secret set into CSD vault.\n";
pk += "p1-08: worktree memory-gate-wt-p1-08; one failing pytest tests/storage/test_tiered_memory_policy.py then TieredMemoryStore.\n";
pk += "pr-ci: Forgejo tzervas/CogSynDelta pulls/1 as user autodev; required checks must have run. Never merge skip-theatre.\n";
pk += "ok=true if the track is closeable this run. evidence=paths.\n";
let pick = agent(pk, #{
    label: "pick-ops",
    capability_mode: "read-only",
    output_schema: ok_schema,
});

phase("Close");
let cp = "";
cp += "Track " + track + ". Isolated worktree. Never main/staging/develop/dev. Never kang-main-wip. Never GitHub bot push.\n";
cp += "Git as ./scripts/csd-autodev-git (identity autodev, CSD vault git/autodev).\n";
if track == "hf" {
    cp += "Do not mint HF tokens via API if unsupported. Write exact operator steps into docs/program/CSD-SECRETS.md if missing. ok=true only if hf/autodev exists in CSD vault or the mint steps are documented and no operator HF token was copied.\n";
}
if track == "p1-08" {
    cp += "In /home/kang/code/personal/tzervas/python-ai/memory-gate-wt-p1-08: one real failing pytest with assert, import memory_gate.storage.sqlite.SqliteDurableStore and MemoryRecord correctly, then implement src/memory_gate/storage/tiered.py. CPU only.\n";
}
if track == "pr-ci" {
    cp += "Inspect https://git.vectorweight.com/tzervas/CogSynDelta/pulls/1 failed jobs (ruff/pytest/trivy/gpu-5080). Fix on feat/agent-harness. Push with csd-autodev-git. Do not merge until required checks ran and succeeded.\n";
}
cp += "ok=true with evidence. Never write tzervas-dev-kb.\n";
let closed = agent(cp, #{
    label: "close-ops",
    capability_mode: "execute",
    output_schema: ok_schema,
});

phase("Verify");
let vp = "";
vp += "CSD vault /akula-data/cabal/csd-vault secret ls must be only gpu/localai-api-key, csd/apply-token, git/autodev, optional hf/autodev. No cabal-forgejo-admin.\n";
vp += "ok=true if allowlist holds. evidence=key names only.\n";
let ver = agent(vp, #{
    label: "verify-vault",
    capability_mode: "execute",
    output_schema: ok_schema,
});

let sat = false;
if pick != () && pick.success && pick.output.ok == true {
    if closed != () && closed.success && closed.output.ok == true {
        if ver != () && ver.success && ver.output.ok == true {
            sat = true;
        }
    }
}
complete(#{
    summary: if sat { "ops leftover closed" } else { "ops leftover remains" },
    sat: sat,
    track: track,
    pick: pick,
    close: closed,
    verify: ver,
});
