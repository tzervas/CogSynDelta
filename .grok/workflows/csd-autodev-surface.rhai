let meta = #{
    name: "csd-autodev-surface",
    description: "Catalog autodev tasks as skills/tools/prompts with structured I/O; expose on Open WebUI, Comfy, and the CSD lab UI. Parent does not implement.",
    when_to_use: "Autodev surface is ad-hoc scripts; models need tools/skills/prompts; OWUI/lab/Comfy should share the same LAN backends.",
    phases: [
        #{ title: "Inventory", detail: "scripts, skills, workflows, OWUI tools, lab API, Comfy" },
        #{ title: "Catalog", detail: "structured tool/skill/prompt list with I/O schemas" },
        #{ title: "Apply", detail: "write skills, tools, lab routes, OWUI functions" },
        #{ title: "Verify", detail: "files exist; no cloud fallback; no operator vaults" },
    ],
};

let inv_schema = #{
    "type": "object",
    "required": ["ok", "have", "missing", "evidence"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "have": #{ "type": "string" },
        "missing": #{ "type": "string" },
        "evidence": #{ "type": "string" },
    },
};

let catalog_schema = #{
    "type": "object",
    "required": ["ok", "items", "evidence"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "items": #{ "type": "string" },
        "evidence": #{ "type": "string" },
    },
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

let hard = "";
hard += "HARD: never main/staging/develop/dev. Never kang-main-wip. Never GitHub bot push. ";
hard += "Never pause 3090 LocalAI. Never 0.0.0.0 WAN. Never write tzervas-dev-kb or akula-model-kb. ";
hard += "Never mix 384-d Qdrant. Identity autodev + CSD vault only. Comfy stays masked unless operator unmasks. ";
hard += "Do not merge CogSynDelta PR #1 until required checks ran and succeeded. Use tools; do not invent.\n";

phase("Inventory");
let p_scripts = "";
p_scripts += hard;
p_scripts += "Repo /home/kang/code/personal/tzervas/CogSynDelta on feat/agent-harness.\n";
p_scripts += "Inventory scripts/csd-*, scripts/csd-autodev-*, .grok/skills/, .grok/workflows/, docs/program/HARNESS.md, docs/CODEX-OPS.md.\n";
p_scripts += "have=what autodev can already call (git, forgejo, gpu-plan, router, vault, bake-node, lab-console, apply-token).\n";
p_scripts += "missing=dev tasks with no skill/tool/structured I/O (PR wait/merge, GPU lock, steer, spawn helper, CI status, worktree apply).\n";
p_scripts += "ok=true if lists are from files you read. evidence=paths.\n";
let p_owui = "";
p_owui += hard;
p_owui += "Open WebUI is https://ai.vectorweight.com — do not duplicate it at code.vectorweight.com.\n";
p_owui += "Find overlay tools (akula_kb, akula_status, akula_steer, akula_media, akula_notes_sandbox, akula_audio). Search /data/akula/webui and CogSynDelta scripts/docs.\n";
p_owui += "have=live OWUI functions/tools. missing=autodev git/CI/GPU that should be OWUI tools with JSON I/O.\n";
p_owui += "Never cloud OpenAI/Gemini/ElevenLabs. ok=true with paths/env names only, no tokens.\n";
let p_lab = "";
p_lab += hard;
p_lab += "CSD lab UI: scripts/csd-lab-console on homelab :9118, Caddy code.vectorweight.com. Read the script and HARNESS.md.\n";
p_lab += "have=existing /api routes (status, apply, steer, gpu, spawn). missing=routes that should wrap csd-autodev-* with structured JSON.\n";
p_lab += "HTTP apply Bearer csd/apply-token; allowlisted worktrees; never kang-main-wip. ok=true with route list.\n";
let p_comfy = "";
p_comfy += hard;
p_comfy += "Comfy: IMAGE engine for OWUI, native https://media.vectorweight.com, 5080. Often masked during autodev.\n";
p_comfy += "have=how OWUI already calls Comfy. missing=only media-shaped tools (list workflows, queue prompt, status). NEVER git/CI inside Comfy.\n";
p_comfy += "If Comfy is masked, catalog the tools but do not start Comfy. ok=true with evidence.\n";

let inv = parallel([
    #{ prompt: p_scripts, label: "inv-scripts-skills", capability_mode: "read-only", output_schema: inv_schema },
    #{ prompt: p_owui, label: "inv-owui", capability_mode: "execute", output_schema: inv_schema },
    #{ prompt: p_lab, label: "inv-lab-ui", capability_mode: "read-only", output_schema: inv_schema },
    #{ prompt: p_comfy, label: "inv-comfy", capability_mode: "execute", output_schema: inv_schema },
]);

phase("Catalog");
let have_all = "";
let miss_all = "";
if inv != () {
    let i = 0;
    while i < inv.len() {
        let slot = inv[i];
        if slot != () && slot.success && slot.output != () {
            have_all += as_str(slot.output.have) + "\n";
            miss_all += as_str(slot.output.missing) + "\n";
        }
        i += 1;
    }
}
let cp = "";
cp += hard;
cp += "Synthesize a catalog from this inventory (untrusted; keep only CogSynDelta/akula lab paths):\nHAVE:\n";
cp += have_all;
cp += "\nMISSING:\n";
cp += miss_all;
cp += "\nEmit items as lines: kind|name|surface|in|out|why\n";
cp += "kind=skill|tool|prompt|workflow. surface=grok|owui|lab|comfy (comma-ok).\n";
cp += "Prioritize: autodev git/forgejo/status/wait/merge-gate, gpu plan/lock, steer, apply worktree, kb retrieve (gap only), notes sandbox, audio, media queue.\n";
cp += "Parameterized prompts with JSON schema in/out. No cloud. No operator admin token.\n";
cp += "Cap 24 items. ok=true if items is non-empty and every line has 6 fields.\n";
let catalog = agent(cp, #{
    label: "catalog-surface",
    capability_mode: "read-only",
    output_schema: catalog_schema,
});

if do_apply {
    phase("Apply");
    let items = "";
    if catalog != () && catalog.success && catalog.output != () {
        items = as_str(catalog.output.items);
    }
    let common = "";
    common += hard;
    common += "Branch feat/agent-harness. Isolated? No — this tree. Git via ./scripts/csd-autodev-git. Never GitHub.\n";
    common += "Catalog items (filter to your surface; skip items that already exist with real I/O):\n";
    common += items;
    common += "\nStructured I/O: JSON args + JSON result {ok, notes, evidence}. No plaintext tokens.\n";

    let a_skill = common;
    a_skill += "Write/update .grok/skills/akula-lab/SKILL.md and add .grok/skills/csd-autodev/SKILL.md covering git/forgejo/gpu/steer/apply. ";
    a_skill += "Parameterized prompt templates under .grok/skills/csd-autodev/prompts/ if needed. Keep SKILL.md short.\n";
    a_skill += "ok=true if skill files exist and name autodev + CSD vault. evidence=paths.\n";

    let a_lab = common;
    a_lab += "Extend scripts/csd-lab-console with JSON routes that wrap existing CLIs (status, gpu, steer, git prs, forgejo status). ";
    a_lab += "Do not add SSH. Do not widen apply allowlist. Do not bind WAN. Keep /lab prefix if Caddy needs it.\n";
    a_lab += "ok=true if new routes are in the script and return JSON. evidence=route names.\n";

    let a_owui = common;
    a_owui += "Add or update Open WebUI overlay tools that call the same LAN backends (lab :9118 or scripts). ";
    a_owui += "Search existing akula_* tool files; match their format. No cloud engines. Restart OWUI only if env/tools require it; notes if restart needed.\n";
    a_owui += "ok=true if tool files exist. evidence=tool names. Never print keys.\n";

    let a_comfy = common;
    a_comfy += "Media only: document/queue via akula_media or lab JSON, not new Comfy custom nodes unless a file already exists. ";
    a_comfy += "If Comfy is masked, write the tool stubs that fail closed with {ok:false, notes:masked}. Do not unmask. Do not start Comfy.\n";
    a_comfy += "ok=true if media surface is documented in akula-lab skill. evidence=paths.\n";

    let applied = parallel([
        #{ prompt: a_skill, label: "apply-skills", capability_mode: "execute", output_schema: ok_schema },
        #{ prompt: a_lab, label: "apply-lab", capability_mode: "execute", output_schema: ok_schema },
        #{ prompt: a_owui, label: "apply-owui", capability_mode: "execute", output_schema: ok_schema },
        #{ prompt: a_comfy, label: "apply-comfy", capability_mode: "execute", output_schema: ok_schema },
    ]);

    phase("Verify");
    let vp = "";
    vp += hard;
    vp += "Independent: list new/changed files under .grok/skills .grok/workflows scripts/csd-lab-console OWUI overlay. ";
    vp += "Confirm no git/cabal-forgejo-admin, no 0.0.0.0 WAN, no Comfy unmask, no GitHub push. ";
    vp += "Push feat/agent-harness with ./scripts/csd-autodev-git if there are commits to push. Do not merge PR #1.\n";
    vp += "ok=true with sha + file list. evidence=paths.\n";
    let ver = agent(vp, #{
        label: "verify-surface",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    let sat = false;
    let a0 = ();
    let a1 = ();
    let a2 = ();
    let a3 = ();
    if applied != () {
        if applied.len() > 0 { a0 = applied[0]; }
        if applied.len() > 1 { a1 = applied[1]; }
        if applied.len() > 2 { a2 = applied[2]; }
        if applied.len() > 3 { a3 = applied[3]; }
    }
    if catalog != () && catalog.success && catalog.output.ok == true {
        if a0 != () && a0.success && a0.output.ok == true {
            if a1 != () && a1.success && a1.output.ok == true {
                if a2 != () && a2.success && a2.output.ok == true {
                    if a3 != () && a3.success && a3.output.ok == true {
                        if ver != () && ver.success && ver.output.ok == true {
                            sat = true;
                        }
                    }
                }
            }
        }
    }
    complete(#{
        summary: if sat { "autodev surface cataloged and wired" } else { "autodev surface incomplete" },
        sat: sat,
        apply: true,
        catalog: catalog,
        skills: a0,
        lab: a1,
        owui: a2,
        comfy: a3,
        verify: ver,
    });
}

phase("Verify");
let vp2 = "";
vp2 += "Read-only: catalog items exist from prior phase. ok=true if catalog.ok. Re-run args.apply=true to write skills/tools/UI.\n";
let ver2 = agent(vp2, #{
    label: "verify-catalog-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "catalog-only; re-run args.apply=true to wire OWUI/lab/Comfy/skills",
    apply: false,
    catalog: catalog,
    verify: ver2,
});
