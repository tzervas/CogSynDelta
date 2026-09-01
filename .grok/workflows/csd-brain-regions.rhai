let meta = #{
    name: "csd-brain-regions",
    description: "CSD brain regions: specialized submodels pretrain on own datasets, then router/interconnect, then whole-mind train. Tiny first. STATUS.md is authority.",
    when_to_use: "Clarifying regions as cognitive specialists in one mind; need dataset map and next autodev increment.",
    phases: [
        #{ title: "Map", detail: "live PoC regions vs intended brain map" },
        #{ title: "Datasets", detail: "one public dataset per intended region" },
        #{ title: "Steer", detail: "autodev: pretrain one region, not whole 14B" },
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
hard += "HARD: Regions = modules of ONE mind (MoE experts), not a swarm of agents. STATUS.md: PoC-1 LatentVAE; PoC-2 ResidualMLP + LatentVAE + softmax top-k; PoC-3 trains the gate. Not mHC, not VL-JEPA, not 10x.\n";
hard += "Order: region-pretrain on specialized data -> train router/interconnect (selective activation) -> train assembled model -> scale sizes to private HF.\n";
hard += "Autodev implements. Grok researches datasets. Never dual 14B. Never GitHub. feat/agent-harness. 1080 Ti guest live for RAG/light if nvidia-smi; 5080 CUDA; 3090 local/code.\n";

phase("Map");
let mp = "";
mp += hard;
mp += "Read STATUS.md, src/cogsyndelta/poc/regions.py, router.py, train.py, train_route.py. ";
mp += "Write docs/program/CSD-BRAIN-REGIONS.md: live regions vs intended specialists (e.g. compress/memory, residual stream, future code/retrieve). ";
mp += "Each row: name, job, pretrain-before-assembly, router trigger. ok=true if file exists and does not claim unimplemented regions are live.\n";
let mapped = agent(mp, #{
    label: "map-regions",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Datasets");
    let dp = "";
    dp += hard;
    dp += "web_search PUBLIC datasets for each INTENDED region (not just the two PoC modules). License MIT/Apache/CC compatible. Size that fits tiny-then-scale. ";
    dp += "Write/extend docs/program/CSD-REGION-DATASETS.md: region, dataset, URL, license, why this specialist, split for pretrain vs later assembly. Cite sources.\n";
    dp += "ok=true if >=3 regions have a cited dataset.\n";
    let data = agent(dp, #{
        label: "region-data",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Steer");
    let sp = "";
    sp += hard;
    sp += "POST lab /api/steer or CSD_STEER: next closeable work is PRETRAIN one existing region (LatentVAE or ResidualMLP) on a real public split, with a failing test first — not whole-model 14B. ";
    sp += "Keep autodev-loop on that goal. 1080 Ti for retrieve-index-light if live; 5080 for tiny CUDA; 3090 for code. ";
    sp += "Commit docs as autodev; do not merge skip-theatre.\n";
    sp += "ok=true if steer next_goal mentions region-pretrain or P1 plus region.\n";
    let steered = agent(sp, #{
        label: "steer-pretrain",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if mapped != () && mapped.success && mapped.output.ok == true {
        if data != () && data.success && data.output.ok == true {
            if steered != () && steered.success && steered.output.ok == true {
                sat = true;
            }
        }
    }
    complete(#{
        summary: if sat { "brain-region curriculum set" } else { "brain-region curriculum incomplete" },
        sat: sat,
        apply: true,
        map: mapped,
        datasets: data,
        steer: steered,
    });
}

phase("Steer");
let vp2 = "";
vp2 += "Read-only map. ok=true if CSD-BRAIN-REGIONS.md exists. Re-run args.apply=true for datasets+steer.\n";
let ver2 = agent(vp2, #{
    label: "map-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "map-only; re-run args.apply=true",
    apply: false,
    map: mapped,
    verify: ver2,
});
