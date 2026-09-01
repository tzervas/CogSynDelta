let meta = #{
    name: "csd-scale-followup",
    description: "Append brain-region curriculum onto completed Regions artifacts, then rewrite Scale: region-pretrain -> router -> assembly -> size ladder + HF + Grafana email.",
    when_to_use: "program-drive Regions already wrote CSD-REGION-DATASETS.md; operator clarified regions are specialized submodels that pretrain first.",
    phases: [
        #{ title: "Append", detail: "patch datasets doc + CSD-BRAIN-REGIONS.md" },
        #{ title: "Scale", detail: "ladder with region-pretrain as rung 0" },
        #{ title: "Steer", detail: "autodev next = one region pretrain" },
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
hard += "HARD: Regions are specialized submodels of one CSD mind. They MUST pretrain on their own datasets BEFORE whole-model train. ";
hard += "STATUS.md: two live PoC regions + softmax router. Not mHC/VL-JEPA/10x. Autodev implements. Never GitHub. feat/agent-harness. ";
hard += "1080 Ti guest live for retrieve-index-light; 5080 CUDA; 3090 local/code. Never dual 14B. hf/autodev if missing = document mint, never copy operator HF token.\n";
hard += "Append docs/program/CSD-REGION-DATASETS.md (already exists from program-drive Regions) with an explicit Pretrain-before-assembly section. ";
hard += "Do not delete cited datasets; add the curriculum order.\n";

phase("Append");
let ap = "";
ap += hard;
ap += "Read CSD-REGION-DATASETS.md and STATUS.md. Write/merge CSD-BRAIN-REGIONS.md if missing. ";
ap += "Append a section: Curriculum (1) per-region pretrain (2) router (3) assembled whole-model (4) scale. ";
ap += "Each existing dataset row: mark pretrain vs later-assembly split. ok=true if the append is in git working tree.\n";
let appended = agent(ap, #{
    label: "append-curriculum",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Scale");
    let sp = "";
    sp += hard;
    sp += "Rewrite docs/program/CSD-SCALE-LADDER.md so rungs are: ";
    sp += "R0 tiny region-pretrain (one specialist, CPU/PoC or 5080 smoke); ";
    sp += "R1 router/interconnect on frozen-then-joint regions; ";
    sp += "R2 assembled tiny whole-mind; ";
    sp += "R3+ larger param counts. Each rung: recipe, eval, private HF tzervas/cogsyndelta-region-<name>-<size> then tzervas/cogsyndelta-<size>. ";
    sp += "Grafana maintainers-email only when a rung has a real STATUS row or pytest/CUDA receipt — no fake sat.\n";
    sp += "ok=true if the ladder starts with region-pretrain.\n";
    let scaled = agent(sp, #{
        label: "rewrite-ladder",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Steer");
    let tp = "";
    tp += hard;
    tp += "Steer autodev: next_goal = pretrain the LatentVAE or ResidualMLP region on the matching public split from the datasets doc, with a failing test first. ";
    tp += "Not 14B assembly. Commit docs as autodev; PR if needed; do not merge skip-theatre.\n";
    tp += "ok=true if CSD_STEER or /api/steer reflects region-pretrain.\n";
    let steered = agent(tp, #{
        label: "steer-r0",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if appended != () && appended.success && appended.output.ok == true {
        if scaled != () && scaled.success && scaled.output.ok == true {
            if steered != () && steered.success && steered.output.ok == true {
                sat = true;
            }
        }
    }
    complete(#{
        summary: if sat { "region curriculum appended; scale ladder starts at pretrain" } else { "scale follow-up incomplete" },
        sat: sat,
        apply: true,
        append: appended,
        scale: scaled,
        steer: steered,
    });
}

phase("Scale");
let vp2 = "";
vp2 += "Read-only. ok=true if CSD-REGION-DATASETS.md exists. Re-run args.apply=true to append+rewrite ladder.\n";
let ver2 = agent(vp2, #{
    label: "exists-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "datasets exist; re-run args.apply=true",
    apply: false,
    append: appended,
    verify: ver2,
});
