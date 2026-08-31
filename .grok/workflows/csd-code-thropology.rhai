let meta = #{
    name: "csd-code-thropology",
    description: "Read-only strata map of CogSynDelta: STATUS vs README vs poc vs ADRs vs archive",
    when_to_use: "When swapping into CogSynDelta or asking what is live versus fossil versus marketing",
    phases: [
        #{ title: "Orient", detail: "HEAD, STATUS, pyproject" },
        #{ title: "Strata", detail: "parallel package + ADR + README vs STATUS" },
        #{ title: "Synthesize", detail: "one strata table, next honest increment" },
    ],
};

let orient_schema = #{
    "type": "object",
    "required": ["branch", "sha", "poc_summary", "python_floor"],
    "properties": #{
        "branch": #{ "type": "string" },
        "sha": #{ "type": "string" },
        "poc_summary": #{ "type": "string" },
        "python_floor": #{ "type": "string" },
        "protected": #{ "type": "boolean" },
    },
};

let strata_schema = #{
    "type": "object",
    "required": ["rows"],
    "properties": #{
        "rows": #{
            "type": "array",
            "maxItems": 12,
            "items": #{
                "type": "object",
                "required": ["layer", "path", "verdict"],
                "properties": #{
                    "layer": #{ "type": "string" },
                    "path": #{ "type": "string" },
                    "verdict": #{ "type": "string" },
                    "note": #{ "type": "string" },
                },
            },
        },
    },
};

let synth_schema = #{
    "type": "object",
    "required": ["table_md", "next_increment"],
    "properties": #{
        "table_md": #{ "type": "string" },
        "next_increment": #{ "type": "string" },
        "readme_overclaim": #{ "type": "string" },
    },
};

phase("Orient");
let op = "";
op += "CogSynDelta code-thropology. Read-only. Use read_file and run_terminal_command.\n";
op += "1. git rev-parse --abbrev-ref HEAD && git rev-parse --short HEAD\n";
op += "2. Read STATUS.md (architecture framing + PoC-1/2/3 in-scope tables).\n";
op += "3. Read pyproject.toml [project] name/description/requires-python.\n";
op += "protected=true if branch is main, staging, develop, or dev.\n";
op += "poc_summary = one sentence from STATUS, not README.\n";
op += "python_floor = requires-python string.\n";
let orient = agent(op, #{
    label: "orient",
    capability_mode: "read-only",
    output_schema: orient_schema,
});
if orient == () || !orient.success {
    complete(#{ summary: "orient failed" });
}

phase("Strata");
let jobs = [];
let p1 = "";
p1 += "List src/cogsyndelta/ top-level packages (ls). Open at least poc/__init__.py, ";
p1 += "poc/cli.py, and one of core/, agents/, quantum/. Mark each live / sediment / stub ";
p1 += "with a file you opened. Do not use README. Output rows with layer,path,verdict,note.";
jobs.push(#{
    prompt: p1,
    label: "packages",
    capability_mode: "read-only",
    output_schema: strata_schema,
});
let p2 = "";
p2 += "Read docs/adr/README.md if present, else list docs/adr/*.md titles. ";
p2 += "Note which ADRs the PoC actually implements vs later research (VSA, V-JEPA, ternary). ";
p2 += "Output rows layer=adr.";
jobs.push(#{
    prompt: p2,
    label: "adrs",
    capability_mode: "read-only",
    output_schema: strata_schema,
});
let p3 = "";
p3 += "Diff README.md Overview bullets against STATUS.md in-scope/out-of-scope. ";
p3 += "Each overclaim is a row: layer=readme, path=README.md, verdict=overclaim, note=STATUS quote. ";
p3 += "Do not invent numbers. Compression: STATUS ~1.3-1.6x bytes, not 10x.";
jobs.push(#{
    prompt: p3,
    label: "readme-vs-status",
    capability_mode: "read-only",
    output_schema: strata_schema,
});
let results = parallel(jobs);

phase("Synthesize");
let blob = json_encode(results);
let sp = "";
sp += "Synthesize a strata markdown table for CogSynDelta from these panel outputs (JSON):\n";
sp += blob;
sp += "\nAlso one next honest increment (single PoC slice, not a fleet). ";
sp += "readme_overclaim = shortest list of README bullets STATUS marks out of scope.\n";
sp += "Cite paths you were given. Fail closed if a panel is empty: say unverified, do not fill from memory.";
let synth = agent(sp, #{
    label: "synthesize",
    capability_mode: "read-only",
    output_schema: synth_schema,
});
if synth == () || !synth.success || synth.output.table_md == () {
    complete(#{ summary: "synthesize failed", branch: orient.output.branch });
}
let report = "";
report += "# CogSynDelta strata\n\n";
report += "HEAD ";
report += orient.output.branch;
report += " @ ";
report += orient.output.sha;
report += "\n\n";
report += synth.output.table_md;
report += "\n\n## Next increment\n\n";
report += synth.output.next_increment;
let path = write_scratch_file("report.md", report);
complete(#{
    summary: "strata map",
    path: path,
    branch: orient.output.branch,
    next: synth.output.next_increment,
});
