let meta = #{
    name: "csd-branch-pr-review",
    description: "Classify CogSynDelta open PRs and trunks: PoC-on-main vs develop sediment vs CI",
    when_to_use: "When reviewing CogSynDelta PRs, asking what fits the PoC, or refreshing docs/GROK-STRATA.md",
    phases: [
        #{ title: "Trunks", detail: "main vs develop vs dev vs staging" },
        #{ title: "PRs", detail: "open PRs targeting main vs develop" },
        #{ title: "Fit", detail: "fits / does not fit STATUS PoC" },
    ],
};

let trunk_schema = #{
    "type": "object",
    "required": ["main_sha", "verdict"],
    "properties": #{
        "main_sha": #{ "type": "string" },
        "develop_sha": #{ "type": "string" },
        "dev_sha": #{ "type": "string" },
        "verdict": #{ "type": "string" },
        "note": #{ "type": "string" },
    },
};

let pr_schema = #{
    "type": "object",
    "required": ["rows"],
    "properties": #{
        "rows": #{
            "type": "array",
            "maxItems": 20,
            "items": #{
                "type": "object",
                "required": ["number", "base", "head", "fit"],
                "properties": #{
                    "number": #{ "type": "string" },
                    "base": #{ "type": "string" },
                    "head": #{ "type": "string" },
                    "fit": #{ "type": "string" },
                    "why": #{ "type": "string" },
                },
            },
        },
    },
};

let fit_schema = #{
    "type": "object",
    "required": ["summary_md", "next"],
    "properties": #{
        "summary_md": #{ "type": "string" },
        "next": #{ "type": "string" },
    },
};

phase("Trunks");
let tp = "";
tp += "Read-only. Repo CogSynDelta. Read docs/GROK-STRATA.md and STATUS.md.\n";
tp += "git fetch origin if needed. Record origin/main, origin/develop, origin/dev SHAs.\n";
tp += "git rev-list --left-right --count origin/main...origin/develop\n";
tp += "verdict=TWO_PROGRAMS if develop is not an ancestor of main. Else ONE_LINE.\n";
let trunks = agent(tp, #{
    label: "trunks",
    capability_mode: "read-only",
    output_schema: trunk_schema,
});

phase("PRs");
let jobs = [];
let a = "";
a += "gh pr list --repo tzervas/CogSynDelta --state open --base main --json number,title,headRefName,baseRefName\n";
a += "Also view #59 if still open. fit=yes only if the PR is PoC CUDA/CI on current main. ";
a += "fit=stale if far behind PoC commits. Output rows.";
jobs.push(#{ prompt: a, label: "prs-main", capability_mode: "read-only", output_schema: pr_schema });
let b = "";
b += "gh pr list --repo tzervas/CogSynDelta --state open --base develop --limit 20 --json number,title,headRefName\n";
b += "Classify the cluster: Jules/maintenance/quality vs real feature. Default fit=no for quality-bot churn. Output rows (cap 12, summarize the rest in why of row 1).";
jobs.push(#{ prompt: b, label: "prs-develop", capability_mode: "read-only", output_schema: pr_schema });
let prs = parallel(jobs);

phase("Fit");
let blob = json_encode(#{ trunks: trunks, prs: prs });
let fp = "";
fp += "Using this JSON, write summary_md: what fits Program A (PoC-on-main / STATUS.md) and what does not.\n";
fp += blob;
fp += "\nnext = one operator action (e.g. rebase #59, do not merge develop). Fail closed if panels empty.";
let fit = agent(fp, #{
    label: "fit",
    capability_mode: "read-only",
    output_schema: fit_schema,
});
if fit == () || !fit.success || fit.output.summary_md == () {
    complete(#{ summary: "fit failed" });
}
let path = write_scratch_file("report.md", fit.output.summary_md);
complete(#{ summary: "branch/PR fit", path: path, next: fit.output.next });
