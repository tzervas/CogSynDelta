let meta = #{
    name: "csd-goal-loop",
    description: "Do-while on program goals: drive, stall-switch, sandbox unblock via gatekeeper, pause if unsafe",
    when_to_use: "Autoloop toward GOALS.md until met, stalled, or operator pause",
    phases: [
        #{ title: "Drive", detail: "pick a non-blocked goal and take one increment" },
        #{ title: "Unblock", detail: "if all blocked, propose sandbox-safe path or pause" },
    ],
};

let pick_schema = #{
    "type": "object",
    "required": ["goal_id", "closeable", "status", "dod", "reason"],
    "properties": #{
        "goal_id": #{ "type": "string" },
        "closeable": #{ "type": "boolean" },
        "status": #{ "type": "string" },
        "dod": #{ "type": "string" },
        "reason": #{ "type": "string" },
        "repo": #{ "type": "string" },
        "branch": #{ "type": "string" },
        "worktree": #{ "type": "string" },
        "refs": #{ "type": "string" },
        "sandbox_ok": #{ "type": "boolean" },
        "unblock": #{ "type": "string" },
        "met_all": #{ "type": "boolean" },
    },
};

let impl_schema = #{
    "type": "object",
    "required": ["goal_id", "status", "summary", "verify_ok"],
    "properties": #{
        "goal_id": #{ "type": "string" },
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "verify_ok": #{ "type": "boolean" },
        "branch": #{ "type": "string" },
        "commit": #{ "type": "string" },
        "pr_url": #{ "type": "string" },
        "progressed": #{ "type": "boolean" },
    },
};

let verdict_schema = #{
    "type": "object",
    "required": ["real", "reason", "evidence"],
    "properties": #{
        "real": #{ "type": "boolean" },
        "reason": #{ "type": "string" },
        "evidence": #{ "type": "string" },
    },
};

fn as_str(x) {
    if type_of(x) == "string" { return x; }
    "";
}

let pin = if args == () { () } else { args.goal };
let last_fp = "";
let stall = 0;
let round = 0;
let last_goal = "";
let last_summary = "";
let last_ok = false;

phase("Drive");
while round < 3 {
    round = round + 1;
    log("goal-loop round " + round.to_string());

    let pick_p = "";
    pick_p += "Read with tools: docs/program/GOAL-LOOP.md, GOALS.md,\n";
    pick_p += "IMPLEMENTER-HANDOFF.md, HANDOFF-NEXT.md, PHASE-1-TASK-BOARD.md.\n";
    pick_p += "Obsidian refs: /akula-data/obsidian/akula-csd-kb/Program/\n";
    pick_p += "Horizon SELF-HOSTED-DRIVE-TARGETS.md is blocked until G-LIFE.\n";
    if pin != () {
        pick_p += "Pinned goal: " + pin + "\n";
    }
    pick_p += "Skip stalled fingerprints. Prefer highest-priority OPEN sandbox goal.\n";
    pick_p += "met_all=true only if every Phase 1 row in GOALS.md is evidenced met.\n";
    pick_p += "If all remaining are blocked, closeable=false, set sandbox_ok and unblock.\n";
    pick_p += "G-TRAIN is never closeable. Use grep/read_file/git. No WAN.\n";
    pick_p += "Return goal_id, closeable, status, dod, reason, repo, branch, worktree, refs.\n";
    let pick = agent(pick_p, #{
        label: "pick-goal",
        capability_mode: "read-only",
        output_schema: pick_schema,
        model: "grok-4.5",
    });
    if pick == () || !pick.success || pick.output == () {
        complete(#{ summary: "pick failed", round: round });
    }
    if pick.output.met_all == true {
        complete(#{ summary: "all phase-1 goals met", goal_id: pick.output.goal_id });
    }

    if pick.output.closeable == false {
        phase("Unblock");
        let u = as_str(pick.output.unblock);
        if u == "" { u = as_str(pick.output.reason); }
        if pick.output.sandbox_ok == true {
            await_user("verification", "Sandbox-safe unblock for " + as_str(pick.output.goal_id) + ": " + u);
        } else {
            await_user("user", "All goals blocked; not sandbox-safe. " + as_str(pick.output.goal_id) + ": " + u);
        }
    } else {
        let impl_p = "";
        impl_p += "Drive ONE increment toward this GOAL (not a random todo).\n";
        impl_p += "Read IMPLEMENTER-HANDOFF.md and GOAL-LOOP.md first.\n";
        impl_p += "Goal: " + as_str(pick.output.goal_id) + "\n";
        impl_p += "DoD: " + as_str(pick.output.dod) + "\n";
        impl_p += "Refs: " + as_str(pick.output.refs) + "\n";
        impl_p += "Repo: " + as_str(pick.output.repo) + " branch: " + as_str(pick.output.branch) + "\n";
        impl_p += "Worktree: " + as_str(pick.output.worktree) + "\n";
        impl_p += "Sandbox only. No WAN. No pause LocalAI. Comfy is masked; do not unmask.\n";
        impl_p += "Never reset kang-main-wip. One architecture change. Conventional commits.\n";
        impl_p += "Push Forgejo as tzervas + git-askpass, credential.helper disabled.\n";
        impl_p += "Merge only if required checks legitimately green (ran and succeeded).\n";
        impl_p += "If this increment cannot progress, set progressed=false and say why.\n";
        impl_p += "Do not start G-TRAIN. Keep 3090 LocalAI loaded. Implement as grok-4.6.\n";
        impl_p += "Return goal_id, status, summary, verify_ok, progressed, branch, commit, pr_url.\n";
        let impl = agent(impl_p, #{
            label: "drive-goal",
            capability_mode: "all",
            output_schema: impl_schema,
            model: "grok-4.6",
        });
        if impl == () || !impl.success || impl.output == () {
            last_summary = "implement failed";
            last_ok = false;
        } else {
            last_goal = as_str(impl.output.goal_id);
            last_summary = as_str(impl.output.summary);
            last_ok = impl.output.verify_ok == true;
            let ver_p = "";
            ver_p += "Adversarially verify the increment against the GOAL DoD. Read the code.\n";
            ver_p += "Goal: " + as_str(pick.output.goal_id) + " DoD: " + as_str(pick.output.dod) + "\n";
            ver_p += json_encode(impl.output) + "\n";
            ver_p += "real=true only with inspected evidence. Fail closed on skip/|| true/merge-red.\n";
            let ver = agent(ver_p, #{
                label: "verify-goal",
                capability_mode: "read-only",
                output_schema: verdict_schema,
                model: "grok-4.5",
            });
            if ver != () && ver.success && ver.output != () && ver.output.real == true {
                last_ok = true;
            } else {
                last_ok = false;
            }
        }
    }

    let fp = fingerprint(as_str(pick.output.goal_id) + as_str(pick.output.reason));
    if fp == last_fp {
        stall = stall + 1;
        log("stall " + stall.to_string() + " on " + as_str(pick.output.goal_id));
    } else {
        stall = 0;
        last_fp = fp;
    }
    if stall >= 2 {
        log("switching off stalled goal " + as_str(pick.output.goal_id));
    }
}

let report = as_str(last_summary);
if report == "" { report = "goal-loop finished"; }
let path = write_scratch_file("csd-goal-loop.md", report);
complete(#{
    path: path,
    goal_id: as_str(last_goal),
    accepted: last_ok,
    summary: report,
    rounds: round,
});
