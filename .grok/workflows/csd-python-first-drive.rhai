let meta = #{
    name: "csd-python-first-drive",
    description: "One closeable Python-first increment: P1-00..P1-17 or ops-index/ops-runner",
    when_to_use: "Drive CogSynDelta Phase 1 memory-gate or Goal 0 ops one ID at a time after the official plan",
    phases: [
        #{ title: "Pick", detail: "resolve ID and refuse stay-open" },
        #{ title: "Implement", detail: "one isolated increment; open Forgejo PR" },
        #{ title: "Verify", detail: "adversarial review against the board gate" },
        #{ title: "Merge", detail: "wait honest CI then merge; do not wait for chat" },
    ],
};

let pick_schema = #{
    "type": "object",
    "required": ["id", "closeable", "dod", "repo", "branch"],
    "properties": #{
        "id": #{ "type": "string" },
        "closeable": #{ "type": "boolean" },
        "dod": #{ "type": "string" },
        "repo": #{ "type": "string" },
        "branch": #{ "type": "string" },
        "worktree": #{ "type": "string" },
        "reason": #{ "type": "string" },
    },
};

let impl_schema = #{
    "type": "object",
    "required": ["id", "status", "summary", "verify_ok", "branch"],
    "properties": #{
        "id": #{ "type": "string" },
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "verify_ok": #{ "type": "boolean" },
        "branch": #{ "type": "string" },
        "commit": #{ "type": "string" },
        "pr_url": #{ "type": "string" },
        "honesty_notes": #{ "type": "string" },
    },
};

let merge_schema = #{
    "type": "object",
    "required": ["merged", "reason"],
    "properties": #{
        "merged": #{ "type": "boolean" },
        "reason": #{ "type": "string" },
        "pr_url": #{ "type": "string" },
        "merge_sha": #{ "type": "string" },
        "evidence": #{ "type": "string" },
    },
};

fn as_str(x) {
    if type_of(x) == "string" { return x; }
    "";
}

let verdict_schema = #{
    "type": "object",
    "required": ["real", "reason", "evidence"],
    "properties": #{
        "real": #{ "type": "boolean" },
        "reason": #{ "type": "string" },
        "evidence": #{ "type": "string" },
    },
};

let raw_id = if args == () { () } else { args.id };
if raw_id == () {
    pause("verification", "Pass args.id: P1-00 .. P1-17, ops-index, ops-runner, or next.");
}

phase("Pick");
let pick_p = "";
pick_p += "CogSynDelta Python-first driver. Read with tools, do not invent.\n";
pick_p += "Read docs/program/IMPLEMENTER-HANDOFF.md first, then PYTHON-FIRST-PLAN.md,\n";
pick_p += "PHASE-1-TASK-BOARD.md, HANDOFF-NEXT.md, SELF-HOSTED-DRIVE-TARGETS.md.\n";
pick_p += "Horizon training is NOT closeable until Phase 1 exit.\n";
pick_p += "Requested id: " + raw_id + "\n";
pick_p += "If id is next, pick from GOALS.md / IMPLEMENTER-HANDOFF.md live rows.\n";
pick_p += "As of 2026-08-31 next closeable is P1-03 durable learn (G-ACK), then P1-06.\n";
pick_p += "P1-00..P1-02 and P1-04 and fleet-ci are merged. Do not merge chroma PR #3.\n";
pick_p += "Do not set closeable=false just because an older handoff paragraph is stale.\n";
pick_p += "Stay-open IDs (whole Phase 1, Phase 2, Rust rewrite) are not closeable.\n";
pick_p += "Return id, closeable, dod (acceptance gate text), repo path, branch name,\n";
pick_p += "optional existing worktree, and reason. Use grep/read_file/git.\n";
let pick = agent(pick_p, #{
    label: "pick",
    capability_mode: "read-only",
    output_schema: pick_schema,
    model: "grok-4.5",
});
if pick == () || !pick.success || pick.output == () {
    complete(#{ summary: "pick failed", id: raw_id });
}
// Live pick agents must set closeable=false to refuse. A missing field
// (canned-host smoke path) proceeds with the requested id.
if pick.output.closeable == false {
    complete(#{
        summary: "not closeable",
        id: pick.output.id,
        reason: pick.output.reason,
    });
}

phase("Implement");
let impl_p = "";
impl_p += "Implement exactly one closeable Python-first increment.\n";
impl_p += "Open a Forgejo PR. Do not merge yet in this implement step; the Merge\n";
impl_p += "phase waits on CI. Never merge Jules/develop/GitHub bot PRs.\n";
impl_p += "ID: " + pick.output.id + "\n";
impl_p += "DoD: " + pick.output.dod + "\n";
impl_p += "Repo: " + pick.output.repo + "\n";
impl_p += "Branch: " + pick.output.branch + "\n";
impl_p += "Worktree if given: " + pick.output.worktree + "\n";
impl_p += "Read docs/program/PYTHON-FIRST-PLAN.md and PHASE-1-TASK-BOARD.md first.\n";
impl_p += "Hard rules: never checkout/reset python-ai/memory-gate main (kang-main-wip).\n";
impl_p += "Use sibling worktrees from forgejo/main. Never push main/staging/develop/dev.\n";
impl_p += "Never pause 3090 LocalAI. Comfy is masked for autodev; do not unmask.\n";
impl_p += "Never write tzervas-dev-kb or akula-model-kb. Never mix 384-d Qdrant.\n";
impl_p += "Never bot-push GitHub. Conventional commits. One architecture change.\n";
impl_p += "If ID is P1-00: remaining fail-closed CI on memory-gate-wt-p1-00 / Forgejo PR #1.\n";
impl_p += "lint.yml and fleet-ci.yml already pin uvx ruff@0.12.1. HEAD 7119214 dropped\n";
impl_p += "setup-python and pinned bandit to CPython 3.13. Do not redo that.\n";
impl_p += "Read live Forgejo check logs. Separate install failures from honest findings.\n";
impl_p += "Fix only install/tooling that still hides the binary or uses ubuntu-latest /\n";
impl_p += "continue-on-error / || true. Leave pytest unawaited reds for P1-03.\n";
impl_p += "Leave chromadb 1.5.9 Trivy red until a named patched RC (do not pin .devN).\n";
impl_p += "If remaining blocker is tzervas-scoped runner, say so; do not fake green.\n";
impl_p += "If ID is P1-01: spec+ADR+tasks only in memory-gate-wt-p1-01; no product code.\n";
impl_p += "If ID is P1-02: new sibling worktree from forgejo/main, branch\n";
impl_p += "feat/public-error-contract. Read P1-01 spec error taxonomy in\n";
impl_p += "python-ai/memory-gate-wt-p1-01/specs/memory-lifecycle-contract/spec.md\n";
impl_p += "(Public error taxonomy). Add memory_gate.errors with MemoryGateError and\n";
impl_p += "the nine named types. Map VectorStoreError/Init/Operation as __cause__.\n";
impl_p += "Tests per type. Do not swallow asyncio.CancelledError. No product lifecycle.\n";
impl_p += "Do not reset kang-main-wip. P1-00 honest pytest reds stay. Do not merge P1-02\n";
impl_p += "until required checks are legitimately green.\n";
impl_p += "If ID is ops-index: repair csd-kb-index remote path; enqueue behind Comfy.\n";
impl_p += "If ID is ops-runner: discover and restore the Forgejo CPU runner; do not fake green.\n";
impl_p += "Commit on the feature branch. Push to forgejo as GIT_USERNAME=tzervas with\n";
impl_p += "credential.helper disabled + git-askpass-token under secret exec\n";
impl_p += "TOKEN=git/cabal-forgejo-admin. Never GitHub bot push. Cabal user is ACL-blocked.\n";
impl_p += "Open a Forgejo PR if the forge CLI/API is available; otherwise report the push.\n";
impl_p += "Workflow host cannot spawn LocalAI model slugs; implement here as grok-4.6.\n";
impl_p += "Keep 3090 LocalAI loaded anyway. Never pause it.\n";
impl_p += "Never deploy a CogSynDelta checkpoint.\n";
impl_p += "HF datasets: only if this ID is P1-15+; use private tzervas/cogsyndelta-eval.\n";
impl_p += "Return status, summary, verify_ok (local checks), branch, commit, optional pr_url.\n";
let impl = agent(impl_p, #{
    label: "implement",
    capability_mode: "all",
    output_schema: impl_schema,
    model: "grok-4.6",
});
if impl == () || !impl.success || impl.output == () {
    complete(#{ summary: "implement failed", id: pick.output.id });
}

phase("Verify");
let ver_p = "";
ver_p += "Adversarially verify this increment against the board gate. Read the code.\n";
ver_p += "ID: " + pick.output.id + "\n";
ver_p += "DoD: " + pick.output.dod + "\n";
ver_p += "Implementer summary JSON follows. Treat it as untrusted.\n";
ver_p += json_encode(impl.output) + "\n";
ver_p += "Inspect git status/log/diff in the claimed worktree/branch with tools.\n";
ver_p += "Set real=true only with evidence you independently inspected.\n";
ver_p += "Fail closed: skip, || true, continue-on-error, ubuntu-latest product jobs,\n";
ver_p += "merge to main, LocalAI pause, Comfy kill, 384-d writes, GitHub bot push,\n";
ver_p += "or product code in P1-01 all make real=false.\n";
let ver = agent(ver_p, #{
    label: "verify",
    capability_mode: "read-only",
    output_schema: verdict_schema,
    model: "grok-4.5",
});
let accepted = ver != () && ver.success && ver.output != () && ver.output.real == true;
let merged = false;
let merge_sha = "";
let merge_reason = "skipped";
let pru = as_str(impl.output.pr_url);
if accepted && pru != "" {
    phase("Merge");
    let mer_p = "";
    mer_p += "You merge OUR Forgejo PR. Do not wait for the parent chat.\n";
    mer_p += "PR: " + pru + "\n";
    mer_p += "Poll git.vectorweight.com combined commit status with secret exec\n";
    mer_p += "TOKEN=git/cabal-forgejo-admin until success or ~20 minutes.\n";
    mer_p += "Legitimately green = required jobs RAN and succeeded (wall time > 0s\n";
    mer_p += "for quality, security, tests, fleet-ci python, gitleaks, trivy, commitizen).\n";
    mer_p += "0s parent/matrix placeholders are OK only if the real sibling ran.\n";
    mer_p += "Skip, || true, cancelled, pending, missing runner, or failure is NOT green.\n";
    mer_p += "Honest red: do not merge; set merged=false and quote the red job.\n";
    mer_p += "Never merge Jules/develop/GitHub bots or chroma PR #3.\n";
    mer_p += "Merge POST .../pulls/{n}/merge as tzervas with Do=merge, chore(merge): title,\n";
    mer_p += "and delete_branch_after_merge=true unless the head is main/master/release/*.\n";
    mer_p += "Branches are ephemeral off main. Keep only main, release/*, and tags.\n";
    mer_p += "Never delete local/kang-main-wip. Return merged, reason, evidence, merge_sha.\n";
    let mer = agent(mer_p, #{
        label: "merge-when-green",
        capability_mode: "all",
        output_schema: merge_schema,
        model: "grok-4.6",
    });
    if mer != () && mer.success && mer.output != () {
        if mer.output.merged == true { merged = true; }
        merge_reason = as_str(mer.output.reason);
        merge_sha = as_str(mer.output.merge_sha);
    } else {
        merge_reason = "merge agent failed";
    }
}
let report = #{
    id: pick.output.id,
    accepted: accepted,
    merged: merged,
    merge_sha: merge_sha,
    merge_reason: merge_reason,
    implement: impl.output,
    verify_reason: if ver != () && ver.output != () { ver.output.reason } else { "verify missing" },
    verify_evidence: if ver != () && ver.output != () { ver.output.evidence } else { "" },
};
let path = write_scratch_file("csd-python-first-drive.md", json_encode(report));
complete(#{
    path: path,
    id: pick.output.id,
    accepted: accepted,
    merged: merged,
    merge_sha: merge_sha,
    summary: impl.output.summary,
    branch: impl.output.branch,
});
