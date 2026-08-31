let meta = #{
    name: "csd-python-first-drive",
    description: "One closeable Python-first increment: P1-00..P1-17 or ops-index/ops-runner",
    when_to_use: "Drive CogSynDelta Phase 1 memory-gate or Goal 0 ops one ID at a time after the official plan",
    phases: [
        #{ title: "Pick", detail: "resolve ID and refuse stay-open" },
        #{ title: "Implement", detail: "one isolated increment, no merge" },
        #{ title: "Verify", detail: "adversarial review against the board gate" },
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
pick_p += "Read docs/program/PYTHON-FIRST-PLAN.md, docs/program/PHASE-1-TASK-BOARD.md,\n";
pick_p += "and docs/program/PYTHON-MEMORY-GATE-GAP-MAP.md.\n";
pick_p += "Requested id: " + raw_id + "\n";
pick_p += "If id is next, pick the first board row whose Depends-on work is landed or\n";
pick_p += "is ops that can proceed without a product-code PR. P1-01 docs may run with P1-00.\n";
pick_p += "Product-code P1-02+ is NOT closeable until P1-00 has a real Forgejo run.\n";
pick_p += "A real Forgejo run exists on memory-gate PR #1 (homelab-cpu); pytest ran on\n";
pick_p += "a66f730 (8 failed / 121 passed). P1-02 is closeable. P1-01 spec lives on\n";
pick_p += "PR #2 / memory-gate-wt-p1-01 (read-only). Do not set closeable=false just\n";
pick_p += "because pytest/gitleaks/trivy-on-chroma are still red or P1-00 is unmerged.\n";
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
impl_p += "Implement exactly one closeable Python-first increment. Do not merge.\n";
impl_p += "ID: " + pick.output.id + "\n";
impl_p += "DoD: " + pick.output.dod + "\n";
impl_p += "Repo: " + pick.output.repo + "\n";
impl_p += "Branch: " + pick.output.branch + "\n";
impl_p += "Worktree if given: " + pick.output.worktree + "\n";
impl_p += "Read docs/program/PYTHON-FIRST-PLAN.md and PHASE-1-TASK-BOARD.md first.\n";
impl_p += "Hard rules: never checkout/reset python-ai/memory-gate main (kang-main-wip).\n";
impl_p += "Use sibling worktrees from forgejo/main. Never push main/staging/develop/dev.\n";
impl_p += "Never pause 3090 LocalAI. Never kill Comfy; enqueue 5080 jobs.\n";
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
impl_p += "Do not reset kang-main-wip. Do not merge. P1-00 honest pytest reds stay.\n";
impl_p += "If ID is ops-index: repair csd-kb-index remote path; enqueue behind Comfy.\n";
impl_p += "If ID is ops-runner: discover and restore the Forgejo CPU runner; do not fake green.\n";
impl_p += "Commit on the feature branch. Push to forgejo as GIT_USERNAME=tzervas with\n";
impl_p += "credential.helper disabled + git-askpass-token under secret exec\n";
impl_p += "TOKEN=git/cabal-forgejo-admin. Never GitHub bot push. Cabal user is ACL-blocked.\n";
impl_p += "Open a Forgejo PR if the forge CLI/API is available; otherwise report the push.\n";
impl_p += "Workflow host cannot spawn LocalAI model slugs; implement here as grok-4.6.\n";
impl_p += "Keep 3090 LocalAI loaded anyway. Never pause it. Never steal the 5080 from Comfy.\n";
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
let report = #{
    id: pick.output.id,
    accepted: accepted,
    implement: impl.output,
    verify_reason: if ver != () && ver.output != () { ver.output.reason } else { "verify missing" },
    verify_evidence: if ver != () && ver.output != () { ver.output.evidence } else { "" },
};
let path = write_scratch_file("csd-python-first-drive.md", json_encode(report));
complete(#{
    path: path,
    id: pick.output.id,
    accepted: accepted,
    summary: impl.output.summary,
    branch: impl.output.branch,
});
