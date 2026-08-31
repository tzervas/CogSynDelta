let meta = #{
    name: "csd-ci-unblock",
    description: "Unblock merge-first CogSynDelta PRs: fix uv/ruff CI, restack, reopen, confirm runners",
    when_to_use: "When #56-#60 required Actions are red or idle and the merge stack must move",
    phases: [
        #{ title: "Fix", detail: "uv flag + ruff on #56, restack, reopen #57/#58" },
        #{ title: "Runners", detail: "homelab listen via SSH; secret exec only on a decrypting host" },
        #{ title: "Verify", detail: "required checks queued or running, not idle-red" },
    ],
};

let fix_schema = #{
    "type": "object",
    "required": ["verdict", "evidence"],
    "properties": #{
        "verdict": #{ "type": "string" },
        "evidence": #{ "type": "string" },
        "sha56": #{ "type": "string" },
        "sha57": #{ "type": "string" },
        "sha58": #{ "type": "string" },
        "sha60": #{ "type": "string" },
        "reopened": #{ "type": "string" },
        "first_error": #{ "type": "string" },
    },
};

let run_schema = #{
    "type": "object",
    "required": ["verdict", "evidence"],
    "properties": #{
        "verdict": #{ "type": "string" },
        "evidence": #{ "type": "string" },
        "cpu_listen": #{ "type": "string" },
        "secret_inject": #{ "type": "string" },
        "first_error": #{ "type": "string" },
    },
};

let ver_schema = #{
    "type": "object",
    "required": ["verdict", "evidence"],
    "properties": #{
        "verdict": #{ "type": "string" },
        "evidence": #{ "type": "string" },
        "queued": #{ "type": "string" },
        "in_progress": #{ "type": "string" },
        "required56": #{ "type": "string" },
        "first_error": #{ "type": "string" },
    },
};

phase("Fix");
let pfix = "";
pfix += "You are the CogSynDelta CI unblock implementer. Use tools. Do not invent results.\n";
pfix += "Repo: tzervas/CogSynDelta. Merge order: #56 -> #57 -> #58 -> #60. Do not squash #56. Do not merge.\n";
pfix += "Existing worktrees (branches already checked out — do NOT create new worktrees):\n";
pfix += "  E0=/home/kang/code/personal/CogSynDelta-wt-poc-e0  branch feat/poc-e0-e2-foundation  PR 56\n";
pfix += "  REG=/home/kang/code/personal/CogSynDelta-wt-poc-registry  branch feat/poc-region-registry  PR 57 CLOSED\n";
pfix += "  CUDA=/home/kang/code/personal/CogSynDelta-wt-poc-cuda  branch feat/poc-cuda-measure  PR 58 CLOSED\n";
pfix += "  P3=/home/kang/code/personal/CogSynDelta-wt-poc3  branch feat/poc-train-route  PR 60\n";
pfix += "  GPU=/home/kang/code/personal/CogSynDelta-wt-akula-gpu  branch ci/akula-gpu-runner  PR 59\n";
pfix += "\nROOT CAUSE (measured on Actions run 32317020349 / 32317020374):\n";
pfix += "1) uv treats env UV_COMPILE_BYTECODE=0 as --compile-bytecode. Combined with CLI --no-compile-bytecode, uv sync exits 2.\n";
pfix += "2) #56 Lint failed: 52 ruff issues in tests/test_compression_calibration.py (W293, F541).\n";
pfix += "\nDO THIS ON E0 (PR 56):\n";
pfix += "A) In .github/workflows/ci.yml DELETE the top-level env key UV_COMPILE_BYTECODE entirely. Keep --no-compile-bytecode --link-mode copy on uv sync.\n";
pfix += "B) In .github/workflows/poc-ci.yml DELETE the job env UV_COMPILE_BYTECODE entirely. Keep --no-compile-bytecode --link-mode copy.\n";
pfix += "C) Run: uvx ruff@0.14.13 check --fix tests/test_compression_calibration.py\n";
pfix += "   then: uvx ruff@0.14.13 format tests/test_compression_calibration.py\n";
pfix += "   then: uvx ruff@0.14.13 check src/ tests/ benchmarks/ scripts/ examples/\n";
pfix += "   Fix any remaining ruff so lint is clean.\n";
pfix += "D) Commit on E0: fix(ci): drop UV_COMPILE_BYTECODE env that conflicts with --no-compile-bytecode\n";
pfix += "   Body: uv 0.7 treats a set UV_COMPILE_BYTECODE as --compile-bytecode even when the value is 0.\n";
pfix += "   Also ruff --fix tests/test_compression_calibration.py so required Lint can pass.\n";
pfix += "E) git push origin feat/poc-e0-e2-foundation\n";
pfix += "\nRESTACK (force-with-lease only):\n";
pfix += "In REG: git fetch origin; git rebase feat/poc-e0-e2-foundation. On workflow conflicts, keep the version with NO UV_COMPILE_BYTECODE env and WITH --no-compile-bytecode --link-mode copy. Push --force-with-lease.\n";
pfix += "In CUDA: rebase onto feat/poc-region-registry. If scripts/ci_local.sh exports UV_COMPILE_BYTECODE=0, delete that export (uv flag conflict). Push --force-with-lease.\n";
pfix += "In P3: rebase onto feat/poc-cuda-measure. Same ci_local.sh / poc-ci.yml rule. Push --force-with-lease.\n";
pfix += "In GPU: rebase onto feat/poc-e0-e2-foundation. Push --force-with-lease.\n";
pfix += "\nREOPEN: gh pr reopen 57 --repo tzervas/CogSynDelta; gh pr reopen 58 --repo tzervas/CogSynDelta\n";
pfix += "Confirm PR 57 head is feat/poc-region-registry (not 6da91e5) and PR 58 head is feat/poc-cuda-measure.\n";
pfix += "If reopen fails, open replacement PRs with the same bases: 57 base feat/poc-e0-e2-foundation, 58 base feat/poc-region-registry.\n";
pfix += "\nDo not print tokens. Conventional commit. Report SHAs after push.\n";

let rfix = agent(pfix, #{
    label: "fix-stack",
    capability_mode: "all",
    output_schema: fix_schema,
});

phase("Runners");
let prun = "";
prun += "Check homelab + akula GHA runners so CogSynDelta jobs can be claimed. Use tools. Never print secret values.\n";
prun += "SECRET TOOLCHAIN: command is `secret` at /home/kang/.local/bin/secret (akula) and /usr/local/bin/secret (homelab).\n";
prun += "Never `secret get` to a terminal. Never pass a value as argv. Never echo decrypted env vars.\n";
prun += "akula-prime age public is age1nwuc0sx0zusxl5aflkfzal59dxkcld8xpsqydy3jmjyuua7u5swsg6n9wh.\n";
prun += "homelab/* leaves on this host are wrapped to laptop + homelab-kang ONLY. `secret exec` of homelab/* on akula WILL fail closed. Do not fight that on this host.\n";
prun += "SSH from akula uses Host homelab in ~/.ssh/config (IdentityFile, User kang). That is leak-safe: no key on argv.\n";
prun += "On the remote, sudo -n works. gha-agent linger is yes. Check:\n";
prun += "  sudo -n -u gha-agent XDG_RUNTIME_DIR=/run/user/995 systemctl --user is-active gha-runner-ctl@cpu\n";
prun += "  sudo -n grep -E '^(GHA_PRIORITY_REPOS|GHA_POOL_MAX_WORKERS|GHA_ALLOWLIST_REPOS)=' /home/gha-agent/.local/share/gha-runner-ctl/instances/cpu.env\n";
prun += "Do NOT grep or print GHA_APP_PRIVATE_KEY or any *TOKEN* / *KEY* / *SECRET* / *PASS* values.\n";
prun += "Expected: cpu listen active; GHA_PRIORITY_REPOS includes tzervas/CogSynDelta; allowlist includes CogSynDelta; max workers 60.\n";
prun += "On akula: systemctl --user is-active gha-runner-ctl@gpu; do not dump gpu.env values.\n";
prun += "Try remote inject (names only on argv) AFTER confirming remote vault can decrypt:\n";
prun += "  ssh homelab 'secret exec HOMELAB_IP=homelab/ip -- python3 -c \"import os; print(\\\"ok\\\", len(os.environ.get(\\\"HOMELAB_IP\\\",\\\"\\\")))\"'\n";
prun += "If decrypt fails, set secret_inject=unavailable-wrong-recipient and do not copy keys. SSH config is enough for this check.\n";
prun += "Mycelium rust-test may occupy a worker; with max_workers=60 that is not a CogSynDelta queue block. Report if no worker can spawn.\n";

let rrun = agent(prun, #{
    label: "runners",
    capability_mode: "execute",
    output_schema: run_schema,
});

phase("Verify");
let pver = "";
pver += "Verify merge-first Actions are processing after the fix push. Use gh. Repo tzervas/CogSynDelta.\n";
pver += "PRs: 56 feat/poc-e0-e2-foundation, 57 feat/poc-region-registry, 58 feat/poc-cuda-measure, 60 feat/poc-train-route.\n";
pver += "Required for #56 merge: Lint & Format Check, Test on Python 3.12, Quality Score Check.\n";
pver += "Commands:\n";
pver += "  gh pr view 56,57,58,60 --repo tzervas/CogSynDelta --json number,state,headRefOid,statusCheckRollup,mergeStateStatus\n";
pver += "  gh run list --repo tzervas/CogSynDelta --status queued --limit 20\n";
pver += "  gh run list --repo tzervas/CogSynDelta --status in_progress --limit 20\n";
pver += "  git ls-remote origin refs/heads/feat/poc-e0-e2-foundation refs/heads/feat/poc-region-registry refs/heads/feat/poc-cuda-measure refs/heads/feat/poc-train-route\n";
pver += "Verdict pass only if #56 HEAD is NEWER than 94a718a AND (queued or in_progress exist OR required checks are pending/success — not the old uv-flag failure on 94a718a).\n";
pver += "If still idle on the old SHA, say so. Do not rerun failed jobs on 94a718a (that SHA is permanently red).\n";

let rver = agent(pver, #{
    label: "verify-actions",
    capability_mode: "execute",
    output_schema: ver_schema,
});

let summary = "csd-ci-unblock finished";
let ev = "";
if rfix != () && rfix.success && rfix.output.evidence != () { ev += "fix=" + rfix.output.evidence + "\n"; }
if rrun != () && rrun.success && rrun.output.evidence != () { ev += "runners=" + rrun.output.evidence + "\n"; }
if rver != () && rver.success && rver.output.evidence != () { ev += "verify=" + rver.output.evidence + "\n"; }
if ev == "" { ev = "agents returned no evidence"; }

let path = write_scratch_file("csd-ci-unblock.md", ev);
complete(#{
    summary: summary,
    path: path,
    report: ev,
    fix_ok: rfix != () && rfix.success,
    runners_ok: rrun != () && rrun.success,
    verify_ok: rver != () && rver.output.verdict == "pass",
});
