let meta = #{
    name: "csd-poc-gpu-drive",
    description: "Measure CogSynDelta PoC CUDA on the 5080, record honest STATUS, then the next gated slice",
    when_to_use: "After homelab/akula runners recover; close the PoC CUDA gap then implement one next increment",
    phases: [
        #{ title: "Env", detail: "5080, VRAM, PoC HEAD SHA" },
        #{ title: "CPU gate", detail: "pytest and ruff on PoC HEAD" },
        #{ title: "CUDA measure", detail: "train, compress, route on 5080" },
        #{ title: "Record", detail: "STATUS CUDA rows, tests, PR" },
        #{ title: "Review", detail: "adversarial claim check" },
    ],
};

let env_schema = #{
    "type": "object",
    "required": ["verdict", "evidence"],
    "properties": #{
        "verdict": #{ "type": "string" },
        "evidence": #{ "type": "string" },
        "gpu_name": #{ "type": "string" },
        "vram_free_mib": #{ "type": "string" },
        "head_sha": #{ "type": "string" },
        "worktree": #{ "type": "string" },
        "first_error": #{ "type": "string" },
    },
};

let gate_schema = #{
    "type": "object",
    "required": ["verdict", "evidence"],
    "properties": #{
        "verdict": #{ "type": "string" },
        "evidence": #{ "type": "string" },
        "first_error": #{ "type": "string" },
        "tests_passed": #{ "type": "string" },
    },
};

let measure_schema = #{
    "type": "object",
    "required": ["verdict", "evidence"],
    "properties": #{
        "verdict": #{ "type": "string" },
        "evidence": #{ "type": "string" },
        "first_error": #{ "type": "string" },
        "need_ollama_stop": #{ "type": "boolean" },
        "need_3090ti": #{ "type": "boolean" },
        "train_loss": #{ "type": "string" },
        "compress_fidelity": #{ "type": "string" },
        "route_load": #{ "type": "string" },
    },
};

let record_schema = #{
    "type": "object",
    "required": ["verdict", "evidence"],
    "properties": #{
        "verdict": #{ "type": "string" },
        "evidence": #{ "type": "string" },
        "branch": #{ "type": "string" },
        "pr_url": #{ "type": "string" },
        "first_error": #{ "type": "string" },
    },
};

let review_schema = #{
    "type": "object",
    "required": ["real", "reason", "evidence"],
    "properties": #{
        "real": #{ "type": "boolean" },
        "reason": #{ "type": "string" },
        "evidence": #{ "type": "string" },
    },
};

let base_branch = "feat/poc-region-registry";
if args != () && args.branch != () {
    base_branch = args.branch;
}

phase("Env");
let ep = "";
ep += "You are on akula-prime. Use run_terminal_command and read_file. Do not invent a GPU.\n";
ep += "Repo: /home/kang/code/personal/CogSynDelta (local main is STALE). Remote: tzervas/CogSynDelta.\n";
ep += "1. nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv\n";
ep += "2. test -e /dev/nvidia0; python3 -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)' if torch exists\n";
ep += "3. git -C /home/kang/code/personal/CogSynDelta fetch origin\n";
ep += "4. Create an isolated worktree if missing:\n";
ep += "   git -C /home/kang/code/personal/CogSynDelta worktree add -B ";
ep += base_branch;
ep += " /tmp/csd-poc-gpu origin/";
ep += base_branch;
ep += "\n";
ep += "   If the worktree exists, reset it to origin/";
ep += base_branch;
ep += " (do not reset if it has uncommitted work you just created).\n";
ep += "5. Record HEAD SHA of that worktree.\n";
ep += "Do NOT stop ollama. Do NOT swap GPUs.\n";
ep += "verdict=PASS if /dev/nvidia0 exists, nvidia-smi lists RTX 5080, and the worktree exists.\n";
ep += "verdict=FAIL_ENV otherwise. evidence=short sanitized facts (gpu name, free MiB, sha).\n";
let envr = agent(ep, #{
    label: "env",
    capability_mode: "execute",
    output_schema: env_schema,
});

let env_ok = false;
let worktree = "/tmp/csd-poc-gpu";
if envr != () && envr.success && envr.output.verdict == "PASS" {
    env_ok = true;
    if envr.output.worktree != () { worktree = envr.output.worktree; }
}
if !env_ok {
    let p = write_scratch_file("report.md", "FAIL_ENV: GPU or worktree not ready.");
    complete(#{ summary: "FAIL_ENV", path: p, next: "fix env" });
}

phase("CPU gate");
let cp = "";
cp += "Worktree: ";
cp += worktree;
cp += "\n";
cp += "This is CogSynDelta PoC HEAD (RegionRegistry). Python floor is >=3.12,<3.14.\n";
cp += "Use run_terminal_command. Do not implement new features.\n";
cp += "cd into the worktree. Use uv (not system pip).\n";
cp += "1. uv sync --group dev (use the repo pyproject; CUDA torch extra/index if present).\n";
cp += "2. uv run ruff check src/cogsyndelta/contracts src/cogsyndelta/poc tests/test_poc_*.py\n";
cp += "3. uv run pytest tests/test_poc_contracts.py tests/test_poc_train.py tests/test_poc_compress.py tests/test_poc_registry.py -q --tb=short\n";
cp += "4. uv run python -m cogsyndelta.poc.cli train --device cpu --steps 10 --checkpoint /tmp/poc_vae_cpu.pt\n";
cp += "5. uv run python -m cogsyndelta.poc.cli compress --device cpu --batch 4\n";
cp += "6. uv run python -m cogsyndelta.poc.cli route --device cpu --batch 8\n";
cp += "verdict=PASS only if ruff, pytest, and all three CLIs succeed. Otherwise FAIL_PRODUCT.\n";
cp += "evidence=counts and first failing command. tests_passed=N/N.\n";
let cpr = agent(cp, #{
    label: "cpu-gate",
    capability_mode: "execute",
    output_schema: gate_schema,
});

let cpu_ok = false;
if cpr != () && cpr.success && cpr.output.verdict == "PASS" { cpu_ok = true; }
if !cpu_ok {
    let p2 = write_scratch_file("report.md", "CPU gate failed; do not claim CUDA or merge.");
    complete(#{ summary: "FAIL_PRODUCT cpu", path: p2, next: "fix PoC CPU" });
}

phase("CUDA measure");
let mp = "";
mp += "Worktree: ";
mp += worktree;
mp += "\n";
mp += "Measure PoC CUDA on the RTX 5080. Do NOT stop ollama. Do NOT invent numbers.\n";
mp += "STATUS.md currently lists Device CUDA path as unknown. Close that with measured values.\n";
mp += "1. nvidia-smi snapshot (used/free MiB) before and after.\n";
mp += "2. uv run python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'\n";
mp += "3. uv run python -m cogsyndelta.poc.cli train --device cuda --steps 20 --checkpoint /tmp/poc_vae_cuda.pt\n";
mp += "4. uv run python -m cogsyndelta.poc.cli compress --device cuda --batch 4\n";
mp += "5. uv run python -m cogsyndelta.poc.cli route --device cuda --batch 8\n";
mp += "6. If a CUDA test file exists, run it. If not, do not invent one yet.\n";
mp += "If CUDA OOM: set need_ollama_stop=true and/or need_3090ti=true, verdict=FAIL_ENV, do not kill ollama.\n";
mp += "verdict=PASS only if train loss decreases on cuda AND compress+route produce JSON with measured fidelity.\n";
mp += "Fill train_loss, compress_fidelity, route_load with the actual numbers.\n";
mp += "evidence=quoted command outputs (short).\n";
let mr = agent(mp, #{
    label: "cuda-measure",
    capability_mode: "execute",
    output_schema: measure_schema,
});

let cuda_ok = false;
let need_swap = false;
if mr != () && mr.success && mr.output.verdict == "PASS" { cuda_ok = true; }
if mr != () && mr.success && mr.output.need_3090ti == true { need_swap = true; }
if mr != () && mr.success && mr.output.need_ollama_stop == true { need_swap = true; }
if !cuda_ok {
    let msg = "CUDA measure failed. Do not update STATUS with fake numbers.";
    if need_swap {
        msg = "CUDA OOM or VRAM blocked. Ask operator to stop ollama or swap 3090Ti. Do not fake results.";
    }
    let p3 = write_scratch_file("report.md", msg);
    complete(#{ summary: "FAIL_ENV cuda", path: p3, next: "vram or 3090Ti" });
}

phase("Record");
let rp = "";
rp += "Worktree: ";
rp += worktree;
rp += "\n";
rp += "Record measured CUDA results honestly. Branch from current worktree HEAD.\n";
rp += "Create branch feat/poc-cuda-measure from HEAD if not already on it.\n";
rp += "Edits allowed:\n";
rp += "- STATUS.md: change Device CUDA path from unknown to measured numbers. No marketing.\n";
rp += "- tests: add tests/test_poc_cuda.py that skips unless torch.cuda.is_available().\n";
rp += "- specs/poc-v1/spec.md only if DoD wording needs a CUDA measured note.\n";
rp += "Do NOT claim 10x compression. Do NOT touch Jules/quality PRs. Do NOT merge #56/#57.\n";
rp += "Conventional commit: test(poc): measure CUDA train/compress/route on RTX 5080\n";
rp += "Push to origin and open PR against feat/poc-region-registry (stacked). Return pr_url.\n";
rp += "If push/PR fails, leave the commit local and set verdict=FAIL_PRODUCT with evidence.\n";
rp += "verdict=PASS only if STATUS has measured CUDA rows AND tests exist AND commit is created.\n";
let rr = agent(rp, #{
    label: "record",
    capability_mode: "all",
    output_schema: record_schema,
});

let rec_ok = false;
let pr_url = "";
if rr != () && rr.success && rr.output.verdict == "PASS" { rec_ok = true; }
if rr != () && rr.success && rr.output.pr_url != () { pr_url = rr.output.pr_url; }
if !rec_ok {
    let p4 = write_scratch_file("report.md", "Record/PR step failed after CUDA measure passed.");
    complete(#{ summary: "FAIL_PRODUCT record", path: p4, next: "finish STATUS/PR" });
}

phase("Review");
let vp = "";
vp += "Adversarially verify the CUDA STATUS claims in worktree ";
vp += worktree;
vp += " and any PR ";
vp += pr_url;
vp += ".\n";
vp += "Read STATUS.md and tests/test_poc_cuda.py (or equivalent). Use git diff and gh pr view.\n";
vp += "Set real=true only if STATUS CUDA numbers match evidence from this run and tests skip without CUDA.\n";
vp += "If any claim is unsubstantiated, real=false with evidence.\n";
let vr = agent(vp, #{
    label: "review",
    capability_mode: "read-only",
    output_schema: review_schema,
});

let confirmed = false;
if vr != () && vr.success && vr.output.real == true { confirmed = true; }

let summary = "CUDA measured; review rejected claims";
if confirmed { summary = "CUDA measured and verified; stacked PR ready"; }

let report = "";
report += "# CogSynDelta PoC GPU drive\n\n";
report += summary;
report += "\n\nPR: ";
report += pr_url;
report += "\n";
let path = write_scratch_file("report.md", report);
complete(#{
    summary: summary,
    path: path,
    pr_url: pr_url,
    confirmed: confirmed,
});
