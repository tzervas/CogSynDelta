#!/usr/bin/env python3
"""Grade the E5 latent-step-vs-sequence-blind receipts against the pre-registered rule.

Deterministic: reads the six receipts named by the run lane from this directory's
`receipts/` subdirectory, extracts each arm/seed's `predictor_battery.recall@1`
(the pre-registered metric), pairs `latent-step` against `sequence-blind` within each
seed, computes the seed spread, and applies the pre-registered go/kill rule verbatim
(`GO_RECALL_FLOOR = 0.40`, `GO_MARGIN_OVER_BLIND = 0.10`, `KILL_MARGIN_OVER_BLIND =
0.05`, mirrored from `src/cogsyndelta/regions/reason_latent_step.py:141-143` so this
grader has no hidden dependency on the worktree's Python environment). Prints a table
and writes nothing itself -- the caller redirects stdout, or reads the returned dict,
to produce results.json. Run twice; both runs must produce byte-identical results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Mirrored constants -- see docstring. Must match reason_latent_step.py exactly.
GO_RECALL_FLOOR = 0.40
GO_MARGIN_OVER_BLIND = 0.10
KILL_MARGIN_OVER_BLIND = 0.05

RECEIPTS_DIR = Path(__file__).resolve().parent / "receipts"

RECEIPT_FILES = {
    ("latent-step", 0): "reason-e5-latent-step-s0-20260906T043659Z.json",
    ("latent-step", 1): "reason-e5-latent-step-s1-20260906T050808Z.json",
    ("latent-step", 2): "reason-e5-latent-step-s2-20260906T060614Z.json",
    ("sequence-blind", 0): "reason-e5-sequence-blind-s0-20260906T044700Z.json",
    ("sequence-blind", 1): "reason-e5-sequence-blind-s1-20260906T051944Z.json",
    ("sequence-blind", 2): "reason-e5-sequence-blind-s2-20260906T085530Z.json",
}

SEEDS = (0, 1, 2)
ARMS = ("latent-step", "sequence-blind")


def load_receipt(arm: str, seed: int) -> dict:
    path = RECEIPTS_DIR / RECEIPT_FILES[(arm, seed)]
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def go_kill_note(this_arm_recall: float, blind_recall: float | None) -> dict:
    """Pre-registered go/kill rule for one seed, given its paired blind score.

    Mirrors `reason_latent_step.go_kill_note` exactly (same constants, same branch
    order) so this grader does not depend on importing the worktree's package.
    """
    if blind_recall is None:
        return {
            "verdict": "pending",
            "this_arm_recall@1": this_arm_recall,
            "blind_recall@1": None,
            "note": "paired sequence-blind run not yet scored",
        }
    margin = this_arm_recall - blind_recall
    if this_arm_recall >= GO_RECALL_FLOOR and margin >= GO_MARGIN_OVER_BLIND:
        verdict = "go"
    elif margin <= KILL_MARGIN_OVER_BLIND:
        verdict = "kill"
    else:
        verdict = "none"
    return {
        "verdict": verdict,
        "this_arm_recall@1": this_arm_recall,
        "blind_recall@1": blind_recall,
        "margin_over_blind": margin,
        "go_floor": GO_RECALL_FLOOR,
        "go_margin": GO_MARGIN_OVER_BLIND,
        "kill_margin": KILL_MARGIN_OVER_BLIND,
    }


def build_results() -> dict:
    receipts = {(arm, seed): load_receipt(arm, seed) for arm in ARMS for seed in SEEDS}

    per_run = {}
    for (arm, seed), receipt in receipts.items():
        per_run[f"{arm}-s{seed}"] = {
            "arm": arm,
            "seed": seed,
            "recall@1": receipt["predictor_battery"]["recall@1"],
            "untrained_recall@1": receipt["untrained_predictor_battery"]["recall@1"],
            "e1_regression_guard_recall@1": receipt["e1_regression_guard"]["recall@1"],
            "elapsed_s": receipt["elapsed_s"],
            "corpus_fingerprint": receipt["corpus_fingerprint"],
            "split_sha256": receipt["split"]["sha256"],
            "batch_order_sha256": receipt["batch_order"]["sha256"],
            "code_revision": receipt["code_revision"]["git_sha"],
            "dirty": receipt["code_revision"]["dirty"],
        }

    # Control check: G26 -- same split + order manifest across every run (arms differ
    # only in objective).
    fingerprints = {r["corpus_fingerprint"] for r in per_run.values()}
    split_shas = {r["split_sha256"] for r in per_run.values()}
    order_shas = {r["batch_order_sha256"] for r in per_run.values()}
    control_check = {
        "single_corpus_fingerprint": len(fingerprints) == 1,
        "single_split_sha256": len(split_shas) == 1,
        "single_batch_order_sha256": len(order_shas) == 1,
        "corpus_fingerprint": next(iter(fingerprints))
        if len(fingerprints) == 1
        else sorted(fingerprints),
        "split_sha256": next(iter(split_shas)) if len(split_shas) == 1 else sorted(split_shas),
        "batch_order_sha256": next(iter(order_shas))
        if len(order_shas) == 1
        else sorted(order_shas),
    }

    per_seed = {}
    for seed in SEEDS:
        latent = per_run[f"latent-step-s{seed}"]["recall@1"]
        blind = per_run[f"sequence-blind-s{seed}"]["recall@1"]
        note = go_kill_note(latent, blind)
        per_seed[str(seed)] = {
            "latent_step_recall@1": latent,
            "sequence_blind_recall@1": blind,
            "margin": latent - blind,
            "verdict": note["verdict"],
        }

    verdicts = [v["verdict"] for v in per_seed.values()]
    margins = [v["margin"] for v in per_seed.values()]
    recalls_latent = [per_seed[str(s)]["latent_step_recall@1"] for s in SEEDS]

    if any(v == "kill" for v in verdicts):
        overall = "kill"
    elif all(v == "go" for v in verdicts):
        overall = "go"
    else:
        overall = "none"

    seed_spread = {
        "latent_step_recall@1": {
            "min": min(recalls_latent),
            "max": max(recalls_latent),
            "spread": max(recalls_latent) - min(recalls_latent),
        },
        "margin_over_blind": {
            "min": min(margins),
            "max": max(margins),
            "spread": max(margins) - min(margins),
        },
    }

    return {
        "schema": "csd-e5-grade/v1",
        "pre_registered_rule": {
            "go": f"predictor acc@1 >= {GO_RECALL_FLOOR:.2f} AND >= sequence-blind + {GO_MARGIN_OVER_BLIND:.2f}, in every seed",
            "kill": f"predictor acc@1 <= sequence-blind + {KILL_MARGIN_OVER_BLIND:.2f}, in any seed",
        },
        "per_run": per_run,
        "per_seed": per_seed,
        "seed_spread": seed_spread,
        "control_check": control_check,
        "overall_verdict": overall,
    }


def print_table(results: dict) -> None:
    print("E5 latent-step vs sequence-blind -- per-run recall@1")
    print(
        f"{'arm':<15}{'seed':<6}{'recall@1':<12}{'untrained':<12}{'e1_guard':<10}{'elapsed_s':<10}"
    )
    for key in (f"{a}-s{s}" for s in SEEDS for a in ARMS):
        r = results["per_run"][key]
        print(
            f"{r['arm']:<15}{r['seed']:<6}{r['recall@1']:<12.4f}"
            f"{r['untrained_recall@1']:<12.4f}{r['e1_regression_guard_recall@1']:<10.4f}"
            f"{r['elapsed_s']:<10.1f}"
        )
    print()
    print("per-seed go/kill (margin = latent-step - sequence-blind)")
    print(f"{'seed':<6}{'latent-step':<14}{'blind':<10}{'margin':<10}{'verdict':<10}")
    for seed in SEEDS:
        v = results["per_seed"][str(seed)]
        print(
            f"{seed:<6}{v['latent_step_recall@1']:<14.4f}{v['sequence_blind_recall@1']:<10.4f}"
            f"{v['margin']:<10.4f}{v['verdict']:<10}"
        )
    print()
    print(f"seed spread (latent-step recall@1): {results['seed_spread']['latent_step_recall@1']}")
    print(
        f"control check (G26 -- single manifest across all 6 runs): "
        f"fingerprint={results['control_check']['single_corpus_fingerprint']} "
        f"split={results['control_check']['single_split_sha256']} "
        f"order={results['control_check']['single_batch_order_sha256']}"
    )
    print()
    print(f"OVERALL VERDICT: {results['overall_verdict'].upper()}")


def main() -> int:
    results = build_results()
    print_table(results)
    out_path = Path(__file__).resolve().parent / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
