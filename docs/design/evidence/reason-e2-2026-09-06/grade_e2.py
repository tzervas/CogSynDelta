#!/usr/bin/env python3
"""Grade E2 (reason region epoch-matched batch ablation) from RUN-MANIFEST.json.

Deterministic: reads only the receipts named in RUN-MANIFEST.json, does no
network or clock-dependent work, and sorts everything it iterates over. Run
twice and diff results.json to confirm.

Pre-registration: docs/design/evidence/reason-region-diagnosis-2026-09-06/README.md:349-368
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "RUN-MANIFEST.json"

TFIDF_R1 = 0.873046875  # diagnosis README.md:349-368, cell reason-b512-s0 lexical receipt
GO_H2_R1_TOL = 0.03
GO_H2_R10_TOL = 0.04
KILL_H6_R10_MARGIN = 0.05
PEAK_FINAL_GATE = 0.02
PEAK_FINAL_ARM_THRESHOLD = 4  # of 6 4000-step runs


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def build_arm_rows(manifest):
    rows = []
    for arm in manifest["required_arms"]:
        row = {
            "batch": arm["batch"],
            "steps": arm["steps"],
            "seed": arm["seed"],
            "status": arm["status"],
        }
        if arm["status"] == "present":
            ev = load_json(arm["eval_receipt"])
            tr = load_json(arm["train_receipt"])
            m = ev["metrics"]
            row["final_r1"] = m["rank.recall@1"]
            row["final_r10"] = m["rank.recall@10"]
            row["final_mrr"] = m["rank.mrr"]
            hist = tr.get("history") or []
            r1_points = [h["held_recall@1"] for h in hist if "held_recall@1" in h]
            r10_points = [h["held_recall@10"] for h in hist if "held_recall@10" in h]
            row["peak_r1"] = max(r1_points) if r1_points else row["final_r1"]
            row["peak_r10"] = max(r10_points) if r10_points else row["final_r10"]
            row["peak_minus_final_r1"] = round(row["peak_r1"] - row["final_r1"], 6)
            row["lexical_fraction_r1"] = round(row["final_r1"] / TFIDF_R1, 6)
        else:
            row["reason"] = arm.get("reason", "missing")
        rows.append(row)
    rows.sort(key=lambda r: (r["seed"], r["steps"], r["batch"]))
    return rows


def seeds_present(rows, batch, steps):
    return sorted(
        r["seed"]
        for r in rows
        if r["batch"] == batch and r["steps"] == steps and r["status"] == "present"
    )


def evaluate_h2_h6(rows):
    """H2/H6 both require, per seed s, all four (batch,steps) arms at that seed.
    With only seed 0 populated (1 of 3 required seeds), neither prediction can
    be evaluated as pre-registered -- both are reported INSUFFICIENT_DATA.
    """
    combos = [(256, 2000), (256, 4000), (512, 2000), (512, 4000)]
    complete_seeds = []
    for seed in (0, 1, 2):
        have = all(
            any(
                r["batch"] == b
                and r["steps"] == s
                and r["seed"] == seed
                and r["status"] == "present"
                for r in rows
            )
            for (b, s) in combos
        )
        if have:
            complete_seeds.append(seed)

    result = {
        "seeds_required": 3,
        "seeds_complete": complete_seeds,
        "h2": "INSUFFICIENT_DATA",
        "h6": "INSUFFICIENT_DATA",
    }
    if not complete_seeds:
        return result

    # (kept for when data exists; unreachable at current lane state)
    h2_ok, h6_ok = [], []
    for seed in complete_seeds:
        by_combo = {
            (r["batch"], r["steps"]): r
            for r in rows
            if r["seed"] == seed and r["status"] == "present"
        }
        a = by_combo[(512, 2000)]
        b = by_combo[(256, 4000)]
        c = by_combo[(512, 4000)]
        same_epoch = (
            abs(a["final_r1"] - b["final_r1"]) <= GO_H2_R1_TOL
            and abs(a["final_r10"] - b["final_r10"]) <= GO_H2_R10_TOL
        )
        beats_b512s4000 = a["final_r1"] > c["final_r1"] and b["final_r1"] > c["final_r1"]
        h2_ok.append(same_epoch and beats_b512s4000)
        h6_ok.append((b["final_r10"] - a["final_r10"]) > KILL_H6_R10_MARGIN)

    result["h2"] = "CONFIRMED" if all(h2_ok) else "NOT_CONFIRMED"
    result["h6"] = "CONFIRMED" if all(h6_ok) else "NOT_CONFIRMED"
    return result


def evaluate_peak_final_gate(rows):
    four_k = [r for r in rows if r["status"] == "present" and r["steps"] == 4000]
    triggered = [r for r in four_k if r["peak_minus_final_r1"] > PEAK_FINAL_GATE]
    return {
        "runs_at_4000_steps_available": len(four_k),
        "runs_at_4000_steps_required": 6,
        "runs_triggering_gate": len(triggered),
        "trigger_threshold_runs": PEAK_FINAL_ARM_THRESHOLD,
        "gate_evaluable": len(four_k) >= 6,
        "gate_fired": len(triggered) >= PEAK_FINAL_ARM_THRESHOLD
        if len(four_k) >= 6
        else "INSUFFICIENT_DATA",
    }


def overall_verdict(manifest, hh, e1):
    present = sum(1 for a in manifest["required_arms"] if a["status"] == "present")
    total = len(manifest["required_arms"])
    return {
        "arms_present": present,
        "arms_required": total,
        "h2": hh["h2"],
        "h6": hh["h6"],
        "e1_battery_verdict": e1["verdict"],
        "decision": "BLOCKED_INSUFFICIENT_DATA",
        "rationale": (
            f"Only {present}/{total} pre-registered arms exist (both reused seed-0 "
            "cells; the other 10 were never trained -- model-matrix run never "
            "executed because /akula-data/csd/matrix/selection.json holds a stale "
            "sticky selection for a different region and this lane is read-only on "
            "the matrix root). H2 and H6 each require a within-seed comparison "
            "across all three seeds; with 1 of 3 seeds populated, neither can be "
            "confirmed or refuted as pre-registered. The matrix row does not move "
            "to an epoch budget, the b256>b512 batch finding is not retired, and no "
            "temperature arm is added to E3 on this evidence. The E1 kill (all "
            "arms <= 0.25 on the corrupted-derivation battery) stands independently "
            "and continues to demote reason diagonal r@1 to a lexical-ceiling "
            "fraction regardless of E2's outcome."
        ),
    }


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = load_json(MANIFEST_PATH)
    rows = build_arm_rows(manifest)
    hh = evaluate_h2_h6(rows)
    peak_final = evaluate_peak_final_gate(rows)
    verdict = overall_verdict(manifest, hh, manifest["e1_battery_receipts"])

    out = {
        "experiment": "E2",
        "pre_registration": manifest["pre_registration"],
        "manifest_sha256": sha256_of(MANIFEST_PATH),
        "arms": rows,
        "h2_h6": hh,
        "best_checkpoint_retention_gate": peak_final,
        "verdict": verdict,
    }

    print(json.dumps(out, indent=2, sort_keys=True))
    return out


if __name__ == "__main__":
    main()
