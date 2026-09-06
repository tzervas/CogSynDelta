#!/usr/bin/env python3
"""Grade E2 (reason region epoch-matched batch ablation) from RUN-MANIFEST.json.

Deterministic: reads only the receipts named in RUN-MANIFEST.json, does no network or
clock-dependent work, and sorts everything it iterates over. Run twice and diff
results.json to confirm.

Pre-registration: docs/design/evidence/reason-region-diagnosis-2026-09-06/README.md:349-368

  H2: (512, 2000) ~= (256, 4000) within 0.03 r@1 and 0.04 r@10 (same ~87 epochs), and
      both beat (512, 4000) on r@1 -- required in every seed.
  H6: (512, 2000) trails (256, 4000) by > 0.05 r@10, same sign -- required in all three
      seeds.
  PRE:1018: peak - final > 0.02 r@1 in >= 4 of the 6 4,000-step runs triggers
      best-checkpoint retention.

Only the twelve required_arms (all trained under code 2fdc8b2, sharing the committed
split manifest) feed H2/H6/PRE:1018. secondary_disclosed_cells (the two a769409 cells
this run superseded) are reported for transparency only and never enter a verdict --
they predate the split.sha256 field and are a code-revision confound against the other
ten arms (see RUN-MANIFEST.json: code_revision_confound_note).
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "RUN-MANIFEST.json"

GO_H2_R1_TOL = 0.03
GO_H2_R10_TOL = 0.04
KILL_H6_R10_MARGIN = 0.05
PEAK_FINAL_GATE = 0.02
PEAK_FINAL_ARM_THRESHOLD = 4  # of 6 4000-step runs
FOUR_K_RUNS_REQUIRED = 6
SEEDS_REQUIRED = (0, 1, 2)
COMBOS = ((256, 2000), (256, 4000), (512, 2000), (512, 4000))


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _scored_row(batch, steps, seed, train_receipt, eval_receipt, tfidf_r1):
    tr = load_json(train_receipt)
    ev = load_json(eval_receipt)
    m = ev["metrics"]
    row = {
        "batch": batch,
        "steps": steps,
        "seed": seed,
        "status": "present",
        "final_r1": m["rank.recall@1"],
        "final_r10": m["rank.recall@10"],
        "final_mrr": m["rank.mrr"],
    }
    untrained = tr.get("untrained_baseline") or {}
    row["untrained_r1"] = untrained.get("recall@1")
    row["untrained_r10"] = untrained.get("recall@10")
    row["vs_untrained_r1_multiple"] = (
        round(row["final_r1"] / row["untrained_r1"], 4) if row["untrained_r1"] else None
    )
    row["lexical_ceiling_fraction_r1"] = round(row["final_r1"] / tfidf_r1, 6)

    hist = tr.get("history") or []
    r1_points = [h["held_recall@1"] for h in hist if "held_recall@1" in h]
    r10_points = [h["held_recall@10"] for h in hist if "held_recall@10" in h]
    peak_r1 = max(r1_points) if r1_points else row["final_r1"]
    peak_r10 = max(r10_points) if r10_points else row["final_r10"]
    row["peak_r1"] = peak_r1
    row["peak_r10"] = peak_r10
    row["peak_minus_final_r1"] = round(peak_r1 - row["final_r1"], 6)

    row["code_sha"] = tr.get("code_revision", {}).get("git_sha")
    row["split_sha256"] = (tr.get("split") or {}).get("sha256")
    return row


def build_arm_rows(manifest):
    tfidf_r1 = manifest["lexical_baseline"]["tfidf_recall@1"]
    rows = []
    for arm in manifest["required_arms"]:
        rows.append(
            _scored_row(
                arm["batch"],
                arm["steps"],
                arm["seed"],
                arm["train_receipt"],
                arm["eval_receipt"],
                tfidf_r1,
            )
        )
    rows.sort(key=lambda r: (r["seed"], r["steps"], r["batch"]))
    return rows


def build_secondary_rows(manifest):
    tfidf_r1 = manifest["lexical_baseline"]["tfidf_recall@1"]
    rows = []
    for arm in manifest.get("secondary_disclosed_cells", []):
        row = _scored_row(
            arm["batch"],
            arm["steps"],
            arm["seed"],
            arm["train_receipt"],
            arm["eval_receipt"],
            tfidf_r1,
        )
        row["note"] = arm.get("note")
        rows.append(row)
    rows.sort(key=lambda r: (r["batch"],))
    return rows


def evaluate_h2_h6(rows):
    """H2 and H6 are both within-seed, four-arm comparisons; evaluated separately for
    each seed in {0, 1, 2}, then aggregated (CONFIRMED only if true in every seed, per
    the pre-registration's "in every seed" / "in all three seeds" wording)."""
    by_key = {(r["batch"], r["steps"], r["seed"]): r for r in rows}
    complete_seeds = [
        seed for seed in SEEDS_REQUIRED if all((b, s, seed) in by_key for (b, s) in COMBOS)
    ]

    per_seed = {}
    for seed in SEEDS_REQUIRED:
        if seed not in complete_seeds:
            per_seed[seed] = {"h2": "INSUFFICIENT_DATA", "h6": "INSUFFICIENT_DATA"}
            continue
        a = by_key[(512, 2000, seed)]  # H2/H6 "(512, 2000)"
        b = by_key[(256, 4000, seed)]  # H2/H6 "(256, 4000)"
        c = by_key[(512, 4000, seed)]  # H2's "(512, 4000)" both must beat

        r1_gap = round(abs(a["final_r1"] - b["final_r1"]), 6)
        r10_gap = round(abs(a["final_r10"] - b["final_r10"]), 6)
        same_epoch = r1_gap <= GO_H2_R1_TOL and r10_gap <= GO_H2_R10_TOL
        beats_512_4000 = a["final_r1"] > c["final_r1"] and b["final_r1"] > c["final_r1"]
        h2_seed = same_epoch and beats_512_4000

        h6_margin = round(b["final_r10"] - a["final_r10"], 6)
        h6_seed = h6_margin > KILL_H6_R10_MARGIN

        per_seed[seed] = {
            "h2": "CONFIRMED" if h2_seed else "NOT_CONFIRMED",
            "h2_r1_gap": r1_gap,
            "h2_r10_gap": r10_gap,
            "h2_beats_512_4000": beats_512_4000,
            "h6": "CONFIRMED" if h6_seed else "NOT_CONFIRMED",
            "h6_r10_margin_256st4000_minus_512st2000": h6_margin,
        }

    result = {
        "seeds_required": list(SEEDS_REQUIRED),
        "seeds_complete": complete_seeds,
        "per_seed": {str(k): v for k, v in per_seed.items()},
    }
    if len(complete_seeds) < len(SEEDS_REQUIRED):
        result["h2"] = "INSUFFICIENT_DATA"
        result["h6"] = "INSUFFICIENT_DATA"
    else:
        result["h2"] = (
            "CONFIRMED"
            if all(per_seed[s]["h2"] == "CONFIRMED" for s in SEEDS_REQUIRED)
            else "NOT_CONFIRMED"
        )
        result["h6"] = (
            "CONFIRMED"
            if all(per_seed[s]["h6"] == "CONFIRMED" for s in SEEDS_REQUIRED)
            else "NOT_CONFIRMED"
        )
    return result


def evaluate_peak_final_gate(rows):
    four_k = sorted((r for r in rows if r["steps"] == 4000), key=lambda r: (r["seed"], r["batch"]))
    per_run = [
        {
            "batch": r["batch"],
            "seed": r["seed"],
            "peak_minus_final_r1": r["peak_minus_final_r1"],
            "triggers": r["peak_minus_final_r1"] > PEAK_FINAL_GATE,
        }
        for r in four_k
    ]
    triggered = sum(1 for r in per_run if r["triggers"])
    evaluable = len(four_k) >= FOUR_K_RUNS_REQUIRED
    return {
        "runs_at_4000_steps_available": len(four_k),
        "runs_at_4000_steps_required": FOUR_K_RUNS_REQUIRED,
        "per_run": per_run,
        "runs_triggering_gate": triggered,
        "trigger_threshold_runs": PEAK_FINAL_ARM_THRESHOLD,
        "gate_evaluable": evaluable,
        "gate_fired": (triggered >= PEAK_FINAL_ARM_THRESHOLD) if evaluable else "INSUFFICIENT_DATA",
    }


def overall_verdict(manifest, hh, peak_final):
    present = len(manifest["required_arms"])
    rationale_bits = [
        f"All {present}/12 pre-registered arms are present, all trained under code "
        f"{manifest['code_sha_all_required_arms']} against the committed split "
        f"({manifest['split_sha256'][:12]}...), resolving the code-revision confound "
        "the two originally-reused seed-0 cells carried (no split.sha256, code "
        "a769409). H2 " + hh["h2"] + ", H6 " + hh["h6"] + "."
    ]
    if hh["h2"] == "CONFIRMED":
        rationale_bits.append(
            "The matrix row moves to an epoch budget and 'b256 > b512' is retired as a "
            "batch finding."
        )
    else:
        rationale_bits.append(
            "The matrix row does not move to an epoch budget; 'b256 > b512' is not retired."
        )
    if hh["h6"] == "CONFIRMED":
        rationale_bits.append("A temperature arm (0.05 vs 0.10) joins E3.")
    else:
        rationale_bits.append("No temperature arm is added to E3 on this evidence.")
    if peak_final["gate_fired"] is True:
        rationale_bits.append("PRE:1018 fires: best-checkpoint retention is added.")
    elif peak_final["gate_fired"] is False:
        rationale_bits.append("PRE:1018 does not fire: best-checkpoint retention is not added.")
    else:
        rationale_bits.append("PRE:1018 is not evaluable (insufficient 4000-step runs).")

    return {
        "arms_present": present,
        "arms_required": 12,
        "h2": hh["h2"],
        "h6": hh["h6"],
        "pre_1018_gate_fired": peak_final["gate_fired"],
        "rationale": " ".join(rationale_bits),
    }


def main():
    manifest = load_json(MANIFEST_PATH)
    rows = build_arm_rows(manifest)
    secondary_rows = build_secondary_rows(manifest)
    hh = evaluate_h2_h6(rows)
    peak_final = evaluate_peak_final_gate(rows)
    verdict = overall_verdict(manifest, hh, peak_final)

    out = {
        "experiment": "E2",
        "pre_registration": manifest["pre_registration"],
        "manifest_sha256": sha256_of(MANIFEST_PATH),
        "lexical_baseline": manifest["lexical_baseline"],
        "arms": rows,
        "secondary_disclosed_cells": secondary_rows,
        "h2_h6": hh,
        "best_checkpoint_retention_gate": peak_final,
        "verdict": verdict,
    }

    print(json.dumps(out, indent=2, sort_keys=True))
    return out


if __name__ == "__main__":
    main()
