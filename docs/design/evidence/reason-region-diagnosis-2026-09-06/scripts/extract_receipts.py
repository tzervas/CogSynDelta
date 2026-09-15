"""Extract the reason-region receipts into one table. Read-only."""

import glob
import json
from pathlib import Path

ROWS = []
CELLS = sorted(glob.glob("/akula-data/csd/matrix/reason-b*-*-7bc2699-20260904"))
OLD = "/akula-data/csd/receipts"


def load(path):
    with open(path) as f:
        return json.load(f)


def one(label, train_p, eval_p, quant_p, evalq_p):
    t = load(train_p)
    e = load(eval_p) if eval_p else None
    q = load(quant_p) if quant_p else None
    eq = load(evalq_p) if evalq_p else None
    h = t["history"]
    row = {
        "cell": label,
        "code": t["code_revision"]["git_sha"][:7],
        "batch": t["config"]["batch_size"],
        "seed": t["config"]["seed"],
        "lr": t["config"]["lr"],
        "warmup": t["config"]["warmup_steps"],
        "steps": t["config"]["steps"],
        "max_len": t["config"]["max_len"],
        "train_pairs": t["corpus"]["train_pairs"],
        "holdout": t["corpus"]["holdout_pairs"],
        "dups_removed": t["corpus"].get("duplicates_removed"),
        "fp": t["corpus"]["fingerprint"],
        "params": t["parameters"],
        "elapsed_s": t["elapsed_s"],
        "train_tokens": t.get("tokenisation", {}).get("train_tokens"),
        "ragged": t.get("tokenisation", {}).get("ragged_ratio"),
        "untrained_r1": t["untrained_baseline"]["recall@1"],
        "untrained_r10": t["untrained_baseline"]["recall@10"],
        "untrained_mrr": t["untrained_baseline"]["mrr"],
        "untrained_emb_std": t["untrained_baseline"]["emb_std"],
        "r1": t["held_out"]["recall@1"],
        "r10": t["held_out"]["recall@10"],
        "mrr": t["held_out"]["mrr"],
        "emb_std": t["held_out"]["emb_std"],
        "final_loss": h[-1]["loss"],
        "final_in_batch_acc": h[-1]["in_batch_acc"],
        "cap_per_param": t.get("capability_per_param"),
        "pooled_pr_rank": t.get("token_aware", {})
        .get("final_block_rank", {})
        .get("pooled_pr_rank"),
        "token_pr_rank": t.get("token_aware", {})
        .get("final_block_rank", {})
        .get("token_global_pr_rank"),
        "pooled_ent_rank": t.get("token_aware", {})
        .get("final_block_rank", {})
        .get("pooled_entropy_rank"),
        "token_ent_rank": t.get("token_aware", {})
        .get("final_block_rank", {})
        .get("token_global_entropy_rank"),
        "history": [
            (
                x["step"],
                round(x["loss"], 3),
                round(x["in_batch_acc"], 3),
                x["held_recall@1"],
                x["held_recall@10"],
            )
            for x in h
        ],
        "contam_pos_content": t["contamination"]["channels"]["positive_content"]["overlap"],
        "contam_anchor_content": t["contamination"]["channels"]["anchor_content"]["overlap"],
        "contam_pos_exact": t["contamination"]["channels"]["positive_exact"]["overlap"],
    }
    if e:
        m = e["metrics"]
        row.update(
            {
                "eval_r1": m["rank.recall@1"],
                "eval_r5": m["rank.recall@5"],
                "eval_r10": m["rank.recall@10"],
                "eval_mrr": m["rank.mrr"],
                "eval_ndcg10": m["rank.ndcg@10"],
                "candidates": m["rank.candidates"],
                "aniso": m["repr.anisotropy"],
                "align": m["repr.alignment"],
                "unif": m["repr.uniformity"],
                "eff_rank": m["repr.effective_rank"],
                "eff_rank_ratio": m["repr.effective_rank_ratio"],
                "lat_p50": m["eff.latency_p50_ms"],
                "peak_vram_mb": m["eff.peak_vram_mb"],
                "eval_s": e.get("seconds"),
                "stored_mb": m["eff.stored_mb"],
            }
        )
    if q:
        row.update(
            {
                "q_fp32": q["fp32_metric_recomputed"],
                "q_metric": q["quantized_metric"],
                "q_drop": q["drop"],
                "q_ratio": q["compression_ratio"],
                "q_hist": q["width_histogram"],
                "q_within": q["within_budget"],
                "q_stored": q["stored_bytes"],
            }
        )
    if eq:
        m = eq["metrics"]
        row.update(
            {
                "eq_r1": m["rank.recall@1"],
                "eq_r10": m["rank.recall@10"],
                "eq_mrr": m["rank.mrr"],
                "eq_ndcg10": m["rank.ndcg@10"],
                "eq_aniso": m["repr.anisotropy"],
                "eq_eff_rank": m["repr.effective_rank"],
            }
        )
    return row


for c in CELLS:
    r = c + "/receipts/"
    tr = sorted(glob.glob(r + "reason-2026*.json"))
    ev = sorted(glob.glob(r + "cogsyndelta-reason-eval-2026*.json"))
    qu = sorted(glob.glob(r + "reason-quant-*.json"))
    eq = sorted(glob.glob(r + "cogsyndelta-reason-eval-quantized-*.json"))
    ROWS.append(
        one(
            Path(c).name,
            tr[-1],
            ev[-1] if ev else None,
            qu[-1] if qu else None,
            eq[-1] if eq else None,
        )
    )

# older baseline receipts
ROWS.append(
    one(
        "OLD reason-20260903T123431Z (b512 s0, 0026a3d)",
        OLD + "/reason-20260903T123431Z.json",
        OLD + "/cogsyndelta-reason-eval-20260903T123729Z.json",
        OLD + "/reason-quant-20260903T123741Z.json",
        None,
    )
)
# the second quant receipt of the old run
q2 = load(OLD + "/reason-quant-20260903T155346Z.json")
print(
    "OLD second quant receipt:",
    {
        k: q2.get(k)
        for k in (
            "fp32_metric_recomputed",
            "quantized_metric",
            "drop",
            "compression_ratio",
            "width_histogram",
        )
    },
)
print("OLD train receipt code rev:", load(OLD + "/reason-20260903T123431Z.json")["code_revision"])
old_t = load(OLD + "/reason-20260903T123431Z.json")
print("OLD train receipt keys:", sorted(old_t.keys()))
print("OLD tokenisation:", old_t.get("tokenisation"))

for r in ROWS:
    print("=" * 100)
    for k, v in r.items():
        print(f"{k}: {v}")

with open(Path(__file__).resolve().parent / "reason_receipts.json", "w") as fh:
    json.dump(ROWS, fh, indent=1, default=str)
