#!/usr/bin/env python3
"""Intake accounting for the gsm8k test split: disjointness, balance, battery size."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path("/home/kang/code/personal/tzervas/csd-worktress/gsm8k-test")
sys.path.insert(0, str(ROOT / "src"))

import pyarrow.parquet as pq  # noqa: E402

from cogsyndelta.corpus import fingerprint_corpus  # noqa: E402
from cogsyndelta.splits import item_id, load_reserved_holdout_ids  # noqa: E402

CORPUS = Path("/mnt/bulk/csd-corpus/reason")
GSM_TRAIN = CORPUS / "gsm8k-main" / "train.parquet"
GSM_TEST = CORPUS / "gsm8k-main" / "test.parquet"
AQUA = CORPUS / "aqua_rat-raw" / "train.parquet"
AQUA_CAP = 4982

out: dict[str, object] = {}


def pairs(path: Path, cols: tuple[str, str]) -> list[tuple[str, str]]:
    t = pq.read_table(str(path), columns=list(cols))
    return [
        (str(a), str(b))
        for a, b in zip(t.column(cols[0]).to_pylist(), t.column(cols[1]).to_pylist(), strict=True)
    ]


gsm_train = pairs(GSM_TRAIN, ("question", "answer"))
gsm_test = pairs(GSM_TEST, ("question", "answer"))
aqua_rows = pq.read_table(str(AQUA), columns=["question"]).num_rows

# ---- 1. corpus fingerprint must not move -------------------------------------------
fp = fingerprint_corpus(
    [str(GSM_TRAIN)],
    columns=["question", "answer"],
    extra_sources=[
        {"shards": [str(AQUA)], "columns": ["question", "rationale"], "limit": AQUA_CAP}
    ],
)
out["reason_corpus_fingerprint"] = fp
out["fingerprint_unchanged_vs_E0_ca364a92"] = fp.startswith("ca364a92")

# ---- 2. item-level disjointness, measured not assumed -------------------------------
train_ids = {item_id(a, b) for a, b in gsm_train}
test_ids = {item_id(a, b) for a, b in gsm_test}
reserved = load_reserved_holdout_ids(splits_dir=ROOT / "config" / "mind" / "splits")
out["gsm8k_train_rows"] = len(gsm_train)
out["gsm8k_test_rows"] = len(gsm_test)
out["test_ids_unique"] = len(test_ids)
out["overlap_test_vs_gsm8k_train_ids"] = len(test_ids & train_ids)
# Question-only overlap: a shared question with a different answer would still be a leak
# of the problem, so it is measured separately rather than folded into the pair check.
out["overlap_test_vs_train_question_text"] = len(
    {" ".join(q.split()).lower() for q, _ in gsm_test}
    & {" ".join(q.split()).lower() for q, _ in gsm_train}
)
out["reserved_manifest_ids"] = len(reserved)
out["reserved_covers_every_test_row"] = test_ids <= reserved


# ---- 3. balance accounting, before and after ----------------------------------------
def balance(shares: dict[str, int]) -> dict[str, object]:
    total = sum(shares.values())
    p = {k: v / total for k, v in shares.items()}
    n_eff = 1.0 / sum(x * x for x in p.values())
    entropy = -sum(x * math.log(x) for x in p.values() if x > 0)
    return {
        "total_rows": total,
        "shares": {k: round(v, 6) for k, v in p.items()},
        "max_share": round(max(p.values()), 6),
        "n_eff_inverse_simpson": round(n_eff, 4),
        "exp_shannon": round(math.exp(entropy), 4),
        "B1_max_share_le_0.40": max(p.values()) <= 0.40,
        "B2_n_eff_ge_3": n_eff >= 3.0,
    }


out["balance"] = {
    "staged_pool_before": balance({"aqua_rat": aqua_rows, "gsm8k": len(gsm_train)}),
    "staged_pool_after": balance({"aqua_rat": aqua_rows, "gsm8k": len(gsm_train) + len(gsm_test)}),
    "staged_pool_after_training_material_only": balance(
        {"aqua_rat": aqua_rows, "gsm8k": len(gsm_train)}
    ),
    "realised_train_split_before": balance({"aqua_rat": AQUA_CAP, "gsm8k": len(gsm_train)}),
    "realised_train_split_after": balance({"aqua_rat": AQUA_CAP, "gsm8k": len(gsm_train)}),
}

# ---- 4. battery eligibility ---------------------------------------------------------
from cogsyndelta.eval.corrupted_derivation import (  # noqa: E402
    CORRUPTION_SEED,
    build_corrupted_battery,
)

items_now = build_corrupted_battery(gsm_test, corruption_seed=CORRUPTION_SEED)
out["battery"] = {
    "test_split_rows": len(gsm_test),
    "eligible_items_from_test_split": len(items_now),
    "eligible_today_from_in_mixture_holdout": 299,
    "mean_annotations": round(sum(i.n_annotations for i in items_now) / max(1, len(items_now)), 4),
}

print(json.dumps(out, indent=2, sort_keys=True))
