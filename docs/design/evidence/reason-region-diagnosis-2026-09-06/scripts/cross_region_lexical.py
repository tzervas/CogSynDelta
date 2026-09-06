"""TF-IDF / BM25 baselines on the EXACT 512-pair holdout of code, compress and retrieve
(seed 0, matrix b512 config), plus a feasibility count for a corrupted-derivation battery on
reason's holdout. Read-only, CPU only."""

import glob
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/home/kang/code/personal/tzervas/CogSynDelta/src")
import numpy as np

from cogsyndelta.regions.pretrain import (
    PretrainConfig,
    _corpus_content_fingerprint,
    build_splits,
)
from cogsyndelta.regions.text_encoder import TextEncoderConfig

C = "/mnt/fleet-datasets/csd/region"
OUT = str(Path(__file__).resolve().parent / "cross_region_lexical.json")
WORD = re.compile(r"[a-z]+|\d+(?:\.\d+)?")


def words(t):
    return WORD.findall(t.lower())


def rank_metrics(scores):
    n = scores.shape[0]
    order = np.argsort(-scores, axis=1)
    ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(n)])
    return {
        "recall@1": float((ranks == 1).mean()),
        "recall@10": float((ranks <= 10).mean()),
        "mrr": float((1.0 / ranks).mean()),
    }


def lexical(holdout):
    A = [a for a, _ in holdout]
    P = [p for _, p in holdout]
    docs = [words(t) for t in A + P]
    df = Counter()
    for d in docs:
        df.update(set(d))
    N = len(docs)
    vocab = {w: i for i, w in enumerate(df)}
    idf = np.array([math.log((N + 1) / (df[w] + 1)) + 1 for w in vocab])

    def tfidf(d):
        v = np.zeros(len(vocab))
        for w, k in Counter(d).items():
            v[vocab[w]] = (1 + math.log(k)) * idf[vocab[w]]
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    TA = np.stack([tfidf(d) for d in docs[: len(A)]])
    TP = np.stack([tfidf(d) for d in docs[len(A) :]])
    rng = np.random.default_rng(0)
    out = {"tfidf": rank_metrics(TA @ TP.T + rng.uniform(0, 1e-9, (len(A), len(P))))}
    # BM25
    k1, b_ = 1.5, 0.75
    pdocs = [Counter(d) for d in docs[len(A) :]]
    dl = np.array([sum(d.values()) for d in pdocs])
    avgdl = dl.mean()
    dfp = Counter()
    for d in pdocs:
        dfp.update(d.keys())
    Np = len(pdocs)
    B = np.zeros((len(A), len(P)))
    for i, q in enumerate(docs[: len(A)]):
        for w in set(q):
            if w not in dfp:
                continue
            idf_w = math.log(1 + (Np - dfp[w] + 0.5) / (dfp[w] + 0.5))
            for j, d in enumerate(pdocs):
                f = d.get(w, 0)
                if f:
                    B[i, j] += idf_w * f * (k1 + 1) / (f + k1 * (1 - b_ + b_ * dl[j] / avgdl))
    out["bm25"] = rank_metrics(B + rng.uniform(0, 1e-9, B.shape))
    return out


def cfg_for(region, shards, cols, extra, max_len):
    return PretrainConfig(
        region=region,
        pair_columns=cols,
        shards=shards,
        extra_sources=extra,
        steps=4000,
        batch_size=512,
        lr=4.242640687119285e-4,
        warmup_steps=266,
        checkpoint_every=200,
        max_len=max_len,
        seed=0,
        eval_every=666,
        holdout_pairs=512,
        tokenizer_path="/mnt/fleet-datasets/tritter/gpt2_tokenizer.json",
        encoder=TextEncoderConfig(dim=256, depth=4, n_heads=4, max_len=max_len),
    )


REGIONS = {
    "code": (
        sorted(
            glob.glob(
                f"{C}/code/codesearchnet-python/**/train-*-of-00004-*.parquet", recursive=True
            )
        ),
        ("docstring", "code"),
        [],
        96,
        "e493273b003f5cf9c239db0ac461b993",
        0.98046875,
    ),
    "compress": (
        [f"{C}/compress/all-nli/pair/train-00000-of-00001.parquet"],
        ("anchor", "positive"),
        [],
        96,
        "65a9938192b5040beda5e92eb4078dbc",
        0.712890625,
    ),
    "retrieve": (
        [f"{C}/retrieve/fiqa-pairs/train.parquet"],
        ("query", "passage"),
        [
            {
                "shards": [f"{C}/retrieve/natural-questions/pair/train-00000-of-00001.parquet"],
                "columns": ["query", "answer"],
                "limit": 0,
            },
            {
                "shards": sorted(
                    glob.glob(f"{C}/retrieve/gooaq/pair/train-0000?-of-00002.parquet")
                ),
                "columns": ["question", "answer"],
                "limit": 400000,
            },
        ],
        96,
        "af4bb36f6f5dcacd0bf7f211df6ef63a",
        0.720703125,
    ),
}
results = {}
for name, (shards, cols, extra, max_len, fp_expected, model_r1) in REGIONS.items():
    t0 = time.time()
    cfg = cfg_for(name, shards, cols, extra, max_len)
    fp = _corpus_content_fingerprint(cfg)
    holdout, train_pairs, meta = build_splits(cfg)
    lex = lexical(holdout)
    results[name] = {
        "fingerprint": fp,
        "matches_receipt": fp == fp_expected,
        "holdout": len(holdout),
        "train_pairs": len(train_pairs),
        "model_r1_receipt_b512s0": model_r1,
        **lex,
        "seconds": round(time.time() - t0, 1),
    }
    print(name, results[name], flush=True)

# reason: feasibility of a corrupted-derivation battery on the held-out gsm8k positives
R = "/mnt/bulk/csd-corpus/reason"
cfg = PretrainConfig(
    region="reason",
    pair_columns=("question", "answer"),
    shards=[f"{R}/gsm8k-main/train.parquet"],
    extra_sources=[
        {
            "shards": [f"{R}/aqua_rat-raw/train.parquet"],
            "columns": ["question", "rationale"],
            "limit": 4982,
        }
    ],
    steps=4000,
    batch_size=256,
    lr=3e-4,
    warmup_steps=266,
    checkpoint_every=200,
    max_len=256,
    seed=0,
    eval_every=666,
    holdout_pairs=512,
    tokenizer_path="/mnt/fleet-datasets/tritter/gpt2_tokenizer.json",
    encoder=TextEncoderConfig(dim=256, depth=4, n_heads=4, max_len=256),
)
holdout, train_pairs, meta = build_splits(cfg)
CALC = re.compile(r"<<([^<>=]+)=([^<>]+)>>")
gsm = [(a, p) for a, p in holdout if "####" in p]
n_calc = [len(CALC.findall(p)) for _, p in gsm]
n_lines = [len([ln for ln in p.split("\n") if ln.strip()]) for _, p in gsm]
results["reason_corruption_feasibility"] = {
    "gsm8k_holdout_items": len(gsm),
    "with_ge1_calc_annotation": int(sum(1 for k in n_calc if k >= 1)),
    "with_ge2_calc_annotations": int(sum(1 for k in n_calc if k >= 2)),
    "with_ge4_calc_annotations": int(sum(1 for k in n_calc if k >= 4)),
    "mean_calc_annotations": float(np.mean(n_calc)),
    "mean_lines_incl_final": float(np.mean(n_lines)),
    "with_ge3_lines": int(sum(1 for k in n_lines if k >= 3)),
}
tr_gsm = [p for _, p in train_pairs if "####" in p]
tr_calc = [len(CALC.findall(p)) for p in tr_gsm]
results["reason_corruption_feasibility"]["train_gsm8k_with_ge2_calc"] = int(
    sum(1 for k in tr_calc if k >= 2)
)
results["reason_corruption_feasibility"]["train_gsm8k_items"] = len(tr_gsm)
print(results["reason_corruption_feasibility"])
with open(OUT, "w") as fh:
    json.dump(results, fh, indent=1)
print("wrote", OUT)
