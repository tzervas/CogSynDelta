"""Read-only CPU analysis of the reason region: corpus lengths, lexical baselines on the
exact 512-pair holdout, and per-item diagnostics of the trained checkpoints.

CUDA_VISIBLE_DEVICES must be empty. Nothing is written outside the scratchpad.
"""

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/home/kang/code/personal/tzervas/CogSynDelta/src")

import numpy as np
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer

from cogsyndelta.regions.pretrain import (
    PretrainConfig,
    _corpus_content_fingerprint,
    build_splits,
    evaluate,
)
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig

ROOT = "/mnt/bulk/csd-corpus/reason"
TOK_PATH = "/mnt/fleet-datasets/tritter/gpt2_tokenizer.json"
OUT = str(Path(__file__).resolve().parent / "reason_analysis.json")
CELLS = {
    "b256-s0": "/akula-data/csd/matrix/reason-b256-s0-7bc2699-20260904/receipts/reason-checkpoints/ab5ba4db/final.pt",
}
# find the b512-s0 final checkpoint
import glob  # noqa: E402

for c in glob.glob(
    "/akula-data/csd/matrix/reason-b512-s0-7bc2699-20260904/receipts/reason-checkpoints/*/final.pt"
):
    CELLS["b512-s0"] = c

results: dict = {}
tok = Tokenizer.from_file(TOK_PATH)


def pct(xs, q):
    return float(np.percentile(xs, q))


def length_stats(texts, label):
    enc = tok.encode_batch(list(texts))
    lens = np.array([len(e.ids) for e in enc])
    st = {
        "n": len(lens),
        "mean": float(lens.mean()),
        "p50": pct(lens, 50),
        "p90": pct(lens, 90),
        "p99": pct(lens, 99),
        "max": int(lens.max()),
        "frac_gt_96": float((lens > 96).mean()),
        "frac_gt_128": float((lens > 128).mean()),
        "frac_gt_256": float((lens > 256).mean()),
    }
    print(label, st)
    return st


# ---------------------------------------------------------------- 1. corpus lengths
gsm = pq.read_table(f"{ROOT}/gsm8k-main/train.parquet").to_pydict()
aq_all = pq.read_table(
    f"{ROOT}/aqua_rat-raw/train.parquet", columns=["question", "rationale", "correct"]
).to_pydict()
results["lengths"] = {
    "gsm8k.question": length_stats(gsm["question"], "gsm8k.question"),
    "gsm8k.answer": length_stats(gsm["answer"], "gsm8k.answer"),
    "aqua_rat_all.question": length_stats(aq_all["question"], "aqua_rat_all.question"),
    "aqua_rat_all.rationale": length_stats(aq_all["rationale"], "aqua_rat_all.rationale"),
}
# aqua_rat rationale quality: how many end with an explicit answer letter, how many are very short
rat = aq_all["rationale"]
short = sum(1 for r in rat if len(r.split()) < 8)
ans_line = sum(1 for r in rat if re.search(r"(answer|ans)\W*(is|:)?\s*\(?[A-E]\)?", r, re.I))
results["aqua_rat_quality"] = {
    "n": len(rat),
    "rationale_lt_8_words": short,
    "rationale_with_answer_letter": ans_line,
    "dup_rationales": len(rat) - len(set(rat)),
    "dup_questions": len(aq_all["question"]) - len(set(aq_all["question"])),
}
print("aqua_rat quality", results["aqua_rat_quality"])
gsm_ans = gsm["answer"]
results["gsm8k_quality"] = {
    "n": len(gsm_ans),
    "with_final_marker": sum(1 for a in gsm_ans if "####" in a),
    "with_calc_annotations": sum(1 for a in gsm_ans if "<<" in a),
    "dup_questions": len(gsm["question"]) - len(set(gsm["question"])),
}
print("gsm8k quality", results["gsm8k_quality"])

# ---------------------------------------------------------------- 2. exact holdout
cfg = PretrainConfig(
    region="reason",
    pair_columns=("question", "answer"),
    shards=[f"{ROOT}/gsm8k-main/train.parquet"],
    extra_sources=[
        {
            "shards": [f"{ROOT}/aqua_rat-raw/train.parquet"],
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
    tokenizer_path=TOK_PATH,
    encoder=TextEncoderConfig(dim=256, depth=4, n_heads=4, max_len=256),
)
fp = _corpus_content_fingerprint(cfg)
holdout, train_pairs, meta = build_splits(cfg)
results["split"] = {
    "fingerprint": fp,
    "matches_receipt": fp == "ca364a92d2c6c5fd259404e0ab6f52a1",
    "holdout": len(holdout),
    "train_pairs": len(train_pairs),
    "duplicates_removed": meta["duplicates_removed"],
    "source_counts": meta["source_counts"],
}
print("split", results["split"])


def is_gsm(positive: str) -> bool:
    return "####" in positive


src = np.array(["gsm8k" if is_gsm(p) else "aqua_rat" for _, p in holdout])
results["holdout_sources"] = dict(Counter(src.tolist()))
print("holdout sources", results["holdout_sources"])
train_src = Counter("gsm8k" if is_gsm(p) else "aqua_rat" for _, p in train_pairs)
results["train_sources"] = dict(train_src)
print("train sources", results["train_sources"])

# epochs
for b in (256, 512, 1280):
    results[f"epochs_b{b}"] = 4000 * b / len(train_pairs)
print("epochs", {k: v for k, v in results.items() if k.startswith("epochs")})


# ---------------------------------------------------------------- 3. lexical baselines
def rank_metrics(scores: np.ndarray, label: str, mask_src=None):
    n = scores.shape[0]
    order = np.argsort(-scores, axis=1)
    ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(n)])
    out = {
        "recall@1": float((ranks == 1).mean()),
        "recall@10": float((ranks <= 10).mean()),
        "mrr": float((1.0 / ranks).mean()),
    }
    for s in ("gsm8k", "aqua_rat"):
        m = src == s
        out[f"recall@1[{s}]"] = float((ranks[m] == 1).mean())
        out[f"recall@10[{s}]"] = float((ranks[m] <= 10).mean())
    # source confusion: is the top-1 from the same source as the anchor
    top1_src = src[order[:, 0]]
    out["top1_same_source"] = float((top1_src == src).mean())
    print(label, out)
    return out, ranks


WORD = re.compile(r"[a-z]+|\d+(?:\.\d+)?")
NUM = re.compile(r"\d+(?:\.\d+)?")


def words(t):
    return WORD.findall(t.lower())


def nums(t):
    return set(NUM.findall(t))


A = [a for a, _ in holdout]
P = [p for _, p in holdout]

# 3a. TF-IDF cosine over words
docs = [words(t) for t in A + P]
df = Counter()
for d in docs:
    df.update(set(d))
N = len(docs)
vocab = {w: i for i, w in enumerate(df)}
idf = np.array([math.log((N + 1) / (df[w] + 1)) + 1 for w in vocab])


def tfidf(d):
    v = np.zeros(len(vocab))
    c = Counter(d)
    for w, k in c.items():
        v[vocab[w]] = (1 + math.log(k)) * idf[vocab[w]]
    nrm = np.linalg.norm(v)
    return v / nrm if nrm else v


TA = np.stack([tfidf(d) for d in docs[: len(A)]])
TP = np.stack([tfidf(d) for d in docs[len(A) :]])
results["baseline_tfidf"], r_tfidf = rank_metrics(TA @ TP.T, "TF-IDF cosine")

# 3b. numeric-token overlap (Jaccard over the numbers in the text)
NA = [nums(t) for t in A]
NP = [nums(t) for t in P]
S = np.zeros((len(A), len(P)))
for i, na in enumerate(NA):
    for j, npj in enumerate(NP):
        u = len(na | npj)
        S[i, j] = (len(na & npj) / u) if u else 0.0
# tiny deterministic tie-break so argsort is stable and not index-biased
rng = np.random.default_rng(0)
S = S + rng.uniform(0, 1e-6, S.shape)
results["baseline_number_jaccard"], r_num = rank_metrics(S, "number Jaccard")

# 3c. BM25 (question as query, positive as document)
k1, b_ = 1.5, 0.75
dl = np.array([len(d) for d in docs[len(A) :]])
avgdl = dl.mean()
pdocs = [Counter(d) for d in docs[len(A) :]]
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
B = B + rng.uniform(0, 1e-6, B.shape)
results["baseline_bm25"], r_bm25 = rank_metrics(B, "BM25")

# ---------------------------------------------------------------- 4. trained checkpoints on CPU
device = torch.device("cpu")


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    enc = TextEncoderConfig(**ck["config"])
    m = TextEncoder(enc)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def embed(model, texts, max_len=256):
    from cogsyndelta.regions.pretrain import _tokenize

    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            ids, mask = _tokenize(tok, texts[i : i + 64], max_len, device)
            outs.append(torch.nn.functional.normalize(model(ids, mask), dim=-1))
    return torch.cat(outs)


def scrub_numbers(t):
    return NUM.sub("N", t)


results["models"] = {}
for name, path in CELLS.items():
    model, ck = load_model(path)
    rep = evaluate(model, tok, holdout, 256, device)
    print(name, "reproduced held_out", rep)
    a = embed(model, A)
    p = embed(model, P)
    sc = (a @ p.T).numpy()
    m_all, r_model = rank_metrics(sc, f"{name} model")
    # numbers scrubbed on both sides: does the model rely on digit matching?
    a2 = embed(model, [scrub_numbers(t) for t in A])
    p2 = embed(model, [scrub_numbers(t) for t in P])
    m_scrub, _ = rank_metrics((a2 @ p2.T).numpy(), f"{name} model, numbers scrubbed")
    # anchor truncated to 96 tokens (the fleet default) at eval only
    rep96 = evaluate(model, tok, holdout, 96, device)
    # within-source pools: rank gsm8k anchors only against gsm8k positives
    within = {}
    for s in ("gsm8k", "aqua_rat"):
        idx = np.where(src == s)[0]
        sub = sc[np.ix_(idx, idx)]
        order = np.argsort(-sub, axis=1)
        ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(len(idx))])
        within[s] = {
            "pool": len(idx),
            "recall@1": float((ranks == 1).mean()),
            "recall@10": float((ranks <= 10).mean()),
            "mrr": float((1 / ranks).mean()),
        }
    print(name, "within-source", within)
    # overlap of the model's hits with the lexical baselines' hits
    hits_model = r_model == 1
    hits_num = r_num == 1
    hits_bm25 = r_bm25 == 1
    hits_tfidf = r_tfidf == 1
    overlap = {
        "model_hits": int(hits_model.sum()),
        "number_hits": int(hits_num.sum()),
        "bm25_hits": int(hits_bm25.sum()),
        "tfidf_hits": int(hits_tfidf.sum()),
        "model_and_bm25": int((hits_model & hits_bm25).sum()),
        "model_and_number": int((hits_model & hits_num).sum()),
        "model_and_tfidf": int((hits_model & hits_tfidf).sum()),
        "model_only_vs_bm25": int((hits_model & ~hits_bm25).sum()),
        "model_hit_rate_when_bm25_hit": float(hits_model[hits_bm25].mean())
        if hits_bm25.any()
        else None,
        "model_hit_rate_when_bm25_miss": float(hits_model[~hits_bm25].mean()),
        "model_rank_median": float(np.median(r_model)),
        "model_rank_p75": float(np.percentile(r_model, 75)),
    }
    print(name, "overlap", overlap)
    # anisotropy / effective rank on the pooled holdout embeddings (both sides)
    both = torch.cat([a, p]).double()
    sim = both @ both.T
    n = both.shape[0]
    aniso = float((sim.sum() - sim.diag().sum()) / (n * (n - 1)))
    cov = torch.cov(both.T.float())
    ev = torch.linalg.eigvalsh(cov).clamp(min=0)
    pr = float(ev.sum() ** 2 / (ev**2).sum())
    pv = ev / ev.sum()
    ent = float(torch.exp(-(pv[pv > 0] * pv[pv > 0].log()).sum()))
    results["models"][name] = {
        "reproduced": rep,
        "reproduced_at_max_len_96": rep96,
        "full_pool": m_all,
        "numbers_scrubbed": m_scrub,
        "within_source": within,
        "overlap_with_lexical": overlap,
        "anisotropy_recomputed": aniso,
        "pr_rank_recomputed": pr,
        "entropy_rank_recomputed": ent,
        "history": ck.get("history", [])[-2:],
    }

# lexical baselines' per-source
with open(OUT, "w") as fh:
    json.dump(results, fh, indent=1, default=str)
print("wrote", OUT)
