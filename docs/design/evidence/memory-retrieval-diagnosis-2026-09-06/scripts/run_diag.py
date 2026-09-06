"""W4 memory-region diagnosis: full-pool BEIR vs held-out diagonal.

Follows the scout's pre-registration exactly. Evaluation only: loads existing
checkpoints, runs no training. Writes results incrementally to results.json.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

REPO = "/home/kang/code/personal/tzervas/csd-worktress/CogSynDelta-wt-memory-diag"
sys.path.insert(0, f"{REPO}/src")

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from cogsyndelta.eval import beir_fiqa  # noqa: E402
from cogsyndelta.regions._checkpoint import load_checkpoint  # noqa: E402
from cogsyndelta.regions.memory import memory_config  # noqa: E402
from cogsyndelta.regions.pretrain import (  # noqa: E402
    _prepare_graded,
    build_splits,
    evaluate,
    evaluate_graded,
)
from cogsyndelta.regions.pretrain import (  # noqa: E402
    _tokenize as tokenize_batch,
)
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig  # noqa: E402

RESULTS_PATH = Path("/akula-data/session-backup-staging/tmp/memory-diag/results.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = {
    "memory": {
        "path": "/akula-data/session-backup-staging/w4-chunked/run-1280/memory-checkpoints/369351bf/final.pt",
        "sha256": "fb3c53fa28c683a333ec8886e7a3d6ead01e8b7743833991f8fb859e194d2a7a",
        "code_revision": "eb735ab46063cb (branch feat/w4-masked-token-loss)",
    },
    "retrieve": {
        "path": "/akula-data/csd/matrix/retrieve-b1280-s0-7bc2699-20260904/receipts/retrieve-checkpoints/1d2bb223/final.pt",
        "sha256": "1351cf45944331b1cc63f94b234c3fc964341034660a624c362f3b5469909540",
        "code_revision": "a7694090903664bc256b4b96d998b37cacd316cf (a769409)",
    },
    "compress": {
        "path": "/akula-data/csd/matrix/compress-b1280-s0-7bc2699-20260904/receipts/compress-checkpoints/116eae1c/final.pt",
        "sha256": "e456293ffe39a719f22b57c521c0f27d3e0e1ee127b84e25300922c24da99f42",
        "code_revision": "a7694090903664bc256b4b96d998b37cacd316cf (a769409)",
    },
}

results: dict = {
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "device": str(DEVICE),
    "arms": {},
    "excluded": {},
}


def save() -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"[saved] {RESULTS_PATH}")


def load_encoder(path: str, expected_sha256: str) -> TextEncoder:
    state = load_checkpoint(path, expected_sha256=expected_sha256, map_location=DEVICE)
    model = TextEncoder(TextEncoderConfig(**state["config"]), name="diag").to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def main() -> None:
    tok_path = "/mnt/fleet-datasets/tritter/gpt2_tokenizer.json"
    tok = Tokenizer.from_file(tok_path)

    def tokenize(texts: list[str]):
        return tokenize_batch(tok, texts, 96, DEVICE)

    # ---- Load checkpoints, verifying sha256 first (load_checkpoint checks before
    # torch.load ever opens the file). Any failure excludes the arm explicitly. ----
    models: dict[str, TextEncoder] = {}
    for name, info in CHECKPOINTS.items():
        try:
            models[name] = load_encoder(info["path"], info["sha256"])
            results["arms"].setdefault(name, {})["checkpoint"] = info["path"]
            results["arms"][name]["checkpoint_sha256"] = info["sha256"]
            results["arms"][name]["code_revision"] = info["code_revision"]
            print(f"[loaded] {name}: {info['path']}")
        except Exception as exc:
            results["excluded"][name] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            print(f"[EXCLUDED] {name}: {exc}")
    save()

    # ---- Untrained baseline: same architecture (dim=256, depth=4, n_heads=4,
    # max_len=96), fresh init, seed=0 (matches memory receipt's own untrained arm). ----
    torch.manual_seed(0)
    untrained_cfg = TextEncoderConfig(vocab_size=50257, dim=256, depth=4, n_heads=4, max_len=96)
    untrained = TextEncoder(untrained_cfg, name="untrained").to(DEVICE)
    untrained.eval()
    results["arms"]["untrained"] = {
        "checkpoint": None,
        "note": "fresh init, seed=0, same architecture",
    }
    save()

    # ---- Full-pool BEIR task: dev split, all 57,638 FiQA passages -- the SAME task,
    # SAME pool, SAME qrels for every arm. Built ONCE and reused. ----
    print("[building] full-pool ranking task (dev split, corpus pool)...")
    task = beir_fiqa.build_ranking_task("dev", pool="corpus")
    results["task_summary"] = task.summary()
    print(f"[task] {task.summary()}")
    save()

    print("[scoring] BM25...")
    bm25 = beir_fiqa.bm25_metrics(task)
    results["arms"]["bm25"] = {"checkpoint": None, "full_pool": bm25}
    print(f"[bm25] {bm25}")
    save()

    for name, model in {**models, "untrained": untrained}.items():
        print(f"[scoring] {name} full-pool...")
        t0 = time.time()
        metrics = beir_fiqa.encoder_rank_metrics(model, tokenize, task)
        metrics["eval_seconds"] = round(time.time() - t0, 1)
        results["arms"][name]["full_pool"] = metrics
        print(f"[{name}] full_pool: {metrics}")
        save()

    # ---- Extra question (a): does memory's own held-out battery (the 512-pair
    # diagonal over the UNION training corpus) still score what its receipt claims?
    # Reproduce build_splits/_prepare_graded EXACTLY as pretrain_region did: same
    # seed=0/split_seed=0(default), same steps/batch_size (only affects the reservoir
    # budget, which does not bind here), same max_len. No training -- evaluate() and
    # evaluate_graded() only. ----
    if "memory" in models:
        print("[reproducing] memory's own held-out split (union corpus, seed=0)...")
        cfg = memory_config(steps=4000, batch_size=1280, max_len=96, holdout_pairs=512, seed=0)
        holdout, train_pairs, split_meta = build_splits(cfg)
        graded, graded_report = _prepare_graded(cfg, train_pairs)
        held = evaluate(models["memory"], tok, holdout, cfg.max_len, DEVICE)
        graded_res = evaluate_graded(models["memory"], tok, graded, cfg.max_len, DEVICE)
        results["own_holdout_reproduction"] = {
            "held_out": held,
            "graded_held_out": graded_res,
            "n_holdout": len(holdout),
            "n_graded": len(graded),
            "split_membership_seed": split_meta.get("membership_seed"),
            "receipt_claimed": {
                "held_out": {
                    "recall@1": 0.853515625,
                    "recall@10": 0.984375,
                    "mrr": 0.9058414697647095,
                },
                "graded_held_out": {"spearman": 0.7893104522457324},
            },
        }
        print(f"[own-holdout] held_out={held}")
        print(f"[own-holdout] graded_held_out={graded_res}")
        save()

    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    print("DONE")


if __name__ == "__main__":
    main()
