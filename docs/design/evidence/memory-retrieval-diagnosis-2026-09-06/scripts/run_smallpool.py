"""Addendum to run_diag.py: the same arms on the SMALL (split) pool, so the lexical
bar exists at memory-store scale as well as at corpus scale. Evaluation only."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = "/home/kang/code/personal/tzervas/csd-worktress/CogSynDelta-wt-memory-diag"
sys.path.insert(0, f"{REPO}/src")

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from cogsyndelta.eval import beir_fiqa  # noqa: E402
from cogsyndelta.regions._checkpoint import load_checkpoint  # noqa: E402
from cogsyndelta.regions.pretrain import _tokenize as tokenize_batch  # noqa: E402
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig  # noqa: E402

OUT = Path("/akula-data/session-backup-staging/tmp/memory-diag/results_smallpool.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT = {
    "memory": (
        "/akula-data/session-backup-staging/w4-chunked/run-1280/memory-checkpoints/369351bf/final.pt",
        "fb3c53fa28c683a333ec8886e7a3d6ead01e8b7743833991f8fb859e194d2a7a",
    ),
    "retrieve": (
        "/akula-data/csd/matrix/retrieve-b1280-s0-7bc2699-20260904/receipts/retrieve-checkpoints/1d2bb223/final.pt",
        "1351cf45944331b1cc63f94b234c3fc964341034660a624c362f3b5469909540",
    ),
}


def main() -> None:
    task = beir_fiqa.build_ranking_task("dev", pool="split")
    res: dict = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(DEVICE),
        "pool": "split",
        "task_summary": {
            "queries": float(len(task.queries)),
            "pool_size": float(len(task.pool_texts)),
            "relevant_per_query": sum(len(g) for g in task.gold) / len(task.gold),
        },
        "arms": {},
    }

    tok = Tokenizer.from_file("/mnt/fleet-datasets/tritter/gpt2_tokenizer.json")

    def tokenize(texts: list[str]):
        return tokenize_batch(tok, texts, 96, DEVICE)

    for name, (path, sha) in CKPT.items():
        state = load_checkpoint(Path(path), expected_sha256=sha)
        model = TextEncoder(TextEncoderConfig(**state["config"]), name="diag").to(DEVICE)
        model.load_state_dict(state["model"])
        model.eval()
        with torch.no_grad():
            res["arms"][name] = beir_fiqa.encoder_rank_metrics(model, tokenize, task)
        del model
        torch.cuda.empty_cache()

    res["arms"]["bm25"] = beir_fiqa.bm25_metrics(task)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
