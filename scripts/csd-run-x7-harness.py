#!/usr/bin/env python3
"""Run row W3's episode harness over the landed X7 items and write a receipt.

WHAT THIS SCRIPT IS FOR. `cogsyndelta.eval.episode_harness` is the deliverable; this is the
reference run of it -- the thing that turns "the harness exists" into a receipt somebody can
read. It builds a CPU mind with X7's own two ablation-pair partners (`language`, `memory`)
plus the episodic store, runs every requested arm of every requested episode, and writes
`harness_receipt` to a file.

WHAT ITS NUMBERS ARE, AND ARE NOT.

  - The **store evidence is real**. `EpisodicStoreImpl` over `SqliteBackend` is E1's durable
    store, and `learn` commits through `StoreBackend.put` before it returns a `WriteReceipt`,
    so "turn 1's write committed before turn 2 was scored" is a property of this run and not
    of a stub. The read is PR #86's query-dependent read. `read_returned_turn1_record` is an
    observation of the call the forward pass made.
  - The **three gates are real**. They key on the store path and on the items' own declared
    counterfactuals, neither of which depends on a trained weight.
  - The **model accuracy is not evidence of anything**. The faculties are real
    `TextFacultyAdapter(TextEncoder(...))` modules at UNTRAINED weights, behind a hashing
    stand-in tokeniser, because `language` and `memory` have no token-aware retrain yet (W1)
    and X7 is not admitted (NSRS `s_r` is GPU-blocked and blocked on W2b). `rank_head_pass`
    is reported so W5 inherits a wired scorer; reading it as a capability number would be the
    error this whole row exists to make impossible.

THREADS. `--threads` pins `torch.set_num_threads` and the receipt records it beside
`OMP_NUM_THREADS`, because CPU reductions are not associative and a run's verdicts must be
reproducible from what the receipt says.

USAGE

    export OMP_NUM_THREADS=1
    export TMPDIR=/akula-data/csd/tmp
    python scripts/csd-run-x7-harness.py \
        --items /akula-data/csd/reserve/x7/x7-eval.jsonl --limit 64 \
        --out docs/design/evidence/w3-episode-harness-2026-09-07/receipt.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cogsyndelta.eval.episode_harness import (  # noqa: E402
    EpisodeArm,
    EpisodeHarness,
    HashingTurnEncoder,
    StoreProbe,
    harness_receipt,
    load_episodes,
)
from cogsyndelta.faculty.adapters import TextFacultyAdapter  # noqa: E402
from cogsyndelta.interconnect.episodic import (  # noqa: E402
    EpisodicStoreImpl,
    SqliteBackend,
    TierBudget,
)
from cogsyndelta.interconnect.mind import (  # noqa: E402
    InterconnectConfig,
    ParticipantSpec,
    WhiteMatter,
)
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig  # noqa: E402

DOMAIN = "episode"
SEQ_LEN = 8
D_W = 64
N_OPTIONS = 5
VOCAB = 1024
CAPACITY_BYTES = 1 << 30
"""Byte capacity handed to the store for this run: large enough that nothing evicts.

Deliberate. DEC-63's dynamic capacity is E1's own gate and is measured there; a harness run
that silently evicted a turn-1 record would report a store failure as a recall failure, which
is precisely the confusion this row exists to prevent. The number is recorded in the receipt
so a reader can see the run was not capacity-bound.
"""


class BoolMaskTextFaculty:
    """`TextFacultyAdapter` with its padding mask cast to `bool`.

    A RECORDED INTEROP GAP, not a design choice. `Faculty.tokens` documents its mask as *"1
    for a real position, 0 for padding"* and does not fix a dtype; `TextEncoder.tokens`
    returns `torch.ones(b, t, dtype=h.dtype)` -- FLOAT -- when the caller passes no attention
    mask, while `WhiteMatter._iterate` combines it with `mask & self._budget_mask(...)`, which
    raises `NotImplementedError: "bitwise_and_cpu" not implemented for 'Float'`. So the real
    text adapter and the real interconnect do not currently compose on a mask-less input.

    Passing an explicit bool `attention_mask` as an `(ids, mask)` tuple is the other way out
    and does not work either: `WhiteMatter._batch_size` infers `B` from the first TENSOR in
    `inputs`, and turn 1 carries no `candidates`, so an all-tuple request has no tensor to
    infer from.

    Neither is this row's to fix -- both sit on a merged forward path and deserve their own
    lane with their own can-fail test. This shim keeps the harness running against the REAL
    `TextEncoder` and the REAL `TextFacultyAdapter` instead of substituting a toy faculty,
    and does exactly one thing so the gap stays visible.
    """

    def __init__(self, adapter: TextFacultyAdapter) -> None:
        """Wrap one adapter.

        Args:
            adapter: The `TextFacultyAdapter` to delegate to.
        """
        self.adapter = adapter
        self.name = adapter.name
        self.faculty = adapter.faculty
        self.token_dim = adapter.token_dim
        self.pooled_dim = adapter.pooled_dim
        self.kv_bytes_per_token = adapter.kv_bytes_per_token
        self.accepts_condition = adapter.accepts_condition

    def tokens(self, inputs, *, context_tokens: int, condition=None):
        """Delegate, then cast the mask.

        Args:
            inputs: `[B, T]` token ids.
            context_tokens: `ctx_r`.
            condition: Must be `None`; the adapter refuses anything else.

        Returns:
            `(h, mask)` with `mask` as `torch.bool`.
        """
        h, mask = self.adapter.tokens(inputs, context_tokens=context_tokens, condition=condition)
        return h, mask.bool()

    def pool(self, h, mask):
        """Delegate pooling unchanged.

        Args:
            h: `[B, T, token_dim]`.
            mask: `[B, T]`.

        Returns:
            `[B, pooled_dim]`.
        """
        return self.adapter.pool(h, mask)


def build_mind(seed: int) -> tuple[WhiteMatter, StoreProbe, EpisodicStoreImpl, Path]:
    """Build the CPU mind, its durable store and the probe wrapping it.

    Args:
        seed: Seed for the untrained region weights, so two runs of this script produce the
            same (meaningless) model scores rather than two different ones.

    Returns:
        `(mind, probe, store, sqlite path)`.
    """
    torch.manual_seed(seed)
    encoder_cfg = TextEncoderConfig(
        vocab_size=VOCAB, dim=D_W, depth=2, n_heads=4, max_len=64, out_dim=D_W
    )
    faculties = {
        name: BoolMaskTextFaculty(
            TextFacultyAdapter(TextEncoder(encoder_cfg, name=name), faculty=name)
        )
        for name in ("language", "memory")
    }
    for faculty in faculties.values():
        faculty.adapter.encoder.eval()
        for parameter in faculty.adapter.encoder.parameters():
            parameter.requires_grad_(False)

    # ctx_min == ctx_max == SEQ_LEN: `TextFacultyAdapter.tokens` REFUSES an input longer than
    # its budget (DEC-15) rather than truncating it, and `_raw_summary` encodes at `ctx_min`.
    # A smaller floor would make the summary pass raise instead of the region encoding less.
    spec = ParticipantSpec(
        ctx_min=SEQ_LEN, ctx_max=SEQ_LEN, token_budget_min=2, token_budget_max=8, phi=1.0
    )
    config = InterconnectConfig(
        participants={
            "language": spec,
            "memory": spec,
            "episodic_store": ParticipantSpec(
                ctx_min=None, ctx_max=None, token_budget_min=2, token_budget_max=8, phi=0.0
            ),
        },
        workspace_dim=D_W,
        latents=8,
        n_iter=2,
        heads=4,
        mlp_ratio=2,
        budget_total_read_tokens=16,
        budget_total_kv_bytes=4_000_000,
        floor_eta=0.15,
        n_cond=4,
        controller_dim=32,
        controller_depth=1,
        controller_heads=2,
        write_back=True,
        k_candidates=N_OPTIONS + 1,
        rank_temperature=1.0,
        allowed_modalities=("text",),
        resident_heads=("text",),
    )

    db_dir = Path(tempfile.mkdtemp(prefix="csd-x7-harness-", dir=os.environ.get("TMPDIR")))
    db_path = db_dir / "episodes.sqlite"
    store = EpisodicStoreImpl(
        {DOMAIN},
        backend=SqliteBackend(db_path),
        capacity_provider=lambda: CAPACITY_BYTES,
        host="harness",
        tier_budget=TierBudget(ram_max_items=8192, max_records=131_072),
    )
    store.start()
    probe = StoreProbe(store)
    return WhiteMatter(config, faculties, probe), probe, store, db_path


def main() -> int:
    """Parse arguments, run the harness, write the receipt.

    Returns:
        `0` when both gates fired and the store was load-bearing; `1` otherwise. A run whose
        gates did not fire is not a passing run with a caveat -- it is a run that measured
        nothing, and the exit code says so.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items",
        default="/akula-data/csd/reserve/x7/x7-eval.jsonl",
        help="X7 JSONL shard to run.",
    )
    parser.add_argument("--limit", type=int, default=64, help="Episodes to run (file order).")
    parser.add_argument("--out", required=True, help="Where to write the receipt JSON.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the untrained weights.")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch.set_num_threads for this run; recorded in the receipt.",
    )
    parser.add_argument(
        "--episode-rows",
        type=int,
        default=None,
        help=(
            "Per-episode rows to keep in the receipt. Gates and summaries are always "
            "computed over the whole run; this only bounds the row listing."
        ),
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    items = load_episodes(args.items, limit=args.limit)
    mind, probe, store, db_path = build_mind(args.seed)
    encoder = HashingTurnEncoder(
        {"language": VOCAB, "memory": VOCAB}, seq_len=SEQ_LEN, workspace_dim=D_W
    )
    harness = EpisodeHarness(mind, probe, encoder, domain=DOMAIN)

    arms = [EpisodeArm.FULL, EpisodeArm.WRONG_TURN1, EpisodeArm.NO_TURN1]
    results = harness.run(items, arms)
    receipt = harness_receipt(
        results,
        config={
            "items": args.items,
            "limit": args.limit,
            "episodes": len(items),
            "arms": [arm.value for arm in arms],
            "seed": args.seed,
            "torch_num_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "torch_version": torch.__version__,
            "store": {
                "implementation": "EpisodicStoreImpl",
                "backend": "SqliteBackend",
                "capacity_bytes": CAPACITY_BYTES,
                "durable_commit": "learn() -> StoreBackend.put -> WriteReceipt",
            },
            "faculties": {
                "adapter": "TextFacultyAdapter(TextEncoder)",
                "weights": "UNTRAINED -- model scores are not evidence",
                "tokeniser": encoder.name,
            },
            "workspace_dim": D_W,
            "k_candidates": N_OPTIONS + 1,
        },
        max_episode_rows=args.episode_rows,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    store.stop()
    # `rmtree`, not `unlink` + `rmdir`: WAL mode leaves `-wal` and `-shm` siblings beside the
    # database, so removing only the file it was asked for left a non-empty directory behind.
    shutil.rmtree(db_path.parent, ignore_errors=True)

    gates = receipt["gates"]
    print(f"episodes: {len(items)}  results: {len(results)}  receipt: {out_path}")
    for arm_name, stats in receipt["arms"].items():
        print(
            f"  {arm_name:<12} oracle_pass {stats['oracle_pass']}/{stats['episodes']}"
            f"  read_returned_turn1 {stats['read_returned_turn1_record']}"
            f"  rank_head_pass {stats['rank_head_pass']}"
        )
    print(f"  gate (i)   negative control : {gates['negative_control']}")
    print(f"  gate (iii) partition reset  : {gates['partition_reset']}")
    print(f"  store load-bearing          : {receipt['store']}")

    fired = (
        gates["negative_control"]["gate_fires"]
        and gates["partition_reset"]["gate_fires"]
        and receipt["store"]["store_is_load_bearing"]
    )
    return 0 if fired else 1


if __name__ == "__main__":
    raise SystemExit(main())
