"""The interconnect's command line -- the harness entry point over `phase_a.py`.

    python -m cogsyndelta.interconnect.cli phase-a --steps 3 --device cpu --out-dir DIR
    python -m cogsyndelta.interconnect.cli phase-a --threads 8 --out-dir DIR
    python -m cogsyndelta.interconnect.cli params

WHY `run_phase_a`'S FIRST STATEMENT IS A THREAD PIN
The trained weights depend on the intra-op thread count: at one seed and one command line,
an unpinned sweep of `OMP_NUM_THREADS` over 1..28 produced eleven distinct
`checkpoint_sha256` and flipped G29's verdict. `phase_a.DEFAULT_PHASE_A_THREADS` carries
the measurement. Omitting `--threads` picks that default and stamps the receipt
`source: "default"`; there is no argv here that produces an unpinned run.

WHY THIS IS A SEPARATE FILE FROM `phase_a.py`
`phase_a.py` defines what phase A trains, freezes, optimises and gates; this file is
argv, a synthetic stream and an exit code. The split follows `cogsyndelta.poc.cli` and
`cogsyndelta.regions.compress`, the repo's two existing shapes for the same thing, and it
keeps the trainable/frozen partition testable without a parser in the way.

WHAT `--stream synthetic` IS, AND WHAT IT IS NOT
It is a planted-rule toy at the spec's own integration dimensions (`D_w = 64`, `L = 8`,
`n_iter = 2`, `B_read = 16`, `k = 4`) over two stand-in faculties. It exists so the real
command line is exercised end to end -- parser, trainer, gates, receipt on disk -- because
a green unit suite has repeatedly failed to predict what a stage script does. It is NOT a
phase-A row: no real region checkpoint is loaded and no corpus is read, so every receipt it
writes is stamped `battery_id: "plumbing"`, which spec section 6 Q5 option (c) reserves for
exactly this ("allowed only with `battery_id: plumbing` so nothing can cite it").

Loading real frozen region checkpoints and a real reserve manifest is the compose-stage
driver's job, which spec section 1 places outside this module and which does not exist
yet. `--stream` therefore accepts one value today; a second value is the driver's arrival,
not a flag change here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from cogsyndelta.interconnect.episodic_store import InMemoryStoreStub, derive_scope
from cogsyndelta.interconnect.kv_bank import STORE_PARTICIPANT
from cogsyndelta.interconnect.mind import InterconnectConfig, ParticipantSpec, WhiteMatter
from cogsyndelta.interconnect.phase_a import (
    DEFAULT_PHASE_A_THREADS,
    PHASE_A_TRAINABLE_R5,
    TABLE_4_TOTAL_R5,
    PhaseABatch,
    PhaseAConfig,
    PhaseATrainer,
    phase_a_parameter_partition,
    pin_threads,
)

__all__ = ["main"]

#: Table 1's four v1 regions plus Table 4a's context budgets, for the `params` subcommand.
V1_PARTICIPANTS: dict[str, ParticipantSpec] = {
    "language": ParticipantSpec(8, 96, 8, 96, 1.0),
    "memory": ParticipantSpec(8, 96, 8, 96, 1.0),
    "reasoning": ParticipantSpec(8, 256, 8, 96, 1.0),
    "visual": ParticipantSpec(8, 64, 8, 96, 2.0),
}
#: Table 1's `episodic_store` row: no `ctx` axis, no `phi`.
V1_STORE_SPEC = ParticipantSpec(None, None, 8, 96, 0.0)
#: The one server-derived scope every synthetic request reads and writes under -- see
#: `synthetic_batches` for why it is shared rather than per batch.
SYNTHETIC_SCOPE = derive_scope("cli-principal", session="synthetic")

#: Table 1's `token_dim` and `kv_bytes_per_token` per region.
V1_REGION_WIDTHS: dict[str, tuple[int, int]] = {
    "language": (256, 4096),
    "memory": (256, 4096),
    "reasoning": (256, 4096),
    "visual": (384, 9216),
}


class ConstructionOnlyFaculty(nn.Module):
    """A participant's construction-time surface and nothing else.

    `WhiteMatter.__init__` reads only `token_dim`, `pooled_dim`, `kv_bytes_per_token` and
    `accepts_condition`; the `params` subcommand never runs a forward pass, so `tokens`
    and `pool` raise rather than return a plausible-looking tensor that would let a
    mistaken caller measure something meaningless.
    """

    def __init__(self, name: str, token_dim: int, kv_bytes_per_token: int) -> None:
        """Declare one participant's four construction-time attributes."""
        super().__init__()
        self.name = name
        self.faculty = name
        self.token_dim = token_dim
        self.pooled_dim = token_dim
        self.kv_bytes_per_token = kv_bytes_per_token
        self.accepts_condition = True

    def tokens(
        self, inputs: Tensor, *, context_tokens: int, condition: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Refuse: this stand-in exists to be counted, never to be run."""
        raise NotImplementedError("ConstructionOnlyFaculty counts parameters; it does not run.")

    def pool(self, h: Tensor, mask: Tensor) -> Tensor:
        """Refuse: this stand-in exists to be counted, never to be run."""
        raise NotImplementedError("ConstructionOnlyFaculty counts parameters; it does not run.")


class SyntheticText(nn.Module):
    """A text-shaped stand-in faculty over integer ids, for `--stream synthetic`."""

    def __init__(self, vocab: int = 64, token_dim: int = 16) -> None:
        """Build the embedding table and the write-back conditioning projection."""
        super().__init__()
        self.name = "language"
        self.faculty = "language"
        self.token_dim = token_dim
        self.pooled_dim = token_dim
        self.kv_bytes_per_token = 64
        self.accepts_condition = True
        self.embed = nn.Embedding(vocab, token_dim)
        self.cond_proj = nn.Linear(token_dim, token_dim)

    def tokens(
        self, inputs: Tensor, *, context_tokens: int, condition: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Embed at most `context_tokens` ids, adding the conditioning prefix's mean.

        Args:
            inputs: `[B, T]` int64 ids.
            context_tokens: DEC-15's `ctx_r`; positions past it are never encoded.
            condition: `[B, n_cond, token_dim]` write-back prefix, or `None`.

        Returns:
            `(h, mask)` at `[B, min(T, ctx), token_dim]` and `[B, min(T, ctx)]`.
        """
        ids = inputs[:, :context_tokens]
        h = self.embed(ids)
        if condition is not None:
            h = h + self.cond_proj(condition.mean(dim=1, keepdim=True))
        mask = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
        return h, mask

    def pool(self, h: Tensor, mask: Tensor) -> Tensor:
        """Return the mask-weighted mean over positions."""
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype)
        return (h * mask.unsqueeze(-1)).sum(dim=1) / denom


class SyntheticVisual(nn.Module):
    """A patch-shaped stand-in faculty that refuses a condition, for `--stream synthetic`."""

    def __init__(self, raw_dim: int = 32, token_dim: int = 24) -> None:
        """Build the patch projection."""
        super().__init__()
        self.name = "visual"
        self.faculty = "visual"
        self.token_dim = token_dim
        self.pooled_dim = token_dim
        self.kv_bytes_per_token = 96
        self.accepts_condition = False
        self.proj = nn.Linear(raw_dim, token_dim)

    def tokens(
        self, inputs: Tensor, *, context_tokens: int, condition: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Project at most `context_tokens` patches.

        Args:
            inputs: `[B, T, raw_dim]` float patch features.
            context_tokens: DEC-15's `ctx_r`.
            condition: Must be `None`; this faculty declares `accepts_condition = False`.

        Returns:
            `(h, mask)`.

        Raises:
            TypeError: `condition` was not `None`.
        """
        if condition is not None:
            raise TypeError("SyntheticVisual.accepts_condition is False; condition must be None.")
        patches = inputs[:, :context_tokens]
        h = self.proj(patches)
        mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        return h, mask

    def pool(self, h: Tensor, mask: Tensor) -> Tensor:
        """Return the mask-weighted mean over positions."""
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype)
        return (h * mask.unsqueeze(-1)).sum(dim=1) / denom


def toy_config(*, k: int = 4, write_back: bool = True) -> InterconnectConfig:
    """Return the spec's own integration-test configuration, with the store admitted.

    Args:
        k: Candidate-set size, `RankHead`'s `k`.
        write_back: Whether the conditioning prefixes are built and trained.

    Returns:
        An `InterconnectConfig` at `D_w = 64`, `L = 8`, `n_iter = 2`, `B_read = 16`.
    """
    participants = {
        "language": ParticipantSpec(2, 8, 2, 8, 1.0),
        "visual": ParticipantSpec(2, 6, 2, 8, 1.5),
        STORE_PARTICIPANT: ParticipantSpec(None, None, 2, 8, 0.0),
    }
    return InterconnectConfig(
        participants=participants,
        workspace_dim=64,
        latents=8,
        n_iter=2,
        heads=4,
        mlp_ratio=2,
        budget_total_read_tokens=16,
        budget_total_kv_bytes=100_000,
        floor_eta=0.15,
        n_cond=4,
        controller_dim=32,
        controller_depth=1,
        controller_heads=2,
        write_back=write_back,
        k_candidates=k,
        rank_temperature=1.0,
        allowed_modalities=("text",),
        resident_heads=("text",),
    )


def synthetic_batches(
    *,
    count: int,
    batch_size: int,
    k: int,
    workspace_dim: int,
    task_seed: int,
    item_seed: int,
) -> list[PhaseABatch]:
    """Build a planted-rule synthetic stream: `k - 1` classes plus a `NULL` general bin.

    The rule is learnable but not trivial. Each item is assigned a class; its text ids and
    its patch features are drawn around that class's own centre, and the candidate bank is
    the SAME `k - 1` fixed vectors for every item, so `recall@1` is only achievable by
    mapping the input to the right candidate. One class in `k` is the general bin, whose
    gold answer is `NULL` at index 0 -- without it `null_recall` has no items to be
    measured on and G36 could never be evaluated.

    Args:
        count: Number of batches.
        batch_size: Items per batch.
        k: Candidate-set size; `k - 1` content candidates plus `NULL`.
        workspace_dim: `D_w`, the candidate embedding width.
        task_seed: Seeds the PLANTED RULE -- the candidate bank and the per-class text and
            patch centres. Train and dev must share it, or the two halves are different
            tasks and the overfit gate measures the seed rather than the run. Operator
            note `identical-seeds-as-control`: arms share the seed and a fixed held-out
            split so one change is what is measured.
        item_seed: Seeds only which items are DRAWN. Train and dev differ here and
            nowhere else, which is what makes the held-out half held out.

    Returns:
        `count` `PhaseABatch`es.
    """
    task_gen = torch.Generator().manual_seed(task_seed)
    bank = torch.randn(k - 1, workspace_dim, generator=task_gen)
    class_ids = torch.randint(0, 64, (k - 1, 8), generator=task_gen)
    class_patches = torch.randn(k - 1, 6, 32, generator=task_gen)
    gen = torch.Generator().manual_seed(item_seed)

    batches: list[PhaseABatch] = []
    for index in range(count):
        labels = torch.randint(0, k, (batch_size,), generator=gen)
        text = torch.randint(0, 64, (batch_size, 8), generator=gen)
        patches = torch.randn(batch_size, 6, 32, generator=gen) * 0.3
        target = torch.zeros(batch_size, dtype=torch.long)
        bins: list[str] = []
        for item in range(batch_size):
            label = int(labels[item])
            if label == k - 1:
                # The general bin: no class centre, gold answer is NULL at index 0.
                bins.append("general")
                continue
            text[item] = class_ids[label]
            patches[item] = patches[item] + class_patches[label]
            target[item] = label + 1
            bins.append("content")
        batches.append(
            PhaseABatch(
                inputs={
                    "language": text,
                    "visual": patches,
                    "candidates": bank.unsqueeze(0).expand(batch_size, -1, -1).clone(),
                    # ONE scope across every batch, deliberately. A per-batch scope means
                    # every read lands in a fresh, empty partition, the store contributes
                    # zero attention mass, and G29 fires on `episodic_store` for a reason
                    # that is an artefact of the stream rather than a property of the run.
                    "scope": SYNTHETIC_SCOPE,
                    "domain": "general",
                    "logical_key": f"turn-{index}",
                },
                target=target,
                bins=tuple(bins),
            )
        )
    return batches


def prime_store(white_matter: WhiteMatter, *, records: int, seed: int) -> int:
    """Write `records` episodes into the store before training, under the shared scope.

    Without this, step 0 reads an empty partition: the stub returns zero unmasked store
    slots, the store gets no attention mass on that step, and its running mean is dragged
    below `eta/R` by an initial condition rather than by anything the run did. Table 6's
    E2 row assumes "episode items X7 and X8 in the mix" -- a store with residents -- and
    this is the synthetic stream's smallest stand-in for that.

    Args:
        white_matter: The built module; its `store` receives the writes.
        records: How many episodes to write.
        seed: Generator seed for the episode vectors.

    Returns:
        The number of records written; `0` when the module has no store.
    """
    if white_matter.store is None:
        return 0
    gen = torch.Generator().manual_seed(seed)
    for index in range(records):
        vector = torch.randn(white_matter.config.workspace_dim, generator=gen)
        white_matter.store.write(SYNTHETIC_SCOPE, "general", f"primer-{index}", vector)
    return records


def _state_sha256(module: nn.Module) -> str:
    """Return a stable sha256 over a module's state dict, for a plumbing frozen-set row."""
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def synthetic_frozen_set(white_matter: WhiteMatter) -> list[dict[str, Any]]:
    """Build Table 7's frozen-set rows for a synthetic run, in the module's own order.

    The shas are real -- measured off each stand-in faculty's own state dict -- and each
    row cites itself, so G33's identity half passes honestly rather than by being handed
    two copies of a constant. `status` says `synthetic` so no reader mistakes these rows
    for retrained region checkpoints.

    Args:
        white_matter: The built module, for its participant order.

    Returns:
        One row per participant.
    """
    rows: list[dict[str, Any]] = []
    for name in white_matter.participant_names:
        if name == STORE_PARTICIPANT:
            rows.append(
                {
                    "name": name,
                    "checkpoint_sha256": "0" * 64,
                    "receipt_checkpoint_sha256": "0" * 64,
                    "receipt_path": "synthetic://episodic-store-stub",
                    "status": "synthetic",
                    "kind": "nonparametric_store",
                    "parametric": False,
                }
            )
            continue
        sha = _state_sha256(white_matter.faculties[name])  # type: ignore[arg-type]
        rows.append(
            {
                "name": name,
                "checkpoint_sha256": sha,
                "receipt_checkpoint_sha256": sha,
                "receipt_path": f"synthetic://{name}",
                "status": "synthetic",
                "kind": "contrastive_encoder",
                "parametric": True,
            }
        )
    return rows


def _build_synthetic(*, k: int, write_back: bool, seed: int) -> WhiteMatter:
    """Build a `WhiteMatter` over the synthetic stand-in faculties and the store stub."""
    torch.manual_seed(seed)
    faculties = {"language": SyntheticText(), "visual": SyntheticVisual()}
    store = InMemoryStoreStub(domain_enum={"general"}, half_life_s=3600.0, importance_default=0.5)
    return WhiteMatter(toy_config(k=k, write_back=write_back), faculties, store)  # type: ignore[arg-type]


def run_phase_a(args: argparse.Namespace) -> int:
    """Run the `phase-a` subcommand and write its receipt.

    Args:
        args: Parsed arguments.

    Returns:
        `0` when every evaluated gate passed, `1` otherwise -- a failing gate is a
        non-zero exit so a run cannot be scripted past without noticing, matching
        `cogsyndelta.regions.compress.main`.

    Raises:
        ValueError: `--threads` is below 1 (`phase_a.pin_threads`).
    """
    # FIRST, before `_build_synthetic` and therefore before any tensor exists. Weight
    # initialisation is a tensor operation, so pinning after construction would leave the
    # run's starting point decided by the ambient environment even though every step after
    # it was pinned. See `phase_a.DEFAULT_PHASE_A_THREADS` for the measurement.
    pin = pin_threads(args.threads)
    white_matter = _build_synthetic(k=args.k, write_back=not args.no_write_back, seed=args.seed)
    trainer = PhaseATrainer(
        white_matter,
        PhaseAConfig(
            delta=args.delta,
            task_weight=args.task_weight,
            lr=args.lr,
            steps=args.steps,
            seed=args.seed,
            # The same value `pin_threads` above already applied, so the trainer's own
            # pin is a no-op re-application rather than a second, different opinion.
            threads=args.threads,
        ),
        synthetic_frozen_set(white_matter),
        device=args.device,
    )
    train_batches = synthetic_batches(
        count=args.train_batches,
        batch_size=args.batch_size,
        k=args.k,
        workspace_dim=white_matter.config.workspace_dim,
        task_seed=args.seed,
        item_seed=args.seed,
    )
    dev_batches = synthetic_batches(
        count=args.dev_batches,
        batch_size=args.batch_size,
        k=args.k,
        workspace_dim=white_matter.config.workspace_dim,
        task_seed=args.seed,
        item_seed=args.seed + 1_000,
    )
    primed = prime_store(white_matter, records=args.prime_store, seed=args.seed)
    result = trainer.run(train_batches, dev_batches)

    identity = {
        "corpus_fingerprint": hashlib.sha256(
            f"synthetic:{args.seed}:{args.k}:{args.batch_size}".encode()
        ).hexdigest(),
        "fingerprint_scheme": "sha256-of-synthetic-generator-arguments",
        # Spec section 6 Q5 option (c): a plumbing-only run is admissible ONLY under this
        # battery id, so nothing downstream can cite its numbers.
        "battery_id": "plumbing",
        "k": args.k,
        "pooling": "frontal",
        "checkpoint_sha256": _state_sha256(white_matter),
        "region": "white_matter",
        "seed": args.seed,
        "split_sha256": hashlib.sha256(
            f"synthetic-split:{args.train_batches}:{args.dev_batches}:{args.seed}".encode()
        ).hexdigest(),
    }
    path = trainer.write_receipt(result, identity, Path(args.out_dir))
    print(
        json.dumps(
            {
                "receipt": str(path),
                # Printed, not merely written, so an operator watching a run sees the pin
                # its weights depend on without opening the receipt. `pin` is what
                # `pin_threads` read back, not what was asked for.
                "threads": pin.as_receipt_knob(),
                "store_records_primed": primed,
                "steps": len(result.steps),
                "loss_first": result.loss_curve[0],
                "loss_last": result.loss_curve[-1],
                "trainable_parameters": trainer.trainable_parameter_count(),
                "frozen_parameters": trainer.frozen_parameter_count(),
                "mean_attention_per_region": result.mean_per_region,
                "collapse_floor": result.guards.collapse_floor,
                "collapsed_in_phase_A": sorted(result.guards.collapsed_in_phase_A),
                "train_recall_at_1": result.train_metric,
                "dev_recall_at_1": result.dev_metric,
                "overfit_gap_points": result.guards.overfit_gap / 0.01,
                "loss_site": result.loss_site,
                "verdict": result.guards.verdict(),
            },
            indent=2,
        )
    )
    return 0 if result.guards.passed else 1


def run_params(args: argparse.Namespace) -> int:
    """Report the measured phase-A partition at the v1 dimensions against Tables 4 and 6.

    Args:
        args: Parsed arguments; `--r` selects the `R = 4` or `R = 5` configuration.

    Returns:
        `0` when the measured numbers match the table at `R = 5`, `1` otherwise. At
        `R = 4` there is no table figure for the trainable set, so the exit is always `0`
        and the numbers are reported for the record.
    """
    participants = dict(V1_PARTICIPANTS)
    store = None
    if args.r == 5:
        participants[STORE_PARTICIPANT] = V1_STORE_SPEC
        store = InMemoryStoreStub(
            domain_enum={"general"}, half_life_s=3600.0, importance_default=0.5
        )
    faculties = {
        name: ConstructionOnlyFaculty(name, token_dim, kv_bytes)
        for name, (token_dim, kv_bytes) in V1_REGION_WIDTHS.items()
    }
    white_matter = WhiteMatter(
        InterconnectConfig(participants=participants),
        faculties,  # type: ignore[arg-type]
        store,
    )
    trainable, frozen = phase_a_parameter_partition(white_matter)
    measured_total = sum(p.numel() for p in white_matter.parameters())
    measured_trainable = sum(p.numel() for p in trainable.values())
    measured_frozen = sum(p.numel() for p in frozen.values())
    matched = args.r != 5 or (
        measured_total == TABLE_4_TOTAL_R5 and measured_trainable == PHASE_A_TRAINABLE_R5
    )
    print(
        json.dumps(
            {
                "R": args.r,
                "module_total": measured_total,
                "table_4_total_r5": TABLE_4_TOTAL_R5 if args.r == 5 else None,
                "phase_a_trainable": measured_trainable,
                "table_6_trainable_r5": PHASE_A_TRAINABLE_R5 if args.r == 5 else None,
                "phase_a_frozen": measured_frozen,
                "frozen_names": sorted({name.split(".")[0] for name in frozen}),
                "matches_tables": matched,
            },
            indent=2,
        )
    )
    return 0 if matched else 1


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for both subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m cogsyndelta.interconnect.cli",
        description="The interconnect's phase-A trainer (spec section 4, Table 6 row A).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    phase_a = sub.add_parser("phase-a", help="Run phase A on the synthetic stream.")
    phase_a.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    phase_a.add_argument(
        "--stream",
        choices=("synthetic",),
        default="synthetic",
        help="Only 'synthetic' today; a real corpus is the compose-stage driver's job.",
    )
    phase_a.add_argument("--steps", type=int, default=20)
    phase_a.add_argument("--batch-size", type=int, default=8)
    phase_a.add_argument("--train-batches", type=int, default=4)
    # The default has to be large enough for G35 to MEAN what it says. `recall@1` over
    # `dev_batches * batch_size` items moves in steps of `1 / N`, so one misranked
    # held-out item is a `100 / N`-point gap. At the original default of 2 batches
    # (`N = 16`) that single item was a 6.25-point gap -- already over G35's own
    # 5.00-point ceiling -- so the gate could not express the tolerance it is defined
    # with: with a perfect train metric it passed only on an EXACTLY equal dev metric.
    # Which side of that a run landed on was then decided by float reduction order
    # (measured: identical seed and command line, `OMP_NUM_THREADS` 1/2/6/8 -> dev 1.0000,
    # 3/4 -> dev 0.9375), not by anything the run did. 8 batches is `N = 64`, a
    # 1.5625-point resolution, 3.2x finer than the ceiling; the same measurement puts the
    # toy's real gap at 0.00-1.56 points across every thread count.
    phase_a.add_argument(
        "--dev-batches",
        type=int,
        default=8,
        help=(
            "Held-out batches. Keep dev-batches * batch-size above 20 items or G35's "
            "5.00-point ceiling is finer than the metric's own resolution."
        ),
    )
    phase_a.add_argument("--k", type=int, default=4)
    phase_a.add_argument("--lr", type=float, default=3e-3)
    phase_a.add_argument("--delta", type=float, default=1.0)
    phase_a.add_argument("--task-weight", type=float, default=1.0)
    phase_a.add_argument("--seed", type=int, default=0)
    # Not optional in effect, only in spelling: omitting it selects
    # DEFAULT_PHASE_A_THREADS and the receipt records `source: "default"`, so there is no
    # argv that produces an UNPINNED run. The alternative -- refusing to start without
    # --threads -- was rejected because it makes every existing invocation fail closed
    # while buying nothing a recorded default does not already buy: what has to be
    # impossible is an unpinned receipt that looks citable, not an unpinned command line.
    phase_a.add_argument(
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Intra-op thread pin, applied before the module is built. Omit for "
            f"{DEFAULT_PHASE_A_THREADS}, which the receipt records as source=default. "
            "The trained weights depend on this number: at one seed and one command "
            "line, an unpinned sweep of OMP_NUM_THREADS over 1..28 produced eleven "
            "distinct checkpoint_sha256 and flipped a gate verdict."
        ),
    )
    phase_a.add_argument("--out-dir", default="receipts")
    phase_a.add_argument(
        "--prime-store",
        type=int,
        default=8,
        help="Episodes written to the store before step 0; see prime_store.",
    )
    phase_a.add_argument(
        "--no-write-back",
        action="store_true",
        help="Build without the conditioning prefixes; Table 6 then freezes them.",
    )

    params = sub.add_parser("params", help="Report the phase-A partition at v1 dimensions.")
    params.add_argument("--r", type=int, choices=(4, 5), default=5)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse `argv` and dispatch to the requested subcommand.

    Args:
        argv: Argument vector, or `None` to read `sys.argv`.

    Returns:
        The subcommand's exit code.
    """
    args = _build_parser().parse_args(argv)
    if args.command == "phase-a":
        return run_phase_a(args)
    return run_params(args)


if __name__ == "__main__":
    raise SystemExit(main())
