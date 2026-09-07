# W3 — the episode harness

**What is here.** `receipt-eval-1024.json` is the reference run of row W3's episode harness over
the whole X7 eval shard: 1,024 episodes, three arms each, 3,072 episode runs. The harness itself
is `src/cogsyndelta/eval/episode_harness.py`; its gates and their can-fail twins are
`tests/interconnect/test_episode_harness.py`; the runner is `scripts/csd-run-x7-harness.py`.

**Why it exists.** Every other reserve shape is one forward pass. X7 is a *sequence* — turn 1
runs, its write commits, turn 2 is scored against the resulting partition — and nothing in this
repo could run that. Revision 2's objection to the store was that *"a single-item eval forward
pass has nothing to read"*; §5.5(e) records that the objection is correct and *"is not answered by
declaring a shape; it is answered by building the harness."*

## Headline

| | |
|---|---|
| episodes | 1,024 (the whole `x7-eval.jsonl` shard) |
| arms per episode | 3 — `full`, `wrong_turn1`, `no_turn1` |
| episode runs | 3,072 |
| gate (i) negative control | **fires** — 1,024 / 1,024 failed, all on the declared wrong option |
| gate (ii) DEC-38 disjointness | **fires** — asserted per item at run time; 0 violations |
| gate (iii) partition reset | **fires** — 1,024 / 1,024 turn-1-less runs still failed |
| turn 2 read turn 1's record | **2,048 / 2,048** episodes that ran a turn 1 |
| store load-bearing | `true` |
| model (`RankHead`) accuracy | 0 / 3,072 — it chose `NULL` every time, **and it is not evidence**; see *The model scorer* below |

## The thing that keeps every other number honest

A harness that reports green while the store contributes nothing is worse than no harness:
`mean(a_store)` passed G29 on a store whose mutual information with the target was exactly zero.
So the store read is **observed, not inferred**. `StoreProbe` wraps the store the forward pass
holds and records every call it makes; the harness reports, per episode, whether turn 2's step-8
read returned the latent turn 1 committed.

| observation | value |
|---|---|
| episodes that committed a turn 1 | 2,048 |
| whose turn-2 read returned that record | 2,048 |
| at rank 0 (first slot the store returned) | 2,048 |
| cosine to the committed latent | 1.0 on every one |
| `no_turn1` runs that read any record | 0 |

Rank 0 on every episode is PR #86's cosine-ranked read doing its job: the query is the item's own
store-free pre-pass `z_N`, the partition holds exactly the record turn 1 wrote, and the read puts
it first. Before PR #86 the read was a constant and this column could not have distinguished
anything.

## Ordering: the commit really precedes the scoring

The W3 cell's headline clause is that turn 1's write commits *before* turn 2 is scored. Three
things make that a property of the run rather than a claim about it.

1. Turn 1 returns a `WriteReceipt`, and the harness refuses the episode if it does not. On
   `EpisodicStoreImpl` that receipt comes from `learn`, which commits through `StoreBackend.put`
   before returning — on `SqliteBackend` (WAL, `synchronous=FULL`) the record is durable at that
   point, not queued.
2. The harness then **reads the partition itself** and requires the record to be resident. The
   receipt alone proves nothing: `WriteReceipt.committed` is `Literal[True]` by construction, so a
   fire-and-forget store returns an identical receipt.
3. `test_an_uncommitted_turn1_leaves_turn2_with_nothing_to_read` swaps in exactly that store. Every
   predicted symptom appears together — empty partition, empty read, abstaining verdict,
   `store_is_load_bearing: false` — which is the design's *"a turn 2 scored against an uncommitted
   write is measuring nothing"* made executable.

## The three gates, each with a red twin

The W3 cell asks for gates *"all constructed to fire"*, so each green assertion has a red one
beside it in `tests/interconnect/test_episode_harness.py`.

| gate | green | red (the reconstructed defect) |
|---|---|---|
| (i) negative control | `test_negative_control_fails_the_item` | `test_negative_control_cannot_fire_when_the_donor_fact_collides` — a donor committing the *same* fact makes the control select the *same* option, and the arm passes |
| (ii) DEC-38 | `test_dec38_holds_on_a_well_formed_episode` | `test_dec38_fires_when_the_turns_share_a_source_row`, plus a donor row reused as the commit row |
| (iii) partition reset | `test_turn1_less_variant_still_fails_after_a_full_run` | `test_the_turn1_less_variant_starts_passing_when_the_reset_is_off` — `reset_partitions=False` pins every episode to one session and the turn-1-less variant scores correct |

Gate (i) is checked in its **strong** form. "Did not pass" is satisfied by an abstention, and an
abstention is also what a dead store produces, so the weak form cannot tell a working control from
a broken one. The item's substituted fact selects an option that is *present and wrong*, so the
verdict is checkable to the index: all 1,024 control runs landed on
`counterfactual.expected_option_index`.

Gate (iii) is checked twice over, because either half alone can hold by accident. The turn-1-less
arm must fail (the design's own test), **and** every one of the 3,072 runs must have started
against an empty partition (the mechanism that makes it fail).

## The recall oracle, and why it is not circular

Turn 2's verdict comes from the record turn 2's read actually returned. The harness matches each
unmasked read slot against the latents it watched turn 1 commit, looks up the fact that turn 1
committed, and renders `fact + probe_answer` into one of the item's five options. It abstains when
no returned slot is a record it watched being written.

This is a **harness instrument, not a model**. Its accuracy is exactly the fidelity of the store
path and is zero when the store is not read — which is what makes all three gates fire
deterministically today, before any region is retrained. It is reported as `oracle_*`, never as a
capability number, and `store_is_load_bearing` sits beside it in the receipt so the two cannot be
read apart.

## The model scorer, and a finding about `NULL`

`rank_head_choice` is `RankHead`'s argmax over the item's options — the seam W5 scores through.
It is wired, and on untrained weights it is not evidence.

**It chose `NULL` on 3,072 of 3,072 runs.** That is not chance — chance would be roughly one in
six — and the mechanism is worth recording because it will outlive this run. `NullCandidate` is
initialised at `randn(D_w) * 0.02`, while `RankHead.proj` is a `Linear` **with a bias**, so
`proj(NULL)` is dominated by that bias; and `FrontalReadout`'s output projects onto the bias far
more strongly than the candidate embeddings do. Every figure below is measured on the very mind
that produced `receipt-eval-1024.json`, not on an isolated probe.

| quantity | value |
|---|---|
| `‖NULL‖` at init | 0.164 |
| `‖proj.bias‖` | 0.552 |
| `cos(proj(NULL), proj.bias)` | **0.987** |
| mean `cos(proj(f), proj.bias)` over 64 read-outs | **0.792** |
| mean `cos(proj(candidate), proj.bias)` | 0.065 |

The query and `NULL` are both bias-aligned; the content candidates are not. `NULL` therefore wins
by default, on every item, regardless of the item.

The consequence for **G36**, the `NULL` recall / false-positive gate: an untrained baseline scores
100% `NULL` recall and a 100% false-positive rate, and reading the recall half alone would look
like a pass. G36's baseline has to be read as a pair.

## What this run's numbers are, and are not

- **Real**: the store path, the commit ordering, the partition reset, the three gates, and the
  per-episode read provenance. None of them depend on a trained weight.
- **Not real**: any capability claim. The faculties are genuine
  `TextFacultyAdapter(TextEncoder(...))` modules at **untrained** weights behind a hashing stand-in
  tokeniser, because `language` and `memory` have no token-aware retrain yet (W1's result) and X7
  is **not admitted** — NSRS `s_r` is GPU-blocked and blocked on W2b, and every item still carries
  `admission.status = "pending"`.

## Where the W3 cell was ambiguous, said rather than chosen silently

1. **`LearnReceipt` has no symbol in this repo.** The W3 cell and §5.5(e) both name one. `learn`
   is lifecycle verb 2/6 in `interconnect/episodic/store.py` and returns `WriteReceipt`, whose own
   docstring records that it *is* memory-gate's `LearnReceipt` ported under the E0 protocol's name.
   Read as the same object; no symbol was invented.
2. **"the partition is reset between episodes" does not say per ARM.** Two arms of one item sharing
   a partition would let the `full` arm's record satisfy the `no_turn1` arm — the exact leak gate
   (iii) exists to catch. The session key is therefore per `(item, arm, run ordinal)`: finer than
   the text requires, in the direction that cannot hide a leak.
3. **The negative control's turn-1 text is not in the item.** `counterfactual.substituted_turn1`
   carries `fact_value`, `key_claim` and `pair_fingerprint`, not a rendered query. The control is
   defined by the fact that gets committed, so the harness stages the donor's `key_claim` as turn
   1's text and commits `fact_value`. Recovering the donor's full query would mean re-reading the
   pinned corpus shard at eval time, which no other scorer here does.
4. **Batch size is one episode per forward pass.** `mind.py` takes per-item `scopes` and
   `logical_keys`, so batching is legal and is W5's optimisation to make. At `B = 1` each turn's
   step-8 read is unambiguously one call, so read provenance is an observation rather than a
   reconstruction — and the harness refuses rather than reporting on a read it cannot identify.

## An interop gap found on the way, and left for its own lane

`TextFacultyAdapter` and `WhiteMatter` **do not currently compose on a mask-less input.**
`TextEncoder.tokens` returns `torch.ones(b, t, dtype=h.dtype)` — a **float** mask — when the caller
passes no attention mask, and `WhiteMatter._iterate` combines it with `mask & budget_mask`, which
raises `NotImplementedError: "bitwise_and_cpu" not implemented for 'Float'`. `Faculty.tokens`
documents the mask as *"1 for a real position, 0 for padding"* and fixes no dtype, so neither side
is obviously wrong.

The other way out does not work either: passing `(ids, bool_mask)` as a tuple is accepted by
`TextFacultyAdapter._split_inputs`, but `WhiteMatter._batch_size` infers `B` from the first
**tensor** in `inputs`, and turn 1 carries no `candidates`, so an all-tuple request has no tensor
to infer from.

Both sit on a merged forward path and deserve their own lane with their own can-fail test, so this
row did not touch them. `scripts/csd-run-x7-harness.py` carries a one-method `BoolMaskTextFaculty`
shim that casts the mask, which keeps the run against the **real** encoder and the **real** adapter
rather than substituting a toy faculty, and keeps the gap visible in one named place.

## Reproducing

```bash
export OMP_NUM_THREADS=1
export TMPDIR=/akula-data/csd/tmp
python scripts/csd-run-x7-harness.py \
    --items /akula-data/csd/reserve/x7/x7-eval.jsonl --limit 1024 \
    --threads 1 --episode-rows 64 \
    --out docs/design/evidence/w3-episode-harness-2026-09-07/receipt-eval-1024.json
```

The run is CPU-only and takes about 43 s. `--episode-rows 64` bounds the receipt's **row listing**
only; every summary and both gates are computed over all 3,072 runs, and `episode_rows` in the
receipt records what was cut. Threads are pinned and recorded because CPU reductions are not
associative and a verdict must be reproducible from what the receipt says. The script exits
non-zero if either gate fails to fire or the store is not load-bearing — a run that measured
nothing is not a passing run with a caveat.
