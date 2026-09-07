# `openai/gsm8k:main` `test` split — corpus intake, 2026-09-06

**Summary.** The 1,319-row `test` split of `openai/gsm8k` (config `main`) is landed as a
**holdout**, not as training material. It is licence-verified at the primary source,
fingerprinted, reserved by item id and by shard path, and it does not change any region's
realised training set. It takes the pre-registered reasoning battery from 299 eligible
items to **1,236**.

Numbers below are reproduced by `analyse.py` in this directory; its output is
`intake.json`. Corpus intake only — no training was run.

---

## 1. Licence, verified at the primary source

The catalogue already recorded this source as `PERMISSIVE_OK` under MIT. That claim was
re-read upstream rather than inherited, because this project has a recorded finding that
mirror tags disagree with upstream terms in ten cases.

| what | where | what it actually says |
|---|---|---|
| primary source | `https://github.com/openai/grade-school-math/blob/master/LICENSE` | MIT License, "Copyright (c) 2021 OpenAI" |
| Hub repo card | `https://huggingface.co/datasets/openai/gsm8k` | `license: [mit]`; "The GSM8K dataset is licensed under the MIT License" |

**Verdict: mirror and source AGREE, and the `test` split is under the same MIT terms as
the `train` split already in the corpus.** The Hub repo is in OpenAI's own namespace — it
is the publisher, not a re-host — which is why there is no third-party tag to disagree
with. Recorded on the catalogue entry's `upstream` field
(`scripts/csd-corpus-expand.py`), which is where the contract's mirror→upstream
discipline is kept, and again in the holdout manifest.

## 2. What landed

| field | value |
|---|---|
| rows | 1,319 (matches the card's `num_examples` for `main/test`) |
| path | `/mnt/bulk/csd-corpus/reason/gsm8k-main/test.parquet` |
| bytes | 409,116 |
| shard sha256 | `2920a9cc0fd639600f38dc86276c8a36554c161fd34ac7dd981c0ca0c56973eb` |
| membership sha256 (1,319 item ids) | `8aeb05279e083e2926958f644d7b72fd09dfd81bf45a2726cfeb5e6ebb3fa593` |
| manifest | `config/mind/splits/reason-gsm8k-test-holdout.json` |

**The reason corpus fingerprint did not move: `ca364a92d2c6c5fd259404e0ab6f52a1`,
unchanged.** `fingerprint_corpus` hashes the shard basenames and byte sizes of the sources
a region *declares*, and `test.parquet` is not one of them. `train.parquet` was verified
byte-identical after the re-fetch (sha256 `1467aefa…`), so every existing seed-0 reason
cell stays comparable and G26 still passes against the committed
`reason-ca364a92-split0.json`.

## 3. Item-level disjointness — measured, not assumed

Contract §2.2: item-level disjointness is mandatory everywhere.

| check | result |
|---|---|
| test rows with a duplicate item id inside the split | 0 (1,319 unique) |
| test item ids also present in `gsm8k` `train` | **0** |
| test question text also present in `gsm8k` `train` (normalised) | **0** |
| test rows covered by the reserved-holdout manifest | 1,319 / 1,319 |

The question-text check is separate on purpose: a shared question with a different answer
would still leak the problem, and the pair-id check alone would not see it.

## 4. It is enforced, not intended

A holdout that is merely intended to be held out is not a holdout. Two guards, and each
one's failure is constructed in `tests/test_gsm8k_test_holdout.py`:

- **G38, by item id.** `cogsyndelta.splits.assert_no_reserved_holdout_in_pairs`, called
  from `build_splits` on the realised training pairs — so it covers every region and every
  composite phase, whatever shards the config happens to name. Fails closed twice: a
  missing pinned manifest is an error rather than an empty id set, and a manifest whose
  ids do not hash to its recorded sha256 is refused.
- **By shard path.** `csd-train-all.py`'s `HELD_OUT_SHARDS`, checked at shard-resolution
  time before torch is imported, so a glob widened from `train.parquet` to `*.parquet`
  refuses to start. `RESERVED_FOR_COMPOSE` could not express this: it is keyed by
  directory, and `reason/gsm8k-main/` holds both splits.

**Mutation evidence.** Deleting the `build_splits` call site turns
`test_build_splits_refuses_a_corpus_containing_a_real_gsm8k_test_row` red (`DID NOT
RAISE`), and `test_neutering_the_call_site_lets_the_leak_through` shows the identical
corpus builds happily with the guard stubbed out — so the raise is the guard's, not some
unrelated failure in the path.

## 5. Balance accounting, before and after

Contract Part 3 checks operate on the **post-dedup, post-cap realised training set**.
The `test` split contributes **zero** training rows, so the realised distribution is
unchanged; the staged pool figures move only because the pool got bigger.

| population | max share | `N_eff` (1/Σp²) | `exp(H)` | B1 ≤ 0.40 | B2 ≥ 3 |
|---|---|---|---|---|---|
| staged pool, before (104,940 rows) | 0.928788 | 1.1524 | 1.2927 | fail | fail |
| staged pool, after, counting the holdout (106,259) | **0.917259** | **1.1790** | 1.3303 | fail | fail |
| staged pool, after, training material only (104,940) | 0.928788 | 1.1524 | 1.2927 | fail | fail |
| **realised train split, before (12,455)** | **0.600000** | **1.9231** | 1.9601 | fail | fail |
| **realised train split, after (12,455)** | **0.600000** | **1.9231** | 1.9601 | fail | fail |

**No bound is breached by this change.** B1 and B2 were already failing for `reason`
before it — that failure is stated in CORPUS-CONTRACT.md §1.6 ("`reason` fails the balance
rule before it is trained, and by more than any region that exists") and carried as a
dated B2 waiver in `config/mind/csd-regions.json`. Every figure this intake moves, it
moves in the improving direction: the staged pool's max share falls 92.88% → 91.73% and
`N_eff` rises 1.152 → 1.179. Nothing that previously passed now fails, so there is no
stop condition here. The real remedy is unchanged and unaddressed: `reason` has two
reasoning shapes where it needs at least four, and the two missing ones (deduction,
multi-hop factual) still have no licence-clean supplier.

**B3, train/eval correspondence, stated plainly.** This holdout is **`in-mixture` at the
source level**, not `held-out-domain`: `gsm8k` contributes rows to `reason`'s training set,
so a battery scored on `gsm8k` `test` measures in-distribution competence on unseen items,
not transfer across distributions. What it does buy is a clean item-level guarantee at
4.1× the population. Claiming distribution-level separation here would be false, and §2.2
reserves that requirement for the composed model's eval set.

## 6. What this does for the battery

| | items |
|---|---|
| eligible today, from the in-mixture 512-pair holdout | 299 (of 320 gsm8k rows) |
| eligible from the landed `test` split | **1,236** (of 1,319 rows, mean 3.41 annotations) |

Eligibility is `build_corrupted_battery`'s own rule — a row needs at least
`MIN_ANNOTATIONS` = 2 `<<expr=result>>` calculator spans to be corruptible — applied
unchanged. At K = 4 corruptions, chance stays 0.20.

Rewiring `scripts/csd-eval-reason-e1.py` onto this population is **not** part of this
intake: that script pins `EXPECTED_ELIGIBLE = 299` against the E0 holdout and its
pre-registration was written for that population. Changing the battery's population is a
pre-registration decision, and this change only makes the population available.

## 7. What the contract asked for that is not here

- **Stage 2 (semantic near-duplicate screen) was not run.** §2.3 requires a non-circular
  encoder — one trained on neither side — and both sides here are `gsm8k`, so every
  available encoder is circular by construction and its result could only be advisory.
  The exact-overlap screen (stage 1) is reported in §3 above and is zero on both the pair
  id and the question text.
- **Stage 3 (re-run against the realised train split) is satisfied trivially** and is
  recorded as such: the realised `reason` train split is unchanged, and the exact-overlap
  screen above is against `gsm8k` `train` in full, which is a superset of it.
- **No allocation-ledger entry was written.** §2.4's ledger governs sources allocated to
  the composed model; this is a region-source holdout reserved against *all* regions,
  which the ledger has no row shape for. The reserved-holdout manifest is the record.
