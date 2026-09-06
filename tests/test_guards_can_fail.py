"""Every test here constructs the condition a guard exists to catch, and asserts it IS caught.

WHY THIS FILE EXISTS
Four guards in this project were correct in reasoning and wrong in scope. Reading them
confirmed intent rather than behaviour, and every one of them reported success it was
structurally incapable of distinguishing from failure:

  - `build_splits` deduplicated anchors on `blake2b(normalise(anchor))` and then asked
    `assert_no_contamination` whether any anchor overlapped, using the IDENTICAL
    normalisation and hash over the same field. Unique keys, partitioned by the split,
    cannot intersect. `overlap: 0` was arithmetic, not evidence -- while an independent
    corpus survey measured 53.7% near-duplicate holdout leakage in `retrieve`.
  - `_fingerprint_corpus` hashed the primary source only, so `retrieve`'s receipt
    fingerprint covered ~3% of its data while `csd-quantize.py` hard-failed on it.
  - `load_pairs` returned the moment `limit` was reached, so every "balance cap" in the
    system meant "take a prefix in file order" and the later shuffle could not repair it.
  - `_decode_split` keyed its cache on `abs(hash(key))`, which PYTHONHASHSEED salts per
    process, so the decoded-image cache never hit and every VL run re-decoded 100k JPEGs.
  - `compress`'s graded/STS-B gate ran once by hand (receipts/compress-
    20260902T153612Z.json, `graded_held_out.spearman` 0.4956) and then silently vanished
    from every receipt after `scripts/csd-train-all.py` became the runner: its
    `PretrainConfig(...)` call never set `graded_shards`, `pretrain_region` correctly
    read the empty default as "no graded set declared", and nothing distinguished that
    from "this region declared one and it went missing". No exception anywhere -- a
    receipt just quietly stopped carrying a key. A first repair of this (keying
    `_assert_graded_gate_present` on `cfg.graded_shards`) was itself unreachable by the
    actual regression: the reachable path is `run_region` resolving a declared
    `GradedSpec` to an EMPTY shard list, which used to print a warning and build
    `PretrainConfig(graded_shards=[], graded_name="stsb-validation")` anyway -- an
    inconsistent config, accepted, that reproduces the exact silent-drop shape the guard
    exists to catch. `_assert_graded_gate_present` now keys on `graded_shards OR
    graded_name`, and `run_region` now raises `GradedSourceMissingError` the moment a
    declared graded glob resolves nothing, matching `regions/compress.py`'s
    `compress_config` (`FileNotFoundError`, "rather than training on nothing") instead of
    being weaker than the path it replaced.
  - `regions/pretrain.py`'s checkpoint `_config_fingerprint` hashed `PretrainConfig`'s
    own fields only -- not the corpus CONTENT at `cfg.shards`, and not the
    split-building code (`build_splits`, `screen_pair_contamination`) that decides what
    a corpus turns into. Commit 883c9d4 changed `build_splits` (an unconditional
    shuffle it had never done before) without touching a single config field, so the
    pre- and post-fix runs fingerprinted identically:
    `/akula-data/csd/receipts/code-checkpoints/` mixes a 16:21 pre-fix
    `step-007998.pt` with 17:07-17:08 post-fix files, and nothing on disk distinguishes
    them. A checkpoint with no fingerprint at all (written before fingerprinting
    existed) was also silently adopted as a valid fresh-start point rather than refused.

A happy-path test passes against all six. So each test below builds the exact failing
input and asserts the guard fires, and several assert the PRE-FIX guard would not have --
that pairing is the point of the file. Add to it whenever a guard is added: a guard with
no failing-case test is a comment with a function signature.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import pytest

from cogsyndelta.eval import (
    assert_no_contamination,
    assert_no_pair_contamination,
    pair_contamination_report,
    screen_pair_contamination,
)
from cogsyndelta.eval.metrics import MetricGroup, MetricIdentity, compare

pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------------------
# DEFECT 1 -- the contamination guard could not fire.
# ---------------------------------------------------------------------------------------


def _clean_pairs(n: int, tag: str = "clean") -> list[tuple[str, str]]:
    """Pairs with nothing in common beyond structure, for the negative controls."""
    return [
        (f"{tag} question number {i} about topic {i}", f"{tag} answer body {i}") for i in range(n)
    ]


def test_swapped_pair_is_invisible_to_the_anchor_only_guard_and_caught_by_the_new_one() -> None:
    """(a, b) in train and (b, a) held out is one leak, and symmetric InfoNCE trains both.

    The anchor-only guard compares train anchors against eval anchors: here they are
    `a` and `b`, which differ, so it sees nothing. This is the shape that proves the old
    guard's blindness was about SCOPE, not about hashing.
    """
    train = [*_clean_pairs(20), ("the mitochondrion is the powerhouse", "a cell biology fact")]
    held_out = [
        *_clean_pairs(5, tag="eval"),
        ("a cell biology fact", "the mitochondrion is the powerhouse"),
    ]

    # The pre-fix guard, on the field it was given: silent.
    assert_no_contamination([a for a, _ in train], [a for a, _ in held_out])

    with pytest.raises(ValueError, match="pair_exact"):
        assert_no_pair_contamination(train, held_out)


def test_function_word_paraphrase_of_a_whole_pair_is_caught() -> None:
    """ "how do you know if X" vs "how to know if X" -- the survey's genuine-paraphrase case.

    Exact normalisation calls these different texts, which is why anchor dedup keeps both
    and the old guard saw a clean split. The content-word key calls them the same claim.
    """
    train = [
        *_clean_pairs(20),
        ("how do you know if a mango is ripe", "press it gently near the stem"),
    ]
    held_out = [
        *_clean_pairs(5, tag="eval"),
        ("how to know if a mango is ripe", "you press it gently near the stem"),
    ]

    assert_no_contamination([a for a, _ in train], [a for a, _ in held_out])

    with pytest.raises(ValueError, match="pair_content"):
        assert_no_pair_contamination(train, held_out)


def test_a_slot_value_template_collision_is_deliberately_not_flagged() -> None:
    """GooAQ's "44 is 25 percent of what number?" vs "...55 percent...".

    Same template, DIFFERENT correct answer. Deduplicating or gating on it would hide the
    encoder's failure to read a slot value rather than fix a leak, so the guard must stay
    quiet here. A guard that fires on everything is as useless as one that fires on
    nothing -- this is the test that keeps the new one honest in the other direction.
    """
    train = [*_clean_pairs(20), ("44 is 25 percent of what number", "176")]
    held_out = [*_clean_pairs(5, tag="eval"), ("44 is 55 percent of what number", "80")]

    report = assert_no_pair_contamination(train, held_out)
    assert report["channels"]["pair_content"]["overlap"] == 0
    assert report["channels"]["anchor_content"]["overlap"] == 0


def test_a_shared_positive_is_counted_but_does_not_stop_the_run() -> None:
    """Anchor-only dedup never touches the positive side, so this channel CAN fire.

    It is reported rather than gated on purpose: one passage answering two genuinely
    different queries is normal in IR and only leaks when the queries are also close.
    The number belongs in the receipt; the halt does not.
    """
    shared = "the same passage of text appearing on both sides"
    train = [*_clean_pairs(20), ("what does the manual say about torque", shared)]
    held_out = [*_clean_pairs(5, tag="eval"), ("who wrote the recipe for sourdough", shared)]

    report = assert_no_pair_contamination(train, held_out)
    assert report["channels"]["positive_exact"]["overlap"] == 1
    assert report["channels"]["positive_exact"]["gated"] is False


def test_duplicate_positives_inside_the_holdout_are_counted() -> None:
    """Two held-out pairs sharing a positive cap recall@1 by construction.

    `evaluate` ranks each anchor against every held-out positive, so two identical
    candidates cannot both be ranked first. Nothing in the training loop can see it.
    """
    duplicated = "one answer used twice in the holdout"
    held_out = [("first question", duplicated), ("second question", duplicated)]
    report = pair_contamination_report(_clean_pairs(10), held_out)
    assert report["eval_duplicate_positives"] == 1


def test_a_genuinely_clean_split_passes_every_channel() -> None:
    """The negative control. Without this, "the guard fires" could just mean "always"."""
    report = assert_no_pair_contamination(_clean_pairs(50), _clean_pairs(10, tag="eval"))
    assert all(channel["overlap"] == 0 for channel in report["channels"].values())
    assert report["train_pairs_seen"] == 50
    assert report["eval_pairs"] == 10


def _write_pairs(path, anchors: list[str], positives: list[str]) -> None:
    """Write an (a, b) parquet shard. Imported locally so this file needs no train group."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"a": anchors, "b": positives}), path)


def test_a_stray_leak_is_removed_from_training_and_counted_rather_than_tolerated() -> None:
    """The common real case: `compress` leaks 1 held-out pair of 512, `retrieve` 35.

    Refusing outright there would block a region from training over a handful of rows,
    which is the pressure that ends with someone raising a tolerance. So the colliding
    TRAINING rows go (the holdout is never touched) and the receipt keeps the PRE-removal
    figure -- a report generated after cleanup would show a zero indistinguishable from a
    corpus that never leaked, which is the failure this whole file is about.
    """
    leak = ("how do you know if a mango is ripe", "press it gently near the stem")
    train = [*_clean_pairs(200), leak]
    held_out = [
        *_clean_pairs(5, tag="eval"),
        ("how to know if a mango is ripe", "you press it gently near the stem"),
    ]

    kept, report = screen_pair_contamination(train, held_out)

    assert leak not in kept, "the leaked training row is still in the training set"
    assert len(kept) == len(train) - 1, "removal took rows it should not have"
    assert report["train_pairs_removed"] == 1
    # Measured BEFORE the removal: the receipt has to say what leaked.
    assert report["channels"]["pair_content"]["overlap"] == 1


def test_a_corpus_that_is_mostly_duplicates_is_refused_instead_of_cleaned() -> None:
    """Removal is a repair for strays. Deleting most of training until the eval looks
    clean is laundering, and the ceiling is on how much of TRAINING a repair deletes --
    not on how much of the holdout leaked, which would have refused `retrieve` outright
    over 35 rows of 505,216."""
    duplicated = [(f"how do you know if item {i} is ready", f"answer {i}") for i in range(50)]
    held_out = [(f"how to know if item {i} is ready", f"answer {i}") for i in range(50)]

    with pytest.raises(ValueError, match="deleting the corpus"):
        screen_pair_contamination(duplicated, held_out)


def test_build_splits_refuses_a_corpus_whose_holdout_is_paraphrased_in_train(tmp_path) -> None:
    """End to end: the guard must stop `build_splits`, not merely be callable.

    The corpus is 100 twinned rows. Each twin differs from its sibling ONLY in function
    words on the anchor and is identical on the positive, so:
      - anchor dedup keeps both (their exact normalised anchors differ), which is why the
        old anchor-only guard passed this corpus with `overlap: 0`;
      - whichever twin lands in the holdout has its sibling in train.

    Every row here is a duplicate, so repairing it would delete the training set -- far
    past the removal ceiling -- and the run stops.
    """
    pytest.importorskip("pyarrow", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.pretrain import PretrainConfig, build_splits

    anchors, positives = [], []
    for i in range(100):
        answer = f"press the release catch on item {i} and lift"
        anchors.append(f"how do you know if item {i} is ready")
        positives.append(answer)
        anchors.append(f"how to know if item {i} is ready")
        positives.append(answer)
    shard = tmp_path / "twinned.parquet"
    _write_pairs(shard, anchors, positives)

    cfg = PretrainConfig(
        region="twinned-test",
        pair_columns=("a", "b"),
        shards=[str(shard)],
        steps=1,
        batch_size=len(anchors),
        holdout_pairs=20,
        seed=0,
    )
    with pytest.raises(ValueError, match="deleting the corpus"):
        build_splits(cfg)


def test_build_splits_still_accepts_a_clean_corpus(tmp_path) -> None:
    """The same shape without the twins must train. A guard that blocks everything is not
    a fix, and this is the assertion that would catch making one."""
    pytest.importorskip("pyarrow", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.pretrain import PretrainConfig, build_splits

    shard = tmp_path / "clean.parquet"
    _write_pairs(
        shard,
        [f"question {i} regarding subject {i}" for i in range(200)],
        [f"answer {i} describing outcome {i}" for i in range(200)],
    )
    cfg = PretrainConfig(
        region="clean-test",
        pair_columns=("a", "b"),
        shards=[str(shard)],
        steps=1,
        batch_size=200,
        holdout_pairs=20,
        seed=0,
    )
    holdout, train_pairs, meta = build_splits(cfg)
    assert len(holdout) == 20
    assert len(train_pairs) == 180
    assert meta["contamination"]["train_pairs_removed"] == 0
    # The channel the pre-fix guard used is still reported -- and still zero. Kept
    # visible precisely so nobody reads that zero as evidence again.
    assert meta["contamination"]["channels"]["anchor_exact"]["overlap"] == 0


# ---------------------------------------------------------------------------------------
# DEFECT 2 -- the receipt fingerprint covered one source out of three.
# ---------------------------------------------------------------------------------------


def _shard(tmp_path, name: str, size: int):
    """A file of a known size. The fingerprint hashes names and sizes, not content."""
    path = tmp_path / name
    path.write_bytes(b"x" * size)
    return str(path)


def test_fingerprint_changes_when_an_extra_source_is_swapped(tmp_path) -> None:
    """The exact blindness: `retrieve`'s primary (FiQA) is ~3% of what it trains on.

    Under the pre-fix rule these two corpora -- identical primary, COMPLETELY different
    extra sources -- produced the same fingerprint, so `csd-quantize.py`'s hard failure
    could not see gooaq or natural-questions being replaced wholesale.
    """
    from cogsyndelta.corpus import fingerprint_corpus

    primary = [_shard(tmp_path, "fiqa-0.parquet", 100)]
    gooaq = [_shard(tmp_path, "gooaq-0.parquet", 4000)]
    replaced = [_shard(tmp_path, "something-else-0.parquet", 9000)]

    with_gooaq = fingerprint_corpus(
        primary,
        columns=["query", "passage"],
        extra_sources=[{"shards": gooaq, "columns": ["question", "answer"], "limit": 400_000}],
    )
    with_replacement = fingerprint_corpus(
        primary,
        columns=["query", "passage"],
        extra_sources=[{"shards": replaced, "columns": ["question", "answer"], "limit": 400_000}],
    )
    primary_only = fingerprint_corpus(primary, columns=["query", "passage"])

    assert with_gooaq != with_replacement
    assert with_gooaq != primary_only


def test_fingerprint_changes_when_a_cap_or_a_column_changes(tmp_path) -> None:
    """The cap and the columns decide which rows reach `build_splits`, so they are corpus
    identity too -- a re-run under a different cap is not the split the receipt describes."""
    from cogsyndelta.corpus import fingerprint_corpus

    primary = [_shard(tmp_path, "primary-0.parquet", 100)]
    extra = [_shard(tmp_path, "extra-0.parquet", 200)]

    def fingerprint(limit: int, columns: list[str]) -> str:
        return fingerprint_corpus(
            primary,
            columns=["a", "b"],
            extra_sources=[{"shards": extra, "columns": columns, "limit": limit}],
        )

    assert fingerprint(400_000, ["q", "a"]) != fingerprint(200_000, ["q", "a"])
    assert fingerprint(400_000, ["q", "a"]) != fingerprint(400_000, ["question", "answer"])


def test_fingerprint_ignores_shard_order_but_not_shard_content(tmp_path) -> None:
    """Sorted, because shard ORDER is a `build_splits` concern, not an identity one."""
    from cogsyndelta.corpus import fingerprint_corpus

    one = _shard(tmp_path, "a.parquet", 10)
    two = _shard(tmp_path, "b.parquet", 20)
    assert fingerprint_corpus([one, two]) == fingerprint_corpus([two, one])
    assert fingerprint_corpus([one, two]) != fingerprint_corpus([one])


def test_a_scheme_change_reports_itself_rather_than_looking_like_corpus_drift() -> None:
    """Including every source changed every existing fingerprint. A bare mismatch would
    send an operator hunting a corpus change that never happened."""
    from cogsyndelta.corpus import (
        CORPUS_FINGERPRINT_SCHEME,
        verify_corpus_fingerprint,
    )

    v1_receipt = {"fingerprint": "0123456789abcdef"}  # no scheme field: pre-fix receipt
    with pytest.raises(RuntimeError, match="SCHEME changed"):
        verify_corpus_fingerprint(v1_receipt, "fedcba9876543210", "retrieve")

    same_scheme = {
        "fingerprint": "0123456789abcdef",
        "fingerprint_scheme": CORPUS_FINGERPRINT_SCHEME,
    }
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        verify_corpus_fingerprint(same_scheme, "fedcba9876543210", "retrieve")

    # Matching under the current scheme is the only case that proceeds.
    verify_corpus_fingerprint(same_scheme, "0123456789abcdef", "retrieve")
    # A receipt with no fingerprint at all predates the check entirely; nothing to verify.
    verify_corpus_fingerprint({}, "fedcba9876543210", "retrieve")


# ---------------------------------------------------------------------------------------
# DEFECT 3 -- a cap meant "take a prefix", never "take a sample".
# ---------------------------------------------------------------------------------------


def _skewed_stream(head: int, tail: int) -> list[str]:
    """`head` rows of one group followed by `tail` of another -- a corpus's natural order.

    Real shards look like this: gooaq, CodeSearchNet (long contiguous runs of one repo),
    any corpus written per-source or per-shard. A prefix of such a file is not a sample
    of it, and that is precisely the fact the old cap ignored.
    """
    return [f"head-{i}" for i in range(head)] + [f"tail-{i}" for i in range(tail)]


def test_a_cap_samples_the_whole_source_instead_of_taking_a_prefix() -> None:
    """The measured defect, in miniature: gooaq's 400,000 cap was the FIRST 13.3% of the file.

    Taking a prefix here would return 400 `head` rows and zero `tail` rows. A uniform
    sample must land near the population share instead -- 400 of 3,000 rows, ~13% head.
    """
    from cogsyndelta.corpus import reservoir_sample, sampling_rng

    population = _skewed_stream(400, 2600)
    sample = reservoir_sample(
        iter(population), 400, sampling_rng(0, ["gooaq-0.parquet"], ["q", "a"])
    )

    assert len(sample) == 400
    head_share = sum(1 for row in sample if row.startswith("head-")) / len(sample)
    assert head_share < 0.5, f"cap took a prefix: {head_share:.0%} of the sample is the file head"
    assert 0.05 < head_share < 0.25, (
        f"sample is not near the 13.3% population share: {head_share:.0%}"
    )


def test_the_sample_is_reproducible_and_seed_dependent() -> None:
    """Reproducibility is load-bearing: receipts carry corpus fingerprints and two runs of
    the same config have to be comparable. A cap that sampled differently every run would
    make every receipt a one-off."""
    from cogsyndelta.corpus import reservoir_sample, sampling_rng

    population = _skewed_stream(400, 2600)
    shards, columns = ["gooaq-0.parquet"], ["q", "a"]

    first = reservoir_sample(iter(population), 400, sampling_rng(0, shards, columns))
    again = reservoir_sample(iter(population), 400, sampling_rng(0, shards, columns))
    other_seed = reservoir_sample(iter(population), 400, sampling_rng(1, shards, columns))

    assert first == again
    assert first != other_seed


def test_two_sources_do_not_sample_in_lockstep() -> None:
    """Sources get independent generators, so their accept/reject decisions are not
    correlated by row index -- a coupling nobody would think to look for."""
    from cogsyndelta.corpus import reservoir_sample, sampling_rng

    population = _skewed_stream(400, 2600)
    gooaq = reservoir_sample(
        iter(population), 400, sampling_rng(0, ["gooaq-0.parquet"], ["q", "a"])
    )
    nq = reservoir_sample(iter(population), 400, sampling_rng(0, ["nq-0.parquet"], ["q", "a"]))
    assert gooaq != nq


def test_a_cap_larger_than_the_source_keeps_everything_in_order() -> None:
    """The common case must be untouched: no cap binding means no sampling at all."""
    from cogsyndelta.corpus import reservoir_sample, sampling_rng

    population = _skewed_stream(10, 10)
    kept = reservoir_sample(iter(population), 500, sampling_rng(0, ["x.parquet"], ["a", "b"]))
    assert kept == population


def test_load_pairs_cap_spans_the_whole_shard(tmp_path) -> None:
    """End to end through parquet: the cap must not inherit the file's ordering."""
    pytest.importorskip("pyarrow", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.pretrain import load_pairs

    groups = _skewed_stream(200, 1300)
    shard = tmp_path / "skewed.parquet"
    _write_pairs(shard, groups, [f"positive for {row}" for row in groups])

    pairs = load_pairs([str(shard)], ("a", "b"), limit=200, seed=0)
    assert len(pairs) == 200
    heads = sum(1 for anchor, _ in pairs if anchor.startswith("head-"))
    # The pre-fix `return`-on-limit gave 200 heads out of 200. 200 of 1,500 rows are
    # heads, so a sample should hold roughly 27.
    assert heads < 100, f"load_pairs took a prefix: {heads}/200 rows are the file head"
    assert load_pairs([str(shard)], ("a", "b"), limit=200, seed=0) == pairs


# ---------------------------------------------------------------------------------------
# DEFECT 4 -- the decoded-image cache key changed every process, so it never hit.
# ---------------------------------------------------------------------------------------

_CACHE_KEY_PROBE = (
    "import json;"
    "from cogsyndelta.corpus import stable_cache_tag;"
    "parts={'shards': ['tiny-imagenet-0.parquet'], 'size': 64, 'limit': 100000,"
    " 'image_col': 'image', 'label_col': 'label'};"
    "print(stable_cache_tag(parts));"
    "print(abs(hash(json.dumps(parts, sort_keys=True))) % 10**16)"
)


def _probe(hash_seed: str) -> tuple[str, str]:
    """Compute both tags in a fresh interpreter with a chosen PYTHONHASHSEED."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    out = subprocess.run(
        [sys.executable, "-c", _CACHE_KEY_PROBE],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    stable, salted = out.stdout.split()
    return stable, salted


def test_the_cache_tag_is_the_same_in_every_process() -> None:
    """Two interpreters, two hash seeds. The digest agrees; `hash()` does not.

    The second assertion is what makes the first one mean something: it demonstrates the
    salting empirically in this environment, so this is a test of the real failure
    condition rather than of a story about it. `_decode_split` used the salted form, so
    its cache filename differed on every run and 100,000 JPEGs were re-decoded each time.
    """
    stable_one, salted_one = _probe("1")
    stable_two, salted_two = _probe("2")

    assert stable_one == stable_two, "cache tag is not stable across processes"
    assert salted_one != salted_two, (
        "PYTHONHASHSEED is not varying str hashing here, so this test is not exercising "
        "the condition it exists to catch"
    )


def test_the_cache_tag_separates_inputs_that_produce_different_pixels() -> None:
    """Stability is worthless if everything collides. Each field must move the tag --
    `label_col` included, which the pre-fix key omitted entirely."""
    from cogsyndelta.corpus import stable_cache_tag

    base = {
        "shards": ["a.parquet"],
        "size": 64,
        "limit": 100,
        "image_col": "image",
        "label_col": "label",
    }
    tag = stable_cache_tag(base)
    for field, changed in (
        ("shards", ["b.parquet"]),
        ("size", 32),
        ("limit", 200),
        ("image_col", "img"),
        ("label_col", "fine_label"),
    ):
        assert stable_cache_tag({**base, field: changed}) != tag, f"{field} does not change the tag"


# ---------------------------------------------------------------------------------------
# DEFECT 5 -- a declared graded gate could vanish from a receipt with nothing raising.
# ---------------------------------------------------------------------------------------


def test_a_declared_graded_gate_missing_from_the_receipt_raises() -> None:
    """R4: reproduce the receipt-vs-config mismatch directly, without training.

    A config that DOES declare `graded_shards` (the same way `regions/compress.py`'s
    `compress_config` does, and the same way `scripts/csd-train-all.py`'s `compress`
    entry now does), paired with the receipt a run would produce if that gate silently
    dropped out of it. `_assert_graded_gate_present` is the guard that must refuse to let
    this receipt be written.
    """
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.pretrain import PretrainConfig, _assert_graded_gate_present

    cfg = PretrainConfig(
        region="compress",
        pair_columns=("anchor", "positive"),
        shards=["train.parquet"],
        graded_shards=["stsb-validation.parquet"],
        graded_columns=("sentence1", "sentence2", "score"),
        graded_name="stsb-validation",
    )
    receipt_missing_gate = {"held_out": {}, "untrained_baseline": {}}  # no graded_held_out

    with pytest.raises(RuntimeError, match="graded_shards"):
        _assert_graded_gate_present(cfg, receipt_missing_gate)


def test_a_region_with_no_graded_shards_declared_is_not_checked() -> None:
    """Negative control: `code`/`retrieve` declare no graded set at all, so a receipt
    without `graded_held_out` for them is correct, not a defect. The guard must fire on
    a promise broken, not on every region that never made one."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.pretrain import PretrainConfig, _assert_graded_gate_present

    cfg = PretrainConfig(
        region="code",
        pair_columns=("docstring", "code"),
        shards=["train.parquet"],
    )
    receipt_no_gate = {"held_out": {}, "untrained_baseline": {}}

    _assert_graded_gate_present(cfg, receipt_no_gate)  # must not raise


def test_a_declared_graded_gate_present_in_the_receipt_does_not_raise() -> None:
    """Positive control: a graded gate that DID make it into the receipt is a pass, not
    a near-miss the guard should also flag."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.pretrain import PretrainConfig, _assert_graded_gate_present

    cfg = PretrainConfig(
        region="compress",
        pair_columns=("anchor", "positive"),
        shards=["train.parquet"],
        graded_shards=["stsb-validation.parquet"],
        graded_columns=("sentence1", "sentence2", "score"),
        graded_name="stsb-validation",
    )
    receipt_with_gate = {
        "held_out": {},
        "untrained_baseline": {},
        "graded_held_out": {"spearman": 0.4956},
    }

    _assert_graded_gate_present(cfg, receipt_with_gate)  # must not raise


# ---------------------------------------------------------------------------------------
# DEFECT 5, continued -- the guard above is unreachable by the actual regression, and the
# runner it replaces is fail-open where the code path it replaced was fail-closed.
# ---------------------------------------------------------------------------------------


def _load_csd_train_all():
    """Import scripts/csd-train-all.py the same way tests/test_reserved_corpus_guard.py
    does (see that file's docstring): the hyphenated filename is not a valid module name,
    so every consumer loads it via `importlib.util.spec_from_file_location`. A distinct
    `sys.modules` key from that file's loader keeps the two loads independent.
    """
    path = Path(__file__).resolve().parents[1] / "scripts" / "csd-train-all.py"
    spec = importlib.util.spec_from_file_location("csd_train_all_graded_gate_test", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_csd_train_all()


def test_compress_graded_spec_in_the_runner_matches_compress_py() -> None:
    """Pin `REGIONS["compress"]`'s `GradedSpec` against `regions/compress.py`'s own
    `GRADED_SHARD`/`graded_columns`/`graded_name` constants.

    `scripts/csd-train-all.py`'s `GradedSpec` docstring and the `REGIONS["compress"]`
    comment both assert "both paths must agree on what the compress graded gate means",
    but nothing checked that before this test -- an edit to either side (a renamed STS-B
    column, a moved shard, a re-tagged gate name) would diverge in silence exactly like
    the rest of R4 did. `pytest.importorskip` covers `compress.py`'s own import chain
    (`regions.pretrain`), not this test's assertions, which touch neither torch nor
    tokenizers.
    """
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.compress import GRADED_SHARD

    _sources, _note, _default_max_len, graded_spec = mod.REGIONS["compress"]
    assert graded_spec is not None, "compress must declare a GradedSpec"
    glob_pat, columns, name = graded_spec

    assert glob_pat.endswith(GRADED_SHARD), (glob_pat, GRADED_SHARD)
    assert columns == ("sentence1", "sentence2", "score")
    assert name == "stsb-validation"


def test_run_region_raises_when_the_declared_graded_source_does_not_resolve(
    tmp_path: Path,
) -> None:
    """Reproduce the exact reachable regression: the training shard is present, the
    graded (STS-B) shard is not -- an unmounted NFS export, a renamed upstream dataset
    directory, or a typo in the glob, all of which leave `graded_spec` declared but its
    glob unresolved. `run_region` must refuse to start rather than build a
    `PretrainConfig` with `graded_shards=[]`/`graded_name="stsb-validation"` and let
    `pretrain_region` write a receipt with no `graded_held_out`.

    Before the fix, this printed one line ("graded source MISSING: ... compress declares
    a graded gate but it will not run this time") and returned a resolved dry-run plan
    instead of raising -- fail-open, and weaker than `regions/compress.py`'s
    `compress_config`, which raises `FileNotFoundError` for the identical missing shard.
    """

    def fake_shards(pattern: str, root: Path = mod.CORPUS) -> list[str]:
        if "all-nli" in pattern:
            return [str(tmp_path / "all-nli-train.parquet")]
        if "stsb" in pattern:
            return []  # the graded shard: declared, unresolved
        return []

    with pytest.MonkeyPatch.context() as m:
        m.setattr(mod, "_shards", fake_shards)
        m.setattr(mod, "_schema_mismatch", lambda shards, cols: None)  # bypass parquet I/O
        with pytest.raises(mod.GradedSourceMissingError, match="stsb-validation"):
            mod.run_region(
                name="compress",
                state=tmp_path,
                steps=1,
                batch=1,
                shard_limit=0,
                dry=True,
            )


def test_run_region_dry_run_resolves_the_graded_gate_when_it_is_present(
    tmp_path: Path,
) -> None:
    """Positive control: when both shards resolve, the dry-run plan carries the graded
    gate through to `graded_shards`/`graded_columns`/`graded_name` rather than raising --
    the guard above must fire on absence, not on every compress run."""

    def fake_shards(pattern: str, root: Path = mod.CORPUS) -> list[str]:
        if "all-nli" in pattern:
            return [str(tmp_path / "all-nli-train.parquet")]
        if "stsb" in pattern:
            return [str(tmp_path / "stsb-validation.parquet")]
        return []

    with pytest.MonkeyPatch.context() as m:
        m.setattr(mod, "_shards", fake_shards)
        m.setattr(mod, "_schema_mismatch", lambda shards, cols: None)
        result = mod.run_region(
            name="compress", state=tmp_path, steps=1, batch=1, shard_limit=0, dry=True
        )

    assert result is None  # dry run never returns a receipt


def test_dry_run_plan_omits_graded_keys_for_a_region_with_no_graded_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`code`/`retrieve`/`reason` declare no `GradedSpec`. Before the fix, the dry-run
    plan still printed `graded_columns: ["sentence1", "sentence2", "score"]` for them --
    `graded_cols` was initialised to the STS-B default before the `if graded_spec is not
    None` branch ran, so a reader of the `code` plan saw STS-B column names attached to a
    region with no graded set at all. `graded_shards`/`graded_columns`/`graded_name` must
    all be `null` in the emitted JSON, not filled with another region's defaults.
    """

    def fake_shards(pattern: str, root: Path = mod.CORPUS) -> list[str]:
        return [str(tmp_path / "codesearchnet-shard.parquet")]

    with pytest.MonkeyPatch.context() as m:
        m.setattr(mod, "_shards", fake_shards)
        m.setattr(mod, "_schema_mismatch", lambda shards, cols: None)
        result = mod.run_region(
            name="code", state=tmp_path, steps=1, batch=1, shard_limit=0, dry=True
        )

    assert result is None
    import json as _json

    printed = capsys.readouterr().out
    plan_json = printed.split("resolved PretrainConfig (dry run, no training started):")[1]
    plan = _json.loads(plan_json)
    assert plan["graded_shards"] is None
    assert plan["graded_columns"] is None
    assert plan["graded_name"] is None


# ---------------------------------------------------------------------------------------
# DEFECT 5, continued -- `_assert_graded_gate_present` itself was unwired and unweighed:
# a review mutation deleted its call site from `pretrain_region` and a review mutation
# that keyed it back on `cfg.graded_shards` alone (undoing the `or cfg.graded_name`
# clause this file's other DEFECT-5 tests exercise only by calling the guard directly)
# both left the full suite green. Every test above this point calls
# `_assert_graded_gate_present` as a bare function; none of them run it AS `pretrain_region`
# runs it, so neither mutation could be caught. The two tests below train for real, on
# CPU, through `pretrain_region` itself.
# ---------------------------------------------------------------------------------------


def _build_tiny_tokenizer(path: Path, vocab_texts: list[str]) -> None:
    """A WordLevel tokenizer covering exactly the vocabulary these tests use."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer

    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106 -- a token id name, not a secret
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tok.train_from_iterator(vocab_texts, trainer=trainer)
    tok.save(str(path))


def _build_tiny_train_shard(path: Path, n_pairs: int) -> tuple[list[str], list[str]]:
    """`n_pairs` distinct (anchor, positive) rows -- distinct anchors so none are dropped
    by `build_splits`'s anchor-fingerprint dedup. Returns the texts so the caller can feed
    the same vocabulary to the tokenizer."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    anchors = [f"train anchor number {i}" for i in range(n_pairs)]
    positives = [f"train anchor number {i} matched" for i in range(n_pairs)]
    pq.write_table(pa.table({"anchor": anchors, "positive": positives}), path)
    return anchors, positives


def _tiny_pretrain_cfg(tmp_path: Path, *, graded_shards: list[str], graded_name: str):
    """The exact shape of run a mutation-tested review reproduced: small enough to train
    in a fraction of a second on CPU, distinguished only by whether a graded gate is
    declared and whether it resolved."""
    from cogsyndelta.regions.pretrain import PretrainConfig
    from cogsyndelta.regions.text_encoder import TextEncoderConfig

    tok_path = tmp_path / "tokenizer.json"
    shard_path = tmp_path / "train.parquet"
    anchors, positives = _build_tiny_train_shard(shard_path, 30)
    _build_tiny_tokenizer(tok_path, anchors + positives + ["graded", "sentence", "example"])

    return PretrainConfig(
        region="compress",
        pair_columns=("anchor", "positive"),
        shards=[str(shard_path)],
        steps=2,
        batch_size=4,
        holdout_pairs=8,
        eval_every=1,
        checkpoint_every=0,
        device="cpu",
        bf16=False,
        encoder=TextEncoderConfig(dim=32, depth=1, n_heads=2, max_len=16),
        max_len=16,
        tokenizer_path=str(tok_path),
        out_dir=str(tmp_path / "run"),
        graded_shards=graded_shards,
        graded_columns=("sentence1", "sentence2", "score"),
        graded_name=graded_name,
    )


def test_pretrain_region_raises_when_graded_name_is_declared_but_shards_did_not_resolve(
    tmp_path: Path,
) -> None:
    """The reachable regression, reproduced through `pretrain_region` itself rather than
    through `_assert_graded_gate_present` called by hand.

    `graded_shards=[]` with `graded_name` set is exactly the config `run_region` used to
    build (before this branch's `GradedSourceMissingError`) when a declared graded glob
    resolved nothing, and exactly the shape a caller other than `run_region` -- a hand-run
    script, a notebook, a future runner -- can still construct directly. `_prepare_graded`
    reads the empty `graded_shards` as "no graded set", so training proceeds and would
    finish with no `graded_held_out` at all in the receipt were it not for the guard --
    this must raise `RuntimeError` before `pretrain_region` returns or writes anything to
    `out_dir`, not merely produce a gate-less receipt.

    Proves two review mutations against `pretrain_region` at once: deleting the
    `_assert_graded_gate_present(cfg, receipt)` call site, and narrowing its condition
    back to `cfg.graded_shards` alone (dropping `or cfg.graded_name`). Either one leaves
    `graded_shards=[]` unguarded and this test would stop raising.
    """
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    pytest.importorskip("pyarrow", reason="train group not installed")
    from cogsyndelta.regions.pretrain import pretrain_region

    cfg = _tiny_pretrain_cfg(tmp_path, graded_shards=[], graded_name="stsb-validation")

    with pytest.raises(RuntimeError, match="graded_shards"):
        pretrain_region(cfg)

    written = list((tmp_path / "run").glob("*.json")) if (tmp_path / "run").is_dir() else []
    assert written == [], f"receipt written despite the missing graded gate: {written}"


def test_pretrain_region_writes_graded_held_out_when_the_gate_resolves(
    tmp_path: Path,
) -> None:
    """Positive control for the test above: a graded gate that DOES resolve must still
    train and write a receipt normally through `pretrain_region` -- the guard fires on
    absence, not on every run that declares one."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    pytest.importorskip("pyarrow", reason="train group not installed")
    import pyarrow as pa
    import pyarrow.parquet as pq

    from cogsyndelta.regions.pretrain import pretrain_region

    graded_path = tmp_path / "stsb-validation.parquet"
    pq.write_table(
        pa.table(
            {
                "sentence1": ["graded example one", "graded example two", "graded example three"],
                "sentence2": [
                    "graded sentence one",
                    "graded sentence two",
                    "graded sentence three",
                ],
                "score": [0.9, 0.4, 0.1],
            }
        ),
        graded_path,
    )
    cfg = _tiny_pretrain_cfg(
        tmp_path, graded_shards=[str(graded_path)], graded_name="stsb-validation"
    )

    receipt = pretrain_region(cfg)

    assert "graded_held_out" in receipt
    assert "spearman" in receipt["beats_untrained"]


# ---------------------------------------------------------------------------------------
# DEFECT 6 (R9) -- the checkpoint fingerprint covered `PretrainConfig`'s own fields only,
# so a corpus rewrite or a change to the split-building CODE under an unchanged config was
# invisible to it, and a legacy checkpoint with no fingerprint at all was silently adopted
# as a fresh start. Commit 883c9d4 is the reproduction: it changed `build_splits`
# (unconditional shuffle) without touching a single `PretrainConfig` field, so the pre-
# and post-fix runs fingerprinted identically and `code-checkpoints/` now mixes a 16:21
# pre-fix `step-007998.pt` with 17:07-17:08 post-fix files with nothing on disk to tell
# them apart.
# ---------------------------------------------------------------------------------------


def test_checkpoint_saved_under_one_fingerprint_refuses_to_resume_under_another(
    tmp_path: Path,
) -> None:
    """A checkpoint trained under fingerprint A must refuse -- not silently resume from,
    not silently discard -- a resume attempt under a different fingerprint B, and the
    refusal must name BOTH fingerprints so an operator reading only the error can tell
    which checkpoint and which config it disagreed with."""
    pytest.importorskip("torch", reason="train group not installed")
    # `cogsyndelta.regions._checkpoint` is model-agnostic and needs only torch, but
    # importing anything under `cogsyndelta.regions` runs that package's `__init__`,
    # which unconditionally imports `regions.pretrain` -- and THAT needs `tokenizers`.
    # Skip on the same condition the rest of this file already does, rather than let an
    # environment with torch but not the full train group fail on an unrelated import.
    pytest.importorskip("tokenizers", reason="train group not installed")
    import torch

    from cogsyndelta.regions._checkpoint import atomic_save, load_resumable

    ckpt_dir = tmp_path / "ckpt"
    fp_a = "a" * 32
    atomic_save(
        {
            "config_fingerprint": fp_a,
            "config_fields": {"lr": 1e-3},
            "model": {"w": torch.zeros(2)},  # tiny synthetic tensor -- no GPU, no real model
        },
        ckpt_dir / "final.pt",
    )

    fp_b = "b" * 32
    with pytest.raises(ValueError, match="refusing to resume") as exc_info:
        load_resumable(ckpt_dir, fp_b, {"lr": 2e-3})

    message = str(exc_info.value)
    assert fp_a in message, "the checkpoint's own fingerprint must be in the message"
    assert fp_b in message, "the current config's fingerprint must be in the message"


def test_unfingerprinted_checkpoint_refuses_to_resume_by_default(tmp_path: Path) -> None:
    """A checkpoint written before fingerprinting existed (or with it bypassed) carries
    no `config_fingerprint` at all. `regions/pretrain.py` -- via
    `PretrainConfig.allow_unfingerprinted_resume` defaulting `False`, and
    `pretrain_region` passing that straight through as `load_resumable`'s
    `allow_unfingerprinted` -- must refuse to resume from it rather than silently
    starting fresh next to it: silently starting fresh is exactly the shape that let a
    pre-883c9d4 checkpoint keep sitting in `code-checkpoints/` beside post-fix files with
    nothing to tell them apart.

    `load_resumable` ITSELF still defaults `allow_unfingerprinted=True` -- unchanged
    behaviour for `regions/vl_pretrain.py` and `regions/classify_pretrain.py`, which call
    it positionally and never opted into the stricter mode; only `regions/pretrain.py`
    is stricter by default. Both ends of that split are asserted here.
    """
    pytest.importorskip("torch", reason="train group not installed")
    # See the sibling test above for why `tokenizers` is skipped here too even though
    # this test touches only `_checkpoint.py`.
    pytest.importorskip("tokenizers", reason="train group not installed")
    import torch

    from cogsyndelta.regions._checkpoint import atomic_save, load_resumable

    ckpt_dir = tmp_path / "ckpt"
    atomic_save({"model": {"w": torch.zeros(2)}, "step": 5}, ckpt_dir / "final.pt")

    # regions/pretrain.py's default: PretrainConfig.allow_unfingerprinted_resume=False.
    with pytest.raises(ValueError, match="no config_fingerprint"):
        load_resumable(ckpt_dir, "current-fingerprint", {}, allow_unfingerprinted=False)

    # load_resumable's OWN default (what vl_pretrain.py/classify_pretrain.py still get):
    # unchanged from before this fix -- a fresh start, never a silent resume from a
    # checkpoint with nothing on it to validate against.
    assert load_resumable(ckpt_dir, "current-fingerprint", {}) is None
    assert load_resumable(ckpt_dir, "current-fingerprint", {}, allow_unfingerprinted=True) is None


def test_pretrain_region_refuses_unfingerprinted_checkpoint_by_default(tmp_path: Path) -> None:
    """The test above exercises `load_resumable` directly, passing
    `allow_unfingerprinted=False` by hand -- it never touches
    `PretrainConfig.allow_unfingerprinted_resume` or the wiring at `pretrain_region`'s
    `load_resumable(...)` call site that is supposed to pass it through. A mutation that
    deletes that wiring entirely (calls `load_resumable(ckpt_dir, fingerprint, fields)`
    with `load_resumable`'s own permissive `allow_unfingerprinted=True` default, silently
    restoring the exact silent-adoption behaviour requirement (2) exists to stop) leaves
    the test above still green, because it never runs `pretrain_region` at all.

    This test plants an unfingerprinted `final.pt` directly in the vintage-fingerprinted
    directory `pretrain_region` itself computes and reads
    (`{out_dir}/{region}-checkpoints/{_vintage_fingerprint(cfg)[:8]}`), then calls
    `pretrain_region(cfg)` -- the real entry point, not `load_resumable` -- and asserts
    both ends: refused by default, and let through only when
    `allow_unfingerprinted_resume=True` is set on the config.
    """
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    pytest.importorskip("pyarrow", reason="train group not installed")
    import torch

    from cogsyndelta.regions._checkpoint import atomic_save
    from cogsyndelta.regions.pretrain import _vintage_fingerprint, pretrain_region

    cfg = _tiny_pretrain_cfg(tmp_path, graded_shards=[], graded_name="")
    assert cfg.allow_unfingerprinted_resume is False, "default under test"

    ckpt_dir = Path(cfg.out_dir) / f"{cfg.region}-checkpoints" / _vintage_fingerprint(cfg)[:8]
    atomic_save({"model": {"w": torch.zeros(2)}, "step": 5}, ckpt_dir / "final.pt")

    with pytest.raises(ValueError, match="no config_fingerprint"):
        pretrain_region(cfg)

    # positive control: the same planted checkpoint, the same cfg, only the flag differs.
    permissive_cfg = replace(cfg, allow_unfingerprinted_resume=True)
    receipt = pretrain_region(permissive_cfg)
    assert "checkpoint" in receipt, "the allow flag must let training proceed to a receipt"


def test_changing_only_the_corpus_content_changes_the_fingerprint(tmp_path: Path) -> None:
    """The actual 883c9d4 shape: byte-identical `PretrainConfig` fields, only what is ON
    DISK at the shard path changes. Before this fix `_config_fingerprint` hashed
    `_resume_fields`, which carried shard PATHS as strings -- a path that never changes
    could not move the fingerprint no matter what the file underneath it held."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.pretrain import PretrainConfig, _config_fingerprint

    shard = tmp_path / "corpus.parquet"
    shard.write_bytes(b"x" * 100)
    cfg = PretrainConfig(
        region="fp-corpus-test",
        pair_columns=("a", "b"),
        shards=[str(shard)],
        out_dir=str(tmp_path / "run"),
    )
    fp_before = _config_fingerprint(cfg)

    # Same path, same PretrainConfig object -- only the file's CONTENT (here, its size)
    # changes, the way a corpus re-fetch or re-filter would.
    shard.write_bytes(b"y" * 250)
    fp_after = _config_fingerprint(cfg)

    assert fp_before != fp_after


def test_config_fingerprint_also_moves_when_the_split_building_code_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Companion to the two tests above: proves the split-building CODE hash is actually
    wired into `_config_fingerprint`, not merely computed and discarded. Editing the real
    source file mid-test is not how to exercise this (see `_split_code_fingerprint`'s own
    docstring for why it hashes file bytes rather than `inspect.getsource`); instead,
    substitute the function everything else here treats as ground truth and check the
    composite moves with it. Commit 883c9d4 is exactly this shape: `build_splits`
    changed, no `PretrainConfig` field did."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    import cogsyndelta.regions.pretrain as pretrain_mod

    cfg = pretrain_mod.PretrainConfig(region="fp-code-test", pair_columns=("a", "b"), shards=[])

    monkeypatch.setattr(pretrain_mod, "_split_code_fingerprint", lambda: "code-version-one")
    fp_one = pretrain_mod._config_fingerprint(cfg)
    monkeypatch.setattr(pretrain_mod, "_split_code_fingerprint", lambda: "code-version-two")
    fp_two = pretrain_mod._config_fingerprint(cfg)

    assert fp_one != fp_two


def test_checkpoint_directory_is_vintage_prefixed_and_receipt_records_a_checksum(
    tmp_path: Path,
) -> None:
    """End to end, and the receipt half of R9: the checkpoint directory a real
    `pretrain_region` run writes into must carry the vintage-fingerprint prefix
    (`_vintage_fingerprint`, corpus content + split-building code), and the receipt
    naming the final checkpoint must carry a content hash of the exact file it names --
    so a reader is not trusting the path alone, and a checkpoint silently swapped or
    truncated on disk after the receipt was written no longer passes as a match."""
    pytest.importorskip("pyarrow", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    import hashlib

    from cogsyndelta.regions.pretrain import PretrainConfig, _vintage_fingerprint, pretrain_region
    from cogsyndelta.regions.text_encoder import TextEncoderConfig
    from tests.test_pretrain_resume import _build_pairs_parquet, _build_tokenizer

    tok_path = tmp_path / "tok.json"
    shard = tmp_path / "pairs.parquet"
    _build_tokenizer(tok_path, 40)
    _build_pairs_parquet(shard, 40)

    cfg = PretrainConfig(
        region="fp8-dir-test",
        pair_columns=("anchor", "positive"),
        shards=[str(shard)],
        steps=2,
        batch_size=4,
        holdout_pairs=4,
        eval_every=2,
        checkpoint_every=0,
        max_len=16,
        seed=0,
        device="cpu",
        encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=16),
        tokenizer_path=str(tok_path),
        out_dir=str(tmp_path / "run"),
    )
    receipt = pretrain_region(cfg)

    fp8 = _vintage_fingerprint(cfg)[:8]
    final_ckpt = Path(cfg.out_dir) / f"{cfg.region}-checkpoints" / fp8 / "final.pt"
    assert final_ckpt.is_file()
    assert receipt["checkpoint"] == str(final_ckpt)
    assert receipt["checkpoint_sha256"] == hashlib.sha256(final_ckpt.read_bytes()).hexdigest()


def test_checkpoint_directory_changes_when_split_code_fingerprint_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Companion to `test_config_fingerprint_also_moves_when_the_split_building_code_does`,
    but for `_vintage_fingerprint` specifically -- the function that names the checkpoint
    DIRECTORY, not the one that decides whether a resume is valid. The two call
    `_split_code_fingerprint()` at separate sites; a test that only proves
    `_config_fingerprint` moves says nothing about whether a code-vintage change actually
    lands in its own directory on disk. Verified by mutation: deleting the
    `h.update(_split_code_fingerprint().encode())` line from `_vintage_fingerprint` left
    every existing test green before this one was added.

    Exercises `pretrain_region` end to end under two different `_split_code_fingerprint`
    return values (standing in for `build_splits`/`screen_pair_contamination` actually
    changing -- commit 883c9d4's shape) with the SAME `PretrainConfig` otherwise, and
    asserts the two runs land in two different checkpoint directories, each holding its
    own `final.pt` -- the exact "code-checkpoints/ mixes pre- and post-fix files" failure
    this defect is named for.
    """
    pytest.importorskip("pyarrow", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    import cogsyndelta.regions.pretrain as pretrain_mod
    from cogsyndelta.regions.text_encoder import TextEncoderConfig
    from tests.test_pretrain_resume import _build_pairs_parquet, _build_tokenizer

    tok_path = tmp_path / "tok.json"
    shard = tmp_path / "pairs.parquet"
    _build_tokenizer(tok_path, 40)
    _build_pairs_parquet(shard, 40)

    def _make_cfg() -> pretrain_mod.PretrainConfig:
        return pretrain_mod.PretrainConfig(
            region="fp8-split-code-test",
            pair_columns=("anchor", "positive"),
            shards=[str(shard)],
            steps=2,
            batch_size=4,
            holdout_pairs=4,
            eval_every=2,
            checkpoint_every=0,
            max_len=16,
            seed=0,
            device="cpu",
            encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=16),
            tokenizer_path=str(tok_path),
            out_dir=str(tmp_path / "run"),
        )

    monkeypatch.setattr(pretrain_mod, "_split_code_fingerprint", lambda: "code-version-A")
    receipt_a = pretrain_mod.pretrain_region(_make_cfg())

    monkeypatch.setattr(pretrain_mod, "_split_code_fingerprint", lambda: "code-version-B")
    receipt_b = pretrain_mod.pretrain_region(_make_cfg())

    dir_a = Path(receipt_a["checkpoint"]).parent
    dir_b = Path(receipt_b["checkpoint"]).parent
    assert dir_a != dir_b
    assert (dir_a / "final.pt").is_file()
    assert (dir_b / "final.pt").is_file()


def test_checkpoint_directory_changes_when_corpus_content_fingerprint_does(
    tmp_path: Path,
) -> None:
    """Symmetric twin of `test_checkpoint_directory_changes_when_split_code_fingerprint_does`,
    for `_vintage_fingerprint`'s OTHER half: the corpus-content line, not the split-code
    one. `_vintage_fingerprint` calls `_corpus_content_fingerprint` and
    `_split_code_fingerprint` independently, so a test that only changes the code side
    says nothing about whether a corpus rewrite under the SAME shard path -- 883c9d4's
    other named case, and the actual R9 gap this branch closes -- lands in its own
    directory rather than colliding with the prior vintage's checkpoint. Verified by
    mutation: deleting `h.update(_corpus_content_fingerprint(cfg).encode())` from
    `_vintage_fingerprint` leaves the entire suite green, including this file, before
    this test was added -- the two runs below land in the SAME directory and the second
    one is refused by `load_resumable`'s pre-existing config-mismatch check instead of
    getting its own vintage.

    Rewrites the shard PARQUET FILE in place at the same path (40 rows -> 60 rows, i.e. a
    byte-size change -- `fingerprint_corpus` hashes shard name + byte size, not full file
    content, so a same-size content edit would not move this fingerprint either) between
    two `pretrain_region` calls that otherwise share the identical `PretrainConfig`
    (literally the same object, so no config field differs at all), and asserts the two
    runs land in two different checkpoint directories, each holding its own `final.pt`.
    """
    pytest.importorskip("pyarrow", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.pretrain import PretrainConfig, pretrain_region
    from cogsyndelta.regions.text_encoder import TextEncoderConfig
    from tests.test_pretrain_resume import _build_pairs_parquet, _build_tokenizer

    tok_path = tmp_path / "tok.json"
    shard = tmp_path / "pairs.parquet"
    _build_tokenizer(tok_path, 60)
    _build_pairs_parquet(shard, 40)

    cfg = PretrainConfig(
        region="fp8-corpus-vintage-test",
        pair_columns=("anchor", "positive"),
        shards=[str(shard)],
        steps=2,
        batch_size=4,
        holdout_pairs=4,
        eval_every=2,
        checkpoint_every=0,
        max_len=16,
        seed=0,
        device="cpu",
        encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=16),
        tokenizer_path=str(tok_path),
        out_dir=str(tmp_path / "run"),
    )

    receipt_a = pretrain_region(cfg)

    # Rewrite the corpus IN PLACE at the same path: same `cfg`, same `cfg.shards`, but
    # different content on disk -- exactly 883c9d4's "corpus content changed under an
    # unchanged config" shape.
    _build_pairs_parquet(shard, 60)
    receipt_b = pretrain_region(cfg)

    dir_a = Path(receipt_a["checkpoint"]).parent
    dir_b = Path(receipt_b["checkpoint"]).parent
    assert dir_a != dir_b
    assert (dir_a / "final.pt").is_file()
    assert (dir_b / "final.pt").is_file()


# DEFECT 6 -- `beats_untrained` was satisfiable by a broken baseline.
#
# receipts/retrieve-20260902T203759Z.json recorded `untrained_baseline["recall@1"] ==
# 0.0` on the real 512-pair FiQA/NQ/GooAQ holdout, so any `held_out["recall@1"] > 0.0`
# -- including pure noise -- passed `beats_untrained["recall@1"]`. Investigating that
# receipt (calling `build_splits` + a freshly constructed, untrained `TextEncoder`
# directly against the real corpus, seed=0, production defaults -- no training) found
# NO bug in `evaluate`/`recall_at_k`/`build_splits`: the score matrix carried no NaNs,
# was not tied, and the untrained encoder's held-out embeddings were genuinely
# near-collapsed (`emb_std` ~0.007, vs ~0.06 for a random unit-norm 256-d direction) --
# the sincos position embedding (~unit scale) dwarfing the Normal(0, 0.02) token
# embeddings at init. That collapse concentrates similarity on one arbitrary "attractor"
# pair per run; whether the attractor happens to sit ON the diagonal (giving 1/N, as a
# from-scratch CPU repro of the same config measured) or OFF it (giving exactly 0, as
# the GPU/bf16 production run did) is close to a coin flip across a 512-row diagonal,
# not a defect in the measurement. So the fix is not a numerical patch to the metric --
# it is refusing to certify `beats_untrained` against a baseline that low at all.
#
# Two real gaps DID turn up alongside that finding, and both are fixed here:
#   - `evaluate` had no NaN guard. A `topk`/`argsort` over an all-NaN score matrix (a
#     genuinely diverged encoder, e.g. a bf16 overflow) does not raise; it returns a
#     plausible-looking float, indistinguishable from a real measurement.
#   - `beats_untrained` had no floor at all: `final[m] > baseline[m]` alone is
#     trivially satisfiable by ANY baseline, however broken.
# ---------------------------------------------------------------------------------------


class _ConstantTextEncoder:
    """Stand-in for `TextEncoder`: ignores its input entirely and emits the SAME nonzero
    vector for every item, for every call -- the literal "collapsed to one point"
    encoder `_beats_untrained_gate`'s docstring distinguishes from the real untrained
    encoder's near-collapse. Implements just enough of `nn.Module`'s surface for
    `evaluate` to drive it (`.eval()`, `.train()`, `.training`, `__call__`)."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.training = True

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True

    def __call__(self, input_ids, attention_mask):
        import torch

        return torch.full((input_ids.size(0), self.dim), 0.5)


class _NaNTextEncoder(_ConstantTextEncoder):
    """Same shape as `_ConstantTextEncoder`, but emits NaN -- the diverged/overflowed
    encoder `evaluate`'s NaN guard exists to catch before it reaches `recall_at_k`."""

    def __call__(self, input_ids, attention_mask):
        import torch

        return torch.full((input_ids.size(0), self.dim), float("nan"))


def _tiny_eval_pairs_and_tokenizer(tmp_path: Path, n: int) -> tuple[list[tuple[str, str]], Path]:
    """`n` distinct (anchor, positive) pairs plus a tokenizer covering their vocabulary --
    enough for `evaluate()` to run without needing a trained model or real corpus."""
    pairs = [(f"anchor item number {i}", f"positive item number {i}") for i in range(n)]
    tok_path = tmp_path / "tokenizer.json"
    vocab = [t for pair in pairs for t in pair]
    _build_tiny_tokenizer(tok_path, vocab)
    return pairs, tok_path


def test_evaluate_on_an_encoder_that_collapses_to_one_point_yields_chance_not_zero(
    tmp_path: Path,
) -> None:
    """R6, part 1: the literal "identical scores" case the receipt's 0.0 looked like it
    might be. `recall_at_k`'s `topk` resolves exact ties to a stable (lowest-index) order,
    so an encoder that maps EVERY input to the same point scores `recall@1 == 1/N`, never
    0.0 -- confirming the real receipt's 0.0 was not this failure mode (see the DEFECT 6
    docstring above)."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from tokenizers import Tokenizer

    from cogsyndelta.regions.pretrain import evaluate

    n = 16
    pairs, tok_path = _tiny_eval_pairs_and_tokenizer(tmp_path, n)
    tok = Tokenizer.from_file(str(tok_path))

    result = evaluate(_ConstantTextEncoder(), tok, pairs, max_len=16, device=_cpu_device())

    assert result["recall@1"] == pytest.approx(1.0 / n)
    assert result["emb_std"] == pytest.approx(0.0, abs=1e-6)  # every embedding IS the same point


def test_evaluate_raises_on_nan_scores(tmp_path: Path) -> None:
    """R6, part 2: a diverged/overflowed encoder must fail the run loudly, not hand back
    a plausible-looking recall number computed over NaN similarity scores."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from tokenizers import Tokenizer

    from cogsyndelta.regions.pretrain import evaluate

    pairs, tok_path = _tiny_eval_pairs_and_tokenizer(tmp_path, 16)
    tok = Tokenizer.from_file(str(tok_path))

    with pytest.raises(ValueError, match="NaN"):
        evaluate(_NaNTextEncoder(), tok, pairs, max_len=16, device=_cpu_device())


def _cpu_device():
    import torch

    return torch.device("cpu")


def test_beats_untrained_gate_rejects_a_baseline_below_chance() -> None:
    """R6, part 3: the exact receipt shape -- `untrained_baseline["recall@1"] == 0.0`
    over a 512-pair eval -- must fail the sanity gate regardless of how good the trained
    model's own number is."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.pretrain import _beats_untrained_gate

    final = {"recall@1": 0.748046875, "recall@10": 0.94921875}
    baseline = {"recall@1": 0.0, "recall@10": 0.021484375}  # the real receipt's numbers

    chance, beats = _beats_untrained_gate(final, baseline, eval_pairs=512)

    assert chance["recall@1"] == pytest.approx(1 / 512)
    assert beats["recall@1"] is False
    assert beats["recall@10"] is False


def test_beats_untrained_gate_accepts_a_sane_baseline_past_the_margin() -> None:
    """Positive control: a baseline AT chance (as an untrained model legitimately can be,
    see the DEFECT 6 docstring) does not itself fail the gate -- only a baseline BELOW
    `chance / 2` does -- and a trained model that clears it by more than the margin
    passes."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.pretrain import _beats_untrained_gate

    final = {"recall@1": 0.75, "recall@10": 0.95}
    baseline = {"recall@1": 1 / 512, "recall@10": 10 / 512}  # exactly at chance

    chance, beats = _beats_untrained_gate(final, baseline, eval_pairs=512)

    assert beats["recall@1"] is True
    assert beats["recall@10"] is True


def test_beats_untrained_gate_rejects_a_win_too_small_to_be_signal() -> None:
    """The margin itself: a trained model that clears baseline and chance by less than
    `_BEATS_UNTRAINED_MARGIN` must not read as `beats_untrained` -- that gap is noise on
    a 512-pair holdout, not evidence of learning."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.pretrain import _beats_untrained_gate

    baseline = {"recall@1": 1 / 512, "recall@10": 10 / 512}
    final = {"recall@1": baseline["recall@1"] + 0.001, "recall@10": baseline["recall@10"] + 0.001}

    _, beats = _beats_untrained_gate(final, baseline, eval_pairs=512)

    assert beats["recall@1"] is False
    assert beats["recall@10"] is False


def test_beats_untrained_gate_baseline_sane_rejects_a_non_zero_value_below_chance_half() -> None:
    """R6 continuation: `baseline_sane`'s threshold is `chance / 2`, not `0`. On real
    (quantised-to-`k/N`) `recall@1` values every nonzero baseline already clears
    `chance / 2` -- see `_beats_untrained_gate`'s "WHAT `baseline_sane`'S THRESHOLD
    ACTUALLY CATCHES" docstring section -- so a test that only ever passes quantised
    baselines could not tell `>= chance / 2` apart from `> 0`, and a mutation that
    weakened the threshold to `> 0` would pass every other test in this file. This
    constructs the one shape that distinguishes them directly: `eval_pairs=8` puts
    `chance / 2` at `0.0625`, and a synthetic (non-quantised, as a real measurement never
    is) `baseline["recall@1"] = 0.05` sits strictly between `0` and that threshold --
    nonzero, and still correctly rejected."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.pretrain import _beats_untrained_gate

    final = {"recall@1": 0.9, "recall@10": 0.99}
    baseline = {"recall@1": 0.05, "recall@10": 0.2}  # 0.05 < chance/2 == 1/16 == 0.0625

    chance, beats = _beats_untrained_gate(final, baseline, eval_pairs=8)

    assert chance["recall@1"] == pytest.approx(0.125)
    assert baseline["recall@1"] > 0  # nonzero -- this is not the `== 0.0` case above
    assert beats["recall@1"] is False
    assert beats["recall@10"] is False


def test_beats_untrained_gate_spearman_hardcoded_true_is_caught() -> None:
    """R7 -- the exact regression this commit fixes: `beats_untrained["spearman"]` used
    to be `graded_final["spearman"] > graded_baseline["spearman"]` with no margin and no
    baseline sanity, so hardcoding it `True` broke no test. Here the trained model's
    spearman is BELOW the untrained baseline's, so a correct gate must read `False`; a
    version that ignores both inputs and returns `True` -- the exact hardcode this guards
    against -- fails this assertion."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.pretrain import _beats_untrained_gate

    final = {"recall@1": 0.9, "recall@10": 0.99}
    baseline = {"recall@1": 0.4, "recall@10": 0.8}  # sane: well above chance/2
    graded_final = {"spearman": 0.10}
    graded_baseline = {"spearman": 0.30}  # trained model did WORSE than untrained

    _, beats = _beats_untrained_gate(
        final, baseline, eval_pairs=512, graded_final=graded_final, graded_baseline=graded_baseline
    )

    assert beats["spearman"] is False


def test_beats_untrained_gate_spearman_below_margin_fails_the_gate() -> None:
    """The margin applied to spearman specifically: a trained model that beats the
    graded baseline by less than `_BEATS_UNTRAINED_MARGIN` must not read as
    `beats_untrained["spearman"]` -- mirrors
    `test_beats_untrained_gate_rejects_a_win_too_small_to_be_signal` for recall@k."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.pretrain import _BEATS_UNTRAINED_MARGIN, _beats_untrained_gate

    final = {"recall@1": 0.9, "recall@10": 0.99}
    baseline = {"recall@1": 0.4, "recall@10": 0.8}
    graded_baseline = {"spearman": 0.30}
    graded_final = {
        "spearman": 0.30 + (_BEATS_UNTRAINED_MARGIN / 2)
    }  # half the margin, not past it

    _, beats = _beats_untrained_gate(
        final, baseline, eval_pairs=512, graded_final=graded_final, graded_baseline=graded_baseline
    )

    assert beats["spearman"] is False


def test_beats_untrained_gate_spearman_past_margin_with_sane_baseline_passes() -> None:
    """Positive control for the two tests above: a spearman win clearly past the margin,
    with a sane recall@1 baseline behind it, must pass -- the gate rejects bad evidence,
    not every claim."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.pretrain import _beats_untrained_gate

    final = {"recall@1": 0.9, "recall@10": 0.99}
    baseline = {"recall@1": 0.4, "recall@10": 0.8}
    graded_baseline = {"spearman": 0.10}
    graded_final = {"spearman": 0.50}

    chance, beats = _beats_untrained_gate(
        final, baseline, eval_pairs=512, graded_final=graded_final, graded_baseline=graded_baseline
    )

    assert chance["spearman"] == 0.0
    assert beats["spearman"] is True


def test_beats_untrained_gate_spearman_inherits_recall1_baseline_insanity() -> None:
    """`baseline_sane` is computed once, off `recall@1`'s baseline, and gates spearman
    too (see the docstring's "SPEARMAN SHARES THE GATE, NOT THE METRIC" section): a
    `recall@1` baseline this broken makes the whole untrained measurement suspect, graded
    included, even when the graded numbers alone look like a clean win."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.pretrain import _beats_untrained_gate

    final = {"recall@1": 0.748046875, "recall@10": 0.94921875}
    baseline = {"recall@1": 0.0, "recall@10": 0.021484375}  # the real receipt's numbers
    graded_baseline = {"spearman": 0.05}
    graded_final = {"spearman": 0.60}  # a clean-looking win on the graded numbers alone

    _, beats = _beats_untrained_gate(
        final, baseline, eval_pairs=512, graded_final=graded_final, graded_baseline=graded_baseline
    )

    assert beats["spearman"] is False


def test_beats_untrained_gate_omits_spearman_when_no_graded_set() -> None:
    """A region with no graded eval set (`graded_final`/`graded_baseline` left at their
    `None` default, matching `pretrain_region`'s `{}` when `graded` is empty) gets no
    `"spearman"` key at all -- not a vacuous `True` or `False`."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from cogsyndelta.regions.pretrain import _beats_untrained_gate

    final = {"recall@1": 0.75, "recall@10": 0.95}
    baseline = {"recall@1": 1 / 512, "recall@10": 10 / 512}

    chance, beats = _beats_untrained_gate(final, baseline, eval_pairs=512)

    assert "spearman" not in chance
    assert "spearman" not in beats


def test_pretrain_region_receipt_records_chance_and_untrained_baseline_seed(
    tmp_path: Path,
) -> None:
    """Wired-through control: the gate above is exercised through `pretrain_region`
    itself, not only as a bare function -- see the DEFECT 5 continuation's reasoning for
    why that distinction catches mutations the bare-function tests cannot. Also pins the
    two receipt fields R6 adds: `chance` (the floor the gate computed against) and
    `untrained_baseline_seed` (config.seed doubles as this already, but every text
    region sharing seed=0 makes that easy to misread as three independent
    measurements -- see the module docstring's investigation note -- so it is named
    explicitly rather than left implicit in `config`)."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    pytest.importorskip("pyarrow", reason="train group not installed")
    from cogsyndelta.regions.pretrain import pretrain_region

    cfg = _tiny_pretrain_cfg(tmp_path, graded_shards=[], graded_name="")

    receipt = pretrain_region(cfg)

    assert receipt["untrained_baseline_seed"] == cfg.seed
    assert receipt["chance"]["recall@1"] == pytest.approx(1 / receipt["held_out"]["n_pairs"])
    assert set(receipt["beats_untrained"]) == {"recall@1", "recall@10"}


def test_pretrain_region_receipt_carries_code_revision_and_trainer_defaults(
    tmp_path: Path,
) -> None:
    """Wired-through control for `_receipt.write_receipt`, mirroring the test above: a
    real `pretrain_region` run must carry `code_revision` (this checkout's real,
    non-"unknown" SHA -- `pretrain_region` itself never patches `write_receipt`'s default
    `capture`, so a mutation that dropped the `write_receipt` call site, or reverted to
    the old inline `path.write_text(...)`, leaves this key absent) and `config.
    trainer_defaults` matching this run's own `steps`/`batch_size`/`lr`/`bf16`/`max_len`."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    pytest.importorskip("pyarrow", reason="train group not installed")
    from cogsyndelta.regions.pretrain import pretrain_region

    cfg = _tiny_pretrain_cfg(tmp_path, graded_shards=[], graded_name="")

    receipt = pretrain_region(cfg)

    assert set(receipt["code_revision"]) == {"git_sha", "dirty", "branch", "describe"}
    assert receipt["code_revision"]["git_sha"] != "unknown"
    assert isinstance(receipt["code_revision"]["dirty"], bool)
    assert receipt["config"]["trainer_defaults"] == {
        "steps": cfg.steps,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "bf16": cfg.bf16,
        "max_len": cfg.max_len,
    }


# ---------------------------------------------------------------------------------------
# DEFECT 7 -- feat/metrics-v2's own review found two guards this project shipped with NO
# proof in this file, this file's own charter ("a guard with no failing-case test is a
# comment with a function signature"): the csd-metrics/v2 refuse-predicate (`compare()`,
# g7-latent-eval-metrics.md §3.3) and the `beats_untrained` -> `beats_untrained_eval`
# rename in `scripts/csd-benchmark.py` (g7 §3.2). Real, non-vacuous proofs already
# existed -- `tests/test_eval_metrics.py::test_compare_refuses_on_each_identity_key_independently`
# and `tests/test_benchmark_metrics_v2_receipt.py::test_fp32_receipt_gates_are_renamed_and_not_anisotropic_is_gone`
# -- but neither lived here. The review named two of that first test's OWN mutation
# proofs as tautological (they rename keys in a local dict and assert the local dict was
# renamed, proving nothing about production code): `test_pre_rename_gate_shape_would_fail_this_files_own_assertions`
# and `test_receipt_missing_the_renamed_fields_would_fail_the_positive_assertions`, both
# in tests/test_benchmark_metrics_v2_receipt.py. The tests below are the SAME assertions
# as the real ones, mirrored here rather than rewritten, driven against production code
# (`compare()` itself; a real tiny CPU pretrain + `csd-benchmark.py`'s own
# `benchmark_region`), not against a local stand-in dict.
# ---------------------------------------------------------------------------------------


def _defect7_identity(**overrides: object) -> MetricIdentity:
    """A baseline `MetricIdentity` every field of which matches its own defaults --
    each test below overrides exactly ONE field so a refusal can be pinned to it.
    Mirrors `tests/test_eval_metrics.py::_identity`."""
    base: dict[str, object] = {
        "metrics_schema": "csd-metrics/v2",
        "corpus_fingerprint": "fp-code-holdout-abc123",
        "fingerprint_scheme": "csd-corpus-fp/v2",
        "battery_id": "eval_holdout",
        "k": None,
        "pooling": "pooled_both",
        "checkpoint_sha256": "127adeba58e39a1a0211e185adad74586d08b9fd0bdda8e2da4f5614f49ad8e1",
        "region": "code",
        "git_sha": "a7694090903664bc256b4b96d998b37cacd316cf",
        "seed": 0,
    }
    base.update(overrides)
    return MetricIdentity(**base)  # type: ignore[arg-type]


def _defect7_group(values: dict[str, float], **identity_overrides: object) -> MetricGroup:
    return MetricGroup(identity=_defect7_identity(**identity_overrides), values=values)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("field", "candidate_override", "expected_receipt_name"),
    [
        ("metrics_schema", {"metrics_schema": "csd-metrics/v1"}, "metrics_schema"),
        ("corpus_fingerprint", {"corpus_fingerprint": "fp-different"}, "corpus.fingerprint"),
        (
            "fingerprint_scheme",
            {"fingerprint_scheme": "csd-corpus-fp/v1"},
            "corpus.fingerprint_scheme",
        ),
        ("battery_id", {"battery_id": "eval_quantized_holdout"}, "battery_id"),
        ("k", {"k": 10}, "k"),
        ("pooling", {"pooling": "anchor"}, "pooling"),
        (
            "checkpoint_sha256",
            {"checkpoint_sha256": "deadbeef" * 8},
            "artifacts.checkpoint_sha256",
        ),
        ("region", {"region": "retrieve"}, "region / producer.component"),
        ("git_sha", {"git_sha": "0" * 40}, "code_revision.git_sha"),
        ("seed", {"seed": 1}, "seed"),
    ],
)
def test_compare_refuse_predicate_catches_every_identity_key(
    field: str, candidate_override: dict[str, object], expected_receipt_name: str
) -> None:
    """MUTATION PROOF: `compare()` (`src/cogsyndelta/eval/metrics.py`) is csd-metrics/v2's
    refuse-predicate -- the v1 version diffed whatever keys two dicts happened to share
    and said nothing about whether they described the same measurement at all, which is
    exactly what let an in-memory plan's `quantized_metric` get read next to a
    packed-artifact eval-quantized `rank.recall@1` as though interchangeable. Stubbing
    the identity loop (`for field_name in MetricIdentity._fields:` -> `for field_name in
    ():`) makes every one of these 10 parametrisations fail: VERIFIED against a scratch
    mutant copy of this tree, not asserted from reading the code alone."""
    result = compare(
        _defect7_group({"recall@1": 0.99}),
        _defect7_group({"recall@1": 0.99}, **candidate_override),
        lower_is_better=set(),
    )
    assert result["refused"] is True, f"expected a refusal when {field!r} differs"
    assert result["mismatched_key"] == expected_receipt_name
    assert result["reason"]  # non-empty, human-readable


@pytest.mark.cpu
def test_compare_refuse_predicate_does_not_diff_shared_keys_across_batteries() -> None:
    """MUTATION PROOF, the specific historical defect this predicate exists to close: a
    `battery_id` mismatch ALONE must refuse even though both sides share the metric name
    `recall@1` with a plausible-looking value -- diffing shared keys across two
    different batteries is exactly what let a `quantized_metric` (an in-memory plan's
    recall@1) get read next to an eval-quantized `rank.recall@1` (the packed artifact's)
    as though they were interchangeable (MM §4)."""
    result = compare(
        _defect7_group({"recall@1": 0.9902}, battery_id="train_holdout"),
        _defect7_group({"recall@1": 0.9902}, battery_id="eval_holdout"),
        lower_is_better=set(),
    )
    assert result["refused"] is True
    assert result["mismatched_key"] == "battery_id"
    assert "metrics" not in result


@pytest.mark.cpu
def test_beats_untrained_eval_gate_name_survives_a_real_pretrain_and_benchmark_run(
    tmp_path: Path,
) -> None:
    """MUTATION PROOF: the g7 §3.2 rename in `scripts/csd-benchmark.py`
    (`"beats_untrained_eval":` -> `"beats_untrained":`) is proved by a REAL tiny CPU
    pretrain + benchmark run, not a hand-built receipt -- the rename is a literal string
    in a dict LITERAL inside the production script, so a stand-in receipt built by hand
    would just assert its own hard-coded key back at itself (the review named exactly
    this vacuity in two tests this branch shipped, elsewhere). Reverting the rename
    makes `rec.gates` carry the OLD `beats_untrained` key again, which collides with a
    TRAINING receipt's separate `beats_untrained_train` predicate under the same
    English name: VERIFIED against a scratch mutant copy of this tree with both
    occurrences of `"beats_untrained_eval":` reverted to `"beats_untrained":`, which
    fails this assertion."""
    pytest.importorskip("torch", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    pytest.importorskip("pyarrow", reason="train group not installed")

    from cogsyndelta.regions.pretrain import PretrainConfig, pretrain_region
    from cogsyndelta.regions.text_encoder import TextEncoderConfig
    from tests.test_benchmark_metrics_v2_receipt import (
        _build_pairs_parquet,
        _build_tokenizer,
    )
    from tests.test_benchmark_metrics_v2_receipt import (
        bench as v2_bench,
    )

    region = "guards7-test"
    tok_path = tmp_path / "tokenizer.json"
    shard_path = tmp_path / "pairs.parquet"
    _build_tokenizer(tok_path, 40)
    _build_pairs_parquet(shard_path, 40)
    receipts_dir = tmp_path / "receipts"

    cfg = PretrainConfig(
        region=region,
        pair_columns=("anchor", "positive"),
        shards=[str(shard_path)],
        steps=2,
        batch_size=4,
        holdout_pairs=4,
        eval_every=2,
        checkpoint_every=2,
        max_len=16,
        seed=3,
        device="cpu",
        encoder=TextEncoderConfig(dim=8, depth=1, n_heads=2, max_len=16),
        tokenizer_path=str(tok_path),
        out_dir=str(receipts_dir),
    )
    pretrain_region(cfg)

    class _Entry:
        sources: ClassVar = [("pairs.parquet", ("anchor", "positive"), 0)]
        root = tmp_path

    def fake_regions_spec() -> dict:
        return {
            "REGIONS": {},
            "_shards": lambda *a, **k: [str(shard_path)],
            "region_spec": lambda name: _Entry(),
        }

    orig_spec = v2_bench._regions_spec
    v2_bench._regions_spec = fake_regions_spec
    try:
        rec = v2_bench.benchmark_region(region, tmp_path)
    finally:
        v2_bench._regions_spec = orig_spec

    assert rec is not None
    assert "beats_untrained_eval" in rec.gates
    assert "beats_untrained" not in rec.gates


# ---------------------------------------------------------------------------------------
# DEFECT 8 / G26 -- the held-out split was a function of the training seed, so seed-0
# and seed-1 cells were different test sets (reason-region diagnosis 2026-09-05 E0).
# Membership is now a hashed manifest; training refuses a fingerprint/sha mismatch or a
# held-out item in a training batch; the benchmark refuses a receipt whose split.sha256
# is not the manifest it is scoring. The full suite lives in tests/test_splits.py (stdlib)
# and tests/test_split_manifests.py (parquet). The tests below are the failing-case
# proofs this file's charter requires.
# ---------------------------------------------------------------------------------------


def test_g26_doctored_split_manifest_is_refused(tmp_path: Path) -> None:
    """A split file with one holdout id swapped must refuse, not train on a silent
    different eval set. Mirrors tests/test_split_manifests.py; kept here so a review
    that only reads this file still sees G26 fire."""
    pytest.importorskip("pyarrow", reason="train group not installed")
    pytest.importorskip("tokenizers", reason="train group not installed")
    from tests.test_split_manifests import test_doctored_manifest_refuses

    test_doctored_manifest_refuses(tmp_path)


def test_g26_held_out_leak_is_refused() -> None:
    from cogsyndelta.splits import SplitGuardError, assert_no_held_out_in_pairs

    holdout = [("the held out question", "the held out answer")]
    train = [("unrelated q", "unrelated a"), ("the held out question", "the held out answer")]
    with pytest.raises(SplitGuardError, match="held-out item"):
        assert_no_held_out_in_pairs(holdout, train)


# ---------------------------------------------------------------------------------------
# DEFECT 9 / G37 -- a representation-geometry comparison (fp32 latents vs quantized
# latents, per docs/design/evidence/visual-ptq-sensitivity-2026-09-06/README.md) is only
# a measurement of quantization if BOTH sides were computed over the same items, in the
# same order, from the same fp32 checkpoint AND the same quantized artifact.
# `cogsyndelta.eval.geometry.verify_geometry_reference` is the guard that refuses a
# comparison across two different item sets, fp32 draws or packed artifacts (a stale
# cached fp32 pass, a split rebuilt under a different seed, a truncated batch on one
# side, a re-trained checkpoint compared against an artifact quantized from a different
# draw, two artifacts quantized from the same checkpoint at different bit widths) before
# `compute_geometry` ever runs -- the same shape G26 already fixed for the text held-out
# split, applied here to a pairwise latent comparison instead of a train/eval partition.
#
# Guard number: G37, not G27. `docs/design/INTERCONNECT-MODULE-SPEC.md` Table 8 (merged
# to main, PR #67) reserves G27 through G36 for the interconnect module's own guards --
# this guard's number was picked from the same registry (`src/cogsyndelta/splits.py`'s
# G26, then the next free number) before that PR landed and had to move. The next new
# guard after this one is G38.
# ---------------------------------------------------------------------------------------

#: Shared valid checkpoint/artifact shas for the tests below that are NOT exercising the
#: checkpoint_sha256 / quantized_sha256 fields themselves -- every `GeometryReference`
#: constructor call needs a value for all four fields now, and reusing one pair of
#: constants keeps the split/n_items-focused tests' intent legible (the pair is
#: irrelevant to what those tests assert, so it stays fixed and out of the way).
_SAME_CHECKPOINT_SHA = "checkpoint" * 4
_SAME_QUANTIZED_SHA = "artifact" * 4


def test_g37_mismatched_split_reference_is_refused() -> None:
    """The fp32 side was scored on one split; the quantized side on another (a stale
    cached fp32 pass reused against a re-built holdout is exactly this shape). Same
    item COUNT, checkpoint and artifact sha on both sides, so a guard keyed on any of
    those alone would miss it."""
    from cogsyndelta.eval.geometry import (
        GeometryReference,
        GeometryReferenceError,
        verify_geometry_reference,
    )

    fp32_reference = GeometryReference(
        split_sha256="split-a" * 4,
        n_items=5400,
        checkpoint_sha256=_SAME_CHECKPOINT_SHA,
        quantized_sha256=_SAME_QUANTIZED_SHA,
    )
    quantized_reference = GeometryReference(
        split_sha256="split-b" * 4,
        n_items=5400,
        checkpoint_sha256=_SAME_CHECKPOINT_SHA,
        quantized_sha256=_SAME_QUANTIZED_SHA,
    )

    with pytest.raises(GeometryReferenceError, match="G37") as exc_info:
        verify_geometry_reference(fp32_reference, quantized_reference)
    assert "split_sha256" in str(exc_info.value)


def test_g37_mismatched_item_count_is_refused() -> None:
    """Same split identity string, different row counts -- a truncated or padded batch
    on one side. A guard keyed on `split_sha256` alone would miss it."""
    from cogsyndelta.eval.geometry import (
        GeometryReference,
        GeometryReferenceError,
        verify_geometry_reference,
    )

    fp32_reference = GeometryReference(
        split_sha256="same-split",
        n_items=5400,
        checkpoint_sha256=_SAME_CHECKPOINT_SHA,
        quantized_sha256=_SAME_QUANTIZED_SHA,
    )
    quantized_reference = GeometryReference(
        split_sha256="same-split",
        n_items=5399,
        checkpoint_sha256=_SAME_CHECKPOINT_SHA,
        quantized_sha256=_SAME_QUANTIZED_SHA,
    )

    with pytest.raises(GeometryReferenceError, match="G37") as exc_info:
        verify_geometry_reference(fp32_reference, quantized_reference)
    assert "n_items" in str(exc_info.value)


def test_g37_mismatched_checkpoint_sha_is_refused() -> None:
    """The fp32 side was loaded from one fp32 checkpoint; the quantized side's
    artifact claims lineage from a different one -- a re-trained fp32 checkpoint
    compared against a stale packed artifact (or vice versa) is exactly this shape.
    Same split, item count and quantized-artifact sha on both sides, so a guard that
    does not check `checkpoint_sha256` would miss it (the hole a second review found:
    previously `GeometryReference` carried only `split_sha256`/`n_items`, and
    `checkpoint_sha256` was threaded into the receipt's `provenance` but never
    compared between the two sides)."""
    from cogsyndelta.eval.geometry import (
        GeometryReference,
        GeometryReferenceError,
        verify_geometry_reference,
    )

    fp32_reference = GeometryReference(
        split_sha256="same-split",
        n_items=5400,
        checkpoint_sha256="checkpoint-a" * 4,
        quantized_sha256=_SAME_QUANTIZED_SHA,
    )
    quantized_reference = GeometryReference(
        split_sha256="same-split",
        n_items=5400,
        checkpoint_sha256="checkpoint-b" * 4,
        quantized_sha256=_SAME_QUANTIZED_SHA,
    )

    with pytest.raises(GeometryReferenceError, match="G37") as exc_info:
        verify_geometry_reference(fp32_reference, quantized_reference)
    assert "checkpoint_sha256" in str(exc_info.value)


def test_g37_matching_checkpoint_sha_is_not_refused() -> None:
    """Positive control for `checkpoint_sha256` specifically: when every field,
    including the new one, genuinely agrees, the guard must not fire. Without this,
    the refusal test above could pass for the wrong reason (e.g. a guard that always
    raises regardless of which field it inspects)."""
    from cogsyndelta.eval.geometry import GeometryReference, verify_geometry_reference

    fp32_reference = GeometryReference(
        split_sha256="same-split",
        n_items=5400,
        checkpoint_sha256="checkpoint-a" * 4,
        quantized_sha256=_SAME_QUANTIZED_SHA,
    )
    quantized_reference = GeometryReference(
        split_sha256="same-split",
        n_items=5400,
        checkpoint_sha256="checkpoint-a" * 4,
        quantized_sha256=_SAME_QUANTIZED_SHA,
    )
    verify_geometry_reference(fp32_reference, quantized_reference)


def test_g37_mismatched_quantized_artifact_sha_is_refused() -> None:
    """Two artifacts quantized from the SAME fp32 checkpoint at different bit widths
    (the concrete shape this guards against): same split, item count and checkpoint
    sha on both sides, but the quantized side's latents were actually scored from a
    different packed file than the one this comparison names."""
    from cogsyndelta.eval.geometry import (
        GeometryReference,
        GeometryReferenceError,
        verify_geometry_reference,
    )

    fp32_reference = GeometryReference(
        split_sha256="same-split",
        n_items=5400,
        checkpoint_sha256=_SAME_CHECKPOINT_SHA,
        quantized_sha256="artifact-3bit" * 4,
    )
    quantized_reference = GeometryReference(
        split_sha256="same-split",
        n_items=5400,
        checkpoint_sha256=_SAME_CHECKPOINT_SHA,
        quantized_sha256="artifact-8bit" * 4,
    )

    with pytest.raises(GeometryReferenceError, match="G37") as exc_info:
        verify_geometry_reference(fp32_reference, quantized_reference)
    assert "quantized_sha256" in str(exc_info.value)


def test_g37_a_genuinely_matching_reference_pair_is_not_refused() -> None:
    """Negative control: without this, "the guard fires" could just mean "always"."""
    from cogsyndelta.eval.geometry import GeometryReference, verify_geometry_reference

    reference = GeometryReference(
        split_sha256="the-real-split",
        n_items=5400,
        checkpoint_sha256="the-real-checkpoint",
        quantized_sha256="the-real-artifact",
    )
    verify_geometry_reference(
        reference,
        GeometryReference(
            split_sha256="the-real-split",
            n_items=5400,
            checkpoint_sha256="the-real-checkpoint",
            quantized_sha256="the-real-artifact",
        ),
    )


def test_g37_end_to_end_through_compute_geometry_would_compare_the_wrong_items() -> None:
    """The regression this guard exists to stop, made concrete: without the guard,
    nothing would have stopped `compute_geometry` from running on two latent matrices
    that only coincidentally share a shape. This test proves the guard is what a caller
    is expected to run FIRST -- `verify_geometry_reference` raising before
    `compute_geometry` is ever reached -- by reproducing exactly the call order
    `scripts/csd-benchmark.py`'s wiring uses."""
    import torch

    from cogsyndelta.eval.geometry import (
        GeometryReference,
        GeometryReferenceError,
        compute_geometry,
        verify_geometry_reference,
    )

    fp32_latents = torch.randn(20, 4)
    quantized_latents = torch.randn(20, 4)  # same SHAPE, not the same split
    fp32_reference = GeometryReference(
        split_sha256="split-a" * 4,
        n_items=20,
        checkpoint_sha256=_SAME_CHECKPOINT_SHA,
        quantized_sha256=_SAME_QUANTIZED_SHA,
    )
    quantized_reference = GeometryReference(
        split_sha256="split-b" * 4,
        n_items=20,
        checkpoint_sha256=_SAME_CHECKPOINT_SHA,
        quantized_sha256=_SAME_QUANTIZED_SHA,
    )

    with pytest.raises(GeometryReferenceError, match="G37"):
        verify_geometry_reference(fp32_reference, quantized_reference)
        compute_geometry(fp32_latents, quantized_latents)  # never reached


# ---------------------------------------------------------------------------------------
# DEFECT 10 / G38+G39 -- mined hard negatives could be their own pair's positive, and a run could
# train on negatives mined against a DIFFERENT corpus, with nothing on disk saying so.
#
# G38 (self-positive disjointness) and G39 (mining provenance) are the two guards
# PREREG-RETRIEVAL-NEGATIVES-2026-09-06 rev 3 section 4.2 pre-registers for the mined
# arm. Both refuse the RUN, not the batch, and both are proven here by construction:
# each test builds a mined negative set that is wrong in exactly one way and asserts the
# refusal, including through `pretrain_region` itself so the guard is shown to be WIRED
# rather than merely present.
# ---------------------------------------------------------------------------------------


def _mining_fixture() -> tuple[list[tuple[str, str]], list[tuple[str, list[tuple[str, str]]]]]:
    """A miniature single-source union: distinct anchors, distinct positives."""
    pairs = [
        (f"question number {i} about topic {i}", f"answer body {i} explains topic {i}")
        for i in range(8)
    ]
    return pairs, [("primary", pairs)]


def _faithful_manifest(tmp_path: Path, m: int = 2) -> tuple[dict, dict, list, list]:
    """Mine the fixture honestly and stamp a manifest for it.

    Returns:
        `(manifest, pools, pairs, sources)` -- everything a verify call needs.
    """
    from cogsyndelta.regions import _mining

    pairs, sources = _mining_fixture()
    qrels_path = tmp_path / "train.parquet"
    qrels_path.write_bytes(b"the qrels artefact G39 hashes")
    empty_qrels: dict[str, list] = {"query_id": [], "doc_id": [], "query": [], "passage": []}
    result = _mining.mine_and_audit(
        train_pairs=pairs, sources=sources, qrels=empty_qrels, fiqa_source="primary", m=m
    )
    manifest = _mining.build_manifest(
        region="memory",
        corpus_fingerprint="6fc0cf23ff8591ff2241278f82c001d2",
        result=result,
        train_pairs=pairs,
        qrels_path=qrels_path,
        m=m,
    )
    return manifest, _mining.build_pools(sources), pairs, sources


def test_g38_fires_when_a_mined_negative_is_its_own_pairs_positive(tmp_path: Path) -> None:
    """The construction G38 exists to refuse: a "hard negative" that IS the gold.

    Mining excludes only the pair's own positive, so a bug in that exclusion -- an index
    off by one, a normalisation that stops matching, a hand-edited manifest -- produces a
    negative set in which some anchors are trained to rank their own answer DOWN. The
    loss curve looks fine.
    """
    from cogsyndelta.regions import _mining

    manifest, pools, pairs, _ = _faithful_manifest(tmp_path)
    negatives = [list(row) for row in manifest["negatives"]]

    # The honest mining result passes.
    source_of_pair = ["primary"] * len(pairs)
    _mining.assert_self_positive_disjoint(pairs, negatives, source_of_pair, pools)

    # Now point pair 3's first negative at its own positive, which IS in the pool.
    own = pools["primary"].texts.index(pairs[3][1])
    negatives[3][0] = own
    with pytest.raises(_mining.MiningGuardError, match="G38"):
        _mining.assert_self_positive_disjoint(pairs, negatives, source_of_pair, pools)


def test_g38_catches_a_reformatted_copy_of_the_positive(tmp_path: Path) -> None:
    """Exact string equality would pass on the copy a corpus is most likely to hold.

    G38 keys on the `pair_exact` normalisation (whitespace and case), so a positive that
    reappears in the pool with different spacing is still refused.
    """
    from cogsyndelta.regions import _mining

    pairs = [("anchor one", "The Answer Body"), ("anchor two", "another answer")]
    pool_texts = ("the   answer   body", "another answer")
    pools = {
        "primary": _mining.MiningPool(
            source="primary", texts=pool_texts, sha256=_mining.pool_sha256(pool_texts)
        )
    }
    with pytest.raises(_mining.MiningGuardError, match="G38"):
        _mining.assert_self_positive_disjoint(pairs, [[0], [1]], ["primary", "primary"], pools)


@pytest.mark.parametrize(
    ("field", "break_it"),
    [
        ("corpus fingerprint", lambda m: m["corpus"].__setitem__("fingerprint", "0" * 32)),
        ("pool sha256", lambda m: m["pools"]["primary"].__setitem__("sha256", "0" * 64)),
        ("qrels sha256", lambda m: m["qrels"].__setitem__("sha256", "0" * 64)),
        ("bm25 k1", lambda m: m["bm25"].__setitem__("k1", 1.2)),
        ("m", lambda m: m.__setitem__("negatives_per_anchor", 7)),
        ("train pair sequence", lambda m: m["train_pairs"].__setitem__("sha256", "0" * 64)),
        ("missing corpus block", lambda m: m.pop("corpus")),
        ("missing schema", lambda m: m.pop("schema")),
    ],
)
def test_g39_refuses_a_manifest_that_disagrees_with_the_run(
    tmp_path: Path, field: str, break_it
) -> None:
    """Every field section 4.2 pins, broken one at a time.

    Each is checked against a value computed from the RUN's own corpus, never against
    another field of the same manifest -- a self-consistent manifest for a different
    corpus is precisely what this refuses. The parametrisation includes two ABSENT
    fields, because a guard that treats a missing field as "nothing to check" fails open.
    """
    from cogsyndelta.regions import _mining

    manifest, pools, pairs, _ = _faithful_manifest(tmp_path)
    qrels_sha = _mining.file_sha256(tmp_path / "train.parquet")

    # The faithful manifest verifies.
    _mining.verify_manifest(
        manifest,
        corpus_fingerprint="6fc0cf23ff8591ff2241278f82c001d2",
        pools=pools,
        qrels_sha256=qrels_sha,
        train_pairs=pairs,
        m=2,
    )

    break_it(manifest)
    # Re-stamp the payload hash, so this test proves the FIELD check fires rather than
    # only the tamper-detection hash.
    if "sha256" in manifest:
        manifest["sha256"] = _mining.manifest_payload_sha256(manifest)
    with pytest.raises(_mining.MiningGuardError, match="G39"):
        _mining.verify_manifest(
            manifest,
            corpus_fingerprint="6fc0cf23ff8591ff2241278f82c001d2",
            pools=pools,
            qrels_sha256=qrels_sha,
            train_pairs=pairs,
            m=2,
        )


def test_g39_refuses_a_manifest_whose_payload_was_edited(tmp_path: Path) -> None:
    """Swapping negatives in an otherwise-correct manifest, without re-stamping its hash."""
    from cogsyndelta.regions import _mining

    manifest, pools, pairs, _ = _faithful_manifest(tmp_path)
    manifest["negatives"][0] = list(reversed(manifest["negatives"][0]))
    with pytest.raises(_mining.MiningGuardError, match="payload sha256"):
        _mining.verify_manifest(
            manifest,
            corpus_fingerprint="6fc0cf23ff8591ff2241278f82c001d2",
            pools=pools,
            qrels_sha256=_mining.file_sha256(tmp_path / "train.parquet"),
            train_pairs=pairs,
            m=2,
        )


def test_g39_refuses_a_missing_manifest(tmp_path: Path) -> None:
    """No manifest at all is a refusal, not an unmined run."""
    from cogsyndelta.regions import _mining

    with pytest.raises(_mining.MiningGuardError, match="G39"):
        _mining.load_manifest(tmp_path / "not-here.json")


# ---------------------------------------------------------------------------------------
# DEFECT 11 / G40 -- a round could be launched with the WRONG OBJECTIVE and complete
# normally. PREREG-RETRIEVAL-NEGATIVES-2026-09-06 rev 3 section 2.1 requires both
# auxiliary weights at 0.0 in every arm, and `scripts/csd-train-all.py` hard-coded
# `memory`'s 0.1/0.1 with no override: an arm started there would have trained the
# production objective, finished, and written a plausible receipt. Nothing failed.
#
# The launcher is fixed, but a launch-time check only protects the launcher somebody
# remembered to fix. G40 sits where the number is READ: `cogsyndelta.eval.prereg` refuses
# to grade a receipt whose MEASURED weights -- recorded at the loss site, not parsed from
# arguments -- are not the pre-registration's declared ones.
# ---------------------------------------------------------------------------------------


def _graded_receipt(token_weight: float, decorr_weight: float, negative_set: str = "in_batch"):
    """A receipt shaped like `pretrain_region`'s, with the measured weights dialled."""
    return {
        "region": "memory",
        "objective_weights": {
            "declared": {"token_loss_weight": 0.0, "decorr_weight": 0.0},
            "measured": {
                "token_loss_weight": token_weight,
                "decorr_weight": decorr_weight,
            },
            "values_seen": [[token_weight, decorr_weight]],
            "steps_measured": 4000,
            "read": "at the loss site, per step",
        },
        "negatives": {"set": negative_set, "bank_size": 0},
    }


def test_g40_refuses_to_grade_a_receipt_trained_at_the_production_weights() -> None:
    """The failing arm: 0.1/0.1 measured against a pre-registration declaring 0.0/0.0.

    This is the run the old launcher would have produced. Its loss curve, its receipt and
    its metrics are all well-formed; the only thing wrong with it is that it answers a
    different question, which is exactly why a human reading the number would not catch
    it.
    """
    from cogsyndelta.eval.prereg import E_N_ARMS, PreregGuardError, assert_receipt_matches_arm

    # The honest arm grades.
    assert_receipt_matches_arm(_graded_receipt(0.0, 0.0), E_N_ARMS["C"])

    with pytest.raises(PreregGuardError, match="G40"):
        assert_receipt_matches_arm(_graded_receipt(0.1, 0.1), E_N_ARMS["C"])
    # One weight is enough: the terms are separate, and 0.1 * L_decorr alone carries the
    # 98.8% of the rank change the control-armed experiment measured.
    with pytest.raises(PreregGuardError, match="G40"):
        assert_receipt_matches_arm(_graded_receipt(0.0, 0.1), E_N_ARMS["C"])


def test_g40_reads_the_measured_weights_not_the_declared_ones() -> None:
    """A receipt whose config says 0.0 while its loss used 0.1 is refused.

    This is the case a check on the parsed arguments cannot see, and it is the reason the
    weights are collected at the multiplication site: `declared` here is 0.0/0.0 -- the
    config was correct -- and the run still trained the wrong objective.
    """
    from cogsyndelta.eval.prereg import E_N_ARMS, PreregGuardError, assert_receipt_matches_arm

    receipt = _graded_receipt(0.1, 0.1)
    assert receipt["objective_weights"]["declared"] == {
        "token_loss_weight": 0.0,
        "decorr_weight": 0.0,
    }
    with pytest.raises(PreregGuardError, match="MEASURED"):
        assert_receipt_matches_arm(receipt, E_N_ARMS["C"])


@pytest.mark.parametrize(
    ("case", "damage"),
    [
        ("no block at all", lambda r: r.pop("objective_weights")),
        ("measured is null", lambda r: r["objective_weights"].__setitem__("measured", None)),
        (
            "weights changed mid-run",
            lambda r: r["objective_weights"].update(
                {"measured": None, "values_seen": [[0.0, 0.0], [0.1, 0.1]]}
            ),
        ),
        (
            "measured is unreadable",
            lambda r: r["objective_weights"].__setitem__("measured", {"token_loss_weight": 0.0}),
        ),
    ],
)
def test_g40_fails_closed_on_a_receipt_that_cannot_say_what_it_trained(case, damage) -> None:
    """Absence is a refusal, not a pass.

    A receipt written before this field existed, one where no step ran, and one whose
    weights moved mid-run all describe runs whose objective is unknown. Grading them
    "because there is nothing to check" is how an unmeasured run passes as a measured one.
    """
    from cogsyndelta.eval.prereg import E_N_ARMS, PreregGuardError, assert_receipt_matches_arm

    receipt = _graded_receipt(0.0, 0.0)
    damage(receipt)
    with pytest.raises(PreregGuardError, match="G40"):
        assert_receipt_matches_arm(receipt, E_N_ARMS["C"])


def test_g40_refuses_an_arm_whose_negative_set_is_not_its_own() -> None:
    """The round changes ONE variable, so a T1 receipt with an in-batch denominator is
    not T1 -- it is the control wearing T1's name."""
    from cogsyndelta.eval.prereg import E_N_ARMS, PreregGuardError, assert_receipt_matches_arm

    assert_receipt_matches_arm(_graded_receipt(0.0, 0.0, negative_set="bank"), E_N_ARMS["T1"])
    with pytest.raises(PreregGuardError, match="G40"):
        assert_receipt_matches_arm(
            _graded_receipt(0.0, 0.0, negative_set="in_batch"), E_N_ARMS["T1"]
        )


def test_g40_stops_the_grader_before_it_produces_a_number() -> None:
    """The refusal has to happen where the RESULT is read, not only where a run starts.

    `grade_contrast` is the whole path from two receipts to a PASS-A verdict; this asserts
    a wrong-objective treatment arm cannot get a bound out of it, however it was launched.
    """
    from cogsyndelta.eval.prereg import E_N_ARMS, PreregGuardError, grade_contrast

    def with_per_query(receipt, values):
        receipt["retrieval"] = {
            "full_pool": {
                "query_ids": [f"q{i}" for i in range(len(values))],
                "per_query": {"trained": {"recall@10": values}},
            }
        }
        return receipt

    control = with_per_query(_graded_receipt(0.0, 0.0), [0.0, 1.0, 0.0, 1.0])
    honest = with_per_query(_graded_receipt(0.0, 0.0, negative_set="bank"), [1.0, 1.0, 1.0, 1.0])
    graded = grade_contrast(control=control, treatment=honest, treatment_arm=E_N_ARMS["T1"], seed=0)
    assert graded["bootstrap"]["point_estimate"] == pytest.approx(0.5)

    wrong_objective = with_per_query(
        _graded_receipt(0.1, 0.1, negative_set="bank"), [1.0, 1.0, 1.0, 1.0]
    )
    with pytest.raises(PreregGuardError, match="G40"):
        grade_contrast(
            control=control, treatment=wrong_objective, treatment_arm=E_N_ARMS["T1"], seed=0
        )
