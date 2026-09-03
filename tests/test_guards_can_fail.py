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

import pytest

from cogsyndelta.eval import (
    assert_no_contamination,
    assert_no_pair_contamination,
    pair_contamination_report,
    screen_pair_contamination,
)

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
