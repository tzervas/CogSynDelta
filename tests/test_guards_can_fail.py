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

A happy-path test passes against all four. So each test below builds the exact failing
input and asserts the guard fires, and several assert the PRE-FIX guard would not have --
that pairing is the point of the file. Add to it whenever a guard is added: a guard with
no failing-case test is a comment with a function signature.
"""

from __future__ import annotations

import pytest

from cogsyndelta.eval import (
    assert_no_contamination,
    assert_no_pair_contamination,
    pair_contamination_report,
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


def test_build_splits_refuses_a_corpus_whose_holdout_is_paraphrased_in_train(tmp_path) -> None:
    """End to end: the guard must stop `build_splits`, not merely be callable.

    The corpus is 100 twinned rows. Each twin differs from its sibling ONLY in function
    words on the anchor and is identical on the positive, so:
      - anchor dedup keeps both (their exact normalised anchors differ), which is why the
        old anchor-only guard passed this corpus with `overlap: 0`;
      - whichever twin lands in the holdout has its sibling in train.
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
    with pytest.raises(ValueError, match="pair_content"):
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
