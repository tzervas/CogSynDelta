"""§4.0's token-aware retrain objective: L_token (masked-token prediction, attached at
the FINAL block) and L_decorr (token-position covariance decorrelation), both opt-in via
``PretrainConfig.token_loss_weight``/``decorr_weight``.

THE CENTRAL CLAIM, AND WHY IT IS TESTED THE WAY IT IS
§4.0 exists because mean-pooled InfoNCE gives every position the SAME gradient (the
pooled vector's), so the pre-pool token surface stays close to whatever a random encoder
already produced -- that is what W1 measured (bet dead, PR ratios 0.66x-1.30x) and W1d
confirmed. The claim under test is that turning the two terms on changes that: the
FINAL block's token-global participation-ratio rank should rise relative to an otherwise
identical run with them off.

``test_token_aware_terms_raise_the_final_block_token_global_pr_rank`` is deliberately a
COMPARISON, not a threshold on one run: it trains the SAME corpus/seed/architecture twice,
terms on and terms off, and asserts the "on" arm's rank is higher. A no-op implementation
of either term (the weight parsed and stored but never added to the loss, or added with a
detached/zero gradient) would make the two arms train identically and this assertion
would FAIL -- which is the point; see that test's docstring for the exact failure mode it
rules out.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from cogsyndelta.eval.benchmark import participation_ratio
from cogsyndelta.regions.pretrain import (
    PretrainConfig,
    _mlm_token_loss,
    _resume_fields,
    _token_decorrelation_loss,
    pretrain_region,
)
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig

pytestmark = pytest.mark.cpu

_VOCAB = [f"w{i}" for i in range(60)]


def _sentence(rng: random.Random, n: int = 8) -> str:
    return " ".join(rng.choice(_VOCAB) for _ in range(n))


def _build_corpus(tmp_path: Path, n_pairs: int) -> tuple[Path, Path]:
    """A tiny corpus with real lexical variety (unlike a fixed template), so the MLM
    head has something non-trivial to predict and the decorrelation term something
    non-degenerate to act on."""
    rng = random.Random(0)  # noqa: S311 -- reproducible test fixture, not cryptography
    anchors = [_sentence(rng) for _ in range(n_pairs)]
    positives = [a + " " + _sentence(rng, 2) for a in anchors]

    tok_path = tmp_path / "tokenizer.json"
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tok.train_from_iterator(anchors + positives, trainer=trainer)
    tok.save(str(tok_path))

    shard_path = tmp_path / "pairs.parquet"
    pq.write_table(pa.table({"anchor": anchors, "positive": positives}), shard_path)
    return tok_path, shard_path


@pytest.fixture
def tiny_cfg(tmp_path: Path) -> PretrainConfig:
    """Terms OFF by default -- callers turn them on with `replace(tiny_cfg, ...)`."""
    tok_path, shard_path = _build_corpus(tmp_path, n_pairs=120)
    return PretrainConfig(
        region="token-aware-test",
        pair_columns=("anchor", "positive"),
        shards=[str(shard_path)],
        steps=120,
        batch_size=16,
        holdout_pairs=32,
        eval_every=60,
        checkpoint_every=0,
        max_len=24,
        seed=0,
        device="cpu",
        bf16=False,
        encoder=TextEncoderConfig(dim=32, depth=2, n_heads=4, max_len=24),
        tokenizer_path=str(tok_path),
        out_dir=str(tmp_path / "run"),
    )


# ---------------------------------------------------------------------------------------
# Unit-level: the two loss functions and the rank statistic in isolation.
# ---------------------------------------------------------------------------------------


def test_participation_ratio_of_a_rank_one_matrix_is_one() -> None:
    """Every row the same direction, different scales -- one live singular value."""
    direction = torch.tensor([1.0, 2.0, -1.0, 0.5])
    x = torch.outer(torch.linspace(1.0, 5.0, 20), direction)
    assert participation_ratio(x) == pytest.approx(1.0, abs=1e-4)


def test_participation_ratio_of_isotropic_noise_approaches_dimension() -> None:
    """Independent Gaussian columns spread mass roughly evenly across all D singular
    values, so PR should sit well above a near-degenerate matrix's."""
    torch.manual_seed(0)
    x = torch.randn(500, 16)
    assert participation_ratio(x) > 10.0


def test_participation_ratio_below_two_rows_is_zero_not_an_error() -> None:
    assert participation_ratio(torch.zeros(1, 4)) == 0.0
    assert participation_ratio(torch.zeros(0, 4)) == 0.0


def test_decorrelation_loss_penalises_correlated_features_more_than_independent_ones() -> None:
    """The whole point of `L_decorr`: two features that are COPIES of each other cost
    more than two independent ones, so minimising it pulls the covariance toward
    diagonal -- which is what raises effective rank."""
    torch.manual_seed(0)
    base = torch.randn(200, 1, 8)
    correlated = base.repeat(1, 1, 2).reshape(200, 1, 16)  # every pair of columns identical
    independent = torch.randn(200, 1, 16)
    mask = torch.ones(200, 1)

    loss_correlated = _token_decorrelation_loss(correlated, mask)
    loss_independent = _token_decorrelation_loss(independent, mask)
    assert loss_correlated.item() > loss_independent.item()


def test_decorrelation_loss_is_zero_with_fewer_than_two_real_positions() -> None:
    h = torch.randn(2, 3, 4)
    mask = torch.tensor([[1, 0, 0], [0, 0, 0]])
    assert _token_decorrelation_loss(h, mask).item() == 0.0


def test_mlm_token_loss_masks_some_real_positions_and_is_differentiable() -> None:
    torch.manual_seed(0)
    cfg = TextEncoderConfig(vocab_size=30, dim=8, depth=1, n_heads=2, max_len=16)
    model = TextEncoder(cfg)
    from cogsyndelta.regions.pretrain import _build_mlm_head

    head, mask_emb = _build_mlm_head(cfg.dim, cfg.vocab_size, torch.device("cpu"))
    ids = torch.randint(1, 30, (4, 10))
    mask = torch.ones(4, 10, dtype=torch.long)

    loss, n_masked = _mlm_token_loss(model, head, mask_emb, ids, mask, mask_prob=0.5)
    assert n_masked > 0
    assert loss.requires_grad
    loss.backward()
    assert mask_emb.grad is not None and mask_emb.grad.abs().sum() > 0


def test_mlm_token_loss_is_a_zero_no_grad_tensor_when_nothing_is_masked() -> None:
    """`mask_prob=0` must not raise or silently pick something to mask anyway."""
    cfg = TextEncoderConfig(vocab_size=10, dim=4, depth=1, n_heads=2, max_len=8)
    model = TextEncoder(cfg)
    from cogsyndelta.regions.pretrain import _build_mlm_head

    head, mask_emb = _build_mlm_head(cfg.dim, cfg.vocab_size, torch.device("cpu"))
    ids = torch.randint(1, 10, (2, 5))
    mask = torch.ones(2, 5, dtype=torch.long)

    loss, n_masked = _mlm_token_loss(model, head, mask_emb, ids, mask, mask_prob=0.0)
    assert n_masked == 0
    assert loss.item() == 0.0
    assert not loss.requires_grad


def test_l_token_gradient_reaches_trunk_parameters_through_the_real_loss_assembly() -> None:
    """B1 direct proof, independent of any rank comparison: with the SAME assembly the
    real training step uses -- `0.5 * (token_loss_a + token_loss_p)` from two real
    `_mlm_token_loss` calls (not a mock, not a re-implementation) -- the trunk's own
    parameters (`model.blocks`) receive nonzero gradient. Detaching `token_loss` -- the
    exact control mutation this test rules out (`token_loss = token_loss.detach()` in
    place of the training loop's assembly line) -- must zero that gradient out entirely.

    Why this exists ALONGSIDE the rank-comparison tests below: a rank comparison can only
    observe L_token's effect indirectly, through a full training run, and B1 showed that
    indirection is exactly where a no-op L_token can hide behind a healthy L_decorr. This
    test needs no training loop and no L_decorr in the picture at all -- it inspects the
    gradient directly, so nothing else can carry the signal.

    A zero-coefficient `anchor` term (`0.0 * sum(p.sum() for p in model.blocks.parameters())`)
    keeps `.backward()` valid even when `token_loss` itself has been detached (a fully
    detached scalar has no `grad_fn`, and `.backward()` on it alone would raise, not
    merely under-count -- `model.pos_embed` will not do for `anchor` either: it is a
    registered BUFFER, not a `Parameter`, so it carries no `grad_fn` on its own). `anchor`
    is built from real trunk `Parameter`s so it always has a live graph, and its `0.0`
    coefficient means it contributes nothing numerically to any parameter's gradient --
    whatever gradient the trunk ends up with is attributable to `token_loss` alone, on or
    off.
    """
    torch.manual_seed(0)
    cfg = TextEncoderConfig(vocab_size=30, dim=8, depth=2, n_heads=2, max_len=16)
    model = TextEncoder(cfg)
    from cogsyndelta.regions.pretrain import _build_mlm_head

    head, mask_emb = _build_mlm_head(cfg.dim, cfg.vocab_size, torch.device("cpu"))
    a_ids = torch.randint(1, 30, (4, 10))
    a_mask = torch.ones(4, 10, dtype=torch.long)
    p_ids = torch.randint(1, 30, (4, 10))
    p_mask = torch.ones(4, 10, dtype=torch.long)

    def trunk_grad_abs_sum(*, detach: bool) -> float:
        model.zero_grad(set_to_none=True)
        token_loss_a, n_masked_a = _mlm_token_loss(model, head, mask_emb, a_ids, a_mask, 0.5)
        token_loss_p, n_masked_p = _mlm_token_loss(model, head, mask_emb, p_ids, p_mask, 0.5)
        assert n_masked_a > 0 and n_masked_p > 0
        # The training loop's own assembly line (regions/pretrain.py's training step):
        token_loss = 0.5 * (token_loss_a + token_loss_p)
        if detach:
            token_loss = token_loss.detach()  # the B1 control mutation, applied here
        anchor = 0.0 * sum(p.sum() for p in model.blocks.parameters())
        total = (
            anchor + 1.0 * token_loss
        )  # weight=1.0, same shape as `cfg.token_loss_weight * token_loss`
        total.backward()
        return sum(
            p.grad.abs().sum().item() for p in model.blocks.parameters() if p.grad is not None
        )

    grad_attached = trunk_grad_abs_sum(detach=False)
    grad_detached = trunk_grad_abs_sum(detach=True)
    assert grad_attached > 0.0, "L_token produced no trunk gradient even when attached"
    assert grad_detached == 0.0, (
        f"detaching L_token should zero the trunk's gradient from this term; got {grad_detached}"
    )


# ---------------------------------------------------------------------------------------
# Config plumbing: opt-in, resume-relevant, and off by default.
# ---------------------------------------------------------------------------------------


def test_token_aware_weights_default_off() -> None:
    cfg = PretrainConfig(region="x", pair_columns=("a", "b"), shards=[])
    assert cfg.token_loss_weight == 0.0
    assert cfg.decorr_weight == 0.0


def test_token_aware_weights_are_resume_relevant(tiny_cfg: PretrainConfig) -> None:
    """A resume under a DIFFERENT weight is a different run, not a continuation."""
    off_fields = _resume_fields(tiny_cfg)
    on_fields = _resume_fields(replace(tiny_cfg, token_loss_weight=0.3, decorr_weight=0.1))
    assert off_fields["token_loss_weight"] != on_fields["token_loss_weight"]
    assert off_fields["decorr_weight"] != on_fields["decorr_weight"]


# ---------------------------------------------------------------------------------------
# Integration: the terms actually move the final-block token surface, through the real
# training loop and the real receipt -- not a mocked loss.
# ---------------------------------------------------------------------------------------


def test_token_aware_terms_raise_the_final_block_token_global_pr_rank(
    tiny_cfg: PretrainConfig,
) -> None:
    """The gate this whole feature exists to pass (§4.0's W4/W7 rank clause) compares the
    FINAL block's token-global participation-ratio rank against the same run with the
    terms off. This is that comparison, constructed so it FAILS on a no-op:

    - If `token_loss_weight`/`decorr_weight` were parsed but never multiplied into the
      loss (the classic "the flag exists but does nothing" bug), the "on" run would train
      IDENTICALLY to the "off" run (same seed, same data, same architecture -- nothing
      else differs), `on_rank == off_rank`, and the `>` assertion below fails.
    - If either loss were computed with a detached/no-grad tensor (added to the receipt's
      stats but never to the backward graph), the trunk would receive no extra gradient
      and the same collapse happens.

    Both arms otherwise share every config field -- same corpus, same tokenizer, same
    seed, same steps/batch/architecture -- via `replace()`, so a rank difference can only
    come from the terms themselves.
    """
    off_cfg = replace(tiny_cfg, out_dir=str(Path(tiny_cfg.out_dir).parent / "off"))
    on_cfg = replace(
        tiny_cfg,
        out_dir=str(Path(tiny_cfg.out_dir).parent / "on"),
        token_loss_weight=1.0,
        decorr_weight=4.0,
    )

    off_receipt = pretrain_region(off_cfg)
    on_receipt = pretrain_region(on_cfg)

    off_rank = off_receipt["token_aware"]["final_block_rank"]
    on_rank = on_receipt["token_aware"]["final_block_rank"]

    assert on_rank["token_global_pr_rank"] > off_rank["token_global_pr_rank"] + 0.1, (
        f"token-aware terms did not raise the final-block token-global PR rank: "
        f"off={off_rank['token_global_pr_rank']:.4f} on={on_rank['token_global_pr_rank']:.4f}"
    )
    # The ratio the W4/W7 gate actually reads (token_global >= 2x pooled) should also
    # move in the same direction, not just the raw numerator.
    off_ratio = off_rank["token_global_pr_rank"] / off_rank["pooled_pr_rank"]
    on_ratio = on_rank["token_global_pr_rank"] / on_rank["pooled_pr_rank"]
    assert on_ratio > off_ratio


def test_each_term_alone_measurably_changes_the_final_block_token_global_pr_rank(
    tiny_cfg: PretrainConfig,
) -> None:
    """B1: the combined-arm comparison above (`token_loss_weight=1.0, decorr_weight=4.0`
    vs both off) passes even when `L_token` is a COMPLETE no-op (the control run: applying
    `token_loss.detach()` at the training loop's assembly line left that test green,
    12/12), because `L_decorr` at weight 4.0 already moves the rank on its own and the
    `>` assertion never distinguishes which term did the work. This test isolates each
    term against the SAME "both off" baseline, so a broken `L_token` can no longer hide
    behind a healthy `L_decorr` (or vice versa):

    - `token_loss_weight>0, decorr_weight=0` vs both off.
    - `token_loss_weight=0, decorr_weight>0` vs both off.

    Asserts a MEASURABLE CHANGE (`!=`), not a raise (`>`), on purpose: on this tiny
    fixture `L_token`'s effect on the PR-rank statistic is small and not reliably signed
    run-to-run (unlike the combined arm's, which is large enough to raise it reliably --
    see the test above -- and unlike `L_decorr`'s alone, which raised it consistently in
    every configuration explored while building this test). Requiring a specific sign
    here would make the guard flaky on the very axis it is supposed to be robust on. What
    a no-op CANNOT survive, whichever way a healthy term happens to move the rank on a
    given fixture: if `token_loss_weight>0` contributed nothing (parsed but never added
    to the loss, or added as `token_loss.detach()`), the token-only arm trains BYTE-
    IDENTICALLY to the all-off arm -- same seed, same corpus, same architecture, zero
    contribution from anything else -- so its rank would be EXACTLY the off arm's, not
    merely close to it. `abs(diff) > 1e-6` catches that exact-equality collapse while
    tolerating ordinary floating-point noise, which this is nowhere near: the mutation
    proof below found the "off" and detached-`token_loss` arms differ by 0.0, not by a
    rounding error.
    """
    off_cfg = replace(tiny_cfg, out_dir=str(Path(tiny_cfg.out_dir).parent / "off2"))
    token_only_cfg = replace(
        tiny_cfg,
        out_dir=str(Path(tiny_cfg.out_dir).parent / "token-only"),
        token_loss_weight=1.0,
        decorr_weight=0.0,
    )
    decorr_only_cfg = replace(
        tiny_cfg,
        out_dir=str(Path(tiny_cfg.out_dir).parent / "decorr-only"),
        token_loss_weight=0.0,
        decorr_weight=4.0,
    )

    off_rank = pretrain_region(off_cfg)["token_aware"]["final_block_rank"]["token_global_pr_rank"]
    token_rank = pretrain_region(token_only_cfg)["token_aware"]["final_block_rank"][
        "token_global_pr_rank"
    ]
    decorr_rank = pretrain_region(decorr_only_cfg)["token_aware"]["final_block_rank"][
        "token_global_pr_rank"
    ]

    assert abs(token_rank - off_rank) > 1e-6, (
        f"L_token ALONE produced no measurable change in the final-block token-global PR "
        f"rank: off={off_rank!r} token_only={token_rank!r} -- a detached/no-op L_token "
        f"would leave this arm byte-identical to 'off' (L_decorr is untouched here, "
        f"weight 0.0, so it cannot be the one carrying this comparison)."
    )
    assert abs(decorr_rank - off_rank) > 1e-6, (
        f"L_decorr ALONE produced no measurable change in the final-block token-global "
        f"PR rank: off={off_rank!r} decorr_only={decorr_rank!r}"
    )


def test_receipt_records_both_rank_definitions_weights_and_the_flag(
    tiny_cfg: PretrainConfig,
) -> None:
    """ "Both ranks are participation ratio, and the receipt says so" (§4.0) -- entropy is
    recorded ALONGSIDE PR, never instead of it, on both the "off" and "on" arm."""
    off_receipt = pretrain_region(tiny_cfg)
    ta = off_receipt["token_aware"]
    assert ta["enabled"] is False
    assert ta["token_loss_weight"] == 0.0
    assert ta["decorr_weight"] == 0.0
    for key in (
        "pooled_pr_rank",
        "pooled_entropy_rank",
        "token_global_pr_rank",
        "token_global_entropy_rank",
    ):
        assert key in ta["final_block_rank"]

    on_cfg = replace(
        tiny_cfg,
        out_dir=str(Path(tiny_cfg.out_dir).parent / "on2"),
        token_loss_weight=0.3,
        decorr_weight=0.2,
        steps=8,
        eval_every=8,
    )
    on_receipt = pretrain_region(on_cfg)
    ta_on = on_receipt["token_aware"]
    assert ta_on["enabled"] is True
    assert ta_on["token_loss_weight"] == pytest.approx(0.3)
    assert ta_on["decorr_weight"] == pytest.approx(0.2)


def test_terms_off_trains_the_plain_infonce_loss_unchanged(tiny_cfg: PretrainConfig) -> None:
    """Regression guard for the refactor that split `model(a_ids, a_mask)` into
    `model.tokens()` + `model.pool()` inside the training loop: with both weights at
    their default 0.0, no `token_loss`/`decorr_loss` key should appear in the receipt's
    history, and the run must still produce a normal-shaped receipt."""
    receipt = pretrain_region(tiny_cfg)
    assert receipt["token_aware"]["enabled"] is False
    for row in receipt["history"]:
        assert "token_loss" not in row
        assert "decorr_loss" not in row
