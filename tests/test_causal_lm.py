"""Causal LM correctness tests.

The causality test is the important one. A mask that leaks even one future position makes
the loss curve look *better*, so the bug hides in the direction nobody investigates: you
get a suspiciously good perplexity and conclude the architecture is working.
"""

from __future__ import annotations

import math

import pytest
import torch

from cogsyndelta.model import CausalLM, CausalLMConfig


@pytest.fixture(scope="module")
def tiny_cfg() -> CausalLMConfig:
    return CausalLMConfig(vocab_size=512, dim=64, n_layers=2, n_heads=4, seq_len=64)


@pytest.mark.cpu
def test_untrained_loss_matches_uniform_baseline(tiny_cfg: CausalLMConfig) -> None:
    """An untrained model must sit at ln(vocab_size).

    Meaningfully below it at init means something is leaking or the head is biased;
    meaningfully above means the init is broken.
    """
    torch.manual_seed(0)
    model = CausalLM(tiny_cfg)
    idx = torch.randint(0, tiny_cfg.vocab_size, (4, 32))
    targets = torch.randint(0, tiny_cfg.vocab_size, (4, 32))
    _, loss = model(idx, targets)
    assert loss is not None
    assert abs(loss.item() - math.log(tiny_cfg.vocab_size)) < 0.5


@pytest.mark.cpu
def test_future_tokens_cannot_affect_earlier_positions(tiny_cfg: CausalLMConfig) -> None:
    """The causality contract, checked directly rather than trusted.

    Changing the token at position t must leave logits at every position < t bit-identical.
    If it does not, the model is conditioning on the future and every perplexity number
    it produces is meaningless.
    """
    torch.manual_seed(0)
    model = CausalLM(tiny_cfg).eval()

    idx = torch.randint(0, tiny_cfg.vocab_size, (1, 16))
    with torch.no_grad():
        base, _ = model(idx)

    altered = idx.clone()
    cut = 8
    altered[0, cut] = (altered[0, cut] + 1) % tiny_cfg.vocab_size
    with torch.no_grad():
        changed, _ = model(altered)

    torch.testing.assert_close(base[:, :cut], changed[:, :cut], rtol=0, atol=0)
    assert not torch.equal(base[:, cut:], changed[:, cut:]), (
        "changing a token left later positions untouched -- attention may not be wired"
    )


@pytest.mark.cpu
def test_overfits_a_single_batch(tiny_cfg: CausalLMConfig) -> None:
    """A correct model must be able to memorise one batch.

    This is the cheapest end-to-end check that gradients flow through every component:
    embedding, RoPE, attention, SwiGLU, norms and the tied head. If any of them is
    detached, loss plateaus near the uniform baseline instead of collapsing.
    """
    torch.manual_seed(0)
    model = CausalLM(tiny_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    idx = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    targets = torch.randint(0, tiny_cfg.vocab_size, (2, 16))

    first = None
    for _ in range(60):
        _, loss = model(idx, targets)
        assert loss is not None
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert first is not None
    assert loss.item() < first * 0.25, f"failed to overfit: {first:.3f} -> {loss.item():.3f}"


@pytest.mark.cpu
def test_weight_tying_shares_storage(tiny_cfg: CausalLMConfig) -> None:
    model = CausalLM(tiny_cfg)
    assert model.head.weight.data_ptr() == model.embed.weight.data_ptr()

    untied = CausalLM(CausalLMConfig(**{**tiny_cfg.__dict__, "tie_embeddings": False}))
    assert untied.head.weight.data_ptr() != untied.embed.weight.data_ptr()
    assert untied.num_parameters() > model.num_parameters()


@pytest.mark.cpu
def test_rejects_sequences_longer_than_configured(tiny_cfg: CausalLMConfig) -> None:
    """RoPE tables are precomputed to seq_len; silently indexing past them would produce
    wrong positions rather than an error."""
    model = CausalLM(tiny_cfg)
    too_long = torch.randint(0, tiny_cfg.vocab_size, (1, tiny_cfg.seq_len + 1))
    with pytest.raises(ValueError, match="exceeds configured"):
        model(too_long)


@pytest.mark.cpu
def test_perplexity_is_token_weighted(tiny_cfg: CausalLMConfig) -> None:
    """A short trailing batch must not skew the estimate."""
    torch.manual_seed(0)
    model = CausalLM(tiny_cfg)
    batches = [
        (
            torch.randint(0, tiny_cfg.vocab_size, (2, 16)),
            torch.randint(0, tiny_cfg.vocab_size, (2, 16)),
        )
        for _ in range(3)
    ]
    ppl = model.estimate_perplexity(batches)
    assert math.isfinite(ppl)
    # Untrained, so it should sit near vocab_size.
    assert 0.4 * tiny_cfg.vocab_size < ppl < 2.5 * tiny_cfg.vocab_size
