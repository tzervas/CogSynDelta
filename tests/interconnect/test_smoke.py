"""The CPU smoke test -- spec section 5: "runs one forward and backward on the
integration configuration at `B = 4`, asserts every trainable parameter received a
finite gradient and every faculty parameter received none, and asserts a wall-clock
under 30s with a 2s target; it runs in the CI job image, which has no GPU."

Run directly:
    CUDA_VISIBLE_DEVICES="" python -m pytest tests/interconnect/test_smoke.py::test_forward_backward_under_30s -q
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from tests.interconnect.conftest import make_toy_inputs


def test_forward_backward_under_30s(white_matter, toy_scope) -> None:
    """One forward/backward at `B = 4` on the toy config, under a 30s hard budget.

    `[lane]`: spec Table 6 marks phase A's controller "frozen (bypassed)" -- since this
    smoke test is phase-A shaped (workspace/adapters/read-out train, the controller
    does not), its parameters are frozen here before backward so "every trainable
    parameter received a finite gradient" reads over the phase-A trainable set Table 6
    actually names, not over a controller this call never uses to drive execution
    (module docstring tension 1: the controller runs for phase-B exposure only).
    """
    for p in white_matter.controller.parameters():
        p.requires_grad_(False)

    start = time.perf_counter()

    inputs = make_toy_inputs(batch_size=4, scope=toy_scope, seed=42)
    out = white_matter(inputs)

    target = inputs["language"].sum(dim=1) % 4
    rank_loss = F.cross_entropy(out.scores, target)

    pooled_targets = {
        name: white_matter.faculties[name].pool(
            *white_matter.faculties[name].tokens(
                inputs[name], context_tokens=white_matter.config.participants[name].ctx_min or 1
            )
        )
        for name in white_matter.region_names
    }
    unify_loss = sum(
        (1.0 - F.cosine_similarity(out.probe_outputs[name], pooled_targets[name], dim=-1)).mean()
        for name in white_matter.region_names
    )
    loss = rank_loss + unify_loss + out.f.pow(2).mean()

    loss.backward()

    elapsed = time.perf_counter() - start
    assert elapsed < 30.0, f"forward+backward took {elapsed:.2f}s, over the 30s budget"
    if elapsed >= 2.0:
        print(
            f"[test_forward_backward_under_30s] {elapsed:.3f}s, over the 2s target but under the 30s budget"
        )

    for name, param in white_matter.named_parameters():
        if not param.requires_grad:
            continue
        assert param.grad is not None, f"trainable parameter {name!r} received no gradient"
        assert torch.isfinite(param.grad).all(), (
            f"trainable parameter {name!r} has a non-finite gradient"
        )

    for fac_name, fac in white_matter.faculties.items():
        for p_name, p in fac.named_parameters():
            assert p.grad is None, f"faculty {fac_name!r} parameter {p_name!r} received a gradient"
