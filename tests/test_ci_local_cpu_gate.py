"""Probe: the test-execution steps of ``scripts/ci_local.sh`` run CPU-only.

WHY (2026-09-04): the PoC route-train test previously diverged between the tracked
``.githooks/pre-push`` hook and CI. The hook ran ``scripts/ci_local.sh`` directly on
whatever machine pushed -- on a GPU box, the "desktop CUDA torch from pyproject" sync
branch installs cu128 wheels, so ``torch.cuda.is_available()`` is true and pytest
exercises CUDA kernels. CI's runner has no GPU at all, so the identical test exercises
CPU kernels there. Same test, same command, two different code paths -- exactly the kind
of drift ``ci_local.sh``'s own header says it exists to prevent.

``scripts/ci_local.sh`` now exports ``CUDA_VISIBLE_DEVICES=""`` immediately before its
pytest / poc-pytest / poc-cli steps (the sync itself is untouched -- CUDA torch is still
installed either way), so those steps see no CUDA device regardless of what the host
actually has. This test is that guarantee made observable: it fails loudly if a future
edit to ``ci_local.sh`` drops the export, because it would then see whatever the host
sees instead of a fixed, CI-matching CPU view.
"""

from __future__ import annotations

import torch


def test_cuda_hidden_during_test_execution() -> None:
    """``torch.cuda.is_available()`` is False under ci_local.sh's test steps.

    Not a claim that this host lacks a GPU -- it may well have one. The claim is that
    ``CUDA_VISIBLE_DEVICES`` has been cleared for this process, matching the GPU-less CI
    runner regardless of what hardware ran the gate.
    """
    assert not torch.cuda.is_available(), (
        "CUDA is visible to this test process; scripts/ci_local.sh should have exported "
        'CUDA_VISIBLE_DEVICES="" before running pytest / poc pytest / poc cli steps.'
    )
