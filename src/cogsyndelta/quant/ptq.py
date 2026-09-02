"""Sensitivity-driven post-training quantization for CSD submodels.

THE IDEA
Not every weight deserves the same precision. Some tensors can drop to 3 or 4 bits with no
measurable effect on the task; others lose the model outright. Selective quantization
spends bits where they buy accuracy and takes them back everywhere else. What makes that
honest rather than decorative is how sensitivity is decided and how size is counted.

SENSITIVITY IS MEASURED ON THE TASK, NOT ON THE WEIGHTS
The tempting proxy is weight error -- quantize a tensor, measure MSE against fp32, call
the noisy ones sensitive. It is cheap and it is wrong: a tensor can absorb large weight
error with no task effect, and a tiny tensor can carry the decision boundary. So each
candidate is quantized ALONE, the real held-out metric is recomputed, and the drop in that
metric is its sensitivity. For a 16M-parameter encoder that is roughly thirty evaluations
-- affordable, and it measures the thing we actually care about.

SIZE IS MEASURED FROM PACKED BYTES, NOT NOMINAL BITS
See quant/packing.py. Storing 4-bit codes in uint8 shrinks the reported bit-width and not
the file. Every size here comes from the packed buffer, and the per-channel scale and
zero-point are counted as the real overhead they are -- for a narrow tensor that metadata
can cost more than the weights it describes, which is exactly the case where quantizing
should be declined.

WHAT STAYS IN FP32, DELIBERATELY
Norm weights, biases and any 1-D parameter. They are a rounding error in size and the
dominant risk in accuracy. Excluding them is not caution, it is the correct trade.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from cogsyndelta.quant.packing import pack_codes, unpack_codes

# Widths the allocator may choose, ascending. 7 is omitted: it costs a byte per eight
# values more than 6 and almost never lands between them in practice.
LADDER: tuple[int, ...] = (2, 3, 4, 5, 6, 8)


@dataclass
class QuantizedTensor:
    """One packed tensor plus the metadata needed to reconstruct it."""

    codes: np.ndarray
    scale: np.ndarray
    zero: np.ndarray
    shape: tuple[int, ...]
    bits: int

    @property
    def nbytes(self) -> int:
        """Real stored size: packed codes plus per-channel scale and zero-point."""
        return int(self.codes.nbytes + self.scale.nbytes + self.zero.nbytes)


def quantize_tensor(w: torch.Tensor, bits: int) -> QuantizedTensor:
    """Asymmetric per-output-channel quantization of a weight tensor.

    Per-channel rather than per-tensor because a single outlier row otherwise stretches
    the range for every other row, and the cost is two floats per output channel.

    Args:
        w: Weight tensor, 2-D or higher; dimension 0 is treated as output channels.
        bits: Width from :data:`LADDER`.

    Returns:
        The packed tensor.
    """
    x = w.detach().float().cpu()
    flat = x.reshape(x.shape[0], -1)
    levels = (1 << bits) - 1
    vmin = flat.min(dim=1).values
    vmax = flat.max(dim=1).values
    scale = ((vmax - vmin) / levels).clamp_min(1e-12)
    rounded = torch.round((flat - vmin.unsqueeze(1)) / scale.unsqueeze(1))
    # Separate name rather than rebinding: the tensor and the numpy array are different
    # types, and reusing one variable for both hides that from the checker.
    codes = rounded.clamp(0, levels).to(torch.int32).numpy().astype(np.uint16)
    return QuantizedTensor(
        codes=pack_codes(codes, bits),
        scale=scale.numpy().astype(np.float32),
        zero=vmin.numpy().astype(np.float32),
        shape=tuple(x.shape),
        bits=bits,
    )


def dequantize_tensor(q: QuantizedTensor) -> torch.Tensor:
    """Reconstruct a float tensor from its packed form."""
    rows = q.shape[0]
    cols = int(np.prod(q.shape[1:])) if len(q.shape) > 1 else 1
    codes = unpack_codes(q.codes, q.bits, rows * cols).astype(np.float32).reshape(rows, cols)
    scale = q.scale.reshape(rows, 1)
    zero = q.zero.reshape(rows, 1)
    return torch.from_numpy(codes * scale + zero).reshape(q.shape)


def quantizable(model: nn.Module, min_elements: int = 4096) -> list[str]:
    """Names of parameters worth quantizing.

    Excludes every 1-D parameter -- norms and biases -- because they are negligible in
    size and dominant in sensitivity. Also excludes tensors small enough that the
    per-channel scale and zero-point would rival the weights they describe: quantizing
    those makes the file bigger, which is the opposite of the point.
    """
    out = []
    for name, p in model.named_parameters():
        if p.dim() < 2 or p.numel() < min_elements:
            continue
        out.append(name)
    return out


def _set_param(model: nn.Module, name: str, value: torch.Tensor) -> None:
    mod = model
    parts = name.split(".")
    for part in parts[:-1]:
        mod = getattr(mod, part)
    getattr(mod, parts[-1]).data.copy_(value)


@dataclass
class SensitivityRow:
    """What one tensor costs the task when quantized alone."""

    name: str
    elements: int
    metric: float
    drop: float


def measure_sensitivity(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    names: list[str],
    bits: int,
    baseline: float,
) -> list[SensitivityRow]:
    """Quantize each tensor alone at ``bits`` and record the real metric drop.

    Args:
        model: Model to probe; restored exactly after each trial.
        eval_fn: Returns the held-out metric for a model. Higher is better.
        names: Parameters to probe.
        bits: Width to probe at -- use the most aggressive one under consideration.
        baseline: The fp32 metric to measure drops against.

    Returns:
        Rows sorted most-sensitive first.
    """
    rows: list[SensitivityRow] = []
    for name in names:
        param = dict(model.named_parameters())[name]
        original = param.detach().clone()
        _set_param(model, name, dequantize_tensor(quantize_tensor(original, bits)))
        try:
            metric = eval_fn(model)
        finally:
            # Restore before anything else can observe the perturbed model, including a
            # failing eval_fn -- one leaked tensor silently poisons every later row.
            _set_param(model, name, original)
        rows.append(
            SensitivityRow(
                name=name, elements=original.numel(), metric=metric, drop=baseline - metric
            )
        )
    return sorted(rows, key=lambda r: r.drop, reverse=True)


@dataclass
class QuantPlan:
    """Per-tensor bit assignment, plus what it actually cost."""

    bits: dict[str, int] = field(default_factory=dict)
    fp32: list[str] = field(default_factory=list)
    metric: float = 0.0
    baseline: float = 0.0
    stored_bytes: int = 0
    fp32_bytes: int = 0
    promotions: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        """Compression against fp32, from measured bytes."""
        return self.fp32_bytes / max(1, self.stored_bytes)

    @property
    def mean_bits(self) -> float:
        """Effective bits per quantized weight, including metadata overhead."""
        n = sum(1 for _ in self.bits)
        return 0.0 if not n else 8.0 * self.stored_bytes / max(1, self.fp32_bytes // 4)


def apply_plan(model: nn.Module, plan: QuantPlan) -> tuple[nn.Module, int]:
    """Quantize a copy of ``model`` per ``plan``; return it and its measured bytes."""
    work = copy.deepcopy(model)
    total = 0
    params = dict(work.named_parameters())
    for name, p in params.items():
        if name in plan.bits:
            q = quantize_tensor(p.detach(), plan.bits[name])
            _set_param(work, name, dequantize_tensor(q))
            total += q.nbytes
        else:
            total += p.numel() * 4
    return work, total


def build_plan(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    baseline: float,
    tolerance: float,
    aggressive_bits: int = 3,
    max_bits: int = 8,
) -> QuantPlan:
    """Assign bits greedily: start aggressive, promote the worst offenders until it holds.

    Fixed tiers ("big tensors get 8, small get 4") assume sensitivity tracks size. It does
    not. This starts with every eligible tensor at the most aggressive width, and while
    the measured metric is outside tolerance, promotes the single most sensitive tensor
    one rung up the ladder and re-measures. Each promotion is therefore bought with an
    observed recovery rather than a rule of thumb, and the result is the cheapest
    assignment found that still meets the accuracy budget.

    Args:
        model: Trained fp32 model.
        eval_fn: Held-out metric, higher is better.
        baseline: fp32 metric.
        tolerance: Largest acceptable absolute drop from ``baseline``.
        aggressive_bits: Starting width for every tensor.
        max_bits: Ceiling; a tensor at this width can only be dropped to fp32.

    Returns:
        The plan, carrying its measured metric and byte count.
    """
    names = quantizable(model)
    sens = measure_sensitivity(model, eval_fn, names, aggressive_bits, baseline)
    order = [r.name for r in sens]

    plan = QuantPlan(bits=dict.fromkeys(names, aggressive_bits), baseline=baseline)
    plan.fp32 = [n for n, _ in model.named_parameters() if n not in plan.bits]
    plan.fp32_bytes = sum(p.numel() * 4 for p in model.parameters())

    for _ in range(len(names) * len(LADDER)):
        probe, size = apply_plan(model, plan)
        metric = eval_fn(probe)
        if baseline - metric <= tolerance:
            plan.metric, plan.stored_bytes = metric, size
            return plan
        # Promote the most sensitive tensor that is not already at the ceiling.
        target = next((n for n in order if plan.bits[n] < max_bits), None)
        if target is None:
            plan.metric, plan.stored_bytes = metric, size
            return plan
        rung = LADDER.index(plan.bits[target])
        plan.bits[target] = LADDER[min(rung + 1, LADDER.index(max_bits))]
        plan.promotions.append(f"{target}->{plan.bits[target]}b")
        if plan.bits[target] >= max_bits:
            order = [n for n in order if n != target] + [target]

    probe, size = apply_plan(model, plan)
    plan.metric, plan.stored_bytes = eval_fn(probe), size
    return plan
