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
from pathlib import Path
from typing import Any

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


def fp32_reference_bytes(model: nn.Module) -> int:
    """The fp32 reference size a compression ratio is measured against: every
    parameter tensor at 4 bytes/element, weights only.

    This is the ONE definition of "how big is the fp32 model" this module uses --
    :func:`build_plan` stamps it as `QuantPlan.fp32_bytes`, and `compression_ratio`
    (``fp32_bytes / stored_bytes``) is only meaningful because both sides count the
    same thing: parameter tensors, never optimizer state, RNG state, or any other
    checkpoint bookkeeping that ``torch.save`` happens to have written alongside the
    weights. A caller measuring "the fp32 size" any other way -- e.g. a checkpoint
    file's ``stat().st_size``, which for a resumable training checkpoint also carries
    the Adam optimizer's momentum and variance buffers (routinely ~2x the weights
    themselves) -- produces a number that is not comparable to this one, and so not
    comparable to a quantized artifact's `packed_stored_bytes` either. Any reader that
    wants a ratio comparable to `compression_ratio` must call this, not reinvent it.
    """
    return sum(p.numel() * 4 for p in model.parameters())


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
    plan.fp32_bytes = fp32_reference_bytes(model)

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


# ------------------------------------------------------------------ persisted artifact
#
# Everything above this line measures a plan; nothing writes it to disk. That gap
# means "quantized" was a number in a receipt, never a smaller file next to the
# checkpoint -- the packed sub-byte tensors this module exists to produce (see
# quant/packing.py) were computed for every sensitivity probe and then discarded.
# `pack_state_dict` is the plan actually applied to `model`'s real weights (not the
# copy `apply_plan` perturbs for measurement) and packed per :data:`QuantizedTensor`;
# `save_packed_artifact` writes that as a single ``torch.save`` file so a consumer can
# reconstruct the model with :func:`load_packed_artifact` + :func:`unpack_state_dict`
# without needing this module's `QuantPlan` at all.
#
# Deliberately a plain dict of tensors/ints/lists/strings -- every value
# `torch.load(..., weights_only=True)` already allow-lists -- not the `QuantizedTensor`
# dataclass (numpy arrays, not tensors) or `QuantPlan` itself, so the artifact loads
# under the same safe-unpickling contract `regions/_checkpoint.py.load_checkpoint`
# enforces for the fp32 checkpoint this artifact sits beside.
#
# EVERY STORED TENSOR IS ON THE CPU, DELIBERATELY
# `torch.save` records each tensor's device in the file. An artifact packed on a CUDA
# host therefore carries `cuda:0` location tags, and two things follow that are both
# wrong for something meant to be published: a CPU-only reader's `torch.load` raises
# unless it happens to pass `map_location`, and the SAME weights packed on a GPU host
# and on a CPU host produce different bytes and so different sha256s -- making the hash
# a receipt records a fact about which machine ran the quantizer rather than about the
# weights. Both are fixed here, at the one place that writes the file, by moving every
# kept tensor to the CPU before it is stored; not by asking every reader to remember
# `map_location`, which only ever fixes the first problem and never the second.

#: The `format` string every artifact carries, and the only one this module reads.
PACKED_FORMAT = "csd-ptq-v1"

#: Sections holding one tensor per quantized parameter. Split out from the other
#: top-level keys because these three, plus the fp32 section, are exactly what
#: :func:`packed_stored_bytes` counts.
_PACKED_TENSOR_SECTIONS: tuple[str, ...] = ("codes", "scale", "zero")

#: Every top-level key :func:`pack_state_dict` writes. :func:`load_packed_artifact`
#: requires all of them, so a file that is a valid tensor pickle but not one of ours is
#: rejected by shape up front rather than by whichever `KeyError` a consumer trips over.
_PACKED_KEYS: tuple[str, ...] = ("format", "bits", *_PACKED_TENSOR_SECTIONS, "shape", "fp32")


def _persistent_buffers(model: nn.Module) -> list[str]:
    """Entries in ``model.state_dict()`` that are not parameters.

    That is exactly the persistent buffers: `state_dict()` carries parameters plus
    buffers registered with ``persistent=True`` (the default), and nothing else.
    A buffer registered ``persistent=False`` -- `regions/text_encoder.py`'s `pos_embed`,
    which is recomputed from the config -- is deliberately absent from `state_dict()`
    and so is deliberately absent here too.
    """
    params = {name for name, _ in model.named_parameters()}
    return sorted(key for key in model.state_dict() if key not in params)


def pack_state_dict(model: nn.Module, plan: QuantPlan) -> dict[str, Any]:
    """The packed, on-disk representation of ``model`` quantized per ``plan``.

    Every tensor named in ``plan.bits`` is packed at its assigned width (codes, the
    per-channel scale and zero-point, and the original shape -- everything
    :func:`dequantize_tensor` needs to reconstruct it). Every tensor NOT in
    ``plan.bits`` -- 1-D parameters, and anything :func:`quantizable` excluded --
    is stored verbatim in fp32, so the artifact alone is a complete model, not a
    diff against the checkpoint.

    Every stored tensor is moved to the CPU first; see this section's header comment
    for why that is a correctness property of the artifact and not a convenience.

    Raises:
        ValueError: if ``model`` has any persistent buffer. This function walks
            ``named_parameters()``, so a persistent buffer would be silently dropped
            and the artifact would load as an incomplete model while still claiming
            to be one. Storing buffers as extra fp32 entries would fix that half and
            break another: ``apply_plan`` counts parameter bytes only, so every
            receipt's ``stored_bytes`` -- and `scripts/csd-publish-checkpoint.py`'s
            check of it against the artifact -- would start disagreeing with the file
            by exactly the buffer bytes. So refuse instead, and make admitting
            buffers a deliberate change to the format AND to that byte accounting
            together. A derived buffer should be registered ``persistent=False``.
    """
    dropped = _persistent_buffers(model)
    if dropped:
        raise ValueError(
            f"{type(model).__name__} has persistent buffer(s) {dropped}, which "
            "pack_state_dict does not store -- the artifact would load as an "
            "incomplete model. Register a derived buffer with persistent=False (see "
            "regions/text_encoder.py's pos_embed), or extend this format and "
            "apply_plan's byte accounting together before packing one."
        )
    packed: dict[str, Any] = {
        "format": PACKED_FORMAT,
        "bits": dict(plan.bits),
        "codes": {},
        "scale": {},
        "zero": {},
        "shape": {},
        "fp32": {},
    }
    for name, p in model.named_parameters():
        if name in plan.bits:
            q = quantize_tensor(p.detach(), plan.bits[name])
            packed["codes"][name] = torch.from_numpy(q.codes)
            packed["scale"][name] = torch.from_numpy(q.scale)
            packed["zero"][name] = torch.from_numpy(q.zero)
            packed["shape"][name] = list(q.shape)
        else:
            packed["fp32"][name] = p.detach().to("cpu").clone()
    return packed


def unpack_state_dict(packed: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Reverse :func:`pack_state_dict`: a plain ``name -> fp32 tensor`` state dict.

    Takes the dict, not a path, so it works on a just-packed artifact as well as a
    loaded one. Read a file with :func:`load_packed_artifact` rather than a bare
    ``torch.load``: it pins the safe-unpickling flags and checks the top-level shape
    this function then indexes into.
    """
    out: dict[str, torch.Tensor] = dict(packed["fp32"])
    for name, bits in packed["bits"].items():
        q = QuantizedTensor(
            codes=packed["codes"][name].numpy(),
            scale=packed["scale"][name].numpy(),
            zero=packed["zero"][name].numpy(),
            shape=tuple(packed["shape"][name]),
            bits=bits,
        )
        out[name] = dequantize_tensor(q)
    return out


def save_packed_artifact(model: nn.Module, plan: QuantPlan, path: Any) -> str:
    """Pack ``model`` per ``plan`` and write it atomically to ``path``.

    Reuses `regions/_checkpoint.py`'s ``atomic_save`` (temp file in the same
    directory, then an atomic rename) so a crash mid-write never leaves a truncated
    artifact at the final name -- the same guarantee the fp32 checkpoint already
    gets. Returns the sha256 of the bytes actually written, computed from the file on
    disk (not from the in-memory dict), so it is a claim about what a later reader
    will actually load, not about what this process intended to write.
    """
    from cogsyndelta.regions._checkpoint import atomic_save, sha256_file

    path = Path(path)
    atomic_save(pack_state_dict(model, plan), path)
    return sha256_file(path)


def load_packed_artifact(path: Any) -> dict[str, Any]:
    """Read an artifact written by :func:`save_packed_artifact`, validated by shape.

    Two flags are pinned here rather than left to each caller. ``weights_only=True``
    is the safe-unpickling contract this format was designed for (see this section's
    header) and must not be relaxed: an artifact path can reach a reader out of a
    receipt on a writable NFS export. ``map_location="cpu"`` makes the load succeed on
    a host with no GPU even for an artifact written before packing moved tensors to
    the CPU, so an old file stays readable instead of raising on its stale device tags.

    The shape check is what turns "some dict of tensors" into "one of ours": a file
    missing a section, or carrying a `format` this module does not know, is rejected
    here with a message naming the file, rather than surfacing later as a `KeyError`
    inside :func:`unpack_state_dict` or as a silently short byte count.

    Raises:
        ValueError: if the file is not a `csd-ptq-v1` artifact of the expected shape.
    """
    path = Path(path)
    obj = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: not a packed artifact -- loaded a {type(obj).__name__}")
    missing = [key for key in _PACKED_KEYS if key not in obj]
    if missing:
        raise ValueError(f"{path}: not a packed artifact -- missing top-level key(s) {missing}")
    if obj["format"] != PACKED_FORMAT:
        raise ValueError(
            f"{path}: packed artifact format {obj['format']!r}, expected {PACKED_FORMAT!r}"
        )
    for key in _PACKED_KEYS[1:]:
        if not isinstance(obj[key], dict):
            raise ValueError(
                f"{path}: packed artifact {key!r} is a {type(obj[key]).__name__}, expected a dict"
            )
    names = set(obj["bits"])
    for key in (*_PACKED_TENSOR_SECTIONS, "shape"):
        if set(obj[key]) != names:
            raise ValueError(
                f"{path}: packed artifact {key!r} covers {sorted(set(obj[key]))}, but "
                f"'bits' covers {sorted(names)} -- the sections disagree on which "
                "tensors are quantized"
            )
    return obj


def packed_stored_bytes(packed: dict[str, Any]) -> int:
    """The artifact's own stored size, counted exactly as :func:`apply_plan` counts a
    plan's.

    `apply_plan` charges a quantized tensor ``QuantizedTensor.nbytes`` (packed codes
    plus the per-channel scale and zero-point) and an unquantized one four bytes per
    element; this walks the same four sections of the written file and charges the
    same way, so a receipt's ``stored_bytes`` and this number are comparable by
    construction rather than by coincidence. That comparison is the point: it is what
    lets `scripts/csd-publish-checkpoint.py` check a receipt's claimed compression
    against the artifact it is about to publish instead of transcribing it.
    """
    total = 0
    for section in _PACKED_TENSOR_SECTIONS:
        total += sum(int(t.numel()) * int(t.element_size()) for t in packed[section].values())
    total += sum(int(t.numel()) * 4 for t in packed["fp32"].values())
    return total


def packed_width_histogram(packed: dict[str, Any]) -> dict[str, int]:
    """Bit-width -> tensor count, measured from the artifact.

    String keys ascending by numeric width, matching the ``width_histogram`` shape
    `scripts/csd-quantize.py` writes into a quant receipt, so the two can be compared
    directly.
    """
    hist: dict[str, int] = {}
    for bits in packed["bits"].values():
        key = str(int(bits))
        hist[key] = hist.get(key, 0) + 1
    return {key: hist[key] for key in sorted(hist, key=int)}
