"""Sub-byte weight packing, so a bit-width claim corresponds to real bytes.

WHY THIS FILE EXISTS AT ALL
contracts/compactor.py already says it plainly: codes land in uint8 for bits<=8 and
uint16 for 9..16, so "measured stored_bytes takes exactly two values across the whole
range. Sub-8-bit buys nothing without real sub-byte packing."

That is fatal for selective quantization specifically. The entire point of assigning 4
bits to insensitive tensors and 8 to sensitive ones is that the 4-bit ones get smaller.
Store both in uint8 and they do not -- the plan reports a shrinking average bit-width
while the file on disk stays exactly the same size. The numbers would look like a
compression result and mean nothing.

So packing here is a real bitstream: n values at b bits occupy ceil(n*b/8) bytes, and
every size this package reports is measured from the packed buffer rather than inferred
from a nominal width.
"""

from __future__ import annotations

import numpy as np


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack unsigned integer codes into a dense bitstream.

    Args:
        codes: Non-negative integers, each < ``2**bits``.
        bits: Width in ``[1, 16]``.

    Returns:
        ``uint8`` buffer of exactly ``ceil(codes.size * bits / 8)`` bytes.

    Raises:
        ValueError: If ``bits`` is out of range or a code does not fit in it.
    """
    if not 1 <= bits <= 16:
        raise ValueError(f"bits must be in [1, 16], got {bits}")
    flat = np.ascontiguousarray(codes).reshape(-1).astype(np.uint16)
    if flat.size and int(flat.max()) >= (1 << bits):
        # Silent truncation is the failure mode this guard exists for: a 12-bit code cast
        # into a narrower field still reconstructs to plausible-looking numbers.
        raise ValueError(f"code {int(flat.max())} does not fit in {bits} bits")
    # Big-endian bit expansion, then keep only the low `bits` of each value.
    as_bytes = flat.view(np.uint8).reshape(-1, 2)[:, ::-1]
    expanded = np.unpackbits(as_bytes, axis=1)[:, -bits:]
    return np.packbits(expanded.reshape(-1))


def unpack_codes(buf: np.ndarray, bits: int, count: int) -> np.ndarray:
    """Reverse :func:`pack_codes`.

    Args:
        buf: Packed ``uint8`` buffer.
        bits: Width used to pack.
        count: Number of codes to recover.

    Returns:
        ``uint16`` array of length ``count``.
    """
    if not 1 <= bits <= 16:
        raise ValueError(f"bits must be in [1, 16], got {bits}")
    need = count * bits
    stream = np.unpackbits(np.ascontiguousarray(buf).reshape(-1))[:need]
    if stream.size < need:
        raise ValueError(f"buffer holds {stream.size} bits, need {need}")
    grid = stream.reshape(count, bits)
    pad = np.zeros((count, 16 - bits), dtype=np.uint8)
    full = np.concatenate([pad, grid], axis=1)
    packed = np.packbits(full, axis=1)  # [count, 2] big-endian
    return (packed[:, 0].astype(np.uint16) << 8) | packed[:, 1].astype(np.uint16)


def packed_bytes(count: int, bits: int) -> int:
    """Bytes a packed buffer of ``count`` values at ``bits`` occupies."""
    return (count * bits + 7) // 8
