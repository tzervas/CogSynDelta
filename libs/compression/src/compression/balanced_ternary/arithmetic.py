"""Balanced Ternary Arithmetic Operations.

Balanced ternary uses digits {-1, 0, +1} for symmetric representation.

Notation:
    -1 represented as: T, -, or ⊖
    0 represented as:  0
    +1 represented as: 1, +, or ⊕

Properties:
    - Symmetric: -x is just flip signs of each digit
    - No sign bit: Sign inherent in representation
    - Unique representation: No +0 vs -0 problem
    - Efficient rounding: Truncation is rounding towards zero

Examples:
    Decimal → Balanced Ternary
    5  → 1TT  (9 - 3 - 1)
    -5 → T11  (-9 + 3 + 1)
    13 → 111  (9 + 3 + 1)
    0  → 0
"""

import torch


class BalancedTernaryArithmetic:
    """Core arithmetic operations for balanced ternary."""

    @staticmethod
    def to_decimal(trits: torch.Tensor) -> torch.Tensor:
        """Convert balanced ternary to decimal.

        Args:
            trits: Balanced ternary digits {-1, 0, 1} (shape: [..., num_trits])

        Returns:
            Decimal values (shape: [...])

        Example:
            >>> trits = torch.tensor([[1, -1, -1]])  # 1TT
            >>> to_decimal(trits)
            tensor([5])  # 9 - 3 - 1 = 5
        """
        # Powers of 3
        num_trits = trits.shape[-1]
        powers = 3 ** torch.arange(num_trits - 1, -1, -1, device=trits.device)

        # Weighted sum
        decimal = (trits * powers).sum(dim=-1)

        return decimal

    @staticmethod
    def from_decimal(decimal: torch.Tensor, num_trits: int) -> torch.Tensor:
        """Convert decimal to balanced ternary.

        Args:
            decimal: Decimal values (shape: [...])
            num_trits: Number of trits for representation

        Returns:
            Balanced ternary digits {-1, 0, 1} (shape: [..., num_trits])

        Example:
            >>> decimal = torch.tensor([5])
            >>> from_decimal(decimal, 9)
            tensor([[1, -1, -1, 0, 0, 0, 0, 0, 0]])  # 1TT000000
        """
        # Initialize output
        shape = list(decimal.shape) + [num_trits]
        trits = torch.zeros(shape, dtype=torch.int8, device=decimal.device)

        # Convert using greedy algorithm
        remaining = decimal.clone().long()

        for i in range(num_trits):
            power = 3 ** (num_trits - 1 - i)

            # Determine trit value: {-1, 0, 1}
            # Use rounding to nearest trit
            trit = torch.round(remaining.float() / power).clamp(-1, 1).to(torch.int8)

            trits[..., i] = trit
            remaining = remaining - trit.long() * power

        return trits

    @staticmethod
    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Add two balanced ternary numbers.

        Args:
            a: First operand (shape: [..., num_trits])
            b: Second operand (shape: [..., num_trits])

        Returns:
            Sum in balanced ternary (shape: [..., num_trits])
        """
        # Convert to decimal, add, convert back
        a_dec = BalancedTernaryArithmetic.to_decimal(a)
        b_dec = BalancedTernaryArithmetic.to_decimal(b)
        sum_dec = a_dec + b_dec

        num_trits = a.shape[-1]
        return BalancedTernaryArithmetic.from_decimal(sum_dec, num_trits)

    @staticmethod
    def multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply two balanced ternary numbers.

        Args:
            a: First operand (shape: [..., num_trits])
            b: Second operand (shape: [..., num_trits])

        Returns:
            Product in balanced ternary (shape: [..., num_trits])
        """
        a_dec = BalancedTernaryArithmetic.to_decimal(a)
        b_dec = BalancedTernaryArithmetic.to_decimal(b)
        prod_dec = a_dec * b_dec

        num_trits = a.shape[-1]
        return BalancedTernaryArithmetic.from_decimal(prod_dec, num_trits)

    @staticmethod
    def negate(a: torch.Tensor) -> torch.Tensor:
        """Negate a balanced ternary number (flip all signs).

        Args:
            a: Operand (shape: [..., num_trits])

        Returns:
            Negated value (shape: [..., num_trits])
        """
        return -a


class BalancedTernaryTensor:
    """Wrapper for balanced ternary tensors with tryte organization.

    A tryte is 9 trits (balanced ternary digits).
    Range: -9841 to +9841 (3^9 = 19683, symmetric around 0)
    """

    def __init__(self, trits: torch.Tensor, trits_per_tryte: int = 9):
        """Initialize balanced ternary tensor.

        Args:
            trits: Balanced ternary digits {-1, 0, 1} (shape: [..., num_trits])
            trits_per_tryte: Number of trits per tryte (default: 9)
        """
        self.trits = trits
        self.trits_per_tryte = trits_per_tryte

        # Validate
        assert trits.shape[-1] % trits_per_tryte == 0, (
            f"Number of trits must be multiple of {trits_per_tryte}"
        )

        self.num_trytes = trits.shape[-1] // trits_per_tryte

    @property
    def shape(self) -> torch.Size:
        """Shape of the tensor (excluding trits dimension)."""
        return self.trits.shape[:-1]

    def to_decimal(self) -> torch.Tensor:
        """Convert to decimal representation.

        Returns:
            Decimal tensor (shape: [...])
        """
        return BalancedTernaryArithmetic.to_decimal(self.trits)

    def to_float(self) -> torch.Tensor:
        """Convert to floating point for neural network operations.

        Returns:
            Float tensor (shape: [...])
        """
        return self.to_decimal().float()

    def __add__(self, other: BalancedTernaryTensor) -> BalancedTernaryTensor:
        """Add two balanced ternary tensors."""
        result_trits = BalancedTernaryArithmetic.add(self.trits, other.trits)
        return BalancedTernaryTensor(result_trits, self.trits_per_tryte)

    def __mul__(self, other: BalancedTernaryTensor) -> BalancedTernaryTensor:
        """Multiply two balanced ternary tensors."""
        result_trits = BalancedTernaryArithmetic.multiply(self.trits, other.trits)
        return BalancedTernaryTensor(result_trits, self.trits_per_tryte)

    def __neg__(self) -> BalancedTernaryTensor:
        """Negate balanced ternary tensor."""
        result_trits = BalancedTernaryArithmetic.negate(self.trits)
        return BalancedTernaryTensor(result_trits, self.trits_per_tryte)

    def __repr__(self) -> str:
        """String representation."""
        return f"BalancedTernaryTensor(shape={self.shape}, trytes={self.num_trytes})"


def tryte_encode(values: torch.Tensor, trits_per_tryte: int = 9) -> BalancedTernaryTensor:
    """Encode decimal/float values as balanced ternary trytes.

    Args:
        values: Decimal values to encode (shape: [...])
        trits_per_tryte: Trits per tryte (default: 9)

    Returns:
        Balanced ternary tensor
    """
    # Round to nearest integer
    int_values = torch.round(values).long()

    # Convert to balanced ternary
    trits = BalancedTernaryArithmetic.from_decimal(int_values, trits_per_tryte)

    return BalancedTernaryTensor(trits, trits_per_tryte)


def tryte_decode(bt_tensor: BalancedTernaryTensor) -> torch.Tensor:
    """Decode balanced ternary trytes to decimal/float.

    Args:
        bt_tensor: Balanced ternary tensor

    Returns:
        Decimal values (shape: [...])
    """
    return bt_tensor.to_float()


class BalancedTernaryMatMul:
    """Optimized matrix multiplication for balanced ternary.

    Leverages sparsity and restricted value set {-1, 0, 1} for efficiency.
    """

    @staticmethod
    def matmul(
        a_trits: torch.Tensor, b_trits: torch.Tensor, trits_per_value: int = 9
    ) -> torch.Tensor:
        """Matrix multiplication of balanced ternary matrices.

        Args:
            a_trits: First matrix trits (shape: [M, K, trits_per_value])
            b_trits: Second matrix trits (shape: [K, N, trits_per_value])
            trits_per_value: Trits per value

        Returns:
            Product in balanced ternary (shape: [M, N, trits_per_value])
        """
        # Convert to decimal for matmul
        a_dec = BalancedTernaryArithmetic.to_decimal(a_trits)  # [M, K]
        b_dec = BalancedTernaryArithmetic.to_decimal(b_trits)  # [K, N]

        # Standard matmul
        c_dec = a_dec @ b_dec  # [M, N]

        # Convert back to balanced ternary
        c_trits = BalancedTernaryArithmetic.from_decimal(c_dec.flatten(), trits_per_value)

        # Reshape
        M, N = c_dec.shape
        c_trits = c_trits.reshape(M, N, trits_per_value)

        return c_trits

    @staticmethod
    def optimized_matmul(a_trits: torch.Tensor, b_trits: torch.Tensor) -> torch.Tensor:
        """Optimized matmul exploiting {-1, 0, 1} values.

        Uses shift-add operations instead of multiplications.

        Args:
            a_trits: First matrix trits (shape: [M, K, trits])
            b_trits: Second matrix trits (shape: [K, N, trits])

        Returns:
            Product trits (shape: [M, N, trits])
        """
        M, K, trits = a_trits.shape
        _, N, _ = b_trits.shape

        # Initialize result
        result = torch.zeros(M, N, trits, dtype=torch.int8, device=a_trits.device)

        # For each trit position
        for t in range(trits):
            power = 3 ** (trits - 1 - t)

            # Get trit slices
            a_t = a_trits[..., t]  # [M, K]
            b_t = b_trits[..., t]  # [K, N]

            # Optimized multiply-accumulate
            # Since trits are {-1, 0, 1}, we can use masking and addition
            for k in range(K):
                a_k = a_t[:, k : k + 1]  # [M, 1]
                b_k = b_t[k : k + 1, :]  # [1, N]

                # Multiply: -1×-1=1, -1×1=-1, etc.
                contrib = a_k * b_k * power  # [M, N]

                # Accumulate
                contrib_trits = BalancedTernaryArithmetic.from_decimal(contrib.flatten(), trits)
                result += contrib_trits.reshape(M, N, trits)

        return result


class BalancedTernaryTryte:
    """High-level tryte representation for neural network weight compression.

    Converts float weights to balanced ternary trytes with packing/unpacking
    for storage efficiency. This is a convenience class that wraps
    BalancedTernaryTensor with additional functionality.

    Attributes:
        trits: Balanced ternary digits {-1, 0, 1}
        trits_per_tryte: Number of trits per tryte (default: 9)
        original_shape: Shape of original weights (excluding trits dimension)
    """

    def __init__(
        self,
        data: torch.Tensor,
        trits_per_tryte: int = 9,
        is_trits: bool = False,
        threshold: float = 0.1,
    ):
        """Initialize balanced ternary tryte.

        Args:
            data: Float weights or pre-computed trits
            trits_per_tryte: Number of trits per tryte (default: 9)
            is_trits: If True, data is already trits; if False, quantize from float
            threshold: Threshold for quantization (if is_trits=False)
        """
        self.trits_per_tryte = trits_per_tryte

        if is_trits:
            # Data is already trits
            self.trits = data.to(torch.int8)
            self.original_shape = data.shape[:-1]
        else:
            # Quantize from float weights
            self.original_shape = data.shape

            # Threshold-based quantization to {-1, 0, +1}
            ternary = torch.zeros_like(data, dtype=torch.int8)
            abs_data = data.abs()
            threshold_val = abs_data.mean() * threshold if threshold < 1 else threshold
            ternary[data > threshold_val] = 1
            ternary[data < -threshold_val] = -1

            # Expand to trits dimension (each value gets trits_per_tryte trits)
            # For simplicity, we use the first trit as the sign, rest are zeros
            self.trits = torch.zeros(
                *data.shape, trits_per_tryte, dtype=torch.int8, device=data.device
            )
            self.trits[..., 0] = ternary

    @property
    def shape(self) -> torch.Size:
        """Shape of the original tensor."""
        return self.original_shape

    def to_decimal(self) -> torch.Tensor:
        """Convert to decimal representation.

        Returns:
            Decimal tensor (same shape as original weights)
        """
        # For simple case where only first trit is used
        return self.trits[..., 0].float()

    def pack(self) -> bytes:
        """Pack trits into compact bytes representation.

        Each trit is {-1, 0, 1} which needs 2 bits (00=0, 01=1, 10=-1).
        We pack 4 trits per byte.

        Returns:
            Packed bytes representation
        """
        # Flatten trits
        flat_trits = self.trits.flatten().tolist()

        # Map {-1, 0, 1} to {2, 0, 1} for 2-bit encoding
        encoded = [t + 1 if t == -1 else t for t in flat_trits]

        # Pack 4 trits per byte
        packed_bytes = []
        for i in range(0, len(encoded), 4):
            byte_val = 0
            for j in range(4):
                if i + j < len(encoded):
                    byte_val |= encoded[i + j] << (6 - j * 2)
            packed_bytes.append(byte_val)

        return bytes(packed_bytes)

    @classmethod
    def unpack(
        cls, packed: bytes, shape: tuple[int, ...], trits_per_tryte: int = 9
    ) -> BalancedTernaryTryte:
        """Unpack bytes to tryte representation.

        Args:
            packed: Packed bytes from pack()
            shape: Original weight shape (excluding trits dimension)
            trits_per_tryte: Trits per tryte

        Returns:
            BalancedTernaryTryte instance
        """
        # Calculate total trits needed
        total_elements = 1
        for s in shape:
            total_elements *= s
        total_trits = total_elements * trits_per_tryte

        # Unpack bytes to trits
        trits = []
        for byte_val in packed:
            for j in range(4):
                if len(trits) >= total_trits:
                    break
                encoded = (byte_val >> (6 - j * 2)) & 0x03
                trit = -1 if encoded == 2 else encoded
                trits.append(trit)

        # Reshape to original shape + trits dimension
        full_shape = (*shape, trits_per_tryte)
        trit_tensor = torch.tensor(trits[:total_trits], dtype=torch.int8).reshape(full_shape)

        return cls(trit_tensor, trits_per_tryte=trits_per_tryte, is_trits=True)

    def __repr__(self) -> str:
        """String representation."""
        return f"BalancedTernaryTryte(shape={self.original_shape}, trits_per_tryte={self.trits_per_tryte})"
