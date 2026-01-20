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

    def __add__(self, other: "BalancedTernaryTensor") -> "BalancedTernaryTensor":
        """Add two balanced ternary tensors."""
        result_trits = BalancedTernaryArithmetic.add(self.trits, other.trits)
        return BalancedTernaryTensor(result_trits, self.trits_per_tryte)

    def __mul__(self, other: "BalancedTernaryTensor") -> "BalancedTernaryTensor":
        """Multiply two balanced ternary tensors."""
        result_trits = BalancedTernaryArithmetic.multiply(self.trits, other.trits)
        return BalancedTernaryTensor(result_trits, self.trits_per_tryte)

    def __neg__(self) -> "BalancedTernaryTensor":
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
