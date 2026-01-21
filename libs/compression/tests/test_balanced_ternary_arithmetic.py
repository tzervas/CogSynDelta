"""Tests for Balanced Ternary Arithmetic."""

from compression.balanced_ternary.arithmetic import BalancedTernaryArithmetic

import pytest
import torch


class TestBalancedTernaryArithmetic:
    """Test suite for balanced ternary arithmetic operations."""

    def test_to_decimal_basic(self):
        """Test converting balanced ternary to decimal."""
        # Example from docstring: 1TT (1, -1, -1) = 9 - 3 - 1 = 5
        trits = torch.tensor([[1, -1, -1, 0, 0, 0, 0, 0, 0]], dtype=torch.int8)

        decimal = BalancedTernaryArithmetic.to_decimal(trits)

        assert decimal.item() == 5

    def test_to_decimal_zero(self):
        """Test zero conversion."""
        trits = torch.zeros((1, 9), dtype=torch.int8)

        decimal = BalancedTernaryArithmetic.to_decimal(trits)

        assert decimal.item() == 0

    def test_to_decimal_negative(self):
        """Test negative number conversion."""
        # -5 = T11 (-9 + 3 + 1)
        trits = torch.tensor([[-1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.int8)

        decimal = BalancedTernaryArithmetic.to_decimal(trits)

        assert decimal.item() == -5

    def test_to_decimal_positive_13(self):
        """Test 13 = 111 (9 + 3 + 1)."""
        trits = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.int8)

        decimal = BalancedTernaryArithmetic.to_decimal(trits)

        assert decimal.item() == 13

    def test_from_decimal_basic(self):
        """Test converting decimal to balanced ternary."""
        decimal = torch.tensor([5])

        trits = BalancedTernaryArithmetic.from_decimal(decimal, num_trits=9)

        # Should be some balanced ternary representation
        assert trits.shape == (1, 9)
        assert set(trits.flatten().tolist()).issubset({-1, 0, 1})

        # Verify roundtrip
        recovered = BalancedTernaryArithmetic.to_decimal(trits)
        assert recovered.item() == 5

    def test_from_decimal_zero(self):
        """Test zero conversion."""
        decimal = torch.tensor([0])

        trits = BalancedTernaryArithmetic.from_decimal(decimal, num_trits=9)

        assert (trits == 0).all()

    def test_from_decimal_negative(self):
        """Test negative number conversion."""
        decimal = torch.tensor([-5])

        trits = BalancedTernaryArithmetic.from_decimal(decimal, num_trits=9)

        # Verify roundtrip
        recovered = BalancedTernaryArithmetic.to_decimal(trits)
        assert recovered.item() == -5

    def test_roundtrip_multiple_values(self):
        """Test roundtrip for multiple values."""
        decimals = torch.tensor([0, 1, -1, 5, -5, 13, -13, 100, -100])

        trits = BalancedTernaryArithmetic.from_decimal(decimals, num_trits=9)
        recovered = BalancedTernaryArithmetic.to_decimal(trits)

        assert torch.allclose(decimals.float(), recovered.float())

    def test_add_basic(self):
        """Test balanced ternary addition."""
        # 1 + 1 = 2
        a = BalancedTernaryArithmetic.from_decimal(torch.tensor([1]), num_trits=9)
        b = BalancedTernaryArithmetic.from_decimal(torch.tensor([1]), num_trits=9)

        result = BalancedTernaryArithmetic.add(a, b)
        decimal_result = BalancedTernaryArithmetic.to_decimal(result)

        assert decimal_result.item() == 2

    def test_add_with_negatives(self):
        """Test addition with negative numbers."""
        # 5 + (-3) = 2
        a = BalancedTernaryArithmetic.from_decimal(torch.tensor([5]), num_trits=9)
        b = BalancedTernaryArithmetic.from_decimal(torch.tensor([-3]), num_trits=9)

        result = BalancedTernaryArithmetic.add(a, b)
        decimal_result = BalancedTernaryArithmetic.to_decimal(result)

        assert decimal_result.item() == 2

    def test_multiply_basic(self):
        """Test balanced ternary multiplication."""
        # 3 × 4 = 12
        a = BalancedTernaryArithmetic.from_decimal(torch.tensor([3]), num_trits=9)
        b = BalancedTernaryArithmetic.from_decimal(torch.tensor([4]), num_trits=9)

        result = BalancedTernaryArithmetic.multiply(a, b)
        decimal_result = BalancedTernaryArithmetic.to_decimal(result)

        assert decimal_result.item() == 12

    def test_multiply_with_negative(self):
        """Test multiplication with negative number."""
        # 5 × (-2) = -10
        a = BalancedTernaryArithmetic.from_decimal(torch.tensor([5]), num_trits=9)
        b = BalancedTernaryArithmetic.from_decimal(torch.tensor([-2]), num_trits=9)

        result = BalancedTernaryArithmetic.multiply(a, b)
        decimal_result = BalancedTernaryArithmetic.to_decimal(result)

        assert decimal_result.item() == -10

    def test_negate_basic(self):
        """Test negation (flip all trits)."""
        # 5 → -5
        trits = BalancedTernaryArithmetic.from_decimal(torch.tensor([5]), num_trits=9)

        negated = BalancedTernaryArithmetic.negate(trits)
        decimal_result = BalancedTernaryArithmetic.to_decimal(negated)

        assert decimal_result.item() == -5

    def test_negate_zero(self):
        """Test that negating zero gives zero."""
        trits = torch.zeros((1, 9), dtype=torch.int8)

        negated = BalancedTernaryArithmetic.negate(trits)

        assert (negated == 0).all()

    def test_negate_symmetric(self):
        """Test that negating twice gives original."""
        trits = BalancedTernaryArithmetic.from_decimal(torch.tensor([13]), num_trits=9)

        double_negated = BalancedTernaryArithmetic.negate(BalancedTernaryArithmetic.negate(trits))

        assert torch.equal(trits, double_negated)

    def test_batch_operations(self):
        """Test batch processing."""
        decimals = torch.tensor([1, 2, 3, 4, 5])

        trits = BalancedTernaryArithmetic.from_decimal(decimals, num_trits=9)

        assert trits.shape == (5, 9)

        # Roundtrip
        recovered = BalancedTernaryArithmetic.to_decimal(trits)
        assert torch.allclose(decimals.float(), recovered.float())

    def test_large_numbers(self):
        """Test with larger numbers within 9-trit range."""
        # 9-trit range: -9841 to +9841 (3^9 = 19683 values)
        decimals = torch.tensor([1000, -1000, 5000, -5000, 9000])

        trits = BalancedTernaryArithmetic.from_decimal(decimals, num_trits=9)
        recovered = BalancedTernaryArithmetic.to_decimal(trits)

        assert torch.allclose(decimals.float(), recovered.float())

    def test_range_limits(self):
        """Test maximum representable values with 9 trits."""
        # Maximum: 3^9 / 2 ≈ 9841
        max_val = 9841
        min_val = -9841

        trits_max = BalancedTernaryArithmetic.from_decimal(torch.tensor([max_val]), num_trits=9)
        trits_min = BalancedTernaryArithmetic.from_decimal(torch.tensor([min_val]), num_trits=9)

        recovered_max = BalancedTernaryArithmetic.to_decimal(trits_max)
        recovered_min = BalancedTernaryArithmetic.to_decimal(trits_min)

        assert recovered_max.item() == max_val
        assert recovered_min.item() == min_val

    def test_unique_representation(self):
        """Test that each number has unique representation."""
        # Different numbers should have different representations
        decimals = torch.tensor([1, 2, 3])

        trits = BalancedTernaryArithmetic.from_decimal(decimals, num_trits=9)

        # Each should be different
        assert not torch.equal(trits[0], trits[1])
        assert not torch.equal(trits[1], trits[2])
        assert not torch.equal(trits[0], trits[2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
