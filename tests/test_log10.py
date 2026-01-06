"""
Comprehensive test suite for log10 operator.

This test suite covers:
- Functional correctness against PyTorch
- Edge cases (zero, negative, very small/large values)
- Different tensor shapes and dimensions
- Different data types (float32, float16)
- Boundary conditions
- Error handling
"""

import pytest
import torch
from operators.log10 import log10


# Test fixtures
@pytest.fixture
def device():
    """Return CUDA device if available, otherwise skip tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def dtype_float32():
    return torch.float32


@pytest.fixture
def dtype_float16():
    return torch.float16


class TestLog10FunctionalCorrectness:
    """Test functional correctness against PyTorch reference."""

    def test_basic_values_float32(self, device, dtype_float32):
        """Test basic log10 values with float32."""
        x = torch.tensor([1.0, 10.0, 100.0, 1000.0], device=device, dtype=dtype_float32)
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert torch.allclose(y_triton, y_torch, rtol=1e-4, atol=1.3e-6)

    def test_basic_values_float16(self, device, dtype_float16):
        """Test basic log10 values with float16."""
        x = torch.tensor([1.0, 10.0, 100.0, 1000.0], device=device, dtype=dtype_float16)
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert torch.allclose(y_triton, y_torch, rtol=1e-4, atol=1e-3)

    def test_small_values_float32(self, device, dtype_float32):
        """Test small positive values."""
        x = torch.tensor([0.1, 0.01, 1e-6, 1e-9], device=device, dtype=dtype_float32)
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert torch.allclose(y_triton, y_torch, rtol=1e-4, atol=1.3e-6)

    def test_large_values_float32(self, device, dtype_float32):
        """Test large positive values."""
        x = torch.tensor([1e6, 1e9, 1e12], device=device, dtype=dtype_float32)
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert torch.allclose(y_triton, y_torch, rtol=1e-4, atol=1.3e-6)

    def test_random_tensor_float32(self, device, dtype_float32):
        """Test with random positive values."""
        torch.manual_seed(42)
        x = torch.rand(1000, device=device, dtype=dtype_float32) * 1000 + 1e-6
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert torch.allclose(y_triton, y_torch, rtol=1e-4, atol=1.3e-6)


class TestLog10EdgeCases:
    """Test edge cases and boundary conditions."""

    def test_value_one(self, device, dtype_float32):
        """Test log10(1) = 0."""
        x = torch.tensor([1.0], device=device, dtype=dtype_float32)
        y = log10(x)
        assert torch.allclose(y, torch.tensor([0.0], device=device), atol=1e-6)

    def test_value_ten(self, device, dtype_float32):
        """Test log10(10) = 1."""
        x = torch.tensor([10.0], device=device, dtype=dtype_float32)
        y = log10(x)
        assert torch.allclose(y, torch.tensor([1.0], device=device), atol=1e-6)

    def test_very_small_positive(self, device, dtype_float32):
        """Test very small positive values."""
        x = torch.tensor([1e-15, 1e-12, 1e-9], device=device, dtype=dtype_float32)
        y_triton = log10(x)
        y_torch = torch.log10(x)
        # For very small values, allow larger tolerance
        assert torch.allclose(y_triton, y_torch, rtol=1e-3, atol=1e-3)

    def test_negative_values(self, device, dtype_float32):
        """Test negative values (should produce NaN)."""
        x = torch.tensor([-1.0, -10.0, -100.0], device=device, dtype=dtype_float32)
        y_triton = log10(x)
        y_torch = torch.log10(x)
        # Both should produce NaN for negative inputs
        assert torch.isnan(y_triton).all()
        assert torch.isnan(y_torch).all()

    def test_zero_value(self, device, dtype_float32):
        """Test zero value (should produce -inf)."""
        x = torch.tensor([0.0], device=device, dtype=dtype_float32)
        y_triton = log10(x)
        y_torch = torch.log10(x)
        # Both should produce -inf
        assert torch.isinf(y_triton).all() and (y_triton < 0).all()
        assert torch.isinf(y_torch).all() and (y_torch < 0).all()

    def test_inf_values(self, device, dtype_float32):
        """Test infinity values."""
        x = torch.tensor([float('inf')], device=device, dtype=dtype_float32)
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert torch.isinf(y_triton).all() and (y_triton > 0).all()
        assert torch.isinf(y_torch).all() and (y_torch > 0).all()

    def test_nan_values(self, device, dtype_float32):
        """Test NaN values."""
        x = torch.tensor([float('nan')], device=device, dtype=dtype_float32)
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert torch.isnan(y_triton).all()
        assert torch.isnan(y_torch).all()


class TestLog10TensorShapes:
    """Test different tensor shapes and dimensions."""

    def test_scalar_tensor(self, device, dtype_float32):
        """Test scalar tensor (0D)."""
        x = torch.tensor(10.0, device=device, dtype=dtype_float32)
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert torch.allclose(y_triton, y_torch, rtol=1e-4, atol=1.3e-6)

    def test_1d_tensor_small(self, device, dtype_float32):
        """Test 1D tensor with small size."""
        x = torch.rand(8, device=device, dtype=dtype_float32) * 100 + 1e-6
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert torch.allclose(y_triton, y_torch, rtol=1e-4, atol=1.3e-6)

    def test_1d_tensor_regular(self, device, dtype_float32):
        """Test 1D tensor with regular size."""
        x = torch.rand(256, device=device, dtype=dtype_float32) * 100 + 1e-6
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert torch.allclose(y_triton, y_torch, rtol=1e-4, atol=1.3e-6)

    def test_1d_tensor_large(self, device, dtype_float32):
        """Test 1D tensor with large size."""
        x = torch.rand(1048576, device=device, dtype=dtype_float32) * 100 + 1e-6
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert torch.allclose(y_triton, y_torch, rtol=1e-4, atol=1.3e-6)

    def test_2d_tensor_small(self, device, dtype_float32):
        """Test 2D tensor with small size."""
        x = torch.rand(8, 8, device=device, dtype=dtype_float32) * 100 + 1e-6
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert torch.allclose(y_triton, y_torch, rtol=1e-4, atol=1.3e-6)

    def test_2d_tensor_regular(self, device, dtype_float32):
        """Test 2D tensor with regular size."""
        x = torch.rand(64, 64, device=device, dtype=dtype_float32) * 100 + 1e-6
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert torch.allclose(y_triton, y_torch, rtol=1e-4, atol=1.3e-6)

    def test_2d_tensor_large(self, device, dtype_float32):
        """Test 2D tensor with large size."""
        x = torch.rand(1024, 1024, device=device, dtype=dtype_float32) * 100 + 1e-6
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert torch.allclose(y_triton, y_torch, rtol=1e-4, atol=1.3e-6)

    def test_3d_tensor(self, device, dtype_float32):
        """Test 3D tensor."""
        x = torch.rand(32, 32, 32, device=device, dtype=dtype_float32) * 100 + 1e-6
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert torch.allclose(y_triton, y_torch, rtol=1e-4, atol=1.3e-6)

    def test_4d_tensor(self, device, dtype_float32):
        """Test 4D tensor."""
        x = torch.rand(16, 16, 16, 16, device=device, dtype=dtype_float32) * 100 + 1e-6
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert torch.allclose(y_triton, y_torch, rtol=1e-4, atol=1.3e-6)

    def test_5d_tensor(self, device, dtype_float32):
        """Test 5D tensor."""
        x = torch.rand(8, 8, 8, 8, 8, device=device, dtype=dtype_float32) * 100 + 1e-6
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert torch.allclose(y_triton, y_torch, rtol=1e-4, atol=1.3e-6)

    def test_empty_tensor(self, device, dtype_float32):
        """Test empty tensor."""
        x = torch.tensor([], device=device, dtype=dtype_float32)
        y_triton = log10(x)
        y_torch = torch.log10(x)
        assert y_triton.shape == y_torch.shape
        assert y_triton.numel() == 0


class TestLog10OutParameter:
    """Test the optional out parameter."""

    def test_out_parameter_float32(self, device, dtype_float32):
        """Test out parameter with float32."""
        x = torch.rand(100, device=device, dtype=dtype_float32) * 100 + 1e-6
        out = torch.empty_like(x)
        result = log10(x, out=out)
        y_torch = torch.log10(x)
        assert result is out
        assert torch.allclose(result, y_torch, rtol=1e-4, atol=1.3e-6)

    def test_out_parameter_float16(self, device, dtype_float16):
        """Test out parameter with float16."""
        x = torch.rand(100, device=device, dtype=dtype_float16) * 100 + 1e-6
        out = torch.empty_like(x)
        result = log10(x, out=out)
        y_torch = torch.log10(x)
        assert result is out
        assert torch.allclose(result, y_torch, rtol=1e-4, atol=1e-3)


class TestLog10ErrorHandling:
    """Test error handling for invalid inputs."""

    def test_cpu_tensor_error(self, dtype_float32):
        """Test that CPU tensor raises error."""
        x = torch.tensor([1.0, 10.0], dtype=dtype_float32)
        with pytest.raises(AssertionError, match="Input must be a CUDA tensor"):
            log10(x)

    def test_unsupported_dtype_error(self, device):
        """Test that unsupported dtype raises error."""
        x = torch.tensor([1.0, 10.0], device=device, dtype=torch.int32)
        with pytest.raises(AssertionError, match="Unsupported dtype"):
            log10(x)

    def test_out_shape_mismatch_error(self, device, dtype_float32):
        """Test that mismatched output shape raises error."""
        x = torch.rand(10, device=device, dtype=dtype_float32)
        out = torch.empty(20, device=device, dtype=dtype_float32)
        with pytest.raises(AssertionError, match="Output shape must match"):
            log10(x, out=out)

    def test_out_dtype_mismatch_error(self, device):
        """Test that mismatched output dtype raises error."""
        x = torch.rand(10, device=device, dtype=torch.float32)
        out = torch.empty(10, device=device, dtype=torch.float16)
        with pytest.raises(AssertionError, match="Output dtype must match"):
            log10(x, out=out)

    def test_out_cpu_error(self, device, dtype_float32):
        """Test that CPU output tensor raises error."""
        x = torch.rand(10, device=device, dtype=dtype_float32)
        out = torch.empty(10, dtype=dtype_float32)
        with pytest.raises(AssertionError, match="Output must be a CUDA tensor"):
            log10(x, out=out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

