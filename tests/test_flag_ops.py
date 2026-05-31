"""
Unit tests for enable_flag_ops() function.

Tests the registration of custom operators to torch.ops.flag_ops namespace.
"""

import pytest
import torch

import flag_gems
from flag_gems.patches.flag_ops import enable_flag_ops


@pytest.mark.flag_ops
def test_registers_operators_to_torch_ops_namespace():
    """After enable_flag_ops(), operators should be accessible via torch.ops.flag_ops."""
    enable_flag_ops()

    # All registered operators should exist
    assert hasattr(torch.ops, "flag_ops")
    assert hasattr(torch.ops.flag_ops, "silu_and_mul")
    assert hasattr(torch.ops.flag_ops, "silu_and_mul_out")
    assert hasattr(torch.ops.flag_ops, "silu_and_mul_with_clamp")
    assert hasattr(torch.ops.flag_ops, "silu_and_mul_with_clamp_out")
    assert hasattr(torch.ops.flag_ops, "cutlass_scaled_mm")


@pytest.mark.flag_ops
def test_idempotent_multiple_calls():
    """Calling enable_flag_ops() multiple times should not raise errors."""
    enable_flag_ops()
    enable_flag_ops()  # Should not raise
    enable_flag_ops()  # Should not raise

    assert hasattr(torch.ops.flag_ops, "silu_and_mul")


@pytest.mark.flag_ops
def test_silu_and_mul_callable():
    """torch.ops.flag_ops.silu_and_mul should be callable with tensors."""
    enable_flag_ops()

    x = torch.randn(4, 8, dtype=torch.float16, device=flag_gems.device)
    y = torch.randn(4, 8, dtype=torch.float16, device=flag_gems.device)

    result = torch.ops.flag_ops.silu_and_mul(x, y)

    assert result.shape == x.shape
    assert result.dtype == x.dtype


@pytest.mark.flag_ops
def test_silu_and_mul_out_callable():
    """torch.ops.flag_ops.silu_and_mul_out should write to output tensor."""
    enable_flag_ops()

    x = torch.randn(4, 8, dtype=torch.float16, device=flag_gems.device)
    y = torch.randn(4, 8, dtype=torch.float16, device=flag_gems.device)
    out = torch.empty_like(x)

    result = torch.ops.flag_ops.silu_and_mul_out(x, y, out=out)

    assert result is out
    assert result.shape == x.shape


@pytest.mark.flag_ops
def test_silu_and_mul_with_clamp_callable():
    """torch.ops.flag_ops.silu_and_mul_with_clamp should accept limit parameter."""
    enable_flag_ops()

    x = torch.randn(4, 8, dtype=torch.float16, device=flag_gems.device)
    y = torch.randn(4, 8, dtype=torch.float16, device=flag_gems.device)
    limit = 1.0

    result = torch.ops.flag_ops.silu_and_mul_with_clamp(x, y, limit)

    assert result.shape == x.shape
    assert result.dtype == x.dtype


@pytest.mark.flag_ops
def test_silu_and_mul_with_clamp_out_callable():
    """torch.ops.flag_ops.silu_and_mul_with_clamp_out should write to output tensor."""
    enable_flag_ops()

    x = torch.randn(4, 8, dtype=torch.float16, device=flag_gems.device)
    y = torch.randn(4, 8, dtype=torch.float16, device=flag_gems.device)
    out = torch.empty_like(x)
    limit = 1.0

    result = torch.ops.flag_ops.silu_and_mul_with_clamp_out(x, y, out=out, limit=limit)

    assert result is out
    assert result.shape == x.shape


@pytest.mark.flag_ops
def test_silu_and_mul_correctness():
    """torch.ops.flag_ops.silu_and_mul should compute silu(x) * y."""
    enable_flag_ops()

    x = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32, device=flag_gems.device)
    y = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=flag_gems.device)

    result = torch.ops.flag_ops.silu_and_mul(x, y)
    expected = torch.nn.functional.silu(x) * y

    torch.testing.assert_close(result, expected)


@pytest.mark.flag_ops
def test_silu_and_mul_with_clamp_correctness():
    """torch.ops.flag_ops.silu_and_mul_with_clamp should clamp the result."""
    enable_flag_ops()

    x = torch.tensor([10.0, 20.0, -1.0], dtype=torch.float32, device=flag_gems.device)
    y = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=flag_gems.device)
    limit = 5.0

    result = torch.ops.flag_ops.silu_and_mul_with_clamp(x, y, limit)

    assert result.max().item() <= limit
    assert result.min().item() >= -limit


@pytest.mark.flag_ops
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_silu_and_mul_dtypes(dtype):
    """torch.ops.flag_ops.silu_and_mul should work with different dtypes."""
    enable_flag_ops()

    x = torch.randn(4, 8, dtype=dtype, device=flag_gems.device)
    y = torch.randn(4, 8, dtype=dtype, device=flag_gems.device)

    result = torch.ops.flag_ops.silu_and_mul(x, y)

    assert result.dtype == dtype
    assert result.shape == x.shape


@pytest.mark.flag_ops
@pytest.mark.parametrize("shape", [(4,), (4, 8), (2, 4, 8), (2, 2, 4, 8)])
def test_silu_and_mul_shapes(shape):
    """torch.ops.flag_ops.silu_and_mul should work with different shapes."""
    enable_flag_ops()

    x = torch.randn(shape, dtype=torch.float16, device=flag_gems.device)
    y = torch.randn(shape, dtype=torch.float16, device=flag_gems.device)

    result = torch.ops.flag_ops.silu_and_mul(x, y)

    assert result.shape == shape
