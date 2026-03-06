"""
Test suite for log10 operator.
"""

import pytest
import torch
import numpy as np
import sys
sys.path.append('.')
from log10 import log10

# Test configurations
SHAPES = [
    (1,),                    # Small 1D
    (8, 8),                  # Small 2D
    (64, 64),                # Medium 2D
    (256, 256),              # Large 2D
    (1024,),                 # Large 1D
    (32, 32, 32),            # 3D
    (16, 16, 16, 16),        # 4D
    (1, 1, 1, 1, 1),         # 5D
]

DTYPES = [
    torch.float32,
    torch.float16,
]

DEVICES = ['cpu']
if torch.cuda.is_available():
    DEVICES.append('cuda')


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
def test_log10_forward(shape, dtype, device):
    """Test forward pass correctness"""
    # Skip invalid combinations
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create input tensor with various values
    x = torch.randn(shape, dtype=dtype, device=device) * 10
    x = torch.abs(x) + 1e-6  # Ensure positive for log10
    
    # Compute with our implementation
    y_custom = log10(x)
    
    # Compute with PyTorch reference
    y_ref = torch.log10(x)
    
    # Compare results
    if dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    else:
        atol, rtol = 1e-6, 1e-5
    
    torch.testing.assert_close(
        y_custom, y_ref,
        atol=atol,
        rtol=rtol,
        msg=f"Failed for shape={shape}, dtype={dtype}"
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_log10_backward(shape, dtype):
    """Test backward pass correctness"""
    x = torch.randn(shape, dtype=dtype, requires_grad=True) * 10
    x = torch.abs(x) + 1e-6
    
    # Custom implementation backward
    y_custom = log10(x)
    loss_custom = y_custom.sum()
    loss_custom.backward()
    grad_custom = x.grad.clone()
    x.grad = None
    
    # PyTorch reference backward
    y_ref = torch.log10(x)
    loss_ref = y_ref.sum()
    loss_ref.backward()
    grad_ref = x.grad
    
    # Compare gradients
    if dtype == torch.float16:
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-5, 1e-5
    
    torch.testing.assert_close(
        grad_custom, grad_ref,
        atol=atol,
        rtol=rtol,
        msg=f"Gradient mismatch for shape={shape}, dtype={dtype}"
    )


def test_log10_edge_cases():
    """Test edge cases"""
    test_cases = [
        (torch.tensor([1.0]), torch.tensor([0.0])),  # log10(1) = 0
        (torch.tensor([10.0]), torch.tensor([1.0])),  # log10(10) = 1
        (torch.tensor([100.0]), torch.tensor([2.0])),  # log10(100) = 2
        (torch.tensor([0.1]), torch.tensor([-1.0])),   # log10(0.1) = -1
        (torch.tensor([1e-6]), None),  # Very small positive
        (torch.tensor([1e6]), None),   # Very large positive
        (torch.tensor([]), torch.tensor([])),  # Empty tensor
        (torch.tensor(3.14), torch.tensor(0.4969)),  # Scalar input
    ]
    
    for x, expected in test_cases:
        if expected is None:
            expected = torch.log10(x)
        
        y = log10(x)
        torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-4)


def test_log10_exception_handling():
    """Test exception handling for invalid inputs"""
    # Test negative values (should produce NaN)
    x = torch.tensor([-1.0, -2.0])
    y = log10(x)
    assert torch.isnan(y).all(), "log10 of negative should be NaN"
    
    # Test zero (should produce -inf)
    x = torch.tensor([0.0])
    y = log10(x)
    assert torch.isinf(y).all() and (y < 0).all(), "log10(0) should be -inf"


@pytest.mark.parametrize("shape", SHAPES)
def test_log10_non_contiguous(shape):
    """Test with non-contiguous tensors"""
    x = torch.randn(shape) * 10
    x = torch.abs(x) + 1e-6
    
    # Create non-contiguous tensor
    if x.dim() >= 2:
        x_noncontig = x.t()
        if not x_noncontig.is_contiguous():
            y_custom = log10(x_noncontig)
            y_ref = torch.log10(x_noncontig)
            torch.testing.assert_close(y_custom, y_ref)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
