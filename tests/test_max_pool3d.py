"""Test cases for max_pool3d operator."""
import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)
from .conftest import QUICK_MODE

# Test configurations for max_pool3d
# Format: (shape, kernel_size, stride, padding, dilation, ceil_mode)
MAXPOOL3D_CONFIGS = [
    # Basic 3D pooling: 3x3x3 kernel, stride 2
    ((2, 3, 16, 16, 16), 3, 2, 1, 1, False),
    # Non-cubic kernel
    ((1, 8, 8, 16, 16), (2, 3, 3), (1, 2, 2), 1, 1, False),
    # Test ceil_mode
    ((2, 4, 15, 15, 15), 3, 2, 1, 1, True),
    # Test dilation
    ((1, 1, 10, 10, 10), 2, 1, 0, 2, False),
    # Larger case
    ((1, 32, 28, 28, 28), 3, 2, 1, 1, False),
    # No padding
    ((2, 8, 16, 16, 16), 2, 2, 0, 1, False),
    # Non-uniform padding
    ((2, 8, 16, 20, 24), 2, 2, (1, 0, 1), 1, False),
    # Small input
    ((1, 1, 4, 4, 4), 2, 1, 0, 1, False),
    # Single element kernel
    ((2, 4, 8, 8, 8), 1, 1, 0, 1, False),
]

# Quick mode: use fewer test cases
if QUICK_MODE:
    MAXPOOL3D_CONFIGS = [MAXPOOL3D_CONFIGS[0], MAXPOOL3D_CONFIGS[2]]
    FLOAT_DTYPES_TEST = [torch.float32]
else:
    FLOAT_DTYPES_TEST = FLOAT_DTYPES


@pytest.mark.max_pool3d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode", MAXPOOL3D_CONFIGS
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_max_pool3d_forward(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype
):
    """Test forward pass of max_pool3d."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    # PyTorch reference
    ref_out = torch.nn.functional.max_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=False,
    )

    # FlagGems implementation
    res_out = flag_gems.max_pool3d(
        inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=False,
    )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.max_pool3d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode", MAXPOOL3D_CONFIGS
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_max_pool3d_with_indices(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype
):
    """Test max_pool3d with return_indices=True."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    # PyTorch reference
    ref_out, ref_indices = torch.nn.functional.max_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )

    # FlagGems implementation
    res_out, res_indices = flag_gems.max_pool3d_with_indices(
        inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    gems_assert_close(res_out, ref_out, dtype)
    gems_assert_equal(res_indices, ref_indices)


@pytest.mark.max_pool3d_backward
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode", MAXPOOL3D_CONFIGS
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_max_pool3d_backward(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype
):
    """Test backward pass of max_pool3d."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, upcast=True)

    # PyTorch reference forward
    ref_out, ref_indices = torch.nn.functional.max_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )

    # FlagGems forward
    res_out, res_indices = flag_gems.max_pool3d_with_indices(
        inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    # Create gradient
    out_grad = torch.randn_like(res_out, device=flag_gems.device)
    ref_grad = to_reference(out_grad, upcast=True)

    # PyTorch reference backward
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    # FlagGems backward
    res_in_grad = flag_gems.max_pool3d_backward(
        out_grad,
        inp,
        res_indices,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.max_pool3d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_max_pool3d_edge_cases(dtype):
    """Test edge cases for max_pool3d."""
    # Test with negative values
    inp = torch.randn((2, 4, 8, 8, 8), dtype=dtype, device=flag_gems.device) - 5.0
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.max_pool3d(ref_inp, kernel_size=2, stride=2)
    res_out = flag_gems.max_pool3d(inp, kernel_size=2, stride=2)

    gems_assert_close(res_out, ref_out, dtype)

    # Test with all same values
    inp = torch.ones((1, 2, 4, 4, 4), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out, ref_indices = torch.nn.functional.max_pool3d(
        ref_inp, kernel_size=2, stride=2, return_indices=True
    )
    res_out, res_indices = flag_gems.max_pool3d_with_indices(
        inp, kernel_size=2, stride=2
    )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.max_pool3d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_max_pool3d_different_strides(dtype):
    """Test max_pool3d with different stride configurations."""
    shape = (2, 4, 16, 16, 16)
    kernel_size = 3

    # Test different stride values
    for stride in [1, 2, 3]:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = to_reference(inp, True)

        ref_out = torch.nn.functional.max_pool3d(
            ref_inp, kernel_size=kernel_size, stride=stride
        )
        res_out = flag_gems.max_pool3d(inp, kernel_size=kernel_size, stride=stride)

        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.max_pool3d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_max_pool3d_different_paddings(dtype):
    """Test max_pool3d with different padding configurations."""
    shape = (2, 4, 16, 16, 16)
    kernel_size = 3
    stride = 2

    # Test different padding values
    # Note: padding must be at most half of kernel_size (i.e., <= 1 for kernel_size=3)
    for padding in [0, 1]:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = to_reference(inp, True)

        ref_out = torch.nn.functional.max_pool3d(
            ref_inp, kernel_size=kernel_size, stride=stride, padding=padding
        )
        res_out = flag_gems.max_pool3d(
            inp, kernel_size=kernel_size, stride=stride, padding=padding
        )

        gems_assert_close(res_out, ref_out, dtype)

    # Test with larger kernel_size to allow larger padding
    kernel_size = 5
    for padding in [0, 1, 2]:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = to_reference(inp, True)

        ref_out = torch.nn.functional.max_pool3d(
            ref_inp, kernel_size=kernel_size, stride=stride, padding=padding
        )
        res_out = flag_gems.max_pool3d(
            inp, kernel_size=kernel_size, stride=stride, padding=padding
        )

        gems_assert_close(res_out, ref_out, dtype)
