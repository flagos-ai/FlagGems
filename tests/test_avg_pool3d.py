"""Test cases for avg_pool3d operator."""
import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference
from .conftest import QUICK_MODE

# Test configurations for avg_pool3d
# Format: (shape, kernel_size, stride, padding, ceil_mode, count_include_pad)
AVGPOOL3D_CONFIGS = [
    # ===== (Small Size) =====
    # Minimal 3D input
    ((1, 1, 4, 4, 4), 2, 1, 0, False, True),
    # Small 3D input with padding
    ((2, 3, 8, 8, 8), 3, 2, 1, False, True),
    # ===== (Medium Size) =====
    # Regular 3D input - basic pooling
    ((2, 3, 16, 16, 16), 3, 2, 1, False, True),
    # Non-cubic kernel
    ((1, 8, 8, 16, 16), (2, 3, 3), (1, 2, 2), 1, False, True),
    # No padding
    ((2, 8, 16, 16, 16), 2, 2, 0, False, True),
    # Non-uniform padding
    ((2, 8, 16, 20, 24), 2, 2, (1, 0, 1), False, True),
    # ===== (Large Size) =====
    # Test ceil_mode
    ((2, 4, 15, 15, 15), 3, 2, 1, True, True),
    # Typical 3D CNN layer
    ((1, 32, 28, 28, 28), 3, 2, 1, False, True),
    # Large 3D volume
    ((1, 16, 32, 32, 32), 3, 2, 1, False, True),
    # ===== (Special Parameter Tests) =====
    # Single element kernel
    ((2, 4, 8, 8, 8), 1, 1, 0, False, True),
    # Test count_include_pad=False
    ((2, 4, 16, 16, 16), 3, 2, 1, False, False),
    # Large kernel
    ((2, 4, 16, 16, 16), 5, 2, 2, False, True),
    # Stride=1 (no downsampling)
    ((2, 4, 8, 8, 8), 3, 1, 1, False, True),
]

# Quick mode: use fewer test cases
if QUICK_MODE:
    AVGPOOL3D_CONFIGS = [AVGPOOL3D_CONFIGS[0], AVGPOOL3D_CONFIGS[2]]
    FLOAT_DTYPES_TEST = [torch.float32]
else:
    FLOAT_DTYPES_TEST = FLOAT_DTYPES


@pytest.mark.avg_pool3d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, ceil_mode, count_include_pad",
    AVGPOOL3D_CONFIGS,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_avg_pool3d_forward(
    shape, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype
):
    """Test forward pass of avg_pool3d."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    # PyTorch reference
    ref_out = torch.nn.functional.avg_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )

    # FlagGems implementation
    res_out = flag_gems.avg_pool3d(
        inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.avg_pool3d_backward
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, ceil_mode, count_include_pad",
    AVGPOOL3D_CONFIGS,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_avg_pool3d_backward(
    shape, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype
):
    """Test backward pass of avg_pool3d."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, upcast=True)

    # PyTorch reference forward
    ref_out = torch.nn.functional.avg_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )

    # FlagGems forward
    res_out = flag_gems.avg_pool3d(
        inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )

    # Create gradient
    out_grad = torch.randn_like(res_out, device=flag_gems.device)
    ref_grad = to_reference(out_grad, upcast=True)

    # PyTorch reference backward
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    # FlagGems backward
    res_in_grad = flag_gems.avg_pool3d_backward(
        out_grad,
        inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=None,
    )

    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.avg_pool3d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_avg_pool3d_edge_cases(dtype):
    """Test edge cases for avg_pool3d."""
    # Test with negative values
    inp = torch.randn((2, 4, 8, 8, 8), dtype=dtype, device=flag_gems.device) - 5.0
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.avg_pool3d(ref_inp, kernel_size=2, stride=2)
    res_out = flag_gems.avg_pool3d(inp, kernel_size=2, stride=2)

    gems_assert_close(res_out, ref_out, dtype)

    # Test with all same values
    inp = torch.ones((1, 2, 4, 4, 4), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.avg_pool3d(ref_inp, kernel_size=2, stride=2)
    res_out = flag_gems.avg_pool3d(inp, kernel_size=2, stride=2)

    gems_assert_close(res_out, ref_out, dtype)

    # Test with zeros
    inp = torch.zeros((1, 2, 4, 4, 4), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.avg_pool3d(ref_inp, kernel_size=2, stride=2)
    res_out = flag_gems.avg_pool3d(inp, kernel_size=2, stride=2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.avg_pool3d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_avg_pool3d_large_input(dtype):
    """Test avg_pool3d with large input sizes."""
    # Large 3D volume
    shape = (1, 32, 64, 64, 64)
    kernel_size = 3
    stride = 2
    padding = 1

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.avg_pool3d(
        ref_inp, kernel_size=kernel_size, stride=stride, padding=padding
    )
    res_out = flag_gems.avg_pool3d(
        inp, kernel_size=kernel_size, stride=stride, padding=padding
    )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.avg_pool3d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_avg_pool3d_different_strides(dtype):
    """Test avg_pool3d with different stride configurations."""
    shape = (2, 4, 16, 16, 16)
    kernel_size = 3

    # Test different stride values
    for stride in [1, 2, 3]:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = to_reference(inp, True)

        ref_out = torch.nn.functional.avg_pool3d(
            ref_inp, kernel_size=kernel_size, stride=stride
        )
        res_out = flag_gems.avg_pool3d(inp, kernel_size=kernel_size, stride=stride)

        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.avg_pool3d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_avg_pool3d_different_paddings(dtype):
    """Test avg_pool3d with different padding configurations."""
    shape = (2, 4, 16, 16, 16)
    kernel_size = 3
    stride = 2

    # Test different padding values
    for padding in [0, 1]:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = to_reference(inp, True)

        ref_out = torch.nn.functional.avg_pool3d(
            ref_inp, kernel_size=kernel_size, stride=stride, padding=padding
        )
        res_out = flag_gems.avg_pool3d(
            inp, kernel_size=kernel_size, stride=stride, padding=padding
        )

        gems_assert_close(res_out, ref_out, dtype)

    # Test with larger kernel_size to allow larger padding
    kernel_size = 5
    for padding in [0, 1, 2]:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = to_reference(inp, True)

        ref_out = torch.nn.functional.avg_pool3d(
            ref_inp, kernel_size=kernel_size, stride=stride, padding=padding
        )
        res_out = flag_gems.avg_pool3d(
            inp, kernel_size=kernel_size, stride=stride, padding=padding
        )

        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.avg_pool3d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_avg_pool3d_different_kernel_sizes(dtype):
    """Test avg_pool3d with different kernel_size configurations."""
    shape = (2, 4, 16, 16, 16)
    stride = 2
    padding = 1

    # Test different kernel sizes
    for kernel_size in [2, 3, 4, 5]:
        # Adjust padding to be valid
        max_padding = (kernel_size - 1) // 2
        test_padding = min(padding, max_padding)

        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = to_reference(inp, True)

        ref_out = torch.nn.functional.avg_pool3d(
            ref_inp, kernel_size=kernel_size, stride=stride, padding=test_padding
        )
        res_out = flag_gems.avg_pool3d(
            inp, kernel_size=kernel_size, stride=stride, padding=test_padding
        )

        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.avg_pool3d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_avg_pool3d_divisor_override(dtype):
    """Test avg_pool3d with divisor_override parameter."""
    shape = (2, 4, 8, 8, 8)
    kernel_size = 2
    stride = 2

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    # Test with divisor_override
    divisor = 10
    ref_out = torch.nn.functional.avg_pool3d(
        ref_inp, kernel_size=kernel_size, stride=stride, divisor_override=divisor
    )
    res_out = flag_gems.avg_pool3d(
        inp, kernel_size=kernel_size, stride=stride, divisor_override=divisor
    )

    gems_assert_close(res_out, ref_out, dtype)
