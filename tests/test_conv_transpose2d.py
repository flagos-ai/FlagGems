import os

import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference

# Test shapes: (input_shape, weight_shape, groups)
# Weight shape for transposed conv: (in_channels, out_channels_per_group, kH, kW)
SHAPE_CONV_TRANSPOSE2D = [
    # Small sizes
    ((1, 2, 3, 3), (2, 1, 2, 2), 1),
    ((2, 4, 4, 4), (4, 2, 2, 2), 1),
    # Regular sizes
    ((4, 8, 8, 8), (8, 16, 3, 3), 1),
    ((8, 16, 16, 16), (16, 32, 3, 3), 1),
    ((4, 32, 32, 32), (32, 64, 3, 3), 1),
    # Large sizes
    ((2, 64, 64, 64), (64, 128, 3, 3), 1),
    ((1, 128, 128, 128), (128, 64, 3, 3), 1),
    # Different kernel sizes
    ((4, 8, 8, 8), (8, 16, 5, 5), 1),
    ((4, 16, 16, 16), (16, 32, 7, 7), 1),
    # Groups
    ((4, 8, 8, 8), (8, 4, 3, 3), 2),
    ((4, 16, 16, 16), (16, 8, 3, 3), 2),
    ((4, 32, 32, 32), (32, 4, 3, 3), 4),
]


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONV_TRANSPOSE2D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("output_padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv_transpose2d(
    shape, kernel, groups, stride, padding, output_padding, dtype, dilation, bias
):
    """Test conv_transpose2d accuracy with various parameter combinations."""
    # Skip invalid combinations
    if output_padding >= stride:
        pytest.skip("output_padding must be smaller than stride")
    if output_padding >= dilation:
        pytest.skip("output_padding must be smaller than dilation")

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False

    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_weight = to_reference(weight, True)

    if bias:
        bias_tensor = torch.randn(
            [kernel[1] * groups],
            dtype=dtype,
            device=flag_gems.device,
            requires_grad=True,
        )
        bias_ref = to_reference(bias_tensor, True)
    else:
        bias_tensor = None
        bias_ref = None

    # Reference output
    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    ).to(dtype)

    # FlagGems output
    res_out = flag_gems.conv_transpose2d(
        inp,
        weight,
        bias=bias_tensor,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )

    gems_assert_close(res_out, ref_out, dtype)

    # Test backward pass
    out_grad = torch.randn_like(ref_out).to(flag_gems.device)
    ref_grad = to_reference(out_grad, True)

    if bias_tensor is not None:
        (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, bias_ref), ref_grad
        )
        (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
            res_out, (inp, weight, bias_tensor), out_grad
        )
        gems_assert_close(res_bias_grad, ref_bias_grad, dtype)
    else:
        (ref_in_grad, ref_weight_grad) = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight), ref_grad
        )
        (res_in_grad, res_weight_grad) = torch.autograd.grad(
            res_out, (inp, weight), out_grad
        )

    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=kernel[2])
    gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=kernel[0])

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


# Edge case tests
@pytest.mark.conv_transpose2d_edge
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_edge_case_1x1_kernel(dtype):
    """Test with 1x1 kernel."""
    shape = (2, 4, 8, 8)
    kernel = (4, 8, 1, 1)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, False)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, False)

    ref_out = torch.nn.functional.conv_transpose2d(ref_inp, ref_weight)
    res_out = flag_gems.conv_transpose2d(inp, weight)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv_transpose2d_edge
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_edge_case_large_stride(dtype):
    """Test with large stride."""
    shape = (2, 4, 4, 4)
    kernel = (4, 8, 3, 3)
    stride = 4

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, False)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, False)

    ref_out = torch.nn.functional.conv_transpose2d(ref_inp, ref_weight, stride=stride)
    res_out = flag_gems.conv_transpose2d(inp, weight, stride=stride)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv_transpose2d_edge
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_edge_case_asymmetric_params(dtype):
    """Test with asymmetric stride and padding."""
    shape = (2, 4, 8, 8)
    kernel = (4, 8, 3, 3)
    stride = (2, 3)
    padding = (1, 2)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, False)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, False)

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp, ref_weight, stride=stride, padding=padding
    )
    res_out = flag_gems.conv_transpose2d(inp, weight, stride=stride, padding=padding)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv_transpose2d_edge
@pytest.mark.parametrize("dtype", [torch.float32])
def test_edge_case_zero_values(dtype):
    """Test with zero input values."""
    shape = (2, 4, 8, 8)
    kernel = (4, 8, 3, 3)

    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, False)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, False)

    ref_out = torch.nn.functional.conv_transpose2d(ref_inp, ref_weight)
    res_out = flag_gems.conv_transpose2d(inp, weight)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv_transpose2d_edge
@pytest.mark.parametrize("dtype", [torch.float32])
def test_edge_case_negative_values(dtype):
    """Test with negative input values."""
    shape = (2, 4, 8, 8)
    kernel = (4, 8, 3, 3)

    inp = -torch.abs(torch.randn(shape, dtype=dtype, device=flag_gems.device))
    ref_inp = to_reference(inp, False)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, False)

    ref_out = torch.nn.functional.conv_transpose2d(ref_inp, ref_weight)
    res_out = flag_gems.conv_transpose2d(inp, weight)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv_transpose2d_edge
@pytest.mark.parametrize("dtype", [torch.float32])
def test_edge_case_single_batch(dtype):
    """Test with batch size 1."""
    shape = (1, 4, 8, 8)
    kernel = (4, 8, 3, 3)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, False)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, False)

    ref_out = torch.nn.functional.conv_transpose2d(ref_inp, ref_weight)
    res_out = flag_gems.conv_transpose2d(inp, weight)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv_transpose2d_edge
@pytest.mark.parametrize("dtype", [torch.float32])
def test_edge_case_large_batch(dtype):
    """Test with large batch size."""
    shape = (32, 4, 8, 8)
    kernel = (4, 8, 3, 3)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, False)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, False)

    ref_out = torch.nn.functional.conv_transpose2d(ref_inp, ref_weight)
    res_out = flag_gems.conv_transpose2d(inp, weight)

    gems_assert_close(res_out, ref_out, dtype)


# Dimension tests
@pytest.mark.conv_transpose2d_dims
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "shape,kernel",
    [
        # 2D only (batch=1, channels, height, width)
        ((1, 1, 8, 8), (1, 1, 3, 3)),
        ((1, 3, 16, 16), (3, 3, 3, 3)),
        ((1, 64, 32, 32), (64, 64, 3, 3)),
    ],
)
def test_various_dimensions(shape, kernel, dtype):
    """Test various input dimensions."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, False)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, False)

    ref_out = torch.nn.functional.conv_transpose2d(ref_inp, ref_weight)
    res_out = flag_gems.conv_transpose2d(inp, weight)

    gems_assert_close(res_out, ref_out, dtype)
