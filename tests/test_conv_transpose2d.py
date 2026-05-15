"""
Unit tests for conv_transpose2d operator.

Test coverage:
- Input shapes: small (1,1,4,4), regular (4,3,32,32), medium, large
- Parameters: stride (1,2,3), padding (0,1,2), output_padding (0,1),
              dilation (1,2), groups (1,2,4)
- Data types: float32, float16, bfloat16
- Edge cases: 1x1 kernel, large kernel, empty batch (N=0), bias/no-bias
- Non-contiguous inputs
- Gradient verification (backward pass)
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# =============================================================================
# Shape definitions: (input_shape, weight_shape, groups)
# weight shape: (C_in, C_out/groups, kH, kW)
# =============================================================================
SHAPE_CONV_TRANSPOSE2D = [
    # Small: 1x1 input with 2x2 kernel
    ((1, 1, 4, 4), (1, 1, 2, 2), 1),
    # Regular: typical training shape
    ((4, 3, 32, 32), (3, 6, 3, 3), 1),
    # Medium: multi-channel
    ((2, 4, 8, 8), (4, 8, 3, 3), 1),
    # Groups=2
    ((2, 4, 8, 8), (4, 2, 3, 3), 2),
    # Groups=4
    ((2, 8, 8, 8), (8, 2, 3, 3), 4),
    # Larger with groups
    ((2, 8, 16, 16), (8, 4, 3, 3), 2),
    # 1x1 kernel (identity-like)
    ((2, 4, 8, 8), (4, 8, 1, 1), 1),
    # Larger kernel
    ((2, 4, 8, 8), (4, 8, 5, 5), 1),
]

# Edge case shapes
SHAPE_EDGE_CASES = [
    # Empty batch
    ((0, 4, 8, 8), (4, 8, 3, 3), 1),
    # Single channel in/out
    ((1, 1, 16, 16), (1, 1, 3, 3), 1),
    # Many channels
    ((1, 32, 8, 8), (32, 16, 3, 3), 1),
]


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONV_TRANSPOSE2D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_conv_transpose2d(
    shape, kernel, stride, padding, groups, dtype, dilation, bias
):
    """Test conv_transpose2d with various parameter combinations."""
    torch.backends.cudnn.allow_tf32 = False

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, True)

    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_weight = utils.to_reference(weight, True)

    if bias:
        out_c = kernel[1] * groups
        bias_tensor = torch.randn(
            [out_c], dtype=dtype, device=flag_gems.device, requires_grad=True
        )
        bias_ref = utils.to_reference(bias_tensor, True)
    else:
        bias_tensor = None
        bias_ref = None

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.conv_transpose2d(
            inp,
            weight,
            bias=bias_tensor,
            groups=groups,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    # Tolerance: reduce_dim accounts for C_in * kH * kW accumulation,
    # with 4x safety factor for different accumulation order and autotuning variance
    reduce_dim = kernel[0] * kernel[2] * kernel[3] * 4
    # bfloat16 has lower precision, use larger atol
    atol = 5e-4 if dtype == torch.bfloat16 else 1e-4
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim, atol=atol)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONV_TRANSPOSE2D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_conv_transpose2d_grad(
    shape, kernel, stride, padding, groups, dtype, dilation, bias
):
    """Test conv_transpose2d backward pass (gradient verification)."""
    torch.backends.cudnn.allow_tf32 = False

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    out_c = kernel[1] * groups
    bias_tensor = (
        torch.randn([out_c], dtype=dtype, device=flag_gems.device, requires_grad=True)
        if bias
        else None
    )

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    bias_ref = utils.to_reference(bias_tensor, True) if bias else None

    # Reference forward + backward
    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    out_grad = torch.randn_like(ref_out).to(flag_gems.device)
    ref_grad = utils.to_reference(out_grad, True)

    ref_grads = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight) + ((bias_ref,) if bias else ()), ref_grad
    )

    # Gems forward + backward
    with flag_gems.use_gems():
        res_out = torch.nn.functional.conv_transpose2d(
            inp,
            weight,
            bias=bias_tensor,
            groups=groups,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    res_grads = torch.autograd.grad(
        res_out,
        (inp, weight) + ((bias_tensor,) if bias else ()),
        out_grad,
    )

    # Check grad_input
    reduce_dim = kernel[0] * kernel[2] * kernel[3] * 2
    utils.gems_assert_close(res_grads[0], ref_grads[0], dtype, reduce_dim=reduce_dim)
    # Check grad_weight
    utils.gems_assert_close(res_grads[1], ref_grads[1], dtype, reduce_dim=reduce_dim)
    # Check grad_bias
    if bias:
        utils.gems_assert_close(res_grads[2], ref_grads[2], dtype)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONV_TRANSPOSE2D)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("padding", [1])
@pytest.mark.parametrize("output_padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_conv_transpose2d_output_padding(
    shape, kernel, stride, padding, output_padding, groups, dtype
):
    """Test conv_transpose2d with output_padding parameter."""
    torch.backends.cudnn.allow_tf32 = False

    if output_padding >= stride:
        pytest.skip("output_padding must be < stride")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.conv_transpose2d(
            inp,
            weight,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    "shape, kernel, groups",
    [
        ((2, 4, 8, 8), (4, 8, 3, 3), 1),
    ],
)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", [1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_conv_transpose2d_noncontiguous(shape, kernel, stride, padding, groups, dtype):
    """Test conv_transpose2d with non-contiguous input."""
    torch.backends.cudnn.allow_tf32 = False

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)

    # Make input non-contiguous by transposing spatial dims
    inp_t = inp.transpose(2, 3).contiguous().transpose(2, 3)

    ref_inp = utils.to_reference(inp_t, True)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp, ref_weight, stride=stride, padding=padding, groups=groups
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.conv_transpose2d(
            inp_t, weight, stride=stride, padding=padding, groups=groups
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_EDGE_CASES)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("bias", [True, False])
def test_conv_transpose2d_edge_cases(
    shape, kernel, stride, padding, groups, dtype, bias
):
    """Test conv_transpose2d with edge case shapes (empty batch, 1ch, many ch)."""
    torch.backends.cudnn.allow_tf32 = False

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)

    out_c = kernel[1] * groups
    bias_tensor = (
        torch.randn([out_c], dtype=dtype, device=flag_gems.device) if bias else None
    )

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    bias_ref = utils.to_reference(bias_tensor, True) if bias else None

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        stride=stride,
        padding=padding,
        groups=groups,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.conv_transpose2d(
            inp,
            weight,
            bias=bias_tensor,
            stride=stride,
            padding=padding,
            groups=groups,
        )

    if shape[0] == 0:
        # Empty batch: just check shape matches
        assert res_out.shape == ref_out.shape
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_conv_transpose2d_stride(stride, dtype):
    """Test conv_transpose2d with various stride values including stride=3."""
    torch.backends.cudnn.allow_tf32 = False

    inp = torch.randn(2, 4, 8, 8, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(4, 8, 3, 3, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp, ref_weight, stride=stride, padding=1
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.conv_transpose2d(
            inp, weight, stride=stride, padding=1
        )

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=4 * 3 * 3 * 2)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("padding", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_conv_transpose2d_padding(padding, dtype):
    """Test conv_transpose2d with various padding values including padding=2."""
    torch.backends.cudnn.allow_tf32 = False

    inp = torch.randn(2, 4, 8, 8, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(4, 8, 3, 3, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp, ref_weight, stride=1, padding=padding
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.conv_transpose2d(
            inp, weight, stride=1, padding=padding
        )

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=4 * 3 * 3 * 2)
