import os

import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference

SHAPE_CONV1D = [
    ((32, 2, 4), (17, 2, 2)),
    ((32, 15, 6), (17, 15, 2)),
    ((64, 64, 64), (128, 64, 7)),
    # ((32, 16, 1024), (1024, 16, 8)),
    # ((32, 12, 9), (17, 12, 3)),
    # ((32, 6, 6), (64, 6, 2)),
]


# @pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.conv1d
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV1D)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("padding", [1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_conv1d(shape, kernel, stride, padding, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv1d(
        ref_inp, ref_weight, bias=None, stride=stride, padding=padding, dilation=1
    )

    res_out = flag_gems.conv1d(
        inp, weight, bias=None, stride=stride, padding=padding, dilation=1
    )
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.conv1d_padding
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV1D)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_conv1d_padding(shape, kernel, stride, padding, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv1d(
        ref_inp, ref_weight, bias=None, stride=stride, padding=padding, dilation=1
    )

    res_out = flag_gems.conv1d(
        inp, weight, bias=None, stride=stride, padding=padding, dilation=1
    )
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


SHAPE_CONV1D_DILATION = [
    ((32, 2, 16), (17, 2, 3)),
    ((32, 15, 32), (17, 15, 3)),
    ((64, 64, 64), (128, 64, 3)),
]


@pytest.mark.conv1d
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV1D_DILATION)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("dilation", [1, 2, (1,), (2,)])
def test_accuracy_conv1d_dilation(shape, kernel, stride, padding, dtype, dilation):
    """Test conv1d with various dilation values, including tuple form.

    This specifically tests the fix where conv1d must properly convert dilation
    to a 2D tuple before delegating to conv2d. Previously, passing dilation as
    a single-element tuple (e.g., (1,)) would cause a ValueError in conv2d.
    """
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv1d(
        ref_inp,
        ref_weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    res_out = flag_gems.conv1d(
        inp, weight, bias=None, stride=stride, padding=padding, dilation=dilation
    )
    gems_assert_close(res_out, ref_out, dtype)


SHAPE_CONV2D = [
    ((1, 2, 5, 5), (1, 2, 3, 3), 1),
    ((2, 3, 9, 9), (1, 3, 3, 3), 1),
    ((32, 8, 8, 8), (32, 8, 2, 2), 1),
    # ((2, 2, 3, 3), (1, 2, 2, 2), 1),
    # ((18, 16, 4, 4), (16, 16, 2, 2), 1),
    # ((9, 16, 4, 4), (128, 4, 2, 2), 4),
    # ((32, 16, 8, 8), (32, 4, 4, 4), 4),
    # ((18, 16, 4, 4), (16, 8, 2, 2), 2),
    # ((9, 16, 4, 4), (128, 8, 2, 2), 2),
    # ((32, 8, 8, 8), (32, 8, 3, 3), 1),
    # ((18, 16, 5, 5), (16, 16, 3, 3), 1),
    # ((9, 16, 7, 7), (128, 4, 3, 3), 4),
    # ((32, 16, 9, 9), (32, 4, 5, 5), 4),
    # ((18, 16, 11, 11), (16, 8, 3, 3), 2),
    # ((9, 16, 6, 6), (128, 8, 3, 3), 2),
]


# @pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RESULT TODOFIX")
# @pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.conv2d
@pytest.mark.parametrize("shape, kernel,groups", SHAPE_CONV2D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv2d(shape, kernel, stride, padding, groups, dtype, dilation, bias):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"
    if flag_gems.vendor_name == "hygon":
        os.environ["TRITON_HIP_USE_NEW_STREAM_PIPELINE"] = "0"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    if bias is True:
        bias = torch.randn(
            [weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=True
        )
        bias_ref = to_reference(bias, True)
    else:
        bias = None
        bias_ref = None

    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv2d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv2d(
        inp,
        weight,
        bias=bias,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(ref_out).to(flag_gems.device)

    ref_grad = to_reference(out_grad, True)
    if bias is not None:
        ref_in_grad, ref_weight_grad, ref_bias_grad = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, bias_ref), ref_grad
        )
        res_in_grad, res_weight_grad, res_bias_grad = torch.autograd.grad(
            res_out, (inp, weight, bias), out_grad
        )
    else:
        ref_in_grad, ref_weight_grad = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight), ref_grad
        )
        res_in_grad, res_weight_grad = torch.autograd.grad(
            res_out, (inp, weight), out_grad
        )

    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=weight.shape[2])

    gems_assert_close(
        res_weight_grad, ref_weight_grad, dtype, reduce_dim=weight.shape[0]
    )
    if bias is not None:
        gems_assert_close(res_bias_grad, ref_bias_grad, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]
    if flag_gems.vendor_name == "hygon":
        del os.environ["TRITON_HIP_USE_NEW_STREAM_PIPELINE"]


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RESULT TODOFIX")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.conv2d_padding
@pytest.mark.parametrize("shape, kernel,groups", SHAPE_CONV2D)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv2d_padding(
    shape, kernel, stride, padding, groups, dtype, dilation, bias
):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    if bias is True:
        bias = torch.randn(
            [weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=True
        )
        bias_ref = to_reference(bias, True)
    else:
        bias = None
        bias_ref = None

    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv2d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv2d(
        inp,
        weight,
        bias=bias,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(ref_out).to(flag_gems.device)

    ref_grad = to_reference(out_grad, True)
    if bias is not None:
        ref_in_grad, ref_weight_grad, ref_bias_grad = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, bias_ref), ref_grad
        )
        res_in_grad, res_weight_grad, res_bias_grad = torch.autograd.grad(
            res_out, (inp, weight, bias), out_grad
        )
    else:
        ref_in_grad, ref_weight_grad = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight), ref_grad
        )
        res_in_grad, res_weight_grad = torch.autograd.grad(
            res_out, (inp, weight), out_grad
        )

    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=weight.shape[2])

    gems_assert_close(
        res_weight_grad, ref_weight_grad, dtype, reduce_dim=weight.shape[0]
    )
    if bias is not None:
        gems_assert_close(res_bias_grad, ref_bias_grad, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


SHAPE_CONV3D = [
    ((1, 2, 5, 5, 5), (1, 2, 3, 3, 3), 1),
    ((2, 3, 9, 9, 9), (1, 3, 3, 3, 3), 1),
    # ((2, 2, 3, 3, 3), (1, 2, 2, 2, 2), 1),
    # ((32, 8, 8, 8, 8), (32, 8, 2, 2, 2), 1),
    # ((18, 16, 4, 4, 4), (16, 16, 2, 2, 2), 1),
    # ((9, 16, 4, 4, 4), (128, 4, 2, 2, 2), 4),
    # ((32, 16, 8, 8, 8), (32, 4, 4, 4, 4), 4),
    # ((18, 16, 4, 4, 4), (16, 8, 2, 2, 2), 2),
    # ((9, 16, 4, 4, 4), (128, 8, 2, 2, 2), 2),
    # ((32, 8, 8, 8, 8), (32, 8, 3, 3, 3), 1),
    # ((18, 16, 5, 5, 5), (16, 16, 3, 3, 3), 1),
    # ((9, 16, 7, 7, 7), (128, 4, 3, 3, 3), 4),
    # ((32, 16, 9, 9, 9), (32, 4, 5, 5, 5), 4),
    # ((18, 16, 11, 11, 11), (16, 8, 3, 3, 3), 2),
    # ((9, 16, 6, 6, 6), (128, 8, 3, 3, 3), 2),
]


# @pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.conv3d
@pytest.mark.parametrize("shape, kernel,groups", SHAPE_CONV3D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv3d(shape, kernel, stride, padding, groups, dtype, dilation, bias):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    if bias is True:
        bias = torch.randn(
            [weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=False
        )
        bias_ref = to_reference(bias, True)
    else:
        bias = None
        bias_ref = None

    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv3d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv3d(
        inp,
        weight,
        bias=bias,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.conv3d_padding
@pytest.mark.parametrize("shape, kernel,groups", SHAPE_CONV3D)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv3d_padding(
    shape, kernel, stride, padding, groups, dtype, dilation, bias
):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    if bias is True:
        bias = torch.randn(
            [weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=False
        )
        bias_ref = to_reference(bias, True)
    else:
        bias = None
        bias_ref = None

    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv3d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv3d(
        inp,
        weight,
        bias=bias,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


SHAPE_DEPTHWISE = [
    ((32, 4, 8, 8), (32, 1, 2, 2), (2, 2)),
    ((18, 16, 4, 4), (16, 1, 2, 2), (2, 2)),
    # ((9, 32, 4, 4), (128, 1, 2, 2), (2, 2)),
    # ((32, 16, 8, 8), (32, 1, 4, 4), (4, 4)),
    # ((18, 8, 4, 4), (16, 1, 2, 2), (2, 2)),
    # ((9, 4, 4, 4), (128, 1, 2, 2), (2, 2)),
    # ((32, 4, 8, 8), (32, 1, 3, 3), (3, 3)),
    # ((18, 16, 13, 13), (16, 1, 5, 5), (5, 5)),
    # ((9, 32, 8, 8), (128, 1, 3, 3), (3, 3)),
    # ((32, 16, 9, 9), (32, 1, 5, 5), (5, 5)),
    # ((18, 8, 7, 7), (16, 1, 3, 3), (3, 3)),
    # ((9, 4, 6, 6), (128, 1, 3, 3), (3, 3)),
]


# test for depthwise conv2d
@pytest.mark.conv_depthwise2d
@pytest.mark.parametrize("shape_input, shape_weight, kernel", SHAPE_DEPTHWISE)
@pytest.mark.parametrize("stride", [[1, 1], [2, 2]])
@pytest.mark.parametrize("padding", [[0, 0], [1, 1]])
@pytest.mark.parametrize("dilation", [[1, 1]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy__conv_depthwise2d(
    shape_input, shape_weight, kernel, stride, padding, dilation, dtype, bias
):
    inp = torch.randn(shape_input, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, False)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(shape_weight, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, False)

    if bias:
        bias_tensor = torch.randn(shape_weight[0], dtype=dtype, device=flag_gems.device)
        ref_bias = to_reference(bias_tensor, False)
    else:
        bias_tensor = None
        ref_bias = None

    ref_out = torch.ops.aten._conv_depthwise2d(
        ref_inp,
        ref_weight,
        kernel,
        ref_bias,
        stride,
        padding,
        dilation,
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten._conv_depthwise2d(
            inp, weight, kernel, bias_tensor, stride, padding, dilation
        )
    gems_assert_close(res_out, ref_out, dtype)


# conv_transpose1d test shapes: (input_shape, weight_shape)
# input: (N, in_channels, L_in)
# weight: (in_channels, out_channels/groups, kernel_width)
SHAPE_CONV_TRANSPOSE1D = [
    ((2, 4, 8), (4, 8, 3)),
    ((4, 8, 16), (8, 16, 3)),
    ((2, 16, 32), (16, 32, 5)),
]


@pytest.mark.conv_transpose1d
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV_TRANSPOSE1D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_conv_transpose1d(shape, kernel, stride, padding, dtype, monkeypatch):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, False)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, False)
    ref_out = torch.nn.functional.conv_transpose1d(
        ref_inp, ref_weight, bias=None, stride=stride, padding=padding, dilation=1
    )

    res_out = flag_gems.conv_transpose1d(
        inp, weight, bias=None, stride=stride, padding=padding, dilation=1
    )
    # Use reduce_dim to account for accumulation precision (in_channels * kernel_width)
    in_channels = kernel[0]
    kernel_width = kernel[2]
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=kernel_width * in_channels)


@pytest.mark.conv_transpose1d_bias
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV_TRANSPOSE1D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_conv_transpose1d_bias(
    shape, kernel, stride, padding, dtype, monkeypatch
):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, False)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, False)
    # weight shape is (in_channels, out_channels/groups, kernel_width)
    out_channels = kernel[1]
    kernel_width = kernel[2]
    bias = torch.randn(out_channels, dtype=dtype, device=flag_gems.device)
    ref_bias = to_reference(bias, False)

    ref_out = torch.nn.functional.conv_transpose1d(
        ref_inp, ref_weight, bias=ref_bias, stride=stride, padding=padding, dilation=1
    )

    res_out = flag_gems.conv_transpose1d(
        inp, weight, bias=bias, stride=stride, padding=padding, dilation=1
    )
    # Use reduce_dim to account for accumulation precision (in_channels * kernel_width)
    in_channels = kernel[0]
    out_channels = kernel[1]
    gems_assert_close(
        res_out,
        ref_out,
        dtype,
        reduce_dim=kernel_width * max(in_channels, out_channels),
    )


@pytest.mark.conv_transpose1d_groups
@pytest.mark.parametrize(
    "shape, kernel, groups",
    [
        ((2, 8, 16), (8, 4, 3), 2),  # 8 in_channels, 2 groups, 4 out_channels per group
        ((4, 12, 32), (12, 4, 3), 3),  # 12 in_channels, 3 groups
    ],
)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_conv_transpose1d_groups(
    shape, kernel, groups, stride, padding, dtype, monkeypatch
):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, False)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, False)

    ref_out = torch.nn.functional.conv_transpose1d(
        ref_inp,
        ref_weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=1,
        groups=groups,
    )

    res_out = flag_gems.conv_transpose1d(
        inp,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=1,
        groups=groups,
    )
    # Use reduce_dim to account for accumulation precision (in_channels_per_group * kernel_width)
    in_channels_per_group = kernel[0] // groups
    kernel_width = kernel[2]
    gems_assert_close(
        res_out, ref_out, dtype, reduce_dim=kernel_width * in_channels_per_group
    )
