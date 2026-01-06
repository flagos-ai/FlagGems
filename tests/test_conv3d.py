import os

import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference

SHAPE_CONV1D = [
    ((32, 2, 4), (17, 2, 2)),
    ((32, 15, 6), (17, 15, 2)),
    ((32, 16, 1024), (1024, 16, 8)),
    ((64, 64, 64), (128, 64, 7)),
    ((32, 12, 9), (17, 12, 3)),
    ((32, 6, 6), (64, 6, 2)),
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


SHAPE_CONV2D = [
    ((1, 2, 5, 5), (1, 2, 3, 3), 1),
    # ((2, 3, 9, 9), (1, 3, 3, 3), 1),
    # ((2, 2, 3, 3), (1, 2, 2, 2), 1),
    # ((32, 8, 8, 8), (32, 8, 2, 2), 1),
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
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("bias", [True])
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
        (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, bias_ref), ref_grad
        )
        (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
            res_out, (inp, weight, bias), out_grad
        )
    else:
        (ref_in_grad, ref_weight_grad) = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight), ref_grad
        )
        (res_in_grad, res_weight_grad) = torch.autograd.grad(
            res_out, (inp, weight), out_grad
        )

    # gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=weight.shape[2])
    gems_assert_close(
        res_weight_grad, ref_weight_grad, dtype, reduce_dim=weight.shape[0]
    )
    if bias is not None:
        gems_assert_close(res_bias_grad, ref_bias_grad, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]
    if flag_gems.vendor_name == "hygon":
        del os.environ["TRITON_HIP_USE_NEW_STREAM_PIPELINE"]


SHAPE_CONV3D = [
    # ((1, 2, 5, 5, 5), (1, 2, 3, 3, 3), 1),
    # ((2, 3, 9, 9, 9), (1, 3, 3, 3, 3), 1),
    # ((2, 2, 3, 3, 3), (1, 2, 2, 2, 2), 1),
    # ((32, 8, 8, 8, 8), (32, 8, 2, 2, 2), 1),
    # ((18, 16, 4, 4, 4), (16, 16, 2, 2, 2), 1),
    ((9, 16, 4, 4, 4), (128, 4, 2, 2, 2), 4),
    # ((32, 16, 8, 8, 8), (32, 4, 4, 4, 4), 4),
    # ((18, 16, 4, 4, 4), (16, 8, 2, 2, 2), 2),
    # ((9, 16, 4, 4, 4), (128, 8, 2, 2, 2), 2),
    ((32, 8, 8, 8, 8), (32, 8, 3, 3, 3), 1),
    ((18, 16, 5, 5, 5), (16, 16, 3, 3, 3), 1),
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


SHAPE_DEPTHWISE = [
    ((32, 4, 8, 8), (32, 1, 2, 2), (2, 2)),
    ((18, 16, 4, 4), (16, 1, 2, 2), (2, 2)),
    ((9, 32, 4, 4), (128, 1, 2, 2), (2, 2)),
    ((32, 16, 8, 8), (32, 1, 4, 4), (4, 4)),
    ((18, 8, 4, 4), (16, 1, 2, 2), (2, 2)),
    ((9, 4, 4, 4), (128, 1, 2, 2), (2, 2)),
    ((32, 4, 8, 8), (32, 1, 3, 3), (3, 3)),
    ((18, 16, 13, 13), (16, 1, 5, 5), (5, 5)),
    ((9, 32, 8, 8), (128, 1, 3, 3), (3, 3)),
    ((32, 16, 9, 9), (32, 1, 5, 5), (5, 5)),
    ((18, 8, 7, 7), (16, 1, 3, 3), (3, 3)),
    ((9, 4, 6, 6), (128, 1, 3, 3), (3, 3)),
]


# test for depthwise depends on specific device
@pytest.mark.skip("conv_depthwise2d introduces failures, disable it temporarily")
@pytest.mark.conv_depthwise2d
@pytest.mark.parametrize("shape_input, shape_weight,kernel ", SHAPE_DEPTHWISE)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("padding", [2])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_depthwise2d(
    shape_input, shape_weight, kernel, stride, padding, dtype
):
    inp = torch.randn(
        shape_input, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_inp = to_reference(inp, False)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(shape_weight, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, False)
    ref_out = torch._C._nn._conv_depthwise2d(
        ref_inp,
        ref_weight,
        kernel,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=1,
    )

    res_out = flag_gems._conv_depthwise2d(
        inp, weight, kernel, bias=None, stride=stride, padding=padding, dilation=1
    )
    gems_assert_close(res_out, ref_out, dtype)
