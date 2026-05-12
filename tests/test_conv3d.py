import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

vendor_name = flag_gems.vendor_name
SHAPE_CONV3D = [
    ((1, 2, 5, 5, 5), (1, 2, 3, 3, 3), 1),
    ((2, 3, 9, 9, 9), (1, 3, 3, 3, 3), 1),
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


@pytest.mark.conv3d
@pytest.mark.parametrize("shape, kernel,groups", SHAPE_CONV3D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_conv3d(
    monkeypatch, shape, kernel, stride, padding, groups, dtype, dilation, bias
):
    if vendor_name == "mthreads" and dtype == torch.float16:
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = utils.to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    if bias is True:
        bias = torch.randn(
            [weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=False
        )
        bias_ref = utils.to_reference(bias, True)
    else:
        bias = None
        bias_ref = None

    ref_weight = utils.to_reference(weight, True)
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

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv3d_padding
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #3023")
@pytest.mark.parametrize("shape, kernel,groups", SHAPE_CONV3D)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_conv3d_padding(
    monkeypatch, shape, kernel, stride, padding, groups, dtype, dilation, bias
):
    if vendor_name == "mthreads" and dtype == torch.float16:
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = utils.to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    if bias is True:
        bias = torch.randn(
            [weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=False
        )
        bias_ref = utils.to_reference(bias, True)
    else:
        bias = None
        bias_ref = None

    ref_weight = utils.to_reference(weight, True)
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

    utils.gems_assert_close(res_out, ref_out, dtype)
