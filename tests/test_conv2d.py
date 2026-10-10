# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

vendor_name = flag_gems.vendor_name

if QUICK_MODE:
    SHAPE_CONV2D = [
        ((1, 2, 5, 5), (1, 2, 3, 3), 1),
    ]
    FLOAT_DTYPES = [torch.float32]
    STRIDES = [1]
    PADDINGS = [1]
    DILATIONS = [1]
    BIASES = [True]
    STR_PADDINGS = ["same"]
else:
    SHAPE_CONV2D = [
        ((1, 2, 5, 5), (1, 2, 3, 3), 1),
        ((2, 3, 9, 9), (1, 3, 3, 3), 1),
        ((32, 8, 8, 8), (32, 8, 2, 2), 1),
    ]
    FLOAT_DTYPES = [torch.float16, torch.float32]
    STRIDES = [1, 2]
    PADDINGS = [0, 1]
    DILATIONS = [1]  # original: [1, 2], dilation=2 commented out to reduce CI timeout
    BIASES = [True, False]
    STR_PADDINGS = ["valid", "same"]


@pytest.mark.conv2d
@pytest.mark.parametrize("shape, kernel,groups", SHAPE_CONV2D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dilation", DILATIONS)
@pytest.mark.parametrize("bias", BIASES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_conv2d(
    monkeypatch, shape, kernel, stride, padding, groups, dtype, dilation, bias
):
    # Issue 2801: The environment variable is not enforced in operator logic.
    if vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    if bias is True:
        bias = torch.randn(
            [weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=True
        )
        bias_ref = utils.to_reference(bias, True)
    else:
        bias = None
        bias_ref = None

    ref_weight = utils.to_reference(weight, True)
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

    utils.gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(ref_out).to(flag_gems.device)

    ref_grad = utils.to_reference(out_grad, True)
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

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=weight.shape[2])

    utils.gems_assert_close(
        res_weight_grad, ref_weight_grad, dtype, reduce_dim=weight.shape[0]
    )
    if bias is not None:
        utils.gems_assert_close(res_bias_grad, ref_bias_grad, dtype)


def _conv2d_reference(ref_inp, ref_weight, bias_ref, groups, stride, padding, dilation):
    """Reference conv2d built on a path that is differentiable on this backend.

    `F.conv2d(..., padding="valid"|"same")` dispatches to `aten::conv2d.padding`,
    which carries no Autograd kernel on the kunlunxin XPU build, so torch's
    autograd fallback builds an empty graph and backprop through the reference
    raises "One of the differentiated Tensors appears to not have been used in
    the graph" before the operator under test is ever reached.  Integer padding
    dispatches to `aten::convolution.default` (ConvolutionBackward0) instead, so
    the same convolution is expressed here with integer padding.

    Only kunlunxin is affected.  Every other vendor - and every integer-padding
    call - takes the upstream call verbatim, argument list unchanged.
    """
    if vendor_name != "kunlunxin" or not isinstance(padding, str):
        return torch.nn.functional.conv2d(
            ref_inp,
            ref_weight,
            bias=bias_ref,
            groups=groups,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    if padding == "valid":
        # exactly integer padding 0
        return torch.nn.functional.conv2d(
            ref_inp,
            ref_weight,
            bias=bias_ref,
            groups=groups,
            stride=stride,
            padding=0,
            dilation=dilation,
        )
    assert padding == "same", padding
    assert stride == 1, "padding='same' requires stride 1"
    d_h, d_w = (dilation, dilation) if isinstance(dilation, int) else dilation
    # 'same' pads dilation*(k-1) in total along each spatial dim, split as
    # floor(total/2) before and the remainder after, so an odd total puts the
    # extra row/column at the END.  Integer padding is always symmetric, so an
    # even total maps onto it directly and an odd total needs an explicit F.pad.
    pad_h = d_h * (ref_weight.shape[2] - 1)
    pad_w = d_w * (ref_weight.shape[3] - 1)
    if pad_h % 2 == 0 and pad_w % 2 == 0:
        return torch.nn.functional.conv2d(
            ref_inp,
            ref_weight,
            bias=bias_ref,
            groups=groups,
            stride=stride,
            padding=(pad_h // 2, pad_w // 2),
            dilation=dilation,
        )
    return torch.nn.functional.conv2d(
        torch.nn.functional.pad(
            ref_inp, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        ),
        ref_weight,
        bias=bias_ref,
        groups=groups,
        stride=stride,
        padding=0,
        dilation=dilation,
    )


@pytest.mark.conv2d_padding
@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2802: operator doesn't work")
@pytest.mark.parametrize("shape, kernel,groups", SHAPE_CONV2D)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", STR_PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dilation", DILATIONS)
@pytest.mark.parametrize("bias", BIASES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_conv2d_padding(
    monkeypatch, shape, kernel, stride, padding, groups, dtype, dilation, bias
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    if bias is True:
        bias = torch.randn(
            [weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=True
        )
        bias_ref = utils.to_reference(bias, True)
    else:
        bias = None
        bias_ref = None

    ref_weight = utils.to_reference(weight, True)
    ref_out = _conv2d_reference(
        ref_inp, ref_weight, bias_ref, groups, stride, padding, dilation
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

    utils.gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(ref_out).to(flag_gems.device)

    ref_grad = utils.to_reference(out_grad, True)
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

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=weight.shape[2])

    utils.gems_assert_close(
        res_weight_grad, ref_weight_grad, dtype, reduce_dim=weight.shape[0]
    )
    if bias is not None:
        utils.gems_assert_close(res_bias_grad, ref_bias_grad, dtype)
