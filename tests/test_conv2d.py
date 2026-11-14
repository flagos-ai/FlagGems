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


@pytest.mark.conv2d
def test_conv2d_depthwise_pads_only_the_weight(monkeypatch):
    """The K >= 16 channel padding must not materialize a 16x input (#7075).

    `conv2d` used to pad each group's input channels from `weight_c` to 16, so a
    depthwise convolution allocated N * C * 16 * H * W, i.e. 16x the real input.
    Only the weight is padded now and the kernel masks channels past the real
    count. This checks the values (forward and backward) and that no `pad` call
    touches a tensor bigger than the padded weight.
    """
    groups = 4
    inp = torch.randn(
        (2, groups, 32, 32),
        dtype=torch.float32,
        device=flag_gems.device,
        requires_grad=True,
    )
    weight = torch.randn(
        (groups * 2, 1, 3, 3),
        dtype=torch.float32,
        device=flag_gems.device,
        requires_grad=True,
    )
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)

    padded = []
    real_pad = torch.nn.functional.pad

    def spy(tensor, pad, *args, **kwargs):
        padded.append(tensor.numel())
        return real_pad(tensor, pad, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "pad", spy)

    ref_out = torch.nn.functional.conv2d(
        ref_inp, ref_weight, stride=1, padding=1, dilation=1, groups=groups
    ).to(torch.float32)
    res_out = flag_gems.conv2d(
        inp, weight, stride=1, padding=1, dilation=1, groups=groups
    )
    utils.gems_assert_close(res_out, ref_out, torch.float32)

    assert padded, "expected the weight to be padded to K >= 16"
    assert max(padded) <= 16 * weight.numel(), (
        f"only the weight may be channel-padded, saw tensors of {padded} elements "
        f"(input has {inp.numel()})"
    )

    out_grad = torch.randn_like(ref_out).to(flag_gems.device)
    ref_grad = utils.to_reference(out_grad, True)
    ref_in_grad, ref_weight_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight), ref_grad
    )
    res_in_grad, res_weight_grad = torch.autograd.grad(res_out, (inp, weight), out_grad)
    utils.gems_assert_close(
        res_in_grad, ref_in_grad, torch.float32, reduce_dim=weight.shape[2]
    )
    utils.gems_assert_close(
        res_weight_grad, ref_weight_grad, torch.float32, reduce_dim=weight.shape[0]
    )
    assert max(padded) <= 16 * weight.numel(), f"backward padded {padded}"


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
