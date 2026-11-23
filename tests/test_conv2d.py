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
    # Issue #2345: fp64 inputs previously failed to compile because the kernel
    # accumulator was hardcoded to fp32. Keep fp64 in the dtype list so this
    # regression stays covered (only when the device supports fp64).
    FLOAT_DTYPES = FLOAT_DTYPES + ([torch.float64] if utils.fp64_is_supported else [])
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


_SPACEMIT_ONLY = pytest.mark.skipif(
    vendor_name != "spacemit",
    reason="two-kernel SpacemiT conv2d (im2col + blocked bmm) only",
)


@pytest.mark.conv2d
@_SPACEMIT_ONLY
@pytest.mark.parametrize(
    "shape,kernel,stride,padding,bias",
    [
        ((1, 3, 8, 8), (4, 3, 3, 3), 1, 1, True),
        ((1, 3, 9, 7), (4, 3, 3, 3), 2, 1, True),
        ((1, 3, 9, 7), (4, 3, 3, 3), 2, 0, False),
    ],
)
def test_spacemit_conv2d_positional_stride_padding(
    shape, kernel, stride, padding, bias
):
    """ATen order is (input, weight, bias, stride, padding, ...).

    ``stride`` and ``padding`` are deliberately unequal so a swapped
    signature changes the output shape as well as the values.
    """
    assert flag_gems.conv2d.__module__.endswith("ops.conv2d")
    gen = torch.Generator(device="cpu").manual_seed(0)
    inp = torch.randn(shape, generator=gen, dtype=torch.float32)
    weight = torch.randn(kernel, generator=gen, dtype=torch.float32)
    bias_t = (
        torch.randn((kernel[0],), generator=gen, dtype=torch.float32)
        if bias
        else None
    )
    ref = torch.nn.functional.conv2d(
        inp.double(),
        weight.double(),
        None if bias_t is None else bias_t.double(),
        stride=stride,
        padding=padding,
    ).float()

    res = flag_gems.conv2d(inp, weight, bias_t, stride, padding)

    assert res.shape == ref.shape
    utils.gems_assert_close(res, ref, torch.float32)
    assert not torch.isnan(res).any()


@pytest.mark.conv2d
@_SPACEMIT_ONLY
def test_spacemit_conv2d_unsupported_scope_is_explicit():
    inp = torch.randn((1, 3, 8, 8), dtype=torch.float32)
    weight = torch.randn((2, 3, 3, 3), dtype=torch.float32)

    with pytest.raises(NotImplementedError):
        flag_gems.conv2d(inp, weight, groups=2)
    with pytest.raises(NotImplementedError):
        flag_gems.conv2d(inp, weight, padding="same")
    with pytest.raises(NotImplementedError):
        flag_gems.conv2d(inp.half(), weight.half())


@pytest.mark.conv2d
@_SPACEMIT_ONLY
def test_spacemit_conv2d_autograd_delegates_to_generic():
    """Forward-only vendor kernels must keep the training contract."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    inp = torch.randn((1, 3, 6, 6), generator=gen, dtype=torch.float32)
    weight = torch.randn((4, 3, 3, 3), generator=gen, dtype=torch.float32)
    bias = torch.randn((4,), generator=gen, dtype=torch.float32)
    inp.requires_grad_(True)
    weight.requires_grad_(True)
    bias.requires_grad_(True)

    res = flag_gems.conv2d(inp, weight, bias, 2, 1)
    ref = torch.nn.functional.conv2d(inp, weight, bias, 2, 1)
    assert res.requires_grad and res.grad_fn is not None
    utils.gems_assert_close(res, ref, torch.float32)

    grad = torch.randn_like(ref)
    res_grads = torch.autograd.grad(res, (inp, weight, bias), grad)
    ref_grads = torch.autograd.grad(ref, (inp, weight, bias), grad)
    for got, expected in zip(res_grads, ref_grads):
        torch.testing.assert_close(got, expected, atol=1e-3, rtol=1e-3)
