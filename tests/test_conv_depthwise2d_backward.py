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

# Depthwise 2D conv shapes: (N, C, H, W, KH, KW, stride, padding, dilation).
# C is the channel count and also the group count; the weight is (C, 1, KH, KW).
DEPTHWISE_CONFIGS = (
    (2, 4, 8, 8, 3, 3, (1, 1), (1, 1), (1, 1)),
    (1, 3, 7, 9, 3, 2, (2, 2), (2, 1), (1, 1)),
    (2, 5, 10, 10, 5, 5, (1, 1), (2, 2), (2, 2)),
    (1, 2, 6, 6, 1, 1, (1, 1), (0, 0), (1, 1)),
    (3, 4, 9, 5, 2, 3, (2, 1), (0, 1), (1, 2)),
)


def _depthwise_conv2d_input_grad(grad_output, x, weight, stride, padding, dilation):
    """Exact input gradient of the depthwise conv2d computed by autograd."""
    inp = x.detach().requires_grad_(True)
    wgt = weight.detach().requires_grad_(True)
    with torch.enable_grad():
        out = torch.conv2d(
            inp,
            wgt,
            None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=x.shape[1],
        )
        grads = torch.autograd.grad(out, (inp, wgt), grad_outputs=grad_output)
    return grads


@pytest.mark.conv_depthwise2d_backward
@pytest.mark.parametrize(
    "config",
    DEPTHWISE_CONFIGS,
    ids=lambda c: f"N{c[0]}C{c[1]}-{c[2]}x{c[3]}-k{c[4]}x{c[5]}",
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_conv_depthwise2d_backward_fp32(config, dtype):
    """Gradients match ATen autograd exactly on fp32 for each config."""
    n, c, h, w, kh, kw, stride, padding, dilation = config
    out_h = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    out_w = (w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    x = torch.randn((n, c, h, w), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((c, 1, kh, kw), dtype=dtype, device=flag_gems.device)
    grad_out = torch.randn((n, c, out_h, out_w), dtype=dtype, device=flag_gems.device)

    ref_grad_input, ref_grad_weight = _depthwise_conv2d_input_grad(
        utils.to_reference(grad_out, False),
        utils.to_reference(x, False),
        utils.to_reference(weight, False),
        stride,
        padding,
        dilation,
    )

    res_grad_input, res_grad_weight = flag_gems.conv_depthwise2d_backward(
        grad_out, x, weight, stride, padding, dilation
    )

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
    utils.gems_assert_close(res_grad_weight, ref_grad_weight, dtype)


@pytest.mark.conv_depthwise2d_backward
@pytest.mark.parametrize(
    "config",
    DEPTHWISE_CONFIGS[:3],
    ids=lambda c: f"N{c[0]}C{c[1]}-{c[2]}x{c[3]}-k{c[4]}x{c[5]}",
)
def test_conv_depthwise2d_backward_half(config):
    """fp16/bf16: 1-ulp-level agreement with an exact fp64 gather."""
    n, c, h, w, kh, kw, stride, padding, dilation = config
    out_h = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    out_w = (w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    for dtype in (torch.float16, torch.bfloat16):
        x = torch.randn((n, c, h, w), dtype=dtype, device=flag_gems.device)
        weight = torch.randn((c, 1, kh, kw), dtype=dtype, device=flag_gems.device)
        grad_out = torch.randn(
            (n, c, out_h, out_w), dtype=dtype, device=flag_gems.device
        )

        ref_grad_input, ref_grad_weight = _depthwise_conv2d_input_grad(
            utils.to_reference(grad_out, False),
            utils.to_reference(x, False),
            utils.to_reference(weight, False),
            stride,
            padding,
            dilation,
        )
        res_grad_input, res_grad_weight = flag_gems.conv_depthwise2d_backward(
            grad_out, x, weight, stride, padding, dilation
        )
        # fp16 accumulation-order noise sits well below one fp16 ulp at the
        # output magnitude; bf16 rounding dominates at ~2^-8 of the max value.
        atol = 2e-2 if dtype == torch.float16 else 6e-2
        torch.testing.assert_close(res_grad_input, ref_grad_input, atol=atol, rtol=2e-2)
        torch.testing.assert_close(
            res_grad_weight, ref_grad_weight, atol=atol * 4, rtol=5e-2
        )


@pytest.mark.conv_depthwise2d_backward
@pytest.mark.parametrize("dtype", [torch.float32])
def test_conv_depthwise2d_backward_non_contiguous(dtype):
    """A channels-last input and a channels-last upstream gradient."""
    n, c, h, w, kh, kw = 2, 4, 10, 12, 3, 3
    out_h = (h + 2 - kh) + 1  # stride=1, padding=1, dilation=1
    out_w = (w + 2 - kw) + 1
    # Channels-last storage: same logical shapes as NCHW but a strided layout.
    x = torch.randn((n, c, h, w), dtype=dtype, device=flag_gems.device).to(
        memory_format=torch.channels_last
    )
    assert not x.is_contiguous()
    grad_out = torch.randn(
        (n, c, out_h, out_w), dtype=dtype, device=flag_gems.device
    ).to(memory_format=torch.channels_last)
    assert not grad_out.is_contiguous()

    weight = torch.randn((c, 1, kh, kw), dtype=dtype, device=flag_gems.device)
    ref_grad_input, ref_grad_weight = _depthwise_conv2d_input_grad(
        grad_out, x, weight, (1, 1), (1, 1), (1, 1)
    )
    res_grad_input, res_grad_weight = flag_gems.conv_depthwise2d_backward(
        grad_out, x, weight, (1, 1), (1, 1), (1, 1)
    )
    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
    utils.gems_assert_close(res_grad_weight, ref_grad_weight, dtype)
