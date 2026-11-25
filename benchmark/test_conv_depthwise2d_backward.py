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

from . import base, consts


class ConvDepthwise2dBackwardBenchmark(base.GenericBenchmark):
    """Depthwise conv2d backward: both gradients through the entry point.

    Shapes are (N, C, H, W, KH, KW, stride, padding) tuples; dilation is kept
    at (1, 1) for the benchmark sweep.  The torch reference uses the native
    autograd of ``torch.conv2d(groups=C)``, which is how ATen evaluates the
    depthwise backward on this build.
    """

    def get_input_iter(self, dtype):
        shapes = [
            (8, 32, 64, 64, 3, 3, (1, 1), (1, 1)),
            (16, 64, 128, 128, 3, 3, (1, 1), (1, 1)),
            (4, 96, 130, 130, 5, 5, (2, 2), (2, 2)),
            (2, 128, 224, 224, 3, 3, (2, 2), (1, 1)),
        ]
        for n, c, h, w, kh, kw, stride, padding in shapes:
            out_h = (h + 2 * padding[0] - (kh - 1) - 1) // stride[0] + 1
            out_w = (w + 2 * padding[1] - (kw - 1) - 1) // stride[1] + 1
            x = torch.randn((n, c, h, w), dtype=dtype, device=self.device)
            weight = torch.randn((c, 1, kh, kw), dtype=dtype, device=self.device)
            grad_output = torch.randn(
                (n, c, out_h, out_w), dtype=dtype, device=self.device
            )
            yield grad_output, x, weight, stride, padding, (1, 1)


def _depthwise_conv2d_backward(grad_output, x, weight, stride, padding, dilation):
    class _DepthwiseConv2dBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, go, inp, wgt):
            ctx.save_for_backward(go, inp, wgt)
            ctx.stride = stride
            ctx.padding = padding
            ctx.dilation = dilation
            return torch.conv2d(
                inp,
                wgt,
                None,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=inp.shape[1],
            )

        @staticmethod
        def backward(ctx, grad_out):
            go, inp, wgt = ctx.saved_tensors
            return torch.ops.aten.convolution_backward(
                grad_out.contiguous(),
                inp,
                wgt,
                None,
                stride,
                padding,
                dilation,
                False,
                (0, 0),
                inp.shape[1],
                (True, True, False),
            )[:2]

    return _DepthwiseConv2dBackward.apply(grad_output, x, weight)


@pytest.mark.conv_depthwise2d_backward
def test_conv_depthwise2d_backward():
    bench = ConvDepthwise2dBackwardBenchmark(
        op_name="conv_depthwise2d_backward",
        torch_op=_depthwise_conv2d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conv_depthwise2d_backward)
    bench.run()
