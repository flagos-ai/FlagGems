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


class ConvDepthwise2DBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # Additional shapes for COMPREHENSIVE mode
        return [
            (1, 64, 224, 224, 3, 3, 1, 1, 1),
            (1, 128, 112, 112, 5, 5, 2, 2, 1),
        ]


def _input_fn(shape, dtype, device):
    (
        batch,
        channels,
        input_h,
        input_w,
        kernel_h,
        kernel_w,
        stride,
        padding,
        dilation,
    ) = shape
    input_shape = (batch, channels, input_h, input_w)
    weight_shape = (channels, 1, kernel_h, kernel_w)
    input_tensor = torch.randn(size=input_shape, device=device, dtype=dtype)
    weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

    # Pass as positional args since the first arg is named 'self' in aten op
    yield (
        input_tensor,
        weight,
        [kernel_h, kernel_w],
        None,  # bias
        [stride, stride],
        [padding, padding],
        [dilation, dilation],
    )


@pytest.mark.conv_depthwise2d
def test_conv_depthwise2d():
    torch.backends.cudnn.allow_tf32 = False

    bench = ConvDepthwise2DBenchmark(
        op_name="conv_depthwise2d",
        input_fn=_input_fn,
        torch_op=torch.ops.aten._conv_depthwise2d,
        gems_op=flag_gems._conv_depthwise2d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
