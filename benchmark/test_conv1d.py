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

from . import base


class Conv1DBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            (32, 64, 512, 64, 3, 1, 0, 1),
            (64, 48, 1024, 128, 5, 2, 2, 1),
            (16, 24, 2048, 96, 7, 1, 3, 2),
            (8, 8, 8192, 16, 11, 4, 5, 1),
            (4, 4, 16384, 4, 15, 2, 7, 1),
            (32, 64, 512, 64, 3, 1, "valid", 1),
            (64, 48, 1024, 128, 5, 2, "valid", 1),
            (16, 24, 2048, 96, 7, 1, "same", 2),
            (8, 8, 8192, 16, 11, 1, "same", 1),
        ]


def conv1d_input_fn(shape, dtype, device):
    (
        batch,
        input_c,
        input_l,
        out_c,
        kernel,
        stride,
        padding,
        groups,
    ) = shape
    input_shape = (batch, input_c, input_l)
    weight_shape = (out_c, input_c // groups, kernel)
    input = torch.randn(size=input_shape, device=device, dtype=dtype)
    weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

    yield {
        "input": input,
        "weight": weight,
        "bias": None,
        "groups": groups,
        "stride": stride,
        "padding": padding,
    },


@pytest.mark.conv1d
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_conv1d():
    torch.backends.cudnn.allow_tf32 = False
    bench = Conv1DBenchmark(
        input_fn=conv1d_input_fn,
        op_name="conv1d",
        torch_op=torch.nn.functional.conv1d,
        dtypes=[
            torch.float16,
            torch.float32,
        ],  # Exclude bfloat16 due to cuDNN limitations
    )
    bench.set_gems(flag_gems.conv1d)
    bench.run()


class Conv1DPaddingBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype):
        shapes = [
            (32, 64, 512, 64, 3, 1, 0, 1),
            (32, 64, 512, 64, 3, 1, 1, 1),
            (32, 64, 512, 64, 3, 1, 2, 1),
            (64, 48, 1024, 128, 5, 2, 0, 1),
            (64, 48, 1024, 128, 5, 2, 2, 1),
            (64, 48, 1024, 128, 5, 2, 4, 1),
            (16, 24, 2048, 96, 7, 1, 0, 2),
            (16, 24, 2048, 96, 7, 1, 3, 2),
            (16, 24, 2048, 96, 7, 1, 6, 2),
            (32, 64, 512, 64, 3, 1, "valid", 1),
            (64, 48, 1024, 128, 5, 2, "valid", 1),
            (16, 24, 2048, 96, 7, 1, "same", 2),
            (8, 8, 8192, 16, 11, 1, "same", 1),
        ]
        for shape in shapes:
            yield from self.input_fn(shape, dtype, self.device)


@pytest.mark.conv1d_padding
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_conv1d_padding():
    torch.backends.cudnn.allow_tf32 = False
    bench = Conv1DPaddingBenchmark(
        input_fn=conv1d_input_fn,
        op_name="conv1d_padding",
        torch_op=torch.nn.functional.conv1d,
        dtypes=[
            torch.float16,
            torch.float32,
        ],  # Exclude bfloat16 due to cuDNN limitations
    )
    bench.set_gems(flag_gems.conv1d)
    bench.run()
