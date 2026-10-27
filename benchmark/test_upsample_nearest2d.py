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


class UpsampleBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        return []


def _input_fn(shape, dtype, device):
    batch, channel, height, weight = shape
    input = torch.randn(size=shape, device=device, dtype=dtype)
    scale_factors = (2, 2)
    output_size = (
        int(height * scale_factors[0]),
        int(weight * scale_factors[1]),
    )
    yield {
        "input": input,
        "output_size": output_size,
        "scales_h": None,
        "scales_w": None,
    },


@pytest.mark.upsample_nearest2d
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_upsample_nearest2d():
    bench = UpsampleBenchmark(
        op_name="upsample_nearest2d",
        input_fn=_input_fn,
        torch_op=torch._C._nn.upsample_nearest2d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
