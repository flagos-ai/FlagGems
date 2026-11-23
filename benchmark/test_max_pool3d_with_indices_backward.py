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

from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils


def max_pool3d_backward_input_fn(shape, dtype, device):
    """Generate inputs for max_pool3d_with_indices_backward benchmark.

    Performs a forward pass to obtain indices, then yields the backward inputs.
    """
    inp = utils.generate_tensor_input(shape, dtype, device)

    # Default config: kernel=3, stride=2, padding=1
    params = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": False,
    }
    output, indices = torch.nn.functional.max_pool3d(inp, return_indices=True, **params)
    grad_output = torch.randn_like(output)
    yield grad_output, inp, params["kernel_size"], params["stride"], params[
        "padding"
    ], params["dilation"], params["ceil_mode"], indices

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # Non-cubic kernel/stride/padding
        if shape[-3] > 5 and shape[-2] > 5 and shape[-1] > 5:
            params2 = {
                "kernel_size": (2, 3, 3),
                "stride": (1, 2, 2),
                "padding": (0, 1, 1),
                "dilation": 1,
                "ceil_mode": False,
            }
            output2, indices2 = torch.nn.functional.max_pool3d(
                inp, return_indices=True, **params2
            )
            grad_output2 = torch.randn_like(output2)
            yield grad_output2, inp, params2["kernel_size"], params2["stride"], params2[
                "padding"
            ], params2["dilation"], params2["ceil_mode"], indices2


class MaxPool3dBackwardBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        # Typical 3D CNN shapes (N, C, D, H, W)
        shapes_5d = [
            (4, 3, 16, 56, 56),  # Input video frame
            (8, 64, 8, 28, 28),  # Early 3D-ResNet layer
            (16, 128, 4, 14, 14),  # Mid 3D-ResNet layer
            (32, 256, 2, 7, 7),  # Late 3D-ResNet layer
        ]

        for shape in shapes_5d:
            yield from self.input_fn(shape, dtype, self.device)


@pytest.mark.max_pool3d_with_indices_backward
def test_max_pool3d_with_indices_backward():
    bench = MaxPool3dBackwardBenchmark(
        input_fn=max_pool3d_backward_input_fn,
        op_name="max_pool3d_with_indices_backward",
        torch_op=torch.ops.aten.max_pool3d_with_indices_backward,
        gems_op=flag_gems.max_pool3d_with_indices_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
