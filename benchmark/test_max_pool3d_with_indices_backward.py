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

from . import base, consts, utils


def _case_fn(shape, dtype):
    """Plan inputs for max_pool3d_with_indices_backward benchmark.

    The default cubic config applies to every shape; the non-cubic config is
    only emitted at COMPREHENSIVE level for large-enough spatial dims.
    """
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"config": "cubic"},
        builder_args=(shape, "cubic"),
    )

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        if shape[-3] > 5 and shape[-2] > 5 and shape[-1] > 5:
            yield base.BenchmarkCasePlan(
                shape={"input": list(shape)},
                params={"config": "noncubic"},
                builder_args=(shape, "noncubic"),
            )


def _build_inputs_fn(plan, dtype, device):
    """Perform a forward pass to obtain indices, then build the backward inputs."""
    shape, config = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)

    if config == "cubic":
        # Default config: kernel=3, stride=2, padding=1
        params = {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "dilation": 1,
            "ceil_mode": False,
        }
    else:
        # Non-cubic kernel/stride/padding
        params = {
            "kernel_size": (2, 3, 3),
            "stride": (1, 2, 2),
            "padding": (0, 1, 1),
            "dilation": 1,
            "ceil_mode": False,
        }
    output, indices = torch.nn.functional.max_pool3d(inp, return_indices=True, **params)
    grad_output = torch.randn_like(output)
    return (
        grad_output,
        inp,
        params["kernel_size"],
        params["stride"],
        params["padding"],
        params["dilation"],
        params["ceil_mode"],
        indices,
    )


class MaxPool3dBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Typical 3D CNN shapes (N, C, D, H, W)
        self.shapes = [
            (4, 3, 16, 56, 56),  # Input video frame
            (8, 64, 8, 28, 28),  # Early 3D-ResNet layer
            (16, 128, 4, 14, 14),  # Mid 3D-ResNet layer
            (32, 256, 2, 7, 7),  # Late 3D-ResNet layer
        ]


@pytest.mark.max_pool3d_with_indices_backward
def test_max_pool3d_with_indices_backward():
    bench = MaxPool3dBackwardBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        op_name="max_pool3d_with_indices_backward",
        torch_op=torch.ops.aten.max_pool3d_with_indices_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
