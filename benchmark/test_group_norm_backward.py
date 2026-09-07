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

from . import base, consts


def group_norm_backward_input_fn(shape, dtype, device):
    """Yield args of torch.ops.aten.native_group_norm_backward.

    Layout: (grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask).
    """
    N, C, H, W = shape
    num_groups = C // 2
    grad_out = torch.randn(shape, dtype=dtype, device=device)
    input = torch.randn(shape, dtype=dtype, device=device)
    mean = torch.randn((N, num_groups), dtype=dtype, device=device)
    rstd = torch.randn((N, num_groups), dtype=dtype, device=device)
    weight = torch.randn((C,), dtype=dtype, device=device)
    yield grad_out, input, mean, rstd, weight, N, C, H * W, num_groups, [
        True,
        True,
        True,
    ]


class GroupNormBackwardBenchmark(base.GenericBenchmark):
    """Benchmark the group_norm_backward operator (native_group_norm_backward)."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (4, 16, 16, 16),
            (8, 8, 32, 32),
            (2, 32, 32, 32),
        ]
        self.shape_desc = "N, C, H, W"


@pytest.mark.group_norm_backward
def test_group_norm_backward():
    bench = GroupNormBackwardBenchmark(
        input_fn=group_norm_backward_input_fn,
        op_name="group_norm_backward",
        torch_op=torch.ops.aten.native_group_norm_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
