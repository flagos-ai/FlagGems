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

GN_BWD_SHAPES = [
    (16, 3, 16, 16),
    (1, 64, 32, 32),
    (2, 64, 128, 128),
]


class GroupNormBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # (N, C, H, W) with channels divisible by the group counts in the builder.
        self.shapes = GN_BWD_SHAPES
        self.shape_desc = "N, C, H, W"


def group_norm_backward_input_fn(shape, dtype, device):
    N, C = shape[0], shape[1]
    HxW = shape[2] * shape[3]
    num_groups = C
    grad_out = torch.randn(shape, dtype=dtype, device=device)
    inp = torch.randn(shape, dtype=dtype, device=device)
    mean = torch.randn([N, num_groups], dtype=dtype, device=device)
    rstd = torch.randn([N, num_groups], dtype=dtype, device=device).abs() + 0.1
    weight = torch.randn(C, dtype=dtype, device=device)
    output_mask = [True, True, True]
    yield (
        grad_out,
        inp,
        mean,
        rstd,
        weight,
        {
            "N": N,
            "C": C,
            "HxW": HxW,
            "group": num_groups,
            "output_mask": output_mask,
        },
    )


@pytest.mark.group_norm_backward
def test_group_norm_backward():
    bench = GroupNormBackwardBenchmark(
        op_name="group_norm_backward",
        torch_op=torch.ops.aten.native_group_norm_backward,
        gems_op=flag_gems.group_norm_backward,
        input_fn=group_norm_backward_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
