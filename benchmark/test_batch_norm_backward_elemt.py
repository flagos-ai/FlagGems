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


class NormBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            (16, 16, 64),
            (16, 16, 1024),
            (16, 16, 4098),
            (1, 8, 4, 4),
            (16, 8, 128, 128),
        ]


def batch_norm_backward_elemt_input_fn(shape, dtype, device):
    C = shape[1]
    spatial = 1
    for s in shape[2:]:
        spatial *= s
    N = shape[0]
    count_val = N * spatial

    grad_output = torch.randn(shape, dtype=dtype, device=device)
    inp = torch.randn(shape, dtype=dtype, device=device)
    # BN statistics are always float32 in PyTorch's native impl
    mean = torch.randn(C, dtype=torch.float32, device=device)
    invstd = torch.randn(C, dtype=torch.float32, device=device).abs() + 0.1
    weight = torch.randn(C, dtype=torch.float32, device=device)
    sum_dy = torch.randn(C, dtype=torch.float32, device=device)
    sum_dy_xmu = torch.randn(C, dtype=torch.float32, device=device)
    count = torch.tensor([count_val], dtype=torch.int32, device=device)

    yield (
        grad_output,
        inp,
        mean,
        invstd,
        weight,
        sum_dy,
        sum_dy_xmu,
        count,
    )


@pytest.mark.batch_norm_backward_elemt
def test_batch_norm_backward_elemt():
    bench = NormBenchmark(
        input_fn=batch_norm_backward_elemt_input_fn,
        op_name="batch_norm_backward_elemt",
        torch_op=torch.batch_norm_backward_elemt,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.batch_norm_backward_elemt)

    bench.run()
