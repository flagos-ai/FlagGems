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
            # 3D shapes represented as [batch_size, channels, hidden_size]
            (16, 16, 64),
            (16, 16, 1024),
            (16, 16, 4098),
            # 4D shapes represented as [batch_size, channels, H, W]
            (1, 8, 4, 4),
            (16, 8, 128, 128),
        ]


def batch_norm_no_update_backward_input_fn(shape, dtype, device):
    C = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((C,), dtype=dtype, device=device)
    running_mean = torch.randn((C,), dtype=dtype, device=device)
    running_var = torch.abs(torch.randn((C,), dtype=dtype, device=device)) + 0.1
    grad_out = torch.randn(shape, dtype=dtype, device=device)
    eps = 1e-5
    yield grad_out, inp, weight, running_mean, running_var, eps


def torch_batch_norm_no_update_backward(
    grad_out, inp, weight, running_mean, running_var, eps
):
    return torch.ops.aten.native_batch_norm_backward.default(
        grad_out,
        inp,
        weight,
        running_mean,
        running_var,
        None,
        None,
        False,
        eps,
        [True, True, True],
    )[0]


@pytest.mark.batch_norm_no_update_backward
def test_batch_norm_no_update_backward():
    bench = NormBenchmark(
        input_fn=batch_norm_no_update_backward_input_fn,
        op_name="batch_norm_no_update_backward",
        torch_op=torch_batch_norm_no_update_backward,
        gems_op=flag_gems.batch_norm_no_update_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
