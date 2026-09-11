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
import torch.nn.functional as F

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


def batch_norm_no_update_input_fn(shape, dtype, device):
    C = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((C,), dtype=dtype, device=device)
    bias = torch.randn((C,), dtype=dtype, device=device)
    running_mean = torch.randn((C,), dtype=dtype, device=device)
    running_var = torch.abs(torch.randn((C,), dtype=dtype, device=device)) + 0.1
    momentum = 0.1
    eps = 1e-5
    yield inp, weight, bias, running_mean, running_var, momentum, eps


def torch_batch_norm_no_update(
    inp, weight, bias, running_mean, running_var, momentum, eps
):
    return F.batch_norm(
        inp, running_mean, running_var, weight, bias, training=False, eps=eps
    )


@pytest.mark.batch_norm_no_update
def test_batch_norm_no_update():
    bench = NormBenchmark(
        input_fn=batch_norm_no_update_input_fn,
        op_name="batch_norm_no_update",
        torch_op=torch_batch_norm_no_update,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
