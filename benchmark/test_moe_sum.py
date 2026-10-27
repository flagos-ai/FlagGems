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


def _input_fn(shape, dtype, device):
    shape = (shape[0], 1, shape[1]) if len(shape) == 2 else shape
    num_tokens, topk, hidden_size = shape
    input_tensor = torch.randn(
        num_tokens,
        topk,
        hidden_size,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )

    output_tensor = torch.empty(
        num_tokens, hidden_size, dtype=dtype, device=device, requires_grad=False
    )
    yield input_tensor, output_tensor


@pytest.mark.moe_sum
def test_moe_sum():
    def torch_op(input_tensor, output_tensor):
        output_tensor.copy_(input_tensor.sum(dim=1))

    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_input_fn,
        op_name="moe_sum",
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.moe_sum)
    bench.run()
