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


class MatmulBiasActivationBenchmark(base.BlasBenchmark):
    """
    benchmark for matmul_bias_activation
    """

    def set_more_shapes(self):
        return None


def matmul_bias_activation_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    # Note: b is ignored as we use (m, k) x (k, n) + bias
    input_tensor = torch.randn([m, k], dtype=cur_dtype, device=device)
    weight = torch.randn([k, n], dtype=cur_dtype, device=device)
    bias = torch.randn([n], dtype=cur_dtype, device=device)
    yield input_tensor, weight, bias


@pytest.mark.matmul_bias_activation
def test_matmul_bias_activation():
    def mma_torch_op(input, weight, bias):
        return torch.relu(torch.mm(input, weight) + bias)

    bench = MatmulBiasActivationBenchmark(
        input_fn=matmul_bias_activation_input_fn,
        op_name="matmul_bias_activation",
        torch_op=mma_torch_op,
        gems_op=flag_gems.matmul_bias_activation,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
