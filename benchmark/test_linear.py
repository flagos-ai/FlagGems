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


class LinearBenchmark(base.BlasBenchmark):
    def get_tflops(self, op, *args, **kwargs):
        input_tensor = args[0]
        weight = args[1]
        batch = input_tensor.shape[0]
        out_features = weight.shape[0]
        in_features = weight.shape[1]
        return batch * out_features * (2 * in_features + 1)


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    input_tensor = torch.randn([m, k], dtype=dtype, device=device)
    weight = torch.randn([n, k], dtype=dtype, device=device)
    bias = torch.randn([n], dtype=dtype, device=device)
    yield input_tensor, weight, bias


@pytest.mark.linear
def test_linear():
    bench = LinearBenchmark(
        op_name="linear",
        input_fn=_input_fn,
        torch_op=torch.nn.functional.linear,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
