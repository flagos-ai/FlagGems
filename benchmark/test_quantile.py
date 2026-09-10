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

from . import base, utils


class QuantileBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        more_shapes_1d = [(4,), (1024,), (65535)]
        more_shapes_2d = [(1024, 2**i) for i in range(0, 15, 3)]
        more_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 3)]
        return more_shapes_1d + more_shapes_2d + more_shapes_3d


def quantile_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    q = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=cur_dtype, device=device)
    yield inp, q, 0


@pytest.mark.quantile
def test_quantile():
    bench = QuantileBenchmark(
        input_fn=quantile_input_fn,
        op_name="quantile",
        torch_op=torch.quantile,
        dtypes=[torch.float32],
    )
    bench.run()
