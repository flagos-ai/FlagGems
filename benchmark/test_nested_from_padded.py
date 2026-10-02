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

import random

import pytest
import torch

from . import base, consts

# (batch_size, max_length, inner_dim)
NESTED_FROM_PADDED_SHAPES = [
    (8, 32, 64),
    (32, 64, 64),
    (64, 128, 64),
    (128, 128, 128),
    (256, 256, 128),
    (512, 256, 128),
]


class NestedFromPaddedBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = NESTED_FROM_PADDED_SHAPES

    def get_input_iter(self, cur_dtype):
        rng = random.Random(42)
        for batch_size, max_length, inner_dim in self.shapes:
            lengths = [rng.randint(1, max_length) for _ in range(batch_size)]
            padded = torch.randn(
                batch_size, max_length, inner_dim, dtype=cur_dtype, device=self.device
            )
            sizes = torch.tensor([[ln, inner_dim] for ln in lengths], dtype=torch.int64)
            yield padded, sizes


@pytest.mark.nested_from_padded
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_nested_from_padded(dtype):
    bench = NestedFromPaddedBenchmark(
        op_name="nested_from_padded",
        torch_op=torch._nested_from_padded,
        dtypes=[dtype],
    )
    bench.run()
