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

# Shapes covering 1D to 4D tensors with various dimension sizes
SPLIT_WITH_SIZES_SHAPES = [
    (10,),
    (10, 4),
    (10, 4, 8),
    (10, 4, 8, 16),
    (16, 32),
    (8, 64, 128),
    (1, 8192),
    (32, 50257),
]


class SplitWithSizesBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPLIT_WITH_SIZES_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            # Benchmark along every splittable dimension for comprehensive
            # coverage, not only dim=0.
            for dim in range(len(shape)):
                dim_size = shape[dim]
                split_sizes = [
                    dim_size // 4,
                    dim_size // 4,
                    dim_size - 2 * (dim_size // 4),
                ]
                yield inp, split_sizes, dim


@pytest.mark.split_with_sizes
def test_split_with_sizes():
    bench = SplitWithSizesBenchmark(
        op_name="split_with_sizes",
        torch_op=torch.split_with_sizes,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
