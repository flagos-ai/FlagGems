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

from . import base

# Range is a non-pointwise op -- we override set_shapes instead of using
# GenericBenchmark which assumes pointwise tensor inputs.
RANGE_SIZES = [4096, 16777216, 1073741824]


class RangeBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = RANGE_SIZES

    def get_input_iter(self, cur_dtype):
        for end in self.shapes:
            yield {"start": 0, "end": end, "dtype": cur_dtype},


@pytest.mark.range
def test_range():
    # torch.range does not support bfloat16 on CUDA
    dtypes = [torch.float16, torch.float32]
    bench = RangeBenchmark(
        op_name="range",
        torch_op=torch.range,
        dtypes=dtypes,
    )
    bench.run()
