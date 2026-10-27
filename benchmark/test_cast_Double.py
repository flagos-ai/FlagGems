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

from .base import Benchmark


class CastDoubleBenchmark(Benchmark):
    def get_input_iter(self, cur_dtype):
        for shape in [(1024,), (65536,), (1048576,), (16777216,)]:
            yield (torch.randn(shape, device=self.device, dtype=cur_dtype),)
        x = torch.randn((1024, 1024), device=self.device, dtype=cur_dtype)
        yield (x.t(),)
        yield (x[:, ::2],)


@pytest.mark.cast_Double
def test_cast_Double():
    bench = CastDoubleBenchmark(
        op_name="cast_Double",
        torch_op=torch.ops.aten._cast_Double.default,
        gems_op=flag_gems.cast_Double,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
    )
    bench.run()
