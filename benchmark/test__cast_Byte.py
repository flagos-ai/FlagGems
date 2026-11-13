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

from .base import Benchmark, BenchmarkCasePlan


class CastByteBenchmark(Benchmark):
    def get_input_iter(self, cur_dtype):
        for shape in [(1024,), (65536,), (1048576,), (16777216,)]:
            yield (torch.randn(shape, device=self.device, dtype=cur_dtype),)
        x = torch.randn((1024, 1024), device=self.device, dtype=cur_dtype)
        yield (x.t(),)
        yield (x[:, ::2],)

    def get_case_iter(self, dtype):
        specs = [((1024,), "contiguous"), ((65536,), "contiguous"),
                 ((1048576,), "contiguous"), ((16777216,), "contiguous"),
                 ((1024, 1024), "transpose"), ((1024, 1024), "slice")]
        for ordinal, (shape, layout) in enumerate(specs):
            result_shape = (1024, 512) if layout == "slice" else shape
            yield self._case_from_plan(dtype, ordinal, BenchmarkCasePlan(
                shape={"input": list(result_shape)}, params={"layout": layout},
                builder_args=(shape, layout)))

    def build_inputs(self, case):
        shape, layout = case.builder_args[0].builder_args
        x = torch.randn(shape, device=self.device, dtype=case.dtype)
        if layout == "transpose":
            x = x.t()
        elif layout == "slice":
            x = x[:, ::2]
        return (x,)



@pytest.mark.underscore_cast_Byte
def test_cast_Byte():
    bench = CastByteBenchmark(
        op_name="_cast_Byte",
        torch_op=torch.ops.aten._cast_Byte.default,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
    )
    bench.run()
