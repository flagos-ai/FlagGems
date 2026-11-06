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
from tests.test_grouped_gemm_w8a8 import deepgemm_grouped_gemm, make_grouped_gemm_case

from .base import Benchmark


class GroupedGemmInt8Benchmark(Benchmark):
    DEFAULT_SHAPES = [(45, 512, 6144, 257), (45, 6144, 256, 257)]
    DEFAULT_SHAPE_DESC = "M,N,K,E"

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            m, n, k, e = shape[:4]
            if len(shape) == 5:
                lengths = shape[4]
            else:
                lengths = [m // e + (expert < m % e) for expert in range(e)]
            if len(lengths) != e or sum(lengths) != m:
                raise ValueError("expert row lengths must match M and E")
            yield make_grouped_gemm_case(lengths, n, k, self.device)


@pytest.mark.grouped_gemm_w8a8
@pytest.mark.skipif(
    "grouped_gemm_w8a8" not in vars(flag_gems),
    reason="requires a registered PPU grouped INT8 GEMM implementation",
)
def test_grouped_gemm_w8a8():
    pytest.importorskip("deep_gemm")
    bench = GroupedGemmInt8Benchmark(
        op_name="grouped_gemm_w8a8",
        torch_op=deepgemm_grouped_gemm,
        gems_op=flag_gems.grouped_gemm_w8a8,
        dtypes=[torch.bfloat16],
    )
    bench.run()
