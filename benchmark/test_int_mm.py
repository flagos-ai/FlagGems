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

from . import base


class IntMMBenchmark(base.Benchmark):
    def __init__(self, op_name, torch_op, gems_op, use_out=False):
        super().__init__(op_name, torch_op, dtypes=[torch.int8])
        self.set_gems(gems_op)
        self.use_out = use_out

    def set_more_shapes(self):
        return [(512, 4096, 4096)]

    def get_input_iter(self, dtype):
        for M, N, K in self.shapes:
            mat1 = torch.randint(
                -128, 128, (M, K), dtype=dtype, device=flag_gems.device
            )
            mat2 = torch.randint(
                -128, 128, (K, N), dtype=dtype, device=flag_gems.device
            )
            if self.use_out:
                out = torch.empty((M, N), dtype=torch.int32, device=flag_gems.device)
                yield mat1, mat2, {"out": out}
            else:
                yield mat1, mat2


@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Native aten._int_mm falls back to CPU on CANN 8.5 and CANN 9.0",
)
@pytest.mark.int_mm
def test_int_mm_benchmark():
    bench = IntMMBenchmark(
        "int_mm",
        torch._int_mm,
        flag_gems.int_mm,
    )
    bench.run()


@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Native aten._int_mm.out falls back to CPU on CANN 8.5 and CANN 9.0",
)
@pytest.mark.int_mm_out
def test_int_mm_out_benchmark():
    bench = IntMMBenchmark(
        "int_mm_out",
        torch.ops.aten._int_mm.out,
        flag_gems.int_mm_out,
        use_out=True,
    )
    bench.run()
