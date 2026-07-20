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


class UpsampleNearestExact1dBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(2, 3, 16), (4, 8, 64), (8, 16, 256), (16, 32, 512)]

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=dtype, device=self.device)
            out_size = [shape[-1] * 2]
            yield x, out_size, None


@pytest.mark.upsample_nearest_exact1d
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_upsample_nearest_exact1d():
    bench = UpsampleNearestExact1dBenchmark(
        op_name="upsample_nearest_exact1d",
        torch_op=torch.ops.aten._upsample_nearest_exact1d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
