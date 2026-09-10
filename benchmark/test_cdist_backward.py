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


class CdistBackwardBenchmark(base.Benchmark):
    def set_more_shapes(self):
        return [
            (2, 16, 32),
            (4, 32, 64),
            (8, 64, 128),
            (16, 128, 256),
        ]

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            batch, n1, dim = shape
            n2 = n1 // 2 + 1
            x1 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            x2 = torch.randn(batch, n2, dim, dtype=cur_dtype, device=self.device)
            cdist = torch.cdist(x1, x2, p=2.0)
            grad = torch.randn(batch, n1, n2, dtype=cur_dtype, device=self.device)
            yield grad, x1, x2, 2.0, cdist


@pytest.mark.cdist_backward
def test_cdist_backward():
    bench = CdistBackwardBenchmark(
        op_name="cdist_backward",
        torch_op=torch.ops.aten._cdist_backward,
        # _cdist_backward uses fp32 accumulation; only float32 is numerically stable
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems._cdist_backward)
    bench.run()
