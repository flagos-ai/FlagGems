# Copyright 2026, The FlagOS Contributors.
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


class PadCircularBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Representative (shape, pad) cases: small to medium tensors covering 1D/2D/3D
        # circular padding. Each pad respects |pad| <= dim (circular wraps at most once).
        self.shapes = [
            ((4096, 4096), [3, 3]),
            ((1024, 1024), [8, 8]),
            ((64, 512, 512), [2, 2, 2, 2]),
            ((256, 128, 128), [4, 4, 4, 4]),
            ((16, 32, 32, 32), [1, 1, 1, 1, 1, 1]),
        ]

    def set_more_shapes(self):
        return None

    def get_case_iter(self, dtype):
        for ordinal, (shape, pad) in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={"pad": pad},
                    builder_args=(shape, pad),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape, pad = plan.builder_args
        cur_dtype = case.dtype
        inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
        return inp, pad


@pytest.mark.pad_circular
@pytest.mark.underscore_pad_circular
def test_pad_circular():
    bench = PadCircularBenchmark(
        op_name="_pad_circular",
        torch_op=torch.ops.aten._pad_circular,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
