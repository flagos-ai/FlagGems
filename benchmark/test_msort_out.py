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

from . import base, consts, utils
from .test_msort import _MsortH800Reference


class MsortOutBenchmark(_MsortH800Reference, base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (8, 1024),
            (32, 4096),
            (128, 8192),
            (512, 4096),
            (32, 131072),
        ]
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend([(8, 262144), (128, 65536), (512, 16384)])

    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape), "out": list(shape)},
                    params={},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        inp = utils.generate_tensor_input(shape, case.dtype, self.device)
        return inp, {"out": torch.empty_like(inp)}


@pytest.mark.msort_out
@pytest.mark.skip_native(
    vendors=["enflame"],
    reason="S60 native msort_out aborts; consume exact native H800 core records",
)
def test_msort_out():
    MsortOutBenchmark(
        op_name="msort_out",
        torch_op=torch.msort,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    ).run()
