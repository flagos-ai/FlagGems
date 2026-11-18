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

# consts.FLOAT_DTYPES omits float64; maximum supports it when the device does.
MAXIMUM_DTYPES = consts.FLOAT_DTYPES + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)


class MaximumBenchmark(base.BinaryPointwiseBenchmark):
    def _input(self, shape, dtype):
        return torch.randn(shape, dtype=dtype, device=self.device)

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"inputs": [shape, shape]},
                    builder_args=(shape,),
                ),
            )

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield self._input(shape, dtype), self._input(shape, dtype)

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        return self._input(shape, case.dtype), self._input(shape, case.dtype)


@pytest.mark.maximum
def test_maximum():
    bench = MaximumBenchmark(
        op_name="maximum",
        torch_op=torch.maximum,
        dtypes=MAXIMUM_DTYPES,
    )
    bench.run()
