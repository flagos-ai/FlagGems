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


@pytest.mark.tanh
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_tanh():
    bench = base.UnaryPointwiseBenchmark(
        op_name="tanh", torch_op=torch.tanh, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.tanh_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_tanh_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="tanh_",
        torch_op=torch.tanh_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


class TanhBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def _inputs_for_shape(self, shape, dtype):
        inp = base.generate_tensor_input(shape, dtype, self.device)
        # Preserve the old forward-produced output and random upstream gradient.
        # Preparation stays outside the direct backward timer.
        output = torch.tanh(inp)
        grad_output = torch.randn_like(output)
        return grad_output, output

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield self._inputs_for_shape(shape, dtype)

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"grad_output": shape, "output": shape},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        return self._inputs_for_shape(shape, case.dtype)


@pytest.mark.tanh_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_tanh_backward():
    bench = TanhBackwardBenchmark(
        op_name="tanh_backward",
        torch_op=torch.ops.aten.tanh_backward.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
