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

from . import base, consts


CASES = [
    ((4, 3, 32, 32), (7, 7)),
    ((8, 64, 56, 56), (7, 7)),
    ((4, 128, 112, 112), (14, 14)),
]


class AdaptiveAvgPool2DBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CASES
        self.shape_desc = "input shape, output size"

    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        for ordinal, (shape, output_size) in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={
                        "self": list(shape),
                        "grad_output": [*shape[:-2], *output_size],
                    },
                    params={},
                    builder_args=(tuple(shape), tuple(output_size)),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape, output_size = plan.builder_args
        inp = torch.randn(shape, dtype=case.dtype, device=self.device)
        grad_output = torch.randn(
            (*shape[:-2], *output_size), dtype=case.dtype, device=self.device
        )
        return grad_output, inp


@pytest.mark.adaptive_avg_pool2d_backward
@pytest.mark.underscore_adaptive_avg_pool2d_backward
def test_adaptive_avg_pool2d_backward():
    AdaptiveAvgPool2DBackwardBenchmark(
        op_name="_adaptive_avg_pool2d_backward",
        torch_op=torch.ops.aten._adaptive_avg_pool2d_backward,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
