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

BENCHMARK_CASES = [
    ((4, 8, 8, 16, 16), (12, 24, 24), (1.5, 1.5, 1.5)),
    ((4, 16, 24, 32, 40), (12, 16, 20), (None, None, None)),
    ((2, 16, 16, 24, 32), (32, 48, 64), (2.0, 2.0, 2.0)),
    ((2, 8, 20, 32, 48), (30, 40, 24), (None, None, None)),
    ((8, 32, 32, 48, 64), (16, 24, 32), (None, None, None)),
]


class UpsampleNearestExact3dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # (input size, output size, explicit scales).  Include upsampling,
        # anisotropic resize and downsampling, while keeping allocations bounded.
        self.shapes = BENCHMARK_CASES

    def set_more_shapes(self):
        return None

    def get_case_iter(self, cur_dtype):
        for ordinal, (input_size, output_size, scales) in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": [*input_size[:2], *output_size]},
                    builder_args=(input_size, output_size, scales),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        input_size, output_size, scales = plan.builder_args
        grad_output = torch.randn(
            (*input_size[:2], *output_size),
            dtype=case.dtype,
            device=self.device,
        )
        return grad_output, output_size, input_size, *scales


class UpsampleNearestExact3dBackwardGradInputBenchmark(
    UpsampleNearestExact3dBackwardBenchmark
):
    def build_inputs(self, case):
        plan = case.builder_args[0]
        input_size, output_size, scales = plan.builder_args
        grad_output = torch.randn(
            (*input_size[:2], *output_size),
            dtype=case.dtype,
            device=self.device,
        )
        grad_input = torch.empty(input_size, dtype=case.dtype, device=self.device)
        return (
            grad_output,
            output_size,
            input_size,
            *scales,
            {"grad_input": grad_input},
        )


@pytest.mark.upsample_nearest_exact3d_backward
@pytest.mark.underscore_upsample_nearest_exact3d_backward
def test_upsample_nearest_exact3d_backward():
    bench = UpsampleNearestExact3dBackwardBenchmark(
        op_name="_upsample_nearest_exact3d_backward",
        torch_op=torch.ops.aten._upsample_nearest_exact3d_backward.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.upsample_nearest_exact3d_backward_grad_input
@pytest.mark.underscore_upsample_nearest_exact3d_backward_grad_input
def test_upsample_nearest_exact3d_backward_grad_input():
    bench = UpsampleNearestExact3dBackwardGradInputBenchmark(
        op_name="_upsample_nearest_exact3d_backward_grad_input",
        torch_op=torch.ops.aten._upsample_nearest_exact3d_backward.grad_input,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
