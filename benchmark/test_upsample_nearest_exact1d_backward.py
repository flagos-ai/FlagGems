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


class UpsampleNearestExact1dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # (N, C, input width, output width): bounded core shapes spanning
        # launch-bound through throughput-bound workloads and both resize directions.
        self.shapes = [
            (1, 1, 8, 16),
            (2, 3, 64, 96),
            (8, 16, 512, 256),
            (16, 64, 2048, 4096),
            (8, 128, 8192, 12288),
        ]
        self.shape_desc = ["N", "C", "IW", "OW"]

    def set_more_shapes(self):
        return []

    def get_case_iter(self, cur_dtype):
        for ordinal, (batch, channels, input_w, output_w) in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": [batch, channels, output_w]},
                    builder_args=((batch, channels, input_w, output_w),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        batch, channels, input_w, output_w = plan.builder_args[0]
        grad_output = torch.randn(
            (batch, channels, output_w),
            dtype=case.dtype,
            device=self.device,
        )
        return grad_output, (output_w,), (batch, channels, input_w)


class UpsampleNearestExact1dBackwardOutBenchmark(
    UpsampleNearestExact1dBackwardBenchmark
):
    def build_inputs(self, case):
        grad_output, output_size, input_size = super().build_inputs(case)
        grad_input = torch.empty(input_size, dtype=case.dtype, device=self.device)
        return grad_output, output_size, input_size, {"grad_input": grad_input}


@pytest.mark.underscore_upsample_nearest_exact1d_backward
def test_upsample_nearest_exact1d_backward():
    bench = UpsampleNearestExact1dBackwardBenchmark(
        op_name="_upsample_nearest_exact1d_backward",
        torch_op=torch.ops.aten._upsample_nearest_exact1d_backward.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.underscore_upsample_nearest_exact1d_backward_grad_input
def test_upsample_nearest_exact1d_backward_grad_input():
    bench = UpsampleNearestExact1dBackwardOutBenchmark(
        op_name="_upsample_nearest_exact1d_backward_grad_input",
        torch_op=torch.ops.aten._upsample_nearest_exact1d_backward.grad_input,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
