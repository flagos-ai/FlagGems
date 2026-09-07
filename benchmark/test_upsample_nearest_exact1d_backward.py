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

from flag_gems.ops._upsample_nearest_exact1d_backward import (
    _upsample_nearest_exact1d_backward,
    _upsample_nearest_exact1d_backward_grad_input,
)

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

    def get_input_iter(self, cur_dtype):
        for batch, channels, input_w, output_w in self.shapes:
            grad_output = torch.randn(
                (batch, channels, output_w),
                dtype=cur_dtype,
                device=self.device,
            )
            yield grad_output, (output_w,), (batch, channels, input_w)


class UpsampleNearestExact1dBackwardOutBenchmark(
    UpsampleNearestExact1dBackwardBenchmark
):
    def get_input_iter(self, cur_dtype):
        for args in super().get_input_iter(cur_dtype):
            grad_output, output_size, input_size = args
            grad_input = torch.empty(input_size, dtype=cur_dtype, device=self.device)
            yield grad_output, output_size, input_size, {"grad_input": grad_input}


@pytest.mark.upsample_nearest_exact1d_backward
def test_upsample_nearest_exact1d_backward():
    bench = UpsampleNearestExact1dBackwardBenchmark(
        op_name="upsample_nearest_exact1d_backward",
        torch_op=torch.ops.aten._upsample_nearest_exact1d_backward.default,
        gems_op=_upsample_nearest_exact1d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.upsample_nearest_exact1d_backward_grad_input
def test_upsample_nearest_exact1d_backward_grad_input():
    bench = UpsampleNearestExact1dBackwardOutBenchmark(
        op_name="upsample_nearest_exact1d_backward_grad_input",
        torch_op=torch.ops.aten._upsample_nearest_exact1d_backward.grad_input,
        gems_op=_upsample_nearest_exact1d_backward_grad_input,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
