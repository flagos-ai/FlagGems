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
from .conftest import Config

CORE_SHAPES = [
    (1, 32, False),
    (8, 64, True),
    (32, 128, False),
    (64, 256, True),
    (128, 512, False),
    (256, 1024, True),
]
COMPREHENSIVE_SHAPES = [
    (512, 1024, False),
    (512, 2048, True),
    (1024, 2048, True),
]
DTYPES = list(consts.FLOAT_DTYPES)
if flag_gems.runtime.device.support_fp64:
    DTYPES.append(torch.float64)


class DifferentiableGRUCellBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_SHAPES)
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += self.set_more_shapes()

    def set_more_shapes(self):
        return list(COMPREHENSIVE_SHAPES)

    def get_input_iter(self, cur_dtype):
        for batch_size, hidden_size, has_bias in self.shapes:
            grad_hy = torch.randn(
                batch_size, hidden_size, dtype=cur_dtype, device=self.device
            )
            input_gates = torch.randn(
                batch_size, 3 * hidden_size, dtype=cur_dtype, device=self.device
            )
            hidden_gates = torch.randn_like(input_gates)
            hx = torch.randn_like(grad_hy)
            if has_bias:
                input_bias = torch.randn(
                    3 * hidden_size, dtype=cur_dtype, device=self.device
                )
                hidden_bias = torch.randn_like(input_bias)
            else:
                input_bias = None
                hidden_bias = None
            yield grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias


@pytest.mark.thnn_differentiable_gru_cell_backward
def test_thnn_differentiable_gru_cell_backward():
    bench = DifferentiableGRUCellBackwardBenchmark(
        op_name="thnn_differentiable_gru_cell_backward",
        torch_op=torch.ops.aten._thnn_differentiable_gru_cell_backward,
        dtypes=DTYPES,
    )
    bench.run()
