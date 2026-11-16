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


def _make_workspace(batch_size, hidden_size, dtype, device):
    input_gates = torch.randn(batch_size, 3 * hidden_size, dtype=dtype, device=device)
    hidden_gates = torch.randn_like(input_gates)
    hx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    input_bias = torch.randn(3 * hidden_size, dtype=dtype, device=device)
    hidden_bias = torch.randn_like(input_bias)
    return torch.ops.aten._thnn_fused_gru_cell(
        input_gates, hidden_gates, hx, input_bias, hidden_bias
    )[1]


class FusedGRUCellBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_SHAPES)
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += list(COMPREHENSIVE_SHAPES)

    def get_input_iter(self, cur_dtype):
        for batch_size, hidden_size, has_bias in self.shapes:
            workspace = _make_workspace(batch_size, hidden_size, cur_dtype, self.device)
            grad_hy = torch.randn(
                batch_size,
                hidden_size,
                dtype=cur_dtype,
                device=self.device,
            )
            yield grad_hy, workspace, has_bias


def _fused_gru_cell_backward_out(
    grad_hy, workspace, has_bias, out0, out1, out2, out3, out4
):
    return torch.ops.aten._thnn_fused_gru_cell_backward.out(
        grad_hy,
        workspace,
        has_bias,
        out0=out0,
        out1=out1,
        out2=out2,
        out3=out3,
        out4=out4,
    )


class FusedGRUCellBackwardOutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(batch, hidden) for batch, hidden, _ in CORE_SHAPES]
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += [
                (batch, hidden) for batch, hidden, _ in COMPREHENSIVE_SHAPES
            ]

    def get_input_iter(self, cur_dtype):
        for batch_size, hidden_size in self.shapes:
            workspace = _make_workspace(batch_size, hidden_size, cur_dtype, self.device)
            grad_hy = torch.randn(
                batch_size,
                hidden_size,
                dtype=cur_dtype,
                device=self.device,
            )
            out0 = torch.empty(
                batch_size, 3 * hidden_size, dtype=cur_dtype, device=self.device
            )
            out1 = torch.empty_like(out0)
            out2 = torch.empty_like(grad_hy)
            out3 = torch.empty(3 * hidden_size, dtype=cur_dtype, device=self.device)
            out4 = torch.empty_like(out3)
            yield (
                grad_hy,
                workspace,
                True,
                out0,
                out1,
                out2,
                out3,
                out4,
            )


@pytest.mark.thnn_fused_gru_cell_backward
def test_thnn_fused_gru_cell_backward():
    bench = FusedGRUCellBackwardBenchmark(
        op_name="thnn_fused_gru_cell_backward",
        torch_op=torch.ops.aten._thnn_fused_gru_cell_backward,
        dtypes=DTYPES,
    )
    bench.run()


@pytest.mark.thnn_fused_gru_cell_backward_out
def test_thnn_fused_gru_cell_backward_out():
    bench = FusedGRUCellBackwardOutBenchmark(
        op_name="thnn_fused_gru_cell_backward_out",
        torch_op=_fused_gru_cell_backward_out,
        dtypes=DTYPES,
    )
    bench.run()
