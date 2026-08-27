# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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


class FusedGRUCellBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_SHAPES)
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += list(COMPREHENSIVE_SHAPES)

    def get_input_iter(self, cur_dtype):
        for batch_size, hidden_size, with_bias in self.shapes:
            input_gates = torch.randn(
                batch_size,
                3 * hidden_size,
                dtype=cur_dtype,
                device=self.device,
            )
            hidden_gates = torch.randn_like(input_gates)
            hx = torch.randn(
                batch_size,
                hidden_size,
                dtype=cur_dtype,
                device=self.device,
            )
            if with_bias:
                input_bias = torch.randn(
                    3 * hidden_size, dtype=cur_dtype, device=self.device
                )
                hidden_bias = torch.randn_like(input_bias)
            else:
                input_bias = None
                hidden_bias = None
            yield input_gates, hidden_gates, hx, input_bias, hidden_bias


def _fused_gru_cell_out(
    input_gates, hidden_gates, hx, input_bias, hidden_bias, out0, out1
):
    return torch.ops.aten._thnn_fused_gru_cell.out(
        input_gates,
        hidden_gates,
        hx,
        input_bias,
        hidden_bias,
        out0=out0,
        out1=out1,
    )


class FusedGRUCellOutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(batch, hidden) for batch, hidden, _ in CORE_SHAPES]
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += [
                (batch, hidden) for batch, hidden, _ in COMPREHENSIVE_SHAPES
            ]

    def get_input_iter(self, cur_dtype):
        for batch_size, hidden_size in self.shapes:
            input_gates = torch.randn(
                batch_size,
                3 * hidden_size,
                dtype=cur_dtype,
                device=self.device,
            )
            hidden_gates = torch.randn_like(input_gates)
            hx = torch.randn(
                batch_size,
                hidden_size,
                dtype=cur_dtype,
                device=self.device,
            )
            input_bias = torch.randn(
                3 * hidden_size, dtype=cur_dtype, device=self.device
            )
            hidden_bias = torch.randn_like(input_bias)
            out0 = torch.empty_like(hx)
            out1 = torch.empty(
                batch_size,
                5 * hidden_size,
                dtype=cur_dtype,
                device=self.device,
            )
            yield (
                input_gates,
                hidden_gates,
                hx,
                input_bias,
                hidden_bias,
                out0,
                out1,
            )


@pytest.mark.thnn_fused_gru_cell
def test_thnn_fused_gru_cell():
    bench = FusedGRUCellBenchmark(
        op_name="thnn_fused_gru_cell",
        torch_op=torch.ops.aten._thnn_fused_gru_cell,
        dtypes=DTYPES,
    )
    bench.run()


@pytest.mark.thnn_fused_gru_cell_out
def test_thnn_fused_gru_cell_out():
    bench = FusedGRUCellOutBenchmark(
        op_name="thnn_fused_gru_cell_out",
        torch_op=_fused_gru_cell_out,
        dtypes=DTYPES,
    )
    bench.run()
