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
from tests import gru_cpu_reference as gru_oracle

from . import base, consts
from .conftest import Config
from .test_thnn_fused_gru_cell import _H800GRUReference

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


def _make_out_workspace(batch_size, hidden_size, dtype, device):
    input_gates = torch.randn(batch_size, 3 * hidden_size, dtype=dtype, device=device)
    hidden_gates = torch.randn_like(input_gates)
    hx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    input_bias = torch.randn(3 * hidden_size, dtype=dtype, device=device)
    hidden_bias = torch.randn_like(input_bias)
    if flag_gems.vendor_name == "ascend":
        return gru_oracle.forward(
            input_gates, hidden_gates, hx, input_bias, hidden_bias
        )[1]
    return torch.ops.aten._thnn_fused_gru_cell(
        input_gates, hidden_gates, hx, input_bias, hidden_bias
    )[1]


class FusedGRUCellBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_SHAPES)
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += list(COMPREHENSIVE_SHAPES)

    def get_case_iter(self, cur_dtype):
        for ordinal, (batch_size, hidden_size, has_bias) in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"grad_hy": [batch_size, hidden_size]},
                    params={"has_bias": has_bias},
                    builder_args=(batch_size, hidden_size, has_bias),
                ),
            )

    def build_inputs(self, case):
        batch_size, hidden_size, has_bias = case.builder_args[0].builder_args
        if flag_gems.vendor_name == "kunlunxin" and not has_bias:
            pytest.skip(
                "Kunlunxin native GRU backward returns biases when has_bias=False"
            )
        cur_dtype = case.dtype
        workspace = _make_workspace(batch_size, hidden_size, cur_dtype, self.device)
        grad_hy = torch.randn(
            batch_size,
            hidden_size,
            dtype=cur_dtype,
            device=self.device,
        )
        return grad_hy, workspace, has_bias


class FusedGRUCellBackwardOutBenchmark(_H800GRUReference, base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(batch, hidden) for batch, hidden, _ in CORE_SHAPES]
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += [
                (batch, hidden) for batch, hidden, _ in COMPREHENSIVE_SHAPES
            ]

    def get_case_iter(self, cur_dtype):
        for ordinal, (batch_size, hidden_size) in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"grad_hy": [batch_size, hidden_size]},
                    params={"has_bias": True},
                    builder_args=(batch_size, hidden_size),
                ),
            )

    def build_inputs(self, case):
        batch_size, hidden_size = case.builder_args[0].builder_args
        cur_dtype = case.dtype
        workspace = _make_out_workspace(batch_size, hidden_size, cur_dtype, self.device)
        grad_hy = torch.randn(
            batch_size, hidden_size, dtype=cur_dtype, device=self.device
        )
        out0 = torch.empty(
            batch_size, 3 * hidden_size, dtype=cur_dtype, device=self.device
        )
        out1 = torch.empty_like(out0)
        out2 = torch.empty_like(grad_hy)
        out3 = torch.empty(3 * hidden_size, dtype=cur_dtype, device=self.device)
        out4 = torch.empty_like(out3)
        return (
            grad_hy,
            workspace,
            True,
            {"out0": out0, "out1": out1, "out2": out2, "out3": out3, "out4": out4},
        )


@pytest.mark.thnn_fused_gru_cell_backward
@pytest.mark.underscore_thnn_fused_gru_cell_backward
def test_thnn_fused_gru_cell_backward():
    bench = FusedGRUCellBackwardBenchmark(
        op_name="_thnn_fused_gru_cell_backward",
        torch_op=torch.ops.aten._thnn_fused_gru_cell_backward,
        dtypes=DTYPES,
    )
    bench.run()


@pytest.mark.skip_native(
    vendors="ascend",
    reason="Native GRU backward reference unavailable; use the verified H800 --base-data comparison",
)
@pytest.mark.underscore_thnn_fused_gru_cell_backward_out
def test_thnn_fused_gru_cell_backward_out():
    bench = FusedGRUCellBackwardOutBenchmark(
        op_name="_thnn_fused_gru_cell_backward_out",
        torch_op=torch.ops.aten._thnn_fused_gru_cell_backward.out,
        dtypes=DTYPES,
    )
    bench.run()
