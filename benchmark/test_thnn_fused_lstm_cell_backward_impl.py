# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

import pytest
import torch

from . import base
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(1, 4), (4, 16), (16, 64)]


class LstmCellBackwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "batch,hidden"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"cx": list(dims), "workspace": [dims[0], 4 * dims[1]]},
        params={"has_bias": True}, builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    torch.manual_seed(0)
    batch, hidden = plan.builder_args[0]
    input_gates = torch.randn(batch, 4 * hidden, dtype=dtype, device=device)
    hidden_gates = torch.randn_like(input_gates)
    cx = torch.randn(batch, hidden, dtype=dtype, device=device)
    input_bias = torch.zeros(4 * hidden, dtype=dtype, device=device)
    hidden_bias = torch.randn(4 * hidden, dtype=dtype, device=device)
    hx, cy, workspace = torch.ops.aten._thnn_fused_lstm_cell(
        input_gates, hidden_gates, cx, input_bias, hidden_bias
    )
    return torch.randn_like(hx), torch.randn_like(cy), cx, cy, workspace, True


@pytest.mark.thnn_fused_lstm_cell_backward_impl
@pytest.mark.underscore_thnn_fused_lstm_cell_backward_impl
def test_thnn_fused_lstm_cell_backward_impl():
    LstmCellBackwardBenchmark(
        op_name="_thnn_fused_lstm_cell_backward_impl",
        torch_op=torch.ops.aten._thnn_fused_lstm_cell_backward_impl,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()
