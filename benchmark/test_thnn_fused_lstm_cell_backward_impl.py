# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark
from .test_thnn_fused_lstm_cell_backward import _FusedCellStaticBaseline

SHAPES = [(1, 4), (4, 16), (16, 64)]


def lstm_forward(input_gates, hidden_gates, cx, input_bias=None, hidden_bias=None):
    dtype = cx.dtype
    compute = torch.float64 if dtype == torch.float64 else torch.float32
    x = input_gates.detach().cpu().to(compute) + hidden_gates.detach().cpu().to(compute)
    if input_bias is not None:
        x = (
            x
            + input_bias.detach().cpu().to(compute)
            + hidden_bias.detach().cpu().to(compute)
        )
    i, f, g, o = x.chunk(4, 1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    c = f * cx.detach().cpu().to(compute) + i * g
    h = o * c.tanh()
    return h.to(dtype), c.to(dtype), torch.cat((i, f, g, o), 1).to(dtype)


def _forward_workspace(*args):
    # PyTorch CUDA RNN.cu stores activated gates in the original scalar dtype.
    # Enflame has no fused-cell CPU kernel behind its native fallback. Preserve
    # target RNG construction, then prepare the dense packet on CPU explicitly.
    if flag_gems.vendor_name != "enflame":
        return torch.ops.aten._thnn_fused_lstm_cell(*args)
    values = lstm_forward(*args)
    return tuple(value.to(args[0].device) for value in values)


class LstmCellBackwardBenchmark(_FusedCellStaticBaseline, OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "batch,hidden"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"cx": list(dims), "workspace": [dims[0], 4 * dims[1]]},
        params={"has_bias": True},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    torch.manual_seed(0)
    batch, hidden = plan.builder_args[0]
    input_gates = torch.randn(batch, 4 * hidden, dtype=dtype, device=device)
    hidden_gates = torch.randn_like(input_gates)
    cx = torch.randn(batch, hidden, dtype=dtype, device=device)
    input_bias = torch.zeros(4 * hidden, dtype=dtype, device=device)
    hidden_bias = torch.randn(4 * hidden, dtype=dtype, device=device)
    hx, cy, workspace = _forward_workspace(
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
