# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(4, 32, 256, 256), (4, 64, 512, 512), (8, 64, 256, 256)]


class QuantizedLstmBenchmark(OperatorBenchmark):
    def _candidate_call(self):
        op, dispatch, overridden = super()._candidate_call()
        if overridden:
            # deq belongs only to the native floating-point reference. The
            # injected quantized_lstm keeps its packed-parameter public ABI.
            def candidate(value, hx, params, deq, *args, **kwargs):
                return op(value, hx, params, *args, **kwargs)

            return candidate, dispatch, overridden
        return op, dispatch, overridden

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "seq,batch,input,hidden"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    seq, batch, inp, hidden = dims
    yield base.BenchmarkCasePlan(
        shape={"input": [seq, batch, inp], "hx": [1, batch, hidden]},
        params={"layers": 1, "bidirectional": False}, builder_args=(dims,),
    )


def _quantized_engine():
    # oneDNN preserves CellParamsBase/linear_unpack on Ascend CPU builds
    # that do not provide FBGEMM.
    if (
        flag_gems.vendor_name == "ascend"
        and "fbgemm" not in torch.backends.quantized.supported_engines
    ):
        return "onednn"
    return "fbgemm"


def _pack(input_size, hidden_size, device, dtype):
    original_engine = torch.backends.quantized.engine
    torch.backends.quantized.engine = _quantized_engine()
    try:
        w_ih = torch.randn(4 * hidden_size, input_size) * 0.3
        w_hh = torch.randn(4 * hidden_size, hidden_size) * 0.3
        b_ih = torch.randn(4 * hidden_size) * 0.1
        b_hh = torch.randn(4 * hidden_size) * 0.1
        q_ih = torch.quantize_per_tensor(w_ih, max(w_ih.abs().max().item() / 127, 1e-8), 0, torch.qint8)
        q_hh = torch.quantize_per_tensor(w_hh, max(w_hh.abs().max().item() / 127, 1e-8), 0, torch.qint8)
        packed_ih = torch.ops.quantized.linear_prepack(q_ih, b_ih)
        packed_hh = torch.ops.quantized.linear_prepack(q_hh, b_hh)
        params = torch.ops.quantized.make_quantized_cell_params_dynamic(
            packed_ih, packed_hh, b_ih, b_hh, True
        )
        deq = [torch.ops.quantized.linear_unpack(packed_ih)[0].dequantize().to(device=device, dtype=dtype),
               torch.ops.quantized.linear_unpack(packed_hh)[0].dequantize().to(device=device, dtype=dtype),
               b_ih.to(device=device, dtype=dtype), b_hh.to(device=device, dtype=dtype)]
        return params, deq
    finally:
        torch.backends.quantized.engine = original_engine


def _build_inputs_fn(plan, dtype, device):
    torch.manual_seed(0)
    seq, batch, inp_size, hidden = plan.builder_args[0]
    value = torch.randn(seq, batch, inp_size, dtype=dtype, device=device)
    hx = [torch.randn(1, batch, hidden, dtype=dtype, device=device),
          torch.randn(1, batch, hidden, dtype=dtype, device=device)]
    params, deq = _pack(inp_size, hidden, device, dtype)
    return value, hx, [params], deq, True, 1, 0.0, False, False, False


def _torch_lstm(value, hx, params, deq, *args):
    del params
    return torch._VF.lstm(value, hx, deq, *args)


@pytest.mark.quantized_lstm
def test_quantized_lstm():
    bench = QuantizedLstmBenchmark(
        op_name="quantized_lstm", torch_op=_torch_lstm,
        case_fn=_case_fn, build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(lambda value, hx, params, deq, *args: flag_gems.quantized_lstm(value, hx, params, *args))
    bench.run()
