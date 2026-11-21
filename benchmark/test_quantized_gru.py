# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(1, 4, 10, 20), (10, 8, 32, 64), (32, 16, 64, 128), (64, 32, 128, 256)]
_FLOAT_PARAMS = {}


class QuantizedGruBenchmark(OperatorBenchmark):
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


def _build_inputs_fn(plan, dtype, device):
    torch.manual_seed(0)
    seq, batch, inp_size, hidden = plan.builder_args[0]
    gru = torch.nn.GRU(inp_size, hidden).eval()
    gru.qconfig = torch.ao.quantization.default_dynamic_qconfig
    qgru = torch.ao.nn.quantized.dynamic.GRU.from_float(gru)
    params = [module.param for module in qgru._all_weight_values]
    _FLOAT_PARAMS[id(params[0])] = [p.detach().to(device=device, dtype=dtype) for p in gru.parameters()]
    value = torch.randn(seq, batch, inp_size, dtype=dtype, device=device)
    hx = torch.randn(1, batch, hidden, dtype=dtype, device=device)
    return value, hx, params, True, 1, 0.0, False, False, False


def _torch_gru(value, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first):
    return torch.ops.aten.gru.input(value, hx, _FLOAT_PARAMS[id(params[0])], has_biases,
                                    num_layers, dropout, train, bidirectional, batch_first)


@pytest.mark.quantized_gru
def test_quantized_gru():
    if flag_gems.vendor_name == "hygon" and not (
        base.Config.list_cases or base.Config.preflight_only
    ):
        pytest.skip("PyTorch has no native quantized_gru accelerator kernel on Hygon")
    bench = QuantizedGruBenchmark(
        op_name="quantized_gru", torch_op=_torch_gru,
        case_fn=_case_fn, build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32, torch.float16],
    )
    registry = base.Config.override_registry
    if registry is None or registry.get_override(bench.op_name) is None:
        bench.set_gems(flag_gems.quantized_gru_input)
    bench.run()
