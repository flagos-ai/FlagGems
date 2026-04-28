import pytest
import torch

import flag_gems

from . import base, consts


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    residual = torch.randn(shape, dtype=dtype, device=device)
    layer_shape = (shape[-1],)
    weight = torch.randn(layer_shape, dtype=dtype, device=device)
    bias = torch.randn(layer_shape, dtype=dtype, device=device)

    yield inp, residual, layer_shape, weight, bias


def torch_op(inp, residual, layer_shape, weight, bias):
    return torch.layer_norm(inp + residual, layer_shape, weight, bias)


@pytest.mark.skip_layer_norm
def test_skip_layernorm():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_input_fn,
        op_name="skip_layernorm",
        gems_op=flag_gems.skip_layer_norm,
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
