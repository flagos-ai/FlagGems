import pytest
import torch

import flag_gems

from . import base, consts


def _input_fn_gemma_rms_norm(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    layer_shape = (shape[-1],)
    weight = torch.randn(layer_shape, dtype=dtype, device=device) * 0.01
    yield inp, layer_shape, weight, 1e-6


def torch_op_gemma_rms_norm(x, layer_shape, weight, eps):
    upcast_x = x.to(torch.float32)
    variance = upcast_x.pow(2).mean(-1, keepdim=True)
    hidden_states = upcast_x * torch.rsqrt(variance + eps)
    return ((1.0 + weight.to(torch.float32)) * hidden_states).to(x.dtype)


def _input_fn_gemma_fused_add_rms_norm(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    residual = torch.randn(shape, dtype=dtype, device=device)
    layer_shape = (shape[-1],)
    weight = torch.randn(layer_shape, dtype=dtype, device=device) * 0.01
    yield inp, residual, layer_shape, weight, 1e-6


def torch_op_gemma_fused_add_rms_norm(x, residual, layer_shape, weight, eps):
    x = x + residual
    upcast_x = x.to(torch.float32)
    variance = upcast_x.pow(2).mean(-1, keepdim=True)
    hidden_states = upcast_x * torch.rsqrt(variance + eps)
    return ((1.0 + weight.to(torch.float32)) * hidden_states).to(x.dtype)


@pytest.mark.gemma_rms_norm
def test_gemma_rms_norm():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_input_fn_gemma_rms_norm,
        op_name="gemma_rms_norm",
        torch_op=torch_op_gemma_rms_norm,
        gems_op=flag_gems.gemma_rms_norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.gemma_fused_add_rms_norm
def test_gemma_fused_add_rms_norm():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_input_fn_gemma_fused_add_rms_norm,
        op_name="gemma_fused_add_rms_norm",
        torch_op=torch_op_gemma_fused_add_rms_norm,
        gems_op=flag_gems.gemma_fused_add_rms_norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
