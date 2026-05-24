import pytest
import torch
import torch.nn as nn

import flag_gems

from . import base, consts


class _MockGemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps


def _input_fn_no_residual(shape, dtype, device):
    module = _MockGemmaRMSNorm(shape[-1], eps=1e-6, dtype=dtype, device=device)
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield module, inp


def _input_fn_with_residual(shape, dtype, device):
    module = _MockGemmaRMSNorm(shape[-1], eps=1e-6, dtype=dtype, device=device)
    inp = torch.randn(shape, dtype=dtype, device=device)
    residual = torch.randn(shape, dtype=dtype, device=device)
    yield module, inp, residual


def torch_op_no_residual(module, x):
    upcast_x = x.to(torch.float32)
    variance = upcast_x.pow(2).mean(-1, keepdim=True)
    hidden_states = upcast_x * torch.rsqrt(variance + module.variance_epsilon)
    return ((1.0 + module.weight.to(torch.float32)) * hidden_states).to(x.dtype)


def torch_op_with_residual(module, x, residual):
    x = x + residual
    upcast_x = x.to(torch.float32)
    variance = upcast_x.pow(2).mean(-1, keepdim=True)
    hidden_states = upcast_x * torch.rsqrt(variance + module.variance_epsilon)
    return ((1.0 + module.weight.to(torch.float32)) * hidden_states).to(x.dtype)


@pytest.mark.gemma_rms_norm
def test_gemma_rms_norm():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_input_fn_no_residual,
        op_name="gemma_rms_norm",
        torch_op=torch_op_no_residual,
        gems_op=flag_gems.gemma_rms_norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.gemma_rms_norm
def test_gemma_rms_norm_with_residual():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_input_fn_with_residual,
        op_name="gemma_rms_norm_with_residual",
        torch_op=torch_op_with_residual,
        gems_op=flag_gems.gemma_rms_norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
