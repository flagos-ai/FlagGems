import pytest
import torch

import flag_gems

from .attri_util import FLOAT_DTYPES
from .performance_utils import (
    GenericBenchmark,
    GenericBenchmarkExcluse1D,
    binary_input_fn,
)


@pytest.mark.gelu_and_mul
def test_perf_gelu_and_mul():
    def torch_op(x, y):
        return torch.mul(torch.nn.functional.gelu(x), y)

    gems_op = flag_gems.gelu_and_mul
    bench = GenericBenchmark(
        input_fn=binary_input_fn,
        op_name="gelu_and_mul",
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()


@pytest.mark.silu_and_mul
def test_perf_silu_and_mul():
    def torch_op(x, y):
        return torch.mul(torch.nn.functional.silu(x), y)

    gems_op = flag_gems.silu_and_mul

    bench = GenericBenchmark(
        input_fn=binary_input_fn,
        op_name="silu_and_mul",
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()


@pytest.mark.skip_layernorm
def test_perf_skip_layernorm():
    def skip_layernorm_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        residual = torch.randn(shape, dtype=dtype, device=device)
        layer_shape = (shape[-1],)
        weight = torch.randn(layer_shape, dtype=dtype, device=device)
        bias = torch.randn(layer_shape, dtype=dtype, device=device)
        yield inp, residual, layer_shape, weight, bias

    def torch_op(inp, residual, layer_shape, weight, bias):
        return torch.layer_norm(inp + residual, layer_shape, weight, bias)

    gems_op = flag_gems.skip_layer_norm

    bench = GenericBenchmarkExcluse1D(
        input_fn=skip_layernorm_input_fn,
        op_name="skip_layernorm",
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()


@pytest.mark.skip_rmsnorm
def test_perf_skip_rmsnorm():
    def skip_rmsnorm_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        residual = torch.randn(shape, dtype=dtype, device=device)
        layer_shape = (shape[-1],)
        weight = torch.randn(layer_shape, dtype=dtype, device=device)
        yield inp, residual, layer_shape, weight, 1e-5

    def torch_op(x, residual, layer_shape, weight, eps):
        x = x + residual
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states

    gems_op = flag_gems.skip_rms_norm

    bench = GenericBenchmarkExcluse1D(
        input_fn=skip_rmsnorm_input_fn,
        op_name="skip_rmsnorm",
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()


# TODO: apply_rotary_pos_emb
