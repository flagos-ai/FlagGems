import torch

import flag_gems

from .performance_utils import (
    FLOAT_DTYPES,
    POINTWISE_BATCH,
    REDUCTION_BATCH,
    SIZES,
    Benchmark,
    binary_args,
)


def test_perf_gelu_and_mul():
    def torch_op(x, y):
        return torch.mul(torch.nn.functional.gelu(x), y)

    gems_op = flag_gems.gelu_and_mul

    bench = Benchmark(
        op_name="gelu_and_mul",
        torch_op=torch_op,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.set_gems(gems_op)
    bench.run()


def test_perf_silu_and_mul():
    def torch_op(x, y):
        return torch.mul(torch.nn.functional.silu(x), y)

    gems_op = flag_gems.silu_and_mul

    bench = Benchmark(
        op_name="silu_and_mul",
        torch_op=torch_op,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.set_gems(gems_op)
    bench.run()


def test_perf_skip_layernorm():
    def skip_layernorm_args(dtype, batch, size):
        inp = torch.randn([batch, size], dtype=dtype, device="cuda")
        residual = torch.randn([batch, size], dtype=dtype, device="cuda")
        weight = torch.randn(
            [
                size,
            ],
            dtype=dtype,
            device="cuda",
        )
        bias = torch.randn(
            [
                size,
            ],
            dtype=dtype,
            device="cuda",
        )
        return (
            inp,
            residual,
            [
                size,
            ],
            weight,
            bias,
        )

    def torch_op(inp, residual, layer_shape, weight, bias):
        return torch.layer_norm(inp + residual, layer_shape, weight, bias)

    gems_op = flag_gems.skip_layer_norm

    bench = Benchmark(
        op_name="skip_layernorm",
        torch_op=torch_op,
        arg_func=skip_layernorm_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.set_gems(gems_op)
    bench.run()


def test_perf_skip_rmsnorm():
    def skip_rmsnorm_args(dtype, batch, size):
        inp = torch.randn([batch, size], dtype=dtype, device="cuda")
        residual = torch.randn([batch, size], dtype=dtype, device="cuda")
        weight = torch.randn(
            [
                size,
            ],
            dtype=dtype,
            device="cuda",
        )
        return (
            inp,
            residual,
            [
                size,
            ],
            weight,
            1e-5,
        )

    def torch_op(x, residual, layer_shape, weight, eps):
        x = x + residual
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states

    gems_op = flag_gems.skip_rms_norm

    bench = Benchmark(
        op_name="skip_rmsnorm",
        torch_op=torch_op,
        arg_func=skip_rmsnorm_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.set_gems(gems_op)
    bench.run()
