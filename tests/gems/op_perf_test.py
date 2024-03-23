import numpy as np
import itertools
import torch
import time
import triton
from gems import *


def run_bench(op, *args, warmups=10, repetitions=1000, **kwargs):    
    for i in range(warmups):
        ref_out = op(*args, **kwargs)
    start = time.time()
    for i in range(repetitions):
        ref_out = op(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()
    ms = (end - start) * 1000
    return ms


class Benchmark:
    def __init__(self, op_name):
        self.op_name = op_name

    def provider_ops(self, gem=None, torch=None):
        assert gem is not None
        assert torch is not None
        self.provider_ops = {'gem': gem, 'torch': torch}

    def bench_params(self, **params):
        self.bench_params = params

    def arg_names(self, *arg_names):
        self.x_names = arg_names
    
    def arg_vals(self, arg_vals):
        self.x_vals = arg_vals
    
    def extra_args(self, **args):
        self.extra_args = args

    def perf(self, fn):
        line_names, line_vals = zip(*self.provider_ops.items())
        bench_param_names, bench_param_vals = zip(*self.bench_params.items())
        benchmarks = (
            triton.testing.Benchmark(
                x_names=self.x_names,
                x_vals=self.x_vals,
                line_arg='op',
                line_names=list(line_names),
                line_vals=list(line_vals),
                styles=[("red", "-"), ("green", "-")],
                ylabel="ms",
                plot_name='test_performance_{}_{}'.format(self.op_name, '_'.join(str(e) for e in bench_param_set)),
                args={**self.extra_args, **dict(zip(bench_param_names, bench_param_set))}
            )
            for bench_param_set in itertools.product(*bench_param_vals)
        )
        return triton.testing.perf_report(benchmarks)(fn)

f16_f32_bf = (torch.float16, torch.float32, torch.bfloat16)
sizes = list(np.arange(1, 20) * 64)
mnk_sizes = list(zip(sizes, sizes, sizes))

addmm_bench = Benchmark('addmm')
addmm_bench.bench_params(dtype=f16_f32_bf)
addmm_bench.provider_ops(gem=addmm, torch=torch.addmm)
addmm_bench.arg_names('M', 'N', 'K')
addmm_bench.arg_vals(mnk_sizes)
addmm_bench.extra_args(alpha=1.0, beta=1.0)
@addmm_bench.perf
def bench_addmm(op, M, N, K, alpha, beta, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device="cuda")
    mat2 = torch.randn((K, N), dtype=dtype, device="cuda")
    bias = torch.randn((N,), dtype=dtype, device="cuda")
    ms = run_bench(op, bias, mat1, mat2, alpha=alpha, beta=beta)
    return ms

bmm_bench = Benchmark('bmm')
bmm_bench.bench_params(dtype=f16_f32_bf)
bmm_bench.provider_ops(gem=bmm, torch=torch.bmm)
bmm_bench.arg_names('M', 'N', 'K')
bmm_bench.arg_vals(mnk_sizes)
bmm_bench.extra_args(batch=4)
@bmm_bench.perf
def bench_bmm(op, batch, M, N, K, dtype):
    tensor_A = torch.randn((batch, M, K), dtype=dtype, device="cuda")
    tensor_B = torch.randn((batch, K, N), dtype=dtype, device="cuda")
    ms = run_bench(op, tensor_A, tensor_B)
    return ms

cumsum_bench = Benchmark('cumsum')
cumsum_bench.bench_params(dtype=f16_f32_bf)
cumsum_bench.provider_ops(gem=cumsum, torch=torch.cumsum)
cumsum_bench.arg_names('N')
cumsum_bench.arg_vals(sizes)
cumsum_bench.extra_args(M=1024, dim=1)
@cumsum_bench.perf
def bench_cumsum(op, M, N, dim, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp, dim=dim)
    return ms

dropout_bench = Benchmark('dropout')
dropout_bench.bench_params(dtype=f16_f32_bf, p=(0.3, 0.6, 0.9))
dropout_bench.provider_ops(gem=dropout, torch=torch.nn.functional.dropout)
dropout_bench.arg_names('N')
dropout_bench.arg_vals(sizes)
dropout_bench.extra_args(M=1024)
@dropout_bench.perf
def bench_dropout(op, M, N, p, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp, p=p, training=True)
    return ms

gelu_bench = Benchmark('gelu')
gelu_bench.bench_params(dtype=f16_f32_bf)
gelu_bench.provider_ops(gem=gelu, torch=torch.nn.functional.gelu)
gelu_bench.arg_names('N')
gelu_bench.arg_vals(sizes)
gelu_bench.extra_args(M=1024)
@gelu_bench.perf
def bench_gelu(op, M, N, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp)


configs_layernorm = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[i * 64 for i in range(1, 20)],
        line_arg="provider",
        line_vals=["gems", "torch"],
        line_names=["gems", "torch"],
        styles=[("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"test_performance_layernorm_{dtype}",
        args={"M": 1024, "dtype": dtype},
    )
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
]


@triton.testing.perf_report(configs_layernorm)
def test_performance_layernorm(M, N, dtype, provider):
    layer_shape = (N,)
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    weight = torch.randn(layer_shape, dtype=dtype, device="cuda")
    bias = torch.randn(layer_shape, dtype=dtype, device="cuda")
    eps = 1e-5

    if provider == "torch":
        for i in range(5):
            ref_out = torch.layer_norm(
                inp, list(layer_shape), weight=weight, bias=bias, eps=eps
            )
        start = time.time()
        for i in range(1000):
            ref_out = torch.layer_norm(
                inp, list(layer_shape), weight=weight, bias=bias, eps=eps
            )
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000
    if provider == "gems":
        for i in range(5):
            res_out = layer_norm(
                inp, list(layer_shape), weight=weight, bias=bias, eps=eps
            )
        start = time.time()
        for i in range(1000):
            res_out = layer_norm(
                inp, list(layer_shape), weight=weight, bias=bias, eps=eps
            )
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000

    return ms


configs_mm = [
    triton.testing.Benchmark(
        x_names=["MNK"],
        x_vals=[i * 64 for i in range(1, 20)],
        line_arg="provider",
        line_vals=["gems", "torch"],
        line_names=["gems", "torch"],
        styles=[("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"test_performance_mm_{dtype}",
        args={"dtype": dtype},
    )
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
]


@triton.testing.perf_report(configs_mm)
def test_performance_mm(MNK, dtype, provider):
    M = N = K = MNK
    tensor_a = torch.randn((M, K), dtype=dtype, device="cuda")
    tensor_b = torch.randn((K, N), dtype=dtype, device="cuda")

    if provider == "torch":
        for i in range(5):
            ref_out = torch.mm(tensor_a, tensor_b)
        start = time.time()
        for i in range(1000):
            ref_out = torch.mm(tensor_a, tensor_b)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000
    if provider == "gems":
        for i in range(5):
            res_out = mm(tensor_a, tensor_b)
        start = time.time()
        for i in range(1000):
            res_out = mm(tensor_a, tensor_b)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000

    return ms


configs_relu = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[i * 64 for i in range(1, 20)],
        line_arg="provider",
        line_vals=["gems", "torch"],
        line_names=["gems", "torch"],
        styles=[("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"test_performance_relu_{dtype}",
        args={"M": 1024, "dtype": dtype},
    )
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
]


@triton.testing.perf_report(configs_relu)
def test_performance_relu(M, N, dtype, provider):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")

    if provider == "torch":
        for i in range(5):
            ref_out = torch.nn.functional.relu(inp)
        start = time.time()
        for i in range(1000):
            ref_out = torch.nn.functional.relu(inp)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000
    if provider == "gems":
        for i in range(5):
            res_out = relu(inp)
        start = time.time()
        for i in range(1000):
            res_out = relu(inp)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000

    return ms


configs_silu = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[i * 64 for i in range(1, 20)],
        line_arg="provider",
        line_vals=["gems", "torch"],
        line_names=["gems", "torch"],
        styles=[("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"test_performance_silu_{dtype}",
        args={"M": 1024, "dtype": dtype},
    )
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
]


@triton.testing.perf_report(configs_silu)
def test_performance_silu(M, N, dtype, provider):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")

    if provider == "torch":
        for i in range(5):
            ref_out = torch.nn.functional.silu(inp)
        start = time.time()
        for i in range(1000):
            ref_out = torch.nn.functional.silu(inp)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000
    if provider == "gems":
        for i in range(5):
            res_out = silu(inp)
        start = time.time()
        for i in range(1000):
            res_out = silu(inp)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000

    return ms


configs_softmax = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[i * 64 for i in range(1, 20)],
        line_arg="provider",
        line_vals=["gems", "torch"],
        line_names=["gems", "torch"],
        styles=[("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"test_performance_softmax_{dtype}",
        args={"M": 1024, "dim": 1, "dtype": dtype},
    )
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
]


@triton.testing.perf_report(configs_softmax)
def test_performance_softmax(M, N, dim, dtype, provider):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")

    if provider == "torch":
        for i in range(5):
            ref_out = torch.nn.functional.softmax(inp, dim=dim)
        start = time.time()
        for i in range(1000):
            ref_out = torch.nn.functional.softmax(inp, dim=dim)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000
    if provider == "gems":
        for i in range(5):
            res_out = softmax(inp, dim=dim)
        start = time.time()
        for i in range(1000):
            res_out = softmax(inp, dim=dim)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000

    return ms


configs_triu = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[i * 64 for i in range(1, 20)],
        line_arg="provider",
        line_vals=["gems", "torch"],
        line_names=["gems", "torch"],
        styles=[("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"test_performance_triu_{dtype}",
        args={"M": 1024, "dtype": dtype},
    )
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
]


@triton.testing.perf_report(configs_triu)
def test_performance_triu(M, N, dtype, provider):
    layer_shape = (N,)
    diagonal = 1
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    weight = torch.randn(layer_shape, dtype=dtype, device="cuda")
    bias = torch.randn(layer_shape, dtype=dtype, device="cuda")
    eps = 1e-5

    if provider == "torch":
        for i in range(5):
            ref_out = torch.triu(inp, diagonal)
        start = time.time()
        for i in range(1000):
            ref_out = torch.triu(inp, diagonal)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000
    if provider == "gems":
        for i in range(5):
            res_out = triu(inp, diagonal)
        start = time.time()
        for i in range(1000):
            res_out = triu(inp, diagonal)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000

    return ms


bench_addmm.run(print_data=True)
bench_bmm.run(print_data=True)
bench_cumsum.run(print_data=True)
bench_dropout.run(print_data=True)
bench_gelu.run(print_data=True)
test_performance_layernorm.run(print_data=True)
test_performance_mm.run(print_data=True)
test_performance_relu.run(print_data=True)
test_performance_silu.run(print_data=True)
test_performance_softmax.run(print_data=True)
test_performance_triu.run(print_data=True)
