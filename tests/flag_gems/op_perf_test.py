import itertools
import torch
import time
import triton
import random
from flag_gems import *


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
        self.provider_ops = {"gem": gem, "torch": torch}

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
                line_arg="op",
                line_names=list(line_names),
                line_vals=list(line_vals),
                styles=[("red", "-"), ("green", "-")],
                ylabel="ms",
                plot_name="test_performance_{}_{}".format(
                    self.op_name, "_".join(str(e) for e in bench_param_set)
                ),
                args={
                    **self.extra_args,
                    **dict(zip(bench_param_names, bench_param_set)),
                },
            )
            for bench_param_set in itertools.product(*bench_param_vals)
        )
        return triton.testing.perf_report(benchmarks)(fn)


f16_f32_bf = (torch.float16, torch.float32, torch.bfloat16)
sizes = [i * 64 for i in range(1, 20)]
mnk_sizes = list(zip(sizes, sizes, sizes))


abs_bench = Benchmark("abs")
abs_bench.bench_params(dtype=f16_f32_bf)
abs_bench.provider_ops(gem=abs, torch=torch.abs)
abs_bench.arg_names("N")
abs_bench.arg_vals(sizes)
abs_bench.extra_args(M=1024)


@abs_bench.perf
def bench_abs(op, M, N, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp)
    return ms


add_bench = Benchmark("add")
add_bench.bench_params(dtype=f16_f32_bf)
add_bench.provider_ops(gem=add, torch=torch.add)
add_bench.arg_names("N")
add_bench.arg_vals(sizes)
add_bench.extra_args(M=1024)


@add_bench.perf
def bench_add(op, M, N, dtype):
    inp1 = torch.randn((M, N), dtype=dtype, device="cuda")
    inp2 = torch.randn((M, N), dtype=dtype, device="cuda")
    alpha = random.random()
    ms = run_bench(op, inp1, inp2, alpha=alpha)
    return ms


@add_bench.perf
def bench_add_scalar(op, M, N, dtype):
    inp1 = torch.randn((M, N), dtype=dtype, device="cuda")
    inp2 = random.random()
    alpha = random.random()
    ms = run_bench(op, inp1, inp2, alpha=alpha)
    return ms


addmm_bench = Benchmark("addmm")
addmm_bench.bench_params(dtype=f16_f32_bf)
addmm_bench.provider_ops(gem=addmm, torch=torch.addmm)
addmm_bench.arg_names("M", "N", "K")
addmm_bench.arg_vals(mnk_sizes)
addmm_bench.extra_args(alpha=1.0, beta=1.0)


@addmm_bench.perf
def bench_addmm(op, M, N, K, alpha, beta, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device="cuda")
    mat2 = torch.randn((K, N), dtype=dtype, device="cuda")
    bias = torch.randn((N,), dtype=dtype, device="cuda")
    ms = run_bench(op, bias, mat1, mat2, alpha=alpha, beta=beta)
    return ms


bmm_bench = Benchmark("bmm")
bmm_bench.bench_params(dtype=f16_f32_bf)
bmm_bench.provider_ops(gem=bmm, torch=torch.bmm)
bmm_bench.arg_names("M", "N", "K")
bmm_bench.arg_vals(mnk_sizes)
bmm_bench.extra_args(batch=4)


@bmm_bench.perf
def bench_bmm(op, batch, M, N, K, dtype):
    tensor_A = torch.randn((batch, M, K), dtype=dtype, device="cuda")
    tensor_B = torch.randn((batch, K, N), dtype=dtype, device="cuda")
    ms = run_bench(op, tensor_A, tensor_B)
    return ms


cumsum_bench = Benchmark("cumsum")
cumsum_bench.bench_params(dtype=f16_f32_bf)
cumsum_bench.provider_ops(gem=cumsum, torch=torch.cumsum)
cumsum_bench.arg_names("N")
cumsum_bench.arg_vals(sizes)
cumsum_bench.extra_args(M=1024, dim=1)


@cumsum_bench.perf
def bench_cumsum(op, M, N, dim, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp, dim=dim)
    return ms


div_bench = Benchmark("div")
div_bench.bench_params(dtype=f16_f32_bf)
div_bench.provider_ops(gem=div, torch=torch.div)
div_bench.arg_names("N")
div_bench.arg_vals(sizes)
div_bench.extra_args(M=1024)


@div_bench.perf
def bench_div(op, M, N, dtype):
    inp1 = torch.randn((M, N), dtype=dtype, device="cuda")
    inp2 = torch.randn((M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp1, inp2)
    return ms


@div_bench.perf
def bench_div_scalar(op, M, N, dtype):
    inp1 = torch.randn((M, N), dtype=dtype, device="cuda")
    inp2 = random.randint(0, 1000)
    ms = run_bench(op, inp1, inp2)
    return ms


dropout_bench = Benchmark("dropout")
dropout_bench.bench_params(dtype=f16_f32_bf, p=(0.3, 0.6, 0.9))
dropout_bench.provider_ops(gem=native_dropout, torch=torch.nn.functional.dropout)
dropout_bench.arg_names("N")
dropout_bench.arg_vals(sizes)
dropout_bench.extra_args(M=1024)


@dropout_bench.perf
def bench_dropout(op, M, N, p, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp, p, True)
    return ms


exp_bench = Benchmark("exp")
exp_bench.bench_params(dtype=f16_f32_bf)
exp_bench.provider_ops(gem=exp, torch=torch.exp)
exp_bench.arg_names("N")
exp_bench.arg_vals(sizes)
exp_bench.extra_args(M=1024)


@exp_bench.perf
def bench_exp(op, M, N, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp)
    return ms


gelu_bench = Benchmark("gelu")
gelu_bench.bench_params(dtype=f16_f32_bf)
gelu_bench.provider_ops(gem=gelu, torch=torch.nn.functional.gelu)
gelu_bench.arg_names("N")
gelu_bench.arg_vals(sizes)
gelu_bench.extra_args(M=1024)


@gelu_bench.perf
def bench_gelu(op, M, N, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp)
    return ms


layernorm_bench = Benchmark("layernorm")
layernorm_bench.bench_params(dtype=f16_f32_bf)
layernorm_bench.provider_ops(gem=layer_norm, torch=torch.nn.functional.layer_norm)
layernorm_bench.arg_names("N")
layernorm_bench.arg_vals(sizes)
layernorm_bench.extra_args(M=1024)


@layernorm_bench.perf
def bench_layernorm(op, M, N, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    weight = torch.randn(N, dtype=dtype, device="cuda")
    bias = torch.randn(N, dtype=dtype, device="cuda")
    eps = 1e-5
    ms = run_bench(
        op,
        inp,
        normalized_shape=[
            N,
        ],
        weight=weight,
        bias=bias,
        eps=eps,
    )
    return ms


mean_bench = Benchmark("mean")
mean_bench.bench_params(dtype=f16_f32_bf)
mean_bench.provider_ops(gem=mean, torch=torch.mean)
mean_bench.arg_names("N")
mean_bench.arg_vals(sizes)
mean_bench.extra_args(M=1024)


@mean_bench.perf
def bench_mean(op, M, N, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp)
    return ms


mm_bench = Benchmark("mm")
mm_bench.bench_params(dtype=f16_f32_bf)
mm_bench.provider_ops(gem=mm, torch=torch.mm)
mm_bench.arg_names("M", "N", "K")
mm_bench.arg_vals(mnk_sizes)
mm_bench.extra_args()


@mm_bench.perf
def bench_mm(op, M, N, K, dtype):
    tensor_a = torch.randn((M, K), dtype=dtype, device="cuda")
    tensor_b = torch.randn((K, N), dtype=dtype, device="cuda")
    ms = run_bench(op, tensor_a, tensor_b)
    return ms


mul_bench = Benchmark("mul")
mul_bench.bench_params(dtype=f16_f32_bf)
mul_bench.provider_ops(gem=mul, torch=torch.mul)
mul_bench.arg_names("N")
mul_bench.arg_vals(sizes)
mul_bench.extra_args(M=1024)


@mul_bench.perf
def bench_mul(op, M, N, dtype):
    inp1 = torch.randn((M, N), dtype=dtype, device="cuda")
    inp2 = torch.randn((M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp1, inp2)
    return ms


@mul_bench.perf
def bench_mul_scalar(op, M, N, dtype):
    inp1 = torch.randn((M, N), dtype=dtype, device="cuda")
    inp2 = random.randint(0, 10000)
    ms = run_bench(op, inp1, inp2)
    return ms


reciprocal_bench = Benchmark("reciprocal")
reciprocal_bench.bench_params(dtype=f16_f32_bf)
reciprocal_bench.provider_ops(gem=reciprocal, torch=torch.reciprocal)
reciprocal_bench.arg_names("N")
reciprocal_bench.arg_vals(sizes)
reciprocal_bench.extra_args(M=1024)


pow_bench = Benchmark("pow")
pow_bench.bench_params(dtype=f16_f32_bf)
pow_bench.provider_ops(gem=pow, torch=torch.pow)
pow_bench.arg_names("N")
pow_bench.arg_vals(sizes)
pow_bench.extra_args(M=1024)


@pow_bench.perf
def bench_pow_tensor_scalar(op, M, N, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    exponent = random.randint(-10, 10)
    ms = run_bench(op, inp, exponent)
    return ms


@pow_bench.perf
def bench_pow_tensor_tensor(op, M, N, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    exponent = torch.randint(-10, 10, (M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp, exponent)
    return ms


@reciprocal_bench.perf
def bench_reciprocal(op, M, N, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp)
    return ms


relu_bench = Benchmark("relu")
relu_bench.bench_params(dtype=f16_f32_bf)
relu_bench.provider_ops(gem=relu, torch=torch.relu)
relu_bench.arg_names("N")
relu_bench.arg_vals(sizes)
relu_bench.extra_args(M=1024)


@relu_bench.perf
def bench_relu(op, M, N, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp)
    return ms


rsqrt_bench = Benchmark("rsqrt")
rsqrt_bench.bench_params(dtype=f16_f32_bf)
rsqrt_bench.provider_ops(gem=rsqrt, torch=torch.rsqrt)
rsqrt_bench.arg_names("N")
rsqrt_bench.arg_vals(sizes)
rsqrt_bench.extra_args(M=1024)


@rsqrt_bench.perf
def bench_rsqrt(op, M, N, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp)
    return ms


silu_bench = Benchmark("silu")
silu_bench.bench_params(dtype=f16_f32_bf)
silu_bench.provider_ops(gem=silu, torch=torch.nn.functional.silu)
silu_bench.arg_names("N")
silu_bench.arg_vals(sizes)
silu_bench.extra_args(M=1024)


@silu_bench.perf
def bench_silu(op, M, N, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp)
    return ms


softmax_bench = Benchmark("softmax")
softmax_bench.bench_params(dtype=f16_f32_bf)
softmax_bench.provider_ops(gem=softmax, torch=torch.nn.functional.softmax)
softmax_bench.arg_names("N")
softmax_bench.arg_vals(sizes)
softmax_bench.extra_args(M=1024, dim=1)


@softmax_bench.perf
def bench_softmax(op, M, N, dim, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp, dim=dim)
    return ms


sub_bench = Benchmark("sub")
sub_bench.bench_params(dtype=f16_f32_bf)
sub_bench.provider_ops(gem=sub, torch=torch.sub)
sub_bench.arg_names("N")
sub_bench.arg_vals(sizes)
sub_bench.extra_args(M=1024)


@sub_bench.perf
def bench_sub(op, M, N, dtype):
    inp1 = torch.randn((M, N), dtype=dtype, device="cuda")
    inp2 = torch.randn((M, N), dtype=dtype, device="cuda")
    alpha = random.randint(0, 10000)
    ms = run_bench(op, inp1, inp2, alpha=alpha)
    return ms


@sub_bench.perf
def bench_sub_scalar(op, M, N, dtype):
    inp1 = torch.randn((M, N), dtype=dtype, device="cuda")
    inp2 = random.randint(0, 10000)
    alpha = random.randint(0, 10000)
    ms = run_bench(op, inp1, inp2, alpha=alpha)
    return ms


triu_bench = Benchmark("triu")
triu_bench.bench_params(dtype=f16_f32_bf)
triu_bench.provider_ops(gem=triu, torch=torch.triu)
triu_bench.arg_names("N")
triu_bench.arg_vals(sizes)
triu_bench.extra_args(M=1024, diagonal=1)


@triu_bench.perf
def bench_triu(op, M, N, diagonal, dtype):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")
    ms = run_bench(op, inp, diagonal=diagonal)
    return ms


bench_abs.run(print_data=True)
bench_add.run(print_data=True)
bench_add_scalar.run(print_data=True)
bench_addmm.run(print_data=True)
bench_bmm.run(print_data=True)
bench_cumsum.run(print_data=True)
bench_exp.run(print_data=True)
bench_dropout.run(print_data=True)
bench_div.run(print_data=True)
bench_div_scalar.run(print_data=True)
bench_gelu.run(print_data=True)
bench_layernorm.run(print_data=True)
bench_mean.run(print_data=True)
bench_mm.run(print_data=True)
bench_mul.run(print_data=True)
bench_mul_scalar.run(print_data=True)
bench_pow_tensor_scalar.run(print_data=True)
bench_pow_tensor_tensor.run(print_data=True)
bench_reciprocal.run(print_data=True)
bench_relu.run(print_data=True)
bench_rsqrt.run(print_data=True)
bench_silu.run(print_data=True)
bench_softmax.run(print_data=True)
bench_sub.run(print_data=True)
bench_sub_scalar.run(print_data=True)
bench_triu.run(print_data=True)
