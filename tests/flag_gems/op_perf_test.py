import torch
import time
import triton
from flag_gems import *

configs_addmm = [
    triton.testing.Benchmark(
        x_names=["MNK"],
        x_vals=[i * 64 for i in range(1, 20)],
        line_arg="provider",
        line_vals=["flag_gems", "torch"],
        line_names=["flag_gems", "torch"],
        styles=[("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"test_performance_addmm_{dtype}",
        args={"alpha": 1.0, "beta": 1.0, "dtype": dtype},
    )
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
]


@triton.testing.perf_report(configs_addmm)
def test_performance_addmm(MNK, alpha, beta, dtype, provider):
    M = N = K = MNK
    mat1 = torch.randn((M, K), dtype=dtype, device="cuda")
    mat2 = torch.randn((K, N), dtype=dtype, device="cuda")
    bias = torch.randn((N,), dtype=dtype, device="cuda")

    if provider == "torch":
        for i in range(5):
            ref_out = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
        start = time.time()
        for i in range(1000):
            ref_out = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000
    if provider == "flag_gems":
        for i in range(5):
            res_out = addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
        start = time.time()
        for i in range(1000):
            res_out = addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000

    return ms


configs_bmm = [
    triton.testing.Benchmark(
        x_names=["MNK"],
        x_vals=[i * 64 for i in range(1, 20)],
        line_arg="provider",
        line_vals=["flag_gems", "torch"],
        line_names=["flag_gems", "torch"],
        styles=[("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"test_performance_bmm_{dtype}",
        args={"batch": 4, "dtype": dtype},
    )
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
]


@triton.testing.perf_report(configs_bmm)
def test_performance_bmm(batch, MNK, dtype, provider):
    M = N = K = MNK
    tensor_A = torch.randn((batch, M, K), dtype=dtype, device="cuda")
    tensor_B = torch.randn((batch, K, N), dtype=dtype, device="cuda")

    if provider == "torch":
        for i in range(5):
            ref_out = torch.bmm(tensor_A, tensor_B)
        start = time.time()
        for i in range(1000):
            ref_out = torch.bmm(tensor_A, tensor_B)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000
    if provider == "flag_gems":
        for i in range(5):
            res_out = bmm(tensor_A, tensor_B)
        start = time.time()
        for i in range(1000):
            res_out = bmm(tensor_A, tensor_B)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000

    return ms


configs_cumsum = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[i * 64 for i in range(1, 20)],
        line_arg="provider",
        line_vals=["flag_gems", "torch"],
        line_names=["flag_gems", "torch"],
        styles=[("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"test_performance_cumsum_{dtype}",
        args={"M": 1024, "dim": 1, "dtype": dtype},
    )
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
]


@triton.testing.perf_report(configs_cumsum)
def test_performance_cumsum(M, N, dim, dtype, provider):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")

    if provider == "torch":
        for i in range(5):
            ref_out = torch.cumsum(inp, dim=dim)
        start = time.time()
        for i in range(1000):
            ref_out = torch.cumsum(inp, dim=dim)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000
    if provider == "flag_gems":
        for i in range(5):
            res_out = cumsum(inp, dim=dim)
        start = time.time()
        for i in range(1000):
            res_out = cumsum(inp, dim=dim)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000

    return ms


configs_dropout = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[i * 64 for i in range(1, 20)],
        line_arg="provider",
        line_vals=["flag_gems", "torch"],
        line_names=["flag_gems", "torch"],
        styles=[("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"test_performance_dropout_{dtype}",
        args={"M": 1024, "p": p, "dtype": dtype},
    )
    for p in [0.3, 0.6, 0.9]
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
]


@triton.testing.perf_report(configs_dropout)
def test_performance_dropout(M, N, p, dtype, provider):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")

    if provider == "torch":
        for i in range(5):
            ref_out = torch.nn.functional.dropout(inp, p, True)
        start = time.time()
        for i in range(1000):
            ref_out = torch.nn.functional.dropout(inp, p, True)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000
    if provider == "flag_gems":
        for i in range(5):
            res_out = dropout(inp, p=p, train=True)
        start = time.time()
        for i in range(1000):
            res_out = dropout(inp, p=p, train=True)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000

    return ms


configs_gelu = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[i * 64 for i in range(1, 20)],
        line_arg="provider",
        line_vals=["flag_gems", "torch"],
        line_names=["flag_gems", "torch"],
        styles=[("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"test_performance_gelu_{dtype}",
        args={"M": 1024, "dtype": dtype},
    )
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
]


@triton.testing.perf_report(configs_gelu)
def test_performance_gelu(M, N, dtype, provider):
    inp = torch.randn((M, N), dtype=dtype, device="cuda")

    if provider == "torch":
        for i in range(5):
            ref_out = torch.nn.functional.gelu(inp)
        start = time.time()
        for i in range(1000):
            ref_out = torch.nn.functional.gelu(inp)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000
    if provider == "flag_gems":
        for i in range(5):
            res_out = gelu(inp)
        start = time.time()
        for i in range(1000):
            res_out = gelu(inp)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000

    return ms


configs_layernorm = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[i * 64 for i in range(1, 20)],
        line_arg="provider",
        line_vals=["flag_gems", "torch"],
        line_names=["flag_gems", "torch"],
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
    if provider == "flag_gems":
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
        line_vals=["flag_gems", "torch"],
        line_names=["flag_gems", "torch"],
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
    if provider == "flag_gems":
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
        line_vals=["flag_gems", "torch"],
        line_names=["flag_gems", "torch"],
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
    if provider == "flag_gems":
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
        line_vals=["flag_gems", "torch"],
        line_names=["flag_gems", "torch"],
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
    if provider == "flag_gems":
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
        line_vals=["flag_gems", "torch"],
        line_names=["flag_gems", "torch"],
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
    if provider == "flag_gems":
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
        line_vals=["flag_gems", "torch"],
        line_names=["flag_gems", "torch"],
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
    if provider == "flag_gems":
        for i in range(5):
            res_out = triu(inp, diagonal)
        start = time.time()
        for i in range(1000):
            res_out = triu(inp, diagonal)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000

    return ms


test_performance_addmm.run(print_data=True)
test_performance_bmm.run(print_data=True)
test_performance_cumsum.run(print_data=True)
test_performance_dropout.run(print_data=True)
test_performance_gelu.run(print_data=True)
test_performance_layernorm.run(print_data=True)
test_performance_mm.run(print_data=True)
test_performance_relu.run(print_data=True)
test_performance_silu.run(print_data=True)
test_performance_softmax.run(print_data=True)
test_performance_triu.run(print_data=True)
