import sys

sys.path.append("../../")
from src.flag_gems import *
import torch
import pytest
import time


@pytest.mark.parametrize(
    "M, N, K",
    [(i * 64, i * 64, i * 64) for i in range(1, 20)],
)
def test_performance_addmm(M, N, K, alpha=1.0, beta=1.0):
    mat1 = torch.randn((M, K), dtype=torch.float16, device="cuda")
    mat2 = torch.randn((K, N), dtype=torch.float16, device="cuda")
    bias = torch.randn((N,), dtype=torch.float16, device="cuda")

    for i in range(5):
        ref_out = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
    start = time.time()
    for i in range(1000):
        ref_out = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
    torch.cuda.synchronize()
    end = time.time()
    torch_time = (end - start) * 1000

    for i in range(5):
        res_out = addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
    start = time.time()
    for i in range(1000):
        res_out = addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
    torch.cuda.synchronize()
    end = time.time()
    triton_time = (end - start) * 1000

    assert (
        False
    ), f"Shape: [{M}, {N}, {K}] | Torch: {torch_time}ms | Triton: {triton_time}ms"


@pytest.mark.parametrize(
    "batch, M, N, K",
    [(4, i * 64, i * 64, i * 64) for i in range(1, 20)],
)
def test_performance_bmm(batch, M, N, K):
    tensor_A = torch.randn((batch, M, K), dtype=torch.float16, device="cuda")
    tensor_B = torch.randn((batch, K, N), dtype=torch.float16, device="cuda")

    for i in range(5):
        ref_out = torch.bmm(tensor_A, tensor_B)
    start = time.time()
    for i in range(1000):
        ref_out = torch.bmm(tensor_A, tensor_B)
    torch.cuda.synchronize()
    end = time.time()
    torch_time = (end - start) * 1000

    for i in range(5):
        res_out = bmm(tensor_A, tensor_B)
    start = time.time()
    for i in range(1000):
        res_out = bmm(tensor_A, tensor_B)
    torch.cuda.synchronize()
    end = time.time()
    triton_time = (end - start) * 1000

    assert (
        False
    ), f"Shape: [{batch}, {M}, {N}, {K}] | Torch: {torch_time}ms | Triton: {triton_time}ms"


@pytest.mark.parametrize(
    "shape",
    [(1024, i * 64) for i in range(1, 20)],
)
def test_performance_cumsum(shape):
    dim = 1
    inp = torch.randn(shape, dtype=torch.float32, device="cuda")

    for i in range(5):
        ref_out = torch.cumsum(inp, dim=dim)
    start = time.time()
    for i in range(1000):
        ref_out = torch.cumsum(inp, dim=dim)
    torch.cuda.synchronize()
    end = time.time()
    torch_time = (end - start) * 1000

    for i in range(5):
        res_out = cumsum(inp, dim=dim)
    start = time.time()
    for i in range(1000):
        res_out = cumsum(inp, dim=dim)
    torch.cuda.synchronize()
    end = time.time()
    triton_time = (end - start) * 1000

    assert False, f"Shape: shape | Torch: {torch_time}ms | Triton: {triton_time}ms"


@pytest.mark.parametrize(
    "shape",
    [(1024, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_performance_gelu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    for i in range(5):
        ref_out = torch.nn.functional.gelu(inp)
    start = time.time()
    for i in range(1000):
        ref_out = torch.nn.functional.gelu(inp)
    torch.cuda.synchronize()
    end = time.time()
    torch_time = (end - start) * 1000

    for i in range(5):
        res_out = gelu(inp)
    start = time.time()
    for i in range(1000):
        res_out = gelu(inp)
    torch.cuda.synchronize()
    end = time.time()
    triton_time = (end - start) * 1000

    assert False, f"Shape: shape | Torch: {torch_time}ms | Triton: {triton_time}ms"


@pytest.mark.parametrize(
    "shape",
    [(1024, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_performance_layernorm(shape, dtype):
    layer_shape = shape[1:]
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.randn(layer_shape, dtype=dtype, device="cuda")
    bias = torch.randn(layer_shape, dtype=dtype, device="cuda")
    eps = 1e-5

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
    torch_time = (end - start) * 1000

    for i in range(5):
        res_out = layer_norm(inp, list(layer_shape), weight=weight, bias=bias, eps=eps)
    start = time.time()
    for i in range(1000):
        res_out = layer_norm(inp, list(layer_shape), weight=weight, bias=bias, eps=eps)
    torch.cuda.synchronize()
    end = time.time()
    triton_time = (end - start) * 1000

    assert False, f"Shape: shape | Torch: {torch_time}ms | Triton: {triton_time}ms"


@pytest.mark.parametrize(
    "shape",
    [(i * 64, i * 64, i * 64) for i in range(1, 20)],
)
def test_performance_mm(shape):
    M, N, K = shape
    tensor_a = torch.randn((M, K), dtype=torch.float16, device="cuda")
    tensor_b = torch.randn((K, N), dtype=torch.float16, device="cuda")

    for i in range(5):
        ref_out = torch.mm(tensor_a, tensor_b)
    start = time.time()
    for i in range(1000):
        ref_out = torch.mm(tensor_a, tensor_b)
    torch.cuda.synchronize()
    end = time.time()
    torch_time = (end - start) * 1000

    for i in range(5):
        res_out = mm(tensor_a, tensor_b)
    for i in range(1000):
        res_out = mm(tensor_a, tensor_b)
    torch.cuda.synchronize()
    end = time.time()
    triton_time = (end - start) * 1000

    assert False, f"Shape: shape | Torch: {torch_time}ms | Triton: {triton_time}ms"


@pytest.mark.parametrize(
    "shape",
    [(1024, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_performance_relu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    for i in range(5):
        ref_out = torch.nn.functional.relu(inp)
    start = time.time()
    for i in range(1000):
        ref_out = torch.nn.functional.relu(inp)
    torch.cuda.synchronize()
    end = time.time()
    torch_time = (end - start) * 1000

    for i in range(5):
        res_out = relu(inp)
    start = time.time()
    for i in range(1000):
        res_out = relu(inp)
    torch.cuda.synchronize()
    end = time.time()
    triton_time = (end - start) * 1000

    assert False, f"Shape: shape | Torch: {torch_time}ms | Triton: {triton_time}ms"


@pytest.mark.parametrize(
    "shape",
    [(1024, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_performance_silu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    for i in range(5):
        ref_out = torch.nn.functional.silu(inp)
    start = time.time()
    for i in range(1000):
        ref_out = torch.nn.functional.silu(inp)
    torch.cuda.synchronize()
    end = time.time()
    torch_time = (end - start) * 1000

    for i in range(5):
        res_out = silu(inp)
    start = time.time()
    for i in range(1000):
        res_out = silu(inp)
    torch.cuda.synchronize()
    end = time.time()
    triton_time = (end - start) * 1000

    assert False, f"Shape: shape | Torch: {torch_time}ms | Triton: {triton_time}ms"


@pytest.mark.parametrize(
    "shape",
    [(1024, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_performance_softmax(shape, dtype):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    for i in range(5):
        ref_out = torch.nn.functional.softmax(inp, dim=dim)
    start = time.time()
    for i in range(1000):
        ref_out = torch.nn.functional.softmax(inp, dim=dim)
    torch.cuda.synchronize()
    end = time.time()
    torch_time = (end - start) * 1000

    for i in range(5):
        res_out = softmax(inp, dim=dim)
    start = time.time()
    for i in range(1000):
        res_out = softmax(inp, dim=dim)
    torch.cuda.synchronize()
    end = time.time()
    triton_time = (end - start) * 1000

    assert False, f"Shape: shape | Torch: {torch_time}ms | Triton: {triton_time}ms"
