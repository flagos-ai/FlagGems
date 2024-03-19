import torch
import pytest
from flag_gems import *


@pytest.mark.parametrize(
    "M, N, K",
    [
        (256, 256, 256),
        (1024, 1024, 1024),
        (1024, 128, 2048),
        (1024, 64, 1280),
        (640, 256, 512),
    ],
)
@pytest.mark.parametrize("alpha", [1.0, 0.5])
@pytest.mark.parametrize("beta", [1.0, 0.5])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_addmm(M, N, K, alpha, beta, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device="cuda")
    mat2 = torch.randn((K, N), dtype=dtype, device="cuda")
    bias = torch.randn((N,), dtype=dtype, device="cuda")

    golden_out = torch.addmm(
        bias.to(torch.float64),
        mat1.to(torch.float64),
        mat2.to(torch.float64),
        alpha=alpha,
        beta=beta,
    )

    ref_out = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
    res_out = addmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    diff_torch = torch.sum(torch.abs(golden_out - ref_out))
    diff_triton = torch.sum(torch.abs(golden_out - res_out))
    assert (
        diff_triton < diff_torch * 1.05
    ), f"Torch diff: {diff_torch}, Triton diff: {diff_triton}"


@pytest.mark.parametrize(
    "batch, M, N, K",
    [
        (1, 1024, 1024, 1024),
        (3, 1024, 1024, 2048),
        (4, 1024, 64, 1280),
        (8, 640, 256, 512),
        (16, 1024, 128, 2048),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_bmm(batch, M, N, K, dtype):
    tensor_A = torch.randn((batch, M, K), dtype=dtype, device="cuda")
    tensor_B = torch.randn((batch, K, N), dtype=dtype, device="cuda")

    golden_out = torch.bmm(tensor_A.to(torch.float64), tensor_B.to(torch.float64))

    ref_out = torch.bmm(tensor_A, tensor_B)
    res_out = bmm(tensor_A, tensor_B)

    diff_torch = torch.sum(torch.abs(golden_out - ref_out))
    diff_triton = torch.sum(torch.abs(golden_out - res_out))
    assert (
        diff_triton < diff_torch * 1.05
    ), f"Torch diff: {diff_torch}, Triton diff: {diff_triton}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_cumsum(shape, dtype):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    golden_out = torch.cumsum(inp.to(torch.float64), dim=dim)

    ref_out = torch.cumsum(inp, dim=dim)
    res_out = cumsum(inp, dim=dim)

    diff_torch = torch.sum(torch.abs(golden_out - ref_out))
    diff_triton = torch.sum(torch.abs(golden_out - res_out))
    assert (
        diff_triton < diff_torch * 1.05
    ), f"Torch diff: {diff_torch}, Triton diff: {diff_triton}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("p", [0.3, 0.6, 0.9])
def test_accuracy_dropout(shape, dtype, p):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.nn.functional.dropout(inp, p, True)
    res_out = dropout(inp, p=p, train=True)

    num_equal = torch.sum(torch.isclose(ref_out, res_out)).item()
    exp_equal = (p * p + (1 - p) * (1 - p)) * inp.numel()
    assert (
        abs(num_equal - exp_equal) / exp_equal <= 0.05
    ), f"num_equal: {num_equal}, exp_equal: {exp_equal}, num_total: {inp.numel()}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_gelu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.nn.functional.gelu(inp)
    res_out = gelu(inp)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 32) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_layernorm(shape, dtype):
    layer_shape = shape[1:]
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.randn(layer_shape, dtype=dtype, device="cuda")
    bias = torch.randn(layer_shape, dtype=dtype, device="cuda")
    eps = 1e-5

    golden_out = torch.layer_norm(
        inp.to(torch.float64),
        list(layer_shape),
        weight=weight.to(torch.float64),
        bias=bias.to(torch.float64),
        eps=eps,
    )

    ref_out = torch.layer_norm(
        inp, list(layer_shape), weight=weight, bias=bias, eps=eps
    )
    res_out = layer_norm(inp, list(layer_shape), weight=weight, bias=bias, eps=eps)

    diff_torch = torch.sum(torch.abs(golden_out - ref_out))
    diff_triton = torch.sum(torch.abs(golden_out - res_out))
    assert (
        diff_triton < diff_torch * 1.05
    ), f"Torch diff: {diff_torch}, Triton diff: {diff_triton}"


@pytest.mark.parametrize(
    "shape",
    [
        (256, 256, 256),
        (1024, 1024, 1024),
        (1024, 128, 2048),
        (1024, 64, 1280),
        (640, 256, 512),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_mm(shape, dtype):
    M, N, K = shape
    tensor_a = torch.randn((M, K), dtype=dtype, device="cuda")
    tensor_b = torch.randn((K, N), dtype=dtype, device="cuda")

    golden_out = torch.mm(tensor_a.to(torch.float64), tensor_b.to(torch.float64))
    ref_out = torch.mm(tensor_a, tensor_b)
    res_out = mm(tensor_a, tensor_b)

    diff_torch = torch.sum(torch.abs(golden_out - ref_out))
    diff_triton = torch.sum(torch.abs(golden_out - res_out))
    assert (
        diff_triton < diff_torch * 1.05
    ), f"Torch diff: {diff_torch}, Triton diff: {diff_triton}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_relu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.nn.functional.relu(inp)
    res_out = relu(inp)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_silu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.nn.functional.silu(inp)
    res_out = silu(inp)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_softmax(shape, dtype):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.nn.functional.softmax(inp, dim=dim)
    res_out = softmax(inp, dim=dim)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (32, 128, 512, 512), (20, 320, 15)],
)
@pytest.mark.parametrize("diagonal", [-3, -1, 0, 1, 3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_triu(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_out = torch.triu(inp, diagonal)

    res_out = triu(inp, diagonal)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"
