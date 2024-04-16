import torch
import pytest
import flag_gems


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_abs(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.abs(inp)
    with flag_gems.use_gems():
        res_out = torch.abs(inp)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_add(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.add(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape_a",
    [(16, 1024, 256)],
)
@pytest.mark.parametrize(
    "shape_b",
    [(1, 256), (1, 1, 256), (16, 1, 256), (1, 1024, 256), (1024, 256)],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_add_broadcast(shape_a, shape_b, alpha, dtype):
    inp1 = torch.randn(shape_a, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape_b, dtype=dtype, device="cuda")

    ref_out = torch.add(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "scalar",
    [111.111, -999.999],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_add_tensor_scalar(shape, scalar, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = scalar

    ref_out = torch.add(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "scalar",
    [111.111, -999.999],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_add_scalar_tensor(shape, scalar, alpha, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.add(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


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
    with flag_gems.use_gems():
        res_out = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)

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
    with flag_gems.use_gems():
        res_out = torch.bmm(tensor_A, tensor_B)

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
    with flag_gems.use_gems():
        res_out = torch.cumsum(inp, dim=dim)

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
def test_accuracy_div(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.div(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3, equal_nan=True
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape_a",
    [(16, 1024, 256)],
)
@pytest.mark.parametrize(
    "shape_b",
    [(1, 256), (1, 1, 256), (16, 1, 256), (1, 1024, 256), (1024, 256)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_div_broadcast(shape_a, shape_b, dtype):
    inp1 = torch.randn(shape_a, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape_b, dtype=dtype, device="cuda")

    ref_out = torch.div(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "scalar",
    [111.111, -999.999],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_div_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = scalar

    ref_out = torch.div(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "scalar",
    [200, 100],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_div_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randint(-5, 5, shape, dtype=dtype, device="cuda")

    ref_out = torch.div(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("p", [0.3, 0.6, 0.9])
def test_accuracy_dropout(shape, dtype, p):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)

    ref_out = torch.nn.functional.dropout(inp, p, True)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.dropout(inp, p, True)

    num_equal = torch.sum(torch.isclose(ref_out, res_out)).item()
    exp_equal = (p * p + (1 - p) * (1 - p)) * inp.numel()
    assert (
        abs(num_equal - exp_equal) / exp_equal <= 0.05
    ), f"num_equal: {num_equal}, exp_equal: {exp_equal}, num_total: {inp.numel()}"

    out_grad = torch.randn_like(inp)
    (ref_in_grad,) = torch.autograd.grad(ref_out, inp, out_grad)
    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    num_equal = torch.sum(torch.isclose(ref_in_grad, res_in_grad)).item()
    exp_equal = (p * p + (1 - p) * (1 - p)) * inp.numel()
    assert (
        abs(num_equal - exp_equal) / exp_equal <= 0.05
    ), f"num_equal: {num_equal}, exp_equal: {exp_equal}, num_total: {inp.numel()}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_exp(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.exp(inp)
    with flag_gems.use_gems():
        res_out = torch.exp(inp)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_gelu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.nn.functional.gelu(inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.gelu(inp)

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
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    weight = torch.randn(layer_shape, dtype=dtype, device="cuda", requires_grad=True)
    bias = torch.randn(layer_shape, dtype=dtype, device="cuda", requires_grad=True)
    eps = 1e-5

    golden_inp = inp.to(torch.float64)
    golden_weight = weight.to(torch.float64)
    golden_bias = bias.to(torch.float64)

    golden_out = torch.layer_norm(
        golden_inp,
        list(layer_shape),
        weight=golden_weight,
        bias=golden_bias,
        eps=eps,
    )
    ref_out = torch.layer_norm(
        inp, list(layer_shape), weight=weight, bias=bias, eps=eps
    )
    with flag_gems.use_gems():
        res_out = torch.layer_norm(
            inp, list(layer_shape), weight=weight, bias=bias, eps=eps
        )

    diff_torch = torch.sum(torch.abs(golden_out - ref_out))
    diff_triton = torch.sum(torch.abs(golden_out - res_out))
    assert (
        diff_triton < diff_torch * 1.05
    ), f"Torch diff: {diff_torch}, Triton diff: {diff_triton}"

    out_grad = torch.randn_like(inp)

    (golden_in_grad, golden_weight_grad, golden_bias_grad) = torch.autograd.grad(
        golden_out, (golden_inp, golden_weight, golden_bias), out_grad.to(torch.float64)
    )
    (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
        ref_out, (inp, weight, bias), out_grad
    )
    with flag_gems.use_gems():
        (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
            res_out, (inp, weight, bias), out_grad
        )

    diff_torch = torch.sum(torch.abs(golden_in_grad - ref_in_grad))
    diff_triton = torch.sum(torch.abs(golden_in_grad - res_in_grad))
    assert (
        diff_triton < diff_torch * 2
    ), f"Input Grad: Torch diff: {diff_torch}, Triton diff: {diff_triton}"
    diff_torch = torch.sum(torch.abs(golden_weight_grad - ref_weight_grad))
    diff_triton = torch.sum(torch.abs(golden_weight_grad - res_weight_grad))
    assert (
        diff_triton < diff_torch * 2
    ), f"Weight Grad: Torch diff: {diff_torch}, Triton diff: {diff_triton}"
    assert torch.allclose(
        ref_bias_grad, res_bias_grad, atol=1e-3, rtol=1e-3
    ), f"bias grad diff"


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 32) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_mean(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.mean(inp)
    with flag_gems.use_gems():
        res_out = torch.mean(inp)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}, {ref_out}, {res_out}"


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
    with flag_gems.use_gems():
        res_out = torch.mm(tensor_a, tensor_b)

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
def test_accuracy_mul(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.mul(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape_a",
    [(16, 1024, 256)],
)
@pytest.mark.parametrize(
    "shape_b",
    [(1, 256), (1, 1, 256), (16, 1, 256), (1, 1024, 256), (1024, 256)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_mul_broadcast(shape_a, shape_b, dtype):
    inp1 = torch.randn(shape_a, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape_b, dtype=dtype, device="cuda")

    ref_out = torch.mul(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "scalar",
    [111.111, -999.999],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_mul_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = scalar

    ref_out = torch.mul(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "scalar",
    [111.111, -999.999],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_mul_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.mul(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "inp",
    [0.9, 1.0, 100.9, -111.9],
)
@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_pow_scalar_tensor(inp, shape, dtype):
    exponent = torch.randint(-5, 5, shape, dtype=dtype, device="cuda")
    ref_out = torch.pow(inp, exponent)
    with flag_gems.use_gems():
        res_out = torch.pow(inp, exponent)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3, equal_nan=True
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "exponent",
    [0.5, 1.5, 5.0, -1.0],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_pow_tensor_scalar(shape, exponent, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.pow(inp, exponent)
    with flag_gems.use_gems():
        res_out = torch.pow(inp, exponent)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3, equal_nan=True
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_pow_tensor_tensor(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    exponent = torch.randint(-10, 10, shape, dtype=dtype, device="cuda")

    ref_out = torch.pow(inp, exponent)
    with flag_gems.use_gems():
        res_out = torch.pow(inp, exponent)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3, equal_nan=True
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape_a",
    [(16, 1024, 256)],
)
@pytest.mark.parametrize(
    "shape_b",
    [(1, 256), (1, 1, 256), (16, 1, 256), (1, 1024, 256), (1024, 256)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_pow_tensor_tensor_broadcast(shape_a, shape_b, dtype):
    inp = torch.randn(shape_a, dtype=dtype, device="cuda")
    exponent = torch.randint(-10, 10, shape_b, dtype=dtype, device="cuda")

    ref_out = torch.pow(inp, exponent)
    with flag_gems.use_gems():
        res_out = torch.pow(inp, exponent)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3, equal_nan=True
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_reciprocal(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.reciprocal(inp)
    with flag_gems.use_gems():
        res_out = torch.reciprocal(inp)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3, equal_nan=True
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_relu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)

    ref_out = torch.nn.functional.relu(inp)
    with flag_gems.use_gems():
        res_out = torch.relu(inp)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"

    out_grad = torch.randn_like(inp)
    (ref_in_grad,) = torch.autograd.grad(ref_out, inp, out_grad)
    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    maxdiff = torch.max(torch.abs(ref_in_grad - res_in_grad))
    assert torch.allclose(
        ref_in_grad, res_in_grad, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_rsqrt(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.rsqrt(inp)
    with flag_gems.use_gems():
        res_out = torch.rsqrt(inp)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3, equal_nan=True
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_rsub(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.rsub(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.rsub(inp1, inp2, alpha=alpha)

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
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)

    ref_out = torch.nn.functional.silu(inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.silu(inp)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"

    out_grad = torch.randn_like(inp)
    (ref_in_grad,) = torch.autograd.grad(ref_out, inp, out_grad)
    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    maxdiff = torch.max(torch.abs(ref_in_grad - res_in_grad))
    assert torch.allclose(
        ref_in_grad, res_in_grad, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_sub(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.sub(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape_a",
    [(16, 1024, 256)],
)
@pytest.mark.parametrize(
    "shape_b",
    [(1, 256), (1, 1, 256), (16, 1, 256), (1, 1024, 256), (1024, 256)],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_sub_broadcast(shape_a, shape_b, alpha, dtype):
    inp1 = torch.randn(shape_a, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape_b, dtype=dtype, device="cuda")

    ref_out = torch.sub(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "scalar",
    [111.111, -999.999],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_sub_tensor_scalar(shape, scalar, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = scalar

    ref_out = torch.sub(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "scalar",
    [111.111, -999.999],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_sub_scalar_tensor(shape, scalar, alpha, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.sub(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

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
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)

    ref_out = torch.nn.functional.softmax(inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.softmax(inp, dim=dim)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"

    out_grad = torch.randn_like(inp)
    (ref_in_grad,) = torch.autograd.grad(ref_out, inp, out_grad)
    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    maxdiff = torch.max(torch.abs(ref_in_grad - res_in_grad))
    assert torch.allclose(
        ref_in_grad, res_in_grad, atol=1e-3, rtol=1e-3
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
    with flag_gems.use_gems():
        res_out = torch.triu(inp, diagonal)

    maxdiff = torch.max(torch.abs(ref_out - res_out))
    assert torch.allclose(
        ref_out, res_out, atol=1e-3, rtol=1e-3
    ), f"max diff: {maxdiff}"
