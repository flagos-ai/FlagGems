import torch
import pytest
import flag_gems


RESOLUTION = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}


def allclose_with_dtype(a, b, dtype, equal_nan=False, reduce_dim=1):
    b = b.to(dtype)
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_abs(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.abs(inp.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.abs(inp)

    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_add(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.add(inp1.to(torch.float64), inp2.to(torch.float64), alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    allclose_with_dtype(res_out, ref_out, dtype)


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

    ref_out = torch.add(inp1.to(torch.float64), inp2.to(torch.float64), alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    allclose_with_dtype(res_out, ref_out, dtype)


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

    ref_out = torch.add(inp1.to(torch.float64), inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    allclose_with_dtype(res_out, ref_out, dtype)


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

    ref_out = torch.add(inp1, inp2.to(torch.float64), alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    allclose_with_dtype(res_out, ref_out, dtype)


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

    ref_out = torch.addmm(
        bias.to(torch.float64),
        mat1.to(torch.float64),
        mat2.to(torch.float64),
        alpha=alpha,
        beta=beta,
    )
    with flag_gems.use_gems():
        res_out = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    allclose_with_dtype(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
def test_accuracy_bitwisenot(shape, dtype):
    inp = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
    )

    ref_out = torch.bitwise_not(inp)
    with flag_gems.use_gems():
        res_out = torch.bitwise_not(inp)

    torch.testing.assert_close(res_out, ref_out, atol=0, rtol=0)


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

    ref_out = torch.bmm(tensor_A.to(torch.float64), tensor_B.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.bmm(tensor_A, tensor_B)

    allclose_with_dtype(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_cos(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_inp = inp.to(torch.float64)
    ref_out = torch.cos(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cos(inp)

    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_cumsum(shape, dtype):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.cumsum(inp.to(torch.float64), dim=dim)
    with flag_gems.use_gems():
        res_out = torch.cumsum(inp, dim=dim)

    allclose_with_dtype(res_out, ref_out, dtype, reduce_dim=shape[dim])


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_div(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.div(inp1.to(torch.float64), inp2.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    allclose_with_dtype(res_out, ref_out, dtype, equal_nan=True)


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

    ref_out = torch.div(inp1.to(torch.float64), inp2.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    allclose_with_dtype(res_out, ref_out, dtype)


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

    ref_out = torch.div(inp1.to(torch.float64), inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    allclose_with_dtype(res_out, ref_out, dtype)


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

    ref_out = torch.div(inp1, inp2.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    allclose_with_dtype(res_out, ref_out, dtype)


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

    ref_out = torch.exp(inp.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.exp(inp)

    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_gelu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.nn.functional.gelu(inp.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.nn.functional.gelu(inp)

    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_isinf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    inp = torch.masked_fill(inp, inp > 1.0, -float("inf"))

    ref_out = torch.isinf(inp.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.isinf(inp)

    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_isnan(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    inp = torch.masked_fill(inp, inp > 1.0, float("nan"))

    ref_out = torch.isnan(inp.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.isnan(inp)

    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_layernorm(shape, dtype):
    M = shape[0]
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    weight = torch.randn(layer_shape, dtype=dtype, device="cuda", requires_grad=True)
    bias = torch.randn(layer_shape, dtype=dtype, device="cuda", requires_grad=True)
    eps = 1e-5

    ref_inp = inp.to(torch.float64)
    ref_weight = weight.to(torch.float64)
    ref_bias = bias.to(torch.float64)

    ref_out = torch.layer_norm(
        ref_inp,
        list(layer_shape),
        weight=ref_weight,
        bias=ref_bias,
        eps=eps,
    )
    (res_out, res_mean, res_rstd) = flag_gems.layer_norm(
        inp, list(layer_shape), weight=weight, bias=bias, eps=eps
    )

    ref_mean = torch.mean(ref_inp, dim=1)
    ref_var = torch.var(ref_inp, dim=1, correction=0)
    ref_rstd = torch.rsqrt(ref_var + eps)
    allclose_with_dtype(res_mean, ref_mean, dtype)
    allclose_with_dtype(res_rstd, ref_rstd, dtype)
    allclose_with_dtype(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight, ref_bias), out_grad.to(torch.float64)
    )
    (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
        res_out, (inp, weight, bias), out_grad
    )
    allclose_with_dtype(res_in_grad, ref_in_grad, dtype, reduce_dim=N)
    allclose_with_dtype(res_weight_grad, ref_weight_grad, dtype, reduce_dim=M)
    allclose_with_dtype(res_bias_grad, ref_bias_grad, dtype, reduce_dim=M)


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_mean(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.mean(inp.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.mean(inp)

    allclose_with_dtype(res_out, ref_out, dtype)


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

    ref_out = torch.mm(tensor_a.to(torch.float64), tensor_b.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.mm(tensor_a, tensor_b)

    allclose_with_dtype(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_mul(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.mul(inp1.to(torch.float64), inp2.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    allclose_with_dtype(res_out, ref_out, dtype)


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

    ref_out = torch.mul(inp1.to(torch.float64), inp2.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    allclose_with_dtype(res_out, ref_out, dtype)


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

    ref_out = torch.mul(inp1.to(torch.float64), inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    allclose_with_dtype(res_out, ref_out, dtype)


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

    ref_out = torch.mul(inp1, inp2.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_neg(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.neg(inp.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.neg(inp)

    allclose_with_dtype(res_out, ref_out, dtype)


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
    ref_out = torch.pow(inp, exponent.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.pow(inp, exponent)

    allclose_with_dtype(res_out, ref_out, dtype, equal_nan=True)


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

    ref_out = torch.pow(inp.to(torch.float64), exponent)
    with flag_gems.use_gems():
        res_out = torch.pow(inp, exponent)

    allclose_with_dtype(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_pow_tensor_tensor(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    exponent = torch.randint(-10, 10, shape, dtype=dtype, device="cuda")

    ref_out = torch.pow(inp.to(torch.float64), exponent.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.pow(inp, exponent)

    allclose_with_dtype(res_out, ref_out, dtype, equal_nan=True)


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

    ref_out = torch.pow(inp.to(torch.float64), exponent.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.pow(inp, exponent)

    allclose_with_dtype(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_reciprocal(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.reciprocal(inp.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.reciprocal(inp)

    allclose_with_dtype(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_relu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)

    ref_inp = inp.to(torch.float64)
    ref_out = torch.nn.functional.relu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.relu(inp)

    allclose_with_dtype(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, out_grad.to(torch.float64))
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    allclose_with_dtype(res_in_grad, ref_in_grad, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_rsqrt(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.rsqrt(inp.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.rsqrt(inp)

    allclose_with_dtype(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_rsub(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.rsub(inp1.to(torch.float64), inp2.to(torch.float64), alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.rsub(inp1, inp2, alpha=alpha)

    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_sigmoid(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)

    ref_inp = inp.to(torch.float64)
    ref_out = torch.sigmoid(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sigmoid(inp)

    allclose_with_dtype(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, out_grad.to(torch.float64))
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    allclose_with_dtype(res_in_grad, ref_in_grad, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_silu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)

    ref_inp = inp.to(torch.float64)
    ref_out = torch.nn.functional.silu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.silu(inp)

    allclose_with_dtype(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, out_grad.to(torch.float64))
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    allclose_with_dtype(res_in_grad, ref_in_grad, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_sin(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_inp = inp.to(torch.float64)
    ref_out = torch.sin(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sin(inp)

    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_sub(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.sub(inp1.to(torch.float64), inp2.to(torch.float64), alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    allclose_with_dtype(res_out, ref_out, dtype)


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

    ref_out = torch.sub(inp1.to(torch.float64), inp2.to(torch.float64), alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    allclose_with_dtype(res_out, ref_out, dtype)


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

    ref_out = torch.sub(inp1.to(torch.float64), inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    allclose_with_dtype(res_out, ref_out, dtype)


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

    ref_out = torch.sub(inp1, inp2.to(torch.float64), alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_softmax(shape, dtype):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)

    ref_inp = inp.to(torch.float64)
    ref_out = torch.nn.functional.softmax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.softmax(inp, dim=dim)

    allclose_with_dtype(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, out_grad.to(torch.float64))
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    allclose_with_dtype(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_tanh(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)

    ref_inp = inp.to(torch.float64)
    ref_out = torch.tanh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.tanh(inp)

    allclose_with_dtype(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, out_grad.to(torch.float64))
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    allclose_with_dtype(res_in_grad, ref_in_grad, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("diagonal", [-3, -1, 0, 1, 3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_triu(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_out = torch.triu(inp.to(torch.float64), diagonal)
    with flag_gems.use_gems():
        res_out = torch.triu(inp, diagonal)

    allclose_with_dtype(res_out, ref_out, dtype)
