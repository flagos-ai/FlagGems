import torch
import pytest
import flag_gems

RESOLUTION = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
    torch.int64: 1.3e-6,
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
def test_accuracy_bitwiseand(shape, dtype):
    inp1 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
    )
    inp2 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
    )

    ref_out = torch.bitwise_and(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_and(inp1, inp2)

    assert torch.equal(res_out, ref_out), f"ref_out: {ref_out}, res_out: {res_out}"


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

    assert torch.equal(res_out, ref_out), f"ref_out: {ref_out}, res_out: {res_out}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
def test_accuracy_bitwiseor(shape, dtype):
    inp1 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
    )
    inp2 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
    )

    ref_out = torch.bitwise_or(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_or(inp1, inp2)

    assert torch.equal(res_out, ref_out), f"ref_out: {ref_out}, res_out: {res_out}"


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
@pytest.mark.parametrize("isnone", [None, "max", "min"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_clamp(shape, isnone, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    maxi = torch.randn(shape, dtype=dtype, device="cuda")
    mini = torch.randn(shape, dtype=dtype, device="cuda")
    if isnone == "min":
        mini = None
    elif isnone == "max":
        maxi = None

    ref_out = torch.clamp(inp, min=mini, max=maxi)
    with flag_gems.use_gems():
        res_out = torch.clamp(inp, min=mini, max=maxi)

    assert torch.equal(res_out, ref_out), f"ref_out: {ref_out}, res_out: {res_out}"


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

    # nz_ref = torch.sum(ref_out == 0.0)
    # nz_res = torch.sum(res_out == 0.0)

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
    "N, C, H, W, num_groups",
    [
        (16, 3, 16, 16, 1),
        (32, 32, 32, 32, 8),
        (1, 32, 32, 32, 8),
        (1, 32, 32, 32, 16),
        (1, 64, 32, 32, 16),
        (1, 64, 32, 32, 32),
        (1, 64, 32, 32, 64),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_groupnorm(N, C, H, W, num_groups, dtype):
    HW = H * W
    inp = torch.randn(size=(N, C, H, W), dtype=dtype, device="cuda", requires_grad=True)
    weight = torch.randn(size=(C,), dtype=dtype, device="cuda", requires_grad=True)
    bias = torch.randn(size=(C,), dtype=dtype, device="cuda", requires_grad=True)
    eps = 1e-5

    ref_inp = inp.to(torch.float64)
    ref_weight = weight.to(torch.float64)
    ref_bias = bias.to(torch.float64)

    ref_out = torch.nn.functional.group_norm(
        ref_inp, num_groups, weight=ref_weight, bias=ref_bias, eps=eps
    )
    ref_mean = torch.mean(ref_inp.reshape([N, num_groups, -1]), dim=2)
    ref_var = torch.var(ref_inp.reshape([N, num_groups, -1]), dim=2, correction=0)
    ref_rstd = torch.rsqrt(ref_var + eps)

    (res_out, res_mean, res_rstd) = flag_gems.group_norm(
        inp, weight, bias, N, C, HW, num_groups, eps
    )

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
    group_size = C // num_groups
    allclose_with_dtype(res_in_grad, ref_in_grad, dtype, reduce_dim=group_size * HW)
    allclose_with_dtype(res_weight_grad, ref_weight_grad, dtype, reduce_dim=N * HW)
    allclose_with_dtype(res_bias_grad, ref_bias_grad, dtype, reduce_dim=N * HW)


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

    assert torch.equal(res_out, ref_out), f"ref_out: {ref_out}, res_out: {res_out}"


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

    assert torch.equal(res_out, ref_out), f"ref_out: {ref_out}, res_out: {res_out}"


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
def test_accuracy_skip_layernorm(shape, dtype):
    M = shape[0]
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=False)
    residual = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=False)
    weight = torch.randn(layer_shape, dtype=dtype, device="cuda", requires_grad=False)
    bias = torch.randn(layer_shape, dtype=dtype, device="cuda", requires_grad=False)
    eps = 1e-5

    ref_inp = inp.to(torch.float64)
    ref_residual = residual.to(torch.float64)
    ref_weight = weight.to(torch.float64)
    ref_bias = bias.to(torch.float64)

    ref_out = torch.layer_norm(
        ref_inp + ref_residual,
        list(layer_shape),
        weight=ref_weight,
        bias=ref_bias,
        eps=eps,
    )
    res_out = flag_gems.skip_layer_norm(
        inp, residual, list(layer_shape), weight=weight, bias=bias, eps=eps
    )

    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_skip_rmsnorm(shape, dtype):
    M = shape[0]
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=False)
    residual = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=False)
    weight = torch.randn(layer_shape, dtype=dtype, device="cuda", requires_grad=False)
    eps = 1e-5

    ref_inp = inp.to(torch.float64)
    ref_residual = residual.to(torch.float64)
    ref_weight = weight.to(torch.float64)

    def _torch_rms_norm(x, residual, weight, eps):
        x = x + residual
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states

    ref_out = _torch_rms_norm(
        ref_inp,
        ref_residual,
        weight=ref_weight,
        eps=eps,
    )

    res_out = flag_gems.skip_rms_norm(
        inp, residual, list(layer_shape), weight=weight, eps=eps
    )

    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_rmsnorm(shape, dtype):
    M = shape[0]
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=False)
    weight = torch.randn(layer_shape, dtype=dtype, device="cuda", requires_grad=False)
    eps = 1e-5

    ref_inp = inp.to(torch.float64)
    ref_weight = weight.to(torch.float64)

    def _torch_rms_norm(x, weight, eps):
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states

    ref_out = _torch_rms_norm(
        ref_inp,
        weight=ref_weight,
        eps=eps,
    )

    res_out = flag_gems.rms_norm(inp, list(layer_shape), weight=weight, eps=eps)

    allclose_with_dtype(res_out, ref_out, dtype)


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
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dim", [-1, 0, 1, None, [1, 0]])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_meandim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_out = torch.mean(inp.to(torch.float64), dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.mean(inp, dim, keepdim)

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


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_max(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.max(inp.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [0, 1])
def test_accuracy_max_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.max(inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.max(inp, dim=dim, keepdim=keepdim)
    ref_out_value, ref_out_index = ref_out
    res_out_value, res_out_index = res_out
    assert torch.equal(
        ref_out_index, res_out_index
    ), f"ref_out_index: {ref_out_index}, res_out_index: {res_out_index}"
    allclose_with_dtype(ref_out_value, res_out_value, dtype)


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_min(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.min(inp.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.min(inp)

    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dim", [-1, 0, 1, None, [1, 0]])
@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_varmean(shape, dim, correction, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_var, ref_mean = torch.var_mean(inp, dim, correction=correction, keepdim=keepdim)
    with flag_gems.use_gems():
        res_var, res_mean = torch.var_mean(
            inp, dim, correction=correction, keepdim=keepdim
        )

    allclose_with_dtype(res_mean, ref_mean, dtype)
    allclose_with_dtype(res_var, ref_var, dtype)


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [0, 1])
def test_accuracy_min_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.min(inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.min(inp, dim=dim, keepdim=keepdim)
    ref_out_value, ref_out_index = ref_out
    res_out_value, res_out_index = res_out
    assert torch.equal(
        ref_out_index, res_out_index
    ), f"ref_out_index: {ref_out_index}, res_out_index: {res_out_index}"

    allclose_with_dtype(ref_out_value, res_out_value, dtype)


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_sum(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.sum(inp.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.sum(inp)

    allclose_with_dtype(res_out, ref_out, dtype, reduce_dim=inp.numel())


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [[0, 1], 0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_sum_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.sum(inp.to(torch.float64), dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.sum(inp, dim=dim, keepdim=keepdim)
    if isinstance(dim, int):
        dim = [dim]
    dim = [d % inp.ndim for d in dim]
    _dim = 1
    for d in dim:
        _dim *= shape[d]
    allclose_with_dtype(res_out, ref_out, dtype, reduce_dim=_dim)


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [[0, 1], [1, 0], 0, 1, None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_amax(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.amax(inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.amax(inp, dim=dim, keepdim=keepdim)

    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [0, 1, None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_argmax(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_out = torch.argmax(inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.argmax(inp, dim=dim, keepdim=keepdim)
    assert torch.equal(ref_out, res_out), f"ref_out: {ref_out}, res_out: {res_out}"


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_prod(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_out = torch.prod(inp.to(torch.float64))
    with flag_gems.use_gems():
        res_out = torch.prod(inp)
    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 64) for i in range(1, 20)],
)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_prod_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.prod(inp.to(torch.float64), dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.prod(inp, dim=dim, keepdim=keepdim)
    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("ord", [2, float("inf"), -float("inf"), 0, 1])
@pytest.mark.parametrize("dim", [-1, 0, 1, None, [1, 0]])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_vectornorm(shape, ord, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_out = torch.linalg.vector_norm(inp.to(torch.float64), ord, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.linalg.vector_norm(inp, ord, dim, keepdim)

    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_log_softmax(shape, dtype):
    dim = 1
    # torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    ref_inp = inp.to(torch.float64)
    ref_out = torch.nn.functional.log_softmax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.log_softmax(inp, dim=dim)
    allclose_with_dtype(res_out, ref_out, dtype)
    out_grad = torch.randn_like(res_out)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, out_grad.to(torch.float64))
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    allclose_with_dtype(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024), (16, 128), (20, 320)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_outer(shape, dtype):
    inp1_shape, inp2_shape = list(shape)
    inp1 = torch.randn(inp1_shape, dtype=dtype, device="cuda", requires_grad=True)
    inp2 = torch.randn(inp2_shape, dtype=dtype, device="cuda", requires_grad=True)

    inp1_f64 = inp1.to(torch.float64)
    inp2_f64 = inp2.to(torch.float64)
    ref_out = torch.outer(inp1_f64, inp2_f64)
    with flag_gems.use_gems():
        res_out = torch.outer(inp1, inp2)
    allclose_with_dtype(res_out, ref_out, dtype)

    out_grad = torch.randn_like(res_out)
    ref_in1_grad, ref_in2_grad = torch.autograd.grad(
        ref_out, (inp1_f64, inp2_f64), out_grad.to(torch.float64)
    )
    res_in1_grad, res_in2_grad = torch.autograd.grad(res_out, (inp1, inp2), out_grad)
    allclose_with_dtype(res_in1_grad, ref_in1_grad, dtype)
    allclose_with_dtype(res_in2_grad, ref_in2_grad, dtype)


def get_rope_cos_sin(max_seq_len, dim, dtype, base=10000, device="cuda"):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


# Copied from transformers.models.llama.modeling_llama.rotate_half
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.cohere.modeling_cohere.rotate_half
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere/modeling_cohere.py
def rotate_interleave(x):
    """Rotates interleave the hidden dims of the input."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def torch_apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids,
    rotary_interleaved: bool = False,
):

    q = q.float()
    k = k.float()
    cos = cos[position_ids].unsqueeze(-2)  # [bs, seq_len, 1, dim/2]
    sin = sin[position_ids].unsqueeze(-2)  # [bs, seq_len, 1, dim/2]
    if rotary_interleaved:
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # [bs, seq_len, 1, dim]
        sin = torch.repeat_interleave(sin, 2, dim=-1)  # [bs, seq_len, 1, dim]
        rotate_fn = rotate_interleave
    else:
        cos = torch.cat([cos, cos], dim=-1)  # [bs, seq_len, 1, dim]
        sin = torch.cat([sin, sin], dim=-1)  # [bs, seq_len, 1, dim]
        rotate_fn = rotate_half

    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)

    return q_embed, k_embed


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("max_seq_len", [512, 2048])
@pytest.mark.parametrize("q_heads,k_heads", [(8, 1), (6, 2), (1, 1), (8, 8)])
@pytest.mark.parametrize("head_dim", [64, 96, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rotary_interleaved", [True, False])
def test_apply_rotary_pos_emb(
    batch_size,
    max_seq_len,
    q_heads,
    k_heads,
    head_dim,
    dtype,
    rotary_interleaved,
):
    seq_len = torch.randint(1, max_seq_len, (1,)).item()
    q = torch.randn(
        (batch_size, seq_len, q_heads, head_dim), dtype=dtype, device="cuda"
    )
    k = torch.randn(
        (batch_size, seq_len, k_heads, head_dim), dtype=dtype, device="cuda"
    )

    position_ids = torch.randint(0, max_seq_len, (batch_size, seq_len), device="cuda")
    cos, sin = get_rope_cos_sin(max_seq_len, head_dim, dtype, device="cuda")

    q_embed_ref, k_embed_ref = torch_apply_rotary_pos_emb(
        q=q,
        k=k,
        cos=cos,
        sin=sin,
        position_ids=position_ids,
        rotary_interleaved=rotary_interleaved,
    )
    q_embed_out, k_embed_out = flag_gems.apply_rotary_pos_emb(
        q=q,
        k=k,
        cos=cos,
        sin=sin,
        position_ids=position_ids,
        rotary_interleaved=rotary_interleaved,
    )

    allclose_with_dtype(q_embed_out, q_embed_ref, dtype)
    allclose_with_dtype(k_embed_out, k_embed_ref, dtype)


@pytest.mark.parametrize(
    "shape",
    [(i, j * 64) for i in [2, 4, 4096] for j in range(1, 10)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16, torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allTrue"])
def test_accuracy_all(shape, dtype, kind):
    if (kind == "allTrue"):
        inp = torch.ones(shape, dtype=dtype, device="cuda")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cuda")

    ref_out = torch.all(inp)
    with flag_gems.use_gems():
        res_out = torch.all(inp)

    assert torch.equal(ref_out, res_out), f"ref_out: {ref_out}, res_out: {res_out}"


@pytest.mark.parametrize(
    "shape",
    [(i, j * 64) for i in [2, 4, 4096] for j in range(1, 10)],
)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [0, 1, -1, None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16, torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allTrue"])
def test_accuracy_all_dim(shape, dim, keepdim, dtype, kind):
    if (kind == "allTrue"):
        inp = torch.ones(shape, dtype=dtype, device="cuda")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cuda")

    ref_out = torch.all(inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.all(inp, dim=dim, keepdim=keepdim)
    assert torch.equal(ref_out, res_out), f"ref_out: {ref_out}, res_out: {res_out}"
    

@pytest.mark.parametrize(
    "shape",
    [(1024, 1024, 16), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15), (2, 3, 5)],
)
@pytest.mark.parametrize("dim", [[1, 0], [1, 2]])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16, torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allTrue"])
def test_accuracy_all_dims(shape, dim, keepdim, dtype, kind):
    if (kind == "allTrue"):
        inp = torch.ones(shape, dtype=dtype, device="cuda")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cuda")

    ref_out = torch.all(inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.all(inp, dim=dim, keepdim=keepdim)
    assert torch.equal(ref_out, res_out), f"ref_out: {ref_out}, res_out: {res_out}"


@pytest.mark.parametrize(
    "shape",
    [(i, j * 64) for i in [2, 4, 4096] for j in range(1, 10)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16, torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allFalse"])
def test_accuracy_any(shape, dtype, kind):
    if (kind == "allFalse"):
        inp = torch.zeros(shape, dtype=dtype, device="cuda")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cuda")

    ref_out = torch.any(inp)
    with flag_gems.use_gems():
        res_out = torch.any(inp)

    assert torch.equal(ref_out, res_out), f"ref_out: {ref_out}, res_out: {res_out}"


@pytest.mark.parametrize(
    "shape",
    [(i, j * 64) for i in [2, 4, 4096] for j in range(1, 10)],
)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [0, 1, -1, None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16, torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allFalse"])
def test_accuracy_any_dim(shape, dim, keepdim, dtype, kind):
    if (kind == "allFalse"):
        inp = torch.zeros(shape, dtype=dtype, device="cuda")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cuda")

    ref_out = torch.any(inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.any(inp, dim=dim, keepdim=keepdim)
    assert torch.equal(ref_out, res_out), f"ref_out: {ref_out}, res_out: {res_out}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024, 16), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15), (2, 3, 5)],
)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [[1, 0], [1, 2]])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16, torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allFalse"])
def test_accuracy_any_dims(shape, dim, keepdim, dtype, kind):
    if (kind == "allFalse"):
        inp = torch.zeros(shape, dtype=dtype, device="cuda")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cuda")

    ref_out = torch.any(inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.any(inp, dim=dim, keepdim=keepdim)
    assert torch.equal(ref_out, res_out), f"ref_out: {ref_out}, res_out: {res_out}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 30)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_silu_and_mul(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    inp1, inp2 = inp.chunk(2, dim=-1)

    ref_out = torch.mul(
        torch.nn.functional.silu(inp1.to(torch.float64)),
        inp2.to(torch.float64),
    )
    with flag_gems.use_gems():
        res_out = flag_gems.silu_and_mul(inp1, inp2)

    allclose_with_dtype(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 30)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_accuracy_gelu_and_mul(shape, approximate, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    inp1, inp2 = inp.chunk(2, dim=-1)

    ref_out = torch.mul(
        torch.nn.functional.gelu(inp1.to(torch.float64), approximate=approximate),
        inp2.to(torch.float64),
    )
    with flag_gems.use_gems():
        res_out = flag_gems.gelu_and_mul(inp1, inp2, approximate)

    allclose_with_dtype(res_out, ref_out, dtype)

