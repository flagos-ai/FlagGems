import pytest
import torch

import flag_gems

from .accuracy_utils import (
    DIM_LIST,
    DIMS_LIST,
    FLOAT_DTYPES,
    REDUCTION_SHAPES,
    gems_assert_close,
    gems_assert_equal,
    skip_expr,
    skip_reason,
    to_reference,
)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allTrue"])
def test_accuracy_all(shape, dtype, kind):
    if kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device="cuda")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.all(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.all(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(skip_expr, reason=skip_reason)
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allTrue"])
def test_accuracy_all_dim(shape, dim, keepdim, dtype, kind):
    if kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device="cuda")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.all(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.all(inp, dim=dim, keepdim=keepdim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(skip_expr, reason=skip_reason)
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allTrue"])
def test_accuracy_all_dims(shape, dim, keepdim, dtype, kind):
    if kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device="cuda")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.all(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.all(inp, dim=dim, keepdim=keepdim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_amax(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.amax(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.amax(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allFalse"])
def test_accuracy_any(shape, dtype, kind):
    if kind == "allFalse":
        inp = torch.zeros(shape, dtype=dtype, device="cuda")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.any(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.any(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(skip_expr, reason=skip_reason)
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allFalse"])
def test_accuracy_any_dim(shape, dim, keepdim, dtype, kind):
    if kind == "allFalse":
        inp = torch.zeros(shape, dtype=dtype, device="cuda")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.any(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.any(inp, dim=dim, keepdim=keepdim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(skip_expr, reason=skip_reason)
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allFalse"])
def test_accuracy_any_dims(shape, dim, keepdim, dtype, kind):
    if kind == "allFalse":
        inp = torch.zeros(shape, dtype=dtype, device="cuda")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.any(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.any(inp, dim=dim, keepdim=keepdim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_argmax(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.argmax(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.argmax(inp, dim=dim, keepdim=keepdim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("size_average", [None, True, False])
@pytest.mark.parametrize("reduce", [None, True, False])
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("ignore_index", [1, 200, -100])
def test_accuracy_cross_entropy_loss(
    shape, dtype, size_average, reduce, ignore_index, reduction
):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    dim = 1
    up_limit = shape[dim] - 1
    target_shape = list(shape)
    del target_shape[dim]
    target = torch.randint(0, up_limit, target_shape, device="cuda")
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target)
    criterion = torch.nn.CrossEntropyLoss(
        size_average=size_average,
        reduce=reduce,
        ignore_index=ignore_index,
        reduction=reduction,
    )

    ref_out = criterion(ref_inp, ref_target)
    with flag_gems.use_gems():
        res_out = criterion(inp, target)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cumsum(shape, dtype):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.cumsum(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.cumsum(inp, dim=dim)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])


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
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_groupnorm(N, C, H, W, num_groups, dtype):
    HW = H * W
    inp = torch.randn(size=(N, C, H, W), dtype=dtype, device="cuda", requires_grad=True)
    weight = torch.randn(size=(C,), dtype=dtype, device="cuda", requires_grad=True)
    bias = torch.randn(size=(C,), dtype=dtype, device="cuda", requires_grad=True)
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)

    ref_out = torch.nn.functional.group_norm(
        ref_inp, num_groups, weight=ref_weight, bias=ref_bias, eps=eps
    )
    ref_mean = torch.mean(ref_inp.reshape([N, num_groups, -1]), dim=2)
    ref_var = torch.var(ref_inp.reshape([N, num_groups, -1]), dim=2, correction=0)
    ref_rstd = torch.rsqrt(ref_var + eps)

    (res_out, res_mean, res_rstd) = flag_gems.group_norm(
        inp, weight, bias, N, C, HW, num_groups, eps
    )

    gems_assert_close(res_mean, ref_mean, dtype)
    gems_assert_close(res_rstd, ref_rstd, dtype)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
    )
    (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
        res_out, (inp, weight, bias), out_grad
    )
    group_size = C // num_groups
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=group_size * HW)
    gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=N * HW)
    gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=N * HW)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
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

    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)

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
    gems_assert_close(res_mean, ref_mean, dtype)
    gems_assert_close(res_rstd, ref_rstd, dtype)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
    )
    (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
        res_out, (inp, weight, bias), out_grad
    )
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=N)
    gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=M)
    gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=M)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log_softmax(shape, dtype):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.log_softmax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.log_softmax(inp, dim=dim)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_max(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_max_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.max(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.max(inp, dim=dim, keepdim=keepdim)
    ref_out_value, ref_out_index = ref_out
    res_out_value, res_out_index = res_out
    gems_assert_equal(res_out_index, ref_out_index)
    gems_assert_equal(res_out_value, ref_out_value)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mean(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.mean(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.mean(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mean_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.mean(inp, dim, keepdim)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_min(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.min(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.min(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_min_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.min(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.min(inp, dim=dim, keepdim=keepdim)
    ref_out_value, ref_out_index = ref_out
    res_out_value, res_out_index = res_out
    gems_assert_equal(res_out_index, ref_out_index)
    gems_assert_equal(res_out_value, ref_out_value)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_prod(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.prod(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.prod(inp)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_prod_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.prod(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.prod(inp, dim=dim, keepdim=keepdim)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rmsnorm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.randn(layer_shape, dtype=dtype, device="cuda")
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)

    def _torch_rms_norm(x, weight, eps):
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states

    ref_out = _torch_rms_norm(ref_inp, weight=ref_weight, eps=eps)

    res_out = flag_gems.rms_norm(inp, list(layer_shape), weight=weight, eps=eps)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_skip_layernorm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    residual = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.randn(layer_shape, dtype=dtype, device="cuda")
    bias = torch.randn(layer_shape, dtype=dtype, device="cuda")
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_residual = to_reference(residual, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)

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

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_skip_rmsnorm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    residual = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.randn(layer_shape, dtype=dtype, device="cuda")
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_residual = to_reference(residual, True)
    ref_weight = to_reference(weight, True)

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

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1])
def test_accuracy_softmax(shape, dtype, dim):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.softmax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.softmax(inp, dim=dim)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sum(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.sum(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sum(inp)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sum_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.sum(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.sum(inp, dim=dim, keepdim=keepdim)

    if isinstance(dim, int):
        dim = [dim]
    dim = [d % inp.ndim for d in dim]
    _dim = 1
    for d in dim:
        _dim *= shape[d]
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=_dim)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_varmean(shape, dim, correction, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_var, ref_mean = torch.var_mean(
        ref_inp, dim, correction=correction, keepdim=keepdim
    )
    with flag_gems.use_gems():
        res_var, res_mean = torch.var_mean(
            inp, dim, correction=correction, keepdim=keepdim
        )

    gems_assert_close(res_mean, ref_mean, dtype)
    gems_assert_close(res_var, ref_var, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("ord", [2, float("inf"), -float("inf"), 0, 1])
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_vectornorm(shape, ord, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.linalg.vector_norm(ref_inp, ord, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.linalg.vector_norm(inp, ord, dim, keepdim)

    gems_assert_close(res_out, ref_out, dtype)
