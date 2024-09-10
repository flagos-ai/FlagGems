import pytest
import torch

import flag_gems

from .accuracy_utils import (
    DIM_LIST,
    DIMS_LIST,
    FLOAT_DTYPES,
    INT_DTYPES,
    ONE_DIM_SHAPES,
    REDUCTION_MNK_SHAPES,
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


@pytest.mark.parametrize("label_smoothing", [0, 0.1, 1])
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("ignore_index", [1, 200, -100])
def test_accuracy_cross_entropy_loss_indices(
    shape, dtype, ignore_index, reduction, label_smoothing
):
    dim = 1
    up_limit = shape[dim] - 1
    target_shape = list(shape)
    del target_shape[dim]

    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    target = torch.randint(0, up_limit, target_shape, device="cuda")
    weight = torch.randn(shape[dim], dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target)
    ref_weight = to_reference(weight, True)
    ref_criterion = torch.nn.CrossEntropyLoss(
        weight=ref_weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    res_criterion = torch.nn.CrossEntropyLoss(
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )

    ref_out = ref_criterion(ref_inp, ref_target)
    with flag_gems.use_gems():
        res_out = res_criterion(inp, target)
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


@pytest.mark.parametrize("label_smoothing", [0, 0.1, 1])
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cross_entropy_loss_probabilities(
    shape, dtype, reduction, label_smoothing
):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    target = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.randn(shape[dim], dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)
    ref_weight = to_reference(weight, True)
    ref_criterion = torch.nn.CrossEntropyLoss(
        weight=ref_weight,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    res_criterion = torch.nn.CrossEntropyLoss(
        weight=weight,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )

    ref_out = ref_criterion(ref_inp, ref_target)
    with flag_gems.use_gems():
        res_out = res_criterion(inp, target)
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


@pytest.mark.parametrize(
    "shape", REDUCTION_SHAPES + ONE_DIM_SHAPES + REDUCTION_MNK_SHAPES
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_cumsum(shape, dtype):
    if shape in REDUCTION_MNK_SHAPES:
        dim = 1
    else:
        dim = -1

    if dtype in INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device="cuda").to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.cumsum(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.cumsum(inp, dim=dim)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])


@pytest.mark.parametrize(
    "shape", REDUCTION_SHAPES + ONE_DIM_SHAPES + REDUCTION_MNK_SHAPES
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + [torch.bool])
def test_accuracy_nonzero(shape, dtype):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, shape, dtype=torch.int, device="cuda").to(dtype)
    elif dtype in INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device="cuda").to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, False)

    ref_out = torch.nonzero(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nonzero(inp)

    gems_assert_equal(res_out, ref_out)


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


@pytest.mark.skip(reason="operator undone")
@pytest.mark.parametrize("src_shape", [(128, 16 * i, 32 * i) for i in range(1, 10, 4)])
@pytest.mark.parametrize("inp_shape", [(512, 32 * i, 64 * i) for i in range(1, 10, 4)])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_scatter_src(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device="cuda")
    src = torch.randn(src_shape, dtype=dtype, device="cuda")
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device="cuda")

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_src = to_reference(src)
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src)
    from src.flag_gems.ops import scatter_src

    res_out = scatter_src(inp, dim, index, src)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.skip(reason="operator undone")
@pytest.mark.parametrize("src_shape", [(2, 2, 2)])
@pytest.mark.parametrize("inp_shape", [(3, 3, 3)])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_scatter_add(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device="cuda")
    src = torch.randn(src_shape, dtype=dtype, device="cuda")
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device="cuda")

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_src = to_reference(src)
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src, reduce="add")
    from src.flag_gems.ops import scatter_reduce

    res_out = scatter_reduce(inp, dim, index, src, reduce="add")

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.skip(reason="operator undone")
@pytest.mark.parametrize("src_shape", [(128, 16 * i, 32 * i) for i in range(1, 10, 4)])
@pytest.mark.parametrize("inp_shape", [(512, 32 * i, 64 * i) for i in range(1, 10, 4)])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_scatter_mul(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device="cuda")
    src = torch.randn(src_shape, dtype=dtype, device="cuda")
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device="cuda")

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_src = to_reference(src)
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src, reduce="multiply")
    from src.flag_gems.ops import scatter_reduce

    res_out = scatter_reduce(inp, dim, index, src, reduce="multiply")

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.skip(reason="operator undone")
@pytest.mark.parametrize("inp_shape", [(512, 32 * i, 64 * i) for i in range(1, 10, 4)])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_gather(inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device="cuda")
    size_dim = inp_shape[dim]

    import random

    index_shape = [
        random.randint(1, inp_shape[0]),
        random.randint(1, inp_shape[1]),
        random.randint(1, inp_shape[2]),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device="cuda")

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_out = torch.gather(ref_inp, dim, ref_index)

    from src.flag_gems.ops import gather

    res_out = gather(inp, dim, index)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.skip(reason="operator undone")
@pytest.mark.parametrize("out_shape", [(128, 16 * i, 32 * i) for i in range(1, 10, 4)])
@pytest.mark.parametrize("inp_shape", [(512, 32 * i, 64 * i) for i in range(1, 10, 4)])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_gather_out(out_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device="cuda")
    size_dim = min(out_shape[dim], inp_shape[dim])

    import random

    index_shape = [
        random.randint(1, min(out_shape[0], inp_shape[0])),
        random.randint(1, min(out_shape[1], inp_shape[1])),
        random.randint(1, min(out_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device="cuda")
    out = torch.randn(tuple(index_shape), dtype=dtype, device="cuda")

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_out = torch.gather(ref_inp, dim, ref_index, sparse_grad=False, out=out)

    from src.flag_gems.ops import gather_out

    res_out = gather_out(inp, dim, index, sparse_grad=False, out=out)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", [(8192, 256 * i) for i in range(1, 10, 2)])
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_index_select(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    index_size = inp.size(dim)
    from math import floor

    index = torch.randint(0, index_size, [floor(index_size * 0.8)], device="cuda")

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_out = torch.index_select(ref_inp, dim, ref_index)
    with flag_gems.use_gems():
        res_out = torch.index_select(inp, dim, index)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_accuracy_masked_select(shape, dtype, threshold):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    mask = torch.randn(shape, dtype=dtype, device="cuda") < threshold

    ref_inp = to_reference(inp)
    ref_mask = to_reference(mask)
    ref_out = torch.masked_select(ref_inp, ref_mask)
    with flag_gems.use_gems():
        res_out = torch.masked_select(inp, mask)

    gems_assert_equal(res_out, ref_out)
