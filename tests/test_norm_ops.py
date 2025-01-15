import math

import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    REDUCTION_SHAPES,
    gems_assert_close,
    to_reference,
)
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else FLOAT_DTYPES
DIMS_LIST = [1] if QUICK_MODE else [0, 1, [0, 1], [1, 0]]
KEEPDIM_DIMS = (
    [(True, DIMS_LIST[0])] if QUICK_MODE else list(zip([True, False] * 2, DIMS_LIST))
)


@pytest.mark.group_norm
@pytest.mark.native_group_norm
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
@pytest.mark.parametrize("wb_none", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_groupnorm(N, C, H, W, num_groups, dtype, wb_none):
    HW = H * W
    inp = torch.randn(
        size=(N, C, H, W), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    if wb_none:
        weight = None
        bias = None
    else:
        weight = torch.randn(
            size=(C,), dtype=dtype, device=flag_gems.device, requires_grad=True
        )
        bias = torch.randn(
            size=(C,), dtype=dtype, device=flag_gems.device, requires_grad=True
        )
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)

    ref_out = torch.nn.functional.group_norm(
        ref_inp, num_groups, weight=ref_weight, bias=ref_bias, eps=eps
    )

    with flag_gems.use_gems():
        res_out = torch.group_norm(inp, num_groups, weight=weight, bias=bias, eps=eps)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    if wb_none:
        (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    else:
        (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
        )
        (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
            res_out, (inp, weight, bias), out_grad
        )
        gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=N * HW)
        gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=N * HW)
    group_size = C // num_groups
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=group_size * HW)


@pytest.mark.layer_norm
@pytest.mark.native_layer_norm
@pytest.mark.parametrize(
    "shape",
    (
        [(1, 40999)]
        if QUICK_MODE
        else [
            (200, 36),
            (4096, 100),
            (1, 40999),
            (100, 40499),
            (4096, 256),
        ]
    ),
)
@pytest.mark.parametrize("wb_none", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_layernorm(shape, dtype, wb_none):
    M = shape[0]
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(
        shape[:2], dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    if wb_none:
        weight = None
        bias = None
    else:
        weight = torch.randn(
            layer_shape, dtype=dtype, device=flag_gems.device, requires_grad=True
        )
        bias = torch.randn(
            layer_shape, dtype=dtype, device=flag_gems.device, requires_grad=True
        )
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
    with flag_gems.use_gems():
        res_out = torch.layer_norm(
            inp,
            list(layer_shape),
            weight=weight,
            bias=bias,
            eps=eps,
        )

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    if wb_none:
        (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    else:
        (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
        )
        (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
            res_out, (inp, weight, bias), out_grad
        )
        gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=M)
        gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=M)
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=N)


@pytest.mark.instance_norm
@pytest.mark.native_instance_norm
@pytest.mark.parametrize(
    "shape",
    (
        [
            (2, 1, 2, 1),
        ]
        if QUICK_MODE
        else [
            (1, 1, 2, 2),
            (2, 1, 2, 2),
            (2, 3, 2, 2),
            (2, 3, 128, 128),
            (4, 16, 8, 8),
            (2, 3, 1024),
            (2, 3, 2048),
            (2, 3, 4096),
            (2, 3, 8192),
            (2, 3, 10240),
        ]
    ),
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("has_weight_bias", [True] if QUICK_MODE else [False, True])
@pytest.mark.parametrize("use_input_stats", [True] if QUICK_MODE else [False, True])
@pytest.mark.parametrize("has_running_stats", [False] if QUICK_MODE else [False, True])
def test_accuracy_instancenorm(
    shape, dtype, has_weight_bias, use_input_stats, has_running_stats
):
    if use_input_stats is False and has_running_stats is False:
        return

    B, C = shape[:2]
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    if has_weight_bias:
        weight = torch.randn(
            size=(C,), dtype=dtype, device=flag_gems.device, requires_grad=True
        )
        bias = torch.randn(
            size=(C,), dtype=dtype, device=flag_gems.device, requires_grad=True
        )
    else:
        weight, bias = None, None
    running_mean = (
        torch.randn(size=(C,), dtype=torch.float32, device=flag_gems.device)
        if has_running_stats
        else None
    )
    running_var = (
        torch.randn(size=(C,), dtype=torch.float32, device=flag_gems.device).abs()
        + 1e-5
        if has_running_stats
        else None
    )
    momentum = 0.1
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)
    ref_running_mean = to_reference(
        running_mean.clone() if has_running_stats else None, True
    )
    ref_running_var = to_reference(
        running_var.clone() if has_running_stats else None, True
    )

    ref_out = torch.nn.functional.instance_norm(
        ref_inp,
        running_mean=ref_running_mean,
        running_var=ref_running_var,
        weight=ref_weight,
        bias=ref_bias,
        use_input_stats=use_input_stats,
        momentum=momentum,
        eps=eps,
    )
    with flag_gems.use_gems():
        res_out = torch.instance_norm(
            inp,
            running_mean=running_mean,
            running_var=running_var,
            weight=weight,
            bias=bias,
            use_input_stats=use_input_stats,
            momentum=momentum,
            eps=eps,
            cudnn_enabled=True,
        )

    gems_assert_close(res_out, ref_out, dtype)
    if has_running_stats:
        gems_assert_close(running_mean, ref_running_mean, running_mean.dtype)
        gems_assert_close(running_var, ref_running_var, running_var.dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    if has_weight_bias:
        (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
        )
        (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
            res_out, (inp, weight, bias), out_grad
        )
    else:
        (ref_in_grad,) = torch.autograd.grad(ref_out, (ref_inp,), ref_grad)
        (res_in_grad,) = torch.autograd.grad(res_out, (inp,), out_grad)
    M = B * C
    N = inp.numel() // M
    if use_input_stats:
        gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=N)
        if has_weight_bias:
            gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=B * N)
            gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=B * N)


WEIGHT_NORM_SHAPE_DIM = list(zip(REDUCTION_SHAPES, [-1] if QUICK_MODE else [0, -1, 1]))


@pytest.mark.weight_norm
@pytest.mark.parametrize("shape, dim", WEIGHT_NORM_SHAPE_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_weightnorm(shape, dtype, dim):
    dim = dim % len(shape)
    v = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    g = torch.randn(
        [1 if i != dim else shape[i] for i in range(v.ndim)],
        dtype=dtype,
        device=flag_gems.device,
        requires_grad=True,
    )
    reduce_size = v.numel() // shape[dim]

    ref_v = to_reference(v, True)
    ref_g = to_reference(g, True)
    ref_w_out = torch._weight_norm(ref_v, ref_g, dim)
    with flag_gems.use_gems():
        res_w_out = torch._weight_norm(v, g, dim)
    gems_assert_close(res_w_out, ref_w_out, dtype, reduce_dim=reduce_size)

    res_w_grad = torch.randn(
        shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_w_grad = to_reference(res_w_grad, True)

    ref_v_grad, ref_g_grad = torch.autograd.grad(
        ref_w_out, (ref_v, ref_g), grad_outputs=ref_w_grad
    )
    res_v_grad, res_g_grad = torch.autograd.grad(
        res_w_out, (v, g), grad_outputs=res_w_grad
    )
    gems_assert_close(res_v_grad, ref_v_grad, dtype, reduce_dim=reduce_size)
    gems_assert_close(res_g_grad, ref_g_grad, dtype, reduce_dim=reduce_size)


WEIGHT_NORM_INTERFACE_SHAPE_DIM = list(
    zip(REDUCTION_SHAPES, [-1] if QUICK_MODE else [0, -1, -1])
)


@pytest.mark.weight_norm_interface
@pytest.mark.parametrize("shape, dim", WEIGHT_NORM_INTERFACE_SHAPE_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_weightnorm_interface(shape, dtype, dim):
    dim = dim % len(shape)
    v = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    g = torch.randn(
        shape[dim], dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    reduce_size = v.numel() // shape[dim]

    ref_v = to_reference(v, True)
    ref_g = to_reference(g, True)

    ref_w_out, ref_norm_out = torch._weight_norm_interface(ref_v, ref_g, dim)
    with flag_gems.use_gems():
        res_w_out, res_norm_out = torch._weight_norm_interface(v, g, dim)
    gems_assert_close(res_w_out, ref_w_out, dtype, reduce_dim=reduce_size)
    gems_assert_close(
        res_norm_out, ref_norm_out, res_norm_out.dtype, reduce_dim=reduce_size
    )

    res_w_grad = torch.randn_like(v)
    ref_w_grad = to_reference(res_w_grad, True)

    ref_v_grad, ref_g_grad = torch.autograd.grad(
        ref_w_out, (ref_v, ref_g), grad_outputs=ref_w_grad
    )
    res_v_grad, res_g_grad = torch.autograd.grad(
        res_w_out, (v, g), grad_outputs=res_w_grad
    )

    gems_assert_close(res_v_grad, ref_v_grad, dtype, reduce_dim=reduce_size)
    gems_assert_close(res_g_grad, ref_g_grad, dtype, reduce_dim=reduce_size)


@pytest.mark.rms_norm
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rmsnorm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape[:2], dtype=dtype, device=flag_gems.device)
    weight = torch.randn(layer_shape, dtype=dtype, device=flag_gems.device)
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


@pytest.mark.skip_layer_norm
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_skip_layernorm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape[:2], dtype=dtype, device=flag_gems.device)
    residual = torch.randn(shape[:2], dtype=dtype, device=flag_gems.device)
    weight = torch.randn(layer_shape, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(layer_shape, dtype=dtype, device=flag_gems.device)
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


@pytest.mark.skip_rms_norm
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_skip_rmsnorm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape[:2], dtype=dtype, device=flag_gems.device)
    residual = torch.randn(shape[:2], dtype=dtype, device=flag_gems.device)
    weight = torch.randn(layer_shape, dtype=dtype, device=flag_gems.device)
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


@pytest.mark.vector_norm
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize(
    "ord", [2] if QUICK_MODE else [2, float("inf"), -float("inf"), 0, 1]
)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIMS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_vectornorm(shape, ord, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.linalg.vector_norm(ref_inp, ord, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.linalg.vector_norm(inp, ord, dim, keepdim)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.batch_norm
@pytest.mark.parametrize(
    "shape",
    [
        (16, 3),
        (32, 32, 32),
        (8, 32, 224, 224),
        (2050, 16, 32, 32),
        (8, 16, 3, 224, 224),
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("require_grad", [True, False])
def test_accuracy_batch_norm(shape, dtype, affine, require_grad):
    C = shape[1]
    inp = torch.randn(
        size=shape, dtype=dtype, device=flag_gems.device, requires_grad=require_grad
    )
    weight = (
        torch.randn(
            size=(C,), dtype=dtype, device=flag_gems.device, requires_grad=require_grad
        )
        if affine
        else None
    )
    bias = (
        torch.randn(
            size=(C,), dtype=dtype, device=flag_gems.device, requires_grad=require_grad
        )
        if affine
        else None
    )

    running_mean = torch.zeros(size=(C,), dtype=dtype, device=flag_gems.device)
    running_var = torch.ones(size=(C,), dtype=dtype, device=flag_gems.device)

    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)
    ref_running_mean = to_reference(running_mean, True)
    ref_running_var = to_reference(running_var, True)

    training = require_grad

    ref_out = torch.nn.functional.batch_norm(
        ref_inp,
        ref_running_mean,
        ref_running_var,
        weight=ref_weight,
        bias=ref_bias,
        training=training,
        eps=eps,
    )

    with flag_gems.use_gems():
        res_out = torch.nn.functional.batch_norm(
            inp,
            running_mean,
            running_var,
            weight=weight,
            bias=bias,
            training=training,
            eps=eps,
        )

    gems_assert_close(res_out, ref_out, dtype)
    gems_assert_close(running_mean, ref_running_mean, dtype)
    gems_assert_close(running_var, ref_running_var, dtype)

    if not require_grad:
        return

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)
    reduce_dim = int(math.prod(shape) / C)

    if affine:
        (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
        )
        (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
            res_out, (inp, weight, bias), out_grad
        )

        gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=reduce_dim)
        gems_assert_close(
            res_weight_grad, ref_weight_grad, dtype, reduce_dim=reduce_dim
        )
        gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=reduce_dim)
    else:
        (ref_in_grad,) = torch.autograd.grad(ref_out, (ref_inp,), ref_grad)
        (res_in_grad,) = torch.autograd.grad(res_out, (inp,), out_grad)

        gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=reduce_dim)
