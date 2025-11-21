import math
import os

import numpy as np
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
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_inp = torch.randn(size=(N, C, H, W), dtype=dtype, device=flag_gems.device)
    if wb_none:
        res_weight = None
        res_bias = None
    else:
        res_weight = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
        res_bias = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    eps = 1e-5

    ref_inp = to_reference(res_inp, True)
    ref_weight = to_reference(res_weight, True)
    ref_bias = to_reference(res_bias, True)

    ref_out = torch.nn.functional.group_norm(
        ref_inp, num_groups, weight=ref_weight, bias=ref_bias, eps=eps
    )

    with flag_gems.use_gems():
        res_out = torch.group_norm(
            res_inp, num_groups, weight=res_weight, bias=res_bias, eps=eps
        )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.group_norm
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
def test_accuracy_groupnorm_backward(N, C, H, W, num_groups, dtype, wb_none):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_inp = torch.randn(size=(N, C, H, W), dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)
    res_mean = torch.randn([N, num_groups], dtype=dtype, device=flag_gems.device)
    res_rstd = torch.randn([N, num_groups], dtype=dtype, device=flag_gems.device)

    if wb_none:
        res_weight = None
        output_mask = [True, False, False]
    else:
        res_weight = torch.randn(C, dtype=dtype, device=flag_gems.device)
        output_mask = [True, True, True]

    ref_inp = to_reference(res_inp, True)
    ref_grad = to_reference(res_grad, True)
    ref_mean = to_reference(res_mean, True)
    ref_rstd = to_reference(res_rstd, True)
    ref_weight = to_reference(res_weight, True)

    group_size = C // num_groups
    HxW = H * W

    (
        ref_in_grad,
        ref_weight_grad,
        ref_bias_grad,
    ) = torch.ops.aten.native_group_norm_backward(
        ref_grad,
        ref_inp,
        ref_mean,
        ref_rstd,
        ref_weight,
        N,
        C,
        HxW,
        num_groups,
        output_mask,
    )
    with flag_gems.use_gems():
        (
            res_in_grad,
            res_weight_grad,
            res_bias_grad,
        ) = torch.ops.aten.native_group_norm_backward(
            res_grad,
            res_inp,
            res_mean,
            res_rstd,
            res_weight,
            N,
            C,
            HxW,
            num_groups,
            output_mask,
        )
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=group_size * HxW)
    if not wb_none:
        gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=N * HxW)
        gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=N * HxW)


@pytest.mark.layer_norm
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
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if wb_none:
        res_weight = None
        res_bias = None
    else:
        res_weight = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
        res_bias = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
    eps = 1e-5

    ref_inp = to_reference(res_inp, True)
    ref_weight = to_reference(res_weight, True)
    ref_bias = to_reference(res_bias, True)

    ref_out = torch.layer_norm(
        ref_inp,
        shape[1:],
        weight=ref_weight,
        bias=ref_bias,
        eps=eps,
    )
    with flag_gems.use_gems():
        res_out = torch.layer_norm(
            res_inp,
            shape[1:],
            weight=res_weight,
            bias=res_bias,
            eps=eps,
        )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.layer_norm
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
def test_accuracy_layernorm_backward(shape, dtype, wb_none):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    if flag_gems.vendor_name == "mthreads":
        # Compatible with older versions of LLVM
        os.environ["DISABLE_LLVM_OPT"] = "1"

    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)
    res_mean = torch.randn(shape[0], dtype=dtype, device=flag_gems.device)
    res_rstd = torch.randn(shape[0], dtype=dtype, device=flag_gems.device)
    if wb_none:
        res_weight = None
        res_bias = None
        output_mask = [True, False, False]
    else:
        res_weight = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
        res_bias = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
        output_mask = [True, True, True]

    normalized_shape = shape[1:]

    ref_inp = to_reference(res_inp, True)
    ref_grad = to_reference(res_grad, True)
    ref_mean = to_reference(res_mean, True)
    ref_rstd = to_reference(res_rstd, True)
    ref_weight = to_reference(res_weight, True)
    ref_bias = to_reference(res_bias, True)

    (
        ref_in_grad,
        ref_weight_grad,
        ref_bias_grad,
    ) = torch.ops.aten.native_layer_norm_backward(
        ref_grad,
        ref_inp,
        normalized_shape,
        ref_mean,
        ref_rstd,
        ref_weight,
        ref_bias,
        output_mask,
    )
    with flag_gems.use_gems():
        (
            res_in_grad,
            res_weight_grad,
            res_bias_grad,
        ) = torch.ops.aten.native_layer_norm_backward(
            res_grad,
            res_inp,
            normalized_shape,
            res_mean,
            res_rstd,
            res_weight,
            res_bias,
            output_mask,
        )

    gems_assert_close(res_in_grad, ref_in_grad, dtype)
    if not wb_none:
        gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=shape[0])
        gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=shape[0])

    if flag_gems.vendor_name == "mthreads":
        # Compatible with older versions of LLVM
        del os.environ["DISABLE_LLVM_OPT"]


@pytest.mark.instance_norm
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

    res_out = flag_gems.instance_norm(
        inp,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        use_input_stats=use_input_stats,
        momentum=momentum,
        eps=eps,
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


@pytest.mark.skipif(
    True, reason="Temporarely skip for ci"
)  # todo: improve backward precision
@pytest.mark.weight_norm
@pytest.mark.parametrize("shape, dim", WEIGHT_NORM_SHAPE_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_weightnorm(shape, dtype, dim):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
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
    res_w_out = flag_gems.weight_norm(v, g, dim)
    gems_assert_close(res_w_out, ref_w_out, dtype, reduce_dim=reduce_size)

    res_w_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_w_grad = to_reference(res_w_grad, True)

    ref_v_grad, ref_g_grad = torch.autograd.grad(
        ref_w_out, (ref_v, ref_g), grad_outputs=ref_w_grad
    )
    res_v_grad, res_g_grad = torch.autograd.grad(
        res_w_out, (v, g), grad_outputs=res_w_grad
    )
    gems_assert_close(
        res_v_grad, ref_v_grad, dtype, reduce_dim=reduce_size, equal_nan=True
    )
    gems_assert_close(
        res_g_grad, ref_g_grad, dtype, reduce_dim=reduce_size, equal_nan=True
    )


WEIGHT_NORM_INTERFACE_SHAPE_DIM = list(
    zip(REDUCTION_SHAPES, [-1] if QUICK_MODE else [0, -1, -1])
)


@pytest.mark.weight_norm
@pytest.mark.parametrize("shape, dim", WEIGHT_NORM_INTERFACE_SHAPE_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_weightnorm_interface(shape, dtype, dim):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    dim = dim % len(shape)
    v = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    g = torch.randn(shape[dim], dtype=dtype, device=flag_gems.device)
    reduce_size = v.numel() // shape[dim]

    ref_v = to_reference(v, True)
    ref_g = to_reference(g, True)

    ref_w_out, ref_norm_out = torch._weight_norm_interface(ref_v, ref_g, dim)
    with flag_gems.use_gems():
        res_w_out, res_norm_out = torch._weight_norm_interface(v, g, dim)
    gems_assert_close(res_w_out, ref_w_out, dtype, reduce_dim=reduce_size)
    gems_assert_close(res_norm_out, ref_norm_out, dtype, reduce_dim=reduce_size)


@pytest.mark.skipif(
    True, reason="Temporarely skip for ci"
)  # todo: improve backward precision
@pytest.mark.weight_norm
@pytest.mark.parametrize("shape, dim", WEIGHT_NORM_INTERFACE_SHAPE_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_weightnorm_interface_backward(shape, dtype, dim):
    dim = dim % len(shape)
    res_w_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_v = torch.randn_like(res_w_grad)
    if flag_gems.vendor_name == "kunlunxin":
        if shape == (4096, 256):
            res_v = res_v.uniform_(-0.01, 0.01)
    res_g = torch.randn(shape[dim], dtype=dtype, device=flag_gems.device)
    res_norm = torch.randn_like(res_g)

    ref_w_grad = to_reference(res_w_grad, True)
    ref_v = to_reference(res_v, True)
    ref_g = to_reference(res_g, True)
    ref_norm = to_reference(res_norm, True)

    ref_v_grad, ref_g_grad = torch.ops.aten._weight_norm_interface_backward(
        ref_w_grad, ref_v, ref_g, ref_norm, dim
    )
    with flag_gems.use_gems():
        res_v_grad, res_g_grad = torch.ops.aten._weight_norm_interface_backward(
            res_w_grad, res_v, res_g, res_norm, dim
        )
    reduce_size = res_v.numel() // shape[dim]
    gems_assert_close(
        res_v_grad, ref_v_grad, dtype, reduce_dim=reduce_size, equal_nan=True
    )
    gems_assert_close(
        res_g_grad, ref_g_grad, dtype, reduce_dim=reduce_size, equal_nan=True
    )


@pytest.mark.rms_norm
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rmsnorm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, shape[:2]).astype(np.float32)
    np_grad = np.random.uniform(-0.01, 0.01, shape[:2]).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, layer_shape).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device, requires_grad=True)
    weight = torch.tensor(
        np_weight, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    eps = 1e-5

    ref_inp = to_reference(inp)
    ref_weight = to_reference(weight)

    def _torch_rms_norm(x, weight, eps):
        upcast_x = x.to(torch.float32)
        variance = upcast_x.pow(2).mean(-1, keepdim=True)
        hidden_states = upcast_x * torch.rsqrt(variance + eps).to(torch.float32)
        hidden_states = hidden_states.to(x.dtype)
        return weight * hidden_states

    ref_out = _torch_rms_norm(ref_inp, weight=ref_weight, eps=eps)
    res_out = flag_gems.rms_norm(inp, list(layer_shape), weight=weight, eps=eps)

    res_grad = torch.tensor(
        np_grad, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_grad = to_reference(res_grad)

    res_grad, res_weight_grad = torch.autograd.grad(res_out, (inp, weight), res_grad)
    ref_grad, ref_weight_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight), ref_grad
    )

    gems_assert_close(res_out, ref_out, dtype)
    if flag_gems.vendor_name == "kunlunxin" and shape == (200, 40999, 3):
        pytest.skip("wait for backward support")
    gems_assert_close(res_grad, ref_grad, dtype)
    gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=N)


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


@pytest.mark.fused_add_rms_norm
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_fused_add_rms_norm(shape, dtype):
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

    def _torch_fused_add_rms_norm(x, residual, weight, eps):
        x = x + residual
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states, x

    ref_out, ref_new_residual = _torch_fused_add_rms_norm(
        ref_inp,
        ref_residual,
        weight=ref_weight,
        eps=eps,
    )

    res_out, res_new_residual = flag_gems.fused_add_rms_norm(
        inp, residual, list(layer_shape), weight=weight, eps=eps
    )

    gems_assert_close(res_out, ref_out, dtype)
    gems_assert_close(res_new_residual, ref_new_residual, dtype)


@pytest.mark.vector_norm
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize(
    "ord", [2] if QUICK_MODE else [2, float("inf"), -float("inf"), 0, 1]
)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIMS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_vectornorm(shape, ord, dim, keepdim, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.linalg.vector_norm(ref_inp, ord, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.linalg.vector_norm(inp, ord, dim, keepdim)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
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
def test_accuracy_batch_norm(shape, dtype, affine):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(23)
        torch.mlu.manual_seed_all(23)
    C = shape[1]
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    weight = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )
    bias = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )

    running_mean = torch.zeros(size=(C,), dtype=dtype, device=flag_gems.device)
    running_var = torch.ones(size=(C,), dtype=dtype, device=flag_gems.device)

    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)
    ref_running_mean = to_reference(running_mean, True)
    ref_running_var = to_reference(running_var, True)

    ref_out = torch.nn.functional.batch_norm(
        ref_inp,
        ref_running_mean,
        ref_running_var,
        weight=ref_weight,
        bias=ref_bias,
        eps=eps,
    )

    with flag_gems.use_gems():
        res_out = torch.nn.functional.batch_norm(
            inp,
            running_mean,
            running_var,
            weight=weight,
            bias=bias,
            eps=eps,
        )

    gems_assert_close(res_out, ref_out, dtype)
    gems_assert_close(running_mean, ref_running_mean, dtype)
    gems_assert_close(running_var, ref_running_var, dtype)


@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
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
def test_accuracy_batch_norm_backward(shape, dtype, affine):
    C = shape[1]
    res_grad = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    res_inp = torch.randn_like(res_grad)
    res_weight = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )
    res_running_mean = torch.zeros(size=(C,), dtype=dtype, device=flag_gems.device)
    res_running_var = torch.ones(size=(C,), dtype=dtype, device=flag_gems.device)
    res_save_mean = torch.randn(C, dtype=torch.float32, device=flag_gems.device)
    res_save_invstd = torch.randn(C, dtype=torch.float32, device=flag_gems.device)

    ref_grad = to_reference(res_grad, True)
    ref_inp = to_reference(res_inp, True)
    ref_weight = to_reference(res_weight, True)
    ref_running_mean = to_reference(res_running_mean, True)
    ref_running_var = to_reference(res_running_var, True)
    ref_save_mean = to_reference(res_save_mean, True)
    ref_save_invstd = to_reference(res_save_invstd, True)

    train = True
    eps = 1e-05
    if affine:
        output_mask = [True, True, True]
    else:
        output_mask = [True, False, False]

    (
        ref_in_grad,
        ref_weight_grad,
        ref_bias_grad,
    ) = torch.ops.aten.native_batch_norm_backward(
        ref_grad,
        ref_inp,
        ref_weight,
        ref_running_mean,
        ref_running_var,
        ref_save_mean,
        ref_save_invstd,
        train,
        eps,
        output_mask,
    )
    with flag_gems.use_gems():
        (
            res_in_grad,
            res_weight_grad,
            res_bias_grad,
        ) = torch.ops.aten.native_batch_norm_backward(
            res_grad,
            res_inp,
            res_weight,
            res_running_mean,
            res_running_var,
            res_save_mean,
            res_save_invstd,
            train,
            eps,
            output_mask,
        )

    reduce_dim = math.prod(shape) // C
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=reduce_dim)
    if affine:
        gems_assert_close(
            res_weight_grad, ref_weight_grad, dtype, reduce_dim=reduce_dim
        )
        gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=reduce_dim)
