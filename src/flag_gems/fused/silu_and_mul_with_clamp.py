# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import lru_cache
from numbers import Real

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.type_utils import ELEMENTWISE_TYPE_PROMOTION_KIND, type_promotion

logger = logging.getLogger(__name__)


def silu_clamp_get_configs(linear=False):
    block_warps = (
        (128, 1),
        (128, 2),
        (256, 1),
        (256, 2),
        (512, 2),
        (512, 4),
        (1024, 4),
        (1024, 8),
        (2048, 4),
        (2048, 8),
        (4096, 4),
        (4096, 8),
    )
    return [
        triton.Config(
            {"BLOCK_SIZE": block, **({"UNMASKED": False} if linear else {})},
            num_warps=warps,
            num_stages=1,
        )
        for block, warps in block_warps
    ]


_FULL_LINEAR_MIN_ELEMENTS = 1 << 20
_LINEAR_BLOCK_SIZES = (128, 256, 512, 1024, 2048, 4096)


def _use_full_linear_kernel(n_elements):
    return n_elements >= _FULL_LINEAR_MIN_ELEMENTS and any(
        n_elements % block_size == 0 for block_size in _LINEAR_BLOCK_SIZES
    )


def silu_clamp_early_prune(configs, named_args, **kwargs):
    n = {**named_args, **kwargs}["n_elements"]
    return [
        config
        for config in configs
        if config.kwargs["BLOCK_SIZE"] >= 32 * config.num_warps
        and config.kwargs["BLOCK_SIZE"] <= 1024 * config.num_warps
        and (
            not config.kwargs.get("UNMASKED", False)
            or n % config.kwargs["BLOCK_SIZE"] == 0
        )
    ]


def silu_clamp_full_early_prune(configs, named_args, **kwargs):
    n = {**named_args, **kwargs}["n_elements"]
    return [
        config
        for config in silu_clamp_early_prune(configs, named_args, **kwargs)
        if n % config.kwargs["BLOCK_SIZE"] == 0
    ]


@triton.jit
def _strided_offset(linear, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    offset = tl.full(linear.shape, 0, tl.int64)
    for dim in tl.static_range(len(SHAPE) - 1, 0, -1):
        index = linear % SHAPE[dim]
        linear = linear // SHAPE[dim]
        offset += index.to(tl.int64) * STRIDES[dim]
    if len(SHAPE) > 0:
        offset += linear.to(tl.int64) * STRIDES[0]
    return offset


@triton.jit
def _silu_clamp_value(x, y, limit):
    gate = tl.minimum(x, limit)
    up = tl.minimum(tl.maximum(y, -limit), limit)
    return tl.fdiv(gate, 1.0 + tl.exp(-gate)) * up


@triton.jit
def _silu_clamp_grad_values(
    x, y, grad, limit, NEED_DX: tl.constexpr, NEED_DY: tl.constexpr
):
    gate = tl.minimum(x, limit)
    sig = 1 / (1 + tl.exp(-gate))
    dx = tl.full(x.shape, 0, tl.float32)
    dy = tl.full(x.shape, 0, tl.float32)
    if NEED_DX:
        up = tl.minimum(tl.maximum(y, -limit), limit)
        derivative = sig * (1 + gate * (1 - sig))
        dx = grad * up * derivative * (x <= limit).to(tl.float32)
    if NEED_DY:
        gate_silu = gate * sig
        dy = grad * gate_silu * ((y >= -limit) & (y <= limit)).to(tl.float32)
    return dx, dy


@libentry()
@libtuner(
    configs=silu_clamp_get_configs(linear=True),
    prune_configs_by={"early_config_prune": silu_clamp_early_prune},
    key=["n_elements", "dtype"],
    strategy="default",
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_with_clamp_linear_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    limit,
    n_elements: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    UNMASKED: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if UNMASKED and n_elements % BLOCK_SIZE == 0:
        mask = tl.full((BLOCK_SIZE,), True, tl.int1)
    else:
        mask = offsets < n_elements
    xo = offsets
    yo = offsets
    oo = offsets
    x = tl.load(x_ptr + xo, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + yo, mask=mask, other=0).to(tl.float32)
    result = _silu_clamp_value(x, y, limit)
    tl.store(out_ptr + oo, result, mask=mask)


@libentry()
@libtuner(
    configs=silu_clamp_get_configs(linear=False),
    prune_configs_by={"early_config_prune": silu_clamp_full_early_prune},
    key=["n_elements", "dtype"],
    strategy="default",
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_with_clamp_linear_full_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    limit,
    n_elements: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets).to(tl.float32)
    y = tl.load(y_ptr + offsets).to(tl.float32)
    result = _silu_clamp_value(x, y, limit)
    tl.store(out_ptr + offsets, result)


@libentry()
@libtuner(
    configs=silu_clamp_get_configs(linear=False),
    prune_configs_by={"early_config_prune": silu_clamp_early_prune},
    key=["n_elements", "dtype", "n_cols", "X_STRIDES", "Y_STRIDES", "OUT_STRIDES"],
    strategy="default",
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_with_clamp_2d_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    limit,
    n_elements,
    dtype: tl.constexpr,
    n_cols: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    row = (offsets // n_cols).to(tl.int64)
    col = offsets % n_cols
    xo = row * X_STRIDES[0] + col * X_STRIDES[1]
    yo = row * Y_STRIDES[0] + col * Y_STRIDES[1]
    oo = row * OUT_STRIDES[0] + col * OUT_STRIDES[1]
    x = tl.load(x_ptr + xo, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + yo, mask=mask, other=0).to(tl.float32)
    result = _silu_clamp_value(x, y, limit)
    tl.store(out_ptr + oo, result, mask=mask)


@libentry()
@libtuner(
    configs=silu_clamp_get_configs(linear=False),
    prune_configs_by={"early_config_prune": silu_clamp_early_prune},
    key=["n_elements", "dtype", "SHAPE", "X_STRIDES", "Y_STRIDES", "OUT_STRIDES"],
    strategy="default",
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_with_clamp_nd_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    limit,
    n_elements,
    dtype: tl.constexpr,
    SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    xo = _strided_offset(offsets, SHAPE, X_STRIDES)
    yo = _strided_offset(offsets, SHAPE, Y_STRIDES)
    oo = _strided_offset(offsets, SHAPE, OUT_STRIDES)
    x = tl.load(x_ptr + xo, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + yo, mask=mask, other=0).to(tl.float32)
    result = _silu_clamp_value(x, y, limit)
    tl.store(out_ptr + oo, result, mask=mask)


@libentry()
@libtuner(
    configs=silu_clamp_get_configs(linear=True),
    prune_configs_by={"early_config_prune": silu_clamp_early_prune},
    key=["n_elements", "dtype", "NEED_DX", "NEED_DY"],
    strategy="default",
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_with_clamp_backward_linear_kernel(
    x_ptr,
    y_ptr,
    grad_ptr,
    dx_ptr,
    dy_ptr,
    limit,
    n_elements: tl.constexpr,
    dtype: tl.constexpr,
    NEED_DX: tl.constexpr,
    NEED_DY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    UNMASKED: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if UNMASKED and n_elements % BLOCK_SIZE == 0:
        mask = tl.full((BLOCK_SIZE,), True, tl.int1)
    else:
        mask = offsets < n_elements
    xo = offsets
    yo = offsets
    go = offsets
    x = tl.load(x_ptr + xo, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + yo, mask=mask, other=0).to(tl.float32)
    grad = tl.load(grad_ptr + go, mask=mask, other=0).to(tl.float32)
    dx, dy = _silu_clamp_grad_values(x, y, grad, limit, NEED_DX, NEED_DY)
    if NEED_DX:
        tl.store(dx_ptr + offsets, dx, mask=mask)
    if NEED_DY:
        tl.store(dy_ptr + offsets, dy, mask=mask)


@libentry()
@libtuner(
    configs=silu_clamp_get_configs(linear=False),
    prune_configs_by={"early_config_prune": silu_clamp_early_prune},
    key=[
        "n_elements",
        "dtype",
        "n_cols",
        "X_STRIDES",
        "Y_STRIDES",
        "G_STRIDES",
        "NEED_DX",
        "NEED_DY",
    ],
    strategy="default",
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_with_clamp_backward_2d_kernel(
    x_ptr,
    y_ptr,
    grad_ptr,
    dx_ptr,
    dy_ptr,
    limit,
    n_elements,
    dtype: tl.constexpr,
    n_cols: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    G_STRIDES: tl.constexpr,
    NEED_DX: tl.constexpr,
    NEED_DY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    row = (offsets // n_cols).to(tl.int64)
    col = offsets % n_cols
    xo = row * X_STRIDES[0] + col * X_STRIDES[1]
    yo = row * Y_STRIDES[0] + col * Y_STRIDES[1]
    go = row * G_STRIDES[0] + col * G_STRIDES[1]
    x = tl.load(x_ptr + xo, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + yo, mask=mask, other=0).to(tl.float32)
    grad = tl.load(grad_ptr + go, mask=mask, other=0).to(tl.float32)
    dx, dy = _silu_clamp_grad_values(x, y, grad, limit, NEED_DX, NEED_DY)
    if NEED_DX:
        tl.store(dx_ptr + offsets, dx, mask=mask)
    if NEED_DY:
        tl.store(dy_ptr + offsets, dy, mask=mask)


@libentry()
@libtuner(
    configs=silu_clamp_get_configs(linear=False),
    prune_configs_by={"early_config_prune": silu_clamp_early_prune},
    key=[
        "n_elements",
        "dtype",
        "SHAPE",
        "X_STRIDES",
        "Y_STRIDES",
        "G_STRIDES",
        "NEED_DX",
        "NEED_DY",
    ],
    strategy="default",
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_with_clamp_backward_nd_kernel(
    x_ptr,
    y_ptr,
    grad_ptr,
    dx_ptr,
    dy_ptr,
    limit,
    n_elements,
    dtype: tl.constexpr,
    SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    G_STRIDES: tl.constexpr,
    NEED_DX: tl.constexpr,
    NEED_DY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    xo = _strided_offset(offsets, SHAPE, X_STRIDES)
    yo = _strided_offset(offsets, SHAPE, Y_STRIDES)
    go = _strided_offset(offsets, SHAPE, G_STRIDES)
    x = tl.load(x_ptr + xo, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + yo, mask=mask, other=0).to(tl.float32)
    grad = tl.load(grad_ptr + go, mask=mask, other=0).to(tl.float32)
    dx, dy = _silu_clamp_grad_values(x, y, grad, limit, NEED_DX, NEED_DY)
    if NEED_DX:
        tl.store(dx_ptr + offsets, dx, mask=mask)
    if NEED_DY:
        tl.store(dy_ptr + offsets, dy, mask=mask)


def _launch_forward(x, y, out, limit, shape, strides, linear):
    n = out.numel()
    args = (x, y, out, limit, n, (x.dtype, y.dtype, out.dtype))
    if linear:
        kernel = (
            silu_and_mul_with_clamp_linear_full_kernel
            if _use_full_linear_kernel(n)
            else silu_and_mul_with_clamp_linear_kernel
        )
    elif len(shape) == 2:
        kernel = silu_and_mul_with_clamp_2d_kernel
        args += (shape[1], *strides)
    else:
        kernel = silu_and_mul_with_clamp_nd_kernel
        args += (shape, *strides)
    with torch_device_fn.device(x.device):
        kernel[lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)](*args)


def _launch_backward(x, y, grad, dx, dy, limit, shape, strides, linear):
    n = grad.numel()
    dtype = dx.dtype if dx is not None else dy.dtype
    args = (x, y, grad, dx, dy, limit, n, (x.dtype, y.dtype, grad.dtype, dtype))
    if linear:
        kernel = silu_and_mul_with_clamp_backward_linear_kernel
    elif len(shape) == 2:
        kernel = silu_and_mul_with_clamp_backward_2d_kernel
        args += (shape[1], *strides)
    else:
        kernel = silu_and_mul_with_clamp_backward_nd_kernel
        args += (shape, *strides)
    args += (dx is not None, dy is not None)
    with torch_device_fn.device(x.device):
        kernel[lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)](*args)


@lru_cache(maxsize=128)
def _rounded_limit(limit, dtype):
    # CPU-only: preserve Python -> input dtype -> FP32 without GPU synchronization.
    return torch.tensor(limit, dtype=dtype, device="cpu").item()


def _normalize_limit(x, limit):
    if not isinstance(limit, Real):
        raise TypeError("silu_and_mul_with_clamp expects a real scalar limit")
    if x.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return _rounded_limit(limit, x.dtype)
    return limit


def _result_dtype(*args):
    tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
    if (
        all(t.dtype == tensors[0].dtype for t in tensors)
        and tensors[0].is_floating_point()
    ):
        return tensors[0].dtype
    return type_promotion(
        *args, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )[1]


def _broadcast_strides(tensor, shape):
    padding = len(shape) - tensor.ndim
    return (0,) * padding + tuple(
        0 if size == 1 else stride
        for size, stride in zip(tensor.shape, tensor.stride())
    )


def _coalesce_layout(shape, strides):
    # Merge only adjacent logical dimensions that are contiguous in *every*
    # operand. Zero strides merge only with zero strides; packed rows keep gaps.
    sizes = []
    merged = [[] for _ in strides]
    for dim, size in enumerate(shape):
        if size == 1:
            continue
        if sizes and all(
            row[-1] == size * stride[dim] for row, stride in zip(merged, strides)
        ):
            sizes[-1] *= size
            for row, stride in zip(merged, strides):
                row[-1] = stride[dim]
        else:
            sizes.append(size)
            for row, stride in zip(merged, strides):
                row.append(stride[dim])
    if not sizes:
        return (1,), tuple((0,) for _ in strides)
    return tuple(sizes), tuple(tuple(row) for row in merged)


def _layout(shape, tensors, allow_dense=False):
    if all(t.shape == shape and t.is_contiguous() for t in tensors):
        # Canonicalize metadata so equal-numel contiguous shapes share tuning.
        return (tensors[0].numel(),), ((1,),) * len(tensors), True
    if (
        allow_dense
        and all(t.shape == shape and t.stride() == tensors[0].stride() for t in tensors)
        and torch.ops.aten.is_non_overlapping_and_dense(tensors[0])
    ):
        # Matching dense transposes/channels-last tensors have the same physical
        # element order. Flatten their storage and preserve that layout in outputs.
        return (tensors[0].numel(),), ((1,),) * len(tensors), True
    task_shape, strides = _coalesce_layout(
        shape, tuple(_broadcast_strides(t, shape) for t in tensors)
    )
    return task_shape, strides, False


def _forward(x, y, limit, out=None):
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    shape = x.shape if x.shape == y.shape else torch.broadcast_shapes(x.shape, y.shape)
    if out is None:
        dtype = _result_dtype(x, y, limit)
        out = (
            torch.empty_like(x, dtype=dtype)
            if x.shape == shape
            else torch.empty(shape, dtype=dtype, device=x.device)
        )
    elif out.device != x.device or out.shape != shape:
        raise ValueError("out must have the broadcast output shape and input device")
    if out.numel() == 0:
        return out
    # Autotuning launches repeatedly: an output sharing input storage must not
    # mutate those inputs between candidate measurements (including partial overlap).
    if torch._C._overlaps(out, x) or torch._C._overlaps(out, y):
        temporary = torch.empty(shape, dtype=out.dtype, device=out.device)
        _forward(x, y, limit, temporary)
        out.copy_(temporary)
        return out
    task_shape, strides, linear = _layout(shape, (x, y, out), allow_dense=True)
    _launch_forward(x, y, out, limit, task_shape, strides, linear)
    return out


class SiluAndMulWithClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, limit):
        ctx.save_for_backward(x, y)
        ctx.limit = _normalize_limit(x, limit)
        logger.debug("GEMS SILU_AND_MUL_WITH_CLAMP_FORWARD")
        return _forward(x, y, ctx.limit)

    @staticmethod
    def backward(ctx, dgrad):
        x, y = ctx.saved_tensors
        need_dx, need_dy = ctx.needs_input_grad[:2]
        logger.debug("GEMS SILU_AND_MUL_WITH_CLAMP_BACKWARD")
        dtype = _result_dtype(x, y, dgrad, ctx.limit)
        shape, strides, linear = _layout(dgrad.shape, (x, y, dgrad), allow_dense=True)

        def allocate_gradient():
            if linear:
                return torch.empty_like(dgrad, dtype=dtype)
            return torch.empty(dgrad.shape, dtype=dtype, device=x.device)

        dx = allocate_gradient() if need_dx else None
        dy = allocate_gradient() if need_dy else None
        n = dgrad.numel()
        if n:
            _launch_backward(x, y, dgrad, dx, dy, ctx.limit, shape, strides, linear)
        # Autograd reduces broadcast dimensions to the original input shapes.
        return dx, dy, None


def silu_and_mul_with_clamp(x, y, limit):
    if not torch.is_grad_enabled() or not (x.requires_grad or y.requires_grad):
        logger.debug("GEMS SILU_AND_MUL_WITH_CLAMP_INFERENCE_FORWARD")
        return _forward(x, y, _normalize_limit(x, limit))
    return SiluAndMulWithClamp.apply(x, y, limit)


def silu_and_mul_with_clamp_out(x, y, out, limit):
    logger.debug("GEMS SILU_AND_MUL_WITH_CLAMP_OUT")
    return _forward(x, y, _normalize_limit(x, limit), out)
