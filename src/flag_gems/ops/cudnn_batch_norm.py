# Copyright 2026, The FlagOS Contributors.
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
#
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)
_MIN_EPSILON = 1e-5
_STAT_BLOCK = 1024
_ELEMENT_BLOCK = 1024


def _spatial_stride(input):
    # Contiguity ignores arbitrary strides on singleton dimensions.
    return next(
        (input.stride(i) for i in range(input.ndim - 1, 1, -1) if input.shape[i] > 1), 1
    )


@libentry()
@triton.jit
def _bn_fused(
    X,
    W,
    Bias,
    RM,
    RV,
    Y,
    Mean,
    Inv,
    M: tl.constexpr,
    S: tl.constexpr,
    NS: tl.constexpr,
    CS: tl.constexpr,
    SS: tl.constexpr,
    UPDATE: tl.constexpr,
    FACTOR: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    DOUBLE: tl.constexpr,
):
    c = tl.program_id(0)
    r = tl.arange(0, B)
    offsets = r // S * NS + c * CS + r % S * SS
    acc: tl.constexpr = tl.float64 if DOUBLE else tl.float32
    x = tl.load(X + offsets, r < M, 0).to(acc)
    anchor = tl.load(X + c * CS).to(acc)
    anchor = tl.where(tl.abs(anchor) < float("inf"), anchor, 0)
    centered = tl.where(r < M, x - anchor, 0)
    shift = tl.sum(centered, 0) / M
    delta = tl.where(r < M, centered - shift, 0)
    var = tl.sum(delta * delta, 0) / M
    inv = tl_extra_shim.rsqrt(var + EPS)
    mean = anchor + shift
    tl.store(Mean + c, mean)
    tl.store(Inv + c, inv)
    if UPDATE:
        old_mean = tl.load(RM + c).to(acc)
        old_var = tl.load(RV + c).to(acc)
        unbiased = var * M / (M - 1) if M > 1 else float("nan")
        tl.store(RM + c, (1 - FACTOR) * old_mean + FACTOR * mean)
        tl.store(RV + c, (1 - FACTOR) * old_var + FACTOR * unbiased)
    w = tl.load(W + c).to(acc)
    bias = tl.load(Bias + c).to(acc)
    tl.store(Y + offsets, delta * inv * w + bias, r < M)


@libentry()
@triton.jit
def _bn_partial(
    X,
    PMean,
    PM2,
    M: tl.constexpr,
    S: tl.constexpr,
    NS: tl.constexpr,
    CS: tl.constexpr,
    SS: tl.constexpr,
    PARTS: tl.constexpr,
    B: tl.constexpr,
    DOUBLE: tl.constexpr,
):
    c = tl.program_id(0)
    p = tl.program_id(1)
    r = p * B + tl.arange(0, B)
    acc: tl.constexpr = tl.float64 if DOUBLE else tl.float32
    x = tl.load(X + r // S * NS + c * CS + r % S * SS, r < M, 0).to(acc)
    anchor = tl.load(X + c * CS).to(acc)
    anchor = tl.where(tl.abs(anchor) < float("inf"), anchor, 0)
    x = tl.where(r < M, x - anchor, 0)
    count = tl.minimum(B, M - p * B)
    mean = tl.sum(x, 0) / count
    delta = tl.where(r < M, x - mean, 0)
    m2 = tl.sum(delta * delta, 0)
    tl.store(PMean + c * PARTS + p, mean)
    tl.store(PM2 + c * PARTS + p, m2)


@libentry()
@triton.jit
def _bn_merge(
    X,
    MeanDelta,
    CS: tl.constexpr,
    PMean,
    PM2,
    Mean,
    Inv,
    RM,
    RV,
    M: tl.constexpr,
    PARTS: tl.constexpr,
    STAT_B: tl.constexpr,
    B: tl.constexpr,
    UPDATE: tl.constexpr,
    FACTOR: tl.constexpr,
    EPS: tl.constexpr,
    DOUBLE: tl.constexpr,
):
    c = tl.program_id(0)
    p = tl.arange(0, B)
    acc: tl.constexpr = tl.float64 if DOUBLE else tl.float32
    means = tl.load(PMean + c * PARTS + p, p < PARTS, 0).to(acc)
    m2 = tl.load(PM2 + c * PARTS + p, p < PARTS, 0).to(acc)
    counts = tl.where(p < PARTS, tl.minimum(STAT_B, M - p * STAT_B), 0)
    mean = tl.sum(means * counts, 0) / M
    delta = means - mean
    var = tl.sum(tl.where(p < PARTS, m2 + counts * delta * delta, 0), 0) / M
    inv = tl_extra_shim.rsqrt(var + EPS)
    tl.store(MeanDelta + c, mean)
    anchor = tl.load(X + c * CS).to(acc)
    anchor = tl.where(tl.abs(anchor) < float("inf"), anchor, 0)
    mean += anchor
    tl.store(Mean + c, mean)
    tl.store(Inv + c, inv)
    if UPDATE:
        old_mean = tl.load(RM + c).to(acc)
        old_var = tl.load(RV + c).to(acc)
        tl.store(RM + c, (1 - FACTOR) * old_mean + FACTOR * mean)
        unbiased = var * M / (M - 1) if M > 1 else float("nan")
        tl.store(RV + c, (1 - FACTOR) * old_var + FACTOR * unbiased)


@libentry()
@triton.jit
def _bn_apply(
    X,
    W,
    Bias,
    Mean,
    Stat,
    Y,
    SIZE: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    TRAIN: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    DOUBLE: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    c = i % C if CHANNELS_LAST else i // S % C
    mask = i < SIZE
    acc: tl.constexpr = tl.float64 if DOUBLE else tl.float32
    x = tl.load(X + i, mask, 0).to(acc)
    w = tl.load(W + c, mask, 0).to(acc)
    bias = tl.load(Bias + c, mask, 0).to(acc)
    mean = tl.load(Mean + c, mask, 0).to(acc)
    stat = tl.load(Stat + c, mask, 0).to(acc)
    if TRAIN:
        anchor = tl.load(X + c * (1 if CHANNELS_LAST else S), mask, 0).to(acc)
        anchor = tl.where(tl.abs(anchor) < float("inf"), anchor, 0)
        x -= anchor
    inv = stat if TRAIN else tl_extra_shim.rsqrt(stat + EPS)
    tl.store(Y + i, (x - mean) * inv * w + bias, mask)


@libentry()
@triton.jit
def _bn_eval_fp64(
    X,
    W,
    Bias,
    RM,
    RV,
    Y,
    M: tl.constexpr,
    S: tl.constexpr,
    NS: tl.constexpr,
    CS: tl.constexpr,
    SS: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
):
    c = tl.program_id(0)
    r = tl.program_id(1) * B + tl.arange(0, B)
    offset = r // S * NS + c * CS + r % S * SS
    # Each program shares one channel's expensive FP64 reciprocal square root.
    mean = tl.load(RM + c)
    inv = tl_extra_shim.rsqrt(tl.load(RV + c) + EPS)
    w = tl.load(W + c)
    bias = tl.load(Bias + c)
    x = tl.load(X + offset, r < M, 0)
    tl.store(Y + offset, (x - mean) * inv * w + bias, r < M)


@libentry()
@triton.jit
def _bn_eval_backward(
    X,
    DY,
    W,
    RM,
    RV,
    DX,
    DW,
    DB,
    M: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    NS: tl.constexpr,
    CS: tl.constexpr,
    SS: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    DOUBLE: tl.constexpr,
):
    c = tl.program_id(0)
    acc: tl.constexpr = tl.float64 if DOUBLE else tl.float32
    mean = tl.load(RM + c).to(acc)
    inv = tl_extra_shim.rsqrt(tl.load(RV + c).to(acc) + EPS)
    w = tl.load(W + c).to(acc)
    dw = tl.full((B,), 0, acc)
    db = tl.full((B,), 0, acc)
    for start in range(tl.cdiv(M, B)):
        r = start * B + tl.arange(0, B)
        offsets = r // S * NS + c * CS + r % S * SS
        x = tl.load(X + offsets, r < M, 0).to(acc)
        # DY is made contiguous; DX retains X's memory format.
        dy = tl.load(DY + r // S * C * S + c * S + r % S, r < M, 0).to(acc)
        norm = (x - mean) * inv
        dw += tl.where(r < M, dy * norm, 0)
        db += dy
        tl.store(DX + offsets, dy * w * inv, r < M)
    tl.store(DW + c, tl.sum(dw, 0))
    tl.store(DB + c, tl.sum(db, 0))


def _validate(input, weight, bias, running_mean, running_var, training, epsilon):
    if input.ndim < 2 or input.ndim > 5:
        raise RuntimeError("cudnn_batch_norm expects input rank 2 through 5")
    if input.dtype not in (torch.float16, torch.float32, torch.float64):
        raise RuntimeError("cudnn_batch_norm supports float16, float32 and float64")
    if input.device.type == "cpu":
        raise RuntimeError("cudnn_batch_norm requires an accelerator tensor")
    if epsilon < _MIN_EPSILON or not math.isfinite(epsilon):
        raise RuntimeError("cudnn_batch_norm requires finite epsilon >= 1e-5")
    if bias is None or weight is None:
        raise RuntimeError("cudnn_batch_norm requires weight and bias")
    if (running_mean is None) != (running_var is None):
        raise RuntimeError("running_mean and running_var must be provided together")
    if not training and running_mean is None:
        raise RuntimeError("inference requires running_mean and running_var")
    fmt = torch.contiguous_format
    if input.ndim == 4 and input.is_contiguous(memory_format=torch.channels_last):
        fmt = torch.channels_last
    elif input.ndim == 5 and input.is_contiguous(memory_format=torch.channels_last_3d):
        fmt = torch.channels_last_3d
    if not input.is_contiguous(memory_format=fmt):
        raise RuntimeError("input must be contiguous in its memory format")
    dtype = torch.float32 if input.dtype == torch.float16 else input.dtype
    for name, t in (
        ("weight", weight),
        ("bias", bias),
        ("running_mean", running_mean),
        ("running_var", running_var),
    ):
        if t is not None and (
            t.device != input.device
            or t.dtype != dtype
            or t.numel() != input.shape[1]
            or not t.is_contiguous()
        ):
            raise RuntimeError(
                f"invalid {name}: expected contiguous C elements with matching device and parameter dtype"
            )
    if input.numel() == 0:
        raise RuntimeError("cudnn_batch_norm does not accept empty input")
    return fmt


def _forward(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    exponential_average_factor,
    epsilon,
    output,
    mean,
    inv,
):
    n, c = input.shape[:2]
    s = math.prod(input.shape[2:])
    m = n * s
    double = input.dtype == torch.float64
    channels_last = input.stride(1) == 1 and not input.is_contiguous()
    with torch_device_fn.device(input.device):
        if not training and double and not channels_last:
            block = min(triton.next_power_of_2(m), _ELEMENT_BLOCK)
            _bn_eval_fp64[(c, triton.cdiv(m, block))](
                input,
                weight,
                bias,
                running_mean,
                running_var,
                output,
                m,
                s,
                input.stride(0),
                input.stride(1),
                _spatial_stride(input),
                epsilon,
                block,
            )
            return
        if training and m <= 2048:
            _bn_fused[(c,)](
                input,
                weight,
                bias,
                running_mean,
                running_var,
                output,
                mean,
                inv,
                m,
                s,
                input.stride(0),
                input.stride(1),
                _spatial_stride(input),
                running_mean is not None,
                exponential_average_factor,
                epsilon,
                triton.next_power_of_2(m),
                double,
            )
            return
        if training:
            parts = triton.cdiv(m, _STAT_BLOCK)
            partial_mean = torch.empty(
                (c, parts), dtype=weight.dtype, device=input.device
            )
            partial_m2 = torch.empty_like(partial_mean)
            mean_delta = torch.empty_like(mean)
            _bn_partial[(c, parts)](
                input,
                partial_mean,
                partial_m2,
                m,
                s,
                input.stride(0),
                input.stride(1),
                _spatial_stride(input),
                parts,
                _STAT_BLOCK,
                double,
            )
            _bn_merge[(c,)](
                input,
                mean_delta,
                input.stride(1),
                partial_mean,
                partial_m2,
                mean,
                inv,
                running_mean,
                running_var,
                m,
                parts,
                _STAT_BLOCK,
                triton.next_power_of_2(parts),
                running_mean is not None,
                exponential_average_factor,
                epsilon,
                double,
            )
        _bn_apply[(triton.cdiv(input.numel(), _ELEMENT_BLOCK),)](
            input,
            weight,
            bias,
            mean_delta if training else running_mean,
            inv if training else running_var,
            output,
            input.numel(),
            c,
            s,
            channels_last,
            training,
            epsilon,
            _ELEMENT_BLOCK,
            double,
        )


def _allocate_forward(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    exponential_average_factor,
    epsilon,
    forward_fn=_forward,
):
    _validate(input, weight, bias, running_mean, running_var, training, epsilon)
    output = torch.empty_strided(
        input.shape, input.stride(), dtype=input.dtype, device=input.device
    )
    mean = torch.empty(
        (input.shape[1] if training else 0,),
        dtype=weight.dtype,
        device=input.device,
    )
    inv = torch.empty_like(mean)
    reserve = torch.empty((0,), dtype=torch.uint8, device=input.device)
    forward_fn(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
        output,
        mean,
        inv,
    )
    return output, mean, inv, reserve


class _CudnnBatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
        forward_fn,
        backward_fn,
    ):
        output, mean, inv, reserve = _allocate_forward(
            input,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            exponential_average_factor,
            epsilon,
            forward_fn,
        )
        ctx.save_for_backward(
            input,
            weight,
            mean if training else running_mean,
            inv if training else running_var,
            reserve,
        )
        ctx.backward_fn = backward_fn
        ctx.training = training
        ctx.epsilon = epsilon
        ctx.bias_shape = bias.shape
        ctx.mark_non_differentiable(mean, inv, reserve)
        return output, mean, inv, reserve

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output, grad_mean, grad_inv, grad_reserve):
        # Call the paired implementation directly: registration may already be gone.
        input, weight, mean, inv, reserve = ctx.saved_tensors
        if ctx.training:
            dx, dw, db = ctx.backward_fn(
                input, grad_output, weight, None, None, mean, inv, ctx.epsilon, reserve
            )
        else:
            dx = torch.empty_strided(
                input.shape, input.stride(), dtype=input.dtype, device=input.device
            )
            dw, db = torch.empty_like(weight), torch.empty_like(weight)
            s = math.prod(input.shape[2:])
            m = input.shape[0] * s
            with torch_device_fn.device(input.device):
                _bn_eval_backward[(input.shape[1],)](
                    input,
                    grad_output.contiguous(),
                    weight,
                    mean,
                    inv,
                    dx,
                    dw,
                    db,
                    m,
                    input.shape[1],
                    s,
                    input.stride(0),
                    input.stride(1),
                    _spatial_stride(input),
                    ctx.epsilon,
                    min(triton.next_power_of_2(m), 1024),
                    input.dtype == torch.float64,
                )
        return (
            dx,
            dw.reshape(weight.shape),
            db.reshape(ctx.bias_shape),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _cudnn_batch_norm_impl(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    exponential_average_factor,
    epsilon,
    forward_fn=_forward,
    backward_fn=None,
):
    """BatchNorm with Gems-owned saved statistics and an empty byte reserve.

    The reserve is only valid with the paired Gems backward. Autograd retains
    that implementation even after temporary dispatcher registration ends.
    """
    args = (
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
    )
    if torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in args[:5]
    ):
        if backward_fn is None:
            from flag_gems.ops.cudnn_batch_norm_backward import (
                cudnn_batch_norm_backward,
            )

            backward_fn = cudnn_batch_norm_backward
        return _CudnnBatchNorm.apply(*args, forward_fn, backward_fn)
    return _allocate_forward(*args, forward_fn)


def cudnn_batch_norm(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    exponential_average_factor,
    epsilon,
):
    logger.debug("GEMS CUDNN_BATCH_NORM")
    return _cudnn_batch_norm_impl(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
    )


def _cudnn_batch_norm_out_impl(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    exponential_average_factor,
    epsilon,
    *,
    out0,
    out1,
    out2,
    out3,
    forward_fn=_forward,
):
    fmt = _validate(input, weight, bias, running_mean, running_var, training, epsilon)
    if torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (
            input,
            weight,
            bias,
            running_mean,
            running_var,
            out0,
            out1,
            out2,
            out3,
        )
    ):
        raise RuntimeError("out= functions do not support automatic differentiation")
    reserve = torch.empty((0,), dtype=torch.uint8, device=input.device)
    if out0.device != input.device or out0.dtype != input.dtype:
        raise RuntimeError("output dtype and device must match the result")
    if out0.shape != input.shape or not out0.is_contiguous(memory_format=fmt):
        raise RuntimeError("out0 must have the input shape and memory format")
    if training:
        for out in (out1, out2):
            if (
                out.device != input.device
                or out.dtype != weight.dtype
                or out.numel() != input.shape[1]
                or not out.is_contiguous()
            ):
                raise RuntimeError(
                    "saved outputs must be contiguous C-element tensors with the parameter dtype"
                )
    outputs = (out0, out1, out2, reserve)
    inputs = (input, weight, bias, running_mean, running_var)
    # Inference leaves the caller's saved-stat placeholders unchanged.
    written_outputs = outputs[:3] if training else outputs[:1]
    for i, out in enumerate(written_outputs):
        if out.numel() and any(
            t is not None and t.numel() and torch._C._overlaps(out, t)
            for t in inputs + written_outputs[:i]
        ):
            raise RuntimeError("overlapping output storage is unsupported")
    forward_fn(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
        out0,
        out1,
        out2,
    )
    return outputs


def cudnn_batch_norm_out(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    exponential_average_factor,
    epsilon,
    *,
    out0,
    out1,
    out2,
    out3,
):
    logger.debug("GEMS CUDNN_BATCH_NORM_OUT")
    return _cudnn_batch_norm_out_impl(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
        out0=out0,
        out1=out1,
        out2=out2,
        out3=out3,
    )
