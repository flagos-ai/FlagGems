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

import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext


@triton.jit
def _prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.jit(do_not_specialize=["eps"])
def post_layer_norm_residual_one_pass_kernel(
    input_ptr,
    residual_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    M,
    N,
    eps,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    SAVE_STATS: tl.constexpr,
):
    pid = ext.program_id(0)
    row_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    row_mask = row_offsets < M
    col_offsets = tl.arange(0, TILE_N)[None, :]
    col_mask = col_offsets < N
    mask = row_mask[:, None] & col_mask
    offsets = row_offsets[:, None] * N + col_offsets

    input = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(input, axis=1) / N
    centered = tl.where(mask, input - mean[:, None], 0.0)
    variance = tl.sum(centered * centered, axis=1) / N
    rstd = tl.math.rsqrt(variance + eps)

    if SAVE_STATS:
        tl.store(mean_ptr + row_offsets, mean, mask=row_mask)
        tl.store(rstd_ptr + row_offsets, rstd, mask=row_mask)

    if weight_ptr is None:
        weight = 1.0
    else:
        weight = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.0)
    if bias_ptr is None:
        bias = 0.0
    else:
        bias = tl.load(bias_ptr + col_offsets, mask=col_mask, other=0.0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    output = centered * rstd[:, None] * weight + bias + residual
    tl.store(output_ptr + offsets, output, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_loop"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def post_layer_norm_residual_loop_kernel(
    input_ptr,
    residual_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    M,
    N,
    eps,
    SAVE_STATS: tl.constexpr,
    TILE_N: tl.constexpr,
):
    pid = ext.program_id(0)

    partial_mean = tl.zeros((TILE_N,), dtype=tl.float32)
    partial_m2 = tl.zeros((TILE_N,), dtype=tl.float32)
    count = tl.zeros((TILE_N,), dtype=tl.int32)
    num_steps = tl.cdiv(N, TILE_N)
    for step in range(0, num_steps - 1):
        col_offsets = step * TILE_N + tl.arange(0, TILE_N)
        input = tl.load(input_ptr + pid * N + col_offsets).to(tl.float32)
        new_mean = partial_mean + (input - partial_mean) / (step + 1)
        partial_m2 += (input - new_mean) * (input - partial_mean)
        partial_mean = new_mean
        count += 1

    for step in range(num_steps - 1, num_steps):
        col_offsets = step * TILE_N + tl.arange(0, TILE_N)
        mask = col_offsets < N
        input = tl.load(input_ptr + pid * N + col_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        new_mean = tl.where(
            mask, partial_mean + (input - partial_mean) / (step + 1), partial_mean
        )
        partial_m2 = tl.where(
            mask,
            partial_m2 + (input - new_mean) * (input - partial_mean),
            partial_m2,
        )
        partial_mean = new_mean
        count += mask.to(tl.int32)

    mean = tl.sum(partial_mean * count) / N
    variance = (
        tl.sum(partial_m2 + count * (partial_mean - mean) * (partial_mean - mean)) / N
    )
    rstd = tl.math.rsqrt(variance + eps)
    if SAVE_STATS:
        tl.store(mean_ptr + pid, mean)
        tl.store(rstd_ptr + pid, rstd)

    previous_multiple = _prev_multiple_of(N, TILE_N)
    for start_n in range(0, TILE_N, TILE_N):
        col_offsets = previous_multiple - start_n + tl.arange(0, TILE_N)
        mask = col_offsets < N
        input = tl.load(
            input_ptr + pid * N + col_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + pid * N + col_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        if weight_ptr is None:
            weight = 1.0
        else:
            weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
        if bias_ptr is None:
            bias = 0.0
        else:
            bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
        output = weight * (input - mean) * rstd + bias + residual
        tl.store(output_ptr + pid * N + col_offsets, output, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        col_offsets = previous_multiple - start_n + tl.arange(0, TILE_N)
        input = tl.load(
            input_ptr + pid * N + col_offsets, eviction_policy="evict_first"
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + pid * N + col_offsets, eviction_policy="evict_first"
        ).to(tl.float32)
        if weight_ptr is None:
            weight = 1.0
        else:
            weight = tl.load(weight_ptr + col_offsets)
        if bias_ptr is None:
            bias = 0.0
        else:
            bias = tl.load(bias_ptr + col_offsets)
        output = weight * (input - mean) * rstd + bias + residual
        tl.store(output_ptr + pid * N + col_offsets, output)


def _normalize_shape(normalized_shape):
    if isinstance(normalized_shape, int):
        return (normalized_shape,)
    try:
        return tuple(normalized_shape)
    except TypeError as error:
        raise TypeError("normalized_shape must be an int or a sequence") from error


def _validate_inputs(input, residual, normalized_shape, weight, bias):
    if not isinstance(input, torch.Tensor) or not isinstance(residual, torch.Tensor):
        raise TypeError("input and residual must be tensors")
    if input.shape != residual.shape:
        raise ValueError("input and residual must have the same shape")
    if input.dtype != residual.dtype:
        raise ValueError("input and residual must have the same dtype")
    if input.device != residual.device:
        raise ValueError("input and residual must be on the same device")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("input dtype must be float16, bfloat16, or float32")
    if not normalized_shape or any(size <= 0 for size in normalized_shape):
        raise ValueError("normalized_shape must contain positive dimensions")
    if len(normalized_shape) > input.ndim:
        raise ValueError("normalized_shape must be a suffix of input.shape")
    if tuple(input.shape[-len(normalized_shape) :]) != normalized_shape:
        raise ValueError("normalized_shape must match the trailing input dimensions")

    for name, tensor in (("weight", weight), ("bias", bias)):
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor or None")
        if tuple(tensor.shape) != normalized_shape:
            raise ValueError(f"{name} shape must match normalized_shape")
        if tensor.dtype != input.dtype:
            raise ValueError(f"{name} must have the same dtype as input")
        if tensor.device != input.device:
            raise ValueError(f"{name} must be on the same device as input")


class PostLayerNormResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, residual, normalized_shape, weight, bias, eps):
        N = math.prod(normalized_shape)
        M = input.numel() // N
        output = torch.empty_like(input)

        needs_layer_norm_grad = (
            input.requires_grad
            or (weight is not None and weight.requires_grad)
            or (bias is not None and bias.requires_grad)
        )
        stats_size = M if needs_layer_norm_grad else 1
        mean = torch.empty(stats_size, dtype=input.dtype, device=input.device)
        rstd = torch.empty_like(mean)

        with torch_device_fn.device(input.device):
            if N <= 4096:
                tile_n = triton.next_power_of_2(N)
                tile_m = triton.cdiv(1024, tile_n) if N <= 128 else 1
                grid = (triton.cdiv(M, tile_m), 1, 1)
                post_layer_norm_residual_one_pass_kernel[grid](
                    input,
                    residual,
                    output,
                    weight,
                    bias,
                    mean,
                    rstd,
                    M,
                    N,
                    eps,
                    TILE_M=tile_m,
                    TILE_N=tile_n,
                    SAVE_STATS=needs_layer_norm_grad,
                )
            else:
                post_layer_norm_residual_loop_kernel[(M, 1, 1)](
                    input,
                    residual,
                    output,
                    weight,
                    bias,
                    mean,
                    rstd,
                    M,
                    N,
                    eps,
                    SAVE_STATS=needs_layer_norm_grad,
                )

        if input.requires_grad or residual.requires_grad or needs_layer_norm_grad:
            weight_saved = input.new_empty(0) if weight is None else weight
            bias_saved = input.new_empty(0) if bias is None else bias
            ctx.save_for_backward(input, weight_saved, bias_saved, mean, rstd)
            ctx.has_weight = weight is not None
            ctx.has_bias = bias is not None
            ctx.input_shape = input.shape
            ctx.normalized_shape = normalized_shape
            ctx.M = M
            ctx.N = N
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "post_layer_norm_residual backward is not implemented"
        )


def post_layer_norm_residual(
    input, residual, normalized_shape, weight=None, bias=None, eps=1e-5
):
    normalized_shape = _normalize_shape(normalized_shape)
    _validate_inputs(input, residual, normalized_shape, weight, bias)

    if input.numel() == 0 or not input.is_contiguous() or not residual.is_contiguous():
        return torch.layer_norm(input, normalized_shape, weight, bias, eps) + residual

    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()
    return PostLayerNormResidual.apply(
        input, residual, normalized_shape, weight, bias, eps
    )
