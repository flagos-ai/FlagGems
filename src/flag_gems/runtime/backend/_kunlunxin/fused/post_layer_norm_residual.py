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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Alignment budget (in elements) for the 2D tile kernel: a 16-row tile of a
# 512-wide reduction is the empirical sweet spot on this backend (larger tiles
# neither raise bandwidth nor lower it, smaller ones are launch bound).
_TILE_BUDGET = 8192
_PREFERRED_TILE_M = 16
# When the row is split into a power-of-two tile plus an explicit 1D tail, the
# full output block (centered * rstd * weight + bias + residual) at 16 rows has
# been observed to corrupt the FP32 statistics on XPU (the reduce result is only
# exact up to 8 rows for this shape). Keep the tail path at 8 rows.
_TAIL_TILE_M = 8
# A 2D axis-1 reduction is only numerically exact when the reduction width is a
# multiple of 8, so the wide part of a row is always the largest power of two
# that fits; whatever is left is handled as an explicit 1D per-row tail.
_MIN_REDUCE_BASE = 8
_MAX_EXPLICIT_TAIL = 16
_BLOCK_PATH_MAX_N = 4096
# 1D per-row tile width. XPU 1D online reduction kernels are only reliable for
# tile widths <= 64 or exactly 1024.
_ROW_TILE_N = 1024


def _reduce_base_and_tail(n):
    """Split ``n`` into ``base + tail`` with ``base`` a power of two."""
    base = 1
    while base * 2 <= n:
        base *= 2
    return base, n - base


@libentry()
@triton.jit(do_not_specialize=["eps"])
def post_layer_norm_residual_block_kernel(
    input_ptr,
    residual_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    eps,
    N: tl.constexpr,
    BASE: tl.constexpr,
    REM: tl.constexpr,
    TILE_M: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
    SAVE_STATS: tl.constexpr,
):
    """Wide 2D tile of ``[TILE_M, BASE]`` plus an explicit per-row tail.

    The wide part is a fully affine, mask-free tile whose reduction width
    ``BASE`` is a power of two (hence a multiple of 8), which is the only
    shape for which the XPU 2D axis-1 reduction is exact. The remaining
    ``REM`` columns are processed as 1D per-row vectors so no masked-memory
    path is generated (masked 2D tiles are miscompiled by this backend).

    ``TILE_M`` always divides the row count, so no row mask is needed. Loading
    weight/bias with 2D column offsets is required: broadcasting a 1D load over
    the 2D tile is miscompiled.
    """
    pid = ext.program_id(0)
    rows = pid * TILE_M + tl.arange(0, TILE_M)
    row_base = rows * N
    cols = tl.arange(0, BASE)[None, :]
    offsets = row_base[:, None] + cols

    input = tl.load(input_ptr + offsets).to(tl.float32)
    total = tl.sum(input, axis=1)
    if REM:
        tail_sum = tl.zeros((TILE_M,), dtype=tl.float32)
        for j in tl.static_range(REM):
            tail_sum += tl.load(input_ptr + row_base + BASE + j).to(tl.float32)
        total += tail_sum
    mean = total / N

    centered = input - mean[:, None]
    m2 = tl.sum(centered * centered, axis=1)
    if REM:
        tail_m2 = tl.zeros((TILE_M,), dtype=tl.float32)
        for j in tl.static_range(REM):
            tail = tl.load(input_ptr + row_base + BASE + j).to(tl.float32) - mean
            tail_m2 += tail * tail
        m2 += tail_m2
    rstd = tl.math.rsqrt(m2 / N + eps)

    if SAVE_STATS:
        tl.store(mean_ptr + rows, mean)
        tl.store(rstd_ptr + rows, rstd)

    output = centered * rstd[:, None]
    if HAS_W:
        output = output * tl.load(weight_ptr + cols)
    if HAS_B:
        output = output + tl.load(bias_ptr + cols)
    output = output + tl.load(residual_ptr + offsets).to(tl.float32)
    tl.store(output_ptr + offsets, output)

    if REM:
        for j in tl.static_range(REM):
            tail = tl.load(input_ptr + row_base + BASE + j).to(tl.float32)
            residual = tl.load(residual_ptr + row_base + BASE + j).to(tl.float32)
            value = (tail - mean) * rstd
            if HAS_W:
                value = value * tl.load(weight_ptr + BASE + j).to(tl.float32)
            if HAS_B:
                value = value + tl.load(bias_ptr + BASE + j).to(tl.float32)
            tl.store(output_ptr + row_base + BASE + j, value + residual)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def post_layer_norm_residual_row_kernel(
    input_ptr,
    residual_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    eps,
    N: tl.constexpr,
    TILE_N: tl.constexpr,
    NEED_TAIL: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
    SAVE_STATS: tl.constexpr,
):
    """One program per row; works for any N (non power-of-two and N > 4096).

    When N is not a multiple of TILE_N every chunk uses the masked path: mixing
    unmasked full chunks with a masked tail in the same kernel is miscompiled by
    the XPU backend (the reduction reads the wrong lanes).
    """
    pid = ext.program_id(0)
    base = pid * N

    partial_sum = tl.zeros((TILE_N,), dtype=tl.float32)
    for offset in range(0, N, TILE_N):
        cols = offset + tl.arange(0, TILE_N)
        if NEED_TAIL:
            chunk_mask = cols < N
            chunk = tl.load(input_ptr + base + cols, mask=chunk_mask, other=0.0)
            partial_sum += tl.where(chunk_mask, chunk.to(tl.float32), 0.0)
        else:
            partial_sum += tl.load(input_ptr + base + cols).to(tl.float32)
    mean = tl.sum(partial_sum) / N

    partial_m2 = tl.zeros((TILE_N,), dtype=tl.float32)
    for offset in range(0, N, TILE_N):
        cols = offset + tl.arange(0, TILE_N)
        if NEED_TAIL:
            chunk_mask = cols < N
            chunk = tl.load(input_ptr + base + cols, mask=chunk_mask, other=0.0)
            centered = tl.where(chunk_mask, chunk.to(tl.float32) - mean, 0.0)
        else:
            centered = tl.load(input_ptr + base + cols).to(tl.float32) - mean
        partial_m2 += centered * centered
    rstd = tl.math.rsqrt(tl.sum(partial_m2) / N + eps)

    if SAVE_STATS:
        tl.store(mean_ptr + pid, mean)
        tl.store(rstd_ptr + pid, rstd)

    for offset in range(0, N, TILE_N):
        cols = offset + tl.arange(0, TILE_N)
        if NEED_TAIL:
            chunk_mask = cols < N
            input = tl.load(input_ptr + base + cols, mask=chunk_mask, other=0.0).to(
                tl.float32
            )
            output = (input - mean) * rstd
            if HAS_W:
                output = output * tl.load(weight_ptr + cols, mask=chunk_mask, other=0.0)
            if HAS_B:
                output = output + tl.load(bias_ptr + cols, mask=chunk_mask, other=0.0)
            output = output + tl.load(
                residual_ptr + base + cols, mask=chunk_mask, other=0.0
            ).to(tl.float32)
            tl.store(output_ptr + base + cols, output, mask=chunk_mask)
        else:
            input = tl.load(input_ptr + base + cols).to(tl.float32)
            output = (input - mean) * rstd
            if HAS_W:
                output = output * tl.load(weight_ptr + cols)
            if HAS_B:
                output = output + tl.load(bias_ptr + cols)
            output = output + tl.load(residual_ptr + base + cols).to(tl.float32)
            tl.store(output_ptr + base + cols, output)


@libentry()
@triton.jit
def post_layer_norm_residual_dx_kernel(
    grad_output_ptr,
    input_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    grad_input_ptr,
    N: tl.constexpr,
    TILE_N: tl.constexpr,
    NEED_TAIL: tl.constexpr,
    HAS_W: tl.constexpr,
):
    """Per-row LayerNorm input gradient.

    ``s2 = sum(dy * w)`` and ``s3 = sum(dy * w * x_hat)`` are accumulated in a
    1D tile (never through a 2D axis-1 reduction) so that non power-of-two N is
    handled correctly on XPU. As in the forward row kernel, when N is not a
    multiple of TILE_N every chunk uses the masked path: mixing unmasked full
    chunks with a masked tail in one kernel is miscompiled by the backend.
    """
    pid = ext.program_id(0)
    base = pid * N
    mean = tl.load(mean_ptr + pid).to(tl.float32)
    rstd = tl.load(rstd_ptr + pid).to(tl.float32)

    partial_2 = tl.zeros((TILE_N,), dtype=tl.float32)
    partial_3 = tl.zeros((TILE_N,), dtype=tl.float32)
    for offset in range(0, N, TILE_N):
        cols = offset + tl.arange(0, TILE_N)
        if NEED_TAIL:
            chunk_mask = cols < N
            dy = tl.load(grad_output_ptr + base + cols, mask=chunk_mask, other=0.0).to(
                tl.float32
            )
            x = tl.load(input_ptr + base + cols, mask=chunk_mask, other=0.0).to(
                tl.float32
            )
            if HAS_W:
                dy = dy * tl.load(weight_ptr + cols, mask=chunk_mask, other=0.0).to(
                    tl.float32
                )
            x_hat = tl.where(chunk_mask, (x - mean) * rstd, 0.0)
            partial_2 += tl.where(chunk_mask, dy, 0.0)
            partial_3 += tl.where(chunk_mask, dy * x_hat, 0.0)
        else:
            dy = tl.load(grad_output_ptr + base + cols).to(tl.float32)
            x = tl.load(input_ptr + base + cols).to(tl.float32)
            if HAS_W:
                dy = dy * tl.load(weight_ptr + cols).to(tl.float32)
            x_hat = (x - mean) * rstd
            partial_2 += dy
            partial_3 += dy * x_hat
    s2 = tl.sum(partial_2)
    s3 = tl.sum(partial_3)

    for offset in range(0, N, TILE_N):
        cols = offset + tl.arange(0, TILE_N)
        if NEED_TAIL:
            chunk_mask = cols < N
            dy = tl.load(grad_output_ptr + base + cols, mask=chunk_mask, other=0.0).to(
                tl.float32
            )
            x = tl.load(input_ptr + base + cols, mask=chunk_mask, other=0.0).to(
                tl.float32
            )
            if HAS_W:
                dy = dy * tl.load(weight_ptr + cols, mask=chunk_mask, other=0.0).to(
                    tl.float32
                )
            x_hat = (x - mean) * rstd
            dx = rstd * (dy - (s2 + x_hat * s3) / N)
            tl.store(grad_input_ptr + base + cols, dx, mask=chunk_mask)
        else:
            dy = tl.load(grad_output_ptr + base + cols).to(tl.float32)
            x = tl.load(input_ptr + base + cols).to(tl.float32)
            if HAS_W:
                dy = dy * tl.load(weight_ptr + cols).to(tl.float32)
            x_hat = (x - mean) * rstd
            dx = rstd * (dy - (s2 + x_hat * s3) / N)
            tl.store(grad_input_ptr + base + cols, dx)


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


def _pow2_divisor(value, cap):
    """Largest power of two <= cap that divides ``value``."""
    divisor = 1
    while divisor * 2 <= cap and value % (divisor * 2) == 0:
        divisor *= 2
    return divisor


def _row_tile_size(n):
    return triton.next_power_of_2(n) if n <= 64 else _ROW_TILE_N


def _uses_block_kernel(M, N):
    base, tail = _reduce_base_and_tail(N)
    if N > _BLOCK_PATH_MAX_N or base < _MIN_REDUCE_BASE or tail > _MAX_EXPLICIT_TAIL:
        return False
    return _block_tile_m(M, N) >= 2


def _block_tile_m(M, N):
    """Rows per program for the 2D tile path.

    ``_PREFERRED_TILE_M`` rows amortize the per-program dispatch cost while the
    ``_TILE_BUDGET`` element cap keeps the per-program working set bounded for
    wide features. ``_pow2_divisor`` keeps the tile free of a row mask.
    """
    base, tail = _reduce_base_and_tail(N)
    budget = max(1, _TILE_BUDGET // base)
    preferred = _PREFERRED_TILE_M if tail == 0 else _TAIL_TILE_M
    # A 2D tile with a singleton row axis trips the backend's 2D address
    # lowering, so the tile path always needs at least two rows per program.
    tile_m = _pow2_divisor(M, min(preferred, budget))
    return max(1, tile_m)


def _post_layer_norm_residual_forward(
    input, residual, normalized_shape, weight, bias, eps, save_stats
):
    N = math.prod(normalized_shape)
    M = input.numel() // N
    output = torch.empty_like(input)

    if save_stats:
        # Backward reductions consume FP32 statistics for every normalized row.
        mean = torch.empty(M, dtype=torch.float32, device=input.device)
        rstd = torch.empty_like(mean)
    else:
        # The constexpr guard removes the stores; a valid tensor avoids two
        # otherwise unnecessary allocations on the inference path.
        mean = output
        rstd = output

    has_w = weight is not None
    has_b = bias is not None
    weight_ptr = input if not has_w else weight
    bias_ptr = input if not has_b else bias

    if M != 0:
        with torch_device_fn.device(input.device):
            if _uses_block_kernel(M, N):
                tile_m = _block_tile_m(M, N)
                base, tail = _reduce_base_and_tail(N)
                grid = (triton.cdiv(M, tile_m), 1, 1)
                post_layer_norm_residual_block_kernel[grid](
                    input,
                    residual,
                    output,
                    weight_ptr,
                    bias_ptr,
                    mean,
                    rstd,
                    eps,
                    N=N,
                    BASE=base,
                    REM=tail,
                    TILE_M=tile_m,
                    HAS_W=has_w,
                    HAS_B=has_b,
                    SAVE_STATS=save_stats,
                )
            else:
                tile_n = _row_tile_size(N)
                grid = (M, 1, 1)
                post_layer_norm_residual_row_kernel[grid](
                    input,
                    residual,
                    output,
                    weight_ptr,
                    bias_ptr,
                    mean,
                    rstd,
                    eps,
                    N=N,
                    TILE_N=tile_n,
                    NEED_TAIL=(N % tile_n) != 0,
                    HAS_W=has_w,
                    HAS_B=has_b,
                    SAVE_STATS=save_stats,
                )

    return output, mean, rstd, M, N


class PostLayerNormResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, residual, normalized_shape, weight, bias, eps):
        # A residual-only gradient does not need LayerNorm statistics.
        needs_layer_norm_grad = (
            input.requires_grad
            or (weight is not None and weight.requires_grad)
            or (bias is not None and bias.requires_grad)
        )
        output, mean, rstd, M, N = _post_layer_norm_residual_forward(
            input,
            residual,
            normalized_shape,
            weight,
            bias,
            eps,
            save_stats=needs_layer_norm_grad,
        )

        if needs_layer_norm_grad:
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
        need_input, need_residual, _, need_weight, need_bias, _ = ctx.needs_input_grad
        output_mask = [need_input, need_weight, need_bias]
        if not any(output_mask):
            return None, grad_output if need_residual else None, None, None, None, None

        input, weight_saved, bias_saved, mean, rstd = ctx.saved_tensors
        weight = weight_saved if ctx.has_weight else None
        bias = bias_saved if ctx.has_bias else None

        # Flatten the normalized suffix to match the public LayerNorm backward API.
        input_2d = input.reshape(ctx.M, ctx.N)
        grad_2d = grad_output.contiguous().reshape(ctx.M, ctx.N)
        weight_1d = None if weight is None else weight.reshape(ctx.N)
        bias_1d = None if bias is None else bias.reshape(ctx.N)

        if ctx.M == 0:
            # Degenerate empty input: every LayerNorm gradient is empty/zero.
            grad_input = torch.empty_like(input_2d) if need_input else None
            grad_weight = torch.zeros_like(weight_1d) if need_weight else None
            grad_bias = torch.zeros_like(bias_1d) if need_bias else None
        else:
            grad_input = None
            if need_input:
                grad_input = torch.empty_like(input_2d)
                tile_n = _row_tile_size(ctx.N)
                with torch_device_fn.device(input.device):
                    post_layer_norm_residual_dx_kernel[(ctx.M, 1, 1)](
                        grad_2d,
                        input_2d,
                        weight_1d if weight_1d is not None else input_2d,
                        mean,
                        rstd,
                        grad_input,
                        N=ctx.N,
                        TILE_N=tile_n,
                        NEED_TAIL=(ctx.N % tile_n) != 0,
                        HAS_W=weight_1d is not None,
                    )

            grad_weight = None
            grad_bias = None
            if need_weight or need_bias:
                # Reuse the backend LayerNorm weight/bias gradient operator. Its
                # dx kernel is not used here (it is numerically unstable on XPU
                # for these small row counts), only its dW/dB reductions are.
                from flag_gems import layer_norm_backward

                _, grad_weight, grad_bias = layer_norm_backward(
                    grad_2d,
                    input_2d,
                    (ctx.N,),
                    mean,
                    rstd,
                    weight_1d,
                    bias_1d,
                    [False, need_weight, need_bias],
                )

        if grad_input is not None:
            grad_input = grad_input.reshape(ctx.input_shape)
        if grad_weight is not None:
            grad_weight = grad_weight.reshape(ctx.normalized_shape)
        if grad_bias is not None:
            grad_bias = grad_bias.reshape(ctx.normalized_shape)
        # The residual branch is an identity, so it needs no additional kernel.
        grad_residual = grad_output if need_residual else None
        return grad_input, grad_residual, None, grad_weight, grad_bias, None


def post_layer_norm_residual(
    input, residual, normalized_shape, weight=None, bias=None, eps=1e-5
):
    """Compute ``LayerNorm(input) + residual`` in one fused operator.

    This ordering follows the GraphCast processor update in Google DeepMind's
    reference implementation and its PyTorch counterpart in NVIDIA PhysicsNeMo:
    https://github.com/google-deepmind/weathernext/blob/7077d40a36db6541e3ed72ccaed1c0d202fa6014/graphcast/deep_typed_graph_net.py#L212-L248
    https://github.com/google-deepmind/weathernext/blob/7077d40a36db6541e3ed72ccaed1c0d202fa6014/graphcast/deep_typed_graph_net.py#L373-L394
    https://github.com/NVIDIA/physicsnemo/blob/65ec388929884436051203bca9dfb912eabd4c18/physicsnemo/nn/module/gnn_layers/mesh_graph_mlp.py#L120-L171
    https://github.com/NVIDIA/physicsnemo/blob/65ec388929884436051203bca9dfb912eabd4c18/physicsnemo/nn/module/mesh_node_block.py#L70-L91
    """
    logger.debug("GEMS_KUNLUNXIN POST_LAYER_NORM_RESIDUAL")

    normalized_shape = _normalize_shape(normalized_shape)
    _validate_inputs(input, residual, normalized_shape, weight, bias)

    if not input.is_contiguous():
        input = input.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()
    if weight is not None and not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    needs_autograd = torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad
        for tensor in (input, residual, weight, bias)
    )
    if not needs_autograd:
        output, _, _, _, _ = _post_layer_norm_residual_forward(
            input,
            residual,
            normalized_shape,
            weight,
            bias,
            eps,
            save_stats=False,
        )
        return output

    return PostLayerNormResidual.apply(
        input, residual, normalized_shape, weight, bias, eps
    )
