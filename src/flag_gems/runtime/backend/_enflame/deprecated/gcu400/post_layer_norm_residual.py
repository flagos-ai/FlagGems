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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


MAX_BLOCK_SIZE = 4096
# gcu400 has 24 SIPs; 48 programs (2 waves) measured best for row-wise kernels
# on L600 (transplanted from the verified skip_layer_norm row-batch kernel).
OPTIMAL_GRID_SIZE = 48
# Row-batch [ROWS_PER_CTA, BLOCK_SIZE] tile budget, same verified-safe domain
# as the skip_layer_norm rows kernel on this backend: tile <= 16384 elements
# with ROWS_PER_CTA <= 32 was verified correct across 225 configurations,
# while R >= 32 with a 32768-element tile is a triton_gcu silent-miscompile
# zone.
_ROW_TILE_ELEM_MAX = 16384
_ROWS_PER_CTA_CAP = 32


def _rows_per_cta(m, n, element_bytes):
    """Pick the row-batch factor R for the [R, BLOCK_SIZE] 2D-tile launch.

    Sweep-measured optimum on gcu400: R tracks the largest power of two
    <= M/32 (grid lands around 32 programs), small M batches 8 rows per
    program, and every R is clamped to the largest power-of-two factor of
    m so that M % R == 0 always holds: no tail body is ever needed, which
    keeps the 2-D hot path branch-free and allows num_warps=2 (a
    compiled-in 1-D tail degrades the whole 2-D path at num_warps >= 2
    on this backend).

    Non-power-of-two N pays a 2x-wide masked tile; 16-bit dtypes collapse
    there at a 16384-element tile (measured N=513 fp16 R=16: 22.6us vs
    R=8 15.0us) while fp32 keeps the full R (R=16 15.3us < R=8 18.7us),
    so only 16-bit halves R.
    """
    block = min(triton.next_power_of_2(n), MAX_BLOCK_SIZE)
    tile_cap = _ROW_TILE_ELEM_MAX // max(block, 1)
    if m < 32:
        rows = 1
    elif m < 96:
        rows = max(1, min(m & (-m), 8, tile_cap))
    else:
        rows = m // 32
        rows = 1 << max(0, rows.bit_length() - 1)
        rows = min(rows, _ROWS_PER_CTA_CAP, tile_cap)
        rows = max(1, min(rows, m & (-m)))
    if n != block and element_bytes < 4:
        rows = max(1, rows // 2)
    return rows


@triton.jit
def _post_layer_norm_residual_rows_tile(
    output_ptr,
    input_ptr,
    residual_ptr,
    w,
    b,
    w_t,
    b_t,
    mean_ptr,
    rstd_ptr,
    row0,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
    EXACT: tl.constexpr,
    TAIL1: tl.constexpr,
    SAVE_STATS: tl.constexpr,
):
    """Full [ROWS_PER_CTA, BLOCK_SIZE] tile body; w/b are [1, BLOCK_SIZE].

    TAIL1 covers N == BLOCK_SIZE + 1 rows: the [R, BLOCK_SIZE] main block is
    unmasked and the single trailing column joins the per-row mean/variance
    as a [R] vector, avoiding the 2x-wide masked tile of the generic path.
    """
    rows = row0 + tl.arange(0, ROWS_PER_CTA)
    cols = tl.arange(0, BLOCK_SIZE)
    x_ptrs = input_ptr + rows[:, None] * N + cols[None, :]
    r_ptrs = residual_ptr + rows[:, None] * N + cols[None, :]
    y_ptrs = output_ptr + rows[:, None] * N + cols[None, :]
    if EXACT:
        x = tl.load(x_ptrs).to(tl.float32)
        r_val = tl.load(r_ptrs).to(tl.float32)
        mean = tl.sum(x, axis=1) / N
        d = x - mean[:, None]
        var = tl.sum(d * d, axis=1) / N
        rstd = tl.math.rsqrt(var + eps)
        if SAVE_STATS:
            tl.store(mean_ptr + rows, mean)
            tl.store(rstd_ptr + rows, rstd)
        y = w * (d * rstd[:, None]) + b
        y = y + r_val
        tl.store(y_ptrs, y.to(output_ptr.dtype.element_ty))
    else:
        if TAIL1:
            t_off = rows * N + BLOCK_SIZE
            x = tl.load(x_ptrs).to(tl.float32)
            xt = tl.load(input_ptr + t_off).to(tl.float32)
            r_val = tl.load(r_ptrs).to(tl.float32)
            rt = tl.load(residual_ptr + t_off).to(tl.float32)
            mean = (tl.sum(x, axis=1) + xt) / N
            d = x - mean[:, None]
            dt = xt - mean
            var = (tl.sum(d * d, axis=1) + dt * dt) / N
            rstd = tl.math.rsqrt(var + eps)
            if SAVE_STATS:
                tl.store(mean_ptr + rows, mean)
                tl.store(rstd_ptr + rows, rstd)
            y = w * (d * rstd[:, None]) + b
            y = y + r_val
            tl.store(y_ptrs, y.to(output_ptr.dtype.element_ty))
            yt = w_t * (dt * rstd) + b_t
            yt = yt + rt
            tl.store(output_ptr + t_off, yt.to(output_ptr.dtype.element_ty))
        else:
            col_mask = cols < N
            mask2d = col_mask[None, :]
            x = tl.load(x_ptrs, mask=mask2d, other=0.0).to(tl.float32)
            r_val = tl.load(r_ptrs, mask=mask2d, other=0.0).to(tl.float32)
            mean = tl.sum(x, axis=1) / N
            d = tl.where(mask2d, x - mean[:, None], 0.0)
            var = tl.sum(d * d, axis=1) / N
            rstd = tl.math.rsqrt(var + eps)
            if SAVE_STATS:
                tl.store(mean_ptr + rows, mean)
                tl.store(rstd_ptr + rows, rstd)
            y = w * (d * rstd[:, None]) + b
            y = y + r_val
            tl.store(y_ptrs, y.to(output_ptr.dtype.element_ty), mask=mask2d)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def post_layer_norm_residual_rows_kernel(
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
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
    EXACT: tl.constexpr,
    TAIL1: tl.constexpr,
    SAVE_STATS: tl.constexpr,
):
    """Persistent row-batch kernel: each program normalizes ROWS_PER_CTA rows
    per tile via a 2D [ROWS_PER_CTA, BLOCK_SIZE] tile, amortizing the
    per-program fixed cost (load dispatch + reduction trees + weight/bias
    loads) that dominates small-tile launches at large M.  Weight/bias are
    loaded once per program as [1, BLOCK_SIZE] 2D tiles: 1-D-vector broadcast
    operands are a verified miscompile window in this kernel family at
    BLOCK 512/1024 with num_warps >= 2.
    """
    pid = ext.program_id(0)
    num_programs = ext.num_programs(0)
    num_tiles = tl.cdiv(M, ROWS_PER_CTA)
    cols = tl.arange(0, BLOCK_SIZE)
    if EXACT:
        w = tl.load(weight_ptr + cols[None, :]).to(tl.float32)
        b = tl.load(bias_ptr + cols[None, :]).to(tl.float32)
    elif TAIL1:
        w = tl.load(weight_ptr + cols[None, :]).to(tl.float32)
        b = tl.load(bias_ptr + cols[None, :]).to(tl.float32)
    else:
        col_mask0 = cols < N
        w = tl.load(weight_ptr + cols[None, :], mask=col_mask0[None, :], other=0.0).to(
            tl.float32
        )
        b = tl.load(bias_ptr + cols[None, :], mask=col_mask0[None, :], other=0.0).to(
            tl.float32
        )
    if TAIL1:
        w_t = tl.load(weight_ptr + BLOCK_SIZE).to(tl.float32)
        b_t = tl.load(bias_ptr + BLOCK_SIZE).to(tl.float32)
    else:
        w_t = w
        b_t = b
    for tile in tl.range(pid, num_tiles, num_programs, num_stages=2):
        row0 = tile * ROWS_PER_CTA
        _post_layer_norm_residual_rows_tile(
            output_ptr,
            input_ptr,
            residual_ptr,
            w,
            b,
            w_t,
            b_t,
            mean_ptr,
            rstd_ptr,
            row0,
            N,
            eps,
            BLOCK_SIZE,
            ROWS_PER_CTA,
            EXACT,
            TAIL1,
            SAVE_STATS,
        )


@triton.jit
def _prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


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
        weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
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
        weight = tl.load(weight_ptr + col_offsets)
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

    with torch_device_fn.device(input.device):
        if N <= 4096:
            block_size = min(triton.next_power_of_2(N), MAX_BLOCK_SIZE)
            # N == 2^k + 1 rows carry a single trailing column past a clean
            # 2^k main block.  Folding that column into the per-row [R]
            # statistics vector beats the 2x-wide masked tile of the generic
            # path for every dtype (measured on N=513: fp16 15.0->10.6us,
            # fp32 15.6->12.7us).
            tail1 = N > 2 and (N & (N - 1)) != 0 and ((N - 1) & (N - 2)) == 0
            if tail1:
                block_size = N - 1
                rows_per_cta = _rows_per_cta(M, N - 1, input.element_size())
            else:
                rows_per_cta = _rows_per_cta(M, N, input.element_size())
            num_tiles = triton.cdiv(M, rows_per_cta)
            grid_size = min(num_tiles, OPTIMAL_GRID_SIZE)
            # One row per program: a single warp is fastest for narrow rows
            # (N <= 1024), wide rows need 4 warps; row-batch tiles use 2.
            if rows_per_cta > 1:
                num_warps = 2
            else:
                num_warps = 1 if N <= 1024 else 4
            post_layer_norm_residual_rows_kernel[(grid_size, 1, 1)](
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
                BLOCK_SIZE=block_size,
                ROWS_PER_CTA=rows_per_cta,
                EXACT=(N == block_size),
                TAIL1=tail1,
                SAVE_STATS=save_stats,
                num_warps=num_warps,
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

        # Resolve the public function at runtime so vendor rebinds remain active.
        from flag_gems import layer_norm_backward

        grad_input, grad_weight, grad_bias = layer_norm_backward(
            grad_2d,
            input_2d,
            (ctx.N,),
            mean,
            rstd,
            weight_1d,
            bias_1d,
            output_mask,
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
    https://github.com/NVIDIA/physicsnemo/blob/65ec388929884436051203bca9dfb912eabd4c18/physicsnemo/nn/module/gnn_layers/mesh_node_block.py#L70-L91
    """
    logger.debug("GEMS_ENFLAME POST_LAYER_NORM_RESIDUAL")
    normalized_shape = _normalize_shape(normalized_shape)
    _validate_inputs(input, residual, normalized_shape, weight, bias)

    # Keep bias outside LayerNorm on the fallback path because some vendor
    # implementations require weight whenever bias is provided.
    if (
        input.numel() == 0
        or weight is None
        or bias is None
        or not input.is_contiguous()
        or not residual.is_contiguous()
    ):
        output = torch.layer_norm(input, normalized_shape, weight, None, eps)
        if bias is not None:
            output = output + bias
        return output + residual

    weight = weight.contiguous()
    bias = bias.contiguous()

    needs_autograd = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (input, residual, weight, bias)
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
