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
from flag_gems.ops.layernorm import (
    _launch_layer_norm_backward_resident,
    _layer_norm_parallel_units,
    layer_norm_backward_kernel,
    weight_bias_backward_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def _layer_norm_backward_resident_dx_config(input, output_mask, M, N):
    if output_mask is None or M <= 0 or N <= 0:
        return None

    compute_dx, compute_dw, compute_db = map(bool, output_mask)
    # This dX launch is paired with the vendor affine reduction below.
    if not compute_dx or not (compute_dw or compute_db):
        return None
    if input.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return None

    heuristics = runtime.get_heuristic_config("layer_norm_backward_resident_dx")
    if heuristics is None:
        return None

    tile_n = triton.next_power_of_2(N)
    args = {
        "M": M,
        "N": N,
        "TILE_N": tile_n,
        "IS_LOW_PRECISION": input.dtype != torch.float32,
    }
    if M < heuristics["MIN_ROWS"](args):
        return None
    if tile_n > heuristics["MAX_RESIDENT_N"](args):
        return None

    parallel_units = _layer_norm_parallel_units(torch_device_fn.current_device())
    if parallel_units is None:
        return None

    if heuristics["FULL_ROW_GRID"](args):
        return 1, tile_n, M

    block_m = max(1, heuristics["TILE_ELEMENTS"](args) // tile_n)
    row_tiles = triton.cdiv(M, block_m)
    program_count = min(
        row_tiles,
        parallel_units * heuristics["PROGRAM_WAVES"](args),
    )
    return block_m, tile_n, program_count


# Stage 1: reduce one row group and one column tile into FP32 partials.
@libentry()
@triton.jit
def weight_bias_backward_partial_kernel(
    dY,
    X,
    Mean,
    Rstd,
    PartialW,
    PartialB,
    M,
    N,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    row_group = ext.program_id(0)
    col_group = ext.program_id(1)
    row_start = row_group * ROWS_PER_PROGRAM
    cols = col_group * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    col_mask = cols < N
    acc_w = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)
    acc_b = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)

    for row_offset in range(0, ROWS_PER_PROGRAM, BLOCK_ROW_SIZE):
        rows = row_start + row_offset + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
        mask = (rows < M) & col_mask[None, :]
        offsets = rows * N + cols[None, :]
        dy = tl.load(dY + offsets, mask=mask, other=0.0).to(tl.float32)
        if PartialW is not None:
            x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
            mean = tl.load(Mean + rows, mask=rows < M, other=0.0).to(tl.float32)
            rstd = tl.load(Rstd + rows, mask=rows < M, other=0.0).to(tl.float32)
            acc_w += tl.sum(dy * (x - mean) * rstd, axis=0)
        if PartialB is not None:
            acc_b += tl.sum(dy, axis=0)

    partial_offsets = row_group * N + cols
    if PartialW is not None:
        tl.store(PartialW + partial_offsets, acc_w, mask=col_mask)
    if PartialB is not None:
        tl.store(PartialB + partial_offsets, acc_b, mask=col_mask)


# Stage 2: reduce row-group partials into the final affine gradients.
@libentry()
@triton.jit
def weight_bias_backward_reduce_kernel(
    PartialW,
    PartialB,
    dW,
    dB,
    row_group_count,
    N,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    cols = ext.program_id(0) * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    col_mask = cols < N
    acc_w = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)
    acc_b = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)

    for group_offset in range(0, row_group_count, BLOCK_GROUP_SIZE):
        groups = group_offset + tl.arange(0, BLOCK_GROUP_SIZE)[:, None]
        mask = (groups < row_group_count) & col_mask[None, :]
        offsets = groups * N + cols[None, :]
        if PartialW is not None:
            acc_w += tl.sum(tl.load(PartialW + offsets, mask=mask, other=0.0), axis=0)
        if PartialB is not None:
            acc_b += tl.sum(tl.load(PartialB + offsets, mask=mask, other=0.0), axis=0)

    if dW is not None:
        tl.store(dW + cols, acc_w, mask=col_mask)
    if dB is not None:
        tl.store(dB + cols, acc_b, mask=col_mask)


def _weight_bias_backward_reduction_config(input, M, N):
    heuristics = runtime.get_heuristic_config("layer_norm_weight_bias_backward")
    if heuristics is None:
        return None

    args = {
        "M": M,
        "N": N,
        "IS_LOW_PRECISION": input.dtype != torch.float32,
    }
    rows_per_program = heuristics["ROWS_PER_PROGRAM"](args)
    row_group_count = triton.cdiv(M, rows_per_program)
    # The temporary reduction buffer pays off only with enough row parallelism.
    if row_group_count < heuristics["MIN_ROW_GROUPS"](args):
        return None

    block_row_size_fn = heuristics.get("BLOCK_ROW_SIZE")
    block_group_size_fn = heuristics.get("BLOCK_GROUP_SIZE")
    block_row_size = block_row_size_fn(args) if block_row_size_fn is not None else 32
    block_group_size = (
        block_group_size_fn(args) if block_group_size_fn is not None else 32
    )
    block_col_size = heuristics["BLOCK_COL_SIZE"](args)
    return (
        row_group_count,
        rows_per_program,
        block_row_size,
        block_col_size,
        block_group_size,
    )


def _launch_weight_bias_backward_reduction(
    grad_out,
    input,
    mean,
    rstd,
    weight_grad,
    bias_grad,
    M,
    N,
):
    config = _weight_bias_backward_reduction_config(input, M, N)
    if config is None:
        return False

    (
        row_group_count,
        rows_per_program,
        block_row_size,
        block_col_size,
        block_group_size,
    ) = config
    # Store inter-stage partials in FP32 so all dtypes use accumulation precision.
    partial_weight = (
        torch.empty((row_group_count, N), dtype=torch.float32, device=input.device)
        if weight_grad is not None
        else None
    )
    partial_bias = (
        torch.empty((row_group_count, N), dtype=torch.float32, device=input.device)
        if bias_grad is not None
        else None
    )

    with torch_device_fn.device(input.device):
        partial_grid = (row_group_count, triton.cdiv(N, block_col_size), 1)
        weight_bias_backward_partial_kernel[partial_grid](
            grad_out,
            input,
            mean,
            rstd,
            partial_weight,
            partial_bias,
            M,
            N,
            ROWS_PER_PROGRAM=rows_per_program,
            BLOCK_ROW_SIZE=block_row_size,
            BLOCK_COL_SIZE=block_col_size,
        )
        final_grid = (triton.cdiv(N, block_col_size), 1, 1)
        weight_bias_backward_reduce_kernel[final_grid](
            partial_weight,
            partial_bias,
            weight_grad,
            bias_grad,
            row_group_count,
            N,
            BLOCK_GROUP_SIZE=block_group_size,
            BLOCK_COL_SIZE=block_col_size,
        )
    return True


def layer_norm_backward(
    grad_out,
    input,
    normalized_shape,
    mean,
    rstd,
    weight=None,
    bias=None,
    output_mask=None,
):
    logger.debug("GEMS LAYERNORM BACKWARD")

    grad_out = grad_out.contiguous()
    input = input.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    # Match LayerNorm semantics when normalized_shape covers only an input suffix.
    N = math.prod(normalized_shape)
    M = input.numel() // N

    # MThreads launches resident dX separately from the grouped affine reduction.
    resident_config = _layer_norm_backward_resident_dx_config(input, output_mask, M, N)
    if resident_config is None:
        in_grad = None
    else:
        in_grad, _, _ = _launch_layer_norm_backward_resident(
            grad_out,
            input,
            mean,
            rstd,
            weight,
            bias,
            M,
            N,
            *resident_config,
            compute_dw=False,
            compute_db=False,
            use_fp32_scratch=False,
        )

    if output_mask[0] and in_grad is None:
        in_grad = torch.empty_like(input)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_ROW_SIZE"]), 1, 1)
        with torch_device_fn.device(input.device):
            layer_norm_backward_kernel[grid](
                grad_out, input, weight, mean, rstd, in_grad, M, N
            )

    if output_mask[1] is False and output_mask[2] is False:
        return in_grad, None, None

    weight_grad = torch.empty_like(weight) if output_mask[1] else None
    bias_grad = torch.empty_like(bias) if output_mask[2] else None
    used_grouped_reduction = _launch_weight_bias_backward_reduction(
        grad_out,
        input,
        mean,
        rstd,
        weight_grad,
        bias_grad,
        M,
        N,
    )
    if not used_grouped_reduction:
        # Small reductions keep the common one-stage affine kernel.
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_COL_SIZE"]), 1, 1)
        with torch_device_fn.device(input.device):
            weight_bias_backward_kernel[grid](
                grad_out, input, mean, rstd, weight_grad, bias_grad, M, N
            )
    return in_grad, weight_grad, bias_grad
