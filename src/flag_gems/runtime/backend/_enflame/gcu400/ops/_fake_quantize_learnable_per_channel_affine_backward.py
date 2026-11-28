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

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

# Upper limits for tile elements / column-chunk width / number of chunks on the compact partial-sum path
_TILE_BLOCK = 4096
_COL_MAX = 16384
_MAX_GRID = 65535
_MAX_RUNS = 1 << 22


@libentry()
@triton.jit
def _fake_quantize_learnable_per_channel_affine_backward_kernel(
    grad_ptr,
    self_ptr,
    scale_ptr,
    zero_point_ptr,
    dX_ptr,
    dScale_vec_ptr,
    dZeroPoint_vec_ptr,
    numel,
    C,
    suffix,
    quant_min,
    quant_max,
    grad_factor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(self_ptr + offsets, mask=mask, other=0.0)

    # Channel index along the quantization axis.
    channel = (offsets // suffix) % C
    scale = tl.load(scale_ptr + channel, mask=mask, other=0.0)
    zero_point = tl.load(zero_point_ptr + channel, mask=mask, other=0.0)

    # Rounded (and clamped) zero point, mirroring _get_rounded_zero_point.
    zp_rounded = tl_extra_shim.rint(zero_point)
    zp_rounded = tl.minimum(tl.maximum(zp_rounded, quant_min), quant_max)

    inv_scale = 1.0 / scale

    # Quantized integer value. aten computes this with a single-rounded fused
    # multiply-add: xq = nearbyint(fma(x, inv_scale, zp_rounded)). A separately
    # rounded rint(x * inv_scale) + zp_rounded disagrees with the aten
    # reference on half-way ties (e.g. x * inv_scale = 127.5 with an odd
    # zp_rounded), so keep the fused form here.
    xq = tl_extra_shim.rint(tl.fma(x, inv_scale, zp_rounded))

    in_range = (xq >= quant_min) & (xq <= quant_max)
    dX = grad * in_range.to(tl.float32)

    # Fake-quantized (clamped) value: (clamp(xq) - zp_rounded) * scale
    xq_clamped = tl.minimum(tl.maximum(xq, quant_min), quant_max)
    xfq = (xq_clamped - zp_rounded) * scale

    # aten picks the saturated dScale branch by casting xq to int64: values the
    # cast cannot represent (|xq| >= 2**63, +-inf, nan) wrap to INT64_MIN and
    # fall into the quant_min branch. Mirror that so saturated-high inputs like
    # +inf match the reference (they take (quant_min - zp), not
    # (quant_max - zp)).
    sat_high = (xq > quant_max) & (xq < 9223372036854775808.0)
    dscale_small = (quant_min - zp_rounded).to(tl.float32)
    dscale_big = (quant_max - zp_rounded).to(tl.float32)

    dScale_sat = (
        tl.where(sat_high, grad * dscale_big, grad * dscale_small) * grad_factor
    )
    dScale_inrange = grad * (xfq - x) * inv_scale * grad_factor
    dScale_vec = tl.where(in_range, dScale_inrange, dScale_sat)

    # dZeroPoint is -grad * scale for every out-of-range xq (including nan,
    # where (xq < quant_min) | (xq > quant_max) would wrongly be all false).
    dZeroPoint_vec = tl.where(in_range, 0.0, grad * (-1.0) * scale * grad_factor)

    tl.store(dX_ptr + offsets, dX, mask=mask)
    tl.store(dScale_vec_ptr + offsets, dScale_vec, mask=mask)
    tl.store(dZeroPoint_vec_ptr + offsets, dZeroPoint_vec, mask=mask)


@triton.jit
def _rows_elementwise_backward(
    grad, x, scale, zero_point, quant_min, quant_max, grad_factor
):
    """Per-element (dX, dScale contribution, dZeroPoint contribution), same contract as the original kernel."""
    zp_rounded = tl_extra_shim.rint(zero_point)
    zp_rounded = tl.minimum(tl.maximum(zp_rounded, quant_min), quant_max)
    inv_scale = 1.0 / scale
    xq = tl_extra_shim.rint(tl.fma(x, inv_scale, zp_rounded))
    in_range = (xq >= quant_min) & (xq <= quant_max)

    dX = grad * in_range.to(tl.float32)

    xq_clamped = tl.minimum(tl.maximum(xq, quant_min), quant_max)
    xfq = (xq_clamped - zp_rounded) * scale

    sat_high = (xq > quant_max) & (xq < 9223372036854775808.0)
    dscale_small = (quant_min - zp_rounded).to(tl.float32)
    dscale_big = (quant_max - zp_rounded).to(tl.float32)

    dScale_sat = (
        tl.where(sat_high, grad * dscale_big, grad * dscale_small) * grad_factor
    )
    dScale_inrange = grad * (xfq - x) * inv_scale * grad_factor
    dScale = tl.where(in_range, dScale_inrange, dScale_sat)

    dZeroPoint = tl.where(in_range, 0.0, grad * (-1.0) * scale * grad_factor)
    return dX, dScale, dZeroPoint


@triton.jit
def _rows_kernel(
    grad_ptr,
    self_ptr,
    scale_ptr,
    zero_point_ptr,
    dX_ptr,
    dScale_partial_ptr,
    dZeroPoint_partial_ptr,
    runs,
    inner,
    C,
    n_col_chunks,
    quant_min,
    quant_max,
    grad_factor,
    COL: tl.constexpr,
    TILE_ROWS: tl.constexpr,
):
    # One program handles the col_chunk-th column chunk among TILE_ROWS consecutive chunks; each row belongs to exactly
    # one channel (channel id = chunk id % C), so summing over rows gives the per-channel partial sum
    col_chunk = tl.program_id(1)
    rows = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    row_mask = rows < runs
    columns = col_chunk * COL + tl.arange(0, COL)
    col_mask = columns < inner
    offsets = rows[:, None] * inner + columns[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    # Masked-out lanes all point to element 0, avoiding reads past the end of the allocation
    safe = tl.where(mask, offsets, 0)

    channel = rows % C
    safe_channel = tl.where(row_mask, channel, 0)
    scale = tl.load(scale_ptr + safe_channel, mask=row_mask, other=1.0)[:, None]
    zero_point = tl.load(zero_point_ptr + safe_channel, mask=row_mask, other=0.0)[
        :, None
    ]
    grad = tl.load(grad_ptr + safe, mask=mask, other=0.0)
    x = tl.load(self_ptr + safe, mask=mask, other=0.0)

    dX, dScale, dZeroPoint = _rows_elementwise_backward(
        grad, x, scale, zero_point, quant_min, quant_max, grad_factor
    )

    tl.store(dX_ptr + safe, dX, mask=mask)
    partial_offset = col_chunk * runs + rows
    tl.store(
        dScale_partial_ptr + partial_offset,
        tl.sum(tl.where(mask, dScale, 0.0), axis=1),
        mask=row_mask,
    )
    tl.store(
        dZeroPoint_partial_ptr + partial_offset,
        tl.sum(tl.where(mask, dZeroPoint, 0.0), axis=1),
        mask=row_mask,
    )


def _pow2_at_most(value, cap):
    block = 1
    while block * 2 <= value and block * 2 <= cap:
        block *= 2
    return max(block, 1)


def _next_pow2(value):
    block = 1
    while block < value:
        block *= 2
    return block


def _backward_with_compact_partials(
    grad,
    self,
    scale,
    zero_point,
    dX,
    C,
    inner,
    numel,
    quant_min,
    quant_max,
    grad_factor,
):
    """Main kernel plus a compact partial-sum pass for inner > 1."""
    runs = numel // inner
    col = min(_next_pow2(inner), _COL_MAX)
    col_chunks = triton.cdiv(inner, col)
    tile_rows = max(min(_TILE_BLOCK // col, runs), 1)
    tile_rows = max(tile_rows, triton.cdiv(runs, _MAX_GRID))
    grid = (triton.cdiv(runs, tile_rows), col_chunks)

    dScale_partial = torch.empty(
        (col_chunks, runs), dtype=torch.float32, device=scale.device
    )
    dZeroPoint_partial = torch.empty_like(dScale_partial)
    _rows_kernel[grid](
        grad,
        self,
        scale,
        zero_point,
        dX,
        dScale_partial,
        dZeroPoint_partial,
        runs,
        inner,
        C,
        col_chunks,
        quant_min,
        quant_max,
        grad_factor,
        COL=col,
        TILE_ROWS=tile_rows,
        num_warps=4 if col * tile_rows >= 2048 else 2,
    )

    # runs == C * outer_count; view as (col_chunks, outer, C) and reduce per channel
    dScale = dScale_partial.view(col_chunks, runs // C, C).sum(dim=(0, 1))
    dZeroPoint = dZeroPoint_partial.view(col_chunks, runs // C, C).sum(dim=(0, 1))
    return dScale, dZeroPoint


def _fake_quantize_learnable_per_channel_affine_backward(
    grad,
    self,
    scale,
    zero_point,
    axis,
    quant_min,
    quant_max,
    grad_factor=1.0,
):

    # The aten op requires all tensors to be float32; enforce it here too.
    assert grad.dtype == torch.float32, "grad must be float32"
    assert self.dtype == torch.float32, "self must be float32"
    assert scale.dtype == torch.float32, "scale must be float32"
    assert zero_point.dtype == torch.float32, "zero_point must be float32"
    grad = grad.contiguous()
    self = self.contiguous()
    scale = scale.contiguous()
    zero_point = zero_point.contiguous()

    numel = self.numel()
    dX = torch.empty_like(self)
    if numel == 0:
        dScale = torch.zeros_like(scale)
        dZeroPoint = torch.zeros_like(zero_point)
        return dX, dScale, dZeroPoint

    C = self.shape[axis]
    # Product of sizes after the quantization axis (for contiguous flatten index).
    suffix = 1
    for s in self.shape[axis + 1 :]:
        suffix *= s
    suffix = max(int(suffix), 1)

    if suffix > 1 and (numel // suffix) <= _MAX_RUNS:
        # inner > 1: the main kernel writes only dX, and per-channel partial sums go through the compact buffer
        dScale, dZeroPoint = _backward_with_compact_partials(
            grad,
            self,
            scale,
            zero_point,
            dX,
            int(C),
            suffix,
            numel,
            float(quant_min),
            float(quant_max),
            float(grad_factor),
        )
        return dX, dScale, dZeroPoint

    dScale_vec = torch.empty_like(self)
    dZeroPoint_vec = torch.empty_like(self)

    # Block size tuned for elementwise map over flattened numel.
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)

    _fake_quantize_learnable_per_channel_affine_backward_kernel[grid](
        grad,
        self,
        scale,
        zero_point,
        dX,
        dScale_vec,
        dZeroPoint_vec,
        numel,
        C,
        suffix,
        quant_min,
        quant_max,
        float(grad_factor),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reduce dScale_vec / dZeroPoint_vec over all axes except `axis`.
    ndim = self.dim()
    reduce_axes = tuple(i for i in range(ndim) if i != axis)
    dScale = dScale_vec.sum(dim=reduce_axes)
    dZeroPoint = dZeroPoint_vec.sum(dim=reduce_axes)

    return dX, dScale, dZeroPoint
