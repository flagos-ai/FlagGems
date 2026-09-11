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

import torch
import triton
import triton.language as tl

from flag_gems.ops._embedding_bag_backward import (
    _eb_backward_chunk_offsets,
    _eb_backward_chunk_offsets_body,
    _eb_backward_finish,
    _eb_backward_finish_body,
    _eb_backward_init,
    _eb_backward_init_body,
    _eb_backward_scatter,
    _eb_backward_scatter_body,
    _eb_backward_sparse,
    _eb_backward_sparse_body,
    _eb_backward_validate,
    _eb_backward_validate_body,
    _embedding_bag_backward_impl,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _eb_backward_init_packed(ACC, FREQ, ERROR, META: tl.constexpr):
    _eb_backward_init_body(
        ACC,
        FREQ,
        ERROR,
        META.value[0],
        META.value[1],
        META.value[2],
        META.value[3],
    )


@libentry()
@triton.jit(debug=True)
def _eb_backward_validate_packed(
    INDICES,
    OFFSETS,
    OFFSET2BAG,
    BAG_SIZE,
    MAXIMUM,
    MAPPING,
    KEEP,
    FREQ,
    CHUNKS,
    ERROR,
    ACC,
    META: tl.constexpr,
):
    _eb_backward_validate_body(
        INDICES,
        OFFSETS,
        OFFSET2BAG,
        BAG_SIZE,
        MAXIMUM,
        MAPPING,
        KEEP,
        FREQ,
        CHUNKS,
        ERROR,
        ACC,
        META.value[0],
        META.value[1],
        META.value[2],
        META.value[3],
        META.value[4],
        META.value[5],
        META.value[6],
        META.value[7],
        META.value[8],
        META.value[9],
        META.value[10],
        META.value[11],
        META.value[12],
        META.value[13],
        META.value[14],
        META.value[15],
        META.value[16],
        META.value[17],
        META.value[18],
        META.value[19],
        META.value[20],
        META.value[21],
    )


@libentry()
@triton.jit
def _eb_backward_chunk_offsets_packed(CHUNKS, ERROR, META: tl.constexpr):
    _eb_backward_chunk_offsets_body(
        CHUNKS,
        ERROR,
        META.value[0],
        META.value[1],
    )


@libentry()
@triton.jit
def _eb_backward_scatter_packed(
    GRAD, INDICES, MAPPING, BAG_SIZE, MAXIMUM, WEIGHTS, ACC, META: tl.constexpr
):
    _eb_backward_scatter_body(
        GRAD,
        INDICES,
        MAPPING,
        BAG_SIZE,
        MAXIMUM,
        WEIGHTS,
        ACC,
        META.value[0],
        META.value[1],
        META.value[2],
        META.value[3],
        META.value[4],
        META.value[5],
        META.value[6],
        META.value[7],
        META.value[8],
        META.value[9],
        META.value[10],
        META.value[11],
        META.value[12],
        META.value[13],
        META.value[14],
        META.value[15],
        META.value[16],
    )


@libentry()
@triton.jit
def _eb_backward_finish_packed(ACC, FREQ, OUT, META: tl.constexpr):
    _eb_backward_finish_body(
        ACC,
        FREQ,
        OUT,
        META.value[0],
        META.value[1],
        META.value[2],
        META.value[3],
    )


@libentry()
@triton.jit
def _eb_backward_sparse_packed(
    GRAD,
    INDICES,
    MAPPING,
    BAG_SIZE,
    WEIGHTS,
    PREFIX,
    CHUNKS,
    OUT_IDX,
    VALUES,
    META: tl.constexpr,
):
    _eb_backward_sparse_body(
        GRAD,
        INDICES,
        MAPPING,
        BAG_SIZE,
        WEIGHTS,
        PREFIX,
        CHUNKS,
        OUT_IDX,
        VALUES,
        META.value[0],
        META.value[1],
        META.value[2],
        META.value[3],
        META.value[4],
        META.value[5],
        META.value[6],
        META.value[7],
        META.value[8],
        META.value[9],
        META.value[10],
        META.value[11],
        META.value[12],
        META.value[13],
        META.value[14],
        META.value[15],
        META.value[16],
    )


_PACKED = {
    _eb_backward_init: _eb_backward_init_packed,
    _eb_backward_validate: _eb_backward_validate_packed,
    _eb_backward_chunk_offsets: _eb_backward_chunk_offsets_packed,
    _eb_backward_scatter: _eb_backward_scatter_packed,
    _eb_backward_finish: _eb_backward_finish_packed,
    _eb_backward_sparse: _eb_backward_sparse_packed,
}


def _launch_hygon(kernel, packed_kernel, grid, pointers, metadata, **options):
    _PACKED[kernel][grid](*pointers, metadata, **options)


@triton.jit
def _validate_max(
    INDICES, OFFSETS, OFFSET2BAG, BAG_SIZE, MAXIMUM, OUT, META: tl.constexpr
):
    _eb_backward_validate_body(
        INDICES,
        OFFSETS,
        OFFSET2BAG,
        BAG_SIZE,
        MAXIMUM,
        INDICES,
        INDICES,
        INDICES,
        INDICES,
        OUT,
        OUT,
        0,
        META.value[0],
        META.value[1],
        META.value[2],
        META.value[3],
        META.value[4],
        META.value[5],
        META.value[8],
        META.value[9],
        META.value[10],
        META.value[11],
        META.value[12],
        META.value[13],
        META.value[14],
        2,
        False,
        False,
        False,
        True,
        True,
        0,
        256,
        False,
    )


@libentry()
@triton.jit(debug=True)
def max_owned_tiles(
    GRAD,
    INDICES,
    OFFSETS,
    OFFSET2BAG,
    BAG_SIZE,
    MAXIMUM,
    ACC,
    OUT,
    META: tl.constexpr,
):
    _validate_max(INDICES, OFFSETS, OFFSET2BAG, BAG_SIZE, MAXIMUM, OUT, META)
    B: tl.constexpr = META.value[1]
    D: tl.constexpr = META.value[2]
    V: tl.constexpr = META.value[3]
    PAD: tl.constexpr = META.value[5]
    SG0: tl.constexpr = META.value[6]
    SG1: tl.constexpr = META.value[7]
    SB: tl.constexpr = META.value[11]
    SX0: tl.constexpr = META.value[12]
    SX1: tl.constexpr = META.value[13]
    FP64: tl.constexpr = META.value[15]
    CAST: tl.constexpr = META.value[16]
    BR: tl.constexpr = META.value[17]
    BD: tl.constexpr = META.value[18]
    BB: tl.constexpr = META.value[19]
    pid = tl.program_id(0).to(tl.int64)
    nc: tl.constexpr = tl.cdiv(D, BD)
    if pid < tl.cdiv(V, BR) * nc:
        row0 = pid // nc * BR
        rows = row0 + tl.arange(0, BR)
        cols = pid % nc * BD + tl.arange(0, BD)
        positions = rows[:, None] * D + cols[None, :]
        output_mask = (rows[:, None] < V) & (cols[None, :] < D)
        tl.store(ACC + positions, 0, output_mask)
        # Each CTA owns all addresses in this output tile. No other CTA writes
        # its zeros, atomics, or cast, so a CTA barrier is sufficient here.
        tl.debug_barrier()
        if FP64:
            acc_dtype = tl.float64
        else:
            acc_dtype = tl.float32
        for start in range(0, B, BB):
            bags = start + tl.arange(0, BB)
            maximum = tl.load(
                MAXIMUM + bags[:, None] * SX0 + cols[None, :] * SX1,
                (bags[:, None] < B) & (cols[None, :] < D),
                other=-1,
            ).to(tl.int64)
            sizes = tl.load(BAG_SIZE + bags * SB, bags < B, other=0)
            active = (
                (bags[:, None] < B)
                & (cols[None, :] < D)
                & (maximum >= row0)
                & (maximum < row0 + BR)
                & (maximum < V)
                & (maximum != PAD)
                & (sizes[:, None] > 0)
            )
            values = tl.load(
                GRAD + bags[:, None] * SG0 + cols[None, :] * SG1,
                active,
                other=0,
            ).to(acc_dtype)
            tl.atomic_add(
                ACC + maximum * D + cols[None, :], values, active, sem="relaxed"
            )
        if CAST:
            tl.debug_barrier()
            values = tl.load(ACC + positions, output_mask, other=0)
            tl.store(OUT + positions, values, output_mask)


def _compute_max_owned(
    grad, indices, offsets, mapping, sizes, maximum, num_weights, padding
):
    b, d = grad.shape
    n = indices.numel()
    br, bd, bb = 512, 8, 256
    cast = grad.dtype in (torch.float16, torch.bfloat16)
    math_blocks = triton.cdiv(num_weights, br) * triton.cdiv(d, bd)
    validate_blocks = triton.cdiv(max(n, offsets.numel(), b, b * d, 1), 256)
    blocks = max(math_blocks, validate_blocks)
    output = torch.empty((num_weights, d), dtype=grad.dtype, device=grad.device)
    accumulator = (
        torch.empty((num_weights, d), dtype=torch.float32, device=grad.device)
        if cast
        else output
    )
    meta = (
        n,
        b,
        d,
        num_weights,
        offsets.numel(),
        padding,
        *grad.stride(),
        indices.stride(0),
        offsets.stride(0),
        mapping.stride(0),
        sizes.stride(0),
        *maximum.stride(),
        mapping.numel() != 0 and d != 0,
        grad.dtype == torch.float64,
        cast,
        br,
        bd,
        bb,
    )
    with torch_device_fn.device(grad.device):
        max_owned_tiles[(blocks,)](
            grad,
            indices,
            offsets,
            mapping,
            sizes,
            maximum,
            accumulator,
            output,
            meta,
            debug=True,
        )
    return output


@libentry()
@triton.jit(debug=True)
def _max_initialize(
    INDICES, OFFSETS, OFFSET2BAG, BAG_SIZE, MAXIMUM, OUT, META: tl.constexpr
):
    _eb_backward_validate_body(
        INDICES,
        OFFSETS,
        OFFSET2BAG,
        BAG_SIZE,
        MAXIMUM,
        INDICES,
        INDICES,
        INDICES,
        INDICES,
        OUT,
        OUT,
        META.value[2] * META.value[3],
        META.value[0],
        META.value[1],
        META.value[2],
        META.value[3],
        META.value[4],
        META.value[5],
        META.value[8],
        META.value[9],
        META.value[10],
        META.value[11],
        META.value[12],
        META.value[13],
        META.value[14],
        2,
        False,
        False,
        False,
        True,
        True,
        0,
        256,
        False,
    )


@triton.jit
def _segment_sum(left_key, left_value, right_key, right_value):
    return right_key, tl.where(
        left_key == right_key, left_value + right_value, right_value
    )


@triton.jit
def _segment_tail(left_key, left_tail, right_key, right_tail):
    return right_key, tl.where(
        left_key == right_key, tl.maximum(left_tail, right_tail), right_tail
    )


@libentry()
@triton.jit
def max_sort_segments(GRAD, BAG_SIZE, MAXIMUM, OUT, META: tl.constexpr):
    B: tl.constexpr = META.value[1]
    D: tl.constexpr = META.value[2]
    V: tl.constexpr = META.value[3]
    PAD: tl.constexpr = META.value[5]
    SG0: tl.constexpr = META.value[6]
    SG1: tl.constexpr = META.value[7]
    SB: tl.constexpr = META.value[11]
    SX0: tl.constexpr = META.value[12]
    SX1: tl.constexpr = META.value[13]
    FP64: tl.constexpr = META.value[15]
    BD: tl.constexpr = META.value[18]
    BB: tl.constexpr = META.value[19]
    cols = tl.program_id(0).to(tl.int64) * BD + tl.arange(0, BD)
    bags = tl.arange(0, BB).to(tl.int64)
    maximum = tl.load(
        MAXIMUM + bags[None, :] * SX0 + cols[:, None] * SX1,
        (bags[None, :] < B) & (cols[:, None] < D),
        other=-1,
    ).to(tl.int64)
    sizes = tl.load(BAG_SIZE + bags * SB, bags < B, other=0)
    active = (
        (bags[None, :] < B)
        & (cols[:, None] < D)
        & (maximum >= 0)
        & (maximum < V)
        & (maximum != PAD)
        & (sizes[None, :] > 0)
    )
    # The bag component makes every active key unique and provides the source
    # position after sorting. The host bounds the product below INT64_MAX.
    packed = tl.where(active, maximum, V) * BB + bags[None, :]
    if (V + 1) * BB < 2147483648:
        packed = packed.to(tl.int32)
    packed = tl.sort(packed, dim=1, descending=False)
    rows = packed // BB
    source_bag = (packed % BB).to(tl.int64)
    values = tl.load(
        GRAD + source_bag * SG0 + cols[:, None] * SG1,
        (rows < V) & (source_bag < B) & (cols[:, None] < D),
        other=0,
    )
    if FP64:
        values = values.to(tl.float64)
    else:
        values = values.to(tl.float32)
    _, sums = tl.associative_scan((rows, values), 1, _segment_sum)
    # Old Triton lacks gather: a reverse scan identifies each segment's tail
    # without a shared/global key temporary or races on partial sums.
    _, tail = tl.associative_scan((rows, packed), 1, _segment_tail, reverse=True)
    tl.store(
        OUT + rows.to(tl.int64) * D + cols[:, None],
        sums,
        (rows < V) & (cols[:, None] < D) & (packed == tail),
    )


_SORT_CONFIG = (1, 4)


def _compute_max_sort(
    grad, indices, offsets, mapping, sizes, maximum, num_weights, padding
):
    b, d = grad.shape
    bb = triton.next_power_of_2(max(b, 1))
    if b > 2048 or num_weights > ((2**63 - 1) // bb - 1):
        return _compute_max_owned(
            grad, indices, offsets, mapping, sizes, maximum, num_weights, padding
        )
    n = indices.numel()
    bd, warps = _SORT_CONFIG
    # Wide feature grids provide enough CTAs for a one-wave bounded sort.
    if d >= 512 and b <= 128:
        warps = 1
    output = torch.empty((num_weights, d), dtype=grad.dtype, device=grad.device)
    blocks = max(
        triton.cdiv(num_weights * d, 1024),
        triton.cdiv(max(n, offsets.numel(), b, b * d, 1), 256),
    )
    meta = (
        n,
        b,
        d,
        num_weights,
        offsets.numel(),
        padding,
        *grad.stride(),
        indices.stride(0),
        offsets.stride(0),
        mapping.stride(0),
        sizes.stride(0),
        *maximum.stride(),
        mapping.numel() != 0 and d != 0,
        grad.dtype == torch.float64,
        False,
        0,
        bd,
        bb,
    )
    with torch_device_fn.device(grad.device):
        _max_initialize[(blocks,)](
            indices, offsets, mapping, sizes, maximum, output, meta, debug=True
        )
        if b and d and num_weights:
            max_sort_segments[(triton.cdiv(d, bd),)](
                grad, sizes, maximum, output, meta, num_warps=warps
            )
    return output


_max_sort_segments_body = max_sort_segments.fn


@libentry()
@triton.jit(debug=True)
def max_fused_sort(
    GRAD, INDICES, OFFSETS, MAPPING, BAG_SIZE, MAXIMUM, OUT, META: tl.constexpr
):
    _validate_max(INDICES, OFFSETS, MAPPING, BAG_SIZE, MAXIMUM, OUT, META)
    V: tl.constexpr = META.value[3]
    D: tl.constexpr = META.value[2]
    BD: tl.constexpr = META.value[18]
    pid = tl.program_id(0).to(tl.int64)
    if pid < tl.cdiv(D, BD):
        for base in range(0, V * BD, 1024):
            x = base + tl.arange(0, 1024)
            row = x // BD
            col = pid * BD + x % BD
            tl.store(OUT + row * D + col, 0, (row < V) & (col < D))
        # This CTA owns every row for its feature tile. Its subsequent unique
        # segment-tail stores cannot race with another CTA's initialization.
        tl.debug_barrier()
        _max_sort_segments_body(GRAD, BAG_SIZE, MAXIMUM, OUT, META)


_FUSED_CONFIG = (8, 4)


def _compute_max_segmented(
    grad, indices, offsets, mapping, sizes, maximum, num_weights, padding
):
    b, d = grad.shape
    if b > 32:
        return _compute_max_sort(
            grad, indices, offsets, mapping, sizes, maximum, num_weights, padding
        )
    bb = triton.next_power_of_2(max(b, 1))
    if num_weights > ((2**63 - 1) // bb - 1):
        return _compute_max_owned(
            grad, indices, offsets, mapping, sizes, maximum, num_weights, padding
        )
    n = indices.numel()
    bd, warps = _FUSED_CONFIG
    output = torch.empty((num_weights, d), dtype=grad.dtype, device=grad.device)
    blocks = max(
        triton.cdiv(d, bd), triton.cdiv(max(n, offsets.numel(), b, b * d, 1), 256)
    )
    meta = (
        n,
        b,
        d,
        num_weights,
        offsets.numel(),
        padding,
        *grad.stride(),
        indices.stride(0),
        offsets.stride(0),
        mapping.stride(0),
        sizes.stride(0),
        *maximum.stride(),
        mapping.numel() != 0 and d != 0,
        grad.dtype == torch.float64,
        False,
        0,
        bd,
        bb,
    )
    with torch_device_fn.device(grad.device):
        max_fused_sort[(blocks,)](
            grad,
            indices,
            offsets,
            mapping,
            sizes,
            maximum,
            output,
            meta,
            num_warps=warps,
            debug=True,
        )
    return output


@libentry()
@triton.jit
def max_scatter_direct(GRAD, BAG_SIZE, MAXIMUM, OUT, META: tl.constexpr):
    B: tl.constexpr = META.value[1]
    D: tl.constexpr = META.value[2]
    V: tl.constexpr = META.value[3]
    PAD: tl.constexpr = META.value[5]
    SG0: tl.constexpr = META.value[6]
    SG1: tl.constexpr = META.value[7]
    SB: tl.constexpr = META.value[11]
    SX0: tl.constexpr = META.value[12]
    SX1: tl.constexpr = META.value[13]
    BLOCK: tl.constexpr = META.value[19]
    x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if D > 0:
        bags = x // D
        cols = x % D
        maximum = tl.load(MAXIMUM + bags * SX0 + cols * SX1, x < B * D, other=-1).to(
            tl.int64
        )
        sizes = tl.load(BAG_SIZE + bags * SB, x < B * D, other=0)
        active = (
            (x < B * D)
            & (maximum >= 0)
            & (maximum < V)
            & (maximum != PAD)
            & (sizes > 0)
        )
        values = tl.load(GRAD + bags * SG0 + cols * SG1, active, other=0).to(
            OUT.dtype.element_ty
        )
        tl.atomic_add(OUT + maximum * D + cols, values, active, sem="relaxed")


_SCATTER_CONFIG = (128, 4)


def _compute_max_direct(
    grad, indices, offsets, mapping, sizes, maximum, num_weights, padding
):
    b, d = grad.shape
    n = indices.numel()
    block, warps = _SCATTER_CONFIG
    output = torch.empty((num_weights, d), dtype=grad.dtype, device=grad.device)
    blocks = max(
        triton.cdiv(num_weights * d, 1024),
        triton.cdiv(max(n, offsets.numel(), b, b * d, 1), 256),
    )
    meta = (
        n,
        b,
        d,
        num_weights,
        offsets.numel(),
        padding,
        *grad.stride(),
        indices.stride(0),
        offsets.stride(0),
        mapping.stride(0),
        sizes.stride(0),
        *maximum.stride(),
        mapping.numel() != 0 and d != 0,
        grad.dtype == torch.float64,
        False,
        0,
        0,
        block,
    )
    with torch_device_fn.device(grad.device):
        _max_initialize[(blocks,)](
            indices, offsets, mapping, sizes, maximum, output, meta, debug=True
        )
        if b and d and num_weights:
            max_scatter_direct[(triton.cdiv(b * d, block),)](
                grad, sizes, maximum, output, meta, num_warps=warps
            )
    return output


def _compute_max(grad, indices, offsets, mapping, sizes, maximum, num_weights, padding):
    if grad.dtype in (torch.float32, torch.float64):
        return _compute_max_direct(
            grad, indices, offsets, mapping, sizes, maximum, num_weights, padding
        )
    return _compute_max_segmented(
        grad, indices, offsets, mapping, sizes, maximum, num_weights, padding
    )


def _embedding_bag_backward(
    grad,
    indices,
    offsets,
    offset2bag,
    bag_size,
    maximum_indices,
    num_weights,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights=None,
    padding_idx=-1,
):
    """DTK 3.1 entry with explicit constexpr-tuple unwrapping."""
    return _embedding_bag_backward_impl(
        grad,
        indices,
        offsets,
        offset2bag,
        bag_size,
        maximum_indices,
        num_weights,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        padding_idx,
        device_assert_enabled=True,
        fused_init_enabled=True,
        launch_fn=_launch_hygon,
        max_fn=_compute_max,
        reuse_accumulator=grad.dtype == torch.float64,
    )
