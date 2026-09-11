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
def _eb_backward_init_packed(
    acc,  # accumulator buffer
    freq,  # embedding occurrence frequencies
    error,
    meta: tl.constexpr,  # packed kernel metadata
):
    _eb_backward_init_body(
        acc,
        freq,
        error,
        meta.value[0],
        meta.value[1],
        meta.value[2],
        meta.value[3],
    )


@libentry()
@triton.jit(debug=True)
def _eb_backward_validate_packed(
    indices,
    offsets,
    offset_to_bag,
    bag_size,
    maximum_indices,
    mapping,
    keep,
    freq,  # embedding occurrence frequencies
    chunks,
    error,
    acc,  # accumulator buffer
    meta: tl.constexpr,  # packed kernel metadata
):
    _eb_backward_validate_body(
        indices,
        offsets,
        offset_to_bag,
        bag_size,
        maximum_indices,
        mapping,
        keep,
        freq,
        chunks,
        error,
        acc,
        meta.value[0],
        meta.value[1],
        meta.value[2],
        meta.value[3],
        meta.value[4],
        meta.value[5],
        meta.value[6],
        meta.value[7],
        meta.value[8],
        meta.value[9],
        meta.value[10],
        meta.value[11],
        meta.value[12],
        meta.value[13],
        meta.value[14],
        meta.value[15],
        meta.value[16],
        meta.value[17],
        meta.value[18],
        meta.value[19],
        meta.value[20],
        meta.value[21],
    )


@libentry()
@triton.jit
def _eb_backward_chunk_offsets_packed(
    chunks,
    error,
    meta: tl.constexpr,  # packed kernel metadata
):
    _eb_backward_chunk_offsets_body(
        chunks,
        error,
        meta.value[0],
        meta.value[1],
    )


@libentry()
@triton.jit
def _eb_backward_scatter_packed(
    grad,  # output gradient
    indices,
    mapping,
    bag_size,
    maximum_indices,
    weights,
    acc,  # accumulator buffer
    meta: tl.constexpr,  # packed kernel metadata
):
    _eb_backward_scatter_body(
        grad,
        indices,
        mapping,
        bag_size,
        maximum_indices,
        weights,
        acc,
        meta.value[0],
        meta.value[1],
        meta.value[2],
        meta.value[3],
        meta.value[4],
        meta.value[5],
        meta.value[6],
        meta.value[7],
        meta.value[8],
        meta.value[9],
        meta.value[10],
        meta.value[11],
        meta.value[12],
        meta.value[13],
        meta.value[14],
        meta.value[15],
        meta.value[16],
    )


@libentry()
@triton.jit
def _eb_backward_finish_packed(
    acc,  # accumulator buffer
    freq,  # embedding occurrence frequencies
    out,  # output buffer
    meta: tl.constexpr,  # packed kernel metadata
):
    _eb_backward_finish_body(
        acc,
        freq,
        out,
        meta.value[0],
        meta.value[1],
        meta.value[2],
        meta.value[3],
    )


@libentry()
@triton.jit
def _eb_backward_sparse_packed(
    grad,  # output gradient
    indices,
    mapping,
    bag_size,
    weights,
    prefix,
    chunks,
    out_idx,  # output sparse row indices
    values,
    meta: tl.constexpr,  # packed kernel metadata
):
    _eb_backward_sparse_body(
        grad,
        indices,
        mapping,
        bag_size,
        weights,
        prefix,
        chunks,
        out_idx,
        values,
        meta.value[0],
        meta.value[1],
        meta.value[2],
        meta.value[3],
        meta.value[4],
        meta.value[5],
        meta.value[6],
        meta.value[7],
        meta.value[8],
        meta.value[9],
        meta.value[10],
        meta.value[11],
        meta.value[12],
        meta.value[13],
        meta.value[14],
        meta.value[15],
        meta.value[16],
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
    indices,
    offsets,
    offset_to_bag,
    bag_size,
    maximum_indices,
    out,  # output buffer
    meta: tl.constexpr,  # packed kernel metadata
):
    _eb_backward_validate_body(
        indices,
        offsets,
        offset_to_bag,
        bag_size,
        maximum_indices,
        indices,
        indices,
        indices,
        indices,
        out,
        out,
        0,
        meta.value[0],
        meta.value[1],
        meta.value[2],
        meta.value[3],
        meta.value[4],
        meta.value[5],
        meta.value[8],
        meta.value[9],
        meta.value[10],
        meta.value[11],
        meta.value[12],
        meta.value[13],
        meta.value[14],
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
    grad,  # output gradient
    indices,
    offsets,
    offset_to_bag,
    bag_size,
    maximum_indices,
    acc,  # accumulator buffer
    out,  # output buffer
    meta: tl.constexpr,  # packed kernel metadata
):
    _validate_max(indices, offsets, offset_to_bag, bag_size, maximum_indices, out, meta)
    num_bags: tl.constexpr = meta.value[1]
    embedding_dim: tl.constexpr = meta.value[2]
    num_weights: tl.constexpr = meta.value[3]
    pad: tl.constexpr = meta.value[5]
    sg0: tl.constexpr = meta.value[6]
    sg1: tl.constexpr = meta.value[7]
    sb: tl.constexpr = meta.value[11]
    sx0: tl.constexpr = meta.value[12]
    sx1: tl.constexpr = meta.value[13]
    fp64: tl.constexpr = meta.value[15]
    CAST: tl.constexpr = meta.value[16]
    BR: tl.constexpr = meta.value[17]
    bd: tl.constexpr = meta.value[18]
    BB: tl.constexpr = meta.value[19]
    pid = tl.program_id(0).to(tl.int64)
    nc: tl.constexpr = tl.cdiv(embedding_dim, bd)
    if pid < tl.cdiv(num_weights, BR) * nc:
        row0 = pid // nc * BR
        rows = row0 + tl.arange(0, BR)
        cols = pid % nc * bd + tl.arange(0, bd)
        positions = rows[:, None] * embedding_dim + cols[None, :]
        output_mask = (rows[:, None] < num_weights) & (cols[None, :] < embedding_dim)
        tl.store(acc + positions, 0, output_mask)
        # Each CTA owns all addresses in this output tile. No other CTA writes
        # its zeros, atomics, or cast, so a CTA barrier is sufficient here.
        tl.debug_barrier()
        if fp64:
            acc_dtype = tl.float64
        else:
            acc_dtype = tl.float32
        for start in range(0, num_bags, BB):
            bags = start + tl.arange(0, BB)
            maximum = tl.load(
                maximum_indices + bags[:, None] * sx0 + cols[None, :] * sx1,
                (bags[:, None] < num_bags) & (cols[None, :] < embedding_dim),
                other=-1,
            ).to(tl.int64)
            sizes = tl.load(bag_size + bags * sb, bags < num_bags, other=0)
            active = (
                (bags[:, None] < num_bags)
                & (cols[None, :] < embedding_dim)
                & (maximum >= row0)
                & (maximum < row0 + BR)
                & (maximum < num_weights)
                & (maximum != pad)
                & (sizes[:, None] > 0)
            )
            values = tl.load(
                grad + bags[:, None] * sg0 + cols[None, :] * sg1,
                active,
                other=0,
            ).to(acc_dtype)
            tl.atomic_add(
                acc + maximum * embedding_dim + cols[None, :],
                values,
                active,
                sem="relaxed",
            )
        if CAST:
            tl.debug_barrier()
            values = tl.load(acc + positions, output_mask, other=0)
            tl.store(out + positions, values, output_mask)


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
    indices,
    offsets,
    offset_to_bag,
    bag_size,
    maximum_indices,
    out,  # output buffer
    meta: tl.constexpr,  # packed kernel metadata
):
    _eb_backward_validate_body(
        indices,
        offsets,
        offset_to_bag,
        bag_size,
        maximum_indices,
        indices,
        indices,
        indices,
        indices,
        out,
        out,
        meta.value[2] * meta.value[3],
        meta.value[0],
        meta.value[1],
        meta.value[2],
        meta.value[3],
        meta.value[4],
        meta.value[5],
        meta.value[8],
        meta.value[9],
        meta.value[10],
        meta.value[11],
        meta.value[12],
        meta.value[13],
        meta.value[14],
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
def max_sort_segments(
    grad,  # output gradient
    bag_size,
    maximum_indices,
    out,  # output buffer
    meta: tl.constexpr,  # packed kernel metadata
):
    num_bags: tl.constexpr = meta.value[1]
    embedding_dim: tl.constexpr = meta.value[2]
    num_weights: tl.constexpr = meta.value[3]
    pad: tl.constexpr = meta.value[5]
    sg0: tl.constexpr = meta.value[6]
    sg1: tl.constexpr = meta.value[7]
    sb: tl.constexpr = meta.value[11]
    sx0: tl.constexpr = meta.value[12]
    sx1: tl.constexpr = meta.value[13]
    fp64: tl.constexpr = meta.value[15]
    bd: tl.constexpr = meta.value[18]
    BB: tl.constexpr = meta.value[19]
    cols = tl.program_id(0).to(tl.int64) * bd + tl.arange(0, bd)
    bags = tl.arange(0, BB).to(tl.int64)
    maximum = tl.load(
        maximum_indices + bags[None, :] * sx0 + cols[:, None] * sx1,
        (bags[None, :] < num_bags) & (cols[:, None] < embedding_dim),
        other=-1,
    ).to(tl.int64)
    sizes = tl.load(bag_size + bags * sb, bags < num_bags, other=0)
    active = (
        (bags[None, :] < num_bags)
        & (cols[:, None] < embedding_dim)
        & (maximum >= 0)
        & (maximum < num_weights)
        & (maximum != pad)
        & (sizes[None, :] > 0)
    )
    # The bag component makes every active key unique and provides the source
    # position after sorting. The host bounds the product below INT64_MAX.
    packed = tl.where(active, maximum, num_weights) * BB + bags[None, :]
    if (num_weights + 1) * BB < 2147483648:
        packed = packed.to(tl.int32)
    packed = tl.sort(packed, dim=1, descending=False)
    rows = packed // BB
    source_bag = (packed % BB).to(tl.int64)
    values = tl.load(
        grad + source_bag * sg0 + cols[:, None] * sg1,
        (rows < num_weights)
        & (source_bag < num_bags)
        & (cols[:, None] < embedding_dim),
        other=0,
    )
    if fp64:
        values = values.to(tl.float64)
    else:
        values = values.to(tl.float32)
    _, sums = tl.associative_scan((rows, values), 1, _segment_sum)
    # Old Triton lacks gather: a reverse scan identifies each segment's tail
    # without a shared/global key temporary or races on partial sums.
    _, tail = tl.associative_scan((rows, packed), 1, _segment_tail, reverse=True)
    tl.store(
        out + rows.to(tl.int64) * embedding_dim + cols[:, None],
        sums,
        (rows < num_weights) & (cols[:, None] < embedding_dim) & (packed == tail),
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
    grad,  # output gradient
    indices,
    offsets,
    mapping,
    bag_size,
    maximum_indices,
    out,  # output buffer
    meta: tl.constexpr,  # packed kernel metadata
):
    _validate_max(indices, offsets, mapping, bag_size, maximum_indices, out, meta)
    num_weights: tl.constexpr = meta.value[3]
    embedding_dim: tl.constexpr = meta.value[2]
    bd: tl.constexpr = meta.value[18]
    pid = tl.program_id(0).to(tl.int64)
    if pid < tl.cdiv(embedding_dim, bd):
        for base in range(0, num_weights * bd, 1024):
            x = base + tl.arange(0, 1024)
            row = x // bd
            col = pid * bd + x % bd
            tl.store(
                out + row * embedding_dim + col,
                0,
                (row < num_weights) & (col < embedding_dim),
            )
        # This CTA owns every row for its feature tile. Its subsequent unique
        # segment-tail stores cannot race with another CTA's initialization.
        tl.debug_barrier()
        _max_sort_segments_body(grad, bag_size, maximum_indices, out, meta)


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
def max_scatter_direct(
    grad,  # output gradient
    bag_size,
    maximum_indices,
    out,  # output buffer
    meta: tl.constexpr,  # packed kernel metadata
):
    num_bags: tl.constexpr = meta.value[1]
    embedding_dim: tl.constexpr = meta.value[2]
    num_weights: tl.constexpr = meta.value[3]
    pad: tl.constexpr = meta.value[5]
    sg0: tl.constexpr = meta.value[6]
    sg1: tl.constexpr = meta.value[7]
    sb: tl.constexpr = meta.value[11]
    sx0: tl.constexpr = meta.value[12]
    sx1: tl.constexpr = meta.value[13]
    block: tl.constexpr = meta.value[19]
    x = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    if embedding_dim > 0:
        bags = x // embedding_dim
        cols = x % embedding_dim
        maximum = tl.load(
            maximum_indices + bags * sx0 + cols * sx1,
            x < num_bags * embedding_dim,
            other=-1,
        ).to(tl.int64)
        sizes = tl.load(bag_size + bags * sb, x < num_bags * embedding_dim, other=0)
        active = (
            (x < num_bags * embedding_dim)
            & (maximum >= 0)
            & (maximum < num_weights)
            & (maximum != pad)
            & (sizes > 0)
        )
        values = tl.load(grad + bags * sg0 + cols * sg1, active, other=0).to(
            out.dtype.element_ty
        )
        tl.atomic_add(
            out + maximum * embedding_dim + cols, values, active, sem="relaxed"
        )


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
