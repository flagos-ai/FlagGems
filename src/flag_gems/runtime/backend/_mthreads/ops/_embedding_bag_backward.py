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
    _eb_backward_sparse,
    _eb_backward_validate_body,
    _embedding_bag_backward_impl,
    _launch_backward,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._mthreads.ops._embedding_bag import (
    _embedding_bag_check_flags,
)
from flag_gems.utils import libentry


@triton.jit
def _round_positive_bf16(value):
    bits = value.to(tl.uint32, bitcast=True)
    bias = 0x7FFF + ((bits >> 16) & 1)
    rounded = (bits + bias) & 0xFFFF0000
    return rounded.to(tl.float32, bitcast=True)


@libentry()
@triton.jit
def _eb_backward_sparse_mthreads(
    grad,  # output gradient
    indices,
    mapping,
    bag_size,
    weights,
    prefix,
    chunks,
    out_idx,  # output sparse row indices
    values,
    num_indices,
    num_bags,
    embedding_dim,
    num_weights,
    pad,  # padding index
    sg0: tl.constexpr,  # output-gradient bag stride
    sg1: tl.constexpr,  # output-gradient feature stride
    si: tl.constexpr,  # indices stride
    sb: tl.constexpr,  # bag-size stride
    sw: tl.constexpr,  # per-sample-weight stride
    mode: tl.constexpr,
    has_weights: tl.constexpr,
    compact: tl.constexpr,
    ranked: tl.constexpr,
    fp64: tl.constexpr,  # float64 accumulation
    bd: tl.constexpr,  # embedding-dimension block size
    bs: tl.constexpr,  # sample block size
):
    pid = tl.program_id(0).to(tl.int64)
    nd = tl.maximum(tl.cdiv(embedding_dim, bd), 1)
    sample = (pid // nd) * bs + tl.arange(0, bs)
    col = (pid % nd) * bd + tl.arange(0, bd)
    idx = tl.load(indices + sample * si, sample < num_indices, other=pad)
    active = (sample < num_indices) & (idx >= 0) & (idx < num_weights) & (idx != pad)
    if compact:
        pos = tl.load(prefix + sample, sample < num_indices, other=0) - 1
        if ranked:
            chunk = sample // 256
            pos += tl.load(chunks + chunk, sample < num_indices, other=0)
    else:
        pos = sample
    pos = pos.to(tl.int64)
    bag = tl.load(mapping + sample, sample < num_indices, other=0)
    active = active & (bag >= 0) & (bag < num_bags)
    if fp64:
        acc_type = tl.float64
    else:
        acc_type = tl.float32
    g = tl.load(
        grad + bag[:, None] * sg0 + col[None, :] * sg1,
        active[:, None] & (col[None, :] < embedding_dim),
        other=0,
    ).to(acc_type)
    if mode == 1:
        # ATen materializes the inverse bag size in grad dtype before its
        # sparse per-occurrence multiplication, including the reduced-dtype
        # rounding of both the denominator and reciprocal.
        size = tl.load(bag_size + bag * sb, active, other=1)
        # This override is selected only for BF16 sparse MEAN. Preserve both
        # round-to-nearest-even stages using integer bits before multiplication.
        # Denominators and inverses here are positive, finite FP32 numbers.
        size = _round_positive_bf16(size.to(tl.float32))
        inverse = _round_positive_bf16(1.0 / tl.maximum(size, 1))
        g = g * inverse[:, None]
    if has_weights:
        weight = tl.load(weights + sample * sw, sample < num_indices, other=0).to(
            acc_type
        )
        g = g * weight[:, None]
    if pid % nd == 0:
        tl.store(out_idx + pos, idx, active)
    tl.store(
        values + pos[:, None] * embedding_dim + col[None, :],
        g,
        active[:, None] & (col[None, :] < embedding_dim),
    )


def _launch_sparse_bf16(kernel, packed_kernel, grid, pointers, metadata, **options):
    if kernel is _eb_backward_sparse:
        _eb_backward_sparse_mthreads[grid](*pointers, *metadata, **options)
    else:
        _launch_backward(kernel, packed_kernel, grid, pointers, metadata, **options)


@triton.jit
def _validate_max(
    indices,
    offsets,
    offset_to_bag,
    bag_size,
    maximum_indices,
    out,  # output buffer
    error,
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
        error,
        out,
        0,
        meta[0],
        meta[1],
        meta[2],
        meta[3],
        meta[4],
        meta[5],
        meta[8],
        meta[9],
        meta[10],
        meta[11],
        meta[12],
        meta[13],
        meta[14],
        2,
        False,
        False,
        False,
        False,
        True,
        0,
        256,
    )


@libentry()
@triton.jit
def max_owned_tiles(
    grad,  # output gradient
    indices,
    offsets,
    offset_to_bag,
    bag_size,
    maximum_indices,
    acc,  # accumulator buffer
    out,  # output buffer
    error,
    meta: tl.constexpr,  # packed kernel metadata
):
    if tl.program_id(0).to(tl.int64) < meta[20]:
        _validate_max(
            indices, offsets, offset_to_bag, bag_size, maximum_indices, out, error, meta
        )
    num_bags: tl.constexpr = meta[1]
    embedding_dim: tl.constexpr = meta[2]
    num_weights: tl.constexpr = meta[3]
    pad: tl.constexpr = meta[5]
    sg0: tl.constexpr = meta[6]
    sg1: tl.constexpr = meta[7]
    sb: tl.constexpr = meta[11]
    sx0: tl.constexpr = meta[12]
    sx1: tl.constexpr = meta[13]
    fp64: tl.constexpr = meta[15]
    CAST: tl.constexpr = meta[16]
    BR: tl.constexpr = meta[17]
    bd: tl.constexpr = meta[18]
    BB: tl.constexpr = meta[19]
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
    br, bd = 512, 8
    bb = min(triton.next_power_of_2(max(b, 1)), 256)
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
    error = torch.empty((validate_blocks,), dtype=torch.int32, device=grad.device)
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
        validate_blocks,
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
            error,
            meta,
        )
        # The MUSA assertion remains in a separate tiny kernel. The arithmetic
        # kernel writes one fresh flag for each input-validation CTA.
        _embedding_bag_check_flags[(1,)](error, validate_blocks, 128, debug=True)
    return output


@libentry()
@triton.jit
def _max_initialize_direct(
    indices,
    offsets,
    offset_to_bag,
    bag_size,
    maximum_indices,
    out,  # output buffer
    error,
    meta: tl.constexpr,  # packed kernel metadata
):
    pid = tl.program_id(0).to(tl.int64)
    z = pid * 1024 + tl.arange(0, 1024)
    tl.store(out + z, 0, z < meta[2] * meta[3])
    # Large output tables require more zeroing CTAs than input-validation CTAs.
    # Only the latter own an error slot, which the checked kernel reads later.
    if pid < meta[20]:
        _validate_max(
            indices, offsets, offset_to_bag, bag_size, maximum_indices, out, error, meta
        )


@libentry()
@triton.jit
def max_scatter_direct(
    grad,  # output gradient
    bag_size,
    maximum_indices,
    out,  # output buffer
    meta: tl.constexpr,  # packed kernel metadata
):
    num_bags: tl.constexpr = meta[1]
    embedding_dim: tl.constexpr = meta[2]
    num_weights: tl.constexpr = meta[3]
    pad: tl.constexpr = meta[5]
    sg0: tl.constexpr = meta[6]
    sg1: tl.constexpr = meta[7]
    sb: tl.constexpr = meta[11]
    sx0: tl.constexpr = meta[12]
    sx1: tl.constexpr = meta[13]
    block: tl.constexpr = meta[19]
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
    validate_blocks = triton.cdiv(max(n, offsets.numel(), b, b * d, 1), 256)
    blocks = max(triton.cdiv(num_weights * d, 1024), validate_blocks)
    error = torch.empty((validate_blocks,), dtype=torch.int32, device=grad.device)
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
        validate_blocks,
    )
    with torch_device_fn.device(grad.device):
        _max_initialize_direct[(blocks,)](
            indices, offsets, mapping, sizes, maximum, output, error, meta
        )
        if b and d and num_weights:
            max_scatter_direct[(triton.cdiv(b * d, block),)](
                grad, sizes, maximum, output, meta, num_warps=warps
            )
        _embedding_bag_check_flags[(1,)](error, validate_blocks, 128, debug=True)
    return output


@libentry()
@triton.jit
def _max_initialize_sort(
    indices,
    offsets,
    offset_to_bag,
    bag_size,
    maximum_indices,
    out,  # output buffer
    error,
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
        error,
        out,
        meta[2] * meta[3],
        meta[0],
        meta[1],
        meta[2],
        meta[3],
        meta[4],
        meta[5],
        meta[8],
        meta[9],
        meta[10],
        meta[11],
        meta[12],
        meta[13],
        meta[14],
        2,
        False,
        False,
        False,
        False,
        True,
        0,
        256,
    )


@triton.jit
def _segment_sum(left_key, left_value, right_key, right_value):
    return right_key, tl.where(
        left_key == right_key, left_value + right_value, right_value
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
    num_bags: tl.constexpr = meta[1]
    embedding_dim: tl.constexpr = meta[2]
    num_weights: tl.constexpr = meta[3]
    pad: tl.constexpr = meta[5]
    sg0: tl.constexpr = meta[6]
    sg1: tl.constexpr = meta[7]
    sb: tl.constexpr = meta[11]
    sx0: tl.constexpr = meta[12]
    sx1: tl.constexpr = meta[13]
    fp64: tl.constexpr = meta[15]
    bd: tl.constexpr = meta[18]
    BB: tl.constexpr = meta[19]
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
    positions = tl.arange(0, BB)[None, :]
    next_positions = tl.broadcast_to(tl.minimum(positions + 1, BB - 1), (bd, BB))
    next_rows = tl.gather(rows, next_positions, 1)
    tail = (positions == BB - 1) | (rows != next_rows)
    tl.store(
        out + rows.to(tl.int64) * embedding_dim + cols[:, None],
        sums,
        (rows < num_weights) & (cols[:, None] < embedding_dim) & tail,
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
    output = torch.empty((num_weights, d), dtype=grad.dtype, device=grad.device)
    blocks = max(
        triton.cdiv(num_weights * d, 1024),
        triton.cdiv(max(n, offsets.numel(), b, b * d, 1), 256),
    )
    error = torch.empty((blocks,), dtype=torch.int32, device=grad.device)
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
        _max_initialize_sort[(blocks,)](
            indices, offsets, mapping, sizes, maximum, output, error, meta
        )
        if b and d and num_weights:
            max_sort_segments[(triton.cdiv(d, bd),)](
                grad, sizes, maximum, output, meta, num_warps=warps
            )
        _embedding_bag_check_flags[(1,)](error, blocks, 128, debug=True)
    return output


def _compute_max(grad, indices, offsets, mapping, sizes, maximum, num_weights, padding):
    if grad.dtype == torch.float32:
        return _compute_max_direct(
            grad, indices, offsets, mapping, sizes, maximum, num_weights, padding
        )
    # A bounded bag sort avoids repeatedly scanning wide reduced-precision
    # output tiles. Large bag counts retain the owned-tile accumulation path.
    if (
        grad.dtype in (torch.float16, torch.bfloat16)
        and grad.shape[0] <= 256
        and grad.shape[1] >= 256
    ):
        return _compute_max_sort(
            grad, indices, offsets, mapping, sizes, maximum, num_weights, padding
        )
    return _compute_max_owned(
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
    """MUSA MAX accumulates in FP32 and keeps validation in a separate checker."""
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
        launch_fn=(
            _launch_sparse_bf16
            if sparse and mode == 1 and grad.dtype == torch.bfloat16
            else _launch_backward
        ),
        max_fn=_compute_max,
    )
