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
    GRAD,
    INDICES,
    MAPPING,
    BAG_SIZE,
    WEIGHTS,
    PREFIX,
    CHUNKS,
    OUT_IDX,
    VALUES,
    N,
    B,
    D,
    V,
    PAD,
    SG0: tl.constexpr,
    SG1: tl.constexpr,
    SI: tl.constexpr,
    SB: tl.constexpr,
    SW: tl.constexpr,
    MODE: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
    COMPACT: tl.constexpr,
    RANKED: tl.constexpr,
    FP64: tl.constexpr,
    BD: tl.constexpr,
    BS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    nd = tl.maximum(tl.cdiv(D, BD), 1)
    sample = (pid // nd) * BS + tl.arange(0, BS)
    col = (pid % nd) * BD + tl.arange(0, BD)
    idx = tl.load(INDICES + sample * SI, sample < N, other=PAD)
    active = (sample < N) & (idx >= 0) & (idx < V) & (idx != PAD)
    if COMPACT:
        pos = tl.load(PREFIX + sample, sample < N, other=0) - 1
        if RANKED:
            chunk = sample // 256
            pos += tl.load(CHUNKS + chunk, sample < N, other=0)
    else:
        pos = sample
    pos = pos.to(tl.int64)
    bag = tl.load(MAPPING + sample, sample < N, other=0)
    active = active & (bag >= 0) & (bag < B)
    if FP64:
        acc_type = tl.float64
    else:
        acc_type = tl.float32
    g = tl.load(
        GRAD + bag[:, None] * SG0 + col[None, :] * SG1,
        active[:, None] & (col[None, :] < D),
        other=0,
    ).to(acc_type)
    if MODE == 1:
        # ATen materializes the inverse bag size in grad dtype before its
        # sparse per-occurrence multiplication, including the reduced-dtype
        # rounding of both the denominator and reciprocal.
        size = tl.load(BAG_SIZE + bag * SB, active, other=1)
        # This override is selected only for BF16 sparse MEAN. Preserve both
        # round-to-nearest-even stages using integer bits before multiplication.
        # Denominators and inverses here are positive, finite FP32 numbers.
        size = _round_positive_bf16(size.to(tl.float32))
        inverse = _round_positive_bf16(1.0 / tl.maximum(size, 1))
        g = g * inverse[:, None]
    if HAS_WEIGHTS:
        weight = tl.load(WEIGHTS + sample * SW, sample < N, other=0).to(acc_type)
        g = g * weight[:, None]
    if pid % nd == 0:
        tl.store(OUT_IDX + pos, idx, active)
    tl.store(
        VALUES + pos[:, None] * D + col[None, :],
        g,
        active[:, None] & (col[None, :] < D),
    )


def _launch_sparse_bf16(kernel, packed_kernel, grid, pointers, metadata, **options):
    if kernel is _eb_backward_sparse:
        _eb_backward_sparse_mthreads[grid](*pointers, *metadata, **options)
    else:
        _launch_backward(kernel, packed_kernel, grid, pointers, metadata, **options)


@triton.jit
def _validate_max(
    INDICES, OFFSETS, OFFSET2BAG, BAG_SIZE, MAXIMUM, OUT, ERROR, META: tl.constexpr
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
        ERROR,
        OUT,
        0,
        META[0],
        META[1],
        META[2],
        META[3],
        META[4],
        META[5],
        META[8],
        META[9],
        META[10],
        META[11],
        META[12],
        META[13],
        META[14],
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
    GRAD,
    INDICES,
    OFFSETS,
    OFFSET2BAG,
    BAG_SIZE,
    MAXIMUM,
    ACC,
    OUT,
    ERROR,
    META: tl.constexpr,
):
    if tl.program_id(0).to(tl.int64) < META[20]:
        _validate_max(INDICES, OFFSETS, OFFSET2BAG, BAG_SIZE, MAXIMUM, OUT, ERROR, META)
    B: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    V: tl.constexpr = META[3]
    PAD: tl.constexpr = META[5]
    SG0: tl.constexpr = META[6]
    SG1: tl.constexpr = META[7]
    SB: tl.constexpr = META[11]
    SX0: tl.constexpr = META[12]
    SX1: tl.constexpr = META[13]
    FP64: tl.constexpr = META[15]
    CAST: tl.constexpr = META[16]
    BR: tl.constexpr = META[17]
    BD: tl.constexpr = META[18]
    BB: tl.constexpr = META[19]
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
    INDICES, OFFSETS, OFFSET2BAG, BAG_SIZE, MAXIMUM, OUT, ERROR, META: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    z = pid * 1024 + tl.arange(0, 1024)
    tl.store(OUT + z, 0, z < META[2] * META[3])
    # Large output tables require more zeroing CTAs than input-validation CTAs.
    # Only the latter own an error slot, which the checked kernel reads later.
    if pid < META[20]:
        _validate_max(INDICES, OFFSETS, OFFSET2BAG, BAG_SIZE, MAXIMUM, OUT, ERROR, META)


@libentry()
@triton.jit
def max_scatter_direct(GRAD, BAG_SIZE, MAXIMUM, OUT, META: tl.constexpr):
    B: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    V: tl.constexpr = META[3]
    PAD: tl.constexpr = META[5]
    SG0: tl.constexpr = META[6]
    SG1: tl.constexpr = META[7]
    SB: tl.constexpr = META[11]
    SX0: tl.constexpr = META[12]
    SX1: tl.constexpr = META[13]
    BLOCK: tl.constexpr = META[19]
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
    INDICES, OFFSETS, OFFSET2BAG, BAG_SIZE, MAXIMUM, OUT, ERROR, META: tl.constexpr
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
        ERROR,
        OUT,
        META[2] * META[3],
        META[0],
        META[1],
        META[2],
        META[3],
        META[4],
        META[5],
        META[8],
        META[9],
        META[10],
        META[11],
        META[12],
        META[13],
        META[14],
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
def max_sort_segments(GRAD, BAG_SIZE, MAXIMUM, OUT, META: tl.constexpr):
    B: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    V: tl.constexpr = META[3]
    PAD: tl.constexpr = META[5]
    SG0: tl.constexpr = META[6]
    SG1: tl.constexpr = META[7]
    SB: tl.constexpr = META[11]
    SX0: tl.constexpr = META[12]
    SX1: tl.constexpr = META[13]
    FP64: tl.constexpr = META[15]
    BD: tl.constexpr = META[18]
    BB: tl.constexpr = META[19]
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
    positions = tl.arange(0, BB)[None, :]
    next_positions = tl.broadcast_to(tl.minimum(positions + 1, BB - 1), (BD, BB))
    next_rows = tl.gather(rows, next_positions, 1)
    tail = (positions == BB - 1) | (rows != next_rows)
    tl.store(
        OUT + rows.to(tl.int64) * D + cols[:, None],
        sums,
        (rows < V) & (cols[:, None] < D) & tail,
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
