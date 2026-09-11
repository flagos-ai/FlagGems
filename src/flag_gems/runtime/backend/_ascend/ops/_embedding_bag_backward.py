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
from packaging import version

from flag_gems.ops._embedding_bag_backward import (
    _eb_backward_init,
    _eb_backward_scatter,
    _eb_backward_scatter_body,
    _eb_backward_sparse,
    _eb_backward_validate,
    _embedding_bag_backward_impl,
)
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
_LEGACY_TRITON = version.parse(triton.__version__) < version.parse("3.5")


@triton.jit
def _round_positive_reduced(
    value,
    bf16: tl.constexpr,  # bfloat16 rounding
):
    bits = value.to(tl.int32, bitcast=True)
    if bf16:
        bias = 0x7FFF + ((bits >> 16) & 1)
        result = ((bits + bias) & -65536).to(tl.float32, bitcast=True)
    else:
        # FP16 normals retain ten fraction bits. Handle overflow to infinity
        # and subnormals separately, including a zero reciprocal after overflow.
        rounded = (bits + 0xFFF + ((bits >> 13) & 1)) & -8192
        normal = tl.where(rounded >= 0x47800000, 0x7F800000, rounded)
        subnormal = bits < 0x38800000
        scaled = tl.where(subnormal, value, 0.0) * 16777216.0
        integral = scaled.to(tl.int32)
        fraction = scaled - integral.to(tl.float32)
        increment = (fraction > 0.5) | ((fraction == 0.5) & ((integral & 1) != 0))
        rounded_subnormal = (integral + increment.to(tl.int32)).to(tl.float32) * (
            1.0 / 16777216.0
        )
        result = tl.where(
            subnormal, rounded_subnormal, normal.to(tl.float32, bitcast=True)
        )
    return result


@libentry()
@triton.jit
def _embedding_bag_sparse_mean(
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
        # Materialize the reduced-dtype RNE stages with integer operations.
        # A cast to the input dtype and back can otherwise be folded away.
        reduced_bf16: tl.constexpr = grad.dtype.element_ty == tl.bfloat16
        size = _round_positive_reduced(tl.maximum(size, 1).to(tl.float32), reduced_bf16)
        inverse = _round_positive_reduced(1.0 / tl.maximum(size, 1), reduced_bf16)
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


@libentry()
@triton.jit
def _legacy_offset2bag(
    offsets,
    mapping,
    num_indices: tl.constexpr,
    num_bags: tl.constexpr,
    so: tl.constexpr,  # offsets stride
    steps: tl.constexpr,
    block: tl.constexpr,  # element block size
):
    x = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    low = tl.full((block,), 0, tl.int64)
    high = tl.full((block,), num_bags, tl.int64)
    for _ in range(steps):
        mid = (low + high) // 2
        boundary = tl.load(
            offsets + mid * so, (x < num_indices) & (mid < num_bags), other=num_indices
        )
        right = (mid < num_bags) & (boundary <= x)
        low = tl.where(right, mid + 1, low)
        high = tl.where(right, high, mid)
    tl.store(mapping + x, tl.maximum(low - 1, 0), x < num_indices)


@libentry()
@triton.jit
def _legacy_scatter(
    grad,  # output gradient
    indices,
    mapping,
    bag_size,
    maximum_indices,
    weights,
    acc,  # accumulator buffer
    num_indices: tl.constexpr,
    num_bags: tl.constexpr,
    embedding_dim: tl.constexpr,
    num_weights: tl.constexpr,
    pad: tl.constexpr,  # padding index
    sg0: tl.constexpr,  # output-gradient bag stride
    sg1: tl.constexpr,  # output-gradient feature stride
    si: tl.constexpr,  # indices stride
    sb: tl.constexpr,  # bag-size stride
    sx0: tl.constexpr,  # maximum-indices row stride
    sx1: tl.constexpr,  # maximum-indices feature stride
    sw: tl.constexpr,  # per-sample-weight stride
    mode: tl.constexpr,
    has_weights: tl.constexpr,
    fp64: tl.constexpr,  # float64 accumulation
    bd: tl.constexpr,  # embedding-dimension block size
    bs: tl.constexpr,  # sample block size
):
    # CANN 8.5's pointer analysis cannot reliably lower these dynamic strides.
    # Fix their metadata at compilation while retaining the generic arithmetic.
    _eb_backward_scatter_body(
        grad,
        indices,
        mapping,
        bag_size,
        maximum_indices,
        weights,
        acc,
        num_indices,
        num_bags,
        embedding_dim,
        num_weights,
        pad,
        sg0,
        sg1,
        si,
        sb,
        sx0,
        sx1,
        sw,
        mode,
        has_weights,
        fp64,
        bd,
        bs,
    )


@libentry()
@triton.jit
def _legacy_sparse_zero_dim_indices(
    indices,
    mapping,
    prefix,
    chunks,
    out_idx,  # output sparse row indices
    num_indices: tl.constexpr,
    num_bags: tl.constexpr,
    num_weights: tl.constexpr,
    pad: tl.constexpr,  # padding index
    si: tl.constexpr,  # indices stride
    compact: tl.constexpr,
    ranked: tl.constexpr,
    block: tl.constexpr,  # element block size
):
    # Empty feature dimensions need only the COO row indices. In old CANN,
    # even fully masked floating-point accesses can touch an empty allocation.
    lane = tl.arange(0, block)
    for base in range(
        tl.program_id(0).to(tl.int64) * block,
        num_indices,
        tl.num_programs(0).to(tl.int64) * block,
    ):
        sample = base.to(tl.int64) + lane
        idx = tl.load(indices + sample * si, sample < num_indices, other=pad)
        bag = tl.load(mapping + sample, sample < num_indices, other=0)
        active = (
            (sample < num_indices) & (idx >= 0) & (idx < num_weights) & (idx != pad)
        )
        active = active & (bag >= 0) & (bag < num_bags)
        if compact:
            pos = (
                tl.load(prefix + sample, sample < num_indices, other=0).to(tl.int64) - 1
            )
            if ranked:
                chunk = tl.load(chunks + sample // 256, sample < num_indices, other=0)
                pos += chunk.to(tl.int64)
        else:
            pos = sample
        tl.store(out_idx + pos, idx, active)


@libentry()
@triton.jit
def _legacy_empty_accumulator_init(
    freq,  # embedding occurrence frequencies
    error,
    num_weights: tl.constexpr,
    count_freq: tl.constexpr,  # count or normalize by embedding frequency
    block: tl.constexpr,  # element block size
):
    # The accumulator has no elements; only real metadata buffers need writes.
    if count_freq and num_weights > 0:
        x = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
        tl.store(freq + x, 0, x < num_weights)
    if tl.program_id(0) == 0:
        tl.store(error, 0)


@libentry()
@triton.jit
def _legacy_empty_indices_validate(
    offsets,
    bag_size,
    maximum_indices,
    error,
    num_bags: tl.constexpr,
    embedding_dim: tl.constexpr,
    num_weights: tl.constexpr,
    num_offsets: tl.constexpr,
    so: tl.constexpr,  # offsets stride
    sb: tl.constexpr,  # bag-size stride
    sx0: tl.constexpr,  # maximum-indices row stride
    sx1: tl.constexpr,  # maximum-indices feature stride
    mode: tl.constexpr,
    sparse: tl.constexpr,
    block: tl.constexpr,  # element block size
):
    # With N=0 no index, mapping, frequency, or compaction storage is accessed.
    # Keep all remaining offset/size/MAX checks, guarded by nonempty extents.
    x = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    invalid = tl.full((block,), 0, tl.int32)
    if num_offsets > 0:
        off = tl.load(offsets + x * so, x < num_offsets, other=0)
        invalid |= tl.where((x < num_offsets) & (off != 0), 2, 0)
    if num_bags > 0 and embedding_dim > 0:
        if mode == 1 or (mode == 2 and not sparse):
            size = tl.load(bag_size + x * sb, x < num_bags, other=0)
            invalid |= tl.where((x < num_bags) & (size != 0), 8, 0)
            if mode == 2:
                row, col = x // embedding_dim, x % embedding_dim
                maximum = tl.load(
                    maximum_indices + row * sx0 + col * sx1,
                    x < num_bags * embedding_dim,
                    other=-1,
                )
                bad = (x < num_bags * embedding_dim) & (
                    (maximum < -1) | (maximum >= tl.maximum(num_weights, 1))
                )
                invalid |= tl.where(bad, 16, 0)
    tl.atomic_max(error, tl.max(invalid, 0), sem="relaxed")


def _launch_ascend_backward(kernel, packed_kernel, grid, pointers, metadata, **options):
    if _LEGACY_TRITON and kernel is _eb_backward_scatter and metadata[3] == 0:
        # There are no output rows. The validator still reports every invalid
        # index, but old CANN must not launch atomics with an empty accumulator.
        return None
    if _LEGACY_TRITON and kernel is _eb_backward_validate and metadata[4] == 0:
        # No row can contribute a frequency when the table has zero rows.
        # Keep index checks and avoid masked atomics to an empty count buffer.
        metadata = (*metadata[:16], False, *metadata[17:])
    if _LEGACY_TRITON and kernel is _eb_backward_init and metadata[0] == 0:
        return _legacy_empty_accumulator_init[grid](
            pointers[1], pointers[2], metadata[1], metadata[2], metadata[3]
        )
    if _LEGACY_TRITON and kernel is _eb_backward_validate and metadata[1] == 0:
        return _legacy_empty_indices_validate[grid](
            pointers[1],
            pointers[3],
            pointers[4],
            pointers[9],
            metadata[2],
            metadata[3],
            metadata[4],
            metadata[5],
            metadata[8],
            metadata[10],
            metadata[11],
            metadata[12],
            metadata[14],
            metadata[15],
            metadata[21],
        )
    if _LEGACY_TRITON and kernel is _eb_backward_sparse and metadata[2] == 0:
        # The generic validator and exact COO size calculation have already run.
        # This launch never receives or touches a zero-sized floating buffer.
        return _legacy_sparse_zero_dim_indices[
            (min(triton.cdiv(metadata[0], 128), 65535),)
        ](
            pointers[1],
            pointers[2],
            pointers[5],
            pointers[6],
            pointers[7],
            metadata[0],
            metadata[1],
            metadata[3],
            metadata[4],
            metadata[7],
            metadata[12],
            metadata[13],
            128,
        )
    if _LEGACY_TRITON and kernel is _eb_backward_validate and not metadata[13]:
        # Reuse the existing disjoint mapping allocation after host validation.
        # The fixed-step search is the one already used by the forward kernel;
        # invalid offsets are still reported by the unchanged validator.
        n, b = metadata[1:3]
        mapping = pointers[5]
        if n:
            _legacy_offset2bag[(triton.cdiv(n, 128),)](
                pointers[1], mapping, n, b, metadata[8], b.bit_length(), 128
            )
        pointers = (*pointers[:2], mapping, *pointers[3:])
        metadata = (*metadata[:9], 1, *metadata[10:13], True, *metadata[14:])
    if (
        kernel is _eb_backward_scatter
        and metadata[12] == 2
        and pointers[4].dtype == torch.int64
    ):
        # CANN's int64-indexed, masked MAX scatter loses contributions with a
        # four-bag, 64-feature tile. A smaller feature tile preserves the full
        # index width and the existing FP32 accumulation contract.
        block_d = min(metadata[15], 32)
        block_s = metadata[16]
        grid = (triton.cdiv(metadata[1], block_s) * triton.cdiv(metadata[2], block_d),)
        metadata = (*metadata[:15], block_d, block_s)
    elif (
        kernel is _eb_backward_sparse
        and metadata[10] == 1
        and pointers[0].dtype in (torch.float16, torch.bfloat16)
    ):
        kernel = _embedding_bag_sparse_mean
    if _LEGACY_TRITON and kernel is _eb_backward_scatter:
        kernel = _legacy_scatter
    return kernel[grid](*pointers, *metadata, **options)


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
    logger.debug("GEMS_ASCEND _EMBEDDING_BAG_BACKWARD")
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
        device_assert_enabled=False,
        fused_init_enabled=False,
        launch_fn=_launch_ascend_backward,
    )
