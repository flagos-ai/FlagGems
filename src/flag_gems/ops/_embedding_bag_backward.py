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
import warnings

import torch
import triton
import triton.language as tl

from flag_gems.ops.cumsum import cumsum
from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# These capabilities are deliberately independent for backend integration.
_USE_DEVICE_ASSERT = runtime_device.vendor_name == "nvidia"
_USE_FUSED_INIT = runtime_device.vendor_name == "nvidia"
_USE_PACKED_META = runtime_device.vendor_name == "nvidia"


@libentry()
@triton.jit
def _eb_backward_init(
    acc,  # accumulator buffer
    freq,  # embedding occurrence frequencies
    error,
    num_acc_elements,  # number of accumulator elements
    num_weights,
    count_freq: tl.constexpr,  # count or normalize by embedding frequency
    block: tl.constexpr,  # element block size
):
    x = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    tl.store(acc + x, 0, x < num_acc_elements)
    if count_freq:
        tl.store(freq + x, 0, x < num_weights)
    if tl.program_id(0) == 0:
        tl.store(error, 0)


@libentry()
@triton.jit
def _eb_backward_validate(
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
    num_acc_elements,  # number of accumulator elements
    num_indices,
    num_bags,
    embedding_dim: tl.constexpr,
    num_weights,
    num_offsets,
    pad,  # padding index
    si: tl.constexpr,  # indices stride
    so: tl.constexpr,  # offsets stride
    sm: tl.constexpr,  # offset-to-bag mapping stride
    sb: tl.constexpr,  # bag-size stride
    sx0: tl.constexpr,  # maximum-indices row stride
    sx1: tl.constexpr,  # maximum-indices feature stride
    has_map: tl.constexpr,  # offset-to-bag mapping is available
    mode: tl.constexpr,
    sparse: tl.constexpr,
    count_freq: tl.constexpr,  # count or normalize by embedding frequency
    ranked: tl.constexpr,
    device_assert: tl.constexpr,
    init_acc: tl.constexpr,  # initialize accumulator
    num_chunks: tl.constexpr,
    block: tl.constexpr,  # element block size
    store_error: tl.constexpr = True,
):
    if init_acc:
        z = tl.program_id(0).to(tl.int64) * 1024 + tl.arange(0, 1024)
        tl.store(acc + z, 0, z < num_acc_elements)
    x = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    idx = tl.load(indices + x * si, x < num_indices, other=0)
    bad_index = (x < num_indices) & ((idx < 0) | (idx >= num_weights))

    off = tl.load(offsets + x * so, x < num_offsets, other=0)
    prev = tl.load(offsets + (x - 1) * so, (x > 0) & (x < num_offsets), other=0)
    bad_offset = (x < num_offsets) & ((off < prev) | (off < 0) | (off > num_indices))
    bad_offset = bad_offset | ((x == 0) & (num_offsets > 0) & (off != 0))
    bad_offset = bad_offset | (
        (num_offsets == num_bags + 1) & (x == num_bags) & (off != num_indices)
    )

    if has_map:
        bag = tl.load(offset_to_bag + x * sm, x < num_indices, other=0).to(tl.int64)
    else:
        # CPU SUM forward can omit offset2bag. Upper bound selects the last
        # equal boundary, which is necessary when consecutive bags are empty.
        left = tl.full((block,), 0, tl.int64)
        right = tl.full((block,), num_bags, tl.int64)
        while tl.sum((left < right).to(tl.int32), 0) > 0:
            mid = (left + right) // 2
            boundary = tl.load(
                offsets + mid * so,
                (mid < num_bags) & (x < num_indices),
                other=num_indices + 1,
            )
            advance = boundary <= x
            active = left < right
            left = tl.where(active & advance, mid + 1, left)
            right = tl.where(active & ~advance, mid, right)
        bag = left - 1
    valid_bag = (bag >= 0) & (bag < num_bags)
    start = tl.load(offsets + bag * so, (x < num_indices) & valid_bag, other=0)
    end = tl.load(
        offsets + (bag + 1) * so,
        (x < num_indices) & valid_bag & (bag + 1 < num_offsets),
        other=num_indices,
    )
    bad_map = (x < num_indices) & (~valid_bag | (x < start) | (x >= end))
    if mode != 2 or sparse:
        tl.store(mapping + x, bag, x < num_indices)
    valid_idx = (x < num_indices) & (idx >= 0) & (idx < num_weights) & (idx != pad)
    if sparse:
        selected = valid_idx.to(tl.int32)
        if ranked:
            tl.store(keep + x, tl.cumsum(selected, 0), x < num_indices)
            count = tl.sum(selected, 0)
            if num_chunks == 1:
                if tl.program_id(0) == 0:
                    tl.store(chunks, 0)
                    tl.store(error + 1, count)
            else:
                tl.store(
                    chunks + tl.program_id(0), count, tl.program_id(0) < num_chunks
                )
        else:
            tl.store(keep + x, selected, x < num_indices)
    if count_freq:
        tl.atomic_add(freq + idx, 1, valid_idx, sem="relaxed")

    bad_size = tl.full((block,), False, tl.int1)
    bad_max = tl.full((block,), False, tl.int1)
    if embedding_dim > 0 and (mode == 1 or (mode == 2 and not sparse)):
        size = tl.load(bag_size + x * sb, x < num_bags, other=0)
        bstart = tl.load(offsets + x * so, x < num_bags, other=0)
        bend = tl.load(
            offsets + (x + 1) * so,
            (x < num_bags) & (x + 1 < num_offsets),
            other=num_indices,
        )
        bad_size = (x < num_bags) & ((size < 0) | (size > bend - bstart))
        if mode == 1:
            sample_size = tl.load(
                bag_size + bag * sb, (x < num_indices) & valid_bag, other=0
            )
            bad_size = bad_size | (valid_idx & (sample_size <= 0))
        elif embedding_dim > 0:
            row = x // embedding_dim
            col = x % embedding_dim
            maximum = tl.load(
                maximum_indices + row * sx0 + col * sx1,
                x < num_bags * embedding_dim,
                other=-1,
            )
            # Ascend uses zero for empty MAX bags, including an empty table.
            bad_max = (x < num_bags * embedding_dim) & (
                (maximum < -1) | (maximum >= tl.maximum(num_weights, 1))
            )
    invalid = (
        bad_index.to(tl.int32)
        + 2 * bad_offset.to(tl.int32)
        + 4 * bad_map.to(tl.int32)
        + 8 * bad_size.to(tl.int32)
        + 16 * bad_max.to(tl.int32)
    )
    if store_error:
        if init_acc:
            # Distinct slots need no prior reset or cross-program atomic ordering.
            tl.store(error + tl.program_id(0), tl.max(invalid, 0))
        else:
            tl.atomic_max(error, tl.max(invalid, 0), sem="relaxed")
    if device_assert:
        tl.device_assert(
            invalid == 0,
            "embedding_bag_backward received invalid indices, offsets, or auxiliary data",
        )


@libentry()
@triton.jit
def _eb_backward_chunk_offsets(
    chunks,
    error,
    num_chunks: tl.constexpr,
    block: tl.constexpr,  # element block size
):
    x = tl.arange(0, block)
    count = tl.load(chunks + x, x < num_chunks, other=0)
    inclusive = tl.cumsum(count, 0)
    tl.store(chunks + x, inclusive - count, x < num_chunks)
    tl.store(error + 1, tl.sum(count, 0))


@libentry()
@triton.jit
def _eb_backward_scatter(
    grad,  # output gradient
    indices,
    mapping,
    bag_size,
    maximum_indices,
    weights,
    acc,  # accumulator buffer
    num_indices,
    num_bags,
    embedding_dim,
    num_weights,
    pad,  # padding index
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
    pid = tl.program_id(0).to(tl.int64)
    nd = tl.cdiv(embedding_dim, bd)
    sample = (pid // nd) * bs + tl.arange(0, bs)
    col = (pid % nd) * bd + tl.arange(0, bd)
    if fp64:
        acc_type = tl.float64
    else:
        acc_type = tl.float32
    if mode == 2:
        idx = tl.load(
            maximum_indices + sample[:, None] * sx0 + col[None, :] * sx1,
            (sample[:, None] < num_bags) & (col[None, :] < embedding_dim),
            other=-1,
        )
        size = tl.load(bag_size + sample * sb, sample < num_bags, other=0)
        g = tl.load(
            grad + sample[:, None] * sg0 + col[None, :] * sg1,
            (sample[:, None] < num_bags) & (col[None, :] < embedding_dim),
            other=0,
        ).to(acc_type)
        active = (
            (sample[:, None] < num_bags)
            & (col[None, :] < embedding_dim)
            & (idx >= 0)
            & (idx < num_weights)
            & (idx != pad)
            & (size[:, None] > 0)
        )
    else:
        idx1 = tl.load(indices + sample * si, sample < num_indices, other=pad)
        bag = tl.load(mapping + sample, sample < num_indices, other=0)
        active1 = (
            (sample < num_indices)
            & (idx1 >= 0)
            & (idx1 < num_weights)
            & (idx1 != pad)
            & (bag >= 0)
            & (bag < num_bags)
        )
        g = tl.load(
            grad + bag[:, None] * sg0 + col[None, :] * sg1,
            active1[:, None] & (col[None, :] < embedding_dim),
            other=0,
        ).to(acc_type)
        if mode == 1:
            size = tl.load(bag_size + bag * sb, active1, other=1).to(acc_type)
            g = g / tl.maximum(size[:, None], 1)
        if has_weights:
            weight = tl.load(weights + sample * sw, sample < num_indices, other=0).to(
                acc_type
            )
            g = g * weight[:, None]
        idx = idx1[:, None]
        active = active1[:, None] & (col[None, :] < embedding_dim)
    tl.atomic_add(
        acc + idx.to(tl.int64) * embedding_dim + col[None, :], g, active, sem="relaxed"
    )


@libentry()
@triton.jit
def _eb_backward_finish(
    acc,  # accumulator buffer
    freq,  # embedding occurrence frequencies
    out,  # output buffer
    num_acc_elements,  # number of accumulator elements
    embedding_dim,
    count_freq: tl.constexpr,  # count or normalize by embedding frequency
    block: tl.constexpr,  # element block size
):
    x = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    val = tl.load(acc + x, x < num_acc_elements, other=0)
    if count_freq:
        count = tl.load(freq + x // embedding_dim, x < num_acc_elements, other=1)
        val = val / tl.maximum(count, 1).to(val.dtype)
    tl.store(out + x, val, x < num_acc_elements)


@libentry()
@triton.jit
def _eb_backward_sparse(
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
        size = size.to(grad.dtype.element_ty).to(acc_type)
        inverse = (1.0 / tl.maximum(size, 1)).to(grad.dtype.element_ty).to(acc_type)
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


# Metadata packing reduces libentry argument processing on validated NVIDIA
# runtimes; each entry inlines exactly the same arithmetic and checks.
_eb_backward_init_body = _eb_backward_init.fn


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
        meta[0],
        meta[1],
        meta[2],
        meta[3],
    )


_eb_backward_validate_body = _eb_backward_validate.fn


@libentry()
@triton.jit
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
        meta[0],
        meta[1],
        meta[2],
        meta[3],
        meta[4],
        meta[5],
        meta[6],
        meta[7],
        meta[8],
        meta[9],
        meta[10],
        meta[11],
        meta[12],
        meta[13],
        meta[14],
        meta[15],
        meta[16],
        meta[17],
        meta[18],
        meta[19],
        meta[20],
        meta[21],
    )


_eb_backward_chunk_offsets_body = _eb_backward_chunk_offsets.fn


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
        meta[0],
        meta[1],
    )


_eb_backward_scatter_body = _eb_backward_scatter.fn


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
        meta[0],
        meta[1],
        meta[2],
        meta[3],
        meta[4],
        meta[5],
        meta[6],
        meta[7],
        meta[8],
        meta[9],
        meta[10],
        meta[11],
        meta[12],
        meta[13],
        meta[14],
        meta[15],
        meta[16],
    )


_eb_backward_finish_body = _eb_backward_finish.fn


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
        meta[0],
        meta[1],
        meta[2],
        meta[3],
    )


_eb_backward_sparse_body = _eb_backward_sparse.fn


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
        meta[0],
        meta[1],
        meta[2],
        meta[3],
        meta[4],
        meta[5],
        meta[6],
        meta[7],
        meta[8],
        meta[9],
        meta[10],
        meta[11],
        meta[12],
        meta[13],
        meta[14],
        meta[15],
        meta[16],
    )


def _launch_backward(kernel, packed_kernel, grid, pointers, metadata, **options):
    if _USE_PACKED_META:
        packed_kernel[grid](*pointers, metadata, **options)
    else:
        kernel[grid](*pointers, *metadata, **options)


def _embedding_bag_backward_impl(
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
    *,
    device_assert_enabled=_USE_DEVICE_ASSERT,
    fused_init_enabled=_USE_FUSED_INIT,
    launch_fn=_launch_backward,
    max_fn=None,
    reuse_accumulator=False,
):
    """ATen embedding-bag backward, with FP32 accumulation for reduced dtypes.

    The low-level ATen sparse path treats MAX as SUM and ignores argmax data.
    Dense MAX ignores scale_grad_by_freq. These are intentional ATen semantics.
    NVIDIA dense data errors are reported asynchronously by a CUDA device assert.
    """
    if grad.ndim != 2 or grad.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise RuntimeError("grad must be a two-dimensional floating-point tensor")
    if mode not in (0, 1, 2):
        raise RuntimeError("embedding_bag mode must be SUM (0), MEAN (1), or MAX (2)")
    if not isinstance(num_weights, int) or num_weights < 0:
        raise RuntimeError("num_weights must be a nonnegative integer")
    if padding_idx < -num_weights or padding_idx >= num_weights:
        if padding_idx != -1:
            raise RuntimeError("padding_idx must be within the number of embeddings")
    if padding_idx < -1:
        padding_idx += num_weights
    if sparse and scale_grad_by_freq:
        raise RuntimeError(
            "embedding_backward: scale_grad_by_freq not supported with sparse gradients"
        )
    n, b, d = indices.numel(), grad.shape[0], grad.shape[1]
    for name, tensor in (
        ("grad", grad),
        ("indices", indices),
        ("offsets", offsets),
        ("offset2bag", offset2bag),
        ("bag_size", bag_size),
        ("maximum_indices", maximum_indices),
    ):
        if tensor.device != grad.device or tensor.layout != torch.strided:
            raise RuntimeError(f"{name} must be strided and on the same device as grad")
        if name != "grad" and tensor.dtype not in (torch.int32, torch.int64):
            raise RuntimeError(f"{name} must have int32 or int64 dtype")
    if indices.ndim != 1 or offsets.ndim != 1 or offset2bag.ndim != 1:
        raise RuntimeError("indices, offsets, and offset2bag must be one-dimensional")
    if offsets.numel() not in (b, b + 1) or (b == 0 and n != 0):
        raise RuntimeError(
            "offsets length must match the number of bags, with an optional final boundary"
        )
    if offset2bag.numel() not in (0, n):
        raise RuntimeError("offset2bag must be empty or contain one entry per index")
    if (mode == 1 or (mode == 2 and not sparse)) and (
        bag_size.ndim != 1 or bag_size.numel() < b
    ):
        raise RuntimeError("bag_size must contain an entry for each bag")
    if mode == 2 and not sparse and maximum_indices.shape != grad.shape:
        raise RuntimeError("maximum_indices must have the same shape as grad for MAX")
    if per_sample_weights is not None:
        if mode != 0:
            raise RuntimeError("per_sample_weights is only supported for mode='sum'")
        if (
            per_sample_weights.ndim != 1
            or per_sample_weights.numel() != n
            or per_sample_weights.dtype != grad.dtype
            or per_sample_weights.device != grad.device
            or per_sample_weights.layout != torch.strided
        ):
            raise RuntimeError(
                "per_sample_weights must match indices length and grad dtype and device"
            )
    if not sparse and n and d and torch.are_deterministic_algorithms_enabled():
        message = "_embedding_bag_backward uses atomic accumulation and does not have a deterministic implementation"
        if torch.is_deterministic_algorithms_warn_only_enabled():
            warnings.warn(message, UserWarning, stacklevel=2)
        else:
            raise RuntimeError(message)

    if max_fn is not None and mode == 2 and not sparse:
        result = max_fn(
            grad,
            indices,
            offsets,
            offset2bag,
            bag_size,
            maximum_indices,
            num_weights,
            padding_idx,
        )
        if result is not None:
            return result

    compact = sparse and padding_idx >= 0 and n > 0
    device_assert = device_assert_enabled and not sparse
    nchunks = triton.cdiv(n, 256)
    ranked = compact and nchunks <= 4096
    count = scale_grad_by_freq and mode != 2 and not sparse
    acc_dtype = torch.float64 if grad.dtype == torch.float64 else torch.float32
    a = 0 if sparse else num_weights * d
    fused_init = device_assert and fused_init_enabled and not count
    extent = max(n, offsets.numel(), b, b * d if mode == 2 and not sparse else 0, 1)
    validate_blocks = max(
        triton.cdiv(extent, 256), triton.cdiv(a, 1024) if fused_init else 0
    )
    # Only dereferenced buffers need allocations. The other kernel pointer
    # arguments are ignored by constexpr branches or have a zero access extent.
    accumulator = (
        torch.empty((num_weights, d), dtype=acc_dtype, device=grad.device)
        if not sparse
        else grad
    )
    frequencies = (
        torch.empty((num_weights,), dtype=torch.int32, device=grad.device)
        if count
        else indices
    )
    error_shape = (validate_blocks,) if fused_init else ((2,) if ranked else ())
    error = torch.empty(error_shape, dtype=torch.int32, device=grad.device)
    chunk_offsets = (
        torch.empty((nchunks,), dtype=torch.int32, device=grad.device)
        if ranked
        else indices
    )
    mapping = torch.empty((n,), dtype=torch.int64, device=grad.device)
    keep = (
        torch.empty((n,), dtype=torch.int32, device=grad.device) if sparse else indices
    )
    maximum_strides = maximum_indices.stride() if maximum_indices.ndim == 2 else (0, 0)
    size_stride = bag_size.stride(0) if bag_size.ndim else 0
    with torch_device_fn.device(grad.device):
        if not fused_init:
            launch_fn(
                _eb_backward_init,
                _eb_backward_init_packed,
                (triton.cdiv(max(a, num_weights if count else 0, 1), 1024),),
                (
                    accumulator,
                    frequencies,
                    error,
                ),
                (
                    a,
                    num_weights,
                    count,
                    1024,
                ),
            )
        launch_fn(
            _eb_backward_validate,
            _eb_backward_validate_packed,
            (validate_blocks,),
            (
                indices,
                offsets,
                offset2bag,
                bag_size,
                maximum_indices,
                mapping,
                keep,
                frequencies,
                chunk_offsets,
                error,
                accumulator,
            ),
            (
                a,
                n,
                b,
                d,
                num_weights,
                offsets.numel(),
                padding_idx,
                indices.stride(0),
                offsets.stride(0),
                offset2bag.stride(0),
                size_stride,
                *maximum_strides,
                offset2bag.numel() != 0 and d != 0,
                mode,
                sparse,
                count,
                ranked,
                device_assert,
                fused_init,
                nchunks,
                256,
            ),
            **{"debug": True} if device_assert else {},
        )
        if ranked and nchunks > 1:
            launch_fn(
                _eb_backward_chunk_offsets,
                _eb_backward_chunk_offsets_packed,
                (1,),
                (
                    chunk_offsets,
                    error,
                ),
                (
                    nchunks,
                    triton.next_power_of_2(nchunks),
                ),
            )
        weight = per_sample_weights if per_sample_weights is not None else grad
        weight_stride = (
            per_sample_weights.stride(0) if per_sample_weights is not None else 0
        )
        bd = min(triton.next_power_of_2(max(d, 1)), 128)
        if sparse:
            if ranked:
                # One metadata transfer supplies both the recoverable error code
                # and exact COO size; only chunk counts need a global scan.
                invalid, nnz = error.tolist()
                prefix = keep
            else:
                prefix = cumsum(keep, 0) if compact else keep
                nnz = int(prefix[-1].item()) if compact else n
            out_indices = torch.empty((1, nnz), dtype=torch.int64, device=grad.device)
            values = torch.empty((nnz, d), dtype=grad.dtype, device=grad.device)
            if nnz:
                launch_fn(
                    _eb_backward_sparse,
                    _eb_backward_sparse_packed,
                    (triton.cdiv(n, 4) * max(triton.cdiv(d, bd), 1),),
                    (
                        grad,
                        indices,
                        mapping,
                        bag_size,
                        weight,
                        prefix,
                        chunk_offsets,
                        out_indices,
                        values,
                    ),
                    (
                        n,
                        b,
                        d,
                        num_weights,
                        padding_idx,
                        *grad.stride(),
                        indices.stride(0),
                        size_stride,
                        weight_stride,
                        mode,
                        per_sample_weights is not None,
                        compact,
                        ranked,
                        grad.dtype == torch.float64,
                        bd,
                        4,
                    ),
                )
        else:
            if n and d:
                samples = b if mode == 2 else n
                launch_fn(
                    _eb_backward_scatter,
                    _eb_backward_scatter_packed,
                    (triton.cdiv(samples, 4) * triton.cdiv(d, bd),),
                    (
                        grad,
                        indices,
                        mapping,
                        bag_size,
                        maximum_indices,
                        weight,
                        accumulator,
                    ),
                    (
                        n,
                        b,
                        d,
                        num_weights,
                        padding_idx,
                        *grad.stride(),
                        indices.stride(0),
                        size_stride,
                        *maximum_strides,
                        weight_stride,
                        mode,
                        per_sample_weights is not None,
                        grad.dtype == torch.float64,
                        bd,
                        4,
                    ),
                )
            if grad.dtype == acc_dtype and not count:
                result = accumulator
            else:
                # Each finish lane owns one accumulator element, so a backend
                # may normalize in place when the output dtype already matches.
                result = (
                    accumulator
                    if reuse_accumulator and grad.dtype == acc_dtype
                    else torch.empty(
                        (num_weights, d), dtype=grad.dtype, device=grad.device
                    )
                )
                if a:
                    launch_fn(
                        _eb_backward_finish,
                        _eb_backward_finish_packed,
                        (triton.cdiv(a, 1024),),
                        (
                            accumulator,
                            frequencies,
                            result,
                        ),
                        (
                            a,
                            max(d, 1),
                            count,
                            1024,
                        ),
                    )

        # Every indirect access is masked even when validation fails. NVIDIA
        # dense calls use an explicitly enabled device assertion, so valid calls
        # need no host synchronization. DEVICE_ASSERT is a constexpr and part of
        # the libentry cache key, keeping checked and ordinary binaries distinct.
        if device_assert:
            return result
        # Other backends report recoverable errors after enqueueing computation.
        if not ranked:
            invalid = error.item()
        if invalid:
            if invalid & 1:
                raise RuntimeError("embedding_bag index is outside [0, num_weights)")
            if invalid & 2:
                raise RuntimeError(
                    "offsets must start at zero, be nondecreasing, and end within indices"
                )
            if invalid & 4:
                raise RuntimeError("offset2bag does not match the bag boundaries")
            if invalid & 8:
                raise RuntimeError(
                    "bag_size is invalid for the bag boundaries or nonpadding indices"
                )
            raise RuntimeError("maximum_indices contains an out-of-range row")
        if sparse:
            return torch.sparse_coo_tensor(
                out_indices,
                values,
                (num_weights, d),
                dtype=grad.dtype,
                device=grad.device,
                is_coalesced=False,
                check_invariants=False,
            )
        return result


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
    logger.debug("GEMS _EMBEDDING_BAG_BACKWARD")
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
    )
