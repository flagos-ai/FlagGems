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

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _scatter_reduce_kernel(
    inp,
    index,
    src,
    out,
    input_size0,
    input_size1,
    input_size2,
    index_size0,
    index_size1,
    index_size2,
    src_size1,
    src_size2,
    DIM: tl.constexpr,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    output_offset = tl.program_id(0)
    z = output_offset % input_size2
    y = (output_offset // input_size2) % input_size1
    x = output_offset // (input_size1 * input_size2)
    source_dim_offsets = tl.arange(0, BLOCK)

    if DIM == 0:
        valid_base = (y < index_size1) & (z < index_size2)
        dim_size = index_size0
        index_base = y * index_size2 + z
        src_base = y * src_size2 + z
        index_stride = index_size1 * index_size2
        src_stride = src_size1 * src_size2
        destination = x
    elif DIM == 1:
        valid_base = (x < index_size0) & (z < index_size2)
        dim_size = index_size1
        index_base = x * index_size1 * index_size2 + z
        src_base = x * src_size1 * src_size2 + z
        index_stride = index_size2
        src_stride = src_size2
        destination = y
    else:
        valid_base = (x < index_size0) & (y < index_size1)
        dim_size = index_size2
        index_base = x * index_size1 * index_size2 + y * index_size2
        src_base = x * src_size1 * src_size2 + y * src_size2
        index_stride = 1
        src_stride = 1
        destination = z

    valid = valid_base & (source_dim_offsets < dim_size)
    index_offsets = index_base + source_dim_offsets * index_stride
    src_offsets = src_base + source_dim_offsets * src_stride
    indices = tl.load(index + index_offsets, mask=valid, other=-1)
    selected = valid & (indices == destination)
    values = tl.load(src + src_offsets, mask=selected, other=0.0).to(tl.float32)
    self_value = tl.load(inp + output_offset).to(tl.float32)
    selected_count = tl.sum(selected.to(tl.int32), axis=0)

    if REDUCE == 0:
        reduced = tl.sum(tl.where(selected, values, 0.0), axis=0)
        if INCLUDE_SELF:
            reduced += self_value
    elif REDUCE == 1:
        reduced = 1.0
        for offset in tl.static_range(BLOCK):
            valid_offset = valid_base & (offset < dim_size)
            index_value = tl.load(
                index + index_base + offset * index_stride,
                mask=valid_offset,
                other=-1,
            )
            value = tl.load(
                src + src_base + offset * src_stride,
                mask=valid_offset,
                other=1.0,
            ).to(tl.float32)
            reduced *= tl.where(index_value == destination, value, 1.0)
        if INCLUDE_SELF:
            reduced *= self_value
    elif REDUCE == 2:
        reduced = tl.sum(tl.where(selected, values, 0.0), axis=0)
        count = selected_count
        if INCLUDE_SELF:
            reduced += self_value
            count += 1
        reduced /= count
    elif REDUCE == 3:
        reduced = tl.max(tl.where(selected, values, -float("inf")), axis=0)
        if INCLUDE_SELF:
            reduced = tl.maximum(reduced, self_value)
    else:
        reduced = tl.min(tl.where(selected, values, float("inf")), axis=0)
        if INCLUDE_SELF:
            reduced = tl.minimum(reduced, self_value)

    if not INCLUDE_SELF:
        reduced = tl.where(selected_count == 0, self_value, reduced)
    tl.store(out + output_offset, reduced)


@libentry()
@triton.jit(do_not_specialize=["input_dim_size", "source_dim_size", "inner_size"])
def _scatter_reduce_prod_kernel(
    inp,
    index,
    src,
    out,
    input_dim_size,
    source_dim_size,
    inner_size,
    INCLUDE_SELF: tl.constexpr,
):
    output_offset = tl.program_id(0)
    inner_offset = output_offset % inner_size
    output_dim_offset = (output_offset // inner_size) % input_dim_size
    outer_offset = output_offset // (input_dim_size * inner_size)
    self_value = tl.load(inp + output_offset).to(tl.float32)
    product = self_value if INCLUDE_SELF else 1.0
    selected_count = 0

    source_dim_offset = 0
    while source_dim_offset < source_dim_size:
        source_offset = (
            outer_offset * source_dim_size + source_dim_offset
        ) * inner_size + inner_offset
        selected = tl.load(index + source_offset) == output_dim_offset
        value = tl.load(src + source_offset).to(tl.float32)
        product = tl.where(selected, product * value, product)
        selected_count += selected.to(tl.int32)
        source_dim_offset += 1

    if not INCLUDE_SELF:
        product = tl.where(selected_count == 0, self_value, product)
    tl.store(out + output_offset, product)


@libentry()
@triton.jit
def _scatter_reduce_prod_3d_kernel(
    inp,
    index,
    src,
    out,
    input_size0,
    input_size1,
    input_size2,
    index_size0,
    index_size1,
    index_size2,
    src_size1,
    src_size2,
    DIM: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
):
    output_offset = tl.program_id(0)
    z = output_offset % input_size2
    y = (output_offset // input_size2) % input_size1
    x = output_offset // (input_size1 * input_size2)
    self_value = tl.load(inp + output_offset).to(tl.float32)

    if DIM == 0:
        valid_base = (y < index_size1) & (z < index_size2)
        dim_size = index_size0
        index_base = y * index_size2 + z
        src_base = y * src_size2 + z
        index_stride = index_size1 * index_size2
        src_stride = src_size1 * src_size2
        destination = x
    elif DIM == 1:
        valid_base = (x < index_size0) & (z < index_size2)
        dim_size = index_size1
        index_base = x * index_size1 * index_size2 + z
        src_base = x * src_size1 * src_size2 + z
        index_stride = index_size2
        src_stride = src_size2
        destination = y
    else:
        valid_base = (x < index_size0) & (y < index_size1)
        dim_size = index_size2
        index_base = x * index_size1 * index_size2 + y * index_size2
        src_base = x * src_size1 * src_size2 + y * src_size2
        index_stride = 1
        src_stride = 1
        destination = z

    product = self_value if INCLUDE_SELF else 1.0
    selected_count = 0
    offset = 0
    if valid_base:
        while offset < dim_size:
            index_value = tl.load(index + index_base + offset * index_stride)
            value = tl.load(src + src_base + offset * src_stride).to(tl.float32)
            matched = index_value == destination
            product = tl.where(matched, product * value, product)
            selected_count += matched.to(tl.int32)
            offset += 1
    if not INCLUDE_SELF:
        product = tl.where(selected_count == 0, self_value, product)
    tl.store(out + output_offset, product)


@libentry()
@triton.jit
def _scatter_reduce_2d_kernel(
    inp,
    index,
    src,
    out,
    input_size0,
    input_size1,
    index_size0,
    index_size1,
    src_size1,
    DIM: tl.constexpr,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    output_offset = tl.program_id(0)
    y = output_offset % input_size1
    x = output_offset // input_size1
    offsets = tl.arange(0, BLOCK)
    if DIM == 0:
        valid_base = y < index_size1
        dim_size, index_base, src_base = index_size0, y, y
        index_stride, src_stride, destination = index_size1, src_size1, x
    else:
        valid_base = x < index_size0
        dim_size, index_base, src_base = index_size1, x * index_size1, x * src_size1
        index_stride, src_stride, destination = 1, 1, y

    valid = valid_base & (offsets < dim_size)
    indices = tl.load(index + index_base + offsets * index_stride, mask=valid, other=-1)
    selected = valid & (indices == destination)
    values = tl.load(
        src + src_base + offsets * src_stride, mask=selected, other=0.0
    ).to(tl.float32)
    self_value = tl.load(inp + output_offset).to(tl.float32)
    selected_count = tl.sum(selected.to(tl.int32), axis=0)
    if REDUCE == 0:
        reduced = tl.sum(tl.where(selected, values, 0.0), axis=0)
        if INCLUDE_SELF:
            reduced += self_value
    elif REDUCE == 1:
        reduced = 1.0
        for offset in tl.static_range(BLOCK):
            valid_offset = valid_base & (offset < dim_size)
            index_value = tl.load(
                index + index_base + offset * index_stride, mask=valid_offset, other=-1
            )
            value = tl.load(
                src + src_base + offset * src_stride, mask=valid_offset, other=1.0
            ).to(tl.float32)
            reduced *= tl.where(valid_offset & (index_value == destination), value, 1.0)
        if INCLUDE_SELF:
            reduced *= self_value
    elif REDUCE == 2:
        reduced = tl.sum(tl.where(selected, values, 0.0), axis=0)
        count = selected_count
        if INCLUDE_SELF:
            reduced += self_value
            count += 1.0
        reduced /= count
    elif REDUCE == 3:
        reduced = tl.max(tl.where(selected, values, float("-inf")), axis=0)
        if INCLUDE_SELF:
            reduced = tl.maximum(reduced, self_value)
    else:
        reduced = tl.min(tl.where(selected, values, float("inf")), axis=0)
        if INCLUDE_SELF:
            reduced = tl.minimum(reduced, self_value)
    if not INCLUDE_SELF:
        reduced = tl.where(selected_count == 0, self_value, reduced)
    tl.store(out + output_offset, reduced)


_REDUCTIONS = {"sum": 0, "prod": 1, "mean": 2, "amax": 3, "amin": 4}


def scatter_reduce(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS_KUNLUNXIN SCATTER_REDUCE")
    if reduce not in _REDUCTIONS:
        raise RuntimeError(
            f"reduce argument must be either sum, prod, mean, amax or amin, got {reduce}"
        )
    if inp.ndim == 0:
        raise RuntimeError(
            "scatter_reduce(): Expected self to have non-zero dimensionality"
        )

    dim %= inp.ndim
    if inp.ndim not in (2, 3) or index.ndim != inp.ndim or src.ndim != inp.ndim:
        raise NotImplementedError(
            "Kunlunxin scatter_reduce currently supports two- or three-dimensional tensors"
        )
    for axis, size in enumerate(index.shape):
        if size > src.shape[axis] or (axis != dim and size > inp.shape[axis]):
            raise RuntimeError(
                "index must not be larger than src or self outside the scatter dimension"
            )

    result = inp.contiguous().clone()
    if index.numel() == 0 or result.numel() == 0:
        return result

    index = index.contiguous()
    src = src.contiguous()
    block = triton.next_power_of_2(index.shape[dim])
    if block > 65536:
        raise RuntimeError(
            "Kunlunxin scatter_reduce supports at most 65536 source elements along dim"
        )

    if inp.ndim == 2:
        with torch_device_fn.device(inp.device):
            _scatter_reduce_2d_kernel[(result.numel(),)](
                inp.contiguous(),
                index,
                src,
                result,
                *inp.shape,
                *index.shape,
                src.shape[1],
                DIM=dim,
                REDUCE=_REDUCTIONS[reduce],
                INCLUDE_SELF=include_self,
                BLOCK=block,
            )
        return result

    input_contiguous = inp.contiguous()
    with torch_device_fn.device(inp.device):
        if reduce == "prod":
            _scatter_reduce_prod_3d_kernel[(result.numel(),)](
                input_contiguous,
                index,
                src,
                result,
                *inp.shape,
                *index.shape,
                src.shape[1],
                src.shape[2],
                DIM=dim,
                INCLUDE_SELF=include_self,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
        else:
            _scatter_reduce_kernel[(result.numel(),)](
                input_contiguous,
                index,
                src,
                result,
                *inp.shape,
                *index.shape,
                src.shape[1],
                src.shape[2],
                DIM=dim,
                REDUCE=_REDUCTIONS[reduce],
                INCLUDE_SELF=include_self,
                BLOCK=block,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
    return result


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS_KUNLUNXIN SCATTER_REDUCE_TWO_")
    result = scatter_reduce(inp, dim, index, src, reduce, include_self=include_self)
    inp.copy_(result)
    return inp


def scatter_reduce_out(inp, dim, index, src, reduce, *, include_self=True, out=None):
    logger.debug("GEMS_KUNLUNXIN SCATTER_REDUCE_TWO_OUT")
    result = scatter_reduce(inp, dim, index, src, reduce, include_self=include_self)
    if tuple(out.shape) != tuple(result.shape):
        out.resize_(result.shape)
    out.copy_(result)
    return out
