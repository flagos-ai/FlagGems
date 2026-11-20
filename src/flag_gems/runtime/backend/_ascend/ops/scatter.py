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

from flag_gems.ops.scatter import scatter as _scatter
from flag_gems.ops.scatter import scatter_ as _scatter_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import restride_dim

from .argsort import argsort
from .sort import sort

logger = logging.getLogger(__name__)
_DEVICE_REDUCE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_WIDE_SCATTER_BATCH_ELEMENTS = 1 << 21


@libentry()
@triton.jit
def _legacy_scatter_segments(
    inp,
    index,
    src,
    order,
    out,
    count,
    SHAPE: tl.constexpr,
    INP_STRIDES: tl.constexpr,
    INDEX_STRIDES: tl.constexpr,
    SRC_STRIDES: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    AXIS: tl.constexpr,
    INNER: tl.constexpr,
    PRODUCT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for start in range(tl.program_id(0) * BLOCK, count, tl.num_programs(0) * BLOCK):
        pos = (start + tl.arange(0, BLOCK)).to(tl.int64)
        valid = pos < count
        remaining = pos
        input_base = tl.full((BLOCK,), 0, tl.int64)
        index_base = tl.full((BLOCK,), 0, tl.int64)
        source_base = tl.full((BLOCK,), 0, tl.int64)
        output_base = tl.full((BLOCK,), 0, tl.int64)
        for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
            coordinate = remaining % SHAPE[axis]
            remaining = remaining // SHAPE[axis]
            if axis != AXIS:
                input_base += coordinate * INP_STRIDES[axis]
                index_base += coordinate * INDEX_STRIDES[axis]
                source_base += coordinate * SRC_STRIDES[axis]
                output_base += coordinate * OUT_STRIDES[axis]

        col = pos // INNER % SHAPE[AXIS]
        source_col = tl.load(order + pos, mask=valid, other=0)
        dest = tl.load(
            index + index_base + source_col * INDEX_STRIDES[AXIS], mask=valid, other=0
        )
        has_prev = valid & (col > 0)
        prev_col = tl.load(order + pos - INNER, mask=has_prev, other=0)
        prev_dest = tl.load(
            index + index_base + prev_col * INDEX_STRIDES[AXIS], mask=has_prev, other=-1
        )
        owner = valid & ((col == 0) | (prev_dest != dest))
        value = tl.load(
            inp + input_base + dest * INP_STRIDES[AXIS], mask=owner, other=0
        ).to(tl.float32)
        active = owner
        cursor = pos
        # Only the first element of each sorted segment writes the result.
        # Read self from an immutable input so a replay repeats the same store.
        while tl.sum(active.to(tl.int32)) > 0:
            update = tl.load(
                src + source_base + source_col * SRC_STRIDES[AXIS],
                mask=active,
                other=0,
            ).to(tl.float32)
            if PRODUCT:
                value = tl.where(active, value * update, value)
            else:
                value = tl.where(active, value + update, value)
            cursor += INNER
            col += 1
            active &= col < SHAPE[AXIS]
            source_col = tl.load(order + cursor, mask=active, other=0)
            next_dest = tl.load(
                index + index_base + source_col * INDEX_STRIDES[AXIS],
                mask=active,
                other=-1,
            )
            active &= next_dest == dest
        tl.store(out + output_base + dest * OUT_STRIDES[AXIS], value, mask=owner)


def _scatter_reduce_device(inp, dim, index, src, reduce):
    if dim < -inp.ndim or dim >= inp.ndim:
        raise IndexError("Dimension out of range")
    dim %= inp.ndim
    count = index.numel()
    if (
        index.ndim == 2
        and dim == 1
        and index.size(1) > 4096
        and count > _WIDE_SCATTER_BATCH_ELEMENTS
        and index.size(0) > 1
    ):
        # The rows have disjoint destinations. Bound the working set and
        # launch size of each independent wide-row batch on Ascend.
        rows_per_batch = max(1, _WIDE_SCATTER_BATCH_ELEMENTS // index.size(1))
        out = inp.clone(memory_format=torch.contiguous_format)
        for start in range(0, index.size(0), rows_per_batch):
            rows = min(rows_per_batch, index.size(0) - start)
            result = _scatter_reduce_device(
                inp.narrow(0, start, rows),
                dim,
                index.narrow(0, start, rows),
                src.narrow(0, start, rows),
                reduce,
            )
            out.narrow(0, start, rows).copy_(result)
        return out

    if count >= 1 << 18 and inp.dtype != torch.float32:
        # Share the large-input kernel across the low-precision dtypes and
        # avoid mixing subword tensor accesses with indirect FP32 updates.
        result = _scatter_reduce_device(
            inp.to(torch.float32), dim, index, src.to(torch.float32), reduce
        )
        return result.to(inp.dtype)

    out = inp.clone(memory_format=torch.contiguous_format)
    if count == 0:
        return out

    if (count >= (1 << 18) or index.size(dim) > 4096) and inp.size(dim) <= (1 << 24):
        # The device sort supports wide FP32 rows. Every valid destination
        # in this range has an exact FP32 representation.
        _, order = sort(index.to(torch.float32), dim, descending=False)
    else:
        order = argsort(index, dim, descending=False)
    order = order.contiguous()
    block = 64
    programs = min(triton.cdiv(count, block), 65535)
    with torch_device_fn.device(inp.device):
        _legacy_scatter_segments[(programs,)](
            inp,
            index,
            src,
            order,
            out,
            count,
            tuple(index.shape),
            inp.stride(),
            index.stride(),
            src.stride(),
            out.stride(),
            dim,
            math.prod(index.shape[dim + 1 :]),
            reduce == "multiply",
            block,
        )
    return out


def _compute_flat_offset(shape, strides, dim, N):
    idx = torch.arange(N, device="cpu", dtype=torch.int64)
    coord = torch.empty((len(shape), N), dtype=torch.int64, device="cpu")
    for i in reversed(range(len(shape))):
        coord[i] = idx % shape[i]
        idx = idx // shape[i]
    offset = torch.zeros(N, dtype=torch.int64, device="cpu")
    for i in range(len(shape)):
        if i != dim:
            offset += coord[i] * strides[i]
    return offset


def scatter_add_no_atomic(inp, dim, index, src):
    out = inp.clone()
    dim = dim % inp.ndim
    dim_stride = inp.stride(dim)

    src_strided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)

    N = index.numel()
    if N == 0:
        return out

    flat_index = index.reshape(-1).to(torch.int64).cpu()
    flat_src = src_strided.reshape(-1).contiguous().cpu()
    out_cpu = out.cpu()

    base_offset = _compute_flat_offset(index.shape, inp_restrided.stride(), dim, N).to(
        torch.int64
    )

    flat_out = out_cpu.reshape(-1)
    for i in range(N):
        idx = flat_index[i].item()
        out_offset = base_offset[i].item() + idx * dim_stride
        flat_out[out_offset] += flat_src[i].item()

    return out_cpu.to(inp.device)


def scatter_reduce_multiply_no_atomic(inp, dim, index, src):
    out = inp.clone()
    dim = dim % inp.ndim
    dim_stride = inp.stride(dim)

    src_strided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)

    N = index.numel()
    if N == 0:
        return out

    flat_index = index.reshape(-1).to(torch.int64).cpu()
    flat_src = src_strided.reshape(-1).contiguous().cpu()
    out_cpu = out.cpu()

    base_offset = _compute_flat_offset(index.shape, inp_restrided.stride(), dim, N).to(
        torch.int64
    )

    flat_out = out_cpu.reshape(-1)
    for i in range(N):
        idx = flat_index[i].item()
        out_offset = base_offset[i].item() + idx * dim_stride
        flat_out[out_offset] *= flat_src[i].item()

    return out_cpu.to(inp.device)


def scatter(inp, dim, index, src, reduce=None):
    logger.debug("GEMS_ASCEND SCATTER")
    if inp.dtype in _DEVICE_REDUCE_DTYPES and reduce in ("add", "multiply"):
        return _scatter_reduce_device(inp, dim, index, src, reduce)
    if reduce == "add":
        return scatter_add_no_atomic(inp, dim, index, src)
    if reduce == "multiply":
        return scatter_reduce_multiply_no_atomic(inp, dim, index, src)
    return _scatter(inp, dim, index, src, reduce)


def scatter_(inp, dim, index, src, reduce=None):
    logger.debug("GEMS_ASCEND SCATTER_")
    if inp.dtype in _DEVICE_REDUCE_DTYPES and reduce in ("add", "multiply"):
        inp.copy_(_scatter_reduce_device(inp, dim, index, src, reduce))
        return inp
    if reduce == "add":
        result = scatter_add_no_atomic(inp, dim, index, src)
        inp.copy_(result)
        return inp
    if reduce == "multiply":
        result = scatter_reduce_multiply_no_atomic(inp, dim, index, src)
        inp.copy_(result)
        return inp
    return _scatter_(inp, dim, index, src, reduce)
