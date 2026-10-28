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

"""MThreads routing for scatter_reduce product reduction."""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.scatter_reduce import (
    _scatter_reduce_rowwise,
    _select_rowwise_strategy,
)
from flag_gems.ops.scatter_reduce import scatter_reduce as _generic_scatter_reduce
from flag_gems.ops.scatter_reduce import scatter_reduce_ as _generic_scatter_reduce_
from flag_gems.ops.scatter_reduce import (
    scatter_reduce_out as _generic_scatter_reduce_out,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_LINK_BUILD_BLOCK = 256
_LINK_BUILD_LOOP = 4
_LINK_FINAL_BLOCK = 128
_MAX_GRID_X = 65535


@triton.jit
def mthreads_scatter_reduce_prod_build_lists_kernel(
    index_ptr,
    src_ptr,
    heads_ptr,
    next_ptr,
    N,
    index_ncols,
    src_ncols,
    out_ncols,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    """Build one lock-free source list per output using integer exchange."""
    pid = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
        tl.int64
    ) * tl.num_programs(0)
    lanes = tl.arange(0, BLOCK).to(tl.int64)
    base = pid * (BLOCK * LOOP) + lanes

    for loop_idx in range(LOOP):
        offsets = base + loop_idx * BLOCK
        mask = offsets < N
        row = offsets // index_ncols
        col = offsets % index_ncols
        source = tl.load(
            src_ptr + row * src_ncols + col,
            mask=mask,
            other=1.0,
        ).to(tl.float32)
        if INCLUDE_SELF:
            changes_value = source != 1.0
        else:
            changes_value = mask
        active = mask & changes_value
        index = tl.load(index_ptr + offsets, mask=active, other=0).to(tl.int64)
        out_offsets = row * out_ncols + index
        previous = tl.atomic_xchg(
            heads_ptr + out_offsets,
            offsets.to(tl.int32),
            mask=active,
            sem="relaxed",
        )
        tl.store(next_ptr + offsets, previous, mask=active)


@triton.jit
def mthreads_scatter_reduce_prod_finalize_lists_kernel(
    inp_ptr,
    src_ptr,
    heads_ptr,
    next_ptr,
    result_ptr,
    out_numel,
    index_ncols,
    src_ncols,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Traverse independent source lists and multiply each source exactly once."""
    pid = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
        tl.int64
    ) * tl.num_programs(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < out_numel
    node = tl.load(heads_ptr + offsets, mask=mask, other=-1)
    touched = node >= 0
    if INCLUDE_SELF:
        value = tl.load(inp_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    else:
        value = tl.full((BLOCK,), 1.0, tl.float32)

    active = mask & touched
    done = ~active
    all_done = False
    while not all_done:
        safe_node = tl.where(active, node, 0).to(tl.int64)
        row = safe_node // index_ncols
        col = safe_node % index_ncols
        source = tl.load(
            src_ptr + row * src_ncols + col,
            mask=active,
            other=1.0,
        ).to(tl.float32)
        value = tl.where(active, value * source, value)
        node = tl.load(next_ptr + safe_node, mask=active, other=-1)
        active &= node >= 0
        done |= ~active
        all_done = tl.sum(done.to(tl.int32)) == BLOCK

    if not INCLUDE_SELF:
        inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        value = tl.where(touched, value, inp)
    tl.store(result_ptr + offsets, value, mask=mask)


def _same_tensor_mapping(lhs, rhs):
    return (
        lhs.data_ptr() == rhs.data_ptr()
        and lhs.shape == rhs.shape
        and lhs.stride() == rhs.stride()
    )


def _rowwise_result_is_safe(inp, src, result):
    """Reject partial storage overlap before a direct-output Triton launch."""
    if result is None:
        return True
    if _same_tensor_mapping(result, inp):
        return not torch._C._overlaps(result, src)
    return not torch._C._overlaps(result, inp) and not torch._C._overlaps(result, src)


def _try_rowwise_product(inp, dim, index, src, include_self, result=None):
    if (
        inp.numel() == 0
        or index.numel() == 0
        or not _rowwise_result_is_safe(inp, src, result)
    ):
        return None
    strategy = _select_rowwise_strategy(
        inp,
        dim,
        index,
        src,
        "prod",
        include_self,
        result,
    )
    if strategy is None:
        return None
    return _scatter_reduce_rowwise(
        inp,
        index,
        src,
        "prod",
        include_self,
        strategy,
        result=result,
    )


def _split_grid(programs):
    return min(programs, _MAX_GRID_X), triton.cdiv(programs, _MAX_GRID_X)


def _can_use_linked_product(inp, dim, index, src, result=None):
    if (
        inp.ndim != 2
        or dim not in (-1, 1)
        or inp.dtype not in (torch.float16, torch.float32, torch.bfloat16)
        or src.dtype != inp.dtype
        or index.dtype != torch.int64
        or not inp.is_contiguous()
        or not index.is_contiguous()
        or not src.is_contiguous()
        or inp.numel() == 0
        or index.numel() == 0
        or index.numel() >= 1 << 31
        or index.shape[0] > inp.shape[0]
        or index.shape[0] > src.shape[0]
        or index.shape[1] > src.shape[1]
    ):
        return False
    if result is None:
        return True
    return (
        result.shape == inp.shape
        and result.dtype == inp.dtype
        and result.device == inp.device
        and result.is_contiguous()
        and _rowwise_result_is_safe(inp, src, result)
    )


def _scatter_reduce_linked_product(inp, index, src, include_self, result=None):
    if result is None:
        result = torch.empty_like(inp)
    heads = torch.full(inp.shape, -1, dtype=torch.int32, device=inp.device)
    next_nodes = torch.empty(index.numel(), dtype=torch.int32, device=inp.device)
    build_programs = triton.cdiv(index.numel(), _LINK_BUILD_BLOCK * _LINK_BUILD_LOOP)
    finalize_programs = triton.cdiv(inp.numel(), _LINK_FINAL_BLOCK)

    with torch_device_fn.device(inp.device):
        mthreads_scatter_reduce_prod_build_lists_kernel[_split_grid(build_programs)](
            index,
            src,
            heads,
            next_nodes,
            index.numel(),
            index.shape[1],
            src.shape[1],
            inp.shape[1],
            include_self,
            BLOCK=_LINK_BUILD_BLOCK,
            LOOP=_LINK_BUILD_LOOP,
        )
        mthreads_scatter_reduce_prod_finalize_lists_kernel[
            _split_grid(finalize_programs)
        ](
            inp,
            src,
            heads,
            next_nodes,
            result,
            inp.numel(),
            index.shape[1],
            src.shape[1],
            include_self,
            BLOCK=_LINK_FINAL_BLOCK,
        )
    return result


def scatter_reduce(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS_MTHREADS SCATTER_REDUCE_TWO")
    if reduce == "prod":
        result = _try_rowwise_product(inp, dim, index, src, include_self)
        if result is not None:
            return result
        if _can_use_linked_product(inp, dim, index, src):
            return _scatter_reduce_linked_product(inp, index, src, include_self)
    return _generic_scatter_reduce(
        inp,
        dim,
        index,
        src,
        reduce,
        include_self=include_self,
    )


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS_MTHREADS SCATTER_REDUCE_TWO_")
    if reduce == "prod":
        result = _try_rowwise_product(
            inp,
            dim,
            index,
            src,
            include_self,
            result=inp,
        )
        if result is not None:
            return result
        if _can_use_linked_product(inp, dim, index, src, inp):
            return _scatter_reduce_linked_product(
                inp,
                index,
                src,
                include_self,
                result=inp,
            )
        result = _generic_scatter_reduce(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self=include_self,
        )
        inp.copy_(result)
        return inp
    return _generic_scatter_reduce_(
        inp,
        dim,
        index,
        src,
        reduce,
        include_self=include_self,
    )


def scatter_reduce_out(
    inp,
    dim,
    index,
    src,
    reduce,
    *,
    include_self=True,
    out=None,
):
    logger.debug("GEMS_MTHREADS SCATTER_REDUCE_TWO_OUT")
    if reduce == "prod":
        if out is not None and out.dtype != inp.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {inp.dtype}, but got {out.dtype} instead"
            )
        if out is not None:
            result = _try_rowwise_product(
                inp,
                dim,
                index,
                src,
                include_self,
                result=out,
            )
            if result is not None:
                return result
            if _can_use_linked_product(inp, dim, index, src, out):
                return _scatter_reduce_linked_product(
                    inp,
                    index,
                    src,
                    include_self,
                    result=out,
                )
        result = _generic_scatter_reduce(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self=include_self,
        )
        if out is not None:
            out.copy_(result)
            return out
        return result
    return _generic_scatter_reduce_out(
        inp,
        dim,
        index,
        src,
        reduce,
        include_self=include_self,
        out=out,
    )
