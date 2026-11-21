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

from flag_gems.ops.row_stack import (
    _allocate,
    _cast_tensors,
    _check_out,
    _prepare,
    _strides_for,
)
from flag_gems.ops.row_stack import row_stack as _generic_row_stack
from flag_gems.ops.row_stack import row_stack_out as _generic_row_stack_out

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)

# Every input contributes one dense run of ``rows * row_width`` elements to the
# output, so the operator is a list of block copies. Below this size the extra
# launch per input dominates and the generic pointwise kernel, which covers a
# group of inputs with a single launch, stays in charge.
_TLE_MIN_BYTES = 1 << 20

# A contiguous stack is a handful of flat copies, which one block-DMA kernel
# covers in a single launch. It beats the per-input TLE transfers up to a few
# MB (measured: 3 MB 8.8 us vs 14.3 us, 6 MB 13.2 us vs 17.0 us, 12 MB 21.7 us
# vs 24.8 us) and loses above ~24 MB (24 MB 38.7 us vs 37.7 us, 48 MB 71.3 us
# vs 66.2 us), where TLE's tile transfers take over.
_FLAT_MAX_BYTES = 8 << 20
_FLAT_MAX_INPUTS = 4


def _flat_block(n_elements: int) -> int:
    """Block-DMA tile for one flat copy, matching the copy_ fast path."""
    if n_elements >= 2**19:
        return 65536
    if n_elements >= 2**16:
        return 32768
    if n_elements >= 2**13:
        return 8192
    if n_elements >= 2**10:
        return 4096
    return 1024


@triton.jit
def _row_stack_flat_kernel(
    src0,
    src1,
    src2,
    src3,
    dst,
    n0,
    n1,
    n2,
    n3,
    b0,
    b1,
    b2,
    o1,
    o2,
    o3,
    BLOCK: tl.constexpr,
    N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Flat copy of ``N`` contiguous inputs into one contiguous output.

    Program ``pid`` belongs to the input whose block range contains it, so the
    whole stack is one launch: ``b0..b2`` are the per-input block counts and
    ``o1..o3`` the output offsets of inputs 1..3. ``NEED_MASK`` is False when
    every input fills its blocks exactly, which keeps the transfer on the
    unmasked block-DMA path.
    """
    pid = tl.program_id(0)
    ptr = src0
    base = 0
    width = n0
    out_base = 0
    if N > 1:
        take = pid >= b0
        ptr = tl.where(take, src1, ptr)
        base = tl.where(take, b0, base)
        width = tl.where(take, n1, width)
        out_base = tl.where(take, o1, out_base)
    if N > 2:
        take = pid >= b0 + b1
        ptr = tl.where(take, src2, ptr)
        base = tl.where(take, b0 + b1, base)
        width = tl.where(take, n2, width)
        out_base = tl.where(take, o2, out_base)
    if N > 3:
        take = pid >= b0 + b1 + b2
        ptr = tl.where(take, src3, ptr)
        base = tl.where(take, b0 + b1 + b2, base)
        width = tl.where(take, n3, width)
        out_base = tl.where(take, o3, out_base)
    off = (pid - base) * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = off < width
        tl.store(dst + out_base + off, tl.load(ptr + off, mask=mask), mask=mask)
    else:
        tl.store(dst + out_base + off, tl.load(ptr + off))


def _flat_stack(tensors, out):
    """Copy already-contiguous, same-dtype inputs into ``out`` in one launch."""
    if len(tensors) > _FLAT_MAX_INPUTS:
        return False
    counts = [t.numel() for t in tensors]
    block = _flat_block(max(counts))
    padded_n = counts + [0] * (_FLAT_MAX_INPUTS - len(counts))
    blocks = [triton.cdiv(count, block) for count in padded_n]
    grid = (sum(blocks),)
    if grid[0] == 0:
        return True
    out_base = [0] * _FLAT_MAX_INPUTS
    for i in range(1, _FLAT_MAX_INPUTS):
        out_base[i] = out_base[i - 1] + padded_n[i - 1]
    ptrs = list(tensors) + [tensors[0]] * (_FLAT_MAX_INPUTS - len(tensors))
    _row_stack_flat_kernel[grid](
        ptrs[0],
        ptrs[1],
        ptrs[2],
        ptrs[3],
        out,
        padded_n[0],
        padded_n[1],
        padded_n[2],
        padded_n[3],
        blocks[0],
        blocks[1],
        blocks[2],
        out_base[1],
        out_base[2],
        out_base[3],
        BLOCK=block,
        N=len(tensors),
        NEED_MASK=any(count % block for count in counts),
        num_warps=32,
        unroll_num=8,
        buffer_size_limit=1024,
    )
    return True


def _tle_stack(tensors, out, row_width, dtype):
    """Copy every input block into ``out`` with one transfer per input.

    Returns False, leaving ``out`` partially written and for the caller to
    overwrite, when a transfer cannot be expressed.
    """
    flat = out.view(-1)
    total_rows = 0
    for tensor in tensors:
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        rows = tensor.shape[0]
        if rows == 0:
            continue
        start = total_rows * row_width
        chunk = flat[start : start + rows * row_width].view(tensor.shape)
        if not tle_copy(tensor, chunk):
            return False
        total_rows += rows
    return True


def _dense_contiguous(tensors, memory_format, row_width, shape, strides, dtype, device):
    """Allocate the output and fill it with DMA block copies, or return None."""
    if memory_format != torch.contiguous_format or strides[0] != row_width:
        return None
    total_elements = sum(t.numel() for t in tensors)
    if total_elements * tensors[0].element_size() < _TLE_MIN_BYTES:
        return None
    out = _allocate(shape, strides, dtype, device)
    if not _tle_stack(
        _cast_tensors(tensors, dtype, memory_format), out, row_width, dtype
    ):
        return None
    return out


def row_stack(tensors: list):
    logger.debug("GEMS_KUNLUNXIN ROW_STACK")

    prepared, memory_format, dtype, row_width, output_shape = _prepare(tensors)
    if row_width == 0 or output_shape[0] == 0:
        return _allocate(
            output_shape,
            _strides_for(output_shape, memory_format),
            dtype,
            prepared[0].device,
        )

    strides = _strides_for(output_shape, memory_format)
    out = _dense_contiguous(
        prepared,
        memory_format,
        row_width,
        output_shape,
        strides,
        dtype,
        prepared[0].device,
    )
    if out is not None:
        return out
    return _generic_row_stack(tensors)


def row_stack_out(tensors: list, *, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN ROW_STACK_OUT")

    prepared, memory_format, dtype, row_width, output_shape = _prepare(tensors)
    if (
        row_width != 0
        and output_shape[0] != 0
        and memory_format == torch.contiguous_format
        and list(out.shape) == list(output_shape)
        and out.dtype == dtype
        and out.is_contiguous()
    ):
        # The dtype/overlap contract of ``cat_out`` still has to hold; the shape
        # already matches, so this validates only and never resizes.
        _check_out(out, output_shape, dtype, memory_format, prepared)
        nbytes = out.numel() * out.element_size()
        flat = (
            nbytes <= _FLAT_MAX_BYTES
            and len(prepared) <= _FLAT_MAX_INPUTS
            and all(t.dtype == dtype and t.is_contiguous() for t in prepared)
        )
        if flat and _flat_stack(prepared, out):
            return out
        if nbytes >= _TLE_MIN_BYTES:
            cast = _cast_tensors(prepared, dtype, memory_format)
            if _tle_stack(cast, out, row_width, dtype):
                return out
    return _generic_row_stack_out(tensors, out=out)
