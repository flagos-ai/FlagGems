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

"""Kunlunxin override for ``row_stack`` / ``row_stack_out``.

Two fast paths are combined:

* ``tle_copy`` block DMA for large contiguous outputs (one transfer per
  input), used when the total payload reaches ``_TLE_MIN_BYTES``.
* A bounded-tile Triton kernel for the remaining dense copies.  The generic
  kernel autotunes ``BLOCK_SIZE`` only up to 4096 and always runs the
  masked-memory path (an always-true mask still costs ~2x on XPU); this path
  uses the recipe already used by the Kunlunxin ``vstack`` / ``copy_``
  overrides: a fixed block-size ladder plus a scalar full-block / masked-tail
  split, so only the last partial block of each input pays the masked path.

``row_stack_out`` is registered here as well; its TLE fast path keeps the
contiguous large-output conditions and the bounded-tile kernel covers smaller
outputs and outputs TLE cannot express.  Non-contiguous memory formats keep
the generic implementation.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.row_stack import _allocate, _cast_tensors, _check_out, _prepare
from flag_gems.ops.row_stack import _run_kernel as _generic_run_kernel
from flag_gems.ops.row_stack import _strides_for
from flag_gems.ops.row_stack import row_stack as _generic_row_stack
from flag_gems.ops.row_stack import row_stack_out as _generic_row_stack_out
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)

# Every input contributes one dense run of ``rows * row_width`` elements to the
# output, so the operator is a list of block copies. Below this size the extra
# launch per input dominates and the bounded-tile kernel, which covers a group
# of inputs with a single launch, stays in charge.
_TLE_MIN_BYTES = 1 << 20


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


def _pick_row_stack_block(n_elements: int) -> int:
    """Bounded tile size (same graduation as copy.py::_pick_flat_block).

    The generic autotune capped the tile at 4096 elements; on large copies that
    leaves too few contiguous bytes per DMA run.  A bounded tile keeps every
    access a contiguous block DMA while still spreading large shapes across
    enough programs.
    """
    if n_elements >= 2**19:
        return 65536
    if n_elements >= 2**16:
        return 32768
    if n_elements >= 2**13:
        return 8192
    if n_elements >= 2**10:
        return 4096
    return 1024


@libentry()
@triton.jit
def row_stack_kernel(
    itensor_ptr0,
    itensor_ptr1,
    itensor_ptr2,
    itensor_ptr3,
    output_ptr,
    local_row0,
    local_row1,
    local_row2,
    local_row3,
    exc_row_offset0,
    exc_row_offset1,
    exc_row_offset2,
    exc_row_offset3,
    total_row_offset,
    row_stride,
    max_tile_elems,
    BLOCK_SIZE: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    tensor_idx = tl.program_id(axis=1)
    col_idx = tl.arange(0, BLOCK_SIZE)

    intensor_ptr = tl.where(tensor_idx == 0, itensor_ptr0, itensor_ptr1)
    intensor_ptr = tl.where(tensor_idx == 2, itensor_ptr2, intensor_ptr)
    intensor_ptr = tl.where(tensor_idx == 3, itensor_ptr3, intensor_ptr)
    base_exc_row_idx = tl.where(tensor_idx == 0, exc_row_offset0, exc_row_offset1)
    base_exc_row_idx = tl.where(tensor_idx == 2, exc_row_offset2, base_exc_row_idx)
    base_exc_row_idx = tl.where(tensor_idx == 3, exc_row_offset3, base_exc_row_idx)
    local_row = tl.where(tensor_idx == 0, local_row0, local_row1)
    local_row = tl.where(tensor_idx == 2, local_row2, local_row)
    local_row = tl.where(tensor_idx == 3, local_row3, local_row)

    # Rows are dense runs of ``row_width`` elements and the destination row
    # pitch equals that width on this path, so a flat tile spans consecutive
    # rows in both operands.
    #
    # Lane offsets stay int32 (an int64 lane compare trips the XPU legalizer:
    # "arith.cmpi ... result type has i1 element type").  The host side only
    # takes this path when max_tile_elems < 2**31, so the int32 tile index
    # cannot overflow; the row-base offset is widened to int64 before the
    # multiply because total rows * row stride can exceed int32.
    end_idx = local_row * row_stride
    block_start = pid_x * BLOCK_SIZE
    idx = block_start + col_idx
    offset_mask = idx < end_idx
    idx64 = idx.to(tl.int64)
    in_offset = intensor_ptr + idx64
    row_stride_offset = (total_row_offset + base_exc_row_idx).to(tl.int64) * row_stride
    out_base = output_ptr + row_stride_offset
    out_offset = out_base + idx64
    if block_start + BLOCK_SIZE <= end_idx:
        # Full block: unmasked block-DMA. The always-true masked-memory path
        # costs ~2x on XPU, so only the tail block of each tensor is masked.
        tl.store(out_offset, tl.load(in_offset))
    elif block_start < end_idx:
        tl.store(out_offset, tl.load(in_offset, mask=offset_mask), mask=offset_mask)


def _run_kernel(tensors, output, row_width, out_row_stride):
    """Copy every input row block into ``output`` through the Triton kernel."""
    if max(tensor.shape[0] for tensor in tensors) * row_width >= 2**31:
        # The fast kernel indexes a tile in int32 (see the comment in the
        # kernel); tensors beyond that bound go through the generic int64
        # kernel instead of overflowing.
        return _generic_run_kernel(tensors, output, row_width, out_row_stride)
    num_tensors = len(tensors)
    outer_iters = triton.cdiv(num_tensors, 4)
    total_row_offset = 0
    for i in range(outer_iters):
        max_rows = 1
        itensors = []
        exclusive_row = []
        local_row = []
        array_row_offset = 0
        scheduled_num_tensors = 0
        for j in range(4):
            tensor_idx = i * 4 + j
            if tensor_idx < num_tensors:
                scheduled_num_tensors += 1
                tensor = tensors[tensor_idx]
                itensors.append(tensor)
                local_row.append(tensor.shape[0])
                exclusive_row.append(array_row_offset)
                array_row_offset += tensor.shape[0]
                max_rows = max(max_rows, tensor.shape[0])
            else:
                empty_tensor = torch.empty(
                    0, dtype=tensors[0].dtype, device=tensors[0].device
                )
                itensors.append(empty_tensor)
                local_row.append(local_row[-1])
                exclusive_row.append(exclusive_row[-1])
        max_tile_elems = max_rows * row_width
        block_size = _pick_row_stack_block(max_tile_elems)
        grid = (triton.cdiv(max_tile_elems, block_size), scheduled_num_tensors)
        # Launch the kernel
        with torch_device_fn.device(tensors[0].device):
            row_stack_kernel[grid](
                itensors[0],
                itensors[1],
                itensors[2],
                itensors[3],
                output,
                local_row[0],
                local_row[1],
                local_row[2],
                local_row[3],
                exclusive_row[0],
                exclusive_row[1],
                exclusive_row[2],
                exclusive_row[3],
                total_row_offset,
                out_row_stride,
                max_tile_elems,
                BLOCK_SIZE=block_size,
                num_warps=32,
                unroll_num=8,
                buffer_size_limit=1024,
            )
            total_row_offset += array_row_offset
    return output


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
    if memory_format != torch.contiguous_format:
        return _generic_row_stack(tensors)

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
    out = _allocate(output_shape, strides, dtype, prepared[0].device)
    c_tensors = _cast_tensors(prepared, dtype, memory_format)
    return _run_kernel(c_tensors, out, row_width, strides[0])


def row_stack_out(tensors: list, *, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN ROW_STACK_OUT")

    prepared, memory_format, dtype, row_width, output_shape = _prepare(tensors)
    _check_out(out, output_shape, dtype, memory_format, prepared)
    if row_width == 0 or output_shape[0] == 0:
        return out
    if memory_format != torch.contiguous_format:
        return _generic_row_stack_out(tensors, out=out)

    c_tensors = _cast_tensors(prepared, dtype, memory_format)
    if (
        out.dtype == dtype
        and out.is_contiguous()
        and out.numel() * out.element_size() >= _TLE_MIN_BYTES
    ):
        if _tle_stack(c_tensors, out, row_width, dtype):
            return out

    # Fast path: ``out`` stores rows as dense runs of ``row_width`` elements and
    # its stride matches the layout the kernel assumes, so a single bounded-tile
    # launch covers every input.  Only an arbitrary strided view falls through
    # to the staging copy below.
    fast = out.dtype == dtype and tuple(out.stride()) == tuple(
        _strides_for(list(out.shape), memory_format)
    )
    if fast:
        return _run_kernel(c_tensors, out, row_width, out.stride(0))

    # Otherwise scatter through a dense staging buffer; ``copy_`` is
    # stride-aware and also honours ``out``'s dtype.  On Kunlunxin this is a
    # device-to-device movement (the vendor copy kernel), never a CPU path.
    strides = _strides_for(output_shape, memory_format)
    staging = _allocate(output_shape, strides, dtype, out.device)
    _run_kernel(c_tensors, staging, row_width, strides[0])
    out.copy_(staging)
    return out
