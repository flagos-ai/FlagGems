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

from flag_gems import runtime
from flag_gems.ops.row_stack import (
    _allocate,
    _cast_tensors,
    _check_out,
    _prepare,
    _strides_for,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("vstack"),
    key=["max_tile_elems"],
)
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
    # GCU300 keeps every lane-varying offset in int32.  The generic kernel
    # widens `pid * BLOCK_SIZE + arange` to int64 before masking and addressing,
    # and this target miscompiles lane-varying int64 arithmetic once a program
    # covers 256 or more lanes: with BLOCK_SIZE >= 256 the copy permutes whole
    # rows (row r received the contents of row r+1 and vice versa), while the
    # same kernel at BLOCK_SIZE=128 is exact.  `ext.program_id` is itself a
    # `to(tl.int64)`, so the tile index comes from `tl.program_id` here, and the
    # row-base offset is applied to the pointer before the vector is added, so
    # the vector arithmetic never leaves int32.
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

    end_idx = local_row * row_stride
    idx = pid_x * BLOCK_SIZE + col_idx
    offset_mask = idx < end_idx

    row_stride_offset = (total_row_offset + base_exc_row_idx) * row_stride
    out_base = output_ptr + row_stride_offset

    out = tl.load(intensor_ptr + idx, mask=offset_mask)
    tl.store(out_base + idx, out, mask=offset_mask)


def _run_kernel(tensors, output, row_width, out_row_stride):
    """Copy every input row block into ``output`` through the Triton kernel."""
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
        grid = lambda META: (
            triton.cdiv(max_tile_elems, META["BLOCK_SIZE"]),
            scheduled_num_tensors,
        )
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
            )
            total_row_offset += array_row_offset
    return output


def _kernel_can_address(dtype):
    """Whether the copy kernel can be compiled for ``dtype`` on GCU300."""
    return dtype not in (torch.float64, torch.complex128)


def _stack_by_copy(tensors, output):
    """Concatenate along dim 0 with aten's strided copy.

    Used for the dtypes whose loads the Triton kernel cannot lower here.
    """
    row = 0
    for tensor in tensors:
        n = tensor.shape[0]
        output[row : row + n].copy_(tensor)
        row += n
    return output


def row_stack(tensors: list):
    logger.debug("GEMS_ENFLAME ROW_STACK")

    tensors, memory_format, dtype, row_width, output_shape = _prepare(tensors)
    device = tensors[0].device
    if row_width == 0 or output_shape[0] == 0:
        return _allocate(
            output_shape, _strides_for(output_shape, memory_format), dtype, device
        )

    strides = _strides_for(output_shape, memory_format)
    output = _allocate(output_shape, strides, dtype, device)
    c_tensors = _cast_tensors(tensors, dtype, memory_format)
    if not _kernel_can_address(dtype):
        # float64 has no addressable gather on this backend (the kernel fails
        # to link on `tops_gather_load_fp64`), so fall back to aten's strided
        # copy, which does work for 64-bit values.
        return _stack_by_copy(c_tensors, output)
    return _run_kernel(c_tensors, output, row_width, strides[0])


def row_stack_out(tensors: list, *, out):
    logger.debug("GEMS_ENFLAME ROW_STACK_OUT")

    tensors, memory_format, dtype, row_width, output_shape = _prepare(tensors)
    _check_out(out, output_shape, dtype, memory_format, tensors)

    if row_width == 0 or output_shape[0] == 0:
        return out

    c_tensors = _cast_tensors(tensors, dtype, memory_format)

    fast = out.dtype == dtype and tuple(out.stride()) == tuple(
        _strides_for(list(out.shape), memory_format)
    )
    if fast:
        if not _kernel_can_address(dtype):
            return _stack_by_copy(c_tensors, out)
        return _run_kernel(c_tensors, out, row_width, out.stride(0))

    strides = _strides_for(output_shape, memory_format)
    staging = _allocate(output_shape, strides, dtype, out.device)
    _run_kernel(c_tensors, staging, row_width, strides[0])
    out.copy_(staging)
    return out
