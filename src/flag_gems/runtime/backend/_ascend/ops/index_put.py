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

import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.index_put import index_put_ as generic_index_put_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext


@triton.jit
def _index_put_first_dim_kernel(
    input_ptr,
    indices_ptr,
    values_ptr,
    n_rows,
    row_size,
    input_rows,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    total = n_rows * row_size
    for start in range(pid * BLOCK, total, tl.num_programs(0) * BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        row = offsets // row_size
        col = offsets % row_size
        mask = offsets < total
        index = tl.load(indices_ptr + row, mask=mask, other=0)
        index = tl.where(index < 0, index + input_rows, index)
        valid = mask & (index >= 0) & (index < input_rows)
        value = tl.load(values_ptr + offsets, mask=mask, other=0)
        tl.store(input_ptr + index * row_size + col, value, mask=valid)


@triton.jit
def _index_put_pairs_scalar_kernel(
    input_ptr,
    row_indices_ptr,
    col_indices_ptr,
    value_ptr,
    n_indices,
    n_rows,
    n_cols,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    for start in range(pid * BLOCK, n_indices, tl.num_programs(0) * BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        in_range = offsets < n_indices
        rows = tl.load(row_indices_ptr + offsets, mask=in_range, other=0)
        cols = tl.load(col_indices_ptr + offsets, mask=in_range, other=0)
        rows = tl.where(rows < 0, rows + n_rows, rows)
        cols = tl.where(cols < 0, cols + n_cols, cols)
        valid = in_range & (rows >= 0) & (rows < n_rows)
        valid = valid & (cols >= 0) & (cols < n_cols)
        value = tl.load(value_ptr)
        tl.store(input_ptr + rows * n_cols + cols, value, mask=valid)


def index_put(inp, indices, values, accumulate=False):
    out = inp.clone()
    return index_put_(out, indices, values, accumulate)


def index_put_(inp, indices, values, accumulate=False):
    if values.dtype != inp.dtype:
        raise RuntimeError(
            "index_put_ requires source and destination tensors to have the same dtype"
        )
    # The generic two-dimensional generated kernel aborts in Ascend's LLVM
    # lowering for Qwen GDN's contiguous first-dimension state write.
    if (
        not accumulate
        and len(indices) >= 1
        and len(indices) <= inp.ndim
        and all(index is None for index in indices[1:])
        and isinstance(indices[0], torch.Tensor)
        and indices[0].ndim == 1
        and indices[0].is_contiguous()
        and indices[0].dtype in (torch.int32, torch.int64)
        and indices[0].device == inp.device
        and inp.is_contiguous()
        and values.is_contiguous()
        and values.device == inp.device
        and indices[0].numel() * math.prod(inp.shape[1:]) < 2**31
        and (
            tuple(values.shape) == (indices[0].numel(), *inp.shape[1:])
            or (indices[0].numel() == 1 and tuple(values.shape) == tuple(inp.shape[1:]))
        )
    ):
        n_rows = indices[0].numel()
        row_size = math.prod(inp.shape[1:])
        total = n_rows * row_size
        if total == 0:
            return inp
        grid = (min(triton.cdiv(total, 1024), 65535),)
        with torch_device_fn.device(inp.device):
            _index_put_first_dim_kernel[grid](
                inp, indices[0], values, n_rows, row_size, inp.shape[0], BLOCK=1024
            )
        return inp
    # vLLM min_tokens masks (request, stop-token) pairs with one scalar -inf.
    # The generic generated rank-2 kernel fails FlagTree's MLIR conversion.
    if (
        not accumulate
        and inp.ndim == 2
        and inp.is_contiguous()
        and len(indices) == 2
        and all(isinstance(index, torch.Tensor) for index in indices)
        and all(index.ndim == 1 for index in indices)
        and all(index.is_contiguous() for index in indices)
        and all(index.dtype in (torch.int32, torch.int64) for index in indices)
        and all(index.device == inp.device for index in indices)
        and indices[0].numel() == indices[1].numel()
        and values.ndim == 0
        and values.device == inp.device
    ):
        n_indices = indices[0].numel()
        if n_indices == 0:
            return inp
        grid = (min(triton.cdiv(n_indices, 256), 65535),)
        with torch_device_fn.device(inp.device):
            _index_put_pairs_scalar_kernel[grid](
                inp,
                indices[0],
                indices[1],
                values,
                n_indices,
                inp.shape[0],
                inp.shape[1],
                BLOCK=256,
            )
        return inp
    return generic_index_put_(inp, indices, values, accumulate)
