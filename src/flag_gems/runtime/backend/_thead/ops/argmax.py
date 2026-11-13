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

from flag_gems.ops.argmax import argmax as _generic_argmax
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
)
_ROW_MIN_N = 256
_BLOCK_MIN_N = 128
_STRIDED_BLOCK = 1024


@triton.jit
def _argmax_pair(a, ai, b, bi):
    replace = b > a
    tied = b == a
    if a.dtype.is_floating():
        replace = replace | ((b != b) & (a == a))
        tied = tied | ((b != b) & (a != a))
    replace = replace | (tied & (bi < ai))
    return tl.where(replace, b, a), tl.where(replace, bi, ai)


@libentry()
@triton.jit
def _argmax_row_kernel(inp, out, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    value_dtype: tl.constexpr = inp.dtype.element_ty
    if value_dtype == tl.float16 or value_dtype == tl.bfloat16:
        value_dtype = tl.float32
    if (
        value_dtype == tl.int1
        or value_dtype == tl.uint8
        or value_dtype == tl.int8
        or value_dtype == tl.int16
    ):
        value_dtype = tl.int32
    minimum = get_dtype_min(value_dtype)
    sentinel: tl.constexpr = 0x7FFFFFFF
    best_value = tl.full((), minimum, value_dtype)
    best_index = tl.full((), sentinel, tl.int32)
    base = row.to(tl.int64) * N

    for start in tl.range(0, N, BLOCK):
        col = start + lane
        mask = col < N
        values = tl.load(inp + base + col, mask=mask, other=minimum).to(value_dtype)
        indices = tl.where(mask, col, sentinel)
        tile_value, tile_index = tl.reduce((values, indices), 0, _argmax_pair)
        best_value, best_index = _argmax_pair(
            best_value, best_index, tile_value, tile_index
        )

    tl.store(out + row, best_index)


@libentry()
@triton.jit
def _argmax_block_kernel(
    inp,
    out,
    M,
    N,
    INNER,
    STRIDE_OUTER,
    STRIDE_REDUCE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    row_mask = rows < M
    outer = rows // INNER
    inner = rows % INNER
    base = inp + outer * STRIDE_OUTER + inner

    value_dtype: tl.constexpr = inp.dtype.element_ty
    if value_dtype == tl.float16 or value_dtype == tl.bfloat16:
        value_dtype = tl.float32
    if (
        value_dtype == tl.int1
        or value_dtype == tl.uint8
        or value_dtype == tl.int8
        or value_dtype == tl.int16
    ):
        value_dtype = tl.int32
    minimum = get_dtype_min(value_dtype)
    sentinel: tl.constexpr = 0x7FFFFFFF
    best_values = tl.full([BLOCK_M], minimum, value_dtype)
    best_indices = tl.full([BLOCK_M], sentinel, tl.int32)

    for start in tl.range(0, N, BLOCK_N):
        n_offset = start + tl.arange(0, BLOCK_N)
        mask = row_mask[:, None] & (n_offset[None, :] < N)
        pointers = base[:, None] + n_offset[None, :].to(tl.int64) * STRIDE_REDUCE
        values = tl.load(pointers, mask=mask, other=minimum).to(value_dtype)
        indices = tl.where(mask, n_offset[None, :], sentinel)
        tile_values, tile_indices = tl.reduce((values, indices), 1, _argmax_pair)
        best_values, best_indices = _argmax_pair(
            best_values, best_indices, tile_values, tile_indices
        )

    tl.store(out + rows, best_indices, row_mask)


@libentry()
@triton.jit
def _argmax_strided_kernel(
    inp,
    out,
    SIZE,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    INDEX64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK).to(tl.int64)
    value_dtype: tl.constexpr = inp.dtype.element_ty
    if value_dtype == tl.float16 or value_dtype == tl.bfloat16:
        value_dtype = tl.float32
    if (
        value_dtype == tl.int1
        or value_dtype == tl.uint8
        or value_dtype == tl.int8
        or value_dtype == tl.int16
    ):
        value_dtype = tl.int32
    index_dtype: tl.constexpr = tl.int64 if INDEX64 else tl.int32
    sentinel: tl.constexpr = 0x7FFFFFFFFFFFFFFF if INDEX64 else 0x7FFFFFFF
    minimum = get_dtype_min(value_dtype)
    best_values = tl.full([BLOCK], minimum, value_dtype)
    best_indices = tl.full([BLOCK], sentinel, index_dtype)

    for start in tl.range(0, SIZE, BLOCK):
        scan = start.to(tl.int64) + lane
        remaining = scan
        physical = tl.zeros([BLOCK], tl.int64)
        for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
            coordinate = remaining % SHAPE[axis]
            remaining = remaining // SHAPE[axis]
            physical += coordinate * STRIDES[axis]
        mask = scan < SIZE
        values = tl.load(inp + physical, mask=mask, other=minimum).to(value_dtype)
        indices = tl.where(mask, scan, sentinel).to(index_dtype)
        best_values, best_indices = _argmax_pair(
            best_values, best_indices, values, indices
        )

    _, index = tl.reduce((best_values, best_indices), 0, _argmax_pair)
    tl.store(out, index)


def _output_shape(inp, dim, keepdim):
    if dim is None:
        return [1] * inp.ndim if keepdim else []
    shape = list(inp.shape)
    if keepdim:
        shape[dim] = 1
    else:
        del shape[dim]
    return shape


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_THEAD ARGMAX")
    ndim = inp.ndim

    if dim is not None:
        original_dim = dim
        if ndim == 0:
            if original_dim not in (0, -1):
                raise IndexError(
                    "Dimension out of range (expected to be in range of [-1, 0], "
                    f"but got {original_dim})"
                )
            return torch.zeros([], dtype=torch.int64, device=inp.device)
        if original_dim < -ndim or original_dim >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-ndim}, {ndim - 1}], but got {original_dim})"
            )
        dim %= ndim

    if dim is None:
        if inp.numel() == 0:
            raise IndexError(
                "argmax(): Expected reduction dim to be specified for "
                "input.numel() == 0."
            )
        n = inp.numel()
    else:
        n = inp.shape[dim]
        if n == 0:
            raise IndexError(
                f"argmax(): Expected reduction dim {dim} to have non-zero size."
            )

    out_shape = _output_shape(inp, dim, keepdim)
    if inp.numel() == 0:
        return torch.empty(out_shape, dtype=torch.int64, device=inp.device)
    if n == 1:
        return torch.zeros(out_shape, dtype=torch.int64, device=inp.device)
    if inp.dtype not in _SUPPORTED_DTYPES:
        return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)

    if not inp.is_contiguous():
        if dim is not None:
            return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)
        size = inp.numel()
        out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
        with torch_device_fn.device(inp.device):
            _argmax_strided_kernel[(1,)](
                inp,
                out,
                size,
                tuple(inp.shape),
                tuple(inp.stride()),
                INDEX64=size > 0x7FFFFFFF,
                BLOCK=_STRIDED_BLOCK,
                num_warps=4,
            )
        return out

    if dim is None:
        return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)
    if n > 0x7FFFFFFF:
        return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)

    m = math.prod(inp.shape[:dim]) if dim else 1
    k = math.prod(inp.shape[dim + 1 :]) if dim + 1 < ndim else 1
    if k == 1 and n < _ROW_MIN_N:
        return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)
    if k > 1 and n < _BLOCK_MIN_N:
        return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)

    out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
    with torch_device_fn.device(inp.device):
        if k == 1:
            block = min(8192, triton.next_power_of_2(n))
            _argmax_row_kernel[(m,)](
                inp,
                out,
                n,
                BLOCK=block,
                num_warps=8 if block >= 4096 else 4,
            )
        else:
            rows = m * k
            block_m = 16 if k <= 128 else 32
            block_n = min(256, triton.next_power_of_2(n))
            _argmax_block_kernel[(triton.cdiv(rows, block_m),)](
                inp,
                out,
                rows,
                n,
                k,
                n * k,
                k,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=4,
            )
    return out


__all__ = ["argmax"]
