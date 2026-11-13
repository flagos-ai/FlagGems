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

from flag_gems.ops.argmax import argmax as base_argmax
from flag_gems.ops.argmax import argmax_kernel_1 as argmax_all_kernel_1
from flag_gems.ops.argmax import argmax_kernel_2 as argmax_all_kernel_2
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


_SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def _flatten_dim(shape, dim):
    dim = dim % len(shape)
    n = shape[dim]
    inner = math.prod(shape[dim + 1 :]) if dim + 1 < len(shape) else 1
    outer = math.prod(shape[:dim]) if dim > 0 else 1
    return dim, n, inner, outer


def _supports_mthreads_argmax(inp):
    return inp.device.type == "musa" and inp.dtype in _SUPPORTED_DTYPES


@libentry()
@triton.jit
def _zero_indices_kernel(out_index, M, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_index + offsets, 0, mask=offsets < M)


@libentry()
@triton.jit
def _argmax_block_kernel(
    inp,
    out_index,
    M,
    N,
    INNER,
    STRIDE_OUTER,
    STRIDE_REDUCE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = rows.to(tl.int64)
    row_mask = rows < M

    outer_idx = rows // INNER
    inner_idx = rows % INNER
    base_ptr = inp + outer_idx * STRIDE_OUTER + inner_idx

    dtype = inp.type.element_ty
    acc_dtype = tl.float32 if dtype is tl.float16 or dtype is tl.bfloat16 else dtype
    min_value = get_dtype_min(dtype)
    max_values = tl.full([BLOCK_M], min_value, acc_dtype)
    argmax_values = tl.zeros([BLOCK_M], tl.int64)

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N).to(tl.int64)
        mask = row_mask[:, None] & (n_offset[None, :] < N)
        inp_ptrs = base_ptr[:, None] + n_offset[None, :] * STRIDE_REDUCE
        inp_vals = tl.load(
            inp_ptrs,
            mask=mask,
            other=min_value,
            cache_modifier=".cg",
        )
        if dtype is tl.float16 or dtype is tl.bfloat16:
            inp_vals = inp_vals.to(tl.float32)
        local_max, local_argmax = tl.max(
            inp_vals,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        local_argmax = local_argmax.to(tl.int64) + start_n
        update = local_max > max_values
        max_values = tl.where(update, local_max, max_values)
        argmax_values = tl.where(update, local_argmax, argmax_values)

    tl.store(out_index + rows, argmax_values, mask=row_mask)


@libentry()
@triton.jit
def _argmax_block_native_kernel(
    inp,
    out_index,
    M,
    N,
    INNER,
    STRIDE_OUTER,
    STRIDE_REDUCE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = rows.to(tl.int64)
    row_mask = rows < M

    outer_idx = rows // INNER
    inner_idx = rows % INNER
    base_ptr = inp + outer_idx * STRIDE_OUTER + inner_idx

    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)
    max_values = tl.full([BLOCK_M], min_value, tl.float32)
    argmax_values = tl.zeros([BLOCK_M], tl.int64)

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N).to(tl.int64)
        mask = row_mask[:, None] & (n_offset[None, :] < N)
        inp_ptrs = base_ptr[:, None] + n_offset[None, :] * STRIDE_REDUCE
        inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
        local_max, local_argmax = tl.max(
            inp_vals,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        local_max = local_max.to(tl.float32)
        local_argmax = local_argmax.to(tl.int64) + start_n
        update = local_max > max_values
        max_values = tl.where(update, local_max, max_values)
        argmax_values = tl.where(update, local_argmax, argmax_values)

    tl.store(out_index + rows, argmax_values, mask=row_mask)


@libentry()
@triton.jit
def _argmax_contiguous_kernel(
    inp,
    out_index,
    M,
    N,
    STRIDE_OUTER,
    STRIDE_REDUCE,
    BLOCK_N: tl.constexpr,
    CACHE_CG: tl.constexpr,
):
    rows = ext.program_id(0) + tl.arange(0, 1).to(tl.int64)
    row_mask = rows < M
    base_ptr = inp + rows * STRIDE_OUTER

    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)
    max_values = tl.full([1], min_value, tl.float32)
    argmax_values = tl.zeros([1], tl.int64)

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N).to(tl.int64)
        mask = row_mask[:, None] & (n_offset[None, :] < N)
        inp_ptrs = base_ptr[:, None] + n_offset[None, :] * STRIDE_REDUCE
        if CACHE_CG:
            inp_vals = tl.load(
                inp_ptrs,
                mask=mask,
                other=min_value,
                cache_modifier=".cg",
            )
        else:
            inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
        local_max, local_argmax = tl.max(
            inp_vals,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        local_max = local_max.to(tl.float32)
        local_argmax = local_argmax.to(tl.int64) + start_n
        update = local_max > max_values
        max_values = tl.where(update, local_max, max_values)
        argmax_values = tl.where(update, local_argmax, argmax_values)

    tl.store(out_index + rows, argmax_values, mask=row_mask)


def _argmax_all(inp, keepdim=False, dtype=None):
    m = inp.numel()
    if dtype is None:
        dtype = inp.dtype

    sqrt_block = triton.next_power_of_2(math.ceil(math.sqrt(m)))
    block_size = min(sqrt_block, 8192)
    mid_size = triton.cdiv(m, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid_value = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
    if keepdim:
        out_shape = [1] * inp.dim()
        out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
    else:
        out = torch.empty([], dtype=torch.int64, device=inp.device)

    with torch_device_fn.device(inp.device):
        argmax_all_kernel_1[(mid_size, 1, 1)](
            inp,
            mid_value,
            mid_index,
            m,
            block_size,
        )
        argmax_all_kernel_2[(1, 1, 1)](
            mid_value,
            mid_index,
            out,
            mid_size,
            block_mid,
        )
    return out


def _block_config(dtype, n, inner):
    if n <= 32:
        return 32, triton.next_power_of_2(n), 8, 1
    if inner <= 128:
        block_m = 16 if dtype == torch.float32 or n >= 512 else 32
        block_n = min(512, triton.next_power_of_2(n))
        num_warps = 8 if dtype != torch.float32 or n >= 512 else 4
        return block_m, block_n, num_warps, 2
    if dtype in (torch.float16, torch.bfloat16):
        return 32, 256, 8, 2
    return 64, 64, 4, 1


def _argmax_blocks(inp, out_index, m, n, inner, stride_outer, stride_reduce):
    if inp.dtype == torch.bfloat16 and m >= 65536:
        with torch_device_fn.device(inp.device):
            _argmax_block_native_kernel[(triton.cdiv(m, 32),)](
                inp,
                out_index,
                m,
                n,
                inner,
                stride_outer,
                stride_reduce,
                BLOCK_M=32,
                BLOCK_N=256,
                num_warps=4,
                num_stages=2,
            )
        return

    block_m, block_n, num_warps, num_stages = _block_config(inp.dtype, n, inner)
    with torch_device_fn.device(inp.device):
        _argmax_block_kernel[(triton.cdiv(m, block_m),)](
            inp,
            out_index,
            m,
            n,
            inner,
            stride_outer,
            stride_reduce,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )


def _argmax_contiguous(inp, out_index, m, n, stride_outer, stride_reduce):
    if n <= 4096:
        block_n = triton.next_power_of_2(n)
        if block_n == 4096:
            block_n = 4096 if m <= 1024 else 2048
        num_warps = 2 if block_n <= 256 else 4
        num_stages = 2 if block_n <= 64 or block_n == 4096 else 1
    else:
        block_n = 4096
        num_warps = 4 if inp.dtype == torch.float32 else 8
        num_stages = 2
    with torch_device_fn.device(inp.device):
        _argmax_contiguous_kernel[(m,)](
            inp,
            out_index,
            m,
            n,
            stride_outer,
            stride_reduce,
            BLOCK_N=block_n,
            CACHE_CG=n >= 4096,
            num_warps=num_warps,
            num_stages=num_stages,
        )


def _prefer_generic_non_inner(inp, m, n, inner):
    is_power_of_two = n > 0 and (n & (n - 1)) == 0
    if n <= 32:
        return is_power_of_two
    if inp.dtype in (torch.float16, torch.bfloat16) and inner <= 128 and n <= 256:
        return n > 128 or is_power_of_two
    if inp.dtype == torch.float16 and inner >= 512 and n >= 512 and m >= 65536:
        return True
    return False


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_MTHREADS ARGMAX")
    if inp.numel() == 0:
        return base_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)

    if not _supports_mthreads_argmax(inp):
        if not inp.is_contiguous():
            inp = inp.contiguous()
        return base_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)

    if dim is None:
        if not inp.is_contiguous():
            inp = inp.contiguous()
        return _argmax_all(inp, keepdim=keepdim, dtype=dtype)

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim

    shape = list(inp.shape)
    dim, n, inner, outer = _flatten_dim(shape, dim)
    m = outer * inner

    out_shape = shape.copy()
    out_shape[dim] = 1
    if n == 1:
        out_index = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
        out_numel = out_index.numel()
        block = 64 if out_numel <= 8192 else 256
        num_warps = 1 if out_numel <= 8192 else 4
        with torch_device_fn.device(inp.device):
            _zero_indices_kernel[(triton.cdiv(out_numel, block),)](
                out_index,
                out_numel,
                BLOCK=block,
                num_warps=num_warps,
            )
        if not keepdim:
            out_index = torch.squeeze(out_index, dim)
        return out_index

    if not inp.is_contiguous():
        inp = inp.contiguous()

    # Contiguous reductions do not benefit from cross-row coalescing. A
    # row-per-CTA kernel avoids the generic dispatcher overhead and streams
    # large rows with explicit tiles; very short rows stay on the generic path.
    if inner == 1:
        if n > 32:
            stride = inp.stride()
            stride_reduce = stride[dim]
            stride_outer = stride_reduce * n
            out_index = torch.empty((m,), dtype=torch.int64, device=inp.device)
            _argmax_contiguous(
                inp,
                out_index,
                m,
                n,
                stride_outer,
                stride_reduce,
            )
            out_index = out_index.view(out_shape)
            if not keepdim:
                out_index = torch.squeeze(out_index, dim)
            return out_index
        return base_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)

    if _prefer_generic_non_inner(inp, m, n, inner):
        return base_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)

    stride = inp.stride()
    stride_reduce = stride[dim]
    stride_outer = stride_reduce * n
    out_index = torch.empty((m,), dtype=torch.int64, device=inp.device)

    _argmax_blocks(
        inp,
        out_index,
        m,
        n,
        inner,
        stride_outer,
        stride_reduce,
    )

    out_index = out_index.view(out_shape)
    if not keepdim:
        out_index = torch.squeeze(out_index, dim)
    return out_index


__all__ = ["argmax"]
