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
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max, get_dtype_min

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
    out_value,
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
    max_value = tl.full((), get_dtype_max(dtype), dtype)
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
        inp_vals = tl.minimum(inp_vals, max_value)
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

    tl.store(out_value + rows, max_values, mask=row_mask)
    tl.store(out_index + rows, argmax_values, mask=row_mask)


@libentry()
@triton.jit
def _argmax_block_native_kernel(
    inp,
    out_index,
    out_value,
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
    max_value = tl.full((), get_dtype_max(dtype), dtype)
    max_values = tl.full([BLOCK_M], min_value, tl.float32)
    argmax_values = tl.zeros([BLOCK_M], tl.int64)

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N).to(tl.int64)
        mask = row_mask[:, None] & (n_offset[None, :] < N)
        inp_ptrs = base_ptr[:, None] + n_offset[None, :] * STRIDE_REDUCE
        inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
        inp_vals = tl.minimum(inp_vals, max_value)
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

    tl.store(out_value + rows, max_values, mask=row_mask)
    tl.store(out_index + rows, argmax_values, mask=row_mask)


@libentry()
@triton.jit
def _argmax_block_nan_fix_kernel(
    inp,
    out_index,
    max_values,
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
    candidates = tl.load(max_values + rows, mask=row_mask, other=float("-inf"))
    fix_mask = row_mask & (candidates == float("inf"))
    if tl.max(fix_mask.to(tl.int32), axis=0) != 0:
        outer_idx = rows // INNER
        inner_idx = rows % INNER
        base_ptr = inp + outer_idx * STRIDE_OUTER + inner_idx
        min_value = get_dtype_min(inp.type.element_ty)
        nan_values = tl.full([BLOCK_M], 0x7FFFFFFFFFFFFFFF, tl.int64)

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
            block_nan = tl.min(
                tl.where(
                    inp_vals != inp_vals,
                    tl.arange(0, BLOCK_N)[None, :],
                    0x7FFFFFFF,
                ),
                1,
            )
            block_nan = tl.where(
                block_nan != 0x7FFFFFFF,
                start_n + block_nan.to(tl.int64),
                0x7FFFFFFFFFFFFFFF,
            )
            nan_values = tl.minimum(nan_values, block_nan)

        update = fix_mask & (nan_values != 0x7FFFFFFFFFFFFFFF)
        tl.store(out_index + rows, nan_values, mask=update)


@libentry()
@triton.jit
def _argmax_contiguous_kernel(
    inp,
    out_index,
    out_value,
    M,
    N,
    STRIDE_OUTER,
    STRIDE_REDUCE,
    BLOCK_N: tl.constexpr,
    CACHE_CG: tl.constexpr,
    FIX_IN_KERNEL: tl.constexpr,
    ENCODE_INF: tl.constexpr,
):
    rows = ext.program_id(0) + tl.arange(0, 1).to(tl.int64)
    row_mask = rows < M
    base_ptr = inp + rows * STRIDE_OUTER

    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)
    max_value = tl.full((), get_dtype_max(dtype), dtype)
    max_values = tl.full([1], min_value, tl.float32)
    argmax_values = tl.zeros([1], tl.int64)

    if FIX_IN_KERNEL:
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
            inp_vals = tl.minimum(inp_vals, max_value)
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

        needs_fix = tl.max((max_values == float("inf")).to(tl.int32), axis=0)
        if needs_fix != 0:
            nan_values = tl.full([1], 0x7FFFFFFFFFFFFFFF, tl.int64)
            for start_n in range(0, N, BLOCK_N):
                n_offset = start_n + tl.arange(0, BLOCK_N).to(tl.int64)
                mask = row_mask[:, None] & (n_offset[None, :] < N)
                inp_ptrs = base_ptr[:, None] + n_offset[None, :] * STRIDE_REDUCE
                inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
                block_nan = tl.min(
                    tl.where(
                        inp_vals != inp_vals,
                        tl.arange(0, BLOCK_N)[None, :],
                        0x7FFFFFFF,
                    ),
                    1,
                )
                block_nan = tl.where(
                    block_nan != 0x7FFFFFFF,
                    start_n + block_nan.to(tl.int64),
                    0x7FFFFFFFFFFFFFFF,
                )
                nan_values = tl.minimum(nan_values, block_nan)
            fix = (max_values == float("inf")) & (nan_values != 0x7FFFFFFFFFFFFFFF)
            argmax_values = tl.where(fix, nan_values, argmax_values)
        tl.store(out_index + rows, argmax_values, mask=row_mask)
    else:
        full_tiles = N // BLOCK_N
        for tile in range(0, full_tiles):
            n_offset = tile * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
            inp_ptrs = base_ptr[:, None] + n_offset[None, :] * STRIDE_REDUCE
            if CACHE_CG:
                inp_vals = tl.load(inp_ptrs, cache_modifier=".cg")
            else:
                inp_vals = tl.load(inp_ptrs)
            inp_vals = tl.minimum(inp_vals, max_value)
            local_max, local_argmax = tl.max(
                inp_vals,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            local_max = local_max.to(tl.float32)
            local_argmax = local_argmax.to(tl.int64) + tile * BLOCK_N
            update = local_max > max_values
            max_values = tl.where(update, local_max, max_values)
            argmax_values = tl.where(update, local_argmax, argmax_values)

        remainder = N - full_tiles * BLOCK_N
        if remainder != 0:
            n_offset = full_tiles * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
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
            inp_vals = tl.minimum(inp_vals, max_value)
            local_max, local_argmax = tl.max(
                inp_vals,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            local_max = local_max.to(tl.float32)
            local_argmax = local_argmax.to(tl.int64) + full_tiles * BLOCK_N
            update = local_max > max_values
            max_values = tl.where(update, local_max, max_values)
            argmax_values = tl.where(update, local_argmax, argmax_values)

        if ENCODE_INF:
            encoded = argmax_values | tl.where(
                max_values == float("inf"), -0x8000000000000000, 0
            )
            tl.store(out_index + rows, encoded, mask=row_mask)
        else:
            tl.store(out_value + rows, max_values, mask=row_mask)
            tl.store(out_index + rows, argmax_values, mask=row_mask)


@libentry()
@triton.jit
def _argmax_contiguous_nan_fix_kernel(
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
    encoded = tl.load(out_index + rows, mask=row_mask, other=0)
    if tl.max((encoded < 0).to(tl.int32), axis=0) != 0:
        base_ptr = inp + rows * STRIDE_OUTER
        min_value = get_dtype_min(inp.type.element_ty)
        nan_values = tl.full([1], 0x7FFFFFFFFFFFFFFF, tl.int64)

        full_tiles = N // BLOCK_N
        for tile in range(0, full_tiles):
            n_offset = tile * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
            inp_ptrs = base_ptr[:, None] + n_offset[None, :] * STRIDE_REDUCE
            if CACHE_CG:
                inp_vals = tl.load(inp_ptrs, cache_modifier=".cg")
            else:
                inp_vals = tl.load(inp_ptrs)
            block_nan = tl.min(
                tl.where(
                    inp_vals != inp_vals,
                    tl.arange(0, BLOCK_N)[None, :],
                    0x7FFFFFFF,
                ),
                1,
            )
            block_nan = tl.where(
                block_nan != 0x7FFFFFFF,
                tile * BLOCK_N + block_nan.to(tl.int64),
                0x7FFFFFFFFFFFFFFF,
            )
            nan_values = tl.minimum(nan_values, block_nan)

        remainder = N - full_tiles * BLOCK_N
        if remainder != 0:
            n_offset = full_tiles * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
            mask = n_offset[None, :] < N
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
            block_nan = tl.min(
                tl.where(
                    inp_vals != inp_vals,
                    tl.arange(0, BLOCK_N)[None, :],
                    0x7FFFFFFF,
                ),
                1,
            )
            block_nan = tl.where(
                block_nan != 0x7FFFFFFF,
                full_tiles * BLOCK_N + block_nan.to(tl.int64),
                0x7FFFFFFFFFFFFFFF,
            )
            nan_values = tl.minimum(nan_values, block_nan)

        fixed = tl.where(
            nan_values != 0x7FFFFFFFFFFFFFFF,
            nan_values,
            encoded & 0x7FFFFFFFFFFFFFFF,
        )
        tl.store(out_index + rows, fixed, mask=row_mask & (encoded < 0))


@libentry()
@triton.jit
def _argmax_all_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    max_value = tl.full((), get_dtype_max(inp.type.element_ty), inp.type.element_ty)
    inp_val = tl.load(inp + offset, mask=mask, other=min_value)
    inp_val = tl.minimum(inp_val, max_value)
    max_val, local_index = tl.max(
        inp_val,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    max_index = pid.to(tl.int64) * BLOCK + local_index.to(tl.int64)
    if max_val == float("inf"):
        block_nan = tl.min(
            tl.where(inp_val != inp_val, tl.arange(0, BLOCK), 0x7FFFFFFF),
            axis=0,
        )
        has_nan = block_nan != 0x7FFFFFFF
        max_index = tl.where(has_nan, pid.to(tl.int64) * BLOCK + block_nan, max_index)
        max_val = tl.where(has_nan, float("nan"), max_val)
    tl.store(mid_value + pid, max_val)
    tl.store(mid_index + pid, max_index)


@libentry()
@triton.jit
def _argmax_all_kernel_2(
    mid_value,
    mid_index,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID).to(tl.int64)
    mask = offset < mid_size
    min_value = get_dtype_min(mid_value.type.element_ty)
    values = tl.load(mid_value + offset, mask=mask, other=min_value)
    if values.dtype is not tl.float32:
        values = values.to(tl.float32)
    bits = values.to(tl.int32, bitcast=True)
    keys = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    keys = tl.where(values != values, 0x7FFFFFFF, keys)
    keys = tl.where(values == 0.0, 0, keys)
    _, max_pos = tl.max(
        keys,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    max_index = tl.load(mid_index + max_pos)
    tl.store(out, max_index)


@libentry()
@triton.jit
def _argmax_all_key_chunk_kernel(
    mid_value,
    mid_index,
    chunk_key,
    chunk_index,
    mid_size,
    CHUNK: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * CHUNK + tl.arange(0, CHUNK)
    mask = offset < mid_size
    min_value = get_dtype_min(mid_value.type.element_ty)
    values = tl.load(mid_value + offset, mask=mask, other=min_value)
    values = values.to(tl.float32)
    bits = values.to(tl.int32, bitcast=True)
    keys = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    keys = tl.where(values != values, 0x7FFFFFFF, keys)
    keys = tl.where(values == 0.0, 0, keys)
    max_key, max_pos = tl.max(
        keys,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    max_index = tl.load(mid_index + pid * CHUNK + max_pos)
    tl.store(chunk_key + pid, max_key)
    tl.store(chunk_index + pid, max_index)


@libentry()
@triton.jit
def _argmax_all_key_final_kernel(
    chunk_key,
    chunk_index,
    out,
    num_chunks,
    BLOCK: tl.constexpr,
):
    offset = tl.arange(0, BLOCK)
    mask = offset < num_chunks
    keys = tl.load(chunk_key + offset, mask=mask, other=-0x80000000)
    _, max_pos = tl.max(
        keys,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    tl.store(out, tl.load(chunk_index + max_pos))


def _argmax_all(inp, keepdim=False, dtype=None):
    m = inp.numel()
    if dtype is None:
        dtype = inp.dtype

    sqrt_block = triton.next_power_of_2(math.ceil(math.sqrt(m)))
    block_size = min(sqrt_block * 2, 8192)
    mid_size = triton.cdiv(m, block_size)
    block_mid = triton.next_power_of_2(mid_size)
    use_tree_reduce = mid_size > 8192
    chunk_size = 1024
    num_chunks = triton.cdiv(mid_size, chunk_size)

    mid_value = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
    if use_tree_reduce:
        chunk_key = torch.empty((num_chunks,), dtype=torch.int32, device=inp.device)
        chunk_index = torch.empty((num_chunks,), dtype=torch.int64, device=inp.device)
    if keepdim:
        out_shape = [1] * inp.dim()
        out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
    else:
        out = torch.empty([], dtype=torch.int64, device=inp.device)

    with torch_device_fn.device(inp.device):
        _argmax_all_kernel_1[(mid_size, 1, 1)](
            inp,
            mid_value,
            mid_index,
            m,
            block_size,
        )
        if use_tree_reduce:
            _argmax_all_key_chunk_kernel[(num_chunks,)](
                mid_value,
                mid_index,
                chunk_key,
                chunk_index,
                mid_size,
                CHUNK=chunk_size,
                num_warps=4,
            )
            _argmax_all_key_final_kernel[(1, 1, 1)](
                chunk_key,
                chunk_index,
                out,
                num_chunks,
                BLOCK=triton.next_power_of_2(num_chunks),
                num_warps=4,
            )
        else:
            _argmax_all_kernel_2[(1, 1, 1)](
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
    if dtype == torch.float16 and inner >= 512 and n >= 512:
        return 64, 128, 4, 1
    if dtype in (torch.float16, torch.bfloat16):
        return 32, 256, 8, 2
    return 64, 64, 4, 1


def _argmax_blocks(inp, out_index, m, n, inner, stride_outer, stride_reduce):
    max_values = torch.empty((m,), dtype=torch.float32, device=inp.device)
    if inp.dtype == torch.bfloat16 and m >= 65536:
        with torch_device_fn.device(inp.device):
            _argmax_block_native_kernel[(triton.cdiv(m, 32),)](
                inp,
                out_index,
                max_values,
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
            _argmax_block_nan_fix_kernel[(triton.cdiv(m, 32),)](
                inp,
                out_index,
                max_values,
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
            max_values,
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
        _argmax_block_nan_fix_kernel[(triton.cdiv(m, block_m),)](
            inp,
            out_index,
            max_values,
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
        fix_in_kernel = True
        encode_inf = False
        block_n = triton.next_power_of_2(n)
        if block_n == 4096 and (m > 1024 or inp.dtype != torch.float32):
            block_n = 2048
        num_warps = 2 if block_n <= 256 else 4
        num_stages = 2 if block_n <= 64 or block_n == 4096 else 1
    else:
        fix_in_kernel = False
        encode_inf = True
        block_n = 4096
        num_warps = 4 if inp.dtype == torch.float32 else 8
        num_stages = 2
    with torch_device_fn.device(inp.device):
        _argmax_contiguous_kernel[(m,)](
            inp,
            out_index,
            out_index,
            m,
            n,
            stride_outer,
            stride_reduce,
            BLOCK_N=block_n,
            CACHE_CG=n >= 4096,
            FIX_IN_KERNEL=fix_in_kernel,
            ENCODE_INF=encode_inf,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        if encode_inf:
            _argmax_contiguous_nan_fix_kernel[(m,)](
                inp,
                out_index,
                m,
                n,
                stride_outer,
                stride_reduce,
                BLOCK_N=block_n,
                CACHE_CG=n >= 4096,
                num_warps=4,
                num_stages=2,
            )


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_MTHREADS ARGMAX")
    if dim is not None:
        rank = inp.ndim
        if rank == 0:
            if dim not in (0, -1):
                raise IndexError(
                    "Dimension out of range (expected to be in range of "
                    f"[-1, 0], but got {dim})"
                )
            return torch.zeros([], dtype=torch.int64, device=inp.device)
        if dim < -rank or dim >= rank:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                f"[{-rank}, {rank - 1}], but got {dim})"
            )
        dim %= rank

    if dim is None:
        if inp.numel() == 0:
            raise IndexError(
                "argmax(): Expected reduction dim to be specified for "
                "input.numel() == 0."
            )
        if not _supports_mthreads_argmax(inp):
            if not inp.is_contiguous():
                inp = inp.contiguous()
            return base_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)
        if not inp.is_contiguous():
            inp = inp.contiguous()
        return _argmax_all(inp, keepdim=keepdim, dtype=dtype)

    shape = list(inp.shape)
    dim, n, inner, outer = _flatten_dim(shape, dim)
    m = outer * inner

    out_shape = shape.copy()
    out_shape[dim] = 1
    if n == 0:
        raise IndexError(
            f"argmax(): Expected reduction dim {dim} to have non-zero size."
        )
    if inp.numel() == 0:
        if not keepdim:
            del out_shape[dim]
        return torch.empty(out_shape, dtype=torch.int64, device=inp.device)
    if not _supports_mthreads_argmax(inp):
        if not inp.is_contiguous():
            inp = inp.contiguous()
        return base_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)

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
    # large rows with explicit tiles.
    if inner == 1:
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
