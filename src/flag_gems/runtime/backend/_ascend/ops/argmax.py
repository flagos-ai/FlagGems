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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max, get_dtype_min

logger = logging.getLogger(__name__)


# Partial/final reduction follows Hopper argmax in FlagGems PR #8084.
@triton.jit
def argmax_maximum_nan(a, b):
    return tl.maximum(a, b, tl.PropagateNan.ALL)


@triton.jit
def argmax_reduce(values, indices, AXIS: tl.constexpr):
    if values.dtype.is_floating():
        maximum = tl.reduce(values, AXIS, argmax_maximum_nan)
        expanded = tl.expand_dims(maximum, AXIS)
        equal = (values == expanded) | ((values != values) & (expanded != expanded))
    else:
        maximum = tl.max(values, AXIS)
        equal = values == tl.expand_dims(maximum, AXIS)
    index = tl.min(tl.where(equal, indices, get_dtype_max(indices.dtype)), AXIS)
    return maximum, index


@libentry()
@triton.jit
def argmax_zero_kernel(out, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    for start in range(ext.program_id(0) * BLOCK, SIZE, ext.num_programs(0) * BLOCK):
        idx = start + tl.arange(0, BLOCK)
        tl.store(out + idx, tl.full((BLOCK,), 0, tl.int64), idx < SIZE)


@libentry()
@triton.jit
def argmax_chunk_kernel(
    inp,
    values_ptr,
    indices_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    SM: tl.constexpr,
    SN: tl.constexpr,
    SK: tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    lane = tl.arange(0, B)
    offset = (row // K).to(tl.int64) * SM + (row % K).to(tl.int64) * SK
    dtype: tl.constexpr = inp.dtype.element_ty
    if dtype == tl.float16 or dtype == tl.bfloat16:
        dtype = tl.float32
    minimum = get_dtype_min(dtype)
    if N > 2147483647:
        chunk = chunk.to(tl.int64)
    col = chunk * CHUNK + lane
    index_max = get_dtype_max(col.dtype)
    values = tl.load(inp + offset + col.to(tl.int64) * SN, col < N, other=minimum).to(
        dtype
    )
    indices = tl.where(col < N, col, index_max)
    for start in range(1, tl.cdiv(CHUNK, B)):
        col = chunk * CHUNK + start * B + lane
        x = tl.load(inp + offset + col.to(tl.int64) * SN, col < N, other=minimum).to(
            dtype
        )
        current = tl.where(col < N, col, index_max)
        # Each lane visits increasing indices, so an equal value keeps its index.
        replace = x > values
        if dtype.is_floating():
            replace = replace | ((x != x) & (values == values))
        values = tl.where(replace, x, values)
        indices = tl.where(replace, current, indices)
    value, index = argmax_reduce(values, indices, 0)
    if S > 1:
        tl.store(values_ptr + row.to(tl.int64) * S + chunk, value)
        tl.store(indices_ptr + row.to(tl.int64) * S + chunk, index)
    else:
        tl.store(indices_ptr + row, index)


@libentry()
@triton.jit
def argmax_row_finalize_kernel(
    values_ptr,
    indices_ptr,
    out,
    M: tl.constexpr,
    S: tl.constexpr,
    B: tl.constexpr,
    R: tl.constexpr,
):
    pid = tl.program_id(0)
    if M > 2147483647:
        pid = pid.to(tl.int64)
    rows = pid * R + tl.arange(0, R)
    col = tl.arange(0, B)
    mask = (rows[:, None] < M) & (col[None, :] < S)
    values = tl.load(
        values_ptr + rows[:, None].to(tl.int64) * S + col[None, :],
        mask,
        other=get_dtype_min(values_ptr.dtype.element_ty),
    )
    indices = tl.load(
        indices_ptr + rows[:, None].to(tl.int64) * S + col[None, :],
        mask,
        other=get_dtype_max(indices_ptr.dtype.element_ty),
    )
    _, index = argmax_reduce(values, indices, 1)
    tl.store(out + rows, index, rows < M)


@libentry()
@triton.jit
def argmax_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    for pid in range(ext.program_id(0), tl.cdiv(M, BLOCK_SIZE), ext.num_programs(0)):
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        inp_ptrs = inp + offset
        mask = offset < M
        min_value = get_dtype_min(inp.type.element_ty)
        inp_val = tl.load(inp_ptrs, mask=mask, other=min_value)
        max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
        max_index = max_index + pid * BLOCK_SIZE
        mid_value_ptr = mid_value + pid
        max_index_ptr = mid_index + pid
        tl.store(mid_value_ptr, max_val)
        tl.store(max_index_ptr, max_index)


@libentry()
@triton.jit
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr):
    min_value = get_dtype_min(mid_value.type.element_ty)
    acc_type = (
        tl.float32
        if mid_value.dtype.element_ty == tl.bfloat16
        else mid_value.dtype.element_ty
    )
    best_value = tl.full((), min_value, acc_type)
    best_index = tl.full((), 0, tl.int64)
    for start in range(0, mid_size, BLOCK_MID):
        offset = start + tl.arange(0, BLOCK_MID)
        values = tl.load(mid_value + offset, offset < mid_size, min_value).to(acc_type)
        value, index = tl.max(
            values, axis=0, return_indices=True, return_indices_tie_break_left=True
        )
        update = (value > best_value) | ((value != value) & (best_value == best_value))
        update = update | (start == 0)
        original_index = tl.load(mid_index + start + index)
        best_value = tl.where(update, value, best_value)
        best_index = tl.where(update, original_index, best_index)
    tl.store(out, best_index)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("argmax"))
@triton.jit
def argmax_kernel(
    inp,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    for task in range(ext.program_id(0), tl.cdiv(M, BLOCK_M) * K, ext.num_programs(0)):
        pid_m = task // K
        pid_k = task % K

        m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

        dtype = inp.type.element_ty
        acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
        min_value = get_dtype_min(dtype)
        max_values = tl.full([BLOCK_M], dtype=acc_type, value=min_value)
        argmax_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
        for start_n in range(0, N, BLOCK_N):
            n_offset = start_n + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
            mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
            inp_ptrs = inp + offset
            inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
            local_max, local_argmax = tl.max(
                inp_vals, 1, return_indices=True, return_indices_tie_break_left=True
            )
            update = (local_max > max_values) | (
                (local_max != local_max) & (max_values == max_values)
            )
            max_values = tl.where(update, local_max, max_values)
            argmax_values = tl.where(update, start_n + local_argmax, argmax_values)

        offset_index = m_offset * K + pid_k
        out_index_ptrs = out_index + offset_index
        mask1 = m_offset < M
        tl.store(out_index_ptrs, argmax_values, mask=mask1)


@libentry()
@triton.jit
def argmax_row_kernel(
    inp,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    for task in range(ext.program_id(0), tl.cdiv(M, BLOCK_M), ext.num_programs(0)):
        rows = task * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        dtype = inp.type.element_ty
        values = tl.load(
            inp + rows[:, None] * N + cols[None, :],
            (rows[:, None] < M) & (cols[None, :] < N),
            get_dtype_min(dtype),
        )
        idx = tl.argmax(values, 1, tie_break_left=True)
        tl.store(out + rows, idx.to(tl.int64), rows < M)


@libentry()
@triton.jit
def argmax_strided_kernel(
    inp,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    k_tiles = tl.cdiv(K, BLOCK_K)
    for task in range(ext.program_id(0), M * k_tiles, ext.num_programs(0)):
        m = task // k_tiles
        ks = (task % k_tiles) * BLOCK_K + tl.arange(0, BLOCK_K)
        ns = tl.arange(0, BLOCK_N)
        dtype = inp.type.element_ty
        acc_type = tl.float32 if dtype == tl.bfloat16 else dtype
        if N <= BLOCK_N:
            values = tl.load(
                inp + m * N * K + ns[:, None] * K + ks[None, :],
                (ns[:, None] < N) & (ks[None, :] < K),
                get_dtype_min(dtype),
            )
            idx = tl.argmax(values, 0, tie_break_left=True)
            tl.store(out + m * K + ks, idx.to(tl.int64), ks < K)
        else:
            best = tl.full((BLOCK_K,), get_dtype_min(dtype), acc_type)
            best_idx = tl.full((BLOCK_K,), 0, tl.int32)
            for start in range(tl.cdiv(N, BLOCK_N)):
                nr = start * BLOCK_N + ns
                values = tl.load(
                    inp + m * N * K + nr[:, None] * K + ks[None, :],
                    (nr[:, None] < N) & (ks[None, :] < K),
                    get_dtype_min(dtype),
                )
                val, idx = tl.max(
                    values, 0, return_indices=True, return_indices_tie_break_left=True
                )
                update = (val > best) | ((val != val) & (best == best))
                best = tl.where(update, val, best)
                best_idx = tl.where(update, start * BLOCK_N + idx, best_idx)
            tl.store(out + m * K + ks, best_idx.to(tl.int64), ks < K)


@libentry()
@triton.jit
def argmax_stream_kernel(
    inp, out, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr, R: tl.constexpr
):
    for task in range(ext.program_id(0), tl.cdiv(M, R), ext.num_programs(0)):
        row = task * R + tl.arange(0, R)
        lane = tl.arange(0, B)
        dtype = inp.dtype.element_ty
        vals = tl.load(
            inp + row[:, None].to(tl.int64) * N + lane[None, :],
            (row[:, None] < M) & (lane[None, :] < N),
            get_dtype_min(dtype),
        )
        if dtype == tl.bfloat16:
            vals = vals.to(tl.float32)
        idx = tl.broadcast_to(tl.where(lane < N, lane, 2147483647)[None, :], (R, B))
        for start in range(1, tl.cdiv(N, B)):
            col = start * B + lane
            x = tl.load(
                inp + row[:, None].to(tl.int64) * N + col[None, :],
                (row[:, None] < M) & (col[None, :] < N),
                get_dtype_min(dtype),
            )
            if dtype == tl.bfloat16:
                x = x.to(tl.float32)
            update = (x > vals) | ((x != x) & (vals == vals))
            vals = tl.where(update, x, vals)
            idx = tl.where(update, col[None, :], idx)
        mx = tl.max(vals, 1)
        match = (vals == mx[:, None]) | ((vals != vals) & (mx[:, None] != mx[:, None]))
        result = tl.min(tl.where(match, idx, 2147483647), 1)
        tl.store(out + row, result.to(tl.int64), row < M)


def argmax_split_rows(inp, out, rows, n):
    if rows >= 64 and n >= 16384 and inp.dtype == torch.float16:
        block, block_rows = (8192, 1) if n >= 262144 else (2048, 4)
        argmax_stream_kernel[(min(65535, triton.cdiv(rows, block_rows)),)](
            inp, out, rows, n, block, block_rows
        )
        return
    # Bound both per-program work and the partial-result workspace.
    block = 4096 if rows == 1 or (rows >= 64 and n >= 16384) else 1024
    if rows == 1:
        target = min(1024, max(64, triton.cdiv(n, 16384)))
    else:
        target = min(16, triton.cdiv(n, 4096), max(1, triton.cdiv(64, rows)))
    splits = min(triton.cdiv(n, block), target)
    chunk = triton.cdiv(n, block * splits) * block
    splits = triton.cdiv(n, chunk)
    if splits > 1:
        dtype = inp.dtype
        values = torch.empty(
            (rows, splits),
            dtype=torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype,
            device=inp.device,
        )
        indices = torch.empty(
            (rows, splits),
            dtype=torch.int64 if n > 2147483647 else torch.int32,
            device=inp.device,
        )
    else:
        values = indices = out
    argmax_chunk_kernel[(rows, splits)](
        inp, values, indices, n, 1, n, 1, 1, block, splits, chunk
    )
    if splits > 1:
        argmax_row_finalize_kernel[(rows,)](
            values, indices, out, rows, splits, triton.next_power_of_2(splits), 1
        )


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ASCEND ARGMAX")
    if inp.ndim == 0 and dim in (0, -1):
        dim = None
    if dim is None:
        M = inp.numel()
        if M == 0:
            raise IndexError(
                "argmax(): Expected reduction dim to be specified for input.numel() == 0."
            )
        inp = inp.contiguous()
        if dtype is None:
            dtype = inp.dtype
        if M >= 4096 and dtype == inp.dtype:
            out_shape = [1] * inp.ndim if keepdim else []
            out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
            with torch_device_fn.device(inp.device):
                argmax_split_rows(inp, out, 1, M)
            return out
        block_size = min(1024, triton.next_power_of_2(math.ceil(math.sqrt(M))))
        mid_size = triton.cdiv(M, block_size)
        block_mid = min(1024, triton.next_power_of_2(mid_size))

        mid_value = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
        out_shape = [1] * inp.ndim if keepdim else []
        out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)

        with torch_device_fn.device(inp.device):
            argmax_kernel_1[(min(mid_size, 65535),)](
                inp,
                mid_value,
                mid_index,
                M,
                block_size,
            )
            argmax_kernel_2[(1, 1, 1)](mid_value, mid_index, out, mid_size, block_mid)
        return out
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        shape = inp.shape
        dim = dim % inp.ndim
        if shape[dim] == 0:
            raise IndexError("argmax(): Expected reduction dim to have non-zero size.")
        if inp.numel() == 0:
            out_shape = list(shape)
            if keepdim:
                out_shape[dim] = 1
            else:
                del out_shape[dim]
            return torch.zeros(out_shape, dtype=torch.int64, device=inp.device)
        N = shape[dim]
        M = math.prod(shape[:dim])
        K = inp.numel() // M // N

        inp = inp.contiguous()

        shape_list = list(shape)
        shape_list[dim] = 1
        out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
        if not keepdim:
            out_index = torch.squeeze(out_index, dim)
        with torch_device_fn.device(inp.device):
            if N == 1:
                argmax_zero_kernel[(min(65535, triton.cdiv(out_index.numel(), 1024)),)](
                    out_index, out_index.numel(), 1024
                )
            elif K == 1:
                if N <= 4096:
                    block_n = triton.next_power_of_2(N)
                    budget = 16384 if inp.dtype == torch.float16 else 8192
                    block_m = min(16, max(1, budget // block_n))
                    argmax_row_kernel[(min(65535, triton.cdiv(M, block_m)),)](
                        inp, out_index, M, N, block_m, block_n
                    )
                elif M <= 65535:
                    argmax_split_rows(inp, out_index, M, N)
                else:
                    grid = lambda meta: (min(65535, triton.cdiv(M, meta["BLOCK_M"])),)
                    argmax_kernel[grid](inp, out_index, M, N, K)
            else:
                block_n = min(
                    128 if inp.dtype == torch.float16 else 512,
                    triton.next_power_of_2(N),
                )
                block_k = min(64, 8192 // block_n, triton.next_power_of_2(K))
                argmax_strided_kernel[(min(65535, M * triton.cdiv(K, block_k)),)](
                    inp, out_index, M, N, K, block_n, block_k
                )

        return out_index
