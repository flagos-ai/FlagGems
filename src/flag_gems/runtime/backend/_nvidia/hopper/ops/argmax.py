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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_max, get_dtype_min

logger = logging.getLogger(__name__)


@triton.jit
def argmax_maximum_nan(a, b):
    return tl.maximum(a, b, tl.PropagateNan.ALL)


@triton.jit
def argmax_pair(a, ai, b, bi):
    replace = b > a
    tied = b == a
    if a.dtype.is_floating():
        replace = replace | ((b != b) & (a == a))
        tied = tied | ((b != b) & (a != a))
    replace = replace | (tied & (bi < ai))
    return tl.where(replace, b, a), tl.where(replace, bi, ai)


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
def argmax_row_kernel(
    inp,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SM: tl.constexpr,
    SN: tl.constexpr,
    SK: tl.constexpr,
    B: tl.constexpr,
    R: tl.constexpr,
):
    pid = tl.program_id(0)
    if M > 2147483647:
        pid = pid.to(tl.int64)
    row = pid * R + tl.arange(0, R)
    col = tl.arange(0, B)
    offset = (row // K).to(tl.int64) * SM + (row % K).to(tl.int64) * SK
    minimum = get_dtype_min(inp.dtype.element_ty)
    values = tl.load(
        inp + offset[:, None] + col[None, :].to(tl.int64) * SN,
        (row[:, None] < M) & (col[None, :] < N),
        other=minimum,
    )
    if values.dtype == tl.float16 or values.dtype == tl.bfloat16:
        values = values.to(tl.float32)
    indices = tl.broadcast_to(tl.where(col < N, col, 2147483647)[None, :], (R, B))
    # A joint reduction keeps large row tiles within the register budget.
    if B >= 32768:
        _, index = tl.reduce((values, indices), 1, argmax_pair)
    else:
        _, index = argmax_reduce(values, indices, 1)
    tl.store(out + row, index, row < M)


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


def argmax_rows(inp, out, rows, n, k, stride_m, stride_n, stride_k):
    """rows = M*K; offset(row) = (row//K)*stride_m+(row%K)*stride_k."""
    if n <= 65536 or (n <= 131072 and rows >= 16):
        block = triton.next_power_of_2(n)
        block_rows = 16 if n <= 32 else (4 if n <= 1024 else 1)
        warps = 4 if n <= 8192 else (8 if n <= 16384 else (16 if n <= 65536 else 32))
        argmax_row_kernel[(triton.cdiv(rows, block_rows),)](
            inp,
            out,
            rows,
            n,
            k,
            stride_m,
            stride_n,
            stride_k,
            block,
            block_rows,
            num_warps=warps,
        )
        return
    num_sms = torch_device_fn.get_device_properties(inp.device).multi_processor_count
    chunk_warps = 4
    if rows == 1 and n > 2097152:
        half = inp.dtype in (torch.float16, torch.bfloat16)
        block = 4096 if half else 8192
        chunk_warps = 4 if half else 8
        splits = min(triton.cdiv(n, block), 16384, 128 * num_sms)
    # Treat a nearly complete CTA wave as full across Hopper SM-count variants.
    elif 32 * rows < 31 * num_sms:
        block = 1024 if n <= 131072 and rows <= 4 else (4096 if n <= 2097152 else 2048)
        splits = min(
            triton.cdiv(n, block),
            max(1, triton.cdiv(512 if rows <= 32 else 8 * num_sms, rows)),
        )
        if n >= 67108864:
            splits = min(2048, 16 * num_sms)
    else:
        block = 2048
        waves = 8 if 8 * rows < 9 * num_sms else (24 if rows >= 3 * num_sms else 16)
        splits = min(
            16,
            triton.cdiv(n, 8192),
            max(triton.cdiv(n, 65536), triton.cdiv(waves * num_sms, rows)),
        )
    chunk = triton.cdiv(n, block * splits) * block
    splits = triton.cdiv(n, chunk)
    if splits == 1:
        argmax_chunk_kernel[(rows, 1)](
            inp,
            out,
            out,
            n,
            k,
            stride_m,
            stride_n,
            stride_k,
            block,
            splits,
            chunk,
            num_warps=chunk_warps,
        )
    else:
        dtype = (
            torch.float32 if inp.dtype in (torch.float16, torch.bfloat16) else inp.dtype
        )
        values = torch.empty((rows, splits), device=inp.device, dtype=dtype)
        index_dtype = torch.int64 if n > 2147483647 else torch.int32
        indices = torch.empty((rows, splits), device=inp.device, dtype=index_dtype)
        argmax_chunk_kernel[(rows, splits)](
            inp,
            values,
            indices,
            n,
            k,
            stride_m,
            stride_n,
            stride_k,
            block,
            splits,
            chunk,
            num_warps=chunk_warps,
        )
        finish_rows = 1 if rows == 1 else 4
        argmax_row_finalize_kernel[(triton.cdiv(rows, finish_rows),)](
            values,
            indices,
            out,
            rows,
            splits,
            triton.next_power_of_2(splits),
            finish_rows,
            num_warps=8 if rows == 1 and splits > 4096 else 4,
        )


@libentry()
@triton.jit
def argmax_column_kernel(
    inp,
    values_ptr,
    indices_ptr,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    SM: tl.constexpr,
    SN: tl.constexpr,
    SK: tl.constexpr,
    CHUNK: tl.constexpr,
    INDEX64: tl.constexpr,
    OFFSET64: tl.constexpr,
):
    # A flat grid avoids CUDA's 65535 limit on grid Y and Z.
    pid = tl.program_id(0)
    m = pid % M
    tile = pid // M
    block_k = tile % triton.cdiv(K, BK)
    if K > 2147483647:
        block_k = block_k.to(tl.int64)
    kk = block_k * BK + tl.arange(0, BK)
    s = tile // triton.cdiv(K, BK)
    start = s.to(tl.int64) * CHUNK if INDEX64 else s * CHUNK
    index_dtype: tl.constexpr = tl.int64 if INDEX64 else tl.int32
    sentinel: tl.constexpr = 0x7FFFFFFFFFFFFFFF if INDEX64 else 0x7FFFFFFF
    value_dtype: tl.constexpr = inp.dtype.element_ty
    if value_dtype == tl.float16 or value_dtype == tl.bfloat16:
        value_dtype = tl.float32
    minimum = get_dtype_min(value_dtype)
    nn = start + tl.arange(0, BN)
    if OFFSET64:
        base_offset = m.to(tl.int64) * SM + kk[None, :].to(tl.int64) * SK
        offset = base_offset + nn[:, None].to(tl.int64) * SN
    else:
        base_offset = m * SM + kk[None, :] * SK
        offset = base_offset + nn[:, None] * SN
    values = tl.load(
        inp + offset, (nn[:, None] < N) & (kk[None, :] < K), other=minimum
    ).to(value_dtype)
    indices = tl.broadcast_to(
        tl.where(nn[:, None] < N, nn[:, None], sentinel), (BN, BK)
    ).to(index_dtype)
    for tile in range(1, triton.cdiv(CHUNK, BN)):
        nn = start + tile * BN + tl.arange(0, BN)
        if OFFSET64:
            offset = base_offset + nn[:, None].to(tl.int64) * SN
        else:
            offset = base_offset + nn[:, None] * SN
        val = tl.load(
            inp + offset, (nn[:, None] < N) & (kk[None, :] < K), other=minimum
        ).to(value_dtype)
        # Later tiles have larger indices, so equal values retain the prior winner.
        replace = val > values
        if value_dtype.is_floating():
            replace = replace | ((val != val) & (values == values))
        values = tl.where(replace, val, values)
        indices = tl.where(replace, nn[:, None], indices)
    value, index = tl.reduce((values, indices), 0, argmax_pair)
    if SPLIT == 1:
        tl.store(out + m.to(tl.int64) * K + kk, index, kk < K)
    else:
        offset = (m.to(tl.int64) * SPLIT + s) * K + kk
        tl.store(values_ptr + offset, value, kk < K)
        tl.store(indices_ptr + offset, index, kk < K)


@libentry()
@triton.jit
def argmax_column_finalize_kernel(
    values_ptr,
    indices_ptr,
    out,
    M: tl.constexpr,
    K: tl.constexpr,
    SPLIT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    INDEX64: tl.constexpr,
):
    # A flat grid avoids CUDA's 65535 limit on grid Y and Z.
    pid = tl.program_id(0)
    m = pid % M
    block_k = pid // M
    if K > 2147483647:
        block_k = block_k.to(tl.int64)
    k = block_k * BK + tl.arange(0, BK)
    s = tl.arange(0, BS)
    mask = (s[:, None] < SPLIT) & (k[None, :] < K)
    offset = (m.to(tl.int64) * SPLIT + s[:, None]) * K + k[None, :]
    value = tl.load(
        values_ptr + offset, mask, other=get_dtype_min(values_ptr.dtype.element_ty)
    )
    sentinel: tl.constexpr = 0x7FFFFFFFFFFFFFFF if INDEX64 else 0x7FFFFFFF
    index = tl.load(indices_ptr + offset, mask, other=sentinel)
    _, ids = tl.reduce((value, index), 0, argmax_pair)
    tl.store(out + m.to(tl.int64) * K + k, ids, k < K)


@libentry()
@triton.jit
def argmax_zero_kernel(out, size, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + offsets, 0, offsets < size)


@libentry()
@triton.jit
def argmax_strided_partial_kernel(
    inp,
    values,
    indices,
    size: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    LOGICAL_STRIDES: tl.constexpr,
    DENSE: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    SPLIT: tl.constexpr,
    INT64: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64:
        pid = pid.to(tl.int64)
    index_dtype: tl.constexpr = tl.int64 if INT64 else tl.int32
    sentinel: tl.constexpr = 0x7FFFFFFFFFFFFFFF if INT64 else 0x7FFFFFFF
    value_dtype: tl.constexpr = inp.dtype.element_ty
    if value_dtype == tl.float16 or value_dtype == tl.bfloat16:
        value_dtype = tl.float32
    minimum = get_dtype_min(value_dtype)
    best_values = tl.full((BLOCK,), minimum, value_dtype)
    best_indices = tl.full((BLOCK,), sentinel, index_dtype)
    lane = tl.arange(0, BLOCK)
    for tile in range(triton.cdiv(CHUNK, BLOCK)):
        scan = pid * CHUNK + tile * BLOCK + lane
        remaining = scan
        logical = tl.full((BLOCK,), 0, index_dtype)
        physical = tl.full((BLOCK,), 0, index_dtype)
        for d in tl.static_range(len(SHAPE) - 1, -1, -1):
            coord = remaining % SHAPE[d]
            remaining = remaining // SHAPE[d]
            logical += coord * LOGICAL_STRIDES[d]
            if not DENSE:
                physical += coord * STRIDES[d]
        if DENSE:
            physical = scan
        value = tl.load(inp + physical, scan < size, other=minimum).to(value_dtype)
        logical = tl.where(scan < size, logical, sentinel)
        if CHUNK == BLOCK:
            best_values, best_indices = value, logical
        else:
            best_values, best_indices = argmax_pair(
                best_values, best_indices, value, logical
            )
    value, index = tl.reduce((best_values, best_indices), 0, argmax_pair)
    tl.store(indices + pid, index)
    if SPLIT > 1:
        tl.store(values + pid, value)


@libentry()
@triton.jit
def argmax_strided_finalize_kernel(
    values, indices, out, size: tl.constexpr, BLOCK: tl.constexpr, INT64: tl.constexpr
):
    offsets = tl.arange(0, BLOCK)
    sentinel: tl.constexpr = 0x7FFFFFFFFFFFFFFF if INT64 else 0x7FFFFFFF
    value = tl.load(
        values + offsets, offsets < size, other=get_dtype_min(values.dtype.element_ty)
    )
    index = tl.load(indices + offsets, offsets < size, other=sentinel)
    _, best_index = tl.reduce((value, index), 0, argmax_pair)
    tl.store(out, best_index)


def argmax_global_strided(inp, out):
    # Scan the fastest physical axis first while retaining logical flatten indices.
    # Broadcast axes repeat each value; coordinate zero wins every index tie.
    dims = [d for d in range(inp.ndim) if inp.shape[d] > 1 and inp.stride(d) != 0]
    dims.sort(key=lambda d: inp.stride(d), reverse=True)
    shape = tuple(inp.shape[d] for d in dims)
    strides = tuple(inp.stride(d) for d in dims)
    logical_strides = tuple(math.prod(inp.shape[d + 1 :]) for d in dims)
    dense = all(strides[d] == math.prod(shape[d + 1 :]) for d in range(len(shape)))
    size = math.prod(shape)
    block = min(4096, triton.next_power_of_2(size))
    # Bound the final reduction, including views with billions of input elements.
    chunk = triton.cdiv(size, block * 4096) * block
    split = triton.cdiv(size, chunk)
    max_offset = sum((s - 1) * st for s, st in zip(inp.shape, inp.stride()))
    use_int64 = max(inp.numel(), max_offset) > 0x7FFFFFFF
    if split > 1:
        value_dtype = (
            torch.float32 if inp.dtype in (torch.float16, torch.bfloat16) else inp.dtype
        )
        index_dtype = torch.int64 if use_int64 else torch.int32
        values = torch.empty((split,), dtype=value_dtype, device=inp.device)
        indices = torch.empty((split,), dtype=index_dtype, device=inp.device)
    else:
        values = indices = out
    argmax_strided_partial_kernel[(split,)](
        inp,
        values,
        indices,
        size,
        shape,
        strides,
        logical_strides,
        dense,
        block,
        chunk,
        split,
        use_int64,
        num_warps=4,
    )
    if split > 1:
        argmax_strided_finalize_kernel[(1,)](
            values,
            indices,
            out,
            split,
            triton.next_power_of_2(split),
            use_int64,
            num_warps=4,
        )


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_NVIDIA ARGMAX")
    if dim is None:
        n = inp.numel()
        out_shape = [1] * inp.ndim if keepdim else []
        m, k, stride_m, stride_n, stride_k = 1, 1, n, 1, 1
        inner = True
    else:
        dim %= inp.ndim
        n = inp.shape[dim]
        out_shape = list(inp.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            del out_shape[dim]
        m = math.prod(inp.shape[:dim])
        k = math.prod(inp.shape[dim + 1 :])
        stride_m, stride_n, stride_k = n * k, k, 1
        inner = k == 1
        if not inp.is_contiguous():
            # Collapse only adjacent output axes that share one linear stride.
            # Their logical order is retained, including broadcast and slice views.
            groups = []
            for axis in reversed(range(inp.ndim)):
                if axis == dim or inp.shape[axis] == 1:
                    continue
                size, stride = inp.shape[axis], inp.stride(axis)
                if groups and stride == groups[-1][0] * groups[-1][1]:
                    groups[-1] = (size * groups[-1][0], groups[-1][1])
                else:
                    groups.append((size, stride))
            groups.reverse()
            stride_n = inp.stride(dim)
            if len(groups) <= 1:
                size, stride = groups[0] if groups else (1, 1)
                if stride_n <= stride or stride == 0:
                    m, k, stride_m, stride_k = size, 1, stride, 1
                    inner = True
                else:
                    m, k, stride_m, stride_k = 1, size, 0, stride
                    inner = False
            elif len(groups) == 2:
                (m, stride_m), (k, stride_k) = groups
                inner = stride_n <= stride_k
            else:
                inp = inp.contiguous()
                stride_m, stride_n, stride_k = n * k, k, 1

    out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
    if out.numel() == 0:
        return out
    with torch_device_fn.device(inp.device):
        if n == 1 or stride_n == 0:
            argmax_zero_kernel[(triton.cdiv(out.numel(), 512),)](
                out, out.numel(), 512, num_warps=4
            )
        elif dim is None and not inp.is_contiguous():
            # The strided global implementation preserves logical flatten indices.
            argmax_global_strided(inp, out)
        elif inner or (n > 65536 and k <= 4 and m * k < 16):
            # Few narrow columns share the stride-aware row path and its cheaper launches.
            argmax_rows(inp, out, m * k, n, k, stride_m, stride_n, stride_k)
        else:
            num_sms = torch_device_fn.get_device_properties(
                inp.device
            ).multi_processor_count
            split_cap = 512
            if k <= 4:
                tile_n = (
                    triton.next_power_of_2(n)
                    if m >= 128 and n <= 8192
                    else min(triton.next_power_of_2(n), 1024)
                )
                tile_k = triton.next_power_of_2(k)
            elif k >= 65536 and 32 < n <= 512:
                # Wide outputs already fill the GPU; stream N without partials.
                tile_n = 16
                tile_k = min(
                    256,
                    max(64, triton.next_power_of_2(triton.cdiv(m * k, 8 * num_sms))),
                )
                split_cap = 1
            elif n > 256 and k >= 256 and m * triton.cdiv(k, 128) >= 16 * num_sms:
                tile_n, tile_k, split_cap = 32, 128, 1
            elif n <= 256:
                tile_n = triton.next_power_of_2(n)
                tile_k = 16 if k < 1024 else 64
                if n <= 32 and k >= 65536:
                    tile_k = min(1024, max(64, 4096 // tile_n))
            elif m >= 64 and k <= 128:
                tile_n, tile_k = 512, min(triton.next_power_of_2(k), 64)
            else:
                tile_n = 256
                tile_k = 32 if m * triton.cdiv(k, 32) >= triton.cdiv(num_sms, 2) else 16
            # Looping large tiles increases live registers and reduces occupancy.
            column_ctas = m * triton.cdiv(k, tile_k)
            if tile_n * tile_k <= 8192:
                split_cap = min(split_cap, triton.cdiv(16 * num_sms, column_ctas))
            # Bound workspace and the final reduction for very long columns.
            chunk = triton.cdiv(n, tile_n * split_cap) * tile_n
            split = triton.cdiv(n, chunk)
            index64 = n > 0x7FFFFFFF
            max_offset = (m - 1) * stride_m + (n - 1) * stride_n + (k - 1) * stride_k
            offset64 = max_offset > 0x7FFFFFFF
            if split > 1:
                value_dtype = (
                    torch.float32
                    if inp.dtype in (torch.float16, torch.bfloat16)
                    else inp.dtype
                )
                values = torch.empty(
                    (m, split, k), dtype=value_dtype, device=inp.device
                )
                indices = torch.empty(
                    (m, split, k),
                    dtype=torch.int64 if index64 else torch.int32,
                    device=inp.device,
                )
            else:
                values = indices = out
            argmax_column_kernel[(m * triton.cdiv(k, tile_k) * split,)](
                inp,
                values,
                indices,
                out,
                m,
                n,
                k,
                tile_n,
                tile_k,
                split,
                stride_m,
                stride_n,
                stride_k,
                chunk,
                index64,
                offset64,
                num_warps=4,
            )
            if split > 1:
                final_tile_k = 32 if split <= 64 else min(16, triton.next_power_of_2(k))
                argmax_column_finalize_kernel[(m * triton.cdiv(k, final_tile_k),)](
                    values,
                    indices,
                    out,
                    m,
                    k,
                    split,
                    triton.next_power_of_2(split),
                    final_tile_k,
                    index64,
                    num_warps=4,
                )
    return out
