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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24
MAX_GRID = NUM_SIPS * 2
INT32_LIMIT = 2**31 - 1


# ========== Global any (flat grid-stride + bool->u8) ==========


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def any_global_kernel(
    inp_ptr,
    mid_ptr,
    N_total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.int8)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        val = tl.load(inp_ptr + off, mask=mask, other=0)
        acc = acc | (val != 0).to(tl.int8)
    result = tl.max(acc, axis=0) > 0
    tl.store(mid_ptr + pid, result)


@libentry()
@triton.jit
def any_reduce_kernel(mid_ptr, out_ptr, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptr + offset, mask=mask, other=0).to(tl.int8)
    result = tl.max(mid_val, axis=0) > 0
    tl.store(out_ptr, result)


# ========== Row any: reduce trailing dims of a contiguous [M, N] view ==========
# acc is int8 (not int1) and the final combine is tl.max: int1 tiles with a
# custom combine_fn reduce compile to far slower code on GCU400.


@libentry()
@triton.jit
def any_row_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_U8: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    if IS_U8:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.uint8)
    else:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int8)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        if IS_U8:
            a = tl.load(inp + cols, mask, other=0)
            acc = acc | a
        else:
            a = tl.load(inp + cols, mask, other=0.0)
            acc = acc | (a != 0).to(tl.int8)
    any_result = tl.max(acc, axis=1) > 0
    tl.store(out, any_result[:, None], row_mask)


# ========== Segmented any: reduce a middle dim range (outer, D, inner) ==========
# Works directly on the original contiguous layout, so no transpose copy is
# needed for mid-dim reductions. When the natural grid is too small the D axis
# is split across programs and partials are merged by any_seg_combine_kernel.


@libentry()
@triton.jit
def any_seg_kernel(
    inp,
    out_ptr,
    mid_ptr,
    OUTER,
    D,
    INNER,
    D_CHUNK,
    SPLIT_RT,
    WRITE_MID: tl.constexpr,
    TILE_D: tl.constexpr,
    TILE_I: tl.constexpr,
    EVEN_D: tl.constexpr,
    EVEN_I: tl.constexpr,
    IS_U8: tl.constexpr,
):
    pid = tl.program_id(0)
    n_i = tl.cdiv(INNER, TILE_I)
    s_idx = pid % SPLIT_RT
    rest = pid // SPLIT_RT
    i_blk = rest % n_i
    a_idx = rest // n_i

    offs_i = i_blk * TILE_I + tl.arange(0, TILE_I)
    mask_i = offs_i < INNER
    d_lo = s_idx * D_CHUNK
    d_hi = tl.minimum(d_lo + D_CHUNK, D)
    base = a_idx * D * INNER
    if IS_U8:
        acc = tl.zeros([TILE_D, TILE_I], dtype=tl.uint8)
    else:
        acc = tl.zeros([TILE_D, TILE_I], dtype=tl.int8)
    for d0 in tl.range(d_lo, d_hi, TILE_D):
        offs_d = d0 + tl.arange(0, TILE_D)[:, None]
        ptrs = inp + base + offs_d * INNER + offs_i[None, :]
        if EVEN_D and EVEN_I:
            a = tl.load(ptrs)
        else:
            m = (offs_d < d_hi) & mask_i[None, :]
            a = tl.load(ptrs, mask=m, other=0.0)
        if IS_U8:
            acc = acc | a
        else:
            acc = acc | (a != 0).to(tl.int8)
    res = tl.max(acc, axis=0) > 0
    if WRITE_MID:
        tl.store(mid_ptr + (s_idx * OUTER + a_idx) * INNER + offs_i, res, mask=mask_i)
    else:
        tl.store(out_ptr + a_idx * INNER + offs_i, res, mask=mask_i)


@libentry()
@triton.jit
def any_seg_combine_kernel(
    mid_ptr,
    out_ptr,
    TOTAL,
    SPLIT_RT,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < TOTAL
    s = tl.arange(0, SPLIT)[:, None]
    v = tl.load(
        mid_ptr + s * TOTAL + offs[None, :],
        mask=(s < SPLIT_RT) & m[None, :],
        other=0,
    ).to(tl.int8)
    res = tl.max(v, axis=0) > 0
    tl.store(out_ptr + offs, res, mask=m)


def _to_u8_if_bool(inp):
    if inp.dtype == torch.bool:
        return inp.view(torch.uint8)
    return inp


def _np2(x):
    return triton.next_power_of_2(max(int(x), 1))


def _launch_global(inp):
    n_total = inp.numel()
    if n_total == 0:
        return torch.tensor(False, dtype=torch.bool, device=inp.device)
    inp = _to_u8_if_bool(inp)
    if n_total <= 4096:
        block = max(_np2(n_total), 1024)
    else:
        block = 16384
    num_blocks = triton.cdiv(n_total, block)
    grid_size = min(num_blocks, MAX_GRID)

    mid = torch.empty((grid_size,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)

    with torch_device_fn.device(inp.device):
        any_global_kernel[(grid_size,)](inp, mid, n_total, block, num_warps=4)
        block_mid = _np2(grid_size)
        any_reduce_kernel[(1,)](mid, out, grid_size, block_mid, num_warps=1)
    return out


def _launch_row(inp, out, M, N):
    block_n = max(64, min(512, _np2(N)))
    block_m = min(128, max(8, 65536 // block_n))
    block_m = min(block_m, _np2(M))
    grid = (triton.cdiv(M, block_m),)
    is_u8 = inp.dtype == torch.uint8
    with torch_device_fn.device(inp.device):
        any_row_kernel[grid](inp, out, M, N, block_m, block_n, is_u8, num_warps=4)


def _launch_seg(inp, out, outer, D, inner):
    tile_i = min(512, _np2(inner))
    tile_d = min(128, _np2(D))
    n_i = triton.cdiv(inner, tile_i)
    grid0 = outer * n_i
    split = 1
    if grid0 < NUM_SIPS and D > tile_d:
        split = min(triton.cdiv(MAX_GRID, grid0), triton.cdiv(D, tile_d))
        split = max(1, min(split, MAX_GRID))
    d_chunk = triton.cdiv(D, split)
    even_d = (
        (D % tile_d == 0)
        if split == 1
        else (d_chunk % tile_d == 0 and D % d_chunk == 0)
    )
    even_i = inner % tile_i == 0
    grid = (grid0 * split,)
    mid = None
    if split > 1:
        mid = torch.empty((split, outer * inner), dtype=torch.uint8, device=inp.device)
    is_u8 = inp.dtype == torch.uint8
    with torch_device_fn.device(inp.device):
        any_seg_kernel[grid](
            inp,
            out,
            mid,
            outer,
            D,
            inner,
            d_chunk,
            split,
            split > 1,
            tile_d,
            tile_i,
            even_d,
            even_i,
            is_u8,
            num_warps=4,
        )
        if split > 1:
            total = outer * inner
            any_seg_combine_kernel[(triton.cdiv(total, 1024),)](
                mid, out, total, split, _np2(split), 1024, num_warps=1
            )


def _reduce_dims_layout(inp, dims):
    """(outer, D, inner) for a contiguous input when dims are an index range."""
    if not inp.is_contiguous():
        return None
    lo, hi = dims[0], dims[-1]
    if hi - lo + 1 != len(dims):
        return None
    shape = inp.shape
    outer, D, inner = 1, 1, 1
    for i in range(lo):
        outer *= shape[i]
    for i in range(lo, hi + 1):
        D *= shape[i]
    for i in range(hi + 1, inp.ndim):
        inner *= shape[i]
    return outer, D, inner


def _any_reduce_dims(inp, dims, keepdim, out_shape):
    numel = inp.numel()
    if numel == 0:
        d_prod = 1
        for d in dims:
            d_prod *= inp.shape[d]
        if d_prod == 0:
            # any over an empty reduce range is False
            out = torch.zeros(out_shape, dtype=torch.bool, device=inp.device)
        else:
            out = torch.empty(out_shape, dtype=torch.bool, device=inp.device)
        if not keepdim:
            for d in reversed(dims):
                out = out.squeeze(dim=d)
        return out

    layout = None
    if numel < INT32_LIMIT:
        layout = _reduce_dims_layout(inp, dims)

    if layout is not None:
        outer, D, inner = layout
        if outer == 1 and inner == 1:
            out = _launch_global(inp).reshape(out_shape)
            if not keepdim:
                for d in reversed(dims):
                    out = out.squeeze(dim=d)
            return out
        inp = _to_u8_if_bool(inp)
        out = torch.empty(out_shape, dtype=torch.bool, device=inp.device)
        if inner == 1:
            _launch_row(inp, out, outer, D)
        else:
            _launch_seg(inp, out, outer, D, inner)
        if not keepdim:
            for d in reversed(dims):
                out = out.squeeze(dim=d)
        return out

    # Fallback: compress reduced dims to the tail (may transpose-copy).
    shape = list(inp.shape)
    N = 1
    for d in dims:
        N *= shape[d]
    inp = dim_compress(inp, dims)
    inp = _to_u8_if_bool(inp)
    M = inp.numel() // N
    out = torch.empty(out_shape, dtype=torch.bool, device=inp.device)
    _launch_row(inp, out, M, N)
    if not keepdim:
        for d in reversed(dims):
            out = out.squeeze(dim=d)
    return out


def any(inp):
    logger.debug("GEMS_ENFLAME ANY")
    return _launch_global(inp)


def any_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_ENFLAME ANY_DIM")
    if dim is None:
        out = any(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
        return out
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    return _any_reduce_dims(inp, [dim], keepdim, _out_shape(inp, [dim]))


def any_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_ENFLAME ANY_DIMS")

    if dim is None or isinstance(dim, int):
        return any_dim(inp, dim=dim, keepdim=keepdim)
    assert all(-inp.ndim <= d < inp.ndim for d in dim), "Invalid dim"

    dims = sorted(set(d % inp.ndim for d in dim))
    if not dims:
        # empty dim list: no reduction, plain cast (matches torch semantics)
        return inp.to(torch.bool)
    return _any_reduce_dims(inp, dims, keepdim, _out_shape(inp, dims))


def _out_shape(inp, dims):
    shape = list(inp.shape)
    for d in dims:
        shape[d] = 1
    return shape
