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
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

NUM_SIPS = 24

INT32_LIMIT = 2**31 - 1
# Max padded BLOCK_C for the copy-free block kernels; K scales inversely with
# BLOCK_C so that a [BLOCK_K, BLOCK_C] tile stays within the per-CTA DSM budget.
BLOCK_C_CAP = 1024
# Per-CTA streaming tile budget in bytes (BLOCK_K * BLOCK_C * element_size).
TILE_BYTES = 262144
# Minimum per-CTA work (B * C elements) to amortize CTA dispatch on the
# copy-free block path; smaller cases fall back to the compress+row path.
MIN_BLOCK_WORK = 1024


@triton.jit
def reduce_all(a, b):
    return a & b


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def all_global_kernel(
    inp_ptr,
    mid_ptr,
    N_total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    acc = tl.full([BLOCK], value=1, dtype=tl.int1)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        val = tl.load(inp_ptr + off, mask=mask, other=1)
        acc = acc & (val != 0)
    result = tl.reduce(acc, axis=0, combine_fn=reduce_all)
    tl.store(mid_ptr + pid, result)


@libentry()
@triton.jit
def all_reduce_kernel(mid_ptr, out_ptr, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptr + offset, mask=mask, other=1).to(tl.int1)
    result = tl.reduce(mid_val, axis=0, combine_fn=reduce_all)
    tl.store(out_ptr, result)


def _keep_config(conf):
    bm = conf.kwargs["BLOCK_M"]
    bn = conf.kwargs["BLOCK_N"]
    if bm * bn > 131072:
        return False
    return True


@libentry()
@triton.jit
def all_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _all = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask

        a = tl.load(inp + cols, mask, other=1.0)
        _all = _all & (a != 0)
    all_result = tl.reduce(_all, axis=1, combine_fn=reduce_all)
    tl.store(out, all_result[:, None], row_mask)


@libentry()
@triton.jit
def all_block_stream_kernel(
    inp,
    out,
    B,
    C,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    EVEN_B: tl.constexpr,
    EVEN_C: tl.constexpr,
):
    # One program per outer block `a`: out[a * C + w] = AND_b inp[a * B * C + b * C + w].
    # The [BLOCK_K, BLOCK_C] tile is a contiguous rectangle, so the streaming
    # load is fully coalesced; the AND stays elementwise and the cross-row
    # reduce runs only once at the end.
    a = tl.program_id(0)
    base = inp + a * B * C
    w = tl.arange(0, BLOCK_C)
    acc = tl.full([BLOCK_K, BLOCK_C], value=1, dtype=tl.int1)
    for off in tl.range(0, B, BLOCK_K):
        k = off + tl.arange(0, BLOCK_K)
        ptrs = base + k[:, None] * C + w[None, :]
        if EVEN_B:
            if EVEN_C:
                tile = tl.load(ptrs)
            else:
                tile = tl.load(ptrs, mask=(w < C)[None, :], other=1)
        else:
            m = (k < B)[:, None]
            if EVEN_C:
                tile = tl.load(ptrs, mask=m, other=1)
            else:
                tile = tl.load(ptrs, mask=m & (w < C)[None, :], other=1)
        acc = acc & (tile != 0)
    res = tl.reduce(acc, axis=0, combine_fn=reduce_all)
    if EVEN_C:
        tl.store(out + a * C + w, res)
    else:
        tl.store(out + a * C + w, res, mask=w < C)


@libentry()
@triton.jit
def all_block_partial_kernel(
    inp,
    mid,
    B,
    C,
    ROWS_PER_CHUNK,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    EVEN_C: tl.constexpr,
):
    # grid = (O, CHUNKS): program (a, ch) reduces rows [ch*RPC, min((ch+1)*RPC, B))
    # of block `a` into mid[a, ch, :]; used when O is too small to fill the SPCs.
    a = tl.program_id(0)
    ch = tl.program_id(1)
    base = inp + a * B * C
    b0 = ch * ROWS_PER_CHUNK
    b1 = tl.minimum(b0 + ROWS_PER_CHUNK, B)
    w = tl.arange(0, BLOCK_C)
    acc = tl.full([BLOCK_K, BLOCK_C], value=1, dtype=tl.int1)
    for off in tl.range(b0, b1, BLOCK_K):
        k = off + tl.arange(0, BLOCK_K)
        ptrs = base + k[:, None] * C + w[None, :]
        if EVEN_C:
            tile = tl.load(ptrs, mask=(k < b1)[:, None], other=1)
        else:
            tile = tl.load(ptrs, mask=(k < b1)[:, None] & (w < C)[None, :], other=1)
        acc = acc & (tile != 0)
    res = tl.reduce(acc, axis=0, combine_fn=reduce_all)
    mid_ptrs = mid + (a * tl.num_programs(1) + ch) * C + w
    if EVEN_C:
        tl.store(mid_ptrs, res)
    else:
        tl.store(mid_ptrs, res, mask=w < C)


@libentry()
@triton.jit
def all_block_mid_kernel(
    mid,
    out,
    C,
    CHUNKS,
    BLOCK_W: tl.constexpr,
    BLOCK_CH: tl.constexpr,
    EVEN_W: tl.constexpr,
):
    # out[a, w] = AND_ch mid[a, ch, w]; w is chunked to bound the tile size.
    a = tl.program_id(0)
    ch = tl.arange(0, BLOCK_CH)
    for woff in tl.range(0, C, BLOCK_W):
        w = woff + tl.arange(0, BLOCK_W)
        m = (ch < CHUNKS)[:, None]
        if not EVEN_W:
            m = m & (w < C)[None, :]
        val = tl.load(
            mid + (a * CHUNKS + ch[:, None]) * C + w[None, :], mask=m, other=1
        ).to(tl.int1)
        res = tl.reduce(val, axis=0, combine_fn=reduce_all)
        if EVEN_W:
            tl.store(out + a * C + w, res)
        else:
            tl.store(out + a * C + w, res, mask=w < C)


def _to_u8_if_bool(inp):
    if inp.dtype == torch.bool:
        return inp.view(torch.uint8)
    return inp


def _block_decompose(inp, red_dims):
    # Copy-free [O, B, C] view of a contiguous input whose reduction dims (with
    # size > 1) form one contiguous block of positions in the (original) dim
    # order: flat element index = o * B * C + b * C + c. Size-1 dims are
    # layout-neutral and fold into O or C. Returns (O, B, C) or None.
    if not inp.is_contiguous():
        return None
    sizes = inp.shape
    big = [i for i in range(inp.ndim) if sizes[i] > 1]
    red_positions = [k for k, d in enumerate(big) if d in red_dims]
    if not red_positions:
        return None
    if red_positions[-1] - red_positions[0] != len(red_positions) - 1:
        return None
    p0, p1 = red_positions[0], red_positions[-1]
    outer_size = 1
    for k in range(p0):
        outer_size *= sizes[big[k]]
    B = 1
    for k in range(p0, p1 + 1):
        B *= sizes[big[k]]
    C = 1
    for k in range(p1 + 1, len(big)):
        C *= sizes[big[k]]
    return outer_size, B, C


def _launch_row_all(inp, out, M, N):
    block_m = 32 if M >= 256 else 8
    block_n = min(triton.next_power_of_2(N), 2048)
    num_warps = 4 if M >= 256 else 1
    grid = (triton.cdiv(M, block_m),)
    with torch_device_fn.device(inp.device):
        all_kernel_dim[grid](
            inp,
            out,
            M,
            N,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )


def _launch_block_all(inp, out, outer_size, B, C):
    elem_size = inp.element_size()
    block_c = triton.next_power_of_2(C)
    even_c = block_c == C
    tile_cap = max(1, TILE_BYTES // (block_c * elem_size))
    k_pref = 128
    if outer_size >= NUM_SIPS * 2:
        kk = min(k_pref, tile_cap)
        block_k = 1 << (kk.bit_length() - 1)
        with torch_device_fn.device(inp.device):
            all_block_stream_kernel[(outer_size,)](
                inp,
                out,
                B,
                C,
                BLOCK_K=block_k,
                BLOCK_C=block_c,
                EVEN_B=(B % block_k == 0),
                EVEN_C=even_c,
                num_warps=2,
            )
    else:
        block_k = 1 << (min(k_pref, tile_cap).bit_length() - 1)
        chunks = max(1, min(triton.cdiv(NUM_SIPS * 2, outer_size), 255, B))
        mid = torch.empty((outer_size, chunks, C), dtype=torch.bool, device=inp.device)
        rows_per_chunk = triton.cdiv(B, chunks)
        block_w = min(triton.next_power_of_2(C), 1024)
        with torch_device_fn.device(inp.device):
            all_block_partial_kernel[(outer_size, chunks)](
                inp,
                mid,
                B,
                C,
                rows_per_chunk,
                BLOCK_K=block_k,
                BLOCK_C=block_c,
                EVEN_C=even_c,
                num_warps=2,
            )
            all_block_mid_kernel[(outer_size,)](
                mid,
                out,
                C,
                chunks,
                BLOCK_W=block_w,
                BLOCK_CH=triton.next_power_of_2(chunks),
                EVEN_W=(C % block_w == 0),
                num_warps=2,
            )


def all(inp):
    logger.debug("GEMS_ENFLAME ALL")
    inp = _to_u8_if_bool(inp)
    N_total = inp.numel()
    if N_total == 0:
        return torch.tensor(True, dtype=torch.bool, device=inp.device)

    if N_total <= 4096:
        BLOCK = max(triton.next_power_of_2(N_total), 1024)
    elif N_total <= 65536:
        BLOCK = 4096
    else:
        BLOCK = 65536
    num_blocks = triton.cdiv(N_total, BLOCK)
    grid_size = min(num_blocks, NUM_SIPS * 2)

    mid = torch.empty((grid_size,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)

    with torch_device_fn.device(inp.device):
        all_global_kernel[(grid_size,)](inp, mid, N_total, BLOCK, num_warps=2)
        block_mid = triton.next_power_of_2(grid_size)
        all_reduce_kernel[(1,)](mid, out, grid_size, block_mid)

    return out


def _all_dim_impl(inp, dims, keepdim):
    shape = list(inp.shape)
    numel = inp.numel()
    N = 1
    for d in dims:
        N *= shape[d]
    out_shape = [1 if i in dims else s for i, s in enumerate(shape)]

    if numel > 0 and N == numel:
        # Full reduction: reuse the tuned global two-stage path. Non-reduced
        # dims can only be size-1 here, so the value is exact; the output shape
        # keeps one size-1 dim per surviving (non-reduced) dim.
        out = all(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
        else:
            surviving = sum(1 for i in range(inp.ndim) if i not in dims)
            out = torch.reshape(out, [1] * surviving)
        return out

    inp_u8 = _to_u8_if_bool(inp)
    dec = None
    if 0 < numel <= INT32_LIMIT:
        dec = _block_decompose(inp, dims)
        if dec is not None:
            outer_size, B, C = dec
            if C > BLOCK_C_CAP or B * C < MIN_BLOCK_WORK:
                dec = None

    if dec is not None:
        outer_size, B, C = dec
        out = torch.empty(out_shape, dtype=torch.bool, device=inp.device)
        if C == 1 and (outer_size >= NUM_SIPS * 2 or B <= 65536):
            # Reduction block at the end: contiguous row reduction. With very
            # few rows and a huge row length, route to the chunked block path
            # below instead so the row loop is not serialized in ~O CTAs.
            _launch_row_all(inp_u8, out, outer_size, B)
        else:
            _launch_block_all(inp_u8, out, outer_size, B, C)
    else:
        inp_c = dim_compress(inp, dims)
        inp_c = _to_u8_if_bool(inp_c)
        M = inp_c.numel() // N
        out = torch.empty(out_shape, dtype=torch.bool, device=inp.device)
        _launch_row_all(inp_c, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=list(dims))
    return out


def all_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_ENFLAME ALL_DIM")
    if dim is None:
        out = all(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
        return out
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    return _all_dim_impl(inp, (dim,), keepdim)


def all_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_ENFLAME ALL_DIMS")

    if dim is None or isinstance(dim, int):
        return all_dim(inp, dim=dim, keepdim=keepdim)
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"

    dims = tuple(d % inp.ndim for d in dim)
    return _all_dim_impl(inp, dims, keepdim)
