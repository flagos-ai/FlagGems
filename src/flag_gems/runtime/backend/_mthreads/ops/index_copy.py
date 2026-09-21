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

from flag_gems.ops.index_copy_ import index_copy as default_index_copy
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.jit
def _index_copy_copy(inp_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    v = tl.load(inp_ptr + offs, mask=m, other=0.0, eviction_policy="evict_first")
    tl.store(out_ptr + offs, v, mask=m, eviction_policy="evict_first")


@libentry()
@triton.jit
def _index_copy_scatter(
    src_ptr,
    index_ptr,
    out_ptr,
    S,
    L,
    post,
    J_TILE: tl.constexpr,
    C_TILE: tl.constexpr,
):
    # Sequential src rows j -> random output rows index[j]. 3D grid (b, j-chunk, col-tile).
    b = tl.program_id(0)
    jc = tl.program_id(1)
    ct = tl.program_id(2)

    j = jc * J_TILE + tl.arange(0, J_TILE)
    jmask = j < L
    p = tl.load(index_ptr + j, mask=jmask, other=0)

    r = ct * C_TILE + tl.arange(0, C_TILE)
    cmask = r < post

    src_off = (b * L + j)[:, None] * post + r[None, :]
    out_off = (b * S + p)[:, None] * post + r[None, :]

    m = jmask[:, None] & cmask[None, :]
    sv = tl.load(src_ptr + src_off, mask=m, other=0.0, eviction_policy="evict_first")
    tl.store(out_ptr + out_off, sv, mask=m, eviction_policy="evict_first")


@libentry()
@triton.jit
def _index_copy_build_inv(index_ptr, inv_ptr, L, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < L
    p = tl.load(index_ptr + offs, mask=mask, other=0)
    tl.store(inv_ptr + p, offs + 1, mask=mask)


@libentry()
@triton.jit
def _index_copy_gather(
    inp_ptr,
    src_ptr,
    inv_ptr,
    out_ptr,
    S,
    L,
    post,
    R_TILE: tl.constexpr,
    C_TILE: tl.constexpr,
):
    # 3D grid: (b, p-chunk, col-tile). inv stores j+1 (0 = unindexed).
    b = tl.program_id(0)
    pg = tl.program_id(1)
    ct = tl.program_id(2)

    p = pg * R_TILE + tl.arange(0, R_TILE)
    pmask = p < S
    i = tl.load(inv_ptr + p, mask=pmask, other=0, eviction_policy="evict_last")
    sel = i > 0
    j = i - 1

    r = ct * C_TILE + tl.arange(0, C_TILE)
    cmask = r < post

    out_off = (b * S + p)[:, None] * post + r[None, :]
    src_off = (b * L + j)[:, None] * post + r[None, :]

    m = pmask[:, None] & cmask[None, :]
    src_val = tl.load(
        src_ptr + src_off,
        mask=sel[:, None] & m,
        other=0.0,
        eviction_policy="evict_first",
    )
    inp_val = tl.load(
        inp_ptr + out_off,
        mask=(~sel)[:, None] & m,
        other=0.0,
        eviction_policy="evict_first",
    )
    val = tl.where(sel[:, None], src_val, inp_val)
    tl.store(out_ptr + out_off, val, mask=m, eviction_policy="evict_first")


@libentry()
@triton.jit
def _index_copy_gather_search(
    inp_ptr,
    src_ptr,
    index_ptr,
    out_ptr,
    S,
    L,
    post,
    R_TILE: tl.constexpr,
    C_TILE: tl.constexpr,
    LP2: tl.constexpr,
    SSTEP: tl.constexpr,
):
    # Same gather, but builds row->src mapping by searching index in registers
    # (index must be unique). Used for small L to avoid any prepass launch.
    b = tl.program_id(0)
    pg = tl.program_id(1)
    ct = tl.program_id(2)

    p = pg * R_TILE + tl.arange(0, R_TILE)
    pmask = p < S
    p64 = p.to(tl.int64)

    i = tl.full((R_TILE,), -1, dtype=tl.int64)
    for jj0 in tl.static_range(0, LP2, SSTEP):
        jj = jj0 + tl.arange(0, SSTEP)
        jmask = jj < L
        idxv = tl.load(index_ptr + jj, mask=jmask, other=-1)
        m = idxv[None, :] == p64[:, None]
        cur = tl.max(tl.where(m, jj.to(tl.int64)[None, :], -1), axis=1)
        i = tl.maximum(i, cur)
    sel = i >= 0

    r = ct * C_TILE + tl.arange(0, C_TILE)
    cmask = r < post

    out_off = (b * S + p)[:, None] * post + r[None, :]
    src_off = (b * L + i)[:, None] * post + r[None, :]

    m = pmask[:, None] & cmask[None, :]
    src_val = tl.load(
        src_ptr + src_off,
        mask=sel[:, None] & m,
        other=0.0,
        eviction_policy="evict_first",
    )
    inp_val = tl.load(
        inp_ptr + out_off,
        mask=(~sel)[:, None] & m,
        other=0.0,
        eviction_policy="evict_first",
    )
    val = tl.where(sel[:, None], src_val, inp_val)
    tl.store(out_ptr + out_off, val, mask=m, eviction_policy="evict_first")


def _pow2_ceil(x):
    return 1 << max(0, (x - 1).bit_length())


def _use_triton_kernel(inp, dim, index, src) -> bool:
    if (
        not isinstance(inp, torch.Tensor)
        or not isinstance(index, torch.Tensor)
        or not isinstance(src, torch.Tensor)
    ):
        return False
    if inp.device.type != "musa" or inp.dtype not in _SUPPORTED_DTYPES:
        return False
    if not inp.is_contiguous() or not src.is_contiguous() or not index.is_contiguous():
        return False
    if inp.numel() == 0:
        return False
    return True


def index_copy(inp, dim, index, src):
    logger.debug("GEMS_MTHREADS INDEX_COPY")
    if not _use_triton_kernel(inp, dim, index, src):
        return default_index_copy(inp, dim, index, src)

    shape = tuple(inp.shape)
    ndim = len(shape)
    d = int(dim)
    if d < 0:
        d += ndim

    pre = 1
    for s in shape[:d]:
        pre *= s
    S = shape[d]
    post = 1
    for s in shape[d + 1 :]:
        post *= s
    L = int(index.shape[0])
    N = 1
    for s in shape:
        N *= s

    out = torch.empty_like(inp)

    if post == 1:
        C_TILE = 1
    else:
        C_TILE = min(1024, _pow2_ceil(post))

    with torch_device_fn.device(inp.device):
        if L <= 64:
            R_TILE = min(max(1, 2048 // C_TILE), _pow2_ceil(S))
            grid = (pre, triton.cdiv(S, R_TILE), triton.cdiv(post, C_TILE))
            LP2 = _pow2_ceil(max(1, L))
            _index_copy_gather_search[grid](
                inp,
                src,
                index,
                out,
                S,
                L,
                post,
                R_TILE=R_TILE,
                C_TILE=C_TILE,
                LP2=LP2,
                SSTEP=32,
            )
            return out

        if post == 1:
            # inv-gather: best for row-of-1 (fp32) scatter shapes like time-1.
            inv = torch.zeros(S, dtype=torch.int32, device=inp.device)
            I_BLK = 1024
            _index_copy_build_inv[(triton.cdiv(L, I_BLK),)](index, inv, L, BLOCK=I_BLK)
            R_TILE = min(max(1, 2048 // C_TILE), _pow2_ceil(S))
            grid = (pre, triton.cdiv(S, R_TILE), triton.cdiv(post, C_TILE))
            _index_copy_gather[grid](
                inp,
                src,
                inv,
                out,
                S,
                L,
                post,
                R_TILE=R_TILE,
                C_TILE=C_TILE,
            )
            return out

        # copy + scatter (sequential src reads, random writes)
        _index_copy_copy[(triton.cdiv(N, 1024),)](inp, out, N, BLOCK=1024, num_warps=16)
        J_TILE = 128
        grid = (pre, triton.cdiv(L, J_TILE), triton.cdiv(post, C_TILE))
        _index_copy_scatter[grid](
            src,
            index,
            out,
            S,
            L,
            post,
            J_TILE=J_TILE,
            C_TILE=C_TILE,
            num_warps=8,
        )
    return out


__all__ = ["index_copy"]
