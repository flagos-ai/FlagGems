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
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# ---- design notes (Kunlunxin XPU, 2026-08-17) -------------------------------
# prod = multiply-reduction.
# * Flat (dim=None): rows-of-8192 main pass with the 2-D row kernel (few wide
#   programs, mask-free), pow2-decomposed unmasked tail chunks (XPU masked-tail
#   loads are unreliable -> never mask), staged fp32/idtree. 8192-lane reduce is
#   the reliable width on this backend.
# * dim path: reduced axes are moved innermost via `permute(order)`; the
#   physical reorder uses the native strided-copy engine
#   (`torch.ops.aten._copy_from`, flag_gems never overrides it) instead of the
#   slow gems `.contiguous()`; then a [BLOCK_M, BLOCK_N] 2D row kernel reduces
#   the innermost N (reduce-INSIDE accumulate in fp32).
# * N == 1 (all reduced dims of size 1): identity -> native `_copy_from`.
# * N == 0: product over an empty dim = 1 -> fill.
# * ints: accumulate in the input width (wrap-around like torch.prod) since
#   fp32 accumulation saturates to inf for large int products.

_REDUCE_BLOCK = 8192  # reliable single-load tl.reduce width
_FAST_BN_FP16 = (1024, 256, 512, 128, 64, 32, 16)
_FAST_BN_FP32 = (512, 256, 1024, 128, 64, 32, 16)
_FAST_BM_FP16 = (128, 64, 32, 256, 16, 8, 4, 2, 1)
_FAST_BM_FP32 = (64, 128, 32, 256, 16, 8, 4, 2, 1)


@triton.jit
def reduce_mul(a, b):
    return a * b


def _pick_fast_tile(M, N, is_fp32):
    """Return (BLOCK_M, BLOCK_N) with M % BLOCK_M == 0 and N % BLOCK_N == 0, so
    the whole reduction runs mask-free, or None."""
    bns = _FAST_BN_FP32 if is_fp32 else _FAST_BN_FP16
    bms = _FAST_BM_FP32 if is_fp32 else _FAST_BM_FP16
    bn = next((b for b in bns if N % b == 0), None)
    if bn is None:
        return None
    bm = next((m for m in bms if M % m == 0), None)
    if bm is None:
        return None
    return bm, bn


def _work_dtype(dt):
    # floats: fp32 accumulation; ints: keep native width (wrap like torch)
    return torch.float32 if dt.is_floating_point else dt


@libentry()
@triton.jit
def prod_row2d(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
    ACC32: tl.constexpr,
):
    # Map the program id to its rows and pre-offset the base pointer so the
    # inner `inp + cols` access is proven contiguous by OffsetAnalysis.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    # reduce-INSIDE accumulation: each [BLOCK_M, BLOCK_N] block is reduced over
    # N (axis=1) first, then multiplied into a [BLOCK_M, 1] accumulator. This
    # is the only form that is numerically reliable on this XPU (the
    # reduce-OUTSIDE variant with a persistent tile miscompiles for bf16 with
    # masked tails). NEED_MASK=False compiles to a fully mask-free kernel.
    if ACC32:
        acc = tl.full([BLOCK_M, 1], value=1.0, dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            if NEED_MASK:
                mask = row_mask and (cols < N)
                a = tl.load(inp + cols, mask, other=1.0).to(tl.float32)
            else:
                a = tl.load(inp + cols).to(tl.float32)
            blk = tl.reduce(a, axis=1, combine_fn=reduce_mul)[:, None]
            acc = acc * blk
    else:
        v0 = tl.load(inp + tl.zeros([BLOCK_M, BLOCK_N], tl.int32))
        acc = tl.full([BLOCK_M, 1], value=1, dtype=v0.dtype)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            if NEED_MASK:
                mask = row_mask and (cols < N)
                a = tl.load(inp + cols, mask, other=1)
            else:
                a = tl.load(inp + cols)
            blk = tl.reduce(a, axis=1, combine_fn=reduce_mul)[:, None]
            acc = acc * blk
    if NEED_MASK:
        tl.store(out, acc, row_mask)
    else:
        tl.store(out, acc)


@libentry()
@triton.jit
def prod_mid_block(inp, mid, BLOCK: tl.constexpr, ACC32: tl.constexpr):
    # one PID per BLOCK (<= 8192) contiguous unmasked chunk -> partial
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if ACC32:
        v = tl.load(inp + offs).to(tl.float32)
    else:
        v = tl.load(inp + offs)
    p = tl.reduce(v, axis=0, combine_fn=reduce_mul)
    tl.store(mid + pid, p)


@libentry()
@triton.jit
def prod_tailk(inp, out, START, WIDTH: tl.constexpr, ACC32: tl.constexpr):
    # exact pow2 slice [START, START+WIDTH) -> fully in-bounds, no mask
    offs = START + tl.arange(0, WIDTH)
    if ACC32:
        v = tl.load(inp + offs).to(tl.float32)
    else:
        v = tl.load(inp + offs)
    p = tl.reduce(v, axis=0, combine_fn=reduce_mul)
    tl.store(out, p)


@libentry()
@triton.jit
def prod_final(inp, out, WIDTH: tl.constexpr, ACC32: tl.constexpr):
    offs = tl.arange(0, WIDTH)
    if ACC32:
        v = tl.load(inp + offs).to(tl.float32)
    else:
        v = tl.load(inp + offs)
    p = tl.reduce(v, axis=0, combine_fn=reduce_mul)
    tl.store(out, p)


def _pow2_decomp(r):
    parts = []
    while r:
        p = 1 << (r.bit_length() - 1)
        parts.append(p)
        r -= p
    return parts


def _reduce_partials(data, n, out, device, acc32):
    """Staged product of `n` fp32/int partials -> scalar `out`, mask-free
    (tails pow2-decomposed; final pad of 1.0s)."""
    with torch_device_fn.device(device):
        while n > _REDUCE_BLOCK:
            k = n // _REDUCE_BLOCK
            r = n - k * _REDUCE_BLOCK
            chunks = _pow2_decomp(r)
            sz = k + len(chunks)
            midn = torch.empty((sz,), dtype=data.dtype, device=device)
            if k:
                prod_mid_block[(k, 1)](
                    data, midn, _REDUCE_BLOCK, acc32, buffer_size_limit=2048
                )
            pos = k * _REDUCE_BLOCK
            for i, w in enumerate(chunks):
                prod_tailk[(1, 1)](
                    data, midn[k + i : k + i + 1], pos, w, acc32,
                    buffer_size_limit=2048,
                )
                pos += w
            data = midn
            n = sz
        width = triton.next_power_of_2(n)
        if width == n:
            prod_final[(1, 1)](data, out, width, acc32, buffer_size_limit=2048)
        else:
            pad = torch.full((width,), 1, dtype=data.dtype, device=device)
            if n:
                torch.ops.aten._copy_from(data, pad[:n], False)
            prod_final[(1, 1)](pad, out, width, acc32, buffer_size_limit=2048)


def _prod_flat(inp, out, device):
    numel = inp.numel()
    block = _REDUCE_BLOCK
    rows = numel // block
    res = numel - rows * block
    is_fp32 = inp.dtype == torch.float32
    acc32 = inp.dtype.is_floating_point
    wdt = _work_dtype(inp.dtype)
    with torch_device_fn.device(device):
        if rows:
            tile = _pick_fast_tile(rows, block, is_fp32)
            bm = tile[0] if tile else 2
            bn = 1024 if not is_fp32 else 512
            chunks = _pow2_decomp(res) if res else []
            mid = torch.empty((rows + len(chunks),), dtype=wdt, device=device)
            prod_row2d[(max(rows // bm, 1), 1)](
                inp, mid, rows, block, bm, bn, False, acc32,
                buffer_size_limit=2048,
            )
            pos = rows * block
            for i, w in enumerate(chunks):
                prod_tailk[(1, 1)](
                    inp, mid[rows + i : rows + i + 1], pos, w, acc32,
                    buffer_size_limit=2048,
                )
                pos += w
        else:
            chunks = _pow2_decomp(res)
            mid = torch.empty((len(chunks),), dtype=wdt, device=device)
            pos = 0
            for i, w in enumerate(chunks):
                prod_tailk[(1, 1)](
                    inp, mid[i : i + 1], pos, w, acc32, buffer_size_limit=2048
                )
                pos += w
        _reduce_partials(mid, mid.numel(), out, device, acc32)


def prod(inp, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN PROD")
    if dtype is None:
        dtype = inp.dtype
    numel = inp.numel()
    out = torch.empty([], dtype=dtype, device=inp.device)
    if numel == 0:
        out.fill_(1)
        return out
    if numel == 1:
        with torch_device_fn.device(inp.device):
            torch.ops.aten._copy_from(inp.reshape([]), out, False)
        return out
    with torch_device_fn.device(inp.device):
        _prod_flat(inp, out, inp.device)
    return out


def prod_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN PROD_DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    if dtype is None:
        dtype = inp.dtype
    shape = list(inp.shape)
    d = dim % inp.ndim
    N = shape[d]
    M = 1
    for s in shape[:d]:
        M *= s
    K = 1
    for s in shape[d + 1 :]:
        K *= s

    out_shape = shape.copy()
    out_shape[d] = 1
    out = torch.empty(out_shape, dtype=dtype, device=inp.device)
    if M == 0 or K == 0:
        if not keepdim:
            out = torch.squeeze(out, d)
        return out
    if N == 0:
        out.fill_(1)
        if not keepdim:
            out = torch.squeeze(out, d)
        return out
    if N == 1:
        # reduce over a size-1 dim is the identity
        with torch_device_fn.device(inp.device):
            torch.ops.aten._copy_from(inp, out, False)
        if not keepdim:
            out = torch.squeeze(out, d)
        return out

    # Move the reduced dim innermost (same order as dim_compress) and make it
    # contiguous with the native strided-copy engine instead of gems'
    # `.contiguous()` (which is ~1000x slower for big transposes).
    order = [i for i in range(inp.dim()) if i != d] + [d]
    view = inp.permute(order)
    if view.is_contiguous():
        src = view
    else:
        src = torch.empty(list(view.shape), dtype=inp.dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            torch.ops.aten._copy_from(view, src, False)

    rows = M * K
    out_flat = out.reshape(rows)
    is_fp32 = dtype == torch.float32
    acc32 = dtype.is_floating_point
    with torch_device_fn.device(inp.device):
        tile = _pick_fast_tile(rows, N, is_fp32)
        if tile is not None:
            bm, bn = tile
            prod_row2d[(max(rows // bm, 1), 1)](
                src, out_flat, rows, N, bm, bn, False, acc32,
                buffer_size_limit=2048,
            )
        else:
            bn = min(triton.next_power_of_2(N), _REDUCE_BLOCK)
            bm = triton.next_power_of_2(min(triton.cdiv(rows, 12), 65536 // bn))
            grid = (triton.cdiv(rows, bm),)
            prod_row2d[grid](
                src, out_flat, rows, N, bm, bn, True, acc32,
                buffer_size_limit=2048,
            )
    if not keepdim:
        out = torch.squeeze(out, d)
    return out