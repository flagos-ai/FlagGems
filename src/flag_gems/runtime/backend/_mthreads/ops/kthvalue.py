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

from flag_gems.ops.kthvalue import kthvalue as default_kthvalue

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


# ---------------------------------------------------------------------------
# kthvalue for MUSA (triton-musa backend).
#
# Reference semantics (torch on MUSA, verified empirically):
#   values[..]  = k-th smallest element along `dim` (ascending order,
#                 NaN treated as largest, like a NaN-last stable sort)
#   indices[..] = the LAST (largest) position along `dim` whose element
#                 equals the k-th smallest value
#
# Algorithm: per-slice binary search over a monotonic integer encoding of
# the values (IEEE-754 float bits mapped to an order-preserving uint, or the
# native int64 value), then a final scan for the last matching position.
# This is exact for ties and NaN, and avoids tl.sort (which is unreliable
# with NaN on this backend).
# ---------------------------------------------------------------------------


@triton.jit
def _count_chunked(
    x_ptr,
    base,
    D,
    cols,
    mid,
    CHUNK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    W64: tl.constexpr,
):
    """Count valid elements <= mid by streaming the slice in chunks."""
    cnt = tl.zeros([], dtype=tl.int64)
    if IS_FLOAT:
        if W64:
            for c in range(0, D, CHUNK):
                offs = c + tl.arange(0, CHUNK)
                m = offs < D
                x = tl.load(x_ptr + base + offs * cols, mask=m, other=0.0)
                xu = x.to(tl.float64).to(tl.uint64, bitcast=True)
                xo = tl.where(
                    xu & 0x8000000000000000 != 0, ~xu, xu | 0x8000000000000000
                )
                cnt += tl.sum((xo <= mid) & m).to(tl.int64)
        else:
            for c in range(0, D, CHUNK):
                offs = c + tl.arange(0, CHUNK)
                m = offs < D
                x = tl.load(x_ptr + base + offs * cols, mask=m, other=0.0)
                xu = x.to(tl.float32).to(tl.uint32, bitcast=True)
                xo = tl.where(xu & 0x80000000 != 0, ~xu, xu | 0x80000000)
                cnt += tl.sum((xo <= mid) & m).to(tl.int64)
    else:
        for c in range(0, D, CHUNK):
            offs = c + tl.arange(0, CHUNK)
            m = offs < D
            x = tl.load(x_ptr + base + offs * cols, mask=m, other=0)
            xi = x.to(tl.int64)
            cnt += tl.sum((xi <= mid) & m).to(tl.int64)
    return cnt


@triton.jit
def _kth_smallk_kernel(
    x_ptr,
    v_ptr,
    i_ptr,
    k,
    D,
    cols,
    num_slices,
    D2: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    W64: tl.constexpr,
    KMAX: tl.constexpr,
):
    """Fast path for small k: iteratively remove the current minimum value
    (all copies) in the ordered-bit domain until the cumulative count reaches
    k; then report that value and the LAST position equal to it."""
    pid = tl.program_id(0)
    if pid < num_slices:
        row = pid // cols
        col = pid % cols
        base = row * D * cols + col
        offs = tl.arange(0, D2)
        mask = offs < D
        x = tl.load(x_ptr + base + offs * cols, mask=mask, other=0.0)

        if IS_FLOAT:
            if W64:
                xo = x.to(tl.float64).to(tl.uint64, bitcast=True)
                xo = tl.where(
                    xo & 0x8000000000000000 != 0, ~xo, xo | 0x8000000000000000
                )
                maxv = tl.full([], 0xFFFFFFFFFFFFFFFF, tl.uint64)
            else:
                xo = x.to(tl.float32).to(tl.uint32, bitcast=True)
                xo = tl.where(xo & 0x80000000 != 0, ~xo, xo | 0x80000000)
                maxv = tl.full([], 0xFFFFFFFF, tl.uint32)
        else:
            xo = x.to(tl.int64)
            maxv = tl.full([], 9223372036854775807, tl.int64)

        cum = tl.zeros([], dtype=tl.int64)
        prev = maxv
        done = tl.zeros([], dtype=tl.int1)
        v_ans = maxv
        i_ans = tl.full([], -1, tl.int64)
        for r in tl.static_range(KMAX):
            if r == 0:
                cand = tl.where(mask, xo, maxv)
            else:
                cand = tl.where(mask & (xo > prev), xo, maxv)
            vmin = tl.min(cand)
            cnt = tl.sum((xo == vmin) & mask).to(tl.int64)
            newcum = cum + cnt
            take = (newcum >= k) & (~done)
            idx_cand = tl.where((xo == vmin) & mask, offs, -1)
            v_ans = tl.where(take, vmin, v_ans)
            i_ans = tl.where(take, tl.max(idx_cand, axis=0).to(tl.int64), i_ans)
            done = done | take
            cum = tl.where(done, cum, newcum)
            prev = tl.where(done, prev, vmin)

        if IS_FLOAT:
            if W64:
                v_bits = tl.where(
                    (v_ans & 0x8000000000000000) != 0,
                    v_ans & 0x7FFFFFFFFFFFFFFF,
                    ~v_ans,
                )
                v = v_bits.to(tl.float64, bitcast=True)
            else:
                v_bits = tl.where((v_ans & 0x80000000) != 0, v_ans & 0x7FFFFFFF, ~v_ans)
                v = v_bits.to(tl.float32, bitcast=True)
        else:
            v = v_ans
        tl.store(v_ptr + pid, v.to(v_ptr.dtype.element_ty))
        tl.store(i_ptr + pid, i_ans.to(tl.int64))


@triton.jit
def _kth_smallk_multi_kernel(
    x_ptr,
    v_ptr,
    i_ptr,
    k,
    D,
    num_slices,
    num_blocks,
    D2: tl.constexpr,
    SPP: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    W64: tl.constexpr,
    KMAX: tl.constexpr,
):
    """cols==1 specialization with power-of-2 D (D2 == D): one program handles
    SPP consecutive slices via a single contiguous SPP*D2-element load reshaped
    to (SPP, D2); per-slice min-removal runs in parallel along axis 1."""
    pid = tl.program_id(0)
    if pid < num_blocks:
        row0 = pid * SPP
        tile = tl.arange(0, SPP * D2)
        lm = tile < (num_slices - row0) * D
        x = tl.load(x_ptr + row0 * D + tile, mask=lm, other=0.0)
        xr = tl.reshape(x, (SPP, D2))

        if IS_FLOAT:
            if W64:
                xu = xr.to(tl.float64).to(tl.uint64, bitcast=True)
                xo = tl.where(
                    xu & 0x8000000000000000 != 0, ~xu, xu | 0x8000000000000000
                )
                maxv = tl.full([SPP], 0xFFFFFFFFFFFFFFFF, tl.uint64)
            else:
                xu = xr.to(tl.float32).to(tl.uint32, bitcast=True)
                xo = tl.where(xu & 0x80000000 != 0, ~xu, xu | 0x80000000)
                maxv = tl.full([SPP], 0xFFFFFFFF, tl.uint32)
        else:
            xo = xr.to(tl.int64)
            maxv = tl.full([SPP], 9223372036854775807, tl.int64)

        col_off = tl.arange(0, D2)[None, :]
        cum = tl.zeros([SPP], dtype=tl.int64)
        prev = maxv
        done = tl.zeros([SPP], dtype=tl.int1)
        v_ans = maxv
        i_ans = tl.full([SPP], -1, tl.int64)
        for r in tl.static_range(KMAX):
            if r == 0:
                cand = xo
            else:
                cand = tl.where(xo > prev[:, None], xo, maxv[:, None])
            vmin = tl.min(cand, axis=1)
            cnt = tl.sum((xo == vmin[:, None]), axis=1).to(tl.int64)
            newcum = cum + cnt
            take = (newcum >= k) & (~done)
            idx_cand = tl.where(xo == vmin[:, None], col_off, -1)
            v_ans = tl.where(take, vmin, v_ans)
            i_ans = tl.where(take, tl.max(idx_cand, axis=1).to(tl.int64), i_ans)
            done = done | take
            cum = tl.where(done, cum, newcum)
            prev = tl.where(done, prev, vmin)

        if IS_FLOAT:
            if W64:
                v_bits = tl.where(
                    (v_ans & 0x8000000000000000) != 0,
                    v_ans & 0x7FFFFFFFFFFFFFFF,
                    ~v_ans,
                )
                v = v_bits.to(tl.float64, bitcast=True)
            else:
                v_bits = tl.where((v_ans & 0x80000000) != 0, v_ans & 0x7FFFFFFF, ~v_ans)
                v = v_bits.to(tl.float32, bitcast=True)
        else:
            v = v_ans
        sid = row0 + tl.arange(0, SPP)
        sm = sid < num_slices
        tl.store(v_ptr + sid, v.to(v_ptr.dtype.element_ty), mask=sm)
        tl.store(i_ptr + sid, i_ans.to(tl.int64), mask=sm)


@triton.jit
def _kth_stream_kernel(
    x_ptr,
    v_ptr,
    i_ptr,
    k,
    D,
    cols,
    num_slices,
    CHUNK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    W64: tl.constexpr,
    UNSIGNED: tl.constexpr,
    NIT: tl.constexpr,
):
    """One program per slice; bounded registers (works for any D)."""
    pid = tl.program_id(0)
    if pid < num_slices:
        row = pid // cols
        col = pid % cols
        base = row * D * cols + col

        if IS_FLOAT:
            if W64:
                lo = tl.zeros([], dtype=tl.uint64)
                hi = tl.full([], 0xFFFFFFFFFFFFFFFF, tl.uint64)
            else:
                lo = tl.zeros([], dtype=tl.uint32)
                hi = tl.full([], 0xFFFFFFFF, tl.uint32)
            for _ in tl.static_range(NIT):
                mid = (lo >> 1) + (hi >> 1) + (lo & hi & 1)
                cnt = _count_chunked(x_ptr, base, D, cols, mid, CHUNK, IS_FLOAT, W64)
                take_hi = cnt >= k
                hi = tl.where(take_hi, mid, hi)
                lo = tl.where(take_hi, lo, mid + 1)
            v_ord = lo
            if W64:
                v_bits = tl.where(
                    (v_ord & 0x8000000000000000) != 0,
                    v_ord & 0x7FFFFFFFFFFFFFFF,
                    ~v_ord,
                )
                v = v_bits.to(tl.float64, bitcast=True)
            else:
                v_bits = tl.where((v_ord & 0x80000000) != 0, v_ord & 0x7FFFFFFF, ~v_ord)
                v = v_bits.to(tl.float32, bitcast=True)
        else:
            if W64:
                lo = tl.full([], -9223372036854775808, tl.int64)
                hi = tl.full([], 9223372036854775807, tl.int64)
            elif UNSIGNED:
                lo = tl.zeros([], dtype=tl.int64)
                hi = tl.full([], 4294967295, tl.int64)
            else:
                lo = tl.full([], -2147483648, tl.int64)
                hi = tl.full([], 2147483647, tl.int64)
            for _ in tl.static_range(NIT):
                mid = (lo >> 1) + (hi >> 1) + (lo & hi & 1)
                cnt = _count_chunked(x_ptr, base, D, cols, mid, CHUNK, IS_FLOAT, W64)
                take_hi = cnt >= k
                hi = tl.where(take_hi, mid, hi)
                lo = tl.where(take_hi, lo, mid + 1)
            v = lo

        # generic last-occurrence scan
        v_is_nan = v != v
        best = tl.full([], -1, tl.int64)
        for c in range(0, D, CHUNK):
            offs = c + tl.arange(0, CHUNK)
            m = offs < D
            x = tl.load(x_ptr + base + offs * cols, mask=m, other=0)
            hits = (x == v) | ((x != x) & v_is_nan)
            cand = tl.where(hits & m, offs, -1)
            best = tl.maximum(best, tl.max(cand, axis=0).to(tl.int64))
        tl.store(v_ptr + pid, v.to(v_ptr.dtype.element_ty))
        tl.store(i_ptr + pid, best.to(tl.int64))


@triton.jit
def _kth_cached_kernel(
    x_ptr,
    v_ptr,
    i_ptr,
    k,
    D,
    cols,
    num_slices,
    D2: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    W64: tl.constexpr,
    UNSIGNED: tl.constexpr,
    NIT: tl.constexpr,
):
    """Register-cached variant: load the slice once, search on registers."""
    pid = tl.program_id(0)
    if pid < num_slices:
        row = pid // cols
        col = pid % cols
        base = row * D * cols + col
        offs = tl.arange(0, D2)
        mask = offs < D
        x = tl.load(x_ptr + base + offs * cols, mask=mask, other=0.0)

        if IS_FLOAT:
            if W64:
                xu = x.to(tl.float64).to(tl.uint64, bitcast=True)
                xo = tl.where(
                    xu & 0x8000000000000000 != 0, ~xu, xu | 0x8000000000000000
                )
                lo = tl.zeros([], dtype=tl.uint64)
                hi = tl.full([], 0xFFFFFFFFFFFFFFFF, tl.uint64)
            else:
                xu = x.to(tl.float32).to(tl.uint32, bitcast=True)
                xo = tl.where(xu & 0x80000000 != 0, ~xu, xu | 0x80000000)
                lo = tl.zeros([], dtype=tl.uint32)
                hi = tl.full([], 0xFFFFFFFF, tl.uint32)
            for _ in tl.static_range(NIT):
                mid = (lo >> 1) + (hi >> 1) + (lo & hi & 1)
                cnt = tl.sum((xo <= mid) & mask).to(tl.int64)
                take_hi = cnt >= k
                hi = tl.where(take_hi, mid, hi)
                lo = tl.where(take_hi, lo, mid + 1)
            v_ord = lo
            if W64:
                v_bits = tl.where(
                    (v_ord & 0x8000000000000000) != 0,
                    v_ord & 0x7FFFFFFFFFFFFFFF,
                    ~v_ord,
                )
                v = v_bits.to(tl.float64, bitcast=True)
            else:
                v_bits = tl.where((v_ord & 0x80000000) != 0, v_ord & 0x7FFFFFFF, ~v_ord)
                v = v_bits.to(tl.float32, bitcast=True)
        else:
            if W64:
                lo = tl.full([], -9223372036854775808, tl.int64)
                hi = tl.full([], 9223372036854775807, tl.int64)
            elif UNSIGNED:
                lo = tl.zeros([], dtype=tl.int64)
                hi = tl.full([], 4294967295, tl.int64)
            else:
                lo = tl.full([], -2147483648, tl.int64)
                hi = tl.full([], 2147483647, tl.int64)
            xi = x.to(tl.int64)
            for _ in tl.static_range(NIT):
                mid = (lo >> 1) + (hi >> 1) + (lo & hi & 1)
                cnt = tl.sum((xi <= mid) & mask).to(tl.int64)
                take_hi = cnt >= k
                hi = tl.where(take_hi, mid, hi)
                lo = tl.where(take_hi, lo, mid + 1)
            v = lo

        v_is_nan = v != v
        hits = (x == v) | ((x != x) & v_is_nan)
        hits = hits & mask
        idx = tl.max(tl.where(hits, offs, -1).to(tl.int64))
        tl.store(v_ptr + pid, v.to(v_ptr.dtype.element_ty))
        tl.store(i_ptr + pid, idx.to(tl.int64))


_D_CACHE = 4096
_CHUNK = 2048


def _specialized_kthvalue(inp, k, dim=-1, keepdim=False):
    if torch.is_tensor(k):
        k = int(k.item())
    if torch.is_tensor(dim):
        dim = int(dim.item())
    k = int(k)

    ndim = inp.dim()
    if ndim == 0:
        raise RuntimeError("kthvalue(): input must be at least 1-d")
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            "Dimension out of range (expected to be in range of [%d, %d], but got %d)"
            % (-ndim, ndim - 1, dim)
        )
    dim = dim % ndim
    D = inp.shape[dim]
    if D == 0:
        raise IndexError(
            "kthvalue(): Expected reduction dim %d to have non-zero size." % dim
        )
    if k < 1 or k > D:
        raise RuntimeError(
            "kthvalue(): selected number k out of range for dimension %d" % dim
        )

    rows = 1
    for s in inp.shape[:dim]:
        rows *= s
    cols = 1
    for s in inp.shape[dim + 1 :]:
        cols *= s
    num_slices = rows * cols

    if not inp.is_contiguous():
        inp = inp.contiguous()

    out_shape = list(inp.shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        del out_shape[dim]

    values = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    indices = torch.empty(out_shape, dtype=torch.int64, device=inp.device)

    dtype = inp.dtype
    if dtype.is_floating_point:
        is_float, unsigned = True, False
        if dtype == torch.float64:
            w64, nit = True, 64
        else:
            w64, nit = False, 32
    else:
        is_float = False
        if dtype == torch.int64:
            w64, unsigned, nit = True, False, 64
        elif dtype in (torch.uint8, torch.uint16, torch.uint32, torch.bool):
            w64, unsigned, nit = False, True, 32
        elif dtype == torch.uint64:
            raise RuntimeError("kthvalue(): uint64 not supported")
        else:
            w64, unsigned, nit = False, False, 32

    v_flat = values.view(num_slices)
    i_flat = indices.view(num_slices)

    if k <= 8 and D <= _D_CACHE:
        D2 = 1
        while D2 < D:
            D2 *= 2
        if cols == 1 and D == D2 and D2 <= 256:
            # multi-slice: one program per SPP consecutive rows (contiguous)
            spp = 512 // D2
            if spp < 1:
                spp = 1
            elif spp > 8:
                spp = 8
            nblocks = (num_slices + spp - 1) // spp
            if D2 <= 128:
                nw = 1
            else:
                nw = 2
            _kth_smallk_multi_kernel[(nblocks,)](
                inp,
                v_flat,
                i_flat,
                k,
                D,
                num_slices,
                nblocks,
                D2=D2,
                SPP=spp,
                IS_FLOAT=is_float,
                W64=w64,
                KMAX=k,
                num_warps=nw,
            )
        else:
            grid = (num_slices,)
            if D2 <= 128:
                nw = 1
            elif D2 <= 256:
                nw = 2
            else:
                nw = 4
            _kth_smallk_kernel[grid](
                inp,
                v_flat,
                i_flat,
                k,
                D,
                cols,
                num_slices,
                D2=D2,
                IS_FLOAT=is_float,
                W64=w64,
                KMAX=k,
                num_warps=nw,
            )
    elif D <= _D_CACHE:
        D2 = 1
        while D2 < D:
            D2 *= 2
        grid = (num_slices,)
        _kth_cached_kernel[grid](
            inp,
            v_flat,
            i_flat,
            k,
            D,
            cols,
            num_slices,
            D2=D2,
            IS_FLOAT=is_float,
            W64=w64,
            UNSIGNED=unsigned,
            NIT=nit,
            num_warps=8 if D2 > 2048 else 4,
        )
    else:
        grid = (num_slices,)
        _kth_stream_kernel[grid](
            inp,
            v_flat,
            i_flat,
            k,
            D,
            cols,
            num_slices,
            CHUNK=_CHUNK,
            IS_FLOAT=is_float,
            W64=w64,
            UNSIGNED=unsigned,
            NIT=nit,
            num_warps=8,
        )

    return values, indices


def kthvalue(inp, k, dim=-1, keepdim=False):
    logger.debug("GEMS_MTHREADS KTHVALUE")
    if (
        isinstance(inp, torch.Tensor)
        and inp.device.type == "musa"
        and inp.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_kthvalue(inp, k, dim, keepdim)
    return default_kthvalue(inp, k, dim, keepdim)
