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

import builtins
import logging
import struct

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MAX_DIMS = 8
_BLOCK = 1024
_G_MINMAX = 64
_CAP = 320
_SMALL = 8192
_MAX_REG_BINS = 4096

_SUPPORTED = (
    torch.float32,
    torch.float64,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
)


# ---------------------------------------------------------------- helpers
def _f64_bits(value):
    u = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
    return u - 0x10000000000000000 if u >= 0x8000000000000000 else u


def _cast_scalar(value, dtype):
    """Replicate ATen's Scalar::to<scalar_t>() for the supported dtypes."""
    if dtype == torch.float32:
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]
    if dtype == torch.float64:
        return float(value)
    iv = int(value)  # truncation toward zero, as in C++
    if dtype == torch.uint8:
        iv &= 0xFF
    elif dtype == torch.int8:
        iv = ((iv + 128) & 0xFF) - 128
    elif dtype == torch.int16:
        iv = ((iv + 32768) & 0xFFFF) - 32768
    elif dtype == torch.int32:
        iv = ((iv + 0x80000000) & 0xFFFFFFFF) - 0x80000000
    else:
        iv = ((iv + 0x8000000000000000) & 0xFFFFFFFFFFFFFFFF) - 0x8000000000000000
    return iv


def _merge_dims(x):
    """Metadata-only: collapse adjacent dims that are contiguous in memory."""
    shp = []
    strd = []
    for sz, st in zip(x.shape, x.stride()):
        sz = int(sz)
        st = int(st)
        if sz == 1:
            continue
        if shp and strd[-1] == sz * st:
            shp[-1] *= sz
        else:
            shp.append(sz)
            strd.append(st)
    if not shp:
        shp = [1]
        strd = [0]
    return shp, strd


# ---------------------------------------------------------------- kernels
@triton.jit
def _offsets(
    idx,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    ND: tl.constexpr,
):
    off = (idx % s0) * t0
    rem = idx // s0
    if ND > 1:
        off += (rem % s1) * t1
        rem = rem // s1
    if ND > 2:
        off += (rem % s2) * t2
        rem = rem // s2
    if ND > 3:
        off += (rem % s3) * t3
        rem = rem // s3
    if ND > 4:
        off += (rem % s4) * t4
        rem = rem // s4
    if ND > 5:
        off += (rem % s5) * t5
        rem = rem // s5
    if ND > 6:
        off += (rem % s6) * t6
        rem = rem // s6
    if ND > 7:
        off += (rem % s7) * t7
    return off


@triton.jit
def _bins_keep(v, m, lo, hi, bins, DT: tl.constexpr):
    keep = m & (v >= lo) & (v <= hi)
    if DT == 0:
        den = hi - lo
        den = tl.where(den == 0, 1, den)
        b = (v - lo) * bins // den
        b = tl.where(lo == hi, bins // 2, b)
        b = tl.minimum(b, bins - 1)
    else:
        den = hi - lo
        den = tl.where(den == 0, 1.0, den)
        b = ((v - lo) * bins / den).to(tl.int32)
        b = tl.where(lo == hi, bins // 2, b)
        b = tl.maximum(tl.minimum(b, bins - 1), 0)
    return b, keep


@triton.jit
def _hist_c_kernel(
    x_ptr,
    ws_ptr,
    pmin_ptr,
    pmax_ptr,
    N,
    bins,
    lo_v,
    hi_v,
    DT: tl.constexpr,
    AUTO: tl.constexpr,
    SINGLE: tl.constexpr,
    G: tl.constexpr,
    NUM_BINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if AUTO:
        gi = tl.arange(0, G)
        lo = tl.min(tl.load(pmin_ptr + gi), axis=0)
        hi = tl.max(tl.load(pmax_ptr + gi), axis=0)
    else:
        if DT == 2:
            lo = tl.cast(lo_v, tl.float64, bitcast=True)
            hi = tl.cast(hi_v, tl.float64, bitcast=True)
        else:
            lo = lo_v
            hi = hi_v

    acc = tl.zeros([NUM_BINS], tl.int32)
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    niter = tl.cdiv(N, BLOCK)
    k = pid
    while k < niter:
        idx = k.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = idx < N
        if DT == 0:
            v = tl.load(x_ptr + idx, mask=m, other=0).to(tl.int64)
        else:
            v = tl.load(x_ptr + idx, mask=m, other=0.0)
        b, keep = _bins_keep(v, m, lo, hi, bins, DT)
        acc += tl.histogram(tl.where(keep, b, bins), NUM_BINS)
        k += nprog
    cols = tl.arange(0, NUM_BINS)
    if SINGLE:
        tl.store(ws_ptr + cols, acc.to(ws_ptr.dtype.element_ty), mask=cols < bins)
    else:
        tl.store(ws_ptr + pid * bins + cols, acc, mask=cols < bins)


@triton.jit
def _hist_s_kernel(
    x_ptr,
    ws_ptr,
    pmin_ptr,
    pmax_ptr,
    N,
    bins,
    lo_v,
    hi_v,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    ND: tl.constexpr,
    DT: tl.constexpr,
    AUTO: tl.constexpr,
    SINGLE: tl.constexpr,
    G: tl.constexpr,
    NUM_BINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if AUTO:
        gi = tl.arange(0, G)
        lo = tl.min(tl.load(pmin_ptr + gi), axis=0)
        hi = tl.max(tl.load(pmax_ptr + gi), axis=0)
    else:
        if DT == 2:
            lo = tl.cast(lo_v, tl.float64, bitcast=True)
            hi = tl.cast(hi_v, tl.float64, bitcast=True)
        else:
            lo = lo_v
            hi = hi_v

    acc = tl.zeros([NUM_BINS], tl.int32)
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    niter = tl.cdiv(N, BLOCK)
    k = pid
    while k < niter:
        idx = k.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = idx < N
        off = _offsets(
            idx, s0, s1, s2, s3, s4, s5, s6, s7, t0, t1, t2, t3, t4, t5, t6, t7, ND
        )
        if DT == 0:
            v = tl.load(x_ptr + off, mask=m, other=0).to(tl.int64)
        else:
            v = tl.load(x_ptr + off, mask=m, other=0.0)
        b, keep = _bins_keep(v, m, lo, hi, bins, DT)
        acc += tl.histogram(tl.where(keep, b, bins), NUM_BINS)
        k += nprog
    cols = tl.arange(0, NUM_BINS)
    if SINGLE:
        tl.store(ws_ptr + cols, acc.to(ws_ptr.dtype.element_ty), mask=cols < bins)
    else:
        tl.store(ws_ptr + pid * bins + cols, acc, mask=cols < bins)


@triton.jit
def _minmax_kernel(
    x_ptr,
    pmin_ptr,
    pmax_ptr,
    N,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    ND: tl.constexpr,
    CONTIG: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    niter = tl.cdiv(N, BLOCK)
    if DT == 0:
        lmin = tl.full([], 2**62, tl.int64)
        lmax = tl.full([], -(2**62), tl.int64)
    else:
        lmin = float("inf")
        lmax = -float("inf")
    k = pid
    while k < niter:
        idx = k.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = idx < N
        if CONTIG:
            off = idx
        else:
            off = _offsets(
                idx, s0, s1, s2, s3, s4, s5, s6, s7, t0, t1, t2, t3, t4, t5, t6, t7, ND
            )
        if DT == 0:
            v = tl.load(x_ptr + off, mask=m, other=0).to(tl.int64)
            cmin = tl.min(tl.where(m, v, 2**62), axis=0)
            cmax = tl.max(tl.where(m, v, -(2**62)), axis=0)
        else:
            v = tl.load(x_ptr + off, mask=m, other=0.0)
            cmin = tl.min(tl.where(m, v, float("inf")), axis=0)
            cmax = tl.max(tl.where(m, v, -float("inf")), axis=0)
        lmin = tl.minimum(lmin, cmin)
        lmax = tl.maximum(lmax, cmax)
        k += nprog
    tl.store(pmin_ptr + pid, lmin)
    tl.store(pmax_ptr + pid, lmax)


@triton.jit
def _reduce_kernel(
    ws_ptr, out_ptr, G, bins, NUM_BINS: tl.constexpr, RBLOCK: tl.constexpr
):
    cols = tl.arange(0, NUM_BINS)
    acc = tl.zeros([NUM_BINS], tl.int32)
    p = 0
    while p < G:
        rows = p + tl.arange(0, RBLOCK)
        v = tl.load(
            ws_ptr + rows[:, None] * bins + cols[None, :],
            mask=(rows < G)[:, None] & (cols < bins)[None, :],
            other=0,
        )
        acc += tl.sum(v, axis=0)
        p += RBLOCK
    tl.store(out_ptr + cols, acc.to(out_ptr.dtype.element_ty), mask=cols < bins)


# fallback for very large bin counts: per-element atomics
@triton.jit
def _hist_atomic_kernel(
    x_ptr,
    acc_ptr,
    pmin_ptr,
    pmax_ptr,
    N,
    bins,
    lo_v,
    hi_v,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    ND: tl.constexpr,
    CONTIG: tl.constexpr,
    DT: tl.constexpr,
    AUTO: tl.constexpr,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if AUTO:
        gi = tl.arange(0, G)
        lo = tl.min(tl.load(pmin_ptr + gi), axis=0)
        hi = tl.max(tl.load(pmax_ptr + gi), axis=0)
    else:
        if DT == 2:
            lo = tl.cast(lo_v, tl.float64, bitcast=True)
            hi = tl.cast(hi_v, tl.float64, bitcast=True)
        else:
            lo = lo_v
            hi = hi_v
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    niter = tl.cdiv(N, BLOCK)
    k = pid
    while k < niter:
        idx = k.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = idx < N
        if CONTIG:
            off = idx
        else:
            off = _offsets(
                idx, s0, s1, s2, s3, s4, s5, s6, s7, t0, t1, t2, t3, t4, t5, t6, t7, ND
            )
        if DT == 0:
            v = tl.load(x_ptr + off, mask=m, other=0).to(tl.int64)
            b, keep = _bins_keep(v, m, lo, hi, bins, DT)
            tl.atomic_add(acc_ptr + b, 1, mask=keep)
        else:
            v = tl.load(x_ptr + off, mask=m, other=0.0)
            b, keep = _bins_keep(v, m, lo, hi, bins, DT)
            tl.atomic_add(acc_ptr + b, 1.0, mask=keep)
        k += nprog


@triton.jit
def _convert_kernel(src_ptr, dst_ptr, bins, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < bins
    v = tl.load(src_ptr + idx, mask=m, other=0)
    tl.store(dst_ptr + idx, v.to(dst_ptr.dtype.element_ty), mask=m)


# ---------------------------------------------------------------- entry
def histc(self, bins=100, min=0, max=0):
    logger.debug("GEMS_HYGON HISTC")
    x = self
    dtype = x.dtype
    if dtype == torch.float16:
        raise RuntimeError("HalfTensor is not supported")
    if dtype not in _SUPPORTED:
        raise NotImplementedError(
            "\"histc\" not implemented for '%s'" % str(dtype).replace("torch.", "")
        )
    bins = int(bins)
    if bins <= 0:
        raise RuntimeError("bins must be > 0")

    lo0 = _cast_scalar(min, dtype)
    hi0 = _cast_scalar(max, dtype)
    if lo0 > hi0:
        raise RuntimeError("max must be larger than min")

    device = x.device
    n = x.numel()
    if n == 0:
        return torch.zeros(bins, dtype=dtype, device=device)

    auto = lo0 == hi0
    is_int = not dtype.is_floating_point
    if is_int:
        dt_code = 0
        lo_v, hi_v = lo0, hi0
    elif dtype == torch.float32:
        dt_code = 1
        lo_v, hi_v = lo0, hi0
    else:
        dt_code = 2
        lo_v, hi_v = _f64_bits(lo0), _f64_bits(hi0)

    contig = x.is_contiguous()
    if contig:
        nd = 1
        shp = [n] + [1] * (_MAX_DIMS - 1)
        strd = [1] + [0] * (_MAX_DIMS - 1)
    else:
        shp, strd = _merge_dims(x)
        nd = len(shp)
        if nd > _MAX_DIMS:
            raise RuntimeError("histc: too many dimensions")
        shp = shp + [1] * (_MAX_DIMS - nd)
        strd = strd + [0] * (_MAX_DIMS - nd)

    num_bins = 1 << bins.bit_length()

    if auto:
        wdtype = torch.int64 if is_int else dtype
        work = torch.empty(2 * _G_MINMAX, dtype=wdtype, device=device)
        pmin = work[:_G_MINMAX]
        pmax = work[_G_MINMAX:]
        _minmax_kernel[(_G_MINMAX,)](
            x,
            pmin,
            pmax,
            n,
            shp[0],
            shp[1],
            shp[2],
            shp[3],
            shp[4],
            shp[5],
            shp[6],
            shp[7],
            strd[0],
            strd[1],
            strd[2],
            strd[3],
            strd[4],
            strd[5],
            strd[6],
            strd[7],
            ND=nd,
            CONTIG=contig,
            DT=dt_code,
            BLOCK=_BLOCK,
        )
    else:
        pmin = x
        pmax = x

    if num_bins > _MAX_REG_BINS:
        narrow_int = is_int and dtype not in (torch.int32, torch.int64)
        target = torch.zeros(
            bins, dtype=(torch.int32 if narrow_int else dtype), device=device
        )
        grid = (builtins.min(triton.cdiv(n, _BLOCK), _CAP),)
        _hist_atomic_kernel[grid](
            x,
            target,
            pmin,
            pmax,
            n,
            bins,
            lo_v,
            hi_v,
            shp[0],
            shp[1],
            shp[2],
            shp[3],
            shp[4],
            shp[5],
            shp[6],
            shp[7],
            strd[0],
            strd[1],
            strd[2],
            strd[3],
            strd[4],
            strd[5],
            strd[6],
            strd[7],
            ND=nd,
            CONTIG=contig,
            DT=dt_code,
            AUTO=auto,
            G=_G_MINMAX,
            BLOCK=_BLOCK,
        )
        if narrow_int:
            out = torch.empty(bins, dtype=dtype, device=device)
            _convert_kernel[(triton.cdiv(bins, _BLOCK),)](
                target, out, bins, BLOCK=_BLOCK
            )
            return out
        return target

    if n <= _SMALL:
        out = torch.empty(bins, dtype=dtype, device=device)
        if contig:
            _hist_c_kernel[(1,)](
                x,
                out,
                pmin,
                pmax,
                n,
                bins,
                lo_v,
                hi_v,
                DT=dt_code,
                AUTO=auto,
                SINGLE=True,
                G=_G_MINMAX,
                NUM_BINS=num_bins,
                BLOCK=_BLOCK,
            )
        else:
            _hist_s_kernel[(1,)](
                x,
                out,
                pmin,
                pmax,
                n,
                bins,
                lo_v,
                hi_v,
                shp[0],
                shp[1],
                shp[2],
                shp[3],
                shp[4],
                shp[5],
                shp[6],
                shp[7],
                strd[0],
                strd[1],
                strd[2],
                strd[3],
                strd[4],
                strd[5],
                strd[6],
                strd[7],
                ND=nd,
                DT=dt_code,
                AUTO=auto,
                SINGLE=True,
                G=_G_MINMAX,
                NUM_BINS=num_bins,
                BLOCK=_BLOCK,
            )
        return out

    g = builtins.min(triton.cdiv(n, _BLOCK), _CAP)
    ws = torch.empty(g * bins, dtype=torch.int32, device=device)
    if contig:
        _hist_c_kernel[(g,)](
            x,
            ws,
            pmin,
            pmax,
            n,
            bins,
            lo_v,
            hi_v,
            DT=dt_code,
            AUTO=auto,
            SINGLE=False,
            G=_G_MINMAX,
            NUM_BINS=num_bins,
            BLOCK=_BLOCK,
        )
    else:
        _hist_s_kernel[(g,)](
            x,
            ws,
            pmin,
            pmax,
            n,
            bins,
            lo_v,
            hi_v,
            shp[0],
            shp[1],
            shp[2],
            shp[3],
            shp[4],
            shp[5],
            shp[6],
            shp[7],
            strd[0],
            strd[1],
            strd[2],
            strd[3],
            strd[4],
            strd[5],
            strd[6],
            strd[7],
            ND=nd,
            DT=dt_code,
            AUTO=auto,
            SINGLE=False,
            G=_G_MINMAX,
            NUM_BINS=num_bins,
            BLOCK=_BLOCK,
        )
    out = torch.empty(bins, dtype=dtype, device=device)
    rblock = 1 if num_bins >= _MAX_REG_BINS else _MAX_REG_BINS // num_bins
    _reduce_kernel[(1,)](ws, out, g, bins, NUM_BINS=num_bins, RBLOCK=rblock)
    return out
