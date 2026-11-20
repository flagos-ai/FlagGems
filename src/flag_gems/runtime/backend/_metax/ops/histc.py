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

import operator

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# device kernels
# ---------------------------------------------------------------------------


@triton.jit
def _mm_contig(x_ptr, mm_ptr, N, BLOCK: tl.constexpr, F64: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if F64:
        x = x.to(tl.float64)
    else:
        x = x.to(tl.float32)
    lmin = tl.min(tl.where(mask, x, float("inf")))
    lmax = tl.max(tl.where(mask, x, float("-inf")))
    tl.store(mm_ptr + 2 * pid, lmin)
    tl.store(mm_ptr + 2 * pid + 1, lmax)


@triton.jit
def _mm_strided(
    x_ptr,
    mm_ptr,
    N,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    src = (
        ((offs // d0) % s0) * t0
        + ((offs // d1) % s1) * t1
        + ((offs // d2) % s2) * t2
        + ((offs // d3) % s3) * t3
        + ((offs // d4) % s4) * t4
        + ((offs // d5) % s5) * t5
        + ((offs // d6) % s6) * t6
        + ((offs // d7) % s7) * t7
    )
    x = tl.load(x_ptr + src, mask=mask, other=0.0)
    if F64:
        x = x.to(tl.float64)
    else:
        x = x.to(tl.float32)
    lmin = tl.min(tl.where(mask, x, float("inf")))
    lmax = tl.max(tl.where(mask, x, float("-inf")))
    tl.store(mm_ptr + 2 * pid, lmin)
    tl.store(mm_ptr + 2 * pid + 1, lmax)


@triton.jit
def _hist_agg_contig(
    x_ptr,
    out_ptr,
    mm_ptr,
    N,
    nprog_mm,
    nbins,
    nbins_f,
    half_f,
    lo,
    hi,
    HAS_RANGE: tl.constexpr,
    NP2: tl.constexpr,
    BINS_PAD: tl.constexpr,
    OUT_F32: tl.constexpr,
    SINGLE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if not HAS_RANGE:
        idx = tl.arange(0, 2 * NP2)
        m = idx < 2 * nprog_mm
        even = (idx % 2) == 0
        odd = (idx % 2) == 1
        v = tl.load(mm_ptr + idx, mask=m, other=0.0)
        lo = tl.min(tl.where(m & even, v, float("inf")))
        hi = tl.max(tl.where(m & odd, v, float("-inf")))
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    cnt = tl.zeros((BINS_PAD,), dtype=tl.int32)
    for start in range(pid * BLOCK, N, step):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        ok = mask & (x >= lo) & (x <= hi)
        span = hi - lo
        raw = ((x - lo) * nbins_f) / span
        bi = tl.where(span > 0.0, raw, half_f).to(tl.int32)
        bi = tl.minimum(tl.maximum(bi, 0), nbins - 1)
        bi = tl.where(ok, bi, nbins)
        cnt += tl.histogram(bi, BINS_PAD)
    bidx = tl.arange(0, BINS_PAD)
    keep = bidx < nbins
    if OUT_F32:
        c = cnt.to(tl.float32)
    else:
        c = cnt
    if SINGLE:
        tl.store(out_ptr + bidx, c, mask=keep)
    else:
        tl.atomic_add(out_ptr + bidx, c, mask=keep)


@triton.jit
def _hist_contig(
    x_ptr,
    out_ptr,
    mm_ptr,
    N,
    nprog,
    nbins,
    nbins_f,
    half_f,
    lo,
    hi,
    HAS_RANGE: tl.constexpr,
    NP2: tl.constexpr,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
    OUT_F32: tl.constexpr,
):
    if not HAS_RANGE:
        idx = tl.arange(0, 2 * NP2)
        m = idx < 2 * nprog
        odd = (idx % 2) == 1
        even = (idx % 2) == 0
        v = tl.load(mm_ptr + idx, mask=m, other=0.0)
        lo = tl.min(tl.where(m & even, v, float("inf")))
        hi = tl.max(tl.where(m & odd, v, float("-inf")))
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if F64:
        x = x.to(tl.float64)
    else:
        x = x.to(tl.float32)
    ok = mask & (x >= lo) & (x <= hi)
    span = hi - lo
    raw = ((x - lo) * nbins_f) / span
    bi = tl.where(span > 0.0, raw, half_f).to(tl.int32)
    bi = tl.minimum(tl.maximum(bi, 0), nbins - 1)
    if OUT_F32:
        tl.atomic_add(out_ptr + bi, 1.0, mask=ok)
    else:
        tl.atomic_add(out_ptr + bi, 1, mask=ok)


@triton.jit
def _hist_strided(
    x_ptr,
    out_ptr,
    mm_ptr,
    N,
    nprog,
    nbins,
    nbins_f,
    half_f,
    lo,
    hi,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    HAS_RANGE: tl.constexpr,
    NP2: tl.constexpr,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
    OUT_F32: tl.constexpr,
):
    if not HAS_RANGE:
        idx = tl.arange(0, 2 * NP2)
        m = idx < 2 * nprog
        odd = (idx % 2) == 1
        even = (idx % 2) == 0
        v = tl.load(mm_ptr + idx, mask=m, other=0.0)
        lo = tl.min(tl.where(m & even, v, float("inf")))
        hi = tl.max(tl.where(m & odd, v, float("-inf")))
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    src = (
        ((offs // d0) % s0) * t0
        + ((offs // d1) % s1) * t1
        + ((offs // d2) % s2) * t2
        + ((offs // d3) % s3) * t3
        + ((offs // d4) % s4) * t4
        + ((offs // d5) % s5) * t5
        + ((offs // d6) % s6) * t6
        + ((offs // d7) % s7) * t7
    )
    x = tl.load(x_ptr + src, mask=mask, other=0.0)
    if F64:
        x = x.to(tl.float64)
    else:
        x = x.to(tl.float32)
    ok = mask & (x >= lo) & (x <= hi)
    span = hi - lo
    raw = ((x - lo) * nbins_f) / span
    bi = tl.where(span > 0.0, raw, half_f).to(tl.int32)
    bi = tl.minimum(tl.maximum(bi, 0), nbins - 1)
    if OUT_F32:
        tl.atomic_add(out_ptr + bi, 1.0, mask=ok)
    else:
        tl.atomic_add(out_ptr + bi, 1, mask=ok)


@triton.jit
def _cvt(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(src_ptr + offs, mask=m, other=0)
    tl.store(dst_ptr + offs, v.to(dst_ptr.dtype.element_ty), mask=m)


# ---------------------------------------------------------------------------
# host wrapper
# ---------------------------------------------------------------------------

_MAX_DIMS = 8
_BLOCK = 4096
_BLOCK_MID = 2048
_BLOCK_MM = 4096
_MAX_AGG_BINS = 2047
_SMS = {}


def _sm_count(device):
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    sm = _SMS.get(idx)
    if sm is None:
        sm = torch.cuda.get_device_properties(idx).multi_processor_count
        _SMS[idx] = sm
    return sm


def histc(self, bins=100, min=0, max=0):
    n_bins = operator.index(bins)
    if n_bins <= 0:
        raise RuntimeError("bins must be > 0")
    lo_s = float(min)
    hi_s = float(max)
    if lo_s > hi_s:
        raise RuntimeError("max must be larger than min")

    x = self
    dtype = x.dtype
    if dtype == torch.float16:
        raise RuntimeError("HalfTensor is not supported")
    if dtype == torch.bfloat16:
        raise NotImplementedError("\"histc\" not implemented for 'BFloat16'")
    n = x.numel()
    device = x.device
    if n == 0:
        return torch.zeros(n_bins, dtype=dtype, device=device)

    auto = lo_s == hi_s
    out_f32 = dtype == torch.float32
    out_int = dtype == torch.int32
    nbins_f = float(n_bins)
    half_f = float(n_bins // 2)

    # fast path: contiguous input, output dtype supports direct atomics
    if (out_f32 or out_int) and n_bins <= _MAX_AGG_BINS and x.is_contiguous():
        bins_pad = triton.next_power_of_2(n_bins + 1)
        sm = _sm_count(device)
        tiles = (n + _BLOCK - 1) // _BLOCK
        if tiles > 4 * sm:
            # enough CTAs to saturate: keep wide tiles, 4 warps
            block = _BLOCK
            warps = 4
        elif tiles > 4:
            # medium launch: double SM coverage at constant per-thread work
            block = _BLOCK_MID
            warps = 8
        else:
            # one or a few CTAs: widen each CTA instead of the grid
            block = _BLOCK
            warps = 8
        nprog = (n + block - 1) // block
        cap = 8 * sm
        if nprog > cap:
            nprog = cap
        single = nprog == 1
        if single:
            out = torch.empty(n_bins, dtype=dtype, device=device)
        else:
            out = torch.zeros(n_bins, dtype=dtype, device=device)
        if auto:
            nprog_mm = (n + _BLOCK_MM - 1) // _BLOCK_MM
            np2 = triton.next_power_of_2(nprog_mm)
            mm = torch.empty(2 * nprog_mm, dtype=torch.float32, device=device)
            _mm_contig[(nprog_mm,)](x, mm, n, _BLOCK_MM, False, num_warps=4)
            _hist_agg_contig[(nprog,)](
                x,
                out,
                mm,
                n,
                nprog_mm,
                n_bins,
                nbins_f,
                half_f,
                lo_s,
                hi_s,
                False,
                np2,
                bins_pad,
                out_f32,
                single,
                block,
                num_warps=warps,
            )
        else:
            _hist_agg_contig[(nprog,)](
                x,
                out,
                out,
                n,
                0,
                n_bins,
                nbins_f,
                half_f,
                lo_s,
                hi_s,
                True,
                1,
                bins_pad,
                out_f32,
                single,
                block,
                num_warps=warps,
            )
        return out

    # general path
    f64 = dtype == torch.float64
    acc = torch.zeros(
        n_bins, dtype=dtype if (out_f32 or out_int) else torch.int32, device=device
    )
    if auto:
        nprog_mm = (n + _BLOCK_MM - 1) // _BLOCK_MM
        np2 = triton.next_power_of_2(nprog_mm)
        mm = torch.empty(
            2 * nprog_mm, dtype=torch.float64 if f64 else torch.float32, device=device
        )
    else:
        nprog_mm = 0
        np2 = 1
        mm = acc
    has_range = not auto
    nprog = (n + _BLOCK - 1) // _BLOCK

    if x.is_contiguous():
        if auto:
            _mm_contig[(nprog_mm,)](x, mm, n, _BLOCK_MM, f64, num_warps=4)
        _hist_contig[(nprog,)](
            x,
            acc,
            mm,
            n,
            nprog_mm,
            n_bins,
            nbins_f,
            half_f,
            lo_s,
            hi_s,
            has_range,
            np2,
            _BLOCK,
            f64,
            out_f32,
            num_warps=4,
        )
    else:
        shape = list(x.shape)
        stride = list(x.stride())
        nd = len(shape)
        divs = [1] * nd
        cur = 1
        for d in range(nd - 1, -1, -1):
            divs[d] = cur
            cur *= shape[d]
        if nd <= _MAX_DIMS:
            pad = _MAX_DIMS - nd
            sargs = shape + [1] * pad + divs + [1] * pad + stride + [0] * pad
            if auto:
                _mm_strided[(nprog_mm,)](x, mm, n, *sargs, _BLOCK_MM, f64, num_warps=4)
            _hist_strided[(nprog,)](
                x,
                acc,
                mm,
                n,
                nprog_mm,
                n_bins,
                nbins_f,
                half_f,
                lo_s,
                hi_s,
                *sargs,
                has_range,
                np2,
                _BLOCK,
                f64,
                out_f32,
                num_warps=4,
            )
        else:
            xc = x.contiguous()
            if auto:
                _mm_contig[(nprog_mm,)](xc, mm, n, _BLOCK_MM, f64, num_warps=4)
            _hist_contig[(nprog,)](
                xc,
                acc,
                mm,
                n,
                nprog_mm,
                n_bins,
                nbins_f,
                half_f,
                lo_s,
                hi_s,
                has_range,
                np2,
                _BLOCK,
                f64,
                out_f32,
                num_warps=4,
            )

    if out_f32 or out_int:
        return acc
    out = torch.empty(n_bins, dtype=dtype, device=device)
    _cvt[(triton.cdiv(n_bins, 256),)](acc, out, n_bins, 256, num_warps=4)
    return out
