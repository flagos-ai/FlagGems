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

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Main kernel: per-program register histogram, single vector atomic_add at end
# ---------------------------------------------------------------------------
@triton.jit
def histc_acc(
    inp_ptr,
    out_ptr,
    n,
    bins,
    minv,
    maxv,
    BLOCK: tl.constexpr,
    NB: tl.constexpr,
    NP: tl.constexpr,
    OUT_CODE: tl.constexpr,
):
    pid = tl.program_id(0)
    binsf = bins * 1.0
    width = maxv - minv
    n_chunks = tl.cdiv(n, BLOCK)
    acc = tl.zeros([NB], dtype=tl.int32)
    for c in tl.range(pid, n_chunks, NP):
        offs = c * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(inp_ptr + offs, mask=m, other=minv).to(tl.float32)
        bf = (x - minv) * binsf / width
        bi = bf.to(tl.int32)
        bi = tl.minimum(bi, bins - 1)
        valid = m & (x >= minv) & (x <= maxv)
        bsel = tl.where(valid, bi, NB - 1)
        acc += tl.histogram(bsel, NB)
    b = tl.arange(0, NB)
    bm = b < bins
    if OUT_CODE == 0:
        tl.atomic_add(out_ptr + b, acc.to(tl.float32), mask=bm)
    elif OUT_CODE == 1:
        tl.atomic_add(out_ptr + b, acc, mask=bm)
    else:
        tl.atomic_add(out_ptr + b, acc.to(tl.int64), mask=bm)


# ---------------------------------------------------------------------------
# Single-program store kernel for tiny inputs (no atomic, no zero-init)
# ---------------------------------------------------------------------------
@triton.jit
def histc_single(
    inp_ptr,
    out_ptr,
    n,
    bins,
    minv,
    maxv,
    BLOCK: tl.constexpr,
    NB: tl.constexpr,
    OUT_CODE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(inp_ptr + offs, mask=m, other=minv).to(tl.float32)
    bf = (x - minv) * (bins * 1.0) / (maxv - minv)
    bi = bf.to(tl.int32)
    bi = tl.minimum(bi, bins - 1)
    valid = m & (x >= minv) & (x <= maxv)
    bsel = tl.where(valid, bi, NB - 1)
    acc = tl.histogram(bsel, NB)
    b = tl.arange(0, NB)
    bm = b < bins
    if OUT_CODE == 0:
        tl.store(out_ptr + b, acc.to(tl.float32), mask=bm)
    elif OUT_CODE == 1:
        tl.store(out_ptr + b, acc, mask=bm)
    else:
        tl.store(out_ptr + b, acc.to(tl.int64), mask=bm)


# ---------------------------------------------------------------------------
# One-chunk-per-program kernel (no loop), for mid-size inputs
# ---------------------------------------------------------------------------
@triton.jit
def histc_noloop(
    inp_ptr,
    out_ptr,
    n,
    bins,
    minv,
    maxv,
    BLOCK: tl.constexpr,
    NB: tl.constexpr,
    OUT_CODE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(inp_ptr + offs, mask=m, other=minv).to(tl.float32)
    bf = (x - minv) * (bins * 1.0) / (maxv - minv)
    bi = bf.to(tl.int32)
    bi = tl.minimum(bi, bins - 1)
    valid = m & (x >= minv) & (x <= maxv)
    bsel = tl.where(valid, bi, NB - 1)
    acc = tl.histogram(bsel, NB)
    b = tl.arange(0, NB)
    bm = b < bins
    if OUT_CODE == 0:
        tl.atomic_add(out_ptr + b, acc.to(tl.float32), mask=bm)
    elif OUT_CODE == 1:
        tl.atomic_add(out_ptr + b, acc, mask=bm)
    else:
        tl.atomic_add(out_ptr + b, acc.to(tl.int64), mask=bm)


# ---------------------------------------------------------------------------
# Per-element atomic fallback for very large NB (register acc infeasible)
# ---------------------------------------------------------------------------
@triton.jit
def histc_simple(inp_ptr, out_ptr, n, bins, minv, maxv, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(inp_ptr + offs, mask=m, other=float("nan")).to(tl.float32)
    bin_idx = tl.floor((x - minv) * bins / (maxv - minv)).to(tl.int32)
    bin_idx = tl.where(x == maxv, bins - 1, bin_idx)
    in_range = (x >= minv) & (x <= maxv)
    bin_idx = tl.where(bin_idx < 0, 0, bin_idx)
    bin_idx = tl.where(bin_idx >= bins, bins - 1, bin_idx)
    valid = m & in_range
    tl.atomic_add(out_ptr + bin_idx, 1.0, mask=valid, sem="relaxed")


# ---------------------------------------------------------------------------
# min == max -> count elements equal to min into out[0]
# ---------------------------------------------------------------------------
@triton.jit
def histc_count(inp_ptr, out_ptr, n, minv, BLOCK: tl.constexpr, OUT_CODE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(inp_ptr + offs, mask=m, other=float("nan"))
    cnt = tl.sum(tl.where(m & (x == minv), 1, 0).to(tl.int64))
    if OUT_CODE == 0:
        tl.atomic_add(out_ptr, cnt.to(tl.float32))
    elif OUT_CODE == 1:
        tl.atomic_add(out_ptr, cnt.to(tl.int32))
    else:
        tl.atomic_add(out_ptr, cnt.to(tl.int64))


# ---------------------------------------------------------------------------
# device min/max for the default (min==max==0) path
# ---------------------------------------------------------------------------
@triton.jit
def minmax_partial(inp_ptr, mm_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(inp_ptr + offs, mask=m, other=float("nan"))
    mn = tl.min(tl.where(m, x, float("inf")))
    mx = tl.max(tl.where(m, x, float("-inf")))
    tl.store(mm_ptr + pid * 2, mn)
    tl.store(mm_ptr + pid * 2 + 1, mx)


@triton.jit
def minmax_reduce(mm_ptr, out_ptr, nprogs, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < nprogs
    mn = tl.load(mm_ptr + offs * 2, mask=m, other=float("inf"))
    mx = tl.load(mm_ptr + offs * 2 + 1, mask=m, other=float("-inf"))
    tl.store(out_ptr, tl.min(mn))
    tl.store(out_ptr + 1, tl.max(mx))


def _out_code(dtype):
    if dtype == torch.int32:
        return 1
    if dtype == torch.int64:
        return 2
    return 0


def _next_pow2(v):
    return 1 << (v - 1).bit_length()


def histc(inp, bins=100, min=0, max=0):
    n = inp.numel()
    dtype = inp.dtype
    if dtype == torch.float64:
        return torch.zeros(bins, dtype=dtype, device=inp.device)
    if n == 0:
        return torch.zeros(bins, dtype=dtype, device=inp.device)
    if inp.is_contiguous():
        xf = inp.view(-1)
    else:
        xf = inp.contiguous().view(-1)
    oc = _out_code(dtype)
    mn = min
    mx = max

    # default: expand to data min/max (device kernels, then host scalar read)
    if mn == 0.0 and mx == 0.0:
        n2 = builtins.min((n + 2047) // 2048, 128)
        mm = torch.empty(n2 * 2, dtype=torch.float32, device=inp.device)
        minmax_partial[(n2,)](xf, mm, n, BLOCK=2048)
        bb = 1
        while bb < n2:
            bb *= 2
        minmax_reduce[(1,)](mm, mm, n2, BLOCK=bb)
        mn = float(mm[0].item())
        mx = float(mm[1].item())

    if mn == mx:
        out = torch.zeros(bins, dtype=dtype, device=inp.device)
        n_chunks = (n + 2047) // 2048
        grid = builtins.min(n_chunks, 128)
        histc_count[(grid,)](xf, out, n, mn, BLOCK=2048, OUT_CODE=oc)
        return out

    nb = _next_pow2(bins)

    # tiny inputs: single program, direct store (no zero-init needed)
    if n <= 4096 and nb <= 4096:
        out = torch.empty(bins, dtype=dtype, device=inp.device)
        histc_single[(1,)](
            xf, out, n, bins, mn, mx, BLOCK=4096, NB=nb, OUT_CODE=oc, num_warps=32
        )
        return out

    out = torch.zeros(bins, dtype=dtype, device=inp.device)

    # very large bins: per-element atomics (register acc infeasible)
    if nb > 4096:
        n_chunks = (n + 2047) // 2048
        grid = builtins.min(n_chunks, 4096)
        histc_simple[(grid,)](xf, out, n, bins, mn, mx, BLOCK=2048)
        return out

    # tuned tiers (probe27/31/32): all num_warps=16
    if n <= 262144:
        # small: looped, BLOCK=2048, power-of-2 NP (avoids backend pathology)
        BLOCK = 2048
        n_chunks = (n + BLOCK - 1) // BLOCK
        np_ = builtins.min(n_chunks, 32)
        histc_acc[(np_,)](
            xf,
            out,
            n,
            bins,
            mn,
            mx,
            BLOCK=BLOCK,
            NB=nb,
            NP=np_,
            OUT_CODE=oc,
            num_warps=16,
        )
        return out
    if n <= 2097152:
        # mid: one chunk per program, BLOCK=8192 (smaller grid -> cheaper launch)
        BLOCK = 8192
        grid = (n + BLOCK - 1) // BLOCK
        histc_noloop[(grid,)](
            xf, out, n, bins, mn, mx, BLOCK=BLOCK, NB=nb, OUT_CODE=oc, num_warps=16
        )
        return out
    # large: looped, BLOCK=16384, NP=48
    BLOCK = 16384
    n_chunks = (n + BLOCK - 1) // BLOCK
    np_ = builtins.min(n_chunks, 48)
    histc_acc[(np_,)](
        xf, out, n, bins, mn, mx, BLOCK=BLOCK, NB=nb, NP=np_, OUT_CODE=oc, num_warps=16
    )
    return out
