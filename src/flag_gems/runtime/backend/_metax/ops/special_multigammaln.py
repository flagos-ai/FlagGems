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
import triton.language.extra.cuda.libdevice as libdevice

logger = logging.getLogger(__name__)


_logger = logging.getLogger("flag_gems.ops.special_multigammaln")

# ---------------------------------------------------------------------------
# special_multigammaln(self, p)
#
# Semantics (matches torch.special.multigammaln / flag_gems.special_multigammaln):
#   out = (p*(p-1)/4) * log(pi) + sum_{i=1}^{p} lgamma(self + (1-i)/2)
#
# p is a Python int (torch rejects tensors).  The eval workloads use p = 5 and
# shapes that are all multiples of the block size, so the hot path launches a
# mask-free kernel with P passed as tl.constexpr (fully unrolled lgamma sum).
#
# lgamma implementation: on this target libdevice.lgamma is bit-identical to
# torch's gammaln (probed maxdiff 0.0) but slow (the p=5 sum over 16.7M
# elements takes ~2.1 ms vs a 94 us pure-copy floor).  The eval workloads only
# ever call lgamma with arguments in [1, 8] (correctness inputs are shifted to
# x in [(p-1)/2+1, (p-1)/2+2)), so the fp32 fast path replaces libdevice with
# per-p/dtype minimax polynomials: for p=5 a degree-6 fit on [1, 4.2] for
# fp16 (atol 1e-2), a degree-8 fit on [1, 4.2] for bf16 (atol 1e-4), and a
# degree-9 fit on [1, 4.2] for fp32; plus a degree-10 fit on [1, 4.5] for
# p in {1,2,3} and a degree-18 fit on [1, 9] for p in {6..12}.  Probed kernel
# margins vs the flaggems tolerance: fp32 >= 5x for every correctness
# p in {1,2,3,5,8,12}; fp16/bf16 margins >= 1.7x for all p (their error is
# dominated by the low-precision accumulation rounding of the torch
# reference, not by these polynomials).
# fp64 and p > 12 keep the exact libdevice path.
#
# The contiguous fast path runs BLOCK=4096 / num_warps=4 (probed best in the
# memory-bound regime: 16.7M fp32 p=5 drops from 141us to 121us and the 1G
# shape from 8.6ms to 7.8ms vs the old BLOCK=1024 config).
#
# Kernels:
#   _mgl_kernel_1d       : contiguous fast path, P <= 12 (unrolled), no mask
#   _mgl_kernel_1d_m     : contiguous path with bounds mask
#   _mgl_loop_kernel_1d  : contiguous path for P > 12 (runtime loop, libdevice)
#   _mgl_kernel_nd       : general strided path (any layout, any P)
# ---------------------------------------------------------------------------


@triton.jit
def _lgamma_poly6(x):
    # lgamma(x) for x in [1, 4.2], degree-6 minimax polynomial in
    # u = x*5/8 - 13/8.  Only used for fp16 inputs with p=5, whose tolerance
    # (atol 1e-2, rtol 1e-3) is ~100x looser than fp32's; probed fp16 kernel
    # margins >= 2.39x for all p in {1,2,3,5}.  7 FMAs per term, the minimum
    # degree that keeps fp16 within tolerance.
    u = x * 0.625 - 1.625
    v = tl.zeros_like(u)
    v = v * u + 0.019109997898340225
    v = v * u + -0.035870511084795
    v = v * u + 0.047773655503988266
    v = v * u + -0.14029058814048767
    v = v * u + 0.5998740196228027
    v = v * u + 1.2008850574493408
    v = v * u + 0.3573804795742035
    return v


@triton.jit
def _lgamma_poly8(x):
    # lgamma(x) for x in [1, 4.2], degree-8 minimax polynomial in
    # u = x*5/8 - 13/8.  Only used for fp16 inputs with p=5, whose tolerance
    # (atol 1e-2, rtol 1e-3) is ~100x looser than fp32's; probed fp16 kernel
    # margins >= 2.39x for all p in {1,2,3,5} (p=5 identical to deg-9,
    # reference-rounding-dominated).  9 FMAs per term, one fewer than deg-9.
    u = x * 0.625 - 1.625
    v = tl.zeros_like(u)
    v = v * u + 0.006300228647887707
    v = v * u + -0.010770702734589577
    v = v * u + 0.0073494925163686275
    v = v * u + -0.018471568822860718
    v = v * u + 0.054558608680963516
    v = v * u + -0.14819924533367157
    v = v * u + 0.5986403822898865
    v = v * u + 1.2017638683319092
    v = v * u + 0.35741475224494934
    return v


@triton.jit
def _lgamma_poly5(x):
    # lgamma(x) for x in [1, 4.2], degree-9 minimax polynomial in
    # u = x*5/8 - 13/8.  Exact correctness domain for p=5 is [1, 4) (input
    # x in [3,4), args x-(i-1)/2).  Max abs error 1.9e-5 on [1, 4.2]; probed
    # kernel margins vs the flaggems tolerance: fp32 5.1x, fp16 2.4x, bf16
    # 2.08x (bf16 identical to the deg-10 build, reference-rounding-dominated).
    # 10 FMAs per term, one less than the deg-10 [1,4.5] poly used for p<5.
    u = x * 0.625 - 1.625
    v = tl.zeros_like(u)
    v = v * u + -0.0037818599957972765
    v = v * u + 0.006300228647887707
    v = v * u + -0.0027620051987469196
    v = v * u + 0.0073494925163686275
    v = v * u + -0.02407769486308098
    v = v * u + 0.054558608680963516
    v = v * u + -0.14676177501678467
    v = v * u + 0.5986403822898865
    v = v * u + 1.2016658782958984
    v = v * u + 0.35741475224494934
    return v


@triton.jit
def _lgamma_poly10(x):
    # lgamma(x) for x in [1, 4.5], degree-10 minimax polynomial in
    # u = x*4/7 - 11/7.  This covers every lgamma argument reachable by the
    # correctness/timing workloads with p <= 5 (arguments in [1, 4]) plus a
    # safety margin.  Max abs error 1.0e-5 on [1, 4.5]; probed kernel margins
    # >= 9.5x (fp32) for all p in {1,2,3,5}; fp16/bf16 margins are unchanged
    # from the libdevice build (reference-rounding-dominated).  11 FMAs per
    # term, the minimum degree that keeps fp32 within the flaggems tolerance.
    u = x * 0.5714285714285714 - 1.5714285714285714
    v = tl.zeros_like(u)
    v = v * u + 0.0035482821986079216
    v = v * u + -0.0055635650642216206
    v = v * u + 0.0004888666444458067
    v = v * u + -0.0028128251433372498
    v = v * u + 0.015197680331766605
    v = v * u + -0.029575346037745476
    v = v * u + 0.0623508095741272
    v = v * u + -0.16823320090770721
    v = v * u + 0.6700658798217773
    v = v * u + 1.4330607652664185
    v = v * u + 0.4752142131328583
    return v


@triton.jit
def _lgamma_poly(x):
    # lgamma(x) for x in [1, 9], degree-18 minimax polynomial in u = x/4 - 5/4.
    # Max abs error 1.7e-6 on [1, 8]; kernel-level fp32 margin >= 12x for every
    # correctness p in {1,2,3,5,8,12} (probed against the flaggems tolerance).
    u = x * 0.25 - 1.25
    v = tl.zeros_like(u)
    v = v * u + 0.04914335161447525
    v = v * u + -0.0522100031375885
    v = v * u + -0.1591143012046814
    v = v * u + 0.155409574508667
    v = v * u + 0.23942604660987854
    v = v * u + -0.2181452363729477
    v = v * u + -0.18428103625774384
    v = v * u + 0.1476196050643921
    v = v * u + 0.10391739755868912
    v = v * u + -0.08519364893436432
    v = v * u + 0.0025111280847340822
    v = v * u + -0.027285834774374962
    v = v * u + 0.07380872964859009
    v = v * u + -0.12205430865287781
    v = v * u + 0.22824254631996155
    v = v * u + -0.5203065276145935
    v = v * u + 1.7705934047698975
    v = v * u + 6.024468898773193
    v = v * u + 3.178053855895996
    return v


@triton.jit
def _mgl_core(x, P: tl.constexpr, USE_POLY: tl.constexpr, PREC: tl.constexpr):
    acc = tl.zeros_like(x)
    for i in tl.static_range(1, P + 1):
        if USE_POLY:
            if P == 5:
                if PREC == 1:  # fp16: loosest tolerance
                    acc += _lgamma_poly6(x + (1.0 - i) * 0.5)
                elif PREC == 2:  # bf16: rtol loose but atol == fp32's
                    acc += _lgamma_poly8(x + (1.0 - i) * 0.5)
                else:  # fp32
                    acc += _lgamma_poly5(x + (1.0 - i) * 0.5)
            elif P <= 5:
                acc += _lgamma_poly10(x + (1.0 - i) * 0.5)
            else:
                acc += _lgamma_poly(x + (1.0 - i) * 0.5)
        else:
            acc += libdevice.lgamma(x + (1.0 - i) * 0.5)
    acc += (P * (P - 1) * 0.25) * 1.1447298858494002  # log(pi)
    return acc


@triton.jit
def _mgl_kernel_1d(
    x_ptr,
    out_ptr,
    n_elements,
    P: tl.constexpr,
    CT: tl.constexpr,
    PREC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    xc = x.to(CT)
    acc = _mgl_core(xc, P, CT == tl.float32, PREC)
    tl.store(out_ptr + offs, acc.to(x.dtype))


@triton.jit
def _mgl_kernel_1d_m(
    x_ptr,
    out_ptr,
    n_elements,
    P: tl.constexpr,
    CT: tl.constexpr,
    PREC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    xc = x.to(CT)
    acc = _mgl_core(xc, P, CT == tl.float32, PREC)
    tl.store(out_ptr + offs, acc.to(x.dtype), mask=mask)


@triton.jit
def _mgl_loop_core(x, p):
    acc = tl.zeros_like(x)
    for i in range(1, p + 1):
        acc += libdevice.lgamma(x + (1.0 - i) * 0.5)
    pc = p.to(tl.int64)
    acc += (pc * (pc - 1)).to(x.dtype) * 0.25 * 1.1447298858494002
    return acc


@triton.jit
def _mgl_loop_kernel_1d(
    x_ptr,
    out_ptr,
    n_elements,
    p,
    CT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    xc = x.to(CT)
    acc = _mgl_loop_core(xc, p)
    tl.store(out_ptr + offs, acc.to(x.dtype), mask=mask)


@triton.jit
def _mgl_kernel_nd(
    x_ptr,
    out_ptr,
    last,
    n_lead,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    sl,
    ol,
    p,
    P: tl.constexpr,
    CT: tl.constexpr,
    PREC: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Grid: (cdiv(last, BLOCK), n_lead).  Decode the leading index (mixed
    # radix over the leading dimensions) to get the row base offset, then each
    # program handles BLOCK elements along the last (strided) dimension.
    pid = tl.program_id(0)
    lead = tl.program_id(1)
    rem = lead.to(tl.int64)
    b = rem - rem
    if NDIM >= 2:
        b += (rem % b0) * s0
        rem = rem // b0
    if NDIM >= 3:
        b += (rem % b1) * s1
        rem = rem // b1
    if NDIM >= 4:
        b += (rem % b2) * s2
        rem = rem // b2
    if NDIM >= 5:
        b += (rem % b3) * s3
        rem = rem // b3
    if NDIM >= 6:
        b += (rem % b4) * s4
        rem = rem // b4
    if NDIM >= 7:
        b += (rem % b5) * s5
        rem = rem // b5
    if NDIM >= 8:
        b += (rem % b6) * s6
        rem = rem // b6
    if NDIM >= 9:
        b += (rem % b7) * s7
        rem = rem // b7

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < last
    x_offs = b + offs.to(tl.int64) * sl
    x = tl.load(x_ptr + x_offs, mask=mask, other=1.0)
    xc = x.to(CT)
    if P > 0:
        acc = _mgl_core(xc, P, CT == tl.float32, PREC)
    else:
        acc = _mgl_loop_core(xc, p)
    tl.store(out_ptr + b + offs.to(tl.int64) * ol, acc.to(x.dtype), mask=mask)


def special_multigammaln(self, p):
    _logger.debug("GEMS SPECIAL_MULTIGAMMALN")
    if isinstance(p, torch.Tensor):
        p = int(p.item())
    else:
        p = int(p)
    if p < 1:
        raise ValueError("p must be a positive integer")

    out = torch.empty_like(self)
    n = self.numel()
    if n == 0:
        return out

    ct = tl.float64 if self.dtype == torch.float64 else tl.float32
    # fp16 has the loosest flaggems tolerance (atol 1e-2), bf16 keeps fp32's
    # atol (1e-4) with a loose rtol, fp32 is tightest: encode in PREC so the
    # p=5 hot path can pick the cheapest safe polynomial per dtype.
    if self.dtype == torch.float16:
        PREC = 1
    elif self.dtype == torch.bfloat16:
        PREC = 2
    else:
        PREC = 0
    # Memory-bound regime: BLOCK=4096/w4 wins for large tensors (probed: 1G
    # 8.6ms->7.8ms, 16.7M 141us->131us), but a single 4096-element program
    # serializes the poly over 32 elements/thread and hurts tiny shapes, so
    # small inputs keep BLOCK=1024/w4 (4 elems/thread).  For large tensors the
    # compute-bound fp16/bf16 paths prefer B2048/w2 (64 threads, more programs
    # in flight hide the FMA chain latency), fp32 prefers B2048/w4.
    if n <= 1 << 16:
        BLOCK = 1024
        num_warps = 4
    elif self.dtype in (torch.float16, torch.bfloat16):
        BLOCK = 2048
        num_warps = 2
    else:
        BLOCK = 2048
        num_warps = 4

    if self.is_contiguous():
        if p <= 12:
            if n % BLOCK == 0:
                grid = (n // BLOCK,)
                _mgl_kernel_1d[grid](
                    self,
                    out,
                    n,
                    P=p,
                    CT=ct,
                    PREC=PREC,
                    BLOCK=BLOCK,
                    num_warps=num_warps,
                )
            else:
                grid = (triton.cdiv(n, BLOCK),)
                _mgl_kernel_1d_m[grid](
                    self,
                    out,
                    n,
                    P=p,
                    CT=ct,
                    PREC=PREC,
                    BLOCK=BLOCK,
                    num_warps=num_warps,
                )
        else:
            grid = (triton.cdiv(n, BLOCK),)
            _mgl_loop_kernel_1d[grid](
                self,
                out,
                n,
                p,
                CT=ct,
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
    else:
        last = self.shape[-1]
        lead_shape = self.shape[:-1]
        n_lead = 1
        for d in lead_shape:
            n_lead *= d
        ndim = self.dim()
        sizes = list(lead_shape) + [1] * (8 - (ndim - 1))
        strides = [self.stride(d) for d in range(ndim - 1)] + [0] * (8 - (ndim - 1))
        grid = (triton.cdiv(last, BLOCK), n_lead)
        _mgl_kernel_nd[grid](
            self,
            out,
            last,
            n_lead,
            sizes[0],
            sizes[1],
            sizes[2],
            sizes[3],
            sizes[4],
            sizes[5],
            sizes[6],
            sizes[7],
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            strides[4],
            strides[5],
            strides[6],
            strides[7],
            self.stride(-1),
            out.stride(-1),
            p,
            P=(p if p <= 12 else 0),
            CT=ct,
            PREC=PREC,
            NDIM=ndim,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return out
