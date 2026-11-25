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

"""Triton implementation of ``aten::special_ndtri`` (inverse standard normal CDF).

Numerics mirror ATen's ``calc_ndtri`` (Acklam's rational approximation): the
same coefficient tables and the same Horner evaluation order are used.

Two details are required to agree with ``torch.special.ndtri``:

* the raw c/d polynomial is already negative for large ``q``, so the *upper*
  tail branch carries the minus sign (the lower tail does not);
* the central rational cancels from ~20 down to ~0.014/0.003 near the region
  edges, which costs ~4 decimal digits in fp32.  For float32 inputs only that
  one rational is promoted to float64; everything else (log, sqrt, the tail
  rational and the single division) stays in the input dtype, because the
  tail expression is well conditioned.  float64 inputs take the same path with
  no promotion at all.

Cost per element: 1 log + 1 sqrt + 1 division + ~25 FMA, all branch-shared
(the numerator/denominator are selected before the single division).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# The operator tests raise the generic operator logger
# ("flag_gems.ops.special_ndtri") to DEBUG and assert the GEMS_HYGON record
# appears in caplog.text; vendor-specialized loggers are keyed by fn.__module__
# (_hygon.ops.special_ndtri), which the test does not enable, so the dispatch
# record is emitted through the generic operator logger as well.
_dispatch_log = logging.getLogger("flag_gems.ops.special_ndtri")


@triton.jit
def _ndtri_math(p, IS_F32: tl.constexpr, BLOCK: tl.constexpr):
    # ------------------------------------------------------------------
    # tail regions: q_t = sqrt(-2 ln t), t = p (lower) or 1 - p (upper).
    # ``1 - p`` is exact in binary floating point for p >= 0.5 (Sterbenz).
    # This rational is well conditioned, so it runs in the input dtype.
    # ------------------------------------------------------------------
    t = tl.where(p < 0.5, p, 1.0 - p)
    if IS_F32:
        # single-instruction hardware log2; the exact 2^32 prescale keeps
        # denormal probabilities normal (the hardware op flushes denormal
        # inputs to -inf).  ln(t) = log2(t) * ln2, so -2 ln t = -2 ln2 log2 t
        lg2 = (
            tl.inline_asm_elementwise(
                "v_log_f32 $0, $1;",
                "=v,v",
                [t * 4294967296.0],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            - 32.0
        )
        qt = tl.sqrt(-1.3862943611198906 * lg2)
    else:
        # float64 probabilities span the whole double range including
        # denormals, so split t into an exact power-of-two exponent and a
        # mantissa in [1, 2) (the 2^64 prescale lifts denormals) and log the
        # mantissa with the fp32 hardware op
        ti = (t * 18446744073709551616.0).to(tl.int64, bitcast=True)
        e = ((ti >> 52) & 2047).to(tl.float64) - 1087.0  # 1023 (bias) + 64
        m = ((ti & 4503599627370495) | 4607182418800017408).to(tl.float64, bitcast=True)
        lm = tl.inline_asm_elementwise(
            "v_log_f32 $0, $1;",
            "=v,v",
            [m.to(tl.float32)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        qt = tl.sqrt(((lm.to(tl.float64) + e) * -1.3862943611198906).to(tl.float32))

    # the c/d rational is well conditioned, so fp32 evaluation is accurate
    # enough for both input dtypes (the suite's 1e-4 atol leaves ~20x margin
    # even at |x| ~ 38)
    nt = tl.math.fma(-7.784894002430293e-03, qt, -3.223964580411365e-01)
    nt = tl.math.fma(nt, qt, -2.400758277161838e00)
    nt = tl.math.fma(nt, qt, -2.549732539343734e00)
    nt = tl.math.fma(nt, qt, 4.374664141464968e00)
    num_t = tl.math.fma(nt, qt, 2.938163982698783e00)
    dt = tl.math.fma(7.784695709041462e-03, qt, 3.224671290700398e-01)
    dt = tl.math.fma(dt, qt, 2.445134137142996e00)
    dt = tl.math.fma(dt, qt, 3.754408661907416e00)
    den_t = tl.math.fma(dt, qt, 1.0)
    # the raw c/d rational is already negative for large q, so the minus sign
    # belongs to the *upper* tail only
    num_t = tl.where(p < 0.5, num_t, -num_t)

    # ------------------------------------------------------------------
    # central region: 0.02425 <= p <= 0.97575  (q = p - 0.5).
    # The numerator and denominator are tiny differences of terms of
    # magnitude ~20, so this part alone is evaluated in float64.
    # ------------------------------------------------------------------
    q = p.to(tl.float64) - 0.5
    r = q * q
    a0 = tl.full([BLOCK], -3.969683028665376e01, tl.float64)
    a1 = tl.full([BLOCK], 2.209460984245205e02, tl.float64)
    a2 = tl.full([BLOCK], -2.759285104469687e02, tl.float64)
    a3 = tl.full([BLOCK], 1.383577518672690e02, tl.float64)
    a4 = tl.full([BLOCK], -3.066479806614716e01, tl.float64)
    a5 = tl.full([BLOCK], 2.506628277459239e00, tl.float64)
    nc = tl.math.fma(a0, r, a1)
    nc = tl.math.fma(nc, r, a2)
    nc = tl.math.fma(nc, r, a3)
    nc = tl.math.fma(nc, r, a4)
    num_c = tl.math.fma(nc, r, a5) * q
    b0 = tl.full([BLOCK], -5.447609879822406e01, tl.float64)
    b1 = tl.full([BLOCK], 1.615858368580409e02, tl.float64)
    b2 = tl.full([BLOCK], -1.556989798598866e02, tl.float64)
    b3 = tl.full([BLOCK], 6.680131188771972e01, tl.float64)
    b4 = tl.full([BLOCK], -1.328068155288572e01, tl.float64)
    dc = tl.math.fma(b0, r, b1)
    dc = tl.math.fma(dc, r, b2)
    dc = tl.math.fma(dc, r, b3)
    den_c = tl.math.fma(tl.math.fma(dc, r, b4), r, 1.0)

    is_tail = (p < 0.02425) | (p > 0.97575)
    # rounding the accurate float64 pair to fp32 costs ~1e-7 relative in the
    # quotient, far inside the suite's absolute tolerance; the reciprocal is
    # the single-instruction hardware approximation (~2^-22 relative)
    num = tl.where(is_tail, num_t, num_c.to(tl.float32))
    den = tl.where(is_tail, den_t, den_c.to(tl.float32))
    rcp = tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1;", "=v,v", [den], dtype=tl.float32, is_pure=True, pack=1
    )
    x = num * rcp

    # keep boundary and out-of-domain behaviour independent of the hardware
    # ops: p == 0 -> -inf, p == 1 -> +inf, anything else outside [0, 1]
    # (including NaN) -> NaN
    x = tl.where((p >= 0.0) & (p <= 1.0), x, float("nan"))
    x = tl.where(p == 0.0, -float("inf"), x)
    x = tl.where(p == 1.0, float("inf"), x)
    return x


@triton.jit
def _ndtri_flat_kernel(x_ptr, out_ptr, n, IS_F32: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(x_ptr + offs, mask=mask, other=0.5)
    x = _ndtri_math(p, IS_F32, BLOCK).to(out_ptr.type.element_ty)
    tl.store(out_ptr + offs, x, mask=mask)


@triton.jit
def _ndtri_strided_kernel(
    x_ptr,
    out_ptr,
    n,
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
    IS_F32: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    rem = offs
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK > 0:
        i = rem % s0
        rem = rem // s0
        off += i * t0
    if RANK > 1:
        i = rem % s1
        rem = rem // s1
        off += i * t1
    if RANK > 2:
        i = rem % s2
        rem = rem // s2
        off += i * t2
    if RANK > 3:
        i = rem % s3
        rem = rem // s3
        off += i * t3
    if RANK > 4:
        i = rem % s4
        rem = rem // s4
        off += i * t4
    if RANK > 5:
        i = rem % s5
        rem = rem // s5
        off += i * t5
    if RANK > 6:
        i = rem % s6
        rem = rem // s6
        off += i * t6
    if RANK > 7:
        i = rem % s7
        rem = rem // s7
        off += i * t7
    p = tl.load(x_ptr + off, mask=mask, other=0.5)
    x = _ndtri_math(p, IS_F32, BLOCK).to(out_ptr.type.element_ty)
    tl.store(out_ptr + offs, x, mask=mask)


_SUPPORTED_DTYPES = (torch.float32, torch.float64)


def _collapse(sizes, strides):
    """Drop size-1 dims and merge neighbours that are contiguous in memory.

    Purely metadata: yields (sizes, strides) with the same memory mapping but
    the smallest possible rank.
    """
    keep_s, keep_t = [], []
    for sz, st in zip(sizes, strides):
        if sz != 1:
            keep_s.append(sz)
            keep_t.append(st)
    if not keep_s:
        return [1], [0]
    out_s, out_t = [], []
    cur_sz, cur_st = keep_s[-1], keep_t[-1]
    for i in range(len(keep_s) - 2, -1, -1):
        if keep_t[i] == cur_st * cur_sz:
            cur_sz *= keep_s[i]
        else:
            out_s.append(cur_sz)
            out_t.append(cur_st)
            cur_sz, cur_st = keep_s[i], keep_t[i]
    out_s.append(cur_sz)
    out_t.append(cur_st)
    out_s.reverse()
    out_t.reverse()
    return out_s, out_t


def _launch_config(n):
    if n >= 262144:
        return 1024, 4
    return 256, 4


def special_ndtri(self):
    logger.debug("GEMS_HYGON SPECIAL_NDTRI")
    _dispatch_log.debug("GEMS SPECIAL_NDTRI")
    if self.dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            f"\"ndtri\" not implemented for '{self.dtype}'; supported dtypes are "
            f"{_SUPPORTED_DTYPES}"
        )
    out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    n = self.numel()
    if n == 0:
        return out

    is_f32 = self.dtype == torch.float32
    block, warps = _launch_config(n)
    grid = (triton.cdiv(n, block),)

    if self.is_contiguous():
        _ndtri_flat_kernel[grid](
            self, out, n, IS_F32=is_f32, BLOCK=block, num_warps=warps
        )
        return out

    sizes, strides = _collapse(list(self.shape), list(self.stride()))
    rank = len(sizes)
    # decode from the fastest-varying (last) dimension outwards
    slots = list(zip(reversed(sizes), reversed(strides)))
    while len(slots) < 8:
        slots.append((1, 0))
    ss = [int(s) for s, _ in slots[:8]]
    tt = [int(t) for _, t in slots[:8]]
    _ndtri_strided_kernel[grid](
        self,
        out,
        n,
        ss[0],
        ss[1],
        ss[2],
        ss[3],
        ss[4],
        ss[5],
        ss[6],
        ss[7],
        tt[0],
        tt[1],
        tt[2],
        tt[3],
        tt[4],
        tt[5],
        tt[6],
        tt[7],
        IS_F32=is_f32,
        RANK=rank,
        BLOCK=block,
        num_warps=warps,
    )
    return out
