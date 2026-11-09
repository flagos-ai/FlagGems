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

"""Triton implementation of aten::avg_pool1d.

Semantics mirror ATen's avg_pool1d (avg_pool2d on an (N, C, 1, L) view with
kernel (1, K), stride (1, S), padding (0, P)).

Output element f in [0, R*L_out) belongs to row r = f // L_out and position
p = f - r*L_out, with

    q0 = p*S - P ; window = [q0, q0+K)
    sum = sum_k (0 <= q0+k < L) ? x[row_base + (q0+k)*in_sl] : 0
    div = count_include_pad ? min(K, L+P-q0) : min(q0+K, L) - max(q0, 0)
    out[f] = cast(sum / div)

Two kernels:

* `_avg_pool1d_pair_kernel` handles the dominant (K=3, S=2, P=1) contiguous-row
  case, where L == 2*L_out makes the flat input address of output f simply
  2*f. Every thread owns two neighbouring outputs and loads the aligned
  4-element slab x[4g .. 4g+3] (one 16 B vector load, 100% sector use) plus the
  single leading element x[4g-1], then stores the two results as one 8 B
  vector. Only the leading element can be padding, so one comparison drives
  the whole mask.

* `_avg_pool1d_kernel` is the general flat kernel (any K/S/P, strides, ranks).
  All host-provable simplifications are constexpr there too: which k can leave
  [0, L) (only the first P below zero, only k >= L-q0max above L), whether the
  tile is entirely in range (EXACT), and whether the divisor is the constant K
  for every window (CONST_DIV).

`run` keeps a single-entry plan cache keyed on the full call signature so the
per-call path is just an output allocation plus the kernel launch.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


_SM = {}
_PLAN = [None, None]


@triton.jit
def _avg_pool1d_pair_kernel(
    x_ptr,
    out_ptr,
    ngroups,
    pmask,
    INV_K: tl.constexpr,
    ACC64: tl.constexpr,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        g = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        g = pid * BLOCK + tl.arange(0, BLOCK)

    slab = 4 * g
    if ACC64:
        a = tl.load(x_ptr + slab[:, None] + tl.arange(0, 4)[None, :]).to(tl.float64)
        lead = tl.load(x_ptr + slab - 1, mask=((2 * g) & pmask) != 0, other=0.0).to(
            tl.float64
        )
    else:
        a = tl.load(x_ptr + slab[:, None] + tl.arange(0, 4)[None, :]).to(tl.float32)
        lead = tl.load(x_ptr + slab - 1, mask=((2 * g) & pmask) != 0, other=0.0).to(
            tl.float32
        )

    ev, od = tl.split(tl.reshape(a, (BLOCK, 2, 2)))
    e0, e1 = tl.split(ev)
    o0, o1 = tl.split(od)

    r0 = (lead + e0 + o0) * INV_K
    r1 = (o0 + e1 + o1) * INV_K
    res = tl.join(r0, r1)

    tl.store(
        out_ptr + 2 * g[:, None] + tl.arange(0, 2)[None, :],
        res.to(out_ptr.dtype.element_ty),
    )


@triton.jit
def _avg_pool1d_kernel(
    x_ptr,
    out_ptr,
    total,
    L_out,
    L,
    step,
    ks,
    noff,
    row_stride,
    C,
    s0,
    s1,
    pmask,
    K: tl.constexpr,
    S: tl.constexpr,
    P: tl.constexpr,
    CIP: tl.constexpr,
    FLAT: tl.constexpr,
    NONAFFINE: tl.constexpr,
    POW2: tl.constexpr,
    EXACT: tl.constexpr,
    LO_N: tl.constexpr,
    HI_FROM: tl.constexpr,
    CONST_DIV: tl.constexpr,
    ACC64: tl.constexpr,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        f = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        f = pid * BLOCK + tl.arange(0, BLOCK)

    if EXACT:
        fmask = None
    else:
        fmask = f < total

    if POW2:
        p = f & pmask
    else:
        p = f % L_out

    if FLAT:
        base = f * step + noff
    elif NONAFFINE:
        r = f // L_out
        n = r // C
        c = r - n * C
        base = n * s0 + c * s1 + p * step + noff
    else:
        r = f // L_out
        base = r * row_stride + p * step + noff

    if (LO_N > 0) or (HI_FROM < K):
        q0 = p * S - P
    else:
        q0 = p

    if ACC64:
        acc = tl.zeros((BLOCK,), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for k in tl.static_range(K):
        if k < LO_N:
            if k >= HI_FROM:
                m = (q0 + k >= 0) & (q0 + k < L)
            else:
                m = q0 + k >= 0
            if not EXACT:
                m = m & fmask
            v = tl.load(x_ptr + base + k * ks, mask=m, other=0.0)
        else:
            if k >= HI_FROM:
                m = q0 + k < L
                if not EXACT:
                    m = m & fmask
                v = tl.load(x_ptr + base + k * ks, mask=m, other=0.0)
            else:
                if EXACT:
                    v = tl.load(x_ptr + base + k * ks)
                else:
                    v = tl.load(x_ptr + base + k * ks, mask=fmask, other=0.0)
        if ACC64:
            acc += v.to(tl.float64)
        else:
            acc += v.to(tl.float32)

    if CONST_DIV:
        res = acc * (1.0 / K)
    else:
        if CIP:
            div = tl.minimum(K, L + P - q0)
        else:
            div = tl.minimum(q0 + K, L) - tl.maximum(q0, 0)
        if ACC64:
            res = acc / div.to(tl.float64)
        else:
            res = acc / div.to(tl.float32)

    if EXACT:
        tl.store(out_ptr + f, res.to(out_ptr.dtype.element_ty))
    else:
        tl.store(out_ptr + f, res.to(out_ptr.dtype.element_ty), mask=fmask)


def _to_int(value, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        return int(value[0])
    return int(value)


def avg_pool1d(
    self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True
):
    logger.debug("GEMS_HYGON AVG_POOL1D")
    x = self
    key = (
        x.shape,
        x.stride(),
        x.dtype,
        x.get_device(),
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
    )
    if key == _PLAN[0]:
        return _PLAN[1](x)

    K = _to_int(kernel_size)
    P = _to_int(padding, 0)
    S = _to_int(stride, None)
    if S is None:
        S = K
    cip = bool(count_include_pad)

    if K <= 0:
        raise RuntimeError("avg_pool1d: kernel_size must be greater than zero.")
    if S <= 0:
        raise RuntimeError("avg_pool1d: stride must be greater than zero.")
    if 2 * P > K:
        raise RuntimeError(
            "avg_pool1d: pad should be smaller than or equal to half of kernel size"
        )

    nd = x.dim()
    if nd not in (2, 3):
        raise RuntimeError(
            "avg_pool1d: Expected 2D or 3D (batch mode) tensor with optional 0 dim "
            "batch size for input, but got: %dD" % nd
        )
    if not x.is_floating_point():
        raise RuntimeError("avg_pool1d: Expected input to be of floating point type")

    shp = x.shape
    L = int(shp[-1])
    numel = int(x.numel())
    R = numel // L if L > 0 else 0

    n = L + 2 * P - K
    L_out = n // S + 1
    if ceil_mode and (L_out - 1) * S >= L + P:
        L_out -= 1

    out_shape = tuple(int(v) for v in shp[:-1]) + (L_out,)
    if L_out <= 0 or R == 0:
        return torch.empty(out_shape, dtype=x.dtype, device=x.device)

    if nd == 2:
        C = 1
        s0 = x.stride(0)
        s1 = 0
        nonaffine = False
        row_stride = s0
    else:
        C = int(shp[-2])
        s0 = x.stride(-3)
        s1 = x.stride(-2)
        if C == 1:
            nonaffine = False
            row_stride = s0
        elif s0 == C * s1:
            nonaffine = False
            row_stride = s1
        else:
            nonaffine = True
            row_stride = 0

    in_sl = x.stride(-1)
    flat = (not nonaffine) and row_stride == L * in_sl and L == S * L_out

    step = S * in_sl
    noff = -P * in_sl

    i64 = (numel > 2147483647) or (row_stride != 0 and abs(row_stride) * R > 2147483647)

    total = R * L_out
    dev = x.get_device()
    nsm = _SM.get(dev)
    if nsm is None:
        try:
            nsm = torch.cuda.get_device_properties(dev).multi_processor_count
        except Exception:
            nsm = 80
        _SM[dev] = nsm

    # Exact ATen window extents: which k need a bound check, and whether the
    # divisor is the constant K for every window.
    q0max = (L_out - 1) * S - P
    lo_n = P if P < K else K
    hi_from = L - q0max
    if hi_from < 0:
        hi_from = 0
    elif hi_from > K:
        hi_from = K
    if cip:
        const_div = (L + P - q0max) >= K
    else:
        const_div = (P == 0) and (q0max + K <= L)

    acc64 = x.dtype == torch.float64
    pow2 = (L_out & (L_out - 1)) == 0
    out_shape = out_shape
    dtype = x.dtype
    device = x.device

    # Vectorised pair path: two neighbouring outputs per thread, one aligned
    # 4-element slab load plus the single leading element.
    if (
        K == 3
        and S == 2
        and P == 1
        and flat
        and in_sl == 1
        and pow2
        and (L_out % 2) == 0
        and const_div
        and hi_from >= K
        and lo_n == 1
    ):
        tg = total >> 1
        gp = tg // (8 * nsm)
        if gp < 32:
            gp = 32
        elif gp > 1024:
            gp = 1024
        else:
            gp = 1 << (gp - 1).bit_length()
        if tg % gp == 0:
            args = (tg, L_out - 1, 1.0 / K, acc64, i64, gp)

            def launch(x, _g=(tg // gp,), _a=args, _s=out_shape, _d=dtype, _v=device):
                out = torch.empty(_s, dtype=_d, device=_v)
                _avg_pool1d_pair_kernel[_g](x, out, *_a, num_warps=4)
                return out

            _PLAN[0] = key
            _PLAN[1] = launch
            return launch(x)

    b = total // (8 * nsm)
    if b < 128:
        b = 128
    elif b > 512:
        b = 512
    else:
        b = 1 << (b - 1).bit_length()
    nw = 2 if b <= 256 else 4
    exact = (total % b) == 0
    args = (
        total,
        L_out,
        L,
        step,
        in_sl,
        noff,
        row_stride,
        C,
        s0,
        s1,
        L_out - 1,
        K,
        S,
        P,
        cip,
        flat,
        nonaffine,
        pow2,
        exact,
        lo_n,
        hi_from,
        const_div,
        acc64,
        i64,
        b,
    )

    def launch(
        x,
        _g=(triton.cdiv(total, b),),
        _a=args,
        _s=out_shape,
        _d=dtype,
        _v=device,
        _w=nw,
    ):
        out = torch.empty(_s, dtype=_d, device=_v)
        _avg_pool1d_kernel[_g](x, out, *_a, num_warps=_w)
        return out

    _PLAN[0] = key
    _PLAN[1] = launch
    return launch(x)
