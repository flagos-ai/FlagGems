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
from triton.language.extra import libdevice

from flag_gems.ops.igamma import igamma as default_igamma
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_BLOCK = 256
_BLOCK_BIG = 512
_MAXIT = 800
_NW = 4
_NW_BIG = 4


@triton.jit
def _log_pref_dev(aa, xx):
    # log(x^a exp(-x) / Gamma(a)) in Stirling form: a*log1p((x-a)/a) and
    # (x-a) stay O(1), so large a keeps its accuracy in fp32.
    ra = 1.0 / aa
    u = (xx - aa) * ra
    return (
        aa * libdevice.log1p(u)
        - (xx - aa)
        + 0.5 * tl.log(aa)
        - 0.9189385332046727
        - ra * 0.08333333333333333
        + ra * ra * ra * 0.002777777777777778
    )


@triton.jit
def _log_pref_naive(aa, xx):
    return aa * tl.log(xx) - xx - libdevice.lgamma(aa)


@triton.jit
def _igamma_core(a_raw, x_raw, EPS: tl.constexpr, MAXIT: tl.constexpr):
    a = a_raw.to(tl.float32)
    x = x_raw.to(tl.float32)

    zero = tl.zeros_like(a)
    one = zero + 1.0
    nanv = zero / zero

    a_zero = a == 0.0
    x_zero = x == 0.0
    a_inf = a == float("inf")
    x_inf = x == float("inf")
    neg = (a < 0.0) | (x < 0.0)
    comp = ~(neg | a_zero | x_zero | a_inf | x_inf)

    aa = tl.where(comp, a, one)
    xx = tl.where(comp, x, one)

    # Choose the prefactor form per block so a uniform block evaluates only
    # the chain it needs; mixed blocks keep the elementwise select.
    if tl.min(aa, axis=0) > 8.0:
        logp = _log_pref_dev(aa, xx)
    elif tl.max(aa, axis=0) <= 8.0:
        logp = _log_pref_naive(aa, xx)
    else:
        logp = tl.where(aa > 8.0, _log_pref_dev(aa, xx), _log_pref_naive(aa, xx))
    pref = tl.exp(logp)
    pref = tl.where(logp < -88.72283905206835, 0.0, pref)

    ser = comp & (xx < aa + 1.0)
    cfm = comp & ~ser

    val = zero

    # lower branch: series expansion for x < a+1. The recurrence runs
    # unmasked (all terms are positive, so the ATen break test is c > eps*sum)
    # and each lane's value is captured once, when it first converges.
    if tl.max(ser.to(tl.int32), axis=0) > 0:
        ap = aa
        term = one / aa
        sm = term
        res = zero
        act = ser & (pref != 0.0)
        it = 0
        while (tl.max(act.to(tl.int32), axis=0) > 0) & (it < MAXIT):
            ap = ap + 1.0
            term = term * xx / ap
            sm = sm + term
            conv = act & ~(term > EPS * sm)
            res = tl.where(conv, sm, res)
            act = act & ~conv
            it += 1
        res = tl.where(act, sm, res)
        val = tl.where(ser, res * pref, val)

    # upper branch: modified Lentz continued fraction for x >= a+1
    if tl.max(cfm.to(tl.int32), axis=0) > 0:
        b = xx + 1.0 - aa
        cc = one / 1e-30
        d = one / b
        h = d
        res = zero
        act = cfm & (pref != 0.0)
        it = 0
        while (tl.max(act.to(tl.int32), axis=0) > 0) & (it < MAXIT):
            it += 1
            nf = zero + it
            an = -nf * (nf - aa)
            b = b + 2.0
            d = an * d + b
            d = tl.where(tl.abs(d) < 1e-30, 1e-30, d)
            cc = b + an / cc
            cc = tl.where(tl.abs(cc) < 1e-30, 1e-30, cc)
            d = one / d
            dl = d * cc
            h = h * dl
            conv = act & ~(tl.abs(dl - 1.0) > EPS)
            res = tl.where(conv, h, res)
            act = act & ~conv
        res = tl.where(act, h, res)
        val = tl.where(cfm, 1.0 - res * pref, val)

    out = val
    out = tl.where(x_inf, one, out)
    out = tl.where(a_inf & x_inf, nanv, out)
    out = tl.where(a_inf & ~x_inf, zero, out)
    out = tl.where(x_zero, zero, out)
    out = tl.where(a_zero & (x > 0.0), one, out)
    out = tl.where(a_zero & ~(x > 0.0), nanv, out)
    out = tl.where(neg, nanv, out)
    return out


@triton.jit
def _igamma_flat(
    a_ptr, x_ptr, o_ptr, n, EPS: tl.constexpr, BLOCK: tl.constexpr, MAXIT: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(a_ptr + offs, mask=m, other=1.0)
    x = tl.load(x_ptr + offs, mask=m, other=1.0)
    r = _igamma_core(a, x, EPS, MAXIT)
    tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _igamma_kernel(
    a_ptr,
    x_ptr,
    o_ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    A0,
    A1,
    A2,
    A3,
    A4,
    A5,
    A6,
    A7,
    X0,
    X1,
    X2,
    X3,
    X4,
    X5,
    X6,
    X7,
    NDIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    MAXIT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    rem = offs
    ia = offs * 0
    ix = offs * 0
    if NDIM > 7:
        idx = rem % s7
        rem = rem // s7
        ia += idx * A7
        ix += idx * X7
    if NDIM > 6:
        idx = rem % s6
        rem = rem // s6
        ia += idx * A6
        ix += idx * X6
    if NDIM > 5:
        idx = rem % s5
        rem = rem // s5
        ia += idx * A5
        ix += idx * X5
    if NDIM > 4:
        idx = rem % s4
        rem = rem // s4
        ia += idx * A4
        ix += idx * X4
    if NDIM > 3:
        idx = rem % s3
        rem = rem // s3
        ia += idx * A3
        ix += idx * X3
    if NDIM > 2:
        idx = rem % s2
        rem = rem // s2
        ia += idx * A2
        ix += idx * X2
    if NDIM > 1:
        idx = rem % s1
        rem = rem // s1
        ia += idx * A1
        ix += idx * X1
    if NDIM > 0:
        ia += rem * A0
        ix += rem * X0

    a = tl.load(a_ptr + ia, mask=m, other=1.0)
    x = tl.load(x_ptr + ix, mask=m, other=1.0)
    r = _igamma_core(a, x, EPS, MAXIT)
    tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty), mask=m)


def _broadcast_strides(t, shape, rank):
    pad = rank - t.dim()
    st = []
    for i in range(rank):
        j = i - pad
        if j < 0:
            st.append(0)
        elif t.shape[j] == 1 and shape[i] != 1:
            st.append(0)
        else:
            st.append(t.stride(j))
    return st


def _fold(shape, sa, sx):
    osh, oa, ox = [], [], []
    for i in range(len(shape)):
        if osh and oa[-1] == sa[i] * shape[i] and ox[-1] == sx[i] * shape[i]:
            osh[-1] *= shape[i]
            oa[-1] = sa[i]
            ox[-1] = sx[i]
        else:
            osh.append(shape[i])
            oa.append(sa[i])
            ox.append(sx[i])
    return osh, oa, ox


def _use_triton_kernel(a, x):
    """Whether the pair can be handled by the MUSA kernel."""
    if not isinstance(a, torch.Tensor) or not isinstance(x, torch.Tensor):
        return False
    if a.device.type != "musa" or x.device.type != "musa":
        return False
    if a.dtype not in _SUPPORTED_DTYPES or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if a.is_complex() or x.is_complex():
        return False
    return True


def igamma(self, other):
    logger.debug("GEMS_MTHREADS IGAMMA")

    if not _use_triton_kernel(self, other):
        return default_igamma(self, other)

    with torch_device_fn.device(self.device):
        return _launch_igamma(self, other)


def _launch_igamma(self, other):
    a, x = self, other
    if a.shape == x.shape:
        shape = tuple(a.shape)
    else:
        shape = torch.broadcast_shapes(tuple(a.shape), tuple(x.shape))
    dt = a.dtype if a.dtype == x.dtype else torch.promote_types(a.dtype, x.dtype)
    if not dt.is_floating_point:
        dt = torch.float32
    out = torch.empty(shape, dtype=dt, device=a.device)
    n = out.numel()
    if n == 0:
        return out

    eps = 1.1920928955078125e-07 if dt != torch.float64 else 2.220446049250313e-16
    if n >= (1 << 20):
        block, nw = _BLOCK_BIG, _NW_BIG
    else:
        block, nw = _BLOCK, _NW
    if (
        a.shape == shape
        and x.shape == shape
        and a.is_contiguous()
        and x.is_contiguous()
    ):
        _igamma_flat[(triton.cdiv(n, block),)](
            a, x, out, n, EPS=eps, BLOCK=block, MAXIT=_MAXIT, num_warps=nw
        )
        return out

    rk = len(shape)
    sa = _broadcast_strides(a, shape, rk)
    sx = _broadcast_strides(x, shape, rk)
    fsh, f_sa, f_sx = _fold(list(shape), sa, sx)
    rank = len(fsh)
    if rank > 8:
        raise RuntimeError("igamma: unsupported broadcast rank > 8")
    pad = 8 - rank
    fs = tuple(fsh + [1] * pad)
    fa = tuple(f_sa + [0] * pad)
    fx = tuple(f_sx + [0] * pad)

    _igamma_kernel[(triton.cdiv(n, block),)](
        a,
        x,
        out,
        n,
        fs[0],
        fs[1],
        fs[2],
        fs[3],
        fs[4],
        fs[5],
        fs[6],
        fs[7],
        fa[0],
        fa[1],
        fa[2],
        fa[3],
        fa[4],
        fa[5],
        fa[6],
        fa[7],
        fx[0],
        fx[1],
        fx[2],
        fx[3],
        fx[4],
        fx[5],
        fx[6],
        fx[7],
        NDIM=rank,
        EPS=eps,
        BLOCK=block,
        MAXIT=_MAXIT,
        num_warps=nw,
    )
    return out
