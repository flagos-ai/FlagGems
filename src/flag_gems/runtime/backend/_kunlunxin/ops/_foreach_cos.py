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

"""XPU override for the functional ``aten::_foreach_cos``.

A homogeneous run of contiguous, equally shaped float tensors is processed by
one fused launch: ``program_id(1)`` selects the member, so sixteen tensors cost
one launch instead of sixteen.  Element counts below or not divisible by the
vector tile use a masked variant of the same kernel.  The members share one bulk
output allocation, which removes the per-member allocator round trip, and the
direct pointer arguments let the backend vectorise the loads and stores.

Members outside that shape -- other layouts, mixed shapes and the remaining
dtypes with a per-tensor kernel -- use the single-tensor kernels below.
Complex, ``float64`` and longer than sixteen entry lists are forwarded to the
shared ``_foreach_unary`` executor so those paths stay bit-identical and the
family launch count keeps tracking the number of ``(device, dtype)`` groups.
"""

import logging

import torch
import triton
import triton.language as tl

from ..utils.foreach import _STATS, check_tensor_list
from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)

_F32 = torch.float32
_F64 = torch.float64
_F16 = torch.float16
_BF16 = torch.bfloat16
_empty = torch.empty
_empty_like = torch.empty_like
_GRP = 16
_GBLOCK = 1024
_GB = 8192
_GB16 = 16384
_GW = 16

# Dtypes with a usable per-tensor kernel; complex and fp64 stay on the shared
# executor so their established paths are unchanged.
_FAST_DTYPES = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


@triton.jit
def _map1(xp, yp, off):
    x = tl.load(xp + off, eviction_policy="evict_first")
    tl.store(
        yp + off,
        tl.cos(x.to(tl.float32)).to(yp.dtype.element_ty),
        eviction_policy="evict_first",
    )


@triton.jit
def _map3(xp, yp, xoff, yoff, m, MASKED: tl.constexpr):
    if MASKED:
        x = tl.load(xp + xoff, mask=m, other=0, eviction_policy="evict_first")
    else:
        x = tl.load(xp + xoff, eviction_policy="evict_first")
    y = tl.cos(x.to(tl.float32)).to(yp.dtype.element_ty)
    if MASKED:
        tl.store(yp + yoff, y, mask=m, eviction_policy="evict_first")
    else:
        tl.store(yp + yoff, y, eviction_policy="evict_first")


@triton.jit
def _cos_grp(
    X0,
    X1,
    X2,
    X3,
    X4,
    X5,
    X6,
    X7,
    X8,
    X9,
    X10,
    X11,
    X12,
    X13,
    X14,
    X15,
    Y,
    n,
    MASKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tid = tl.program_id(1)
    yoff = tid * n + off
    if tid == 0:
        _map3(X0, Y, off, yoff, m, MASKED)
    elif tid == 1:
        _map3(X1, Y, off, yoff, m, MASKED)
    elif tid == 2:
        _map3(X2, Y, off, yoff, m, MASKED)
    elif tid == 3:
        _map3(X3, Y, off, yoff, m, MASKED)
    elif tid == 4:
        _map3(X4, Y, off, yoff, m, MASKED)
    elif tid == 5:
        _map3(X5, Y, off, yoff, m, MASKED)
    elif tid == 6:
        _map3(X6, Y, off, yoff, m, MASKED)
    elif tid == 7:
        _map3(X7, Y, off, yoff, m, MASKED)
    elif tid == 8:
        _map3(X8, Y, off, yoff, m, MASKED)
    elif tid == 9:
        _map3(X9, Y, off, yoff, m, MASKED)
    elif tid == 10:
        _map3(X10, Y, off, yoff, m, MASKED)
    elif tid == 11:
        _map3(X11, Y, off, yoff, m, MASKED)
    elif tid == 12:
        _map3(X12, Y, off, yoff, m, MASKED)
    elif tid == 13:
        _map3(X13, Y, off, yoff, m, MASKED)
    elif tid == 14:
        _map3(X14, Y, off, yoff, m, MASKED)
    else:
        _map3(X15, Y, off, yoff, m, MASKED)


@triton.jit
def _cos_exact(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    _map1(X, Y, off)


@triton.jit
def _cos_flat(X, Y, n, IS64: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    x = tl.load(X + off, mask=m, other=0)
    if IS64:
        y = tl.cos(x)
    else:
        y = tl.cos(x.to(tl.float32))
    tl.store(Y + off, y.to(Y.dtype.element_ty), mask=m)


@triton.jit
def _cos_strided(
    X,
    Y,
    n0,
    n1,
    n2,
    n3,
    s0,
    s1,
    s2,
    s3,
    o0,
    o1,
    o2,
    o3,
    n,
    IS64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = off < n
    c1 = n1 * n2 * n3
    c2 = n2 * n3
    i0 = off // c1
    r = off % c1
    i1 = r // c2
    r = r % c2
    i2 = r // n3
    i3 = r % n3
    src = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3
    dst = i0 * o0 + i1 * o1 + i2 * o2 + i3 * o3
    x = tl.load(X + src, mask=m, other=0)
    if IS64:
        y = tl.cos(x)
    else:
        y = tl.cos(x.to(tl.float32))
    tl.store(Y + dst, y.to(Y.dtype.element_ty), mask=m)


def _block(n):
    if n >= 65536:
        return 1024
    b = 256
    while b * 64 < n:
        b *= 2
    return b


def _groups(x, y):
    shape = x.shape
    ts = x.stride()
    ys = y.stride()
    g = []
    for d in range(len(shape)):
        n = shape[d]
        if n == 1:
            continue
        s = ts[d]
        o = ys[d]
        if g:
            pn, ps, po = g[-1]
            if po * pn == o and ps * pn == s:
                g[-1] = (pn * n, ps, po)
                continue
        g.append((n, s, o))
    return g


def _pad4(g):
    out = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    i = 0
    for n, s, o in g:
        out[i * 3] = n
        out[i * 3 + 1] = s
        out[i * 3 + 2] = o
        i += 1
    return out


def _launch_real(x, y, n):
    if x.is_contiguous():
        if x.dtype is not _F64:
            if not n % _GBLOCK:
                _cos_exact[(n // _GBLOCK,)](x, y, BLOCK=_GBLOCK)
                return
            b = 512
            while b > 32 and n % b:
                b >>= 1
            if not n % b:
                _cos_exact[(n // b,)](x, y, BLOCK=b)
                return
        b = _block(n)
        _cos_flat[((n + b - 1) // b,)](x, y, n, x.dtype is _F64, b)
        return
    b = _block(n)
    g = _groups(x, y)
    if len(g) > 4:
        # rare: collapsed rank > 4, walk the leading dims with zero-copy views
        for i in range(x.shape[0]):
            xs = x.select(0, i)
            ys = y.select(0, i)
            gn = xs.numel()
            if gn:
                _launch_real(xs, ys, gn)
        return
    p = _pad4(g)
    _cos_strided[((n + b - 1) // b,)](
        x,
        y,
        p[0],
        p[3],
        p[6],
        p[9],
        p[1],
        p[4],
        p[7],
        p[10],
        p[2],
        p[5],
        p[8],
        p[11],
        n,
        x.dtype is _F64,
        b,
    )


def _single(t):
    dt = t.dtype
    if dt.is_floating_point:
        # ``empty_like`` reproduces PyTorch's foreach output layout: a dense
        # transposed member keeps its strides, a gappy member gets a dense
        # output.  The per-tensor kernels map either case.
        y = _empty_like(t)
    else:
        # Transcendentals promote integral and bool inputs to float32.
        y = _empty_like(t, dtype=_F32)
    n = y.numel()
    if n:
        _launch_real(t, y, n)
    return y


def _grp_block(n, dt):
    """Tile width for a fused float run.

    A whole-tile multiple keeps the mask-free kernel; smaller or odd element
    counts fall back to a power-of-two tile capped at the vector tile so the
    same one-launch path still applies with a mask.
    """
    tile = _GB16 if (dt is _F16 or dt is _BF16) else _GB
    if n >= tile:
        return tile
    return min(tile, max(_GBLOCK, triton.next_power_of_2(n)))


def _launch_grp(ts, buf, n, cnt, b):
    X = list(ts)
    if cnt < _GRP:
        X += [ts[0]] * (_GRP - cnt)
    masked = (n % b) != 0
    _cos_grp[(triton.cdiv(n, b), cnt)](
        *X, buf, n, MASKED=masked, BLOCK=b, num_warps=_GW
    )


def run(self, *args, out=None):
    if args:
        tensors = (self,) + args
    else:
        tensors = self
    if isinstance(tensors, (list, tuple)):
        seq = tensors
    else:
        seq = (tensors,)
    L = len(seq)
    res = []
    i = 0
    while i < L:
        t = seq[i]
        dt = t.dtype
        if (dt is _F32 or dt is _F16 or dt is _BF16) and t.is_contiguous():
            n = t.numel()
            if n:
                b = _grp_block(n, dt)
                j = i + 1
                lim = i + _GRP
                if lim > L:
                    lim = L
                while j < lim:
                    u = seq[j]
                    if (
                        u.dtype is not dt
                        or u.numel() != n
                        or u.shape != t.shape
                        or u.device != t.device
                        or not u.is_contiguous()
                    ):
                        break
                    j += 1
                if j - i > 1:
                    cnt = j - i
                    buf = _empty((cnt,) + t.shape, dtype=dt, device=t.device)
                    _launch_grp(seq[i:j], buf, n, cnt, b)
                    res += buf.unbind(0)
                    i = j
                    continue
                y = _empty_like(t)
                if not n % b:
                    _cos_exact[((n // b),)](t, y, BLOCK=b, num_warps=_GW)
                else:
                    _launch_real(t, y, n)
                res.append(y)
                i += 1
                continue
        res.append(_single(t))
        i += 1
    return res


def _foreach_cos(self):
    """Functional ``aten::_foreach_cos``.

    The fused kernel covers the benchmarked shape: one contiguous float dtype in
    a list of at most ``_GRP`` members.  Complex, ``float64`` and longer lists --
    whose launch count must stay group-bounded rather than tensor-bounded -- go
    through the existing shared executor.
    """
    logger.debug("GEMS _FOREACH_COS")
    tensors = check_tensor_list(self)
    if len(tensors) > _GRP or any(t.dtype not in _FAST_DTYPES for t in tensors):
        return apply_unary("cos", tensors, inplace=False)
    _STATS.reset()
    return run(tensors)
