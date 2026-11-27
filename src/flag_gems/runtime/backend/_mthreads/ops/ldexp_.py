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
"""In-place ldexp (aten::ldexp_) implemented with Triton.

self <- ldexp(self, other) with full broadcasting, arbitrary strides and
bit-exact agreement with the target reference semantics.

The target reference computes the scale as ``self * exp2(other)`` with the
exp2 result rounded to the storage dtype first, so 0 * inf -> NaN for huge
exponents, inf * 0 -> NaN for very negative ones and deep-subnormal products
are flushed the same way.  The kernel performs exactly that computation, so
it agrees with the reference bitwise over the whole exponent range.

The original Tensor object/storage is written in place and returned.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flag_gems.ops.ldexp_ import ldexp_ as default__ldexp_
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)
BLOCK = 2048
WARPS = 8

_E_OK = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
for _n in ("uint16", "uint32", "uint64"):
    _dt = getattr(torch, _n, None)
    if _dt is not None:
        _E_OK.add(_dt)
# ATen's ldexp is ``self * exp2(other)`` for any numeric ``other``: a floating
# exponent keeps its fractional part rather than being rejected or truncated
# (``ldexp_(1.0, 1.5) == 2.8284``).  The C library's integer-only ``ldexp``
# signature does not carry over to the tensor overload.
_E_OK |= {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_S_OK = {torch.float16, torch.bfloat16, torch.float32, torch.float64}

PZERO = 0.0


# ---------------------------------------------------------------- kernels
@triton.jit
def _scale(x, e, pzero):
    """Reference-equivalent ldexp: x * exp2(n) with the scale rounded to the
    storage dtype first (that rounding is what produces the target's
    0*inf -> NaN / inf*0 -> NaN behaviour at saturated exponents).

    pzero is a runtime 0.0 whose only job is to hide the integer-valued
    exponent from LLVM's ``fmul x, exp2(int-valued y) -> llvm.ldexp`` combine,
    whose vector form the mthreads linker cannot relocate.

    A floating exponent goes to fp32 directly: routing it through ``int32``
    like an integral one would truncate the fractional part and compute
    ``exp2(1)`` for ``1.5``.  The branch is on a trace-time dtype, so each
    specialization still emits a single conversion.
    """
    if e.dtype.is_floating():
        t = libdevice.exp2(e.to(tl.float32) + pzero)
    else:
        # Clamp before narrowing: a bare ``to(int32)`` wraps, so an
        # exponent past the int32 range (e.g. 2**32) folds back to 0 and
        # scales by 1.0 instead of saturating to inf / 0.
        ec = tl.minimum(tl.maximum(e, -2147483648), 2147483647)
        t = libdevice.exp2(ec.to(tl.int32).to(tl.float32) + pzero)
    return x * t.to(x.dtype)


@triton.jit
def _ldexp_c(x_ptr, e_ptr, n, e_s, pzero, BLOCK: tl.constexpr, E_MODE: tl.constexpr):
    """self contiguous (stride 1). E_MODE: 0 scalar exp, 1 contiguous exp, 2 strided exp."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    if E_MODE == 0:
        e = tl.load(e_ptr)
    elif E_MODE == 1:
        e = tl.load(e_ptr + offs, mask=mask)
    else:
        e = tl.load(e_ptr + offs * e_s, mask=mask)
    tl.store(x_ptr + offs, _scale(x, e, pzero), mask=mask)


@triton.jit
def _ldexp_s(
    x_ptr, e_ptr, n, x_s, e_s, pzero, BLOCK: tl.constexpr, E_MODE: tl.constexpr
):
    """self strided along a single collapsed dimension."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    xo = offs * x_s
    x = tl.load(x_ptr + xo, mask=mask)
    if E_MODE == 0:
        e = tl.load(e_ptr)
    else:
        e = tl.load(e_ptr + offs * e_s, mask=mask)
    tl.store(x_ptr + xo, _scale(x, e, pzero), mask=mask)


@triton.jit
def _ldexp_nd(
    x_ptr,
    e_ptr,
    nblk,
    pzero,
    d0,
    xs0,
    es0,
    d1,
    xs1,
    es1,
    d2,
    xs2,
    es2,
    d3,
    xs3,
    es3,
    d4,
    xs4,
    es4,
    d5,
    xs5,
    es5,
    d6,
    xs6,
    es6,
    d7,
    xs7,
    es7,
    BLOCK: tl.constexpr,
    ND: tl.constexpr,
    E0: tl.constexpr,
):
    """Multi-dimensional collapsed traversal.

    Grid is flattened as row * nblk + blk so there is no grid-dim limit.
    Only scalar arithmetic is spent decomposing the row index; the fastest
    collapsed dimension is walked contiguously by the whole block.
    """
    pid = tl.program_id(0)
    row = pid // nblk
    blk = pid - row * nblk
    rem = row
    bx = 0
    be = 0
    if ND > 1:
        i = rem % d1
        rem = rem // d1
        bx += i * xs1
        be += i * es1
    if ND > 2:
        i = rem % d2
        rem = rem // d2
        bx += i * xs2
        be += i * es2
    if ND > 3:
        i = rem % d3
        rem = rem // d3
        bx += i * xs3
        be += i * es3
    if ND > 4:
        i = rem % d4
        rem = rem // d4
        bx += i * xs4
        be += i * es4
    if ND > 5:
        i = rem % d5
        rem = rem // d5
        bx += i * xs5
        be += i * es5
    if ND > 6:
        i = rem % d6
        rem = rem // d6
        bx += i * xs6
        be += i * es6
    if ND > 7:
        i = rem % d7
        rem = rem // d7
        bx += i * xs7
        be += i * es7
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < d0
    xo = bx + offs * xs0
    x = tl.load(x_ptr + xo, mask=mask)
    if E0:
        e = tl.load(e_ptr + be)
    else:
        e = tl.load(e_ptr + be + offs * es0, mask=mask)
    tl.store(x_ptr + xo, _scale(x, e, pzero), mask=mask)


# ---------------------------------------------------------------- wrapper
def _bad_exponent(odt):
    name = str(odt).replace("torch.", "")
    raise RuntimeError(
        "ldexp: Exponent tensor must be of integral type, but got: " + name
    )


def _bad_self(sdt):
    raise RuntimeError(
        "ldexp_: result type Float can't be cast to the desired output type "
        + str(sdt).replace("torch.", "")
    )


def _bad_broadcast(sshape, oshape):
    raise RuntimeError(
        "output with shape ["
        + ", ".join(str(v) for v in sshape)
        + "] doesn't match the broadcast shape ["
        + ", ".join(str(v) for v in _broadcast(sshape, oshape))
        + "]"
    )


def _broadcast(a, b):
    nd = max(len(a), len(b))
    a = (1,) * (nd - len(a)) + tuple(a)
    b = (1,) * (nd - len(b)) + tuple(b)
    out = []
    for x, y in zip(a, b):
        if x == 1:
            out.append(y)
        elif y == 1 or y == x:
            out.append(x)
        else:
            out.append(-1)
    return out


def _check(odt, sdt):
    if odt not in _E_OK:
        _bad_exponent(odt)
    _bad_self(sdt)


def ldexp_(self, other):
    logger.debug("GEMS_MTHREADS LDEXP_")
    if not isinstance(self, torch.Tensor) or self.device.type != "musa":
        return default__ldexp_(self, other)
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, device=self.device)
    odt = other.dtype
    sdt = self.dtype
    if odt not in _E_OK or sdt not in _S_OK:
        _check(odt, sdt)

    sshape = self.shape
    oshape = other.shape
    nd = len(sshape)
    ond = len(oshape)
    if ond > nd:
        _bad_broadcast(sshape, oshape)
    if oshape != sshape:
        off = nd - ond
        for i in range(ond):
            o = oshape[i]
            if o != 1 and o != sshape[off + i]:
                _bad_broadcast(sshape, oshape)

    n = self.numel()
    if n == 0:
        return self

    if has_internal_overlapping(self) == MemOverlap.Yes:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to "
            "tensor refers to a single memory location. Please clone() the "
            "tensor before performing the operation."
        )

    pzero = PZERO

    # ---- fast path: contiguous self
    if self.is_contiguous():
        if other.numel() == n:
            if other.is_contiguous():
                _ldexp_c[((n + BLOCK - 1) // BLOCK,)](
                    self, other, n, 1, pzero, BLOCK=BLOCK, E_MODE=1, num_warps=WARPS
                )
                return self
        elif other.numel() == 1:
            _ldexp_c[((n + BLOCK - 1) // BLOCK,)](
                self, other, n, 1, pzero, BLOCK=BLOCK, E_MODE=0, num_warps=WARPS
            )
            return self

    # ---- general path: collapse (shape, self_stride, other_stride) dims
    ss = self.stride()
    os_ = other.stride()
    off = nd - ond
    dims = []
    for d in range(nd):
        sz = sshape[d]
        if sz > 1:
            if d >= off:
                ostr = os_[d - off] if oshape[d - off] != 1 else 0
            else:
                ostr = 0
            dims.append((sz, ss[d], ostr))
    dims.sort(key=lambda t: abs(t[1]))

    merged = []
    for sz, xs, es in dims:
        if merged:
            psz, pxs, pes = merged[-1]
            if xs == psz * pxs and es == psz * pes:
                merged[-1] = (psz * sz, pxs, pes)
                continue
        merged.append((sz, xs, es))

    nm = len(merged)
    if nm == 0:
        _ldexp_c[(1,)](self, other, 1, 1, pzero, BLOCK=32, E_MODE=0, num_warps=1)
        return self

    if nm == 1:
        sz, xs, es = merged[0]
        if xs == 1:
            mode = 0 if es == 0 else 1
            _ldexp_c[((n + BLOCK - 1) // BLOCK,)](
                self, other, n, 1, pzero, BLOCK=BLOCK, E_MODE=mode, num_warps=WARPS
            )
        else:
            mode = 0 if es == 0 else 2
            _ldexp_s[((n + BLOCK - 1) // BLOCK,)](
                self, other, n, xs, es, pzero, BLOCK=BLOCK, E_MODE=mode, num_warps=WARPS
            )
        return self

    if nm > 8:
        raise RuntimeError("ldexp_: too many non-collapsible dimensions")

    d0, xs0, es0 = merged[0]
    outer = 1
    for t in merged[1:]:
        outer *= t[0]
    nblk = (d0 + BLOCK - 1) // BLOCK
    args = [self, other, nblk, pzero, d0, xs0, es0]
    for k in range(1, 8):
        if k < nm:
            sz, xs, es = merged[k]
            args.append(sz)
            args.append(xs)
            args.append(es)
        else:
            args.append(1)
            args.append(0)
            args.append(0)
    _ldexp_nd[(nblk * outer,)](
        *args, BLOCK=BLOCK, ND=nm, E0=(es0 == 0), num_warps=WARPS
    )
    return self
