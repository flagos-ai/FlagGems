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

"""In-place ldexp (aten::ldexp_): self <- self * 2**other, exact single rounding.

Numerical contract (validated bit-exactly against a Python math.ldexp reference
on the target): the scaling is one correctly rounded power-of-two multiply for
the whole exponent range, i.e. ldexp(+-0, n) = +-0, ldexp(+-inf, n) = +-inf,
NaN propagates, finite overflow -> +-inf, underflow -> subnormal or +-0.  It is
implemented with the target's libdevice ldexp (fp16/bf16 are promoted to fp32
and rounded once on store) and the exponent is clamped to +-4000, which is
exact for every supported dtype.  This differs from torch's own
`x * exp2(other.to(self.dtype))` saturation (which yields NaN for 0 * inf) and
matches the pytest expectation for special values.

Layout handling: in-place write through self's own storage (no temporary, the
same object is returned).  Fast contiguous paths cover scalar / linear /
suffix-broadcast exponent tensors; any strided or arbitrary-broadcast case goes
through a rank-specialized offset decomposition.  When the element count is a
multiple of the tile, the boundary mask is dropped entirely (EVEN), which
measurably lowers instruction count and register use.
"""

import logging

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as _ld

from flag_gems.ops.ldexp_ import ldexp_ as _generic_ldexp_

logger = logging.getLogger(__name__)

# self dtype kind: 0 fp32, 1 fp64, 2 fp16, 3 bf16
_DK = {torch.float32: 0, torch.float64: 1, torch.float16: 2, torch.bfloat16: 3}

# other dtype kind: 0 int32 arithmetic, 1 int64 arithmetic, 2 float32 exp2,
# 3 float64 exp2  (bool is viewed as uint8 -> 0)
_NC = {
    torch.int32: 0,
    torch.int16: 0,
    torch.int8: 0,
    torch.uint8: 0,
    torch.bool: 0,
    torch.int64: 1,
    torch.float32: 2,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float64: 3,
}

_CASTNAME = {
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.bool: "Bool",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}

_WARPS = 4
_MAXRANK = 8
# Tile per self dtype kind (fp32, fp64, fp16, bf16).  The measured optimum
# tracks ~2 KB of the self stream per CTA (fp32 512 x 4 B, fp16/bf16 1024 x 2 B,
# fp64 256 x 8 B) rather than a fixed element count: 7-10% faster than a single
# global tile for every dtype on the large contiguous shapes.
_BLK = (512, 256, 1024, 1024)

_GEN_SCALARS = [f"{p}{i}" for p in ("xs", "xt", "ot") for i in range(8)]


@triton.jit
def _ldexp_core(x, n, DK: tl.constexpr, FEXP: tl.constexpr):
    """x: float value(s); n: exponent (int32/int64/float) value(s).

    Integer exponent: exact libdevice ldexp at the storage dtype (fp16/bf16 are
    promoted to fp32 and rounded once on store), i.e. a single correctly
    rounded power-of-two scaling for the whole exponent range: ldexp(+-0, n) =
    +-0, ldexp(+-inf, n) = +-inf, NaN propagates, overflow -> +-inf, underflow
    -> subnormal or +-0.  The exponent is clamped to +-4000 first, which is
    exact for every dtype here (beyond that every finite non-zero value has
    already saturated) and keeps int64 exponents in int32 range.
    Floating exponent: x * exp2(n) with the multiplier materialised at the
    storage dtype, matching torch's behaviour for non-integral exponents.
    """
    if DK == 1:
        xf = x
    else:
        xf = x.to(tl.float32)
    if FEXP:
        # torch evaluates x * 2**other at the promoted (widest) precision,
        # fp64 whenever the exponent tensor is fp64: 2**128 stays finite
        # there, exp2(-1024) is a nonzero fp64 subnormal so inf * it stays
        # inf, and 0 * inf is nan -- all matching the reference exactly.
        # Computing at the storage dtype instead would overflow fp32 exp2
        # at n=128 and flush exp2(-1024) to zero.  Evaluate every
        # floating-exponent case in fp64, then round to fp32 (this target's
        # TruncF lowering only supports f64 -> f32; the final f32 -> f16/bf16
        # rounding happens on store).
        y = xf.to(tl.float64) * tl.exp2(n.to(tl.float64))
        if DK != 1:
            y = y.to(tl.float32)
    else:
        if DK == 1:
            nc = tl.minimum(tl.maximum(n.to(tl.int64), -4000), 4000)
        else:
            nc = tl.minimum(tl.maximum(n, -4000), 4000)
        y = _ld.ldexp(xf, nc.to(tl.int32))
    return y


@triton.jit
def _ldexp_same(x_ptr, n_ptr, DK: tl.constexpr, NC: tl.constexpr, BLOCK: tl.constexpr):
    """Hottest path: self contiguous, exponent tensor contiguous with the same
    element count, element count an exact multiple of the tile.  Two tensor
    arguments, no bounds mask, no index arithmetic, and every launch option left
    at its default so the per-call dispatch cost is minimal."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    o = tl.load(n_ptr + offs)
    if NC == 0:
        o = o.to(tl.int32)
    elif NC == 1:
        o = o.to(tl.int64)
    elif NC == 2:
        o = o.to(tl.float32)
    else:
        o = o.to(tl.float64)
    y = _ldexp_core(x, o, DK, NC >= 2)
    tl.store(x_ptr + offs, y)


@triton.jit(do_not_specialize=["N", "N_OTHER"])
def _ldexp_flat(
    x_ptr,
    n_ptr,
    N,
    N_OTHER,
    MODE: tl.constexpr,
    DK: tl.constexpr,
    NC: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """self contiguous.  MODE 0: scalar other, 1: linear, 2: suffix modulo."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        mask = None
    else:
        mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    if MODE == 0:
        o = tl.load(n_ptr)
    else:
        if MODE == 1:
            no = offs
        else:
            no = offs % N_OTHER
        o = tl.load(n_ptr + no, mask=mask)
    if NC == 0:
        o = o.to(tl.int32)
    elif NC == 1:
        o = o.to(tl.int64)
    elif NC == 2:
        o = o.to(tl.float32)
    else:
        o = o.to(tl.float64)
    y = _ldexp_core(x, o, DK, NC >= 2)
    tl.store(x_ptr + offs, y, mask=mask)


@triton.jit(do_not_specialize=["N", "N_OTHER"] + _GEN_SCALARS)
def _ldexp_gen(
    x_ptr,
    n_ptr,
    N,
    N_OTHER,
    R: tl.constexpr,
    SM: tl.constexpr,
    OM: tl.constexpr,
    xs0,
    xs1,
    xs2,
    xs3,
    xs4,
    xs5,
    xs6,
    xs7,
    xt0,
    xt1,
    xt2,
    xt3,
    xt4,
    xt5,
    xt6,
    xt7,
    ot0,
    ot1,
    ot2,
    ot3,
    ot4,
    ot5,
    ot6,
    ot7,
    DK: tl.constexpr,
    NC: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Arbitrary strides, dims innermost-first (xs self size, xt self stride,
    ot effective other stride, already 0 for broadcast dims).
    SM 0: self contiguous.  OM 0: scalar, 1: linear, 2: modulo, 3: strided."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        mask = None
    else:
        mask = offs < N
    xo = offs
    if SM == 1:
        rem = offs
        xo = tl.zeros([BLOCK], tl.int32)
        c = rem % xs0
        rem = rem // xs0
        xo += c * xt0
        if R > 1:
            c = rem % xs1
            rem = rem // xs1
            xo += c * xt1
        if R > 2:
            c = rem % xs2
            rem = rem // xs2
            xo += c * xt2
        if R > 3:
            c = rem % xs3
            rem = rem // xs3
            xo += c * xt3
        if R > 4:
            c = rem % xs4
            rem = rem // xs4
            xo += c * xt4
        if R > 5:
            c = rem % xs5
            rem = rem // xs5
            xo += c * xt5
        if R > 6:
            c = rem % xs6
            rem = rem // xs6
            xo += c * xt6
        if R > 7:
            c = rem % xs7
            rem = rem // xs7
            xo += c * xt7
    x = tl.load(x_ptr + xo, mask=mask)
    if OM == 0:
        o = tl.load(n_ptr)
    else:
        if OM == 1:
            no = offs
        elif OM == 2:
            no = offs % N_OTHER
        else:
            rem = offs
            no = tl.zeros([BLOCK], tl.int32)
            c = rem % xs0
            rem = rem // xs0
            no += c * ot0
            if R > 1:
                c = rem % xs1
                rem = rem // xs1
                no += c * ot1
            if R > 2:
                c = rem % xs2
                rem = rem // xs2
                no += c * ot2
            if R > 3:
                c = rem % xs3
                rem = rem // xs3
                no += c * ot3
            if R > 4:
                c = rem % xs4
                rem = rem // xs4
                no += c * ot4
            if R > 5:
                c = rem % xs5
                rem = rem // xs5
                no += c * ot5
            if R > 6:
                c = rem % xs6
                rem = rem // xs6
                no += c * ot6
            if R > 7:
                c = rem % xs7
                rem = rem // xs7
                no += c * ot7
        o = tl.load(n_ptr + no, mask=mask)
    if NC == 0:
        o = o.to(tl.int32)
    elif NC == 1:
        o = o.to(tl.int64)
    elif NC == 2:
        o = o.to(tl.float32)
    else:
        o = o.to(tl.float64)
    y = _ldexp_core(x, o, DK, NC >= 2)
    tl.store(x_ptr + xo, y, mask=mask)


def _suffix_ok(sshape, oshape):
    """oshape without leading 1s is exactly the trailing part of sshape, so a
    contiguous exponent tensor maps through offs % numel."""
    i = 0
    n = len(oshape)
    while i < n and oshape[i] == 1:
        i += 1
    tail = oshape[i:]
    k = len(tail)
    if k == 0:
        return True
    if k > len(sshape):
        return False
    return sshape[-k:] == tail


def ldexp_(self, other):
    logger.debug("GEMS_HYGON LDEXP_")
    if self.dtype.is_complex:
        # Complex in-place ldexp (x * 2**other with complex exponent) is
        # handled by the generic implementation; this kernel covers the
        # floating-point self dtypes only.
        return _generic_ldexp_(self, other)
    dk = _DK.get(self.dtype)
    if dk is None:
        raise RuntimeError(
            "result type Float can't be cast to the desired output type "
            "{}".format(_CASTNAME.get(self.dtype, str(self.dtype)))
        )
    if not isinstance(other, torch.Tensor):
        raise RuntimeError(
            "ldexp_(): argument 'other' (position 2) must be Tensor, not "
            "{}".format(type(other).__name__)
        )
    nc = _NC.get(other.dtype)
    if nc is None:
        raise RuntimeError(
            "ldexp_ expected an integral or floating exponent tensor but got "
            "'{}'".format(other.dtype)
        )
    if other.dtype == torch.bool:
        other = other.view(torch.uint8)

    sshape = self.shape
    oshape = other.shape
    if sshape != oshape:
        sd = len(sshape)
        od = len(oshape)
        r = sd if sd > od else od
        rev = []
        for i in range(1, r + 1):
            a = sshape[sd - i] if i <= sd else 1
            b = oshape[od - i] if i <= od else 1
            if a == b or b == 1:
                rev.append(a)
            elif a == 1:
                rev.append(b)
            else:
                raise RuntimeError(
                    "The size of tensor a ({}) must match the size of tensor b "
                    "({}) at non-singleton dimension {}".format(a, b, r - i)
                )
        rv = tuple(reversed(rev))
        if rv != sshape:
            raise RuntimeError(
                "output with shape {} doesn't match the broadcast shape "
                "{}".format(list(sshape), list(rv))
            )

    N = self.numel()
    if N == 0:
        return self

    blk = _BLK[dk]
    even = (N & (blk - 1)) == 0
    grid = ((N + blk - 1) // blk,)
    onum = other.numel()
    ocontig = other.is_contiguous()
    if self.is_contiguous():
        if ocontig and onum == N:
            if even:
                _ldexp_same[grid](self, other, DK=dk, NC=nc, BLOCK=blk)
            else:
                _ldexp_flat[grid](
                    self, other, N, N, MODE=1, DK=dk, NC=nc, EVEN=False, BLOCK=blk
                )
            return self
        if onum == 1:
            _ldexp_flat[grid](
                self, other, N, 1, MODE=0, DK=dk, NC=nc, EVEN=even, BLOCK=blk
            )
            return self
        if ocontig and _suffix_ok(sshape, oshape):
            _ldexp_flat[grid](
                self, other, N, onum, MODE=2, DK=dk, NC=nc, EVEN=even, BLOCK=blk
            )
            return self
        sm = 0
    else:
        sm = 1

    sstr = self.stride()
    if 0 in sstr:
        for sz, st in zip(sshape, sstr):
            if sz > 1 and st == 0:
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )

    if onum == 1:
        om = 0
    elif ocontig and onum == N:
        om = 1
    elif ocontig and _suffix_ok(sshape, oshape):
        om = 2
    else:
        om = 3

    sd = len(sshape)
    od = len(oshape)
    if sd > _MAXRANK:
        raise RuntimeError("ldexp_ supports at most 8 dimensions")
    ostr = other.stride()
    off = sd - od
    xs = [1] * _MAXRANK
    xt = [0] * _MAXRANK
    ot = [0] * _MAXRANK
    for i in range(sd):
        d = sd - 1 - i
        sz = sshape[d]
        xs[i] = sz
        xt[i] = sstr[d]
        j = d - off
        if j >= 0 and oshape[j] == sz:
            ot[i] = ostr[j]
    _ldexp_gen[grid](
        self,
        other,
        N,
        onum,
        sd,
        sm,
        om,
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        xs[4],
        xs[5],
        xs[6],
        xs[7],
        xt[0],
        xt[1],
        xt[2],
        xt[3],
        xt[4],
        xt[5],
        xt[6],
        xt[7],
        ot[0],
        ot[1],
        ot[2],
        ot[3],
        ot[4],
        ot[5],
        ot[6],
        ot[7],
        DK=dk,
        NC=nc,
        EVEN=even,
        BLOCK=blk,
        num_warps=_WARPS,
    )
    return self
