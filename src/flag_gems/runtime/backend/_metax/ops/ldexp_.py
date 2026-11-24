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

"""In-place ldexp: self = self * 2**other  (aten::ldexp_ semantics).

Target measurements showed torch's in-place kernel is
    self <- self * exp2(other.to(self.dtype))
with the power of two evaluated in `self`'s dtype, so the exponent saturates in
that dtype (fp16: 2**16 -> inf, 2**-30 -> 0; bf16/fp32 analogous; fp64 in fp64).
Reproducing that structure gives bit-exact agreement for fp16/bf16/fp32 and
<=1 ulp for fp64 (torch uses pow there).

The literal multiply turns the saturating cases 0*inf and inf*0 into NaN, which
IEEE ldexp does not do, so `_scale` restores signed zero / signed infinity for
those two combinations (a NaN operand still propagates).

Two kernels:
  * _ldexp_flat : fast path, contiguous self, exponent either flat or scalar
  * _ldexp_gen  : correctness path for non-contiguous self / broadcast exponent
"""

import torch
import triton
import triton.language as tl

_MAX_DIMS = 8

_N_FLAT = 0  # exponent offset == flat offset
_N_ZERO = 1  # exponent has a single value

_INT_DTYPE_NAMES = {
    torch.uint8: "Byte",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.bool: "Bool",
}


@triton.jit
def _scale(x, p):
    """x * 2**p with IEEE-ldexp handling of the saturation cases.

    The power of two is saturated in the operand dtype (2**n -> inf / 0), so a
    literal multiply turns 0*inf and inf*0 into NaN.  ldexp keeps signed zero and
    signed infinity; only a NaN operand (input or exponent) propagates.
    """
    y = x * p
    keep = ((x == 0.0) | (tl.abs(x) == float("inf"))) & (p == p)
    return tl.where(keep, x, y)


@triton.jit
def _ldexp_flat(
    X, N, numel, NMODE: tl.constexpr, IS_FP64: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(X + offs, mask=mask)
    if NMODE == 0:
        nn = tl.load(N + offs, mask=mask)
    else:
        nn = tl.load(N)
    if IS_FP64:
        y = _scale(x, tl.exp2(nn.to(tl.float64)))
    else:
        p = tl.exp2(nn.to(tl.float32)).to(x.dtype)
        y = _scale(x, p)
    tl.store(X + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _ldexp_gen(
    X,
    N,
    numel,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    XS0,
    XS1,
    XS2,
    XS3,
    XS4,
    XS5,
    XS6,
    XS7,
    NS0,
    NS1,
    NS2,
    NS3,
    NS4,
    NS5,
    NS6,
    NS7,
    RANK: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    rem = offs
    x_off = tl.zeros((BLOCK,), dtype=tl.int32)
    n_off = tl.zeros((BLOCK,), dtype=tl.int32)

    if RANK >= 1:
        idx = rem % S0
        rem = rem // S0
        x_off += idx * XS0
        n_off += idx * NS0
    if RANK >= 2:
        idx = rem % S1
        rem = rem // S1
        x_off += idx * XS1
        n_off += idx * NS1
    if RANK >= 3:
        idx = rem % S2
        rem = rem // S2
        x_off += idx * XS2
        n_off += idx * NS2
    if RANK >= 4:
        idx = rem % S3
        rem = rem // S3
        x_off += idx * XS3
        n_off += idx * NS3
    if RANK >= 5:
        idx = rem % S4
        rem = rem // S4
        x_off += idx * XS4
        n_off += idx * NS4
    if RANK >= 6:
        idx = rem % S5
        rem = rem // S5
        x_off += idx * XS5
        n_off += idx * NS5
    if RANK >= 7:
        idx = rem % S6
        rem = rem // S6
        x_off += idx * XS6
        n_off += idx * NS6
    if RANK >= 8:
        idx = rem % S7
        rem = rem // S7
        x_off += idx * XS7
        n_off += idx * NS7

    x = tl.load(X + x_off, mask=mask)
    nn = tl.load(N + n_off, mask=mask)
    if IS_FP64:
        y = _scale(x, tl.exp2(nn.to(tl.float64)))
    else:
        p = tl.exp2(nn.to(tl.float32)).to(x.dtype)
        y = _scale(x, p)
    tl.store(X + x_off, y.to(x.dtype), mask=mask)


def _broadcast_shape(a, b):
    nd = max(len(a), len(b))
    out = []
    for i in range(nd):
        da = a[len(a) - 1 - i] if i < len(a) else 1
        db = b[len(b) - 1 - i] if i < len(b) else 1
        if da == db or db == 1:
            out.append(da)
        elif da == 1:
            out.append(db)
        else:
            raise RuntimeError(
                "The size of tensor a ({}) must match the size of tensor b ({}) "
                "at non-singleton dimension {}".format(da, db, nd - 1 - i)
            )
    out.reverse()
    return tuple(out)


def _launch_config(numel):
    if numel <= 1024:
        return 256, 2
    if numel <= 1 << 18:
        return 1024, 4
    return 2048, 4


def ldexp_(self, other):
    if not isinstance(self, torch.Tensor):
        raise TypeError("ldexp_(): argument 'self' must be Tensor")
    if not isinstance(other, torch.Tensor):
        raise TypeError("ldexp_(): argument 'other' must be Tensor")
    if not self.is_floating_point():
        raise RuntimeError(
            "result type Float can't be cast to the desired output type "
            + _INT_DTYPE_NAMES.get(self.dtype, "Int")
        )

    # torch refuses to write through a view whose elements alias each other
    for size, stride in zip(self.shape, self.stride()):
        if size > 1 and stride == 0:
            raise RuntimeError(
                "unsupported operation: more than one element of the written-to "
                "tensor refers to a single memory location. Please clone() the "
                "tensor before performing the operation."
            )

    self_shape = tuple(self.shape)
    bshape = _broadcast_shape(self_shape, tuple(other.shape))
    if bshape != self_shape:
        raise RuntimeError(
            "output with shape {} doesn't match the broadcast shape {}".format(
                list(self_shape), list(bshape)
            )
        )

    numel = self.numel()
    if numel == 0:
        return self

    nsrc = other
    if nsrc.dtype == torch.bool:
        nsrc = nsrc.view(torch.uint8)

    is_fp64 = self.dtype == torch.float64
    x_contig = self.is_contiguous()
    block, warps = _launch_config(numel)
    grid = (triton.cdiv(numel, block),)

    if numel == 1:
        _ldexp_flat[(1,)](self, nsrc, 1, _N_ZERO, is_fp64, BLOCK=64, num_warps=1)
        return self

    if x_contig and nsrc.numel() == 1:
        _ldexp_flat[grid](
            self, nsrc, numel, _N_ZERO, is_fp64, BLOCK=block, num_warps=warps
        )
        return self

    if x_contig and nsrc.is_contiguous() and nsrc.shape == self_shape:
        _ldexp_flat[grid](
            self, nsrc, numel, _N_FLAT, is_fp64, BLOCK=block, num_warps=warps
        )
        return self

    # general path: arbitrary self strides and/or broadcast exponent
    nd = self.ndim
    ss = list(self.shape)
    xs = list(self.stride())
    pad = nd - nsrc.ndim
    nst = [0] * pad + list(nsrc.stride())
    nsh = [1] * pad + list(nsrc.shape)
    nst = [0 if nsh[i] == 1 else nst[i] for i in range(nd)]

    keep = [i for i in range(nd) if ss[i] != 1]
    ss = [ss[i] for i in keep]
    xs = [xs[i] for i in keep]
    nst = [nst[i] for i in keep]
    if not ss:
        ss, xs, nst = [1], [0], [0]

    cs, cx, cn = [ss[0]], [xs[0]], [nst[0]]
    for d in range(1, len(ss)):
        if cx[-1] == xs[d] * ss[d] and (
            cn[-1] == nst[d] * ss[d] or (cn[-1] == 0 and nst[d] == 0)
        ):
            cs[-1] *= ss[d]
            cx[-1] = xs[d]
            cn[-1] = nst[d]
        else:
            cs.append(ss[d])
            cx.append(xs[d])
            cn.append(nst[d])

    rank = len(cs)
    if rank > _MAX_DIMS:
        raise RuntimeError("ldexp_: tensor rank too large after collapse")

    # kernel dim 0 is the innermost (fastest varying) dimension
    cs.reverse()
    cx.reverse()
    cn.reverse()
    if x_contig:
        ls = [1] * rank
        for i in range(1, rank):
            ls[i] = ls[i - 1] * cs[i - 1]
        cx = ls

    scal = cs + [1] * (_MAX_DIMS - rank)
    xscal = cx + [0] * (_MAX_DIMS - rank)
    nscal = cn + [0] * (_MAX_DIMS - rank)

    _ldexp_gen[grid](
        self,
        nsrc,
        numel,
        scal[0],
        scal[1],
        scal[2],
        scal[3],
        scal[4],
        scal[5],
        scal[6],
        scal[7],
        xscal[0],
        xscal[1],
        xscal[2],
        xscal[3],
        xscal[4],
        xscal[5],
        xscal[6],
        xscal[7],
        nscal[0],
        nscal[1],
        nscal[2],
        nscal[3],
        nscal[4],
        nscal[5],
        nscal[6],
        nscal[7],
        RANK=rank,
        IS_FP64=is_fp64,
        BLOCK=block,
        num_warps=warps,
    )
    return self
