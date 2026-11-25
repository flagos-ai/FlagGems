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

from flag_gems.ops._foreach_unary import _foreach_rsqrt as default__foreach_rsqrt

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_BLOCK = 1024
_MAX_RANK = 8
_MAX_FUSE = 16
# Multi-tensor fusion only pays below the per-launch-overhead crossover.
_FUSE_MAX_BYTES = 1 << 20

_F16 = torch.float16
_BF16 = torch.bfloat16
_F32 = torch.float32
_F64 = torch.float64
_C64 = torch.complex64
_C128 = torch.complex128

# ---------------------------------------------------------------------------
# y = rsqrt(x) with ATen-observable semantics
#   * fp16/bf16 -> computed in fp32, rounded once on store
#   * int/bool  -> converted to fp32 (ATen promotes to the default float dtype)
#   * fp64      -> stays fp64
#   * rsqrt(-0.0) must be -inf (hardware rsqrt yields +inf), fixed up exactly
# ---------------------------------------------------------------------------


@triton.jit
def _rsqrt_val(x, IS_F64: tl.constexpr):
    if IS_F64:
        y = tl.rsqrt(x)
        neg = x.to(tl.int64, bitcast=True) < 0
    else:
        x = x.to(tl.float32)
        y = tl.rsqrt(x)
        neg = x.to(tl.int32, bitcast=True) < 0
    return tl.where(neg & (x == 0.0), -float("inf"), y)


@triton.jit
def _rsqrt_flat(x_ptr, o_ptr, n, IS_F64: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, _rsqrt_val(x, IS_F64), mask=mask)


@triton.jit
def _rsqrt_fused(
    x0,
    o0,
    x1,
    o1,
    x2,
    o2,
    x3,
    o3,
    x4,
    o4,
    x5,
    o5,
    x6,
    o6,
    x7,
    o7,
    x8,
    o8,
    x9,
    o9,
    x10,
    o10,
    x11,
    o11,
    x12,
    o12,
    x13,
    o13,
    x14,
    o14,
    x15,
    o15,
    n,
    SLOTS: tl.constexpr,
    IS_F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """SLOTS tensors sharing one contiguous shape (numel n) in a single launch."""
    pid = tl.program_id(0)
    nblk = tl.cdiv(n, BLOCK)
    xp = x0
    op = o0
    loc = pid
    if SLOTS >= 2:
        if pid >= nblk:
            xp = x1
            op = o1
            loc = pid - nblk
    if SLOTS >= 3:
        if pid >= 2 * nblk:
            xp = x2
            op = o2
            loc = pid - 2 * nblk
    if SLOTS >= 4:
        if pid >= 3 * nblk:
            xp = x3
            op = o3
            loc = pid - 3 * nblk
    if SLOTS >= 5:
        if pid >= 4 * nblk:
            xp = x4
            op = o4
            loc = pid - 4 * nblk
    if SLOTS >= 6:
        if pid >= 5 * nblk:
            xp = x5
            op = o5
            loc = pid - 5 * nblk
    if SLOTS >= 7:
        if pid >= 6 * nblk:
            xp = x6
            op = o6
            loc = pid - 6 * nblk
    if SLOTS >= 8:
        if pid >= 7 * nblk:
            xp = x7
            op = o7
            loc = pid - 7 * nblk
    if SLOTS >= 9:
        if pid >= 8 * nblk:
            xp = x8
            op = o8
            loc = pid - 8 * nblk
    if SLOTS >= 10:
        if pid >= 9 * nblk:
            xp = x9
            op = o9
            loc = pid - 9 * nblk
    if SLOTS >= 11:
        if pid >= 10 * nblk:
            xp = x10
            op = o10
            loc = pid - 10 * nblk
    if SLOTS >= 12:
        if pid >= 11 * nblk:
            xp = x11
            op = o11
            loc = pid - 11 * nblk
    if SLOTS >= 13:
        if pid >= 12 * nblk:
            xp = x12
            op = o12
            loc = pid - 12 * nblk
    if SLOTS >= 14:
        if pid >= 13 * nblk:
            xp = x13
            op = o13
            loc = pid - 13 * nblk
    if SLOTS >= 15:
        if pid >= 14 * nblk:
            xp = x14
            op = o14
            loc = pid - 14 * nblk
    if SLOTS >= 16:
        if pid >= 15 * nblk:
            xp = x15
            op = o15
            loc = pid - 15 * nblk
    offs = loc * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(xp + offs, mask=mask)
    tl.store(op + offs, _rsqrt_val(x, IS_F64), mask=mask)


# ---------------------------------------------------------------------------
# strided kernel; complex tensors are handled through .real/.imag views,
# which share one stride pattern for both components.
# ---------------------------------------------------------------------------


@triton.jit
def _rsqrt_strided(
    x0,
    x1,
    o0,
    o1,
    n,
    sh0,
    sh1,
    sh2,
    sh3,
    sh4,
    sh5,
    sh6,
    sh7,
    sx0,
    sx1,
    sx2,
    sx3,
    sx4,
    sx5,
    sx6,
    sx7,
    so0,
    so1,
    so2,
    so3,
    so4,
    so5,
    so6,
    so7,
    RANK: tl.constexpr,
    COMPLEX: tl.constexpr,
    IS_F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx
    xo = tl.zeros((BLOCK,), dtype=tl.int32)
    oo = tl.zeros((BLOCK,), dtype=tl.int32)
    if RANK >= 8:
        i = rem % sh7
        rem = rem // sh7
        xo += i * sx7
        oo += i * so7
    if RANK >= 7:
        i = rem % sh6
        rem = rem // sh6
        xo += i * sx6
        oo += i * so6
    if RANK >= 6:
        i = rem % sh5
        rem = rem // sh5
        xo += i * sx5
        oo += i * so5
    if RANK >= 5:
        i = rem % sh4
        rem = rem // sh4
        xo += i * sx4
        oo += i * so4
    if RANK >= 4:
        i = rem % sh3
        rem = rem // sh3
        xo += i * sx3
        oo += i * so3
    if RANK >= 3:
        i = rem % sh2
        rem = rem // sh2
        xo += i * sx2
        oo += i * so2
    if RANK >= 2:
        i = rem % sh1
        rem = rem // sh1
        xo += i * sx1
        oo += i * so1
    if RANK >= 1:
        i = rem % sh0
        xo += i * sx0
        oo += i * so0
    if COMPLEX:
        ar = tl.load(x0 + xo, mask=mask)
        ai = tl.load(x1 + xo, mask=mask)
        if not IS_F64:
            ar = ar.to(tl.float32)
            ai = ai.to(tl.float32)
        r = tl.sqrt(ar * ar + ai * ai)
        t1 = tl.sqrt((r + ar) * 0.5)
        t2 = tl.sqrt((r - ar) * 0.5)
        pos = ar >= 0.0
        if IS_F64:
            neg_ai = ai.to(tl.int64, bitcast=True) < 0
        else:
            neg_ai = ai.to(tl.int32, bitcast=True) < 0
        wr = tl.where(pos, t1, tl.abs(ai) / (2.0 * t2))
        wi = tl.where(pos, ai / (2.0 * t1), tl.where(neg_ai, -t2, t2))
        # 1/sqrt(z) = conj(sqrt(z)) / |z|, matching ATen's complex reciprocal
        # semantics: a zero magnitude yields (inf, nan), an infinite one (0, 0).
        real = wr / r
        imag = -(wi / r)
        is_inf = r == float("inf")
        real = tl.where(is_inf, 0.0, real)
        imag = tl.where(is_inf, 0.0, imag)
        is_zero = r == 0.0
        real = tl.where(is_zero, float("inf"), real)
        imag = tl.where(is_zero, float("nan"), imag)
        tl.store(o0 + oo, real, mask=mask)
        tl.store(o1 + oo, imag, mask=mask)
    else:
        x = tl.load(x0 + xo, mask=mask)
        tl.store(o0 + oo, _rsqrt_val(x, IS_F64), mask=mask)


def _launch_strided(x0, x1, o0, o1, n, is_f64, complex_):
    shape = x0.shape
    rank = len(shape)
    if rank > _MAX_RANK:
        # extremely rare: peel the leading dimension with zero-copy views
        d0 = shape[0]
        step = n // d0
        for i in range(d0):
            _launch_strided(
                x0.narrow(0, i, 1).squeeze(0),
                x1.narrow(0, i, 1).squeeze(0),
                o0.narrow(0, i, 1).squeeze(0),
                o1.narrow(0, i, 1).squeeze(0),
                step,
                is_f64,
                complex_,
            )
        return
    if rank == 0:  # 0-dim, numel 1, no strides
        if not complex_:
            _rsqrt_flat[(1,)](x0, o0, n, IS_F64=is_f64, BLOCK=_BLOCK, num_warps=4)
            return
        shape, sx, so, rank = (1,), (0,), (0,), 1
    else:
        # the linear index walks elements in logical row-major order; input and
        # output use their own (possibly different) stride decompositions
        sx = x0.stride()
        so = o0.stride()
    pad = _MAX_RANK - rank
    sh = list(shape) + [1] * pad
    sx = list(sx) + [0] * pad
    so = list(so) + [0] * pad
    block = _BLOCK
    _rsqrt_strided[((n + block - 1) // block,)](
        x0,
        x1,
        o0,
        o1,
        n,
        sh[0],
        sh[1],
        sh[2],
        sh[3],
        sh[4],
        sh[5],
        sh[6],
        sh[7],
        sx[0],
        sx[1],
        sx[2],
        sx[3],
        sx[4],
        sx[5],
        sx[6],
        sx[7],
        so[0],
        so[1],
        so[2],
        so[3],
        so[4],
        so[5],
        so[6],
        so[7],
        RANK=rank,
        COMPLEX=complex_,
        IS_F64=is_f64,
        BLOCK=block,
        num_warps=4,
    )


def _launch_fused(xs, os_, n, is_f64, block, k):
    if k < _MAX_FUSE:
        xs = xs + [xs[0]] * (_MAX_FUSE - k)
        os_ = os_ + [os_[0]] * (_MAX_FUSE - k)
    grid = (((n + block - 1) // block) * k,)
    _rsqrt_fused[grid](
        xs[0],
        os_[0],
        xs[1],
        os_[1],
        xs[2],
        os_[2],
        xs[3],
        os_[3],
        xs[4],
        os_[4],
        xs[5],
        os_[5],
        xs[6],
        os_[6],
        xs[7],
        os_[7],
        xs[8],
        os_[8],
        xs[9],
        os_[9],
        xs[10],
        os_[10],
        xs[11],
        os_[11],
        xs[12],
        os_[12],
        xs[13],
        os_[13],
        xs[14],
        os_[14],
        xs[15],
        os_[15],
        n,
        SLOTS=k,
        IS_F64=is_f64,
        BLOCK=block,
        num_warps=4,
    )


def _foreach_rsqrt(self):
    logger.debug("GEMS_MTHREADS _FOREACH_RSQRT")
    _seq = self if isinstance(self, (list, tuple)) else [self]
    if any(not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in _seq):
        return default__foreach_rsqrt(self)
    if isinstance(self, torch.Tensor):
        raise TypeError(
            "_foreach_rsqrt(): argument 'self' (position 1) must be tuple of Tensors, not Tensor"
        )
    tensors = list(self)
    n_t = len(tensors)
    if n_t == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    outs = []
    ap = outs.append
    for t in tensors:
        dt = t.dtype
        if dt.is_floating_point or dt.is_complex:
            ap(torch.empty_like(t))
        else:
            ap(torch.empty(t.shape, dtype=_F32, device=t.device))

    block = _BLOCK
    if n_t > 1:
        t0 = tensors[0]
        dt0 = t0.dtype
        if dt0 is _F32 or dt0 is _F16 or dt0 is _BF16 or dt0 is _F64:
            n0 = t0.numel()
            # Fuse only when each tensor is small enough that per-launch cost
            # dominates: above this footprint one launch per tensor is faster
            # (measured: fusion loses ~1.8x once a tensor is tens of MB).
            fuse = 0 < n0 and n0 * t0.element_size() <= _FUSE_MAX_BYTES
            if fuse:
                for t in tensors:
                    if t.dtype is not dt0 or t.numel() != n0 or not t.is_contiguous():
                        fuse = False
                        break
            if fuse:
                is_f64 = dt0 is _F64
                i = 0
                while i < n_t:
                    k = n_t - i
                    if k > _MAX_FUSE:
                        k = _MAX_FUSE
                    _launch_fused(
                        tensors[i : i + k], outs[i : i + k], n0, is_f64, block, k
                    )
                    i += k
                return tuple(outs)

    for t, o in zip(tensors, outs):
        n = t.numel()
        if n == 0:
            continue
        dt = t.dtype
        if dt is _C64 or dt is _C128:
            _launch_strided(t.real, t.imag, o.real, o.imag, n, dt is _C128, True)
        else:
            src = t.view(torch.uint8) if dt is torch.bool else t
            is_f64 = dt is _F64
            if src.is_contiguous():
                _rsqrt_flat[((n + block - 1) // block,)](
                    src, o, n, IS_F64=is_f64, BLOCK=block, num_warps=4
                )
            else:
                _launch_strided(src, src, o, o, n, is_f64, False)
    return tuple(outs)
