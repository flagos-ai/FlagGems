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

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tld

# ---------------------------------------------------------------------------
# aten::_foreach_pow_.Scalar semantics (pow_tensor_scalar_kernel_impl, CUDA):
#   isnan(base) or exp == 1 -> base
#   exp == 2   -> base * base          exp == 3    -> base * base * base
#   exp == 0.5 -> sqrt(base)           exp == -0.5 -> 1 / sqrt(base)
#   exp == -1  -> 1 / base             exp == -2   -> 1 / (base * base)
#   exp == 0   -> 1                    else        -> std::pow(base, exp) (powf)
# fp16/bf16 are evaluated through float (opmath) and stored back with RN.
# The op is fully in-place: no output is allocated, the call returns None.
_MODE_IDENTITY = 0
_MODE_ZERO = 1
_MODE_SQUARE = 2
_MODE_CUBE = 3
_MODE_SQRT = 4
_MODE_RSQRT = 5
_MODE_RECIP = 6
_MODE_RECIP2 = 7
_MODE_GENERIC = 8

_MODE_TABLE = {
    1.0: _MODE_IDENTITY,
    0.0: _MODE_ZERO,
    2.0: _MODE_SQUARE,
    3.0: _MODE_CUBE,
    0.5: _MODE_SQRT,
    -0.5: _MODE_RSQRT,
    -1.0: _MODE_RECIP,
    -2.0: _MODE_RECIP2,
}

_MAXSLOT = 16


@triton.jit
def _upcast(x, F32: tl.constexpr):
    if F32:
        y = x.to(tl.float32)
    else:
        y = x
    return y


@triton.jit
def _pow_value(x, expon, MODE: tl.constexpr):
    if MODE == 0:
        y = x
    elif MODE == 1:
        y = tl.where(x == x, 1.0, x)
    elif MODE == 2:
        y = x * x
    elif MODE == 3:
        y = x * x * x
    elif MODE == 4:
        y = tl.sqrt(x)
    elif MODE == 5:
        y = 1.0 / tl.sqrt(x)
    elif MODE == 6:
        y = 1.0 / x
    elif MODE == 7:
        y = 1.0 / (x * x)
    else:
        y = tld.pow(x, expon)
    return y


@triton.jit
def _slot_body(
    p, offs, n, expon, MODE: tl.constexpr, F32: tl.constexpr, EVEN: tl.constexpr
):
    # Each element is read once and written once per call and never re-read, so
    # the loads are marked ".cg" (L2 only) to keep the one-pass stream from
    # allocating dead L1 lines.
    if EVEN:
        x = _upcast(tl.load(p + offs, cache_modifier=".cg"), F32)
        tl.store(
            p + offs,
            _pow_value(x, expon, MODE).to(p.dtype.element_ty),
            cache_modifier=".cg",
        )
    else:
        m = offs < n
        x = _upcast(tl.load(p + offs, mask=m, other=0.0, cache_modifier=".cg"), F32)
        tl.store(
            p + offs,
            _pow_value(x, expon, MODE).to(p.dtype.element_ty),
            mask=m,
            cache_modifier=".cg",
        )


@triton.jit(do_not_specialize=["n"])
def _pow_one_kernel(
    p,
    n,
    expon,
    MODE: tl.constexpr,
    F32: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    _slot_body(p, offs, n, expon, MODE, F32, EVEN)


@triton.jit(do_not_specialize=["n"])
def _pow_fused_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    n,
    expon,
    MODE: tl.constexpr,
    F32: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    k = tl.program_id(1)
    if k == 0:
        _slot_body(p0, offs, n, expon, MODE, F32, EVEN)
    elif k == 1:
        _slot_body(p1, offs, n, expon, MODE, F32, EVEN)
    elif k == 2:
        _slot_body(p2, offs, n, expon, MODE, F32, EVEN)
    elif k == 3:
        _slot_body(p3, offs, n, expon, MODE, F32, EVEN)
    elif k == 4:
        _slot_body(p4, offs, n, expon, MODE, F32, EVEN)
    elif k == 5:
        _slot_body(p5, offs, n, expon, MODE, F32, EVEN)
    elif k == 6:
        _slot_body(p6, offs, n, expon, MODE, F32, EVEN)
    elif k == 7:
        _slot_body(p7, offs, n, expon, MODE, F32, EVEN)
    elif k == 8:
        _slot_body(p8, offs, n, expon, MODE, F32, EVEN)
    elif k == 9:
        _slot_body(p9, offs, n, expon, MODE, F32, EVEN)
    elif k == 10:
        _slot_body(p10, offs, n, expon, MODE, F32, EVEN)
    elif k == 11:
        _slot_body(p11, offs, n, expon, MODE, F32, EVEN)
    elif k == 12:
        _slot_body(p12, offs, n, expon, MODE, F32, EVEN)
    elif k == 13:
        _slot_body(p13, offs, n, expon, MODE, F32, EVEN)
    elif k == 14:
        _slot_body(p14, offs, n, expon, MODE, F32, EVEN)
    else:
        _slot_body(p15, offs, n, expon, MODE, F32, EVEN)


@triton.jit
def _pow_fused_even_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    MODE: tl.constexpr,
    F32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Same fused body as _pow_fused_kernel, but the mask and the exponent are
    # folded away so the launch carries only the 16 storage pointers. This is
    # the common path: every benchmark shape is an exact multiple of BLOCK and
    # no benchmark exponent needs libdevice powf.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    k = tl.program_id(1)
    if k == 0:
        _slot_body(p0, offs, 0, 0.0, MODE, F32, True)
    elif k == 1:
        _slot_body(p1, offs, 0, 0.0, MODE, F32, True)
    elif k == 2:
        _slot_body(p2, offs, 0, 0.0, MODE, F32, True)
    elif k == 3:
        _slot_body(p3, offs, 0, 0.0, MODE, F32, True)
    elif k == 4:
        _slot_body(p4, offs, 0, 0.0, MODE, F32, True)
    elif k == 5:
        _slot_body(p5, offs, 0, 0.0, MODE, F32, True)
    elif k == 6:
        _slot_body(p6, offs, 0, 0.0, MODE, F32, True)
    elif k == 7:
        _slot_body(p7, offs, 0, 0.0, MODE, F32, True)
    elif k == 8:
        _slot_body(p8, offs, 0, 0.0, MODE, F32, True)
    elif k == 9:
        _slot_body(p9, offs, 0, 0.0, MODE, F32, True)
    elif k == 10:
        _slot_body(p10, offs, 0, 0.0, MODE, F32, True)
    elif k == 11:
        _slot_body(p11, offs, 0, 0.0, MODE, F32, True)
    elif k == 12:
        _slot_body(p12, offs, 0, 0.0, MODE, F32, True)
    elif k == 13:
        _slot_body(p13, offs, 0, 0.0, MODE, F32, True)
    elif k == 14:
        _slot_body(p14, offs, 0, 0.0, MODE, F32, True)
    else:
        _slot_body(p15, offs, 0, 0.0, MODE, F32, True)


@triton.jit(do_not_specialize=["n"])
def _pow_strided_kernel(
    p,
    expon,
    n,
    d0,
    d1,
    d2,
    s0,
    s1,
    s2,
    s3,
    MODE: tl.constexpr,
    F32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    idx = offs
    i0 = idx % d0
    idx = idx // d0
    i1 = idx % d1
    idx = idx // d1
    i2 = idx % d2
    i3 = idx // d2
    ptrs = p + i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3
    x = _upcast(tl.load(ptrs, mask=m, other=0.0), F32)
    tl.store(ptrs, _pow_value(x, expon, MODE).to(p.dtype.element_ty), mask=m)


def _cfg(n, dt):
    if n <= 8192:
        return 1024, 4
    if dt is torch.float32 or dt is torch.float64:
        return 4096, 8
    return 2048, 2


def _collapse(shape, strides):
    shp = []
    strd = []
    for s, st in zip(reversed(shape), reversed(strides)):
        if s == 1:
            continue
        if shp and st * s == strd[-1]:
            shp[-1] *= s
        else:
            shp.append(s)
            strd.append(st)
    return shp[::-1], strd[::-1]


def _launch_strided(t, fe, mode, f32, block, warps):
    n = t.numel()
    shp, strd = _collapse(t.shape, t.stride())
    if len(shp) > 4:
        raise RuntimeError("foreach_pow_: unsupported tensor rank")
    # innermost-first, padded to 4 dims
    sizes = shp[::-1] + [1, 1, 1]
    sts = strd[::-1] + [0, 0, 0, 0]
    _pow_strided_kernel[(triton.cdiv(n, block),)](
        t,
        fe,
        n,
        sizes[0],
        sizes[1],
        sizes[2],
        sts[0],
        sts[1],
        sts[2],
        sts[3],
        MODE=mode,
        F32=f32,
        BLOCK=block,
        num_warps=warps,
    )


def foreach_pow_scalar_(self, exponent):
    if self is None:
        return None
    if isinstance(self, (list, tuple)):
        tensors = self
    else:
        tensors = list(self)
    K = len(tensors)
    if K == 0:
        return None

    e = exponent
    if isinstance(e, torch.Tensor):
        e = e.item()
    fe = float(e)
    mode = _MODE_TABLE.get(fe, _MODE_GENERIC)
    if mode == _MODE_IDENTITY:
        return None

    t0 = tensors[0]
    n0 = t0.numel()
    if n0 == 0:
        return None
    dt = t0.dtype
    f32 = dt is not torch.float64

    fused = True
    ptrs_seen = []
    for t in tensors:
        if t.dtype is not dt or t.numel() != n0 or not t.is_contiguous():
            fused = False
            break
        ptrs_seen.append(t.data_ptr())

    # aten applies the power to each list entry sequentially, so a single fused
    # launch is only equivalent when no two entries share storage.
    if fused and len(set(ptrs_seen)) != K:
        fused = False

    block, warps = _cfg(n0, dt)
    if fused:
        even = (n0 % block) == 0
        nb = (n0 + block - 1) // block
        fast = even and mode != _MODE_GENERIC
        if K < _MAXSLOT:
            ptrs = list(tensors) + [tensors[-1]] * (_MAXSLOT - K)
        elif K == _MAXSLOT:
            ptrs = tensors
        else:
            ptrs = None
        if ptrs is not None:
            if fast:
                _pow_fused_even_kernel[(nb, K)](
                    *ptrs, mode, f32, block, num_warps=warps
                )
            else:
                _pow_fused_kernel[(nb, K)](
                    *ptrs, n0, fe, mode, f32, even, block, num_warps=warps
                )
        else:
            for s in range(0, K, _MAXSLOT):
                grp = tensors[s : s + _MAXSLOT]
                g = len(grp)
                if g < _MAXSLOT:
                    grp = list(grp) + [grp[-1]] * (_MAXSLOT - g)
                if fast:
                    _pow_fused_even_kernel[(nb, g)](
                        *grp, mode, f32, block, num_warps=warps
                    )
                else:
                    _pow_fused_kernel[(nb, g)](
                        *grp, n0, fe, mode, f32, even, block, num_warps=warps
                    )
        return None

    for t in tensors:
        n = t.numel()
        if n == 0:
            continue
        b, w = _cfg(n, t.dtype)
        use_f32 = t.dtype is not torch.float64
        if t.is_contiguous():
            _pow_one_kernel[((n + b - 1) // b,)](
                t,
                n,
                fe,
                mode,
                use_f32,
                (n % b) == 0,
                b,
                num_warps=w,
            )
        else:
            _launch_strided(t, fe, mode, use_f32, b, w)
    return None
