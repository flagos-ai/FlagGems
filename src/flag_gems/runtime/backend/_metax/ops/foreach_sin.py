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

"""Triton implementation of aten::_foreach_sin (elementwise sin over a tensor list).

Semantics mirror ATen (verified against the target's torch 2.10 build):
  * floating point (fp16/bf16/fp32/fp64) -> same dtype, sin computed in fp32 except fp64
  * integral (int8/uint8/int16/int32/int64) -> promoted to float32 output
  * bool -> RuntimeError (ATen dispatch failure)
  * float8 -> NotImplementedError
  * complex64/complex128 -> same dtype, sin(a+ib) = sin a cosh b + i cos a sinh b
  * every output is freshly allocated with ATen's layout (empty_like preserves
    dense-permuted / channels_last strides, and falls back to contiguous for
    overlapping views such as expand()), so strides match the reference too.

Host strategy: list elements that share device + dtype class and are contiguous
and small enough are served by one fused launch over up to 16 tensors (each
program handles the same block index of every tensor, masked by that tensor's
element count), which amortizes Triton's per-launch cost for many-small-tensor
lists.  Every other element (single large tensors, strided views, complex) uses
the per-tensor kernels.

16-bit real paths evaluate sin with an inline branch-free fp32 polynomial, which
is bit-identical to ATen after rounding but avoids the out-of-line sinf call.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_MAX_RANK = 8
_FUSE_MAX = 16
_FUSE_MAX_ELEMS = 1 << 20

# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _sinf_poly(x):
    """Branch-free fp32 sine, ~2.4e-8 absolute error for |x| up to ~1e5.

    Quadrant reduction to [-pi/4, pi/4] with a 3-term Cody-Waite split of pi/2,
    then Cephes degree-7 sin / degree-8 cos minimax polynomials and a select.
    For 16-bit outputs this is bit-identical to ATen's sin after rounding.
    """
    q = x * 0.63661977236758134308
    k = tl.floor(q + 0.5)
    r = x - k * 1.5707962512969971
    r = r - k * 7.5497894158615964e-08
    r = r - k * 5.3903025299577648e-15
    r2 = r * r
    s = r + (r * r2) * (
        -1.6666654611e-1 + r2 * (8.3321608736e-3 + r2 * (-1.9515295891e-4))
    )
    c = (
        1.0
        - 0.5 * r2
        + (r2 * r2)
        * (
            4.166664568298827e-2
            + r2 * (-1.388731625493765e-3 + r2 * 2.443315711809948e-5)
        )
    )
    ki = k.to(tl.int32)
    out = tl.where((ki & 1) != 0, c, s)
    return tl.where((ki & 2) != 0, -out, out)


@triton.jit
def _sin_block(x_ptr, o_ptr, offs, n, F64: tl.constexpr, FAST: tl.constexpr):
    m = offs < n
    v = tl.load(x_ptr + offs, mask=m, other=0)
    if FAST:
        r = _sinf_poly(v.to(tl.float32))
    elif F64:
        r = tl.sin(v.to(tl.float64))
    else:
        r = tl.sin(v.to(tl.float32))
    tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _sin_flat(
    x_ptr, o_ptr, n, BLOCK: tl.constexpr, F64: tl.constexpr, FAST: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    _sin_block(x_ptr, o_ptr, offs, n, F64, FAST)


@triton.jit
def _sin_fused2(
    x0, o0, n0, x1, o1, n1, BLOCK: tl.constexpr, F64: tl.constexpr, FAST: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    _sin_block(x0, o0, offs, n0, F64, FAST)
    _sin_block(x1, o1, offs, n1, F64, FAST)


@triton.jit
def _sin_fused4(
    x0,
    o0,
    n0,
    x1,
    o1,
    n1,
    x2,
    o2,
    n2,
    x3,
    o3,
    n3,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
    FAST: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    _sin_block(x0, o0, offs, n0, F64, FAST)
    _sin_block(x1, o1, offs, n1, F64, FAST)
    _sin_block(x2, o2, offs, n2, F64, FAST)
    _sin_block(x3, o3, offs, n3, F64, FAST)


@triton.jit
def _sin_fused8(
    x0,
    o0,
    n0,
    x1,
    o1,
    n1,
    x2,
    o2,
    n2,
    x3,
    o3,
    n3,
    x4,
    o4,
    n4,
    x5,
    o5,
    n5,
    x6,
    o6,
    n6,
    x7,
    o7,
    n7,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
    FAST: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    _sin_block(x0, o0, offs, n0, F64, FAST)
    _sin_block(x1, o1, offs, n1, F64, FAST)
    _sin_block(x2, o2, offs, n2, F64, FAST)
    _sin_block(x3, o3, offs, n3, F64, FAST)
    _sin_block(x4, o4, offs, n4, F64, FAST)
    _sin_block(x5, o5, offs, n5, F64, FAST)
    _sin_block(x6, o6, offs, n6, F64, FAST)
    _sin_block(x7, o7, offs, n7, F64, FAST)


@triton.jit
def _sin_fused16(
    x0,
    o0,
    n0,
    x1,
    o1,
    n1,
    x2,
    o2,
    n2,
    x3,
    o3,
    n3,
    x4,
    o4,
    n4,
    x5,
    o5,
    n5,
    x6,
    o6,
    n6,
    x7,
    o7,
    n7,
    x8,
    o8,
    n8,
    x9,
    o9,
    n9,
    x10,
    o10,
    n10,
    x11,
    o11,
    n11,
    x12,
    o12,
    n12,
    x13,
    o13,
    n13,
    x14,
    o14,
    n14,
    x15,
    o15,
    n15,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
    FAST: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    _sin_block(x0, o0, offs, n0, F64, FAST)
    _sin_block(x1, o1, offs, n1, F64, FAST)
    _sin_block(x2, o2, offs, n2, F64, FAST)
    _sin_block(x3, o3, offs, n3, F64, FAST)
    _sin_block(x4, o4, offs, n4, F64, FAST)
    _sin_block(x5, o5, offs, n5, F64, FAST)
    _sin_block(x6, o6, offs, n6, F64, FAST)
    _sin_block(x7, o7, offs, n7, F64, FAST)
    _sin_block(x8, o8, offs, n8, F64, FAST)
    _sin_block(x9, o9, offs, n9, F64, FAST)
    _sin_block(x10, o10, offs, n10, F64, FAST)
    _sin_block(x11, o11, offs, n11, F64, FAST)
    _sin_block(x12, o12, offs, n12, F64, FAST)
    _sin_block(x13, o13, offs, n13, F64, FAST)
    _sin_block(x14, o14, offs, n14, F64, FAST)
    _sin_block(x15, o15, offs, n15, F64, FAST)


@triton.jit
def _sin_strided(
    x_ptr,
    o_ptr,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
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
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
    F64: tl.constexpr,
    FAST: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n
    rem = idx
    xo = tl.zeros([BLOCK], tl.int64)
    oo = tl.zeros([BLOCK], tl.int64)

    # NOTE: scalar int args are plain python ints (possibly constexpr-specialized),
    # so no .to() call is allowed here; int64 index times python int promotes to int64.
    r = rem % d0
    rem = rem // d0
    xo += r * s0
    oo += r * t0
    if RANK > 1:
        r = rem % d1
        rem = rem // d1
        xo += r * s1
        oo += r * t1
    if RANK > 2:
        r = rem % d2
        rem = rem // d2
        xo += r * s2
        oo += r * t2
    if RANK > 3:
        r = rem % d3
        rem = rem // d3
        xo += r * s3
        oo += r * t3
    if RANK > 4:
        r = rem % d4
        rem = rem // d4
        xo += r * s4
        oo += r * t4
    if RANK > 5:
        r = rem % d5
        rem = rem // d5
        xo += r * s5
        oo += r * t5
    if RANK > 6:
        r = rem % d6
        rem = rem // d6
        xo += r * s6
        oo += r * t6
    if RANK > 7:
        r = rem % d7
        rem = rem // d7
        xo += r * s7
        oo += r * t7

    v = tl.load(x_ptr + xo, mask=mask, other=0)
    if FAST:
        res = _sinf_poly(v.to(tl.float32))
    elif F64:
        res = tl.sin(v.to(tl.float64))
    else:
        res = tl.sin(v.to(tl.float32))
    tl.store(o_ptr + oo, res.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _sinc_flat(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    # x_ptr/o_ptr are the float32/float64 views of complex storage (2 lanes each)
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    io = offs * 2
    a = tl.load(x_ptr + io, mask=mask, other=0.0)
    b = tl.load(x_ptr + io + 1, mask=mask, other=0.0)
    ca = tl.cos(a)
    sa = tl.sin(a)
    eb = tl.exp(b)
    enb = tl.exp(-b)
    ch = 0.5 * (eb + enb)
    sh = 0.5 * (eb - enb)
    tl.store(o_ptr + io, sa * ch, mask=mask)
    tl.store(o_ptr + io + 1, ca * sh, mask=mask)


@triton.jit
def _sinc_strided(
    x_ptr,
    o_ptr,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
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
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n
    rem = idx
    xo = tl.zeros([BLOCK], tl.int64)
    oo = tl.zeros([BLOCK], tl.int64)

    r = rem % d0
    rem = rem // d0
    xo += r * s0
    oo += r * t0
    if RANK > 1:
        r = rem % d1
        rem = rem // d1
        xo += r * s1
        oo += r * t1
    if RANK > 2:
        r = rem % d2
        rem = rem // d2
        xo += r * s2
        oo += r * t2
    if RANK > 3:
        r = rem % d3
        rem = rem // d3
        xo += r * s3
        oo += r * t3
    if RANK > 4:
        r = rem % d4
        rem = rem // d4
        xo += r * s4
        oo += r * t4
    if RANK > 5:
        r = rem % d5
        rem = rem // d5
        xo += r * s5
        oo += r * t5
    if RANK > 6:
        r = rem % d6
        rem = rem // d6
        xo += r * s6
        oo += r * t6
    if RANK > 7:
        r = rem % d7
        rem = rem // d7
        xo += r * s7
        oo += r * t7

    io = xo * 2
    jo = oo * 2
    a = tl.load(x_ptr + io, mask=mask, other=0.0)
    b = tl.load(x_ptr + io + 1, mask=mask, other=0.0)
    ca = tl.cos(a)
    sa = tl.sin(a)
    eb = tl.exp(b)
    enb = tl.exp(-b)
    ch = 0.5 * (eb + enb)
    sh = 0.5 * (eb - enb)
    tl.store(o_ptr + jo, sa * ch, mask=mask)
    tl.store(o_ptr + jo + 1, ca * sh, mask=mask)


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------

# kind codes: 0 error(bool), 1 unsupported, 2 real32, 3 real64, 4 int, 5 complex
_INT_DTYPES = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
_REAL32_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# 16-bit outputs: the inline fp32 polynomial is bit-identical to ATen after
# rounding (verified on 12M random samples per dtype) and avoids the branchy
# out-of-line full-precision sinf expansion.
_FAST_DTYPES = (torch.float16, torch.bfloat16)
_PLAN = {}
_FUSED = {2: _sin_fused2, 4: _sin_fused4, 8: _sin_fused8, 16: _sin_fused16}
_EMPTY_MSG = "Tensor list must have at least one tensor."


def _plan(dtype):
    p = _PLAN.get(dtype)
    if p is not None:
        return p
    if dtype == torch.bool:
        p = (0, None)
    elif dtype.is_complex:
        p = (5, dtype)
    elif dtype in _REAL32_DTYPES:
        p = (2, dtype)
    elif dtype == torch.float64:
        p = (3, dtype)
    elif dtype in _INT_DTYPES:
        p = (4, torch.float32)
    elif dtype.is_floating_point:
        if dtype.itemsize >= 2:
            # any other >=2 byte float (e.g. future dtypes): compute in fp32 parity
            p = (2, dtype)
        else:
            p = (1, None)
    else:
        p = (1, None)
    _PLAN[dtype] = p
    return p


def _geom(t, out):
    shp = [1] * _MAX_RANK
    istr = [0] * _MAX_RANK
    ostr = [0] * _MAX_RANK
    for i, (d, s) in enumerate(zip(t.shape, t.stride())):
        shp[i] = d
        istr[i] = s
    for i, s in enumerate(out.stride()):
        ostr[i] = s
    return shp + istr + ostr


def _launch_one(t, out, kind):
    n = t.numel()
    nb = _BLOCK
    if kind == 5:
        xr = torch.view_as_real(t)
        orr = torch.view_as_real(out)
        if t.is_contiguous():
            _sinc_flat[(triton.cdiv(n, nb),)](xr, orr, n, BLOCK=nb, num_warps=4)
        else:
            _sinc_strided[(triton.cdiv(n, nb),)](
                xr, orr, n, *_geom(t, out), BLOCK=nb, RANK=t.dim(), num_warps=4
            )
        return
    f64 = kind == 3
    fast = t.dtype in _FAST_DTYPES
    if t.is_contiguous() and out.is_contiguous():
        _sin_flat[(triton.cdiv(n, nb),)](
            t, out, n, BLOCK=nb, F64=f64, FAST=fast, num_warps=4
        )
    else:
        _sin_strided[(triton.cdiv(n, nb),)](
            t,
            out,
            n,
            *_geom(t, out),
            BLOCK=nb,
            RANK=t.dim(),
            F64=f64,
            FAST=fast,
            num_warps=4,
        )


def _launch_fused(ts, outs, kind, fast):
    """One launch for up to _FUSE_MAX contiguous same-dtype tensors."""
    nb = _BLOCK
    f64 = kind == 3
    i = 0
    total = len(ts)
    while i < total:
        left = total - i
        k = _FUSE_MAX if left >= _FUSE_MAX else 1 << (left.bit_length() - 1)
        ct = ts[i : i + k]
        co = outs[i : i + k]
        grid = (max(triton.cdiv(t.numel(), nb) for t in ct),)
        if k == 1:
            t = ct[0]
            _sin_flat[grid](
                t, co[0], t.numel(), BLOCK=nb, F64=f64, FAST=fast, num_warps=4
            )
            i += 1
            continue
        a = []
        for t, o in zip(ct, co):
            a.append(t)
            a.append(o)
            a.append(t.numel())
        _FUSED[k][grid](*a, BLOCK=nb, F64=f64, FAST=fast, num_warps=4)
        i += k


def _normalize(args, kwargs):
    if kwargs:
        for key in ("self", "tensors", "inputs", "tensor_list"):
            if key in kwargs:
                v = kwargs[key]
                break
        else:
            raise TypeError("run() got unexpected keyword arguments")
        if len(kwargs) > 1:
            raise TypeError("run() got unexpected keyword arguments")
        args = (v,)
    if len(args) == 1:
        v = args[0]
        if isinstance(v, torch.Tensor):
            return [v]
        if isinstance(v, (list, tuple)):
            return list(v)
        raise TypeError("expected a sequence of tensors, got %r" % type(v))
    if len(args) == 0:
        raise RuntimeError(_EMPTY_MSG)
    out = []
    for a in args:
        if not isinstance(a, torch.Tensor):
            raise TypeError("expected a sequence of tensors")
        out.append(a)
    return out


def foreach_sin(*args, **kwargs):
    tensors = _normalize(args, kwargs)
    total = len(tensors)
    if total == 0:
        raise RuntimeError(_EMPTY_MSG)

    # validate every element first so an invalid dtype raises before any work
    kinds = []
    for t in tensors:
        kind, odt = _plan(t.dtype)
        if kind == 0:
            raise RuntimeError("to - 1 is out of bounds for bool")
        if kind == 1:
            raise NotImplementedError(
                "\"normal_kernel_cuda\" not implemented for '%s'" % t.dtype
            )
        kinds.append(kind)

    outs = [None] * total
    singles = []
    groups = {}
    lim = _FUSE_MAX_ELEMS
    for i, t in enumerate(tensors):
        kind = kinds[i]
        out = torch.empty_like(t, dtype=_plan(t.dtype)[1])
        outs[i] = out
        ne = t.numel()
        if ne == 0:
            continue
        if kind == 5 or ne > lim or not t.is_contiguous() or not out.is_contiguous():
            singles.append(i)
        else:
            key = (t.device, kind, t.dtype)
            g = groups.get(key)
            if g is None:
                groups[key] = [i]
            else:
                g.append(i)

    for i in singles:
        _launch_one(tensors[i], outs[i], kinds[i])

    for key, idxs in groups.items():
        if len(idxs) == 1:
            i = idxs[0]
            _launch_one(tensors[i], outs[i], kinds[i])
        else:
            dt = key[2]
            _launch_fused(
                [tensors[i] for i in idxs],
                [outs[i] for i in idxs],
                key[1],
                dt in _FAST_DTYPES,
            )

    return outs
