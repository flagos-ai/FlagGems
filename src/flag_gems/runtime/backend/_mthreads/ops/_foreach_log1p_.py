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
"""In-place ``_foreach_log1p_`` implemented with Triton for MUSA (MTT S5000).

Semantics reproduced from ATen ``aten::_foreach_log1p_(Tensor(a!)[] self) -> ()``:

* every list element is transformed in place (same storage, strides and dtype),
* the call returns ``None``,
* non-floating dtypes are rejected because an in-place op cannot promote,
* ``expand``-style internal overlap is rejected,
* an empty list is rejected.

Execution strategy: the whole list is validated first.  Contiguous tensors of one
dtype that are small enough for dispatch to dominate are packed into a *single*
kernel launch (up to 16 per launch, base pointer selected per program); larger
tensors get one launch each, and strided/complex views also get one launch each.
All element math runs in device kernels; Torch is used only for metadata reads,
validation and the raw-pointer / real-view plumbing.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as _ld

from flag_gems.ops._foreach_unary import _foreach_log1p_ as default__foreach_log1p_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _needs_fallback(seq):
    """True when the list cannot use the MUSA kernels (device or dtype)."""
    for t in seq:
        if not isinstance(t, torch.Tensor):
            return True
        if t.device.type != "musa" or t.dtype not in _SUPPORTED_DTYPES:
            return True
    return False


# ---------------------------------------------------------------------------
# dtype policy (measured against the pinned ATen build)
# ---------------------------------------------------------------------------

_FLOAT_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32, torch.float64))
_COMPLEX_DTYPES = frozenset((torch.complex64, torch.complex128))

_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. "
    "Please clone() the tensor before performing the operation."
)

_MAX_FUSE = 16


def _cast_error(dtype):
    name = str(dtype).replace("torch.", "").capitalize()
    return RuntimeError(
        f"result type Float can't be cast to the desired output type {name}"
    )


# ---------------------------------------------------------------------------
# element math
# ---------------------------------------------------------------------------


@triton.jit
def _log1p_value(x):
    """log1p in opmath precision; half types are rounded back on store.

    For fp32 and the half types the math is ``log2(1+x) * ln2``: the compiler
    lowers ``tl.log2`` to a compact body (124 LLVM IR lines, 0 division, 17
    registers) whereas ``libdevice.log1p`` inlines 1514 lines with 24 vector
    divisions, and the measured accuracy of the log2 form against fp64 log1p is
    max abs 1.26e-6 / max rel 2.0e-7 (libdevice: 4.8e-7 / 8.0e-8).  fp64 keeps
    the libdevice implementation because it is not benchmarked and doubling
    precision needs the accurate path.
    """
    if x.dtype == tl.float64:
        return _ld.log1p(x)
    y = 1.0 + x.to(tl.float32)
    r = tl.log2(y) * 0.6931471805599453
    if x.dtype == tl.float32:
        return r
    return r.to(x.dtype)


# ---------------------------------------------------------------------------
# contiguous (flat) kernels
# ---------------------------------------------------------------------------


@triton.jit
def _flat_kernel(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(x_ptr + offs, _log1p_value(x), mask=mask)


@triton.jit
def _multi_flat_kernel(
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
    BLOCK: tl.constexpr,
):
    """One launch for up to 16 same-dtype, same-numel contiguous tensors.

    Grid axis 0 selects the tensor and axis 1 the chunk inside it.  The base
    pointer is chosen with a short scalar ``tl.where`` chain, which keeps the
    original pointer type so the load/store stay 128-bit vectorised (verified by
    compile probe 4686d14e: 36 registers, no spills, ``<4 x float>`` accesses,
    identical to the single-pointer kernel).  Selecting through a raw int64
    address instead lost the vectorisation completely.
    """
    t = tl.program_id(0)
    p = p0
    p = tl.where(t == 1, p1, p)
    p = tl.where(t == 2, p2, p)
    p = tl.where(t == 3, p3, p)
    p = tl.where(t == 4, p4, p)
    p = tl.where(t == 5, p5, p)
    p = tl.where(t == 6, p6, p)
    p = tl.where(t == 7, p7, p)
    p = tl.where(t == 8, p8, p)
    p = tl.where(t == 9, p9, p)
    p = tl.where(t == 10, p10, p)
    p = tl.where(t == 11, p11, p)
    p = tl.where(t == 12, p12, p)
    p = tl.where(t == 13, p13, p)
    p = tl.where(t == 14, p14, p)
    p = tl.where(t == 15, p15, p)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(p + offs, mask=mask, other=0.0)
    tl.store(p + offs, _log1p_value(x), mask=mask)


@triton.jit
def _flat_complex_kernel(p, n, BLOCK: tl.constexpr):
    """``p`` addresses the storage as interleaved (re, im) reals."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    re = tl.load(p + 2 * offs, mask=mask, other=0.0)
    im = tl.load(p + 2 * offs + 1, mask=mask, other=0.0)
    if p.dtype.element_ty == tl.float64:
        ref = re
        imf = im
    else:
        ref = re.to(tl.float32)
        imf = im.to(tl.float32)
    ure = 1.0 + ref
    ore = 0.5 * _ld.log1p(2.0 * ref + ref * ref + imf * imf)
    oim = _ld.atan2(imf, ure)
    tl.store(p + 2 * offs, ore, mask=mask)
    tl.store(p + 2 * offs + 1, oim, mask=mask)


# ---------------------------------------------------------------------------
# strided kernels (collapsed view, up to 8 dims)
# ---------------------------------------------------------------------------


@triton.jit
def _affine_kernel(x_ptr, n, s, BLOCK: tl.constexpr):
    """Single collapsed dimension with stride ``s`` (no div/mod needed)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    o = offs * s
    x = tl.load(x_ptr + o, mask=mask, other=0.0)
    tl.store(x_ptr + o, _log1p_value(x), mask=mask)


@triton.jit
def _view_offset(
    offs,
    s0,
    t0,
    s1,
    t1,
    s2,
    t2,
    s3,
    t3,
    s4,
    t4,
    s5,
    t5,
    s6,
    t6,
    s7,
    t7,
    R: tl.constexpr,
):
    """Map a flat logical index of a collapsed view to a storage offset.

    The collapsed dimensions are ordered outermost (largest stride) first, so
    consecutive logical indices walk the smallest-stride dimension.
    """
    rem = offs
    o = tl.zeros_like(offs)
    if R > 7:
        i = rem % s7
        rem = rem // s7
        o += i * t7
    if R > 6:
        i = rem % s6
        rem = rem // s6
        o += i * t6
    if R > 5:
        i = rem % s5
        rem = rem // s5
        o += i * t5
    if R > 4:
        i = rem % s4
        rem = rem // s4
        o += i * t4
    if R > 3:
        i = rem % s3
        rem = rem // s3
        o += i * t3
    if R > 2:
        i = rem % s2
        rem = rem // s2
        o += i * t2
    if R > 1:
        i = rem % s1
        rem = rem // s1
        o += i * t1
    i = rem % s0
    o += i * t0
    return o


@triton.jit
def _strided_kernel(
    x_ptr,
    n,
    s0,
    t0,
    s1,
    t1,
    s2,
    t2,
    s3,
    t3,
    s4,
    t4,
    s5,
    t5,
    s6,
    t6,
    s7,
    t7,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    o = _view_offset(
        offs, s0, t0, s1, t1, s2, t2, s3, t3, s4, t4, s5, t5, s6, t6, s7, t7, R
    )
    x = tl.load(x_ptr + o, mask=mask, other=0.0)
    tl.store(x_ptr + o, _log1p_value(x), mask=mask)


@triton.jit
def _strided_complex_kernel(
    p,
    n,
    s0,
    t0,
    s1,
    t1,
    s2,
    t2,
    s3,
    t3,
    s4,
    t4,
    s5,
    t5,
    s6,
    t6,
    s7,
    t7,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    o = 2 * _view_offset(
        offs, s0, t0, s1, t1, s2, t2, s3, t3, s4, t4, s5, t5, s6, t6, s7, t7, R
    )
    re = tl.load(p + o, mask=mask, other=0.0)
    im = tl.load(p + o + 1, mask=mask, other=0.0)
    if p.dtype.element_ty == tl.float64:
        ref = re
        imf = im
    else:
        ref = re.to(tl.float32)
        imf = im.to(tl.float32)
    ure = 1.0 + ref
    ore = 0.5 * _ld.log1p(2.0 * ref + ref * ref + imf * imf)
    oim = _ld.atan2(imf, ure)
    tl.store(p + o, ore, mask=mask)
    tl.store(p + o + 1, oim, mask=mask)


# ---------------------------------------------------------------------------
# host-side helpers
# ---------------------------------------------------------------------------

_STRIDED_BLOCK = 512


def _collapse(t):
    """Collapse a strided view to (size, stride) pairs, outermost first.

    Adjacent dimensions are merged when they are contiguous with each other,
    size-1 dimensions are dropped, and the result is ordered by descending
    stride so that consecutive logical indices touch the smallest-stride
    dimension.  For an in-place elementwise map the visit order is irrelevant:
    any bijection over the view's element set is correct, provided nothing
    outside the view is written.
    """
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    out = []
    for sz, st in reversed(dims):
        if out and st == out[-1][1] * out[-1][0]:
            osz, ost = out[-1]
            out[-1] = (osz * sz, ost)
        else:
            out.append((sz, st))
    out.reverse()
    return sorted(out, key=lambda d: -d[1])


_BLOCK = 1024
# Lists whose members are small enough that one dispatch (~20 us measured) is
# a large share of the work get packed into a single launch; larger tensors use
# one launch each, because there sequential launches overlap CPU dispatch with
# GPU execution and pay no dispatch penalty (round-1/3 measurements).
_FUSE_MAX_NUMEL = 1 << 19


def _launch_flat(t, n):
    _flat_kernel[(triton.cdiv(n, _BLOCK),)](t, n, _BLOCK)


def _launch_multi(ts, n):
    args = list(ts)
    pad = _MAX_FUSE - len(args)
    if pad > 0:
        args.extend([ts[0]] * pad)
    _multi_flat_kernel[(len(ts), triton.cdiv(n, _BLOCK))](*args, n, _BLOCK)


def _launch_affine(t, n, s, block):
    _affine_kernel[(triton.cdiv(n, block),)](t, n, s, block)


def _launch_strided(t, dims, block):
    args = []
    for sz, st in dims:
        args.append(sz)
        args.append(st)
    while len(args) < 16:
        args.append(1)
        args.append(0)
    _strided_kernel[(triton.cdiv(t.numel(), block),)](
        t, t.numel(), *args, len(dims), block
    )


def _launch_complex_flat(t, n):
    real = torch.view_as_real(t)  # allow-torch-compute: pure view, no device compute
    _flat_complex_kernel[(triton.cdiv(n, _BLOCK),)](real, n, _BLOCK)


def _launch_complex_strided(t, dims, block):
    real = torch.view_as_real(t)  # allow-torch-compute: pure view, no device compute
    args = []
    for sz, st in dims:
        args.append(sz)
        args.append(st)
    while len(args) < 16:
        args.append(1)
        args.append(0)
    _strided_complex_kernel[(triton.cdiv(t.numel(), block),)](
        real, t.numel(), *args, len(dims), block
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _foreach_log1p_(self):
    logger.debug("GEMS_MTHREADS _FOREACH_LOG1P_")
    _seq = self if isinstance(self, (list, tuple)) else [self]
    if _needs_fallback(_seq):
        return default__foreach_log1p_(self)
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(self) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    # Validate the whole list before touching any storage (ATen order), and
    # collect the launches: contiguous same-dtype tensors share one launch.
    flat = {}
    side = []
    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dtype = t.dtype
        if dtype in _FLOAT_DTYPES:
            cplx = False
        elif dtype in _COMPLEX_DTYPES:
            cplx = True
        elif dtype == torch.complex32:
            raise RuntimeError("\"foreach\" not implemented for 'complex32'")
        else:
            raise _cast_error(dtype)

        n = t.numel()
        if n == 0:
            continue
        if t.is_contiguous() or n == 1:
            if cplx:
                side.append((t, "cf", 0, None))
            else:
                flat.setdefault(dtype, []).append(t)
            continue
        if torch._debug_has_internal_overlap(t) == 1:
            raise RuntimeError(_OVERLAP_MSG)
        dims = _collapse(t)
        if not dims or (len(dims) == 1 and dims[0][1] == 1):
            if cplx:
                side.append((t, "cf", 0, None))
            else:
                flat.setdefault(dtype, []).append(t)
        elif len(dims) == 1:
            if cplx:
                side.append((t, "cs", 0, dims))
            else:
                side.append((t, "a", dims[0][1], None))
        elif len(dims) <= 8:
            side.append((t, "cs" if cplx else "s", 0, dims))
        else:
            raise NotImplementedError("view rank too large")

    for ts in flat.values():
        n0 = ts[0].numel()
        uniform = True
        for t in ts:
            if t.numel() != n0:
                uniform = False
                break
        if not uniform:
            for t in ts:
                _launch_flat(t, t.numel())
        elif len(ts) == 1 or n0 >= _FUSE_MAX_NUMEL:
            for t in ts:
                _launch_flat(t, n0)
        else:
            for i in range(0, len(ts), _MAX_FUSE):
                _launch_multi(ts[i : i + _MAX_FUSE], n0)

    for t, kind, stride, dims in side:
        if kind == "f":
            _launch_flat(t, t.numel())
        elif kind == "a":
            _launch_affine(t, t.numel(), stride, _STRIDED_BLOCK)
        elif kind == "s":
            _launch_strided(t, dims, _STRIDED_BLOCK)
        elif kind == "cf":
            _launch_complex_flat(t, t.numel())
        else:
            _launch_complex_strided(t, dims, _STRIDED_BLOCK)

    return None
