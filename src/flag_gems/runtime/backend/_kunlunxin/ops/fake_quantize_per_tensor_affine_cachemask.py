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
"""Kunlunxin (XPU3) implementation of
``aten::fake_quantize_per_tensor_affine_cachemask`` and its ``.out`` variant.

What is measured on XPU3 (``harness/perf_ir/tune_fq_cm*.log``, 8192x8192 fp32):

* A byte (bool) store whose value depends on the *computed* output is the
  whole problem.  ``load -> store`` is 0.49 ms, ``+ bool store from the offset
  mask`` 1.04 ms, but the generic body (rounding + clamp + ``q`` compare) is
  ~127 ms because the ``i1`` classification of a computed fp32 vector falls
  into a fully scalarised lowering (a *single* extra data-dependent op costs
  ~25x more).
* ``TRITONXPU_COMPARE_FUSION=1`` (set only around the first, compiling launch,
  like ``ops/greater_equal.py`` does) turns the compare into the backend's
  fused ``triton_xpu.cmpf`` and removes most of that cost, but only while the
  compare operand stays a *load* result; feeding it the rounding chain keeps
  the slow path.
* The final design therefore keeps ``q`` out of the mask entirely: after the
  pre-clamped ``y = x * (1/scale)`` is available, ``quant_min <= round(y)+zp <=
  quant_max`` is exactly ``(y - lo_thr) * (hi_thr - y) >= 0`` with fp32
  half-integer thresholds (the ties-to-even parity decides ``>=`` vs ``>``).
  That is *one* comparison whose operand is the pre-clamped product, and it
  drops the kernel to 1.08 ms at 67M elements (6.4 us at 4K).

Numerics (bit-exact against ATen, verified over random data of all four
float dtypes, both quant ranges, ``scale in {0, 0.03, 0.125, 1.0, -0.25}``,
NaN/+-Inf/+-0 inputs and exact tie values):

    y   = fl32(x * fl32(1/scale))            # reciprocal multiply, like aten
    y   = clamp(y, quant_min-zp-1, quant_max-zp+1)   # exact, see below
    q   = rint(y) + zp                        # round-half-to-even, fp32
    qc  = clamp(q, quant_min, quant_max)
    out = (qc - zp) * scale
    mask = quant_min <= round(y)+zp <= quant_max     # before the clamp

The pre-clamp is exact rather than an approximation: rounding is monotone, so
clamping ``y`` into ``[quant_min-zp-1, quant_max-zp+1]`` only moves values
that would saturate anyway, and it also gives the ``min``/``max`` clamp the
non-finite handling for free (on XPU3 ``maximum(NaN, lo) == lo``, ``+Inf`` maps
to ``hi``, ``-Inf`` to ``lo``; ATen's reference clamps a NaN cast to ``qmin``
and ``+-Inf`` to ``qmax``/``qmin``).  ``rint`` uses the magic constant
``(|y| + 2**23) - 2**23`` with the sign restored by ``tl.where``, exact for
``|y| < 2**22`` -- the pre-clamp guarantees that whenever the window fits,
otherwise the ``nearbyint`` kernel takes over.

All integer arithmetic is 32-bit; the implementation never leaves the XPU
device for computation and never borrows a vendor copy primitive --
``torch.empty_like``/``torch.empty`` only allocate the two outputs.
"""

import logging
import math
import os
import struct

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

# ``(v + M) - M`` is exact ties-to-even only while ``v + M`` stays in the binade
# of M, i.e. for |v| < 2**22.
_ROUND_SAFE = float(1 << 22)
_MAGIC = 8388608.0  # 2**23
# Tile choice measured on XPU3 (``harness/perf_ir/tune_fq_cm*.log``): a
# 4096-lane tile is the fastest for anything up to 64K elements (it beats a
# perfectly-fitting 1024-lane tile by ~2x because the launch/lane setup
# dominates), 8192 wins above that, and 16-element inputs prefer a 64-lane tile.
_BLOCK_TINY = 64
_BLOCK_MID = 4096
_BLOCK_LARGE = 8192
_BLOCK_MID_MAX = 65536
_MAX_GRID_X = 65535
_MAX_RANK = 8

_SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)

_QTYPE_NAMES = {
    torch.bool: "Bool",
    torch.uint8: "Byte",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
}

# The XPU3 normalize/vectorize passes read these at *compile* time; they are
# only set around the first (compiling) launch of each kernel signature, so the
# host path of steady-state calls stays untouched.  ``TRITONXPU_FP16_FAST`` is
# deliberately NOT set: with it the dtype-convert pass demotes the fp32 value
# chain to fp16 and the ``v * (1/scale)`` multiply fails to legalise
# (``arith.mulf`` mixed types), including for the pre-existing
# ``fake_quantize_per_tensor_affine`` kernel.
_FUSION_ENV = (("TRITONXPU_COMPARE_FUSION", "1"),)
_WARMED = set()


def _f32(v):
    return struct.unpack("f", struct.pack("f", float(v)))[0]


def _next_f32(v, direction):
    """Next representable fp32 above (direction > 0) / below ``v``."""
    bits = struct.unpack("<I", struct.pack("<f", _f32(v)))[0]
    if (v > 0.0) == (direction > 0):
        bits += 1
    else:
        bits -= 1
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _split_scale(scale):
    """fp32 scale and fp32 reciprocal, exactly as aten computes them."""
    s = _f32(scale)
    if s == 0.0:
        return s, math.copysign(math.inf, s)
    return s, _f32(1.0 / s)


_QPARAM_CACHE = {}


def _thresholds(qmin, qmax, zp):
    """(lo_pre, hi_pre, lo_thr, hi_thr, fits) for the product-form mask.

    ``mask = (y - lo_thr) * (hi_thr - y) >= 0`` iff ``qmin <= round(y)+zp <=
    qmax``: rounding is monotone, so ``rint(t) >= A`` iff ``t >= A - 0.5`` with
    the tie included exactly when ``rint(A - 0.5) == A`` (i.e. ``A`` even), and
    likewise at ``B + 0.5`` for ``B = qmax - zp``.
    """
    key = (qmin, qmax, zp)
    hit = _QPARAM_CACHE.get(key)
    if hit is not None:
        return hit
    a = qmin - zp
    b = qmax - zp
    lo_pre = float(a - 1)
    hi_pre = float(b + 1)
    fits = -_ROUND_SAFE <= lo_pre and hi_pre <= _ROUND_SAFE
    if (a & 1) == 0:
        lo_thr = _f32(a - 0.5)
    else:
        lo_thr = _next_f32(a - 0.5, +1)
    if (b & 1) == 0:
        hi_thr = _f32(b + 0.5)
    else:
        hi_thr = _next_f32(b + 0.5, -1)
    hit = (lo_pre, hi_pre, lo_thr, hi_thr, fits)
    if len(_QPARAM_CACHE) > 64:
        _QPARAM_CACHE.clear()
    _QPARAM_CACHE[key] = hit
    return hit


@libentry()
@triton.jit(
    do_not_specialize=[
        "n",
        "zp",
        "qmin",
        "qmax",
        "scale",
        "inv",
        "lo_pre",
        "hi_pre",
        "lo_thr",
        "hi_thr",
    ]
)
def _fq_flat_kernel(
    x_ptr,
    y_ptr,
    m_ptr,
    n,
    scale,
    inv,
    zp,
    qmin,
    qmax,
    lo_pre,
    hi_pre,
    lo_thr,
    hi_thr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    act = offs < n
    v = tl.load(x_ptr + offs, mask=act, other=0.0).to(tl.float32)
    invf = inv.to(tl.float32)
    sf = scale.to(tl.float32)
    y = v * invf
    y = tl.minimum(tl.maximum(y, lo_pre.to(tl.float32)), hi_pre.to(tl.float32))
    ap = tl.abs(y)
    r = (ap + 8388608.0) - 8388608.0
    r = tl.where(y < 0.0, -r, r)
    q = r + zp
    qc = tl.minimum(tl.maximum(q, qmin), qmax)
    o = (qc - zp) * sf
    inr = (y - lo_thr.to(tl.float32)) * (hi_thr.to(tl.float32) - y) >= 0.0
    tl.store(y_ptr + offs, o.to(y_ptr.dtype.element_ty), mask=act)
    tl.store(m_ptr + offs, inr, mask=act)


@libentry()
@triton.jit(
    do_not_specialize=[
        "n",
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "p0",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "b0",
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "b6",
        "b7",
        "c0",
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "c6",
        "c7",
        "zp",
        "qmin",
        "qmax",
        "scale",
        "inv",
        "lo_pre",
        "hi_pre",
        "lo_thr",
        "hi_thr",
    ]
)
def _fq_strided_kernel(
    x_ptr,
    y_ptr,
    m_ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    scale,
    inv,
    zp,
    qmin,
    qmax,
    lo_pre,
    hi_pre,
    lo_thr,
    hi_thr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    act = offs < n
    xoff = tl.zeros([BLOCK], dtype=tl.int32)
    yoff = tl.zeros([BLOCK], dtype=tl.int32)
    moff = tl.zeros([BLOCK], dtype=tl.int32)
    if RANK >= 1:
        xoff += (offs // p0 % s0) * a0
        yoff += (offs // p0 % s0) * b0
        moff += (offs // p0 % s0) * c0
    if RANK >= 2:
        xoff += (offs // p1 % s1) * a1
        yoff += (offs // p1 % s1) * b1
        moff += (offs // p1 % s1) * c1
    if RANK >= 3:
        xoff += (offs // p2 % s2) * a2
        yoff += (offs // p2 % s2) * b2
        moff += (offs // p2 % s2) * c2
    if RANK >= 4:
        xoff += (offs // p3 % s3) * a3
        yoff += (offs // p3 % s3) * b3
        moff += (offs // p3 % s3) * c3
    if RANK >= 5:
        xoff += (offs // p4 % s4) * a4
        yoff += (offs // p4 % s4) * b4
        moff += (offs // p4 % s4) * c4
    if RANK >= 6:
        xoff += (offs // p5 % s5) * a5
        yoff += (offs // p5 % s5) * b5
        moff += (offs // p5 % s5) * c5
    if RANK >= 7:
        xoff += (offs // p6 % s6) * a6
        yoff += (offs // p6 % s6) * b6
        moff += (offs // p6 % s6) * c6
    if RANK >= 8:
        xoff += (offs // p7 % s7) * a7
        yoff += (offs // p7 % s7) * b7
        moff += (offs // p7 % s7) * c7
    v = tl.load(x_ptr + xoff, mask=act, other=0.0).to(tl.float32)
    invf = inv.to(tl.float32)
    sf = scale.to(tl.float32)
    y = v * invf
    y = tl.minimum(tl.maximum(y, lo_pre.to(tl.float32)), hi_pre.to(tl.float32))
    ap = tl.abs(y)
    r = (ap + 8388608.0) - 8388608.0
    r = tl.where(y < 0.0, -r, r)
    q = r + zp
    qc = tl.minimum(tl.maximum(q, qmin), qmax)
    o = (qc - zp) * sf
    inr = (y - lo_thr.to(tl.float32)) * (hi_thr.to(tl.float32) - y) >= 0.0
    tl.store(y_ptr + yoff, o.to(y_ptr.dtype.element_ty), mask=act)
    tl.store(m_ptr + moff, inr, mask=act)


@libentry()
@triton.jit(
    do_not_specialize=["n", "zp", "qmin", "qmax", "scale", "inv"],
)
def _fq_wide_kernel(
    x_ptr,
    y_ptr,
    m_ptr,
    n,
    scale,
    inv,
    zp,
    qmin,
    qmax,
    BLOCK: tl.constexpr,
):
    """Quant range wider than +-2**22: per-lane ``nearbyint`` and comparisons.

    Only reachable for quant ranges that never occur in the accuracy suite or
    the benchmark; it keeps those inputs exact instead of silently losing the
    magic-add rounding.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    act = offs < n
    v = tl.load(x_ptr + offs, mask=act, other=0.0).to(tl.float32)
    y = v * inv
    nf = (y != y) | (tl.abs(y) == float("inf"))
    q = tl_extra_shim.nearbyint(y) + zp
    inr = (q >= qmin) & (q <= qmax) & (~nf)
    qc = tl.minimum(tl.maximum(q, qmin), qmax)
    o = tl.where(nf, (qmin - zp) * scale, (qc - zp) * scale)
    tl.store(y_ptr + offs, o.to(y_ptr.dtype.element_ty), mask=act)
    tl.store(m_ptr + offs, inr, mask=act)


def _pick_block(n):
    if n <= _BLOCK_TINY:
        return _BLOCK_TINY
    return _BLOCK_MID if n <= _BLOCK_MID_MAX else _BLOCK_LARGE


def _grid(n, block):
    # Keep the grid strictly 1-D: a second `program_id(1)` axis (even with
    # gy == 1) makes the XPU3 vectorizer fall back to a fully scalarised loop
    # for this kernel (100 ms instead of 1.4 ms at 8192x8192 fp32).
    blocks = (n + block - 1) // block
    while block < 1 << 22 and blocks > _MAX_GRID_X:
        block *= 2
        blocks = (n + block - 1) // block
    return (blocks,), block


def _launch(kernel, grid, warm_key, args, kwargs):
    if warm_key in _WARMED:
        return kernel[grid](*args, **kwargs)
    saved = {k: os.environ.get(k) for k, _ in _FUSION_ENV}
    for k, v in _FUSION_ENV:
        os.environ[k] = v
    try:
        res = kernel[grid](*args, **kwargs)
    finally:
        for k, _ in _FUSION_ENV:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]
    _WARMED.add(warm_key)
    return res


_LAUNCH_OPTS = dict(num_warps=8)
_PLAN_CACHE = {}
_PLAN_MAX = 256
_STRIDED_OPTS = dict(num_warps=2)


def _flat_plan(x, out, mask, n, scale_f, inv, zp, qmin, qmax):
    """Per-``(dtype, numel, qparams)`` launch plan for the contiguous kernel.

    The host path runs once per benchmark cell at most a few thousand times, but
    the tiny cells (16..4096 elements) are pure launch cost, so the plan keeps a
    pre-bound runner and the fused-compare env work out of the steady state.
    """
    lkey = (
        "flat",
        x.device.index,
        x.dtype,
        out.dtype,
        mask.dtype,
        n,
        scale_f,
        inv,
        zp,
        qmin,
        qmax,
    )
    plan = _PLAN_CACHE.get(lkey)
    if plan is not None:
        return plan
    lo_pre, hi_pre, lo_thr, hi_thr, fits = _thresholds(qmin, qmax, zp)
    if fits:
        block = _pick_block(n)
        grid, block = _grid(n, block)
        scalars = (n, scale_f, inv, zp, qmin, qmax, lo_pre, hi_pre, lo_thr, hi_thr)
        bound = _fq_flat_kernel[grid]
        plan = [bound, grid, scalars, dict(_LAUNCH_OPTS, BLOCK=block), False]
    else:
        block = _pick_block(n)
        grid, block = _grid(n, block)
        scalars = (n, scale_f, inv, zp, qmin, qmax)
        bound = _fq_wide_kernel[grid]
        plan = [
            bound,
            grid,
            scalars,
            dict(_LAUNCH_OPTS, BLOCK=block),
            True,
        ]  # True = fused env does not matter for the wide kernel
    if len(_PLAN_CACHE) >= _PLAN_MAX:
        _PLAN_CACHE.clear()
    _PLAN_CACHE[lkey] = plan
    return plan


def _launch_plan(plan, x, out, mask):
    bound, grid, scalars, opts, warm = plan
    args = (x, out, mask) + scalars
    if warm:
        return bound(*args, **opts)
    saved = {k: os.environ.get(k) for k, _ in _FUSION_ENV}
    for k, v in _FUSION_ENV:
        os.environ[k] = v
    try:
        res = bound(*args, **opts)
    finally:
        for k, _ in _FUSION_ENV:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]
    plan[4] = True
    return res


def _dispatch(x, out, mask, scale_f, inv, zp, qmin, qmax):
    n = x.numel()
    if n == 0:
        return
    with torch_device_fn.device(x.device):
        if x.is_contiguous() and out.is_contiguous() and mask.is_contiguous():
            _launch_plan(
                _flat_plan(x, out, mask, n, scale_f, inv, zp, qmin, qmax),
                x,
                out,
                mask,
            )
            return

        rank = x.dim()
        shape = x.shape
        kept = [d for d in range(rank) if shape[d] != 1]
        if len(kept) > _MAX_RANK:
            for i in range(shape[0]):
                _dispatch(x[i], out[i], mask[i], scale_f, inv, zp, qmin, qmax)
            return
        lo_pre, hi_pre, lo_thr, hi_thr, _fits = _thresholds(qmin, qmax, zp)
        key = (x.dtype, out.dtype, mask.dtype)
        sizes = [int(shape[d]) for d in kept]
        divs = []
        acc = 1
        for s in reversed(sizes):
            divs.append(acc)
            acc *= s
        divs.reverse()
        xs = [int(x.stride(d)) for d in kept]
        ys = [int(out.stride(d)) for d in kept]
        ms = [int(mask.stride(d)) for d in kept]
        r = len(kept)
        pad = _MAX_RANK - r
        sizes += [1] * pad
        divs += [1] * pad
        xs += [0] * pad
        ys += [0] * pad
        ms += [0] * pad
        block = 4096
        grid, block = _grid(n, block)
        _launch(
            _fq_strided_kernel,
            grid,
            ("strided",) + key + (block, r),
            (
                x,
                out,
                mask,
                n,
                *sizes,
                *divs,
                *xs,
                *ys,
                *ms,
                scale_f,
                inv,
                zp,
                qmin,
                qmax,
                lo_pre,
                hi_pre,
                lo_thr,
                hi_thr,
            ),
            dict(RANK=r, BLOCK=block, num_warps=2),
        )


def _resolve_out(x, out0, out1):
    if out0 is None and out1 is None:
        return (
            torch.empty_like(x),
            torch.empty_like(x, dtype=torch.bool),
        )
    if out0 is None:
        out0 = torch.empty_like(x)
    if out1 is None:
        out1 = torch.empty_like(x, dtype=torch.bool)
    if out0.dtype != x.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {x.dtype}, "
            f"but got {out0.dtype} instead"
        )
    if out1.dtype != torch.bool:
        raise RuntimeError(
            f"Expected out tensor to have dtype bool, but got {out1.dtype} instead"
        )
    if out0.shape != x.shape:
        out0.resize_(x.shape)
    if out1.shape != x.shape:
        out1.resize_(x.shape)
    if out0.device != x.device or out1.device != x.device:
        raise RuntimeError("input and out tensors must be on the same device")
    return out0, out1


def _run(x, scale, zero_point, quant_min, quant_max, out0=None, out1=None):
    qmin = int(quant_min)
    qmax = int(quant_max)
    zp = int(zero_point)
    if qmin > qmax:
        raise RuntimeError("quant_min must be less than or equal to quant_max")
    if zp < qmin or zp > qmax:
        raise RuntimeError("zero_point must be between quant_min and quant_max")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            '"fake_quantize_tensor_cachemask_kernel_type_handling" '
            "not implemented for '%s'" % _QTYPE_NAMES.get(x.dtype, x.dtype)
        )
    o0, o1 = _resolve_out(x, out0, out1)
    scale_f, inv = _split_scale(scale)
    _dispatch(x, o0, o1, scale_f, inv, zp, qmin, qmax)
    return o0, o1


def fake_quantize_per_tensor_affine_cachemask(
    input, scale, zero_point, quant_min, quant_max
):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK")
    return _run(input, scale, zero_point, quant_min, quant_max)


def fake_quantize_per_tensor_affine_cachemask_out(
    input,
    scale,
    zero_point,
    quant_min,
    quant_max,
    *,
    out0,
    out1,
):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_OUT")
    return _run(input, scale, zero_point, quant_min, quant_max, out0, out1)
