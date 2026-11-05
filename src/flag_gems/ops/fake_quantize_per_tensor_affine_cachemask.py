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
"""Triton implementation of aten::fake_quantize_per_tensor_affine_cachemask.

Numerics (verified bit-exactly against aten on the S60 target):

    inv    = float32(1.0 / float32(scale))          # fp32 division of fp32 operands
    y      = float32(x) * inv                       # correctly rounded fp32 product
    q      = rint(y) + zero_point                   # round-half-to-even, fp32
    mask   = (q >= quant_min) & (q <= quant_max)    # computed before clamping
    qc     = min(max(q, quant_min), quant_max)      # fp32 min/max
    out    = cast_to_x_dtype((qc - zero_point) * float32(scale))

Target-specific facts that drive the code below:

* Rounding is done with the classic ``(|y| + 2**23) - 2**23`` trick instead of
  ``libdevice.rint``: on this backend the extern rint is exact but scalarises
  badly and makes the whole kernel ~5x slower (measured 187us -> 42us on 1M
  fp32).  The trick must be evaluated on |y| with the sign restored, because
  the fused ``(y + 2**23) - 2**23`` chain this backend emits is *not*
  round-half-to-even for negative operands (p = -22.398 -> -22.5).
* aten divides nowhere: it multiplies by the fp32 reciprocal, so ``x / scale``
  is not bit-equivalent and must not be used.

No int64 arithmetic is used (the S60 backend does not support int64).
"""

import logging
import math
import struct

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_MAGIC_C = tl.constexpr(8388608.0)
_BLOCK_MIN = 1024
_BLOCK_MAX = 131072
_BLOCK_MAX_HUGE = 262144
_STRIDED_BLOCK = 4096
_STRIDED_WARPS = 2
_MAX_RANK = 8
_MAX_GRID_X = 65535


def _pick_block(n):
    """One program per tensor up to 128K elements: launch cost is 11.2us +
    0.46us per program on this vendor, so few large tiles win by a wide margin."""
    block = 1 << max(10, (n - 1).bit_length())
    if block > _BLOCK_MAX:
        block = _BLOCK_MAX
    if (n + block - 1) // block > 256:
        block = _BLOCK_MAX_HUGE
    return block


def _pick_warps(block, grid):
    if block <= 16384:
        return 2
    if block <= 32768:
        return 4
    if grid <= 2:
        return 8
    if grid <= 4:
        return 4
    return 2


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

_SCALAR_NAMES = {
    torch.bool: "bool",
    torch.uint8: "byte",
    torch.int8: "char",
    torch.int16: "short",
    torch.int32: "int",
    torch.int64: "long",
    torch.float16: "half",
    torch.bfloat16: "bfloat16",
    torch.float32: "float",
    torch.float64: "double",
    torch.complex64: "complex<float>",
    torch.complex128: "complex<double>",
}


def _f32(v):
    return struct.unpack("f", struct.pack("f", float(v)))[0]


def _split_scale(scale):
    """Return (float32 scale, float32 reciprocal) exactly as aten does."""
    s = _f32(scale)
    if s == 0.0:
        return s, math.copysign(math.inf, s)
    return s, _f32(1.0 / s)


def _scalar_name(dtype):
    return _SCALAR_NAMES.get(dtype, str(dtype))


@libentry()
@triton.jit(do_not_specialize=["n", "gx", "zp", "qmin", "qmax"])
def fake_quantize_per_tensor_affine_cachemask_kernel(
    x_ptr,
    y_ptr,
    m_ptr,
    n,
    gx,
    scale,
    inv_scale,
    zp,
    qmin,
    qmax,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0) + tl.program_id(1) * gx
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    act = offs < n
    v = tl.load(x_ptr + offs, mask=act, other=0.0).to(tl.float32)
    sf = scale.to(tl.float32)
    invf = inv_scale.to(tl.float32)
    y = v * invf
    # round-half-to-even on the magnitude (|y| + 2**23 lands in a binade whose
    # ulp is one).  Done on |y| because this backend's fused chain is *not*
    # ties-to-even for negative operands.  libdevice.rint is exact but ~5x
    # slower end-to-end here, so the arithmetic form is used.
    ap = tl.abs(y)
    r = (ap + _MAGIC_C) - _MAGIC_C
    r = tl.where(y < 0.0, -r, r)
    q = r + zp
    inr = (q >= qmin) & (q <= qmax)
    qc = tl.minimum(tl.maximum(q, qmin), qmax)
    o = (qc - zp) * sf
    tl.store(y_ptr + offs, o.to(y_ptr.dtype.element_ty), mask=act)
    tl.store(m_ptr + offs, inr, mask=act)


@libentry()
@triton.jit(
    do_not_specialize=[
        "n",
        "gx",
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
    ]
)
def _fq_strided_kernel(
    x_ptr,
    y_ptr,
    m_ptr,
    n,
    gx,
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
    inv_scale,
    zp,
    qmin,
    qmax,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0) + tl.program_id(1) * gx
    offs = pid * BLOCK + tl.arange(0, BLOCK)
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
    sf = scale.to(tl.float32)
    invf = inv_scale.to(tl.float32)
    y = v * invf
    ap = tl.abs(y)
    r = (ap + _MAGIC_C) - _MAGIC_C
    r = tl.where(y < 0.0, -r, r)
    q = r + zp
    inr = (q >= qmin) & (q <= qmax)
    qc = tl.minimum(tl.maximum(q, qmin), qmax)
    o = (qc - zp) * sf
    tl.store(y_ptr + yoff, o.to(y_ptr.dtype.element_ty), mask=act)
    tl.store(m_ptr + moff, inr, mask=act)


def _grid(n, block):
    blocks = (n + block - 1) // block
    # Grow the tile (never the y dimension) so every launched program is busy.
    while block < 262144 and blocks > _MAX_GRID_X:
        block *= 2
        blocks = (n + block - 1) // block
    gx = blocks if blocks <= _MAX_GRID_X else _MAX_GRID_X
    gy = (blocks + gx - 1) // gx
    return (gx, gy), gx, block


def _launch(x, out, mask, scale, inv_scale, zp, qmin, qmax):
    n = x.numel()
    if n == 0:
        return
    with torch_device_fn.device(x.device):
        if x.is_contiguous() and out.is_contiguous() and mask.is_contiguous():
            block = _pick_block(n)
            nblocks = (n + block - 1) // block
            grid, gx, block = _grid(n, block)
            fake_quantize_per_tensor_affine_cachemask_kernel[grid](
                x,
                out,
                mask,
                n,
                gx,
                scale,
                inv_scale,
                zp,
                qmin,
                qmax,
                BLOCK=block,
                num_warps=_pick_warps(block, nblocks),
            )
            return

        rank = x.dim()
        shape = x.shape
        kept = [d for d in range(rank) if shape[d] != 1]
        if len(kept) > _MAX_RANK:
            # Degenerate fallback: peel the leading dimension (zero-copy views).
            for i in range(shape[0]):
                _launch(x[i], out[i], mask[i], scale, inv_scale, zp, qmin, qmax)
            return

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

        grid, gx, block = _grid(n, _STRIDED_BLOCK)
        _fq_strided_kernel[grid](
            x,
            out,
            mask,
            n,
            gx,
            *sizes,
            *divs,
            *xs,
            *ys,
            *ms,
            scale,
            inv_scale,
            zp,
            qmin,
            qmax,
            RANK=r,
            BLOCK=block,
            num_warps=_STRIDED_WARPS,
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
    scale_f, inv_scale = _split_scale(scale)
    _launch(x, o0, o1, scale_f, inv_scale, zp, qmin, qmax)
    return o0, o1


def fake_quantize_per_tensor_affine_cachemask(
    input, scale, zero_point, quant_min, quant_max
):
    logger.debug("GEMS FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK")
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
    logger.debug("GEMS FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_OUT")
    return _run(input, scale, zero_point, quant_min, quant_max, out0, out1)
