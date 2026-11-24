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

"""Out-of-place ``uniform`` (uniform_ semantics) for the MUSA target.

``run(self, from_=0.0, to=1.0, *, generator=None)`` returns a new tensor of the
same dtype/shape whose entries are uniform in ``[from_, to)``.

The fill is a single streaming store pass.  One counter-based 32-bit hash per
*element* (fp32 output) or per *pair* of elements (16-bit outputs: two 16-bit
uniforms packed from one mixed word and re-interleaved so each thread still
owns one wide contiguous store) feeds an fp32 affine map that is clamped to the
representable ``[lo, hi]`` window of the output dtype and stored with a single
rounding cast.  The uniform sample comes from the classic mantissa trick: the
hash bits are placed in the mantissa of a constant exponent, giving a value in
``[1, 2)`` whose affine image needs only one fused multiply-add.
"""

import logging
import math
import struct

import torch
import triton
import triton.language as tl

from flag_gems.ops.uniform import uniform as default_uniform
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

# ---------------------------------------------------------------------------
# launch configuration
# ---------------------------------------------------------------------------
_BLOCK = 1024
_NUM_WARPS = 4

# Per-call changing stream base.  Odd, so Triton's "divisible by 16" argument
# specialisation never flips and the kernel keeps a single compiled variant.
_MULT = 2654435761
_counter = 0


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
@triton.jit
def _mix32(v):
    """splitmix/murmur style 32-bit integer mixing (int32-safe constants)."""
    v = v ^ ((v.to(tl.uint32) >> 16).to(tl.int32, bitcast=True))
    v = v * (-2048144777)  # 0x85EBCA77
    v = v ^ ((v.to(tl.uint32) >> 13).to(tl.int32, bitcast=True))
    v = v * (-1028477387)  # 0xC2B2AE3D
    v = v ^ ((v.to(tl.uint32) >> 16).to(tl.int32, bitcast=True))
    return v


@triton.jit
def _uniform_fill(out_ptr, N, seed, lo, hi, scale, bias, BLOCK: tl.constexpr):
    """One 23-bit uniform per element (fp32 output)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mi = _mix32(offs ^ seed)
    # mantissa trick: hash bits -> 1.xxxxxx in [1, 2); scale/bias carry the map.
    bits = (mi.to(tl.uint32) >> 9).to(tl.int32, bitcast=True) | 0x3F800000
    f = bits.to(tl.float32, bitcast=True)
    # Clamp into the representable window [lo, hi] of the output dtype: the
    # scaled value can round onto the exclusive upper bound, and can round
    # below the inclusive lower bound for low-precision dtypes.
    val = tl.minimum(tl.maximum(f * scale + bias, lo), hi)
    tl.store(out_ptr + offs, val.to(out_ptr.dtype.element_ty), mask=offs < N)


@triton.jit
def _uniform_fill_packed(out_ptr, N, seed, lo, hi, scale, bias, BLOCK: tl.constexpr):
    """Two 16-bit uniforms per mixed word (fp16/bf16 output).

    Half as many hashes as stored elements; the two uniforms land on adjacent
    output elements so every thread still stores one wide contiguous range.
    """
    HALF: tl.constexpr = BLOCK // 2
    pid = tl.program_id(0)
    j = pid * HALF + tl.arange(0, HALF)
    u = _mix32(j ^ seed).to(tl.uint32)
    b0 = (u >> 9).to(tl.int32, bitcast=True) | 0x3F800000
    b1 = ((u << 7).to(tl.int32, bitcast=True) & 0x7FFF80) | 0x3F800000
    f0 = b0.to(tl.float32, bitcast=True)
    f1 = b1.to(tl.float32, bitcast=True)
    out_ty: tl.constexpr = out_ptr.dtype.element_ty
    v0 = tl.minimum(tl.maximum(f0 * scale + bias, lo), hi).to(out_ty)
    v1 = tl.minimum(tl.maximum(f1 * scale + bias, lo), hi).to(out_ty)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.interleave(v0, v1), mask=offs < N)


# ---------------------------------------------------------------------------
# host side dtype-interval helpers (pure Python scalar arithmetic)
# ---------------------------------------------------------------------------
def _f32_round(x):
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.inf if x > 0 else -math.inf


def _f16_round(x):
    try:
        return struct.unpack("<e", struct.pack("<e", x))[0]
    except OverflowError:
        return math.inf if x > 0 else -math.inf


def _bf16_round(x):
    if x != x or x == math.inf or x == -math.inf:
        return x
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    rb = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
    if rb >= 0x7F80:
        return math.inf if x > 0 else -math.inf
    return struct.unpack("<f", struct.pack("<I", rb << 16))[0]


def _to_key(bits, w):
    full = 1 << w
    sign = full >> 1
    if bits & sign:
        return (full - 1) - bits
    return bits + sign


def _from_key(key, w):
    full = 1 << w
    sign = full >> 1
    if key >= sign:
        return key - sign
    return (full - 1) - key


def _bit_ops(dtype):
    if dtype == torch.float16:
        return (
            16,
            lambda v: struct.unpack("<H", struct.pack("<e", v))[0],
            lambda b: struct.unpack("<e", struct.pack("<H", b))[0],
            _f16_round,
        )
    if dtype == torch.bfloat16:
        return (
            16,
            lambda v: (struct.unpack("<I", struct.pack("<f", v))[0] >> 16) & 0xFFFF,
            lambda b: struct.unpack("<f", struct.pack("<I", b << 16))[0],
            _bf16_round,
        )
    return (
        32,
        lambda v: struct.unpack("<I", struct.pack("<f", v))[0],
        lambda b: struct.unpack("<f", struct.pack("<I", b))[0],
        _f32_round,
    )


def _next_dt(bits, w, step):
    return _from_key(_to_key(bits, w) + step, w)


def _compute_limits(dtype, frm, to):
    """Largest/smallest output-dtype values that satisfy ``[from_, to)``."""
    w, to_bits, from_bits, rnd = _bit_ops(dtype)
    r = rnd(frm)
    if math.isfinite(r) and not (r < frm):
        lo = r
    elif math.isfinite(r):
        lo = from_bits(_next_dt(to_bits(r), w, 1))
    else:
        lo = r
    r2 = rnd(to)
    if math.isfinite(r2) and r2 < to:
        hi = r2
    else:
        hi = from_bits(_next_dt(to_bits(r2), w, -1))
    if not (hi < to):
        hi = from_bits(_next_dt(to_bits(hi), w, -1))
    if not (lo <= hi):
        hi = lo
    if dtype == torch.float32 and hi != 0.0 and abs(hi) < 1.1754943508222875e-38:
        # The exact predecessor is a denormal; pick a normal value that is
        # still strictly inside the interval instead of relying on denormal
        # comparison behaviour in the kernel.
        alt = _f32_round(to - (to - frm) * 5.9604644775390625e-08)
        if alt < to:
            hi = alt
    return lo, hi


_LIMIT_CACHE = {}


def _limits(dtype, frm, to):
    key = (dtype, frm, to)
    got = _LIMIT_CACHE.get(key)
    if got is None:
        got = _compute_limits(dtype, frm, to)
        _LIMIT_CACHE[key] = got
    return got


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernel."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def uniform(self, from_=0.0, to=1.0, *, generator=None):
    logger.debug("GEMS_MTHREADS UNIFORM")

    if not _use_triton_kernel(self):
        return default_uniform(self, from_=from_, to=to, generator=generator)

    frm = float(from_)
    end = float(to)
    if frm > end:
        raise RuntimeError(
            "uniform_ expects to return a [from, to) range, but found "
            f"from={frm:g} > to={end:g}"
        )
    out = torch.empty_like(self)
    n = out.numel()
    if n == 0:
        return out
    with torch_device_fn.device(self.device):
        _launch_uniform(self, out, n, frm, end)
    return out


def _launch_uniform(self, out, n, frm, end):
    global _counter
    _counter += 1
    seed = ((_counter * _MULT) & 0x7FFFFFFE) | 1
    lo, hi = _limits(out.dtype, frm, end)
    scale = end - frm
    bias = frm - scale
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    if out.dtype in (torch.float16, torch.bfloat16):
        _uniform_fill_packed[grid](
            out, n, seed, lo, hi, scale, bias, BLOCK=_BLOCK, num_warps=_NUM_WARPS
        )
    else:
        _uniform_fill[grid](
            out, n, seed, lo, hi, scale, bias, BLOCK=_BLOCK, num_warps=_NUM_WARPS
        )
