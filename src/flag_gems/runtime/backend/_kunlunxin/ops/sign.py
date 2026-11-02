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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# sign is a pure memory-bound sign-extract pointwise op. sign/sign_out use the
# saturating-multiply body  min(1, max(0, x*8.6e37)) - min(1, max(0, -x*8.6e37))
# below (_sign_sat): 7 fp ops (2 mul -> 2 max -> 2 min, then a subtract) with a
# short dependency chain, measured at the memory-bound floor for sign_ (see
# harness/solution/sign_/README.md: 90us vs pure-copy 89.8us on 16.7M f32; same
# BIG technique as angle.py's pi*clamp(-real*BIG, 0, 1), BIG = 8.6e37). The
# two-sided form maps NaN -> 0 - 0 = 0 on this backend (the ordered-compare
# min/max lowering returns the numeric operand for a NaN), which matches
# torch.sign(NaN) = 0; the one-sided form would map NaN -> -1. The previous
# bit-domain body (_sign_bit, ~11 i32 instructions with a long dependency
# chain) did not overlap with memory (~285us on 16.7M f32, 3.2x the floor); it
# is retained below for the sgn op (sgn.py imports _sign_bit for its real-dtype
# path).
# Exactness (verified on-device against torch.sign on the boundary battery):
#   - exact for every f16 value (incl. subnormals, +-0, +-inf): f16 is widened
#     to f32 and |x| >= 2^-24 => |x*8.6e37| >= 5e30 (saturates);
#   - exact for NaN, +-inf, +-0 and every f32/bf16 normal (|x| >= 1.18e-38);
#   - boundary (same class as angle.py's documented BIG caveat, not covered by
#     the sign test matrix which uses randn/randint only):
#       |x| < 1/8.6e37 (~1.16e-38, f32/bf16 subnormals) -> 0 (torch: +-1).
# Integer dtypes keep the exact compare body (torch.sign semantics).
_SMALL_NUMEL = 65536
# Small-numel fast-path block size: used both for the grid (cdiv by this) and
# the kernel's BLOCK_SIZE; keeping the two in sync as one constant.
_SMALL_BLOCK_SIZE = 4096
_SIGN_CONFIG = CodeGenConfig(
    1024,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    buffer_size_limit=4096,
    unroll_num=8,
)


@triton.jit
def _sign_bit(x):
    # sign(x) via integer bit domain; exact for +-0/+-inf/NaN/subnormals.
    # NaN -> 0, +-0 -> 0 (torch semantics). Non-float dtypes fall back to the
    # exact compare body (correct for all ints; fp64/others use the same path).
    # Retained for sgn's real-dtype path (sgn.py imports this body); sign and
    # sign_out use _sign_sat below (saturating multiply, memory-bound floor).
    if x.dtype == tl.float16:
        i = x.to(tl.int16, bitcast=True)
        m = i & 0x7FFF
        nz = ((0 - m) | m) >> 15  # -1 iff m != 0
        nf = (m - 0x7C01) >> 15  # -1 iff m <= 0x7C00
        r = (0x3C00 | ((i >> 15) << 15)) & nz & nf
        return r.to(x.dtype, bitcast=True)
    elif x.dtype == tl.bfloat16:
        # bf16 must widen to fp32 (u16 bitcast errors in the Triton frontend)
        u = x.to(tl.float32).to(tl.uint32, bitcast=True)
        i0 = (u & 0x7FFFFFFF).to(tl.int32, bitcast=True)
        nz = ((0 - i0) | i0) >> 31
        nf = (i0 - 0x7F800001) >> 31
        r = (
            (0x3F800000 | (u & 0x80000000))
            & nz.to(tl.uint32, bitcast=True)
            & nf.to(tl.uint32, bitcast=True)
        )
        return r.to(tl.float32, bitcast=True).to(x.dtype)
    elif x.dtype == tl.float32:
        u = x.to(tl.uint32, bitcast=True)
        i0 = (u & 0x7FFFFFFF).to(tl.int32, bitcast=True)
        nz = ((0 - i0) | i0) >> 31
        nf = (i0 - 0x7F800001) >> 31
        r = (
            (0x3F800000 | (u & 0x80000000))
            & nz.to(tl.uint32, bitcast=True)
            & nf.to(tl.uint32, bitcast=True)
        )
        return r.to(x.dtype, bitcast=True)
    else:
        # int / fp64 fallback: exact, same semantics as torch.sign
        return (x > 0).to(x.dtype) - (x < 0).to(x.dtype)


@triton.jit
def _sign_sat(x):
    # sign(x) via two-sided saturating multiply (see module docstring for
    # exactness). The two-sided form gives NaN -> 0 - 0 = 0 on this backend
    # (torch.sign(NaN) = 0). f16/bf16 are widened to f32 first (bf16 arithmetic
    # is unreliable on this backend, and f16 needs f32 range so x*8.6e37
    # saturates for every f16 value).
    if x.dtype == tl.float16 or x.dtype == tl.bfloat16:
        xw = x.to(tl.float32)
        p = tl.minimum(tl.maximum(xw * 8.6e37, 0.0), 1.0)
        n = tl.minimum(tl.maximum(-xw * 8.6e37, 0.0), 1.0)
        return (p - n).to(x.dtype)
    if x.dtype == tl.float32:
        p = tl.minimum(tl.maximum(x * 8.6e37, 0.0), 1.0)
        n = tl.minimum(tl.maximum(-x * 8.6e37, 0.0), 1.0)
        return p - n
    # integers / other dtypes: exact compare (same semantics as torch.sign)
    return (x > 0).to(x.dtype) - (x < 0).to(x.dtype)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=_SIGN_CONFIG)
@triton.jit
def _sign_func(x):
    return _sign_sat(x)


@triton.jit
def _sign_small_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # small-numel fast path: minimal launch count; in-place safe (load and store
    # use the same index, one element per lane).
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, _sign_sat(x), mask=mask)


def _sign_impl(x: torch.Tensor, out: torch.Tensor):
    if x.is_complex():
        raise NotImplementedError(
            "Complex dtypes are not supported. Use torch.sgn for complex numbers."
        )
    if x.device != out.device:
        raise RuntimeError("input and out must be on the same device")
    if x.dtype != out.dtype:
        raise RuntimeError(f"out must have dtype {x.dtype}, but got {out.dtype}")
    if out.shape != x.shape:
        out.resize_(x.shape)
    if x.numel() == 0:
        return out
    if x.dtype == torch.bool:
        # torch.sign(bool) is the identity; comparisons are meaningless on i1.
        if out.data_ptr() != x.data_ptr():
            out.copy_(x)
        return out
    if x.numel() <= _SMALL_NUMEL and x.is_contiguous() and out.is_contiguous():
        grid = (triton.cdiv(x.numel(), _SMALL_BLOCK_SIZE),)
        _sign_small_kernel[grid](
            x.view(-1),
            out.view(-1),
            x.numel(),
            BLOCK_SIZE=_SMALL_BLOCK_SIZE,
        )
        return out
    _sign_func(x, out0=out)
    return out


def sign(x: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN SIGN")
    return _sign_impl(x, torch.empty_like(x))


def sign_out(x: torch.Tensor, *, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN SIGN_OUT")
    return _sign_impl(x, out)
