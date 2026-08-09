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

# Kunlunxin (XPU) override of gcd / gcd_ / gcd_out.
#
# Perf root cause & fix (see harness/solution/performance/gcd_out_xpu7_20260816.md
# and gcd_out_xpu1_20260816.md):
#   1. Euclidean modulo loop now runs entirely in int32 lane math (int16 `%`
#      is emulated on this backend); values fit: 16-bit magnitudes <= 32768,
#      int32 abs(INT32_MIN) wraps to itself - both handled by keeping
#      INT_MIN lanes on their native value (torch bit-exact C-modulo sign
#      chain, e.g. gcd(INT16_MIN, 6) == -2, gcd(81, INT16_MIN) == -1).
#   2. Iteration caps: 24 for int8/int16 (worst 23 steps, fib(23)=28657),
#      48 for int32 (worst 46), 96 for int64.
#   3. num_warps=8 is the dominant XPU lever (BLOCK=128 / nw1 -> nw8:
#      ~25ms -> ~4.3ms at 16.7M elements, measured idle-window); plain
#      launch (no buffer_size_limit / isCloseVectorization); BLOCK>=256
#      does not compile (uni_sram 2KB pass limit for loop-carried values).
#   4. `out=` path writes directly into the provided tensor when it is
#      dtype/shape/contiguity compatible with the promoted result,
#      skipping the extra copy pass.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.gcd import _materialize_inputs

logger = logging.getLogger(__name__)

# Euclidean worst case (consecutive Fibonacci inputs): ~1.44*log2(max)+C.
# 8-bit:   max steps 10 (fib(11)=89 <= 127, fib(12)=144 > 127)
# 16-bit:  max steps 23 (fib(23)=28657 <= 32767) -> 24
# 32-bit:  max steps 46 (fib(46)=1.8e9 <= 2^31-1) -> 48
# 64-bit:  max steps ~93 -> 96
_ITERS_32 = 48
_ITERS_64 = 96
_ITERS = {torch.int8: 24, torch.int16: 24, torch.int32: _ITERS_32}


@triton.jit
def gcd_kernel_32(
    x_ptr, y_ptr, out_ptr, n_elements, ITERS: tl.constexpr, MINV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    xi = x.to(tl.int32)
    yi = y.to(tl.int32)
    # All gcd math in int32 (full-speed XPU lane ops). INT_MIN lanes keep
    # their native value (see module docstring); others run on abs.
    a0 = tl.where(xi == MINV, xi, tl.abs(xi))
    b0 = tl.where(yi == MINV, yi, tl.abs(yi))
    for _ in range(ITERS):
        nz = b0 != 0
        bb = tl.where(nz, b0, 1)
        r = a0 % bb
        a0 = tl.where(nz, b0, a0)
        b0 = tl.where(nz, r, b0)
    tl.store(out_ptr + offsets, a0.to(out_ptr.type.element_ty), mask=mask)


@triton.jit
def gcd_kernel_64(
    x_ptr, y_ptr, out_ptr, n_elements, ITERS: tl.constexpr, MINV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    # INT64_MIN lanes keep their signed value; everything else runs on
    # the abs value (C-modulo sign propagation, torch bit-exact).
    a0 = tl.where(x == MINV, x, tl.abs(x))
    b0 = tl.where(y == MINV, y, tl.abs(y))
    for _ in range(ITERS):
        nz = b0 != 0
        bb = tl.where(nz, b0, 1)
        r = a0 % bb
        a0 = tl.where(nz, b0, a0)
        b0 = tl.where(nz, r, b0)
    tl.store(out_ptr + offsets, a0.to(out_ptr.type.element_ty), mask=mask)


def _kernel_meta(dtype):
    if dtype in (torch.int8, torch.int16, torch.int32):
        minv = -(1 << 31) if dtype == torch.int32 else torch.iinfo(dtype).min
        return gcd_kernel_32, _ITERS[dtype], minv, 128, 8
    if dtype == torch.int64:
        return gcd_kernel_64, _ITERS_64, -(1 << 63), 128, 8
    raise TypeError(f"unsupported dtype for gcd: {dtype}")


def _launch_gcd(lhs, rhs, out):
    numel = out.numel()
    if numel == 0:
        return out
    kernel, iters, minv, block, num_warps = _kernel_meta(out.dtype)
    grid = (triton.cdiv(numel, block),)
    kernel[grid](
        lhs, rhs, out, numel, ITERS=iters, MINV=minv, BLOCK=block,
        num_warps=num_warps,
    )
    return out


def gcd(self, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN GCD")
    if self.numel() == 0 or other.numel() == 0:
        promoted_dtype = torch.promote_types(self.dtype, other.dtype)
        shape = torch.broadcast_shapes(self.shape, other.shape)
        result = torch.empty(shape, dtype=promoted_dtype, device=self.device)
        if out is None:
            return result
        out.copy_(result)
        return out
    lhs, rhs, promoted_dtype = _materialize_inputs(self, other)
    if (
        out is not None
        and out.dtype == promoted_dtype
        and out.shape == lhs.shape
        and out.is_contiguous()
        and out.device == lhs.device
    ):
        # Direct-write into the provided out tensor: skips the extra
        # copy pass (the copy itself is a slow BLOCK-128 kernel here).
        _launch_gcd(lhs.reshape(-1), rhs.reshape(-1), out.reshape(-1))
        return out
    result = torch.empty_like(lhs, dtype=promoted_dtype)
    _launch_gcd(lhs.reshape(-1), rhs.reshape(-1), result.reshape(-1))
    result = result.view(lhs.shape)
    if out is None:
        return result
    out.copy_(result)
    return out


def gcd_out(lhs, rhs, *, out=None):
    if out is None:
        return gcd(lhs, rhs)
    return gcd(lhs, rhs, out=out)


def gcd_(A, B):
    lhs, rhs, promoted_dtype = _materialize_inputs(A, B)
    if A.is_contiguous() and A.dtype == promoted_dtype:
        # In-place direct write: the kernel loads x/y block-wide before
        # storing, so aliasing A as both source and destination is safe,
        # and the extra copy pass (a slow BLOCK-128 flag-triton copy_,
        # ~equal to the kernel itself at 16.7M elements) is eliminated.
        _launch_gcd(lhs.reshape(-1), rhs.reshape(-1), A.reshape(-1))
        return A
    flat_out = torch.empty(lhs.numel(), dtype=promoted_dtype, device=A.device)
    _launch_gcd(lhs.reshape(-1), rhs.reshape(-1), flat_out)
    # Use the native copy engine, not the gems copy_ override (dispatch loop).
    torch.ops.aten._copy_from(flat_out.view(A.shape), A, False)
    return A