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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)

# GCU400 flat-kernel override of the pointwise_dynamic path for trunc/trunc_.
# The generated block-pointer kernel (tile 65536 / grid <= 48 / num_warps 4)
# plus the extern `tl_extra_shim.trunc` body leave ~4x on the table:
#   512M fp16 10039 -> 969us (flat + int-domain body), 2.5x faster than the
#   torch native in-place kernel (2407us);  16.7M fp16 315 -> 38.5us;  the
#   copy floor is 669us @512M fp16, so the 7-int-op body sits 1.45x above it.
# The integer body clears the mantissa bits below 2^0 directly on the fp32
# bit pattern: e = biased exponent, s = mantissa bits to drop; |x| < 1
# (s >= 24) clears everything but the sign bit, |x| >= 2^23 / inf / nan
# (s <= 0) keeps the input untouched.  It is bit-exact on the whole fp32
# domain (sign of zero preserved: trunc(-0.5) = -0.0), unlike the extern
# shim which needs a sign-fixup select chain.
# Integer inputs are integral already: trunc_ becomes a host no-op and trunc
# a flat copy (int64 via the int32 word view, byte-exact under the fake-Long
# physical layout).  Small (< 2048), non-contiguous, fp64 and > int32-safe
# inputs keep the generic pointwise path.
NUM_SIPS = 24
_TINY_FLOOR = 2048
_SINGLE_CTA_N = 8192
_INT32_SAFE_LIMIT = 0x7FFFFFFF
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _trunc_flat_kernel(X, Out, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        m = off < N
        x = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
        u = x.to(tl.int32, bitcast=True)
        e = (u >> 23) & 0xFF
        # mantissa bits below 2^0 for 1 <= |x| < 2^23; all but the sign bit
        # for |x| < 1 (s >= 24 <=> e <= 126); nothing for |x| >= 2^23 / inf
        # / nan (s <= 0 after the max).
        s = tl.maximum(150 - e, 0)
        s = tl.where(s >= 24, 31, s)
        keep = -1 << s
        r = (u & keep).to(tl.float32, bitcast=True)
        tl.store(Out + off, r.to(Out.dtype.element_ty), mask=m)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _copy_flat_kernel(X, Out, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        m = off < N
        v = tl.load(X + off, mask=m, other=0)
        tl.store(Out + off, v, mask=m)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def trunc_func(x):
    x_fp32 = x.to(tl.float32)
    t = tl_extra_shim.trunc(x_fp32)
    # Truncation is towards zero: rebuild zero results with the input sign bit,
    # otherwise -0.5 / -0.0 silently become +0.0 on GCU400.
    xi = x_fp32.to(tl.int32, bitcast=True)
    ti = t.to(tl.int32, bitcast=True)
    ti = tl.where(t == 0.0, xi & (-2147483648), ti)
    t = ti.to(tl.float32, bitcast=True)
    return t.to(x.dtype)


def _choose_block(N, itemsize):
    # do_bench(median) sweep on GCU400 (in-place and out-of-place agree):
    #   tiny: one exact next_pow2 tile, num_warps 4;
    #   mid:  fill the 48-CTA grid (np2 of cdiv(n, 48)), 2B 65536 / 4B 32768;
    #   huge: 2B 131072 above 2^28 (65536 below), 4B 65536; num_warps 2.
    if N <= _SINGLE_CTA_N:
        return max(64, triton.next_power_of_2(N))
    fill = triton.next_power_of_2(max(1, triton.cdiv(N, NUM_SIPS * 2)))
    if itemsize >= 4:
        return min(max(fill, 4096), 32768) if N <= (1 << 24) else 65536
    if N > (1 << 28):
        return 131072
    return min(max(fill, 4096), 65536)


def _launch_flat(kernel, src, dst, N, itemsize):
    BLOCK = _choose_block(N, itemsize)
    grid_size = min(triton.cdiv(N, BLOCK), NUM_SIPS * 2)
    num_warps = 4 if N <= _SINGLE_CTA_N else 2
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared nodes);
    # retrying the launch recompiles and succeeds (threshold / rad2deg /
    # atan2 precedents for the same toolchain family).
    for attempt in range(3):
        try:
            with torch_device_fn.device(src.device):
                kernel[(grid_size,)](src, dst, N, BLOCK=BLOCK, num_warps=num_warps)
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "trunc flat kernel launch failed (attempt %d): %s", attempt + 1, exc
            )


def _flat_ok(A):
    return (
        A.is_contiguous()
        and A.dtype in _FLOAT_DTYPES
        and _TINY_FLOOR <= A.numel() <= _INT32_SAFE_LIMIT - 65536
    )


def trunc(A):
    logger.debug("GEMS_ENFLAME TRUNC")
    n = A.numel()
    if _flat_ok(A):
        out = torch.empty_like(A)
        _launch_flat(_trunc_flat_kernel, A, out, n, A.element_size())
        return out
    if A.is_contiguous() and A.dtype in _INT_DTYPES and 0 < n:
        out = torch.empty_like(A)
        try:
            if A.dtype == torch.int64:
                src = A.reshape(-1).view(torch.int32)
                dst = out.reshape(-1).view(torch.int32)
                words = src.numel()
            else:
                src, dst, words = A, out, n
            if words <= _INT32_SAFE_LIMIT - 65536:
                _launch_flat(_copy_flat_kernel, src, dst, words, src.element_size())
                return out
        except Exception as exc:  # pragma: no cover - exotic view failures
            logger.warning("trunc int fast path failed, falling back: %s", exc)
    return trunc_func(A)


def trunc_(A):
    logger.debug("GEMS_ENFLAME TRUNC_")
    n = A.numel()
    if _flat_ok(A):
        _launch_flat(_trunc_flat_kernel, A, A, n, A.element_size())
        return A
    if A.is_contiguous() and A.dtype in _INT_DTYPES:
        # Integral inputs are already truncated: the in-place op is a no-op.
        return A
    trunc_func(A, out0=A)
    return A
