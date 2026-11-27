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


@triton.jit
def _trunc_scalar(x0, IS_FP16: tl.constexpr, IS_BF16: tl.constexpr):
    if IS_FP16 or IS_BF16:
        u = x0.to(tl.int16, bitcast=True)
        zero = u ^ u
        if IS_BF16:
            # bf16: 8 exponent bits at offset 7, 7 mantissa bits.  Bits to
            # drop: max(127 + 7 - e, 0); |x| < 1 (s >= 8) clears all but the
            # sign bit; e >= 134 (|x| >= 2^7) and inf/NaN keep the input.
            e = (u >> 7) & 0xFF
            s = tl.maximum(134 - e, zero)
            s = tl.where(s >= zero + 8, zero + 15, s)
        else:
            # fp16: 5 exponent bits at offset 10, 10 mantissa bits.  Bits to
            # drop: max(15 + 10 - e, 0); |x| < 1 (s >= 11) clears all but the
            # sign bit; e >= 25 (|x| >= 2^10) and inf/NaN keep the input.
            e = (u >> 10) & 0x1F
            s = tl.maximum(25 - e, zero)
            s = tl.where(s >= zero + 11, zero + 15, s)
        return ((u >> s) << s).to(x0.dtype, bitcast=True)
    else:
        u = x0.to(tl.int32, bitcast=True)
        e = (u >> 23) & 0xFF
        # mantissa bits below 2^0 for 1 <= |x| < 2^23; all but the sign bit
        # for |x| < 1 (s >= 24); nothing for |x| >= 2^23 / inf / nan.
        s = tl.maximum(150 - e, 0)
        s = tl.where(s >= 24, 31, s)
        keep = -1 << s
        return (u & keep).to(tl.float32, bitcast=True)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _trunc_flat_kernel(
    X,
    Out,
    N,
    BLOCK: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x0 = tl.load(X + off)
            tl.store(Out + off, _trunc_scalar(x0, IS_FP16, IS_BF16))
        else:
            m = off < N
            x0 = tl.load(X + off, mask=m, other=0.0)
            tl.store(Out + off, _trunc_scalar(x0, IS_FP16, IS_BF16), mask=m)


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


def _choose_trunc_launch(n_total, dtype):
    # in-place/out-of-place do_bench(median) sweep on GCU400 (2026-11-16),
    # floor/abs family retunes for the shared small bands:
    #   tiny (<= 8192): B1024/nw4 (4K: 3.3us);
    #   small-mid (<= 65536): B8192/nw2;
    #   fp32: B32768/nw2 mid, B65536/nw2 above 2^24 (DRAM floor 1337us @512M);
    #   16-bit mid (<= 2^26): B65536/nw4 (16.7M: 27.2us);
    #   16-bit huge: B131072/nw4 (512M fp16 673.6us / bf16 674.7us).
    if n_total <= 8192:
        return 1024, 4
    if n_total <= 65536:
        return 8192, 2
    if dtype == torch.float32:
        return (65536, 2) if n_total > (1 << 24) else (32768, 2)
    if n_total <= (1 << 26):
        return 65536, 4
    return 131072, 4


def _choose_copy_block(N, itemsize):
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


def _launch_trunc(src, dst, n, dtype):
    block, num_warps = _choose_trunc_launch(n, dtype)
    grid_size = min(triton.cdiv(n, block), NUM_SIPS * 2)
    even = n % block == 0
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared nodes);
    # retrying the launch recompiles and succeeds (threshold / rad2deg /
    # atan2 precedents for the same toolchain family).
    for attempt in range(3):
        try:
            with torch_device_fn.device(src.device):
                _trunc_flat_kernel[(grid_size,)](
                    src,
                    dst,
                    n,
                    BLOCK=block,
                    IS_FP16=dtype == torch.float16,
                    IS_BF16=dtype == torch.bfloat16,
                    EVEN=even,
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "trunc flat kernel launch failed (attempt %d): %s", attempt + 1, exc
            )


def _launch_copy(src, dst, words):
    block = _choose_copy_block(words, src.element_size())
    grid_size = min(triton.cdiv(words, block), NUM_SIPS * 2)
    num_warps = 4 if words <= _SINGLE_CTA_N else 2
    for attempt in range(3):
        try:
            with torch_device_fn.device(src.device):
                _copy_flat_kernel[(grid_size,)](
                    src, dst, words, BLOCK=block, num_warps=num_warps
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "trunc copy kernel launch failed (attempt %d): %s", attempt + 1, exc
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
        _launch_trunc(A, out, n, A.dtype)
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
                _launch_copy(src, dst, words)
                return out
        except Exception as exc:  # pragma: no cover - exotic view failures
            logger.warning("trunc int fast path failed, falling back: %s", exc)
    return trunc_func(A)


def trunc_(A):
    logger.debug("GEMS_ENFLAME TRUNC_")
    n = A.numel()
    if _flat_ok(A):
        _launch_trunc(A, A, n, A.dtype)
        return A
    if A.is_contiguous() and A.dtype in _INT_DTYPES:
        # Integral inputs are already truncated: the in-place op is a no-op.
        return A
    trunc_func(A, out0=A)
    return A
