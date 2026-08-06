# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of lcm / lcm_.
#
# Same root cause as gcd: the generic `flag_gems/ops/lcm.py` binary-GCD helper
# uses `libdevice.ffs` (returns None on XPU -> compile error). We compute the
# gcd via a fixed-iteration Euclidean loop on *signed magnitudes* using
# `tl.abs` + C-style modulo, matching torch.gcd exactly (including INT_MIN
# sign propagation for int16/int32; INT64_MIN->0 quirk for int64 to match
# the XPU-native reference). Then lcm = ax // gcd * ay in unsigned domain
# for int32/int64 to avoid the signed multiply overflow producing wrong
# sign; for int8/int16 the truncating cast into the output dtype naturally
# reproduces torch's wrap semantics.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.lcm import _materialize_inputs

from .gcd import _ITERS_32, _ITERS_64

logger = logging.getLogger(__name__)


@triton.jit
def lcm_kernel_32(x_ptr, y_ptr, out_ptr, n_elements, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    # Signed Euclidean gcd (matches torch.gcd bit-for-bit incl. INT_MIN).
    a = tl.abs(x)
    b = tl.abs(y)
    for _ in range(ITERS):
        nz = b != 0
        bb = tl.where(nz, b, 1)
        r = a % bb
        a = tl.where(nz, b, a)
        b = tl.where(nz, r.to(b.dtype), b)
    g = a
    # lcm = |x| / gcd * |y|. Compute magnitudes in unsigned domain to avoid
    # signed overflow when |x|*|y| exceeds INT_MAX (torch wraps in signed).
    ax = tl.abs(x).to(tl.uint32)
    ay = tl.abs(y).to(tl.uint32)
    gu = g.to(tl.uint32)
    zero_g = g == 0
    safe_g = tl.where(zero_g, 1, gu)
    result_u = (ax // safe_g) * ay
    result_u = tl.where(zero_g, 0, result_u)
    tl.store(out_ptr + offsets, result_u.to(out_ptr.type.element_ty), mask=mask)


@triton.jit
def lcm_kernel_64(x_ptr, y_ptr, out_ptr, n_elements, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    int64_min: tl.constexpr = -9223372036854775808
    x0 = tl.where(x == int64_min, 0, x)
    y0 = tl.where(y == int64_min, 0, y)
    a = tl.abs(x0)
    b = tl.abs(y0)
    for _ in range(ITERS):
        nz = b != 0
        bb = tl.where(nz, b, 1)
        r = a % bb
        a = tl.where(nz, b, a)
        b = tl.where(nz, r.to(b.dtype), b)
    g = a
    ax = tl.abs(x0).to(tl.uint64)
    ay = tl.abs(y0).to(tl.uint64)
    gu = g.to(tl.uint64)
    zero_g = g == 0
    safe_g = tl.where(zero_g, 1, gu)
    result_u = (ax // safe_g) * ay
    result_u = tl.where(zero_g, 0, result_u)
    tl.store(out_ptr + offsets, result_u.to(out_ptr.type.element_ty), mask=mask)


def _kernel_meta(dtype):
    if dtype in (torch.int8, torch.int16, torch.int32):
        return lcm_kernel_32, _ITERS_32, 128, 1
    if dtype == torch.int64:
        return lcm_kernel_64, _ITERS_64, 64, 1
    raise TypeError(f"unsupported dtype for lcm: {dtype}")


def lcm(self, other):
    logger.debug("GEMS_KUNLUNXIN LCM")
    lhs, rhs, promoted_dtype = _materialize_inputs(self, other)
    result = torch.empty_like(lhs, dtype=promoted_dtype)
    numel = result.numel()
    if numel == 0:
        return result
    kernel, iters, block, num_warps = _kernel_meta(promoted_dtype)
    grid = (triton.cdiv(numel, block),)
    kernel[grid](
        lhs.reshape(-1),
        rhs.reshape(-1),
        result.reshape(-1),
        numel,
        ITERS=iters,
        BLOCK=block,
        num_warps=num_warps,
        buffer_size_limit=2048,
        isCloseVectorization=True,
    )
    return result.view(lhs.shape)


def lcm_(self, other):
    logger.debug("GEMS_KUNLUNXIN LCM_")
    result = lcm(self, other)
    self.copy_(result)
    return self
