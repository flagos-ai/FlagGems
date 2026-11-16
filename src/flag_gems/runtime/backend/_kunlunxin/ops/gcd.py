import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.gcd import _materialize_inputs

logger = logging.getLogger(__name__)

_ITERS_32 = 48
_ITERS_64 = 96
_ITERS = {torch.int8: 24, torch.int16: 24, torch.int32: _ITERS_32}

_GCD_BLOCK = 128
_GCD_NCHUNK = 8
_GCD_FAST_MIN_NUMEL = 65536


@triton.jit
def gcd_kernel_32(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    ITERS: tl.constexpr,
    MINV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    xi = x.to(tl.int32)
    yi = y.to(tl.int32)
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
def gcd_kernel_32_fast(
    x_ptr,
    y_ptr,
    out_ptr,
    ITERS: tl.constexpr,
    MINV: tl.constexpr,
    BLOCK: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * (BLOCK * NCHUNK)
    for c in tl.static_range(NCHUNK):
        offsets = base + c * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        xi = x.to(tl.int32)
        yi = y.to(tl.int32)
        a0 = tl.where(xi == MINV, xi, tl.abs(xi))
        b0 = tl.where(yi == MINV, yi, tl.abs(yi))
        for _ in range(ITERS):
            nz = b0 != 0
            bb = tl.where(nz, b0, 1)
            r = a0 % bb
            a0 = tl.where(nz, b0, a0)
            b0 = tl.where(nz, r, b0)
        tl.store(out_ptr + offsets, a0.to(out_ptr.type.element_ty))


@triton.jit
def gcd_kernel_64(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    ITERS: tl.constexpr,
    MINV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    a0 = tl.where(x == MINV, x, tl.abs(x))
    b0 = tl.where(y == MINV, y, tl.abs(y))
    for _ in range(ITERS):
        nz = b0 != 0
        bb = tl.where(nz, b0, 1)
        r = a0 % bb
        a0 = tl.where(nz, b0, a0)
        b0 = tl.where(nz, r, b0)
    tl.store(out_ptr + offsets, a0.to(out_ptr.type.element_ty), mask=mask)


@triton.jit
def gcd_kernel_64_fast(
    x_ptr,
    y_ptr,
    out_ptr,
    ITERS: tl.constexpr,
    MINV: tl.constexpr,
    BLOCK: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * (BLOCK * NCHUNK)
    for c in tl.static_range(NCHUNK):
        offsets = base + c * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        a0 = tl.where(x == MINV, x, tl.abs(x))
        b0 = tl.where(y == MINV, y, tl.abs(y))
        for _ in range(ITERS):
            nz = b0 != 0
            bb = tl.where(nz, b0, 1)
            r = a0 % bb
            a0 = tl.where(nz, b0, a0)
            b0 = tl.where(nz, r, b0)
        tl.store(out_ptr + offsets, a0.to(out_ptr.type.element_ty))


def _kernel_meta(dtype):
    if dtype in (torch.int8, torch.int16, torch.int32):
        minv = -(1 << 31) if dtype == torch.int32 else torch.iinfo(dtype).min
        return gcd_kernel_32, gcd_kernel_32_fast, _ITERS[dtype], minv, _GCD_BLOCK, 8
    if dtype == torch.int64:
        return (
            gcd_kernel_64,
            gcd_kernel_64_fast,
            _ITERS_64,
            -(1 << 63),
            _GCD_BLOCK,
            8,
        )
    raise TypeError(f"unsupported dtype for gcd: {dtype}")


def _launch_gcd(lhs, rhs, out):
    numel = out.numel()
    if numel == 0:
        return out
    kernel, fast_kernel, iters, minv, block, num_warps = _kernel_meta(out.dtype)
    chunk = block * _GCD_NCHUNK
    if numel >= _GCD_FAST_MIN_NUMEL and numel % chunk == 0:
        grid = (triton.cdiv(numel, chunk),)
        fast_kernel[grid](
            lhs,
            rhs,
            out,
            ITERS=iters,
            MINV=minv,
            BLOCK=block,
            NCHUNK=_GCD_NCHUNK,
            num_warps=num_warps,
        )
        return out
    grid = (triton.cdiv(numel, block),)
    kernel[grid](
        lhs,
        rhs,
        out,
        numel,
        ITERS=iters,
        MINV=minv,
        BLOCK=block,
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
        _launch_gcd(lhs.reshape(-1), rhs.reshape(-1), A.reshape(-1))
        return A
    flat_out = torch.empty(lhs.numel(), dtype=promoted_dtype, device=A.device)
    _launch_gcd(lhs.reshape(-1), rhs.reshape(-1), flat_out)
    torch.ops.aten._copy_from(flat_out.view(A.shape), A, False)
    return A
