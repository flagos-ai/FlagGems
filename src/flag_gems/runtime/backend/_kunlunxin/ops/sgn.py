# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from ..utils.pointwise_dynamic import pointwise_dynamic
from .sign import _SIGN_CONFIG, _sign_bit

logger = logging.getLogger(__name__)

# For real dtypes torch.sgn equals torch.sign (both map +-0 -> 0, +-inf -> +-1,
# NaN -> 0), so the real path shares sign's pure integer-ALU bit-domain body
# `_sign_bit` (see harness/solution/performance/sign_xpu5_20260813.md). The
# generic fp compare `(x > 0).to(x.dtype) - (x < 0).to(x.dtype)` lowers on XPU
# to fp compare -> i1 -> select -> sub and costs ~2.8x the bit-domain
# formulation. Complex tensors keep the magnitude-normalization kernels below
# (torch.sgn complex semantics differ from torch.sign).
_SMALL_NUMEL = 65536


@triton.jit
def _sgn_small_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # small-numel fast path: single masked 1D launch, bit-domain body
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, _sign_bit(x), mask=mask)


@triton.jit
def _sgn_complex_kernel(
    x_ri_ptr,
    out_ri_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_IN_FP32: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    base = offsets * 2
    real = tl.load(x_ri_ptr + base, mask=mask, other=0.0)
    imag = tl.load(x_ri_ptr + base + 1, mask=mask, other=0.0)
    real_compute = real.to(tl.float32) if COMPUTE_IN_FP32 else real
    imag_compute = imag.to(tl.float32) if COMPUTE_IN_FP32 else imag
    scale = tl.maximum(tl.abs(real_compute), tl.abs(imag_compute))
    is_zero = scale == 0.0
    safe_scale = tl.where(is_zero, 1.0, scale)
    finite_norm = scale * tl.sqrt(
        (real_compute / safe_scale) * (real_compute / safe_scale)
        + (imag_compute / safe_scale) * (imag_compute / safe_scale)
    )
    has_inf = (tl.abs(real_compute) == float("inf")) | (
        tl.abs(imag_compute) == float("inf")
    )
    has_nan = (real_compute != real_compute) | (imag_compute != imag_compute)
    norm = tl.where(
        has_inf,
        float("nan"),
        tl.where(has_nan, float("nan"), finite_norm),
    )
    safe_norm = tl.where(is_zero, 1.0, norm)
    tl.store(
        out_ri_ptr + base,
        tl.where(is_zero, 0.0, real_compute / safe_norm),
        mask=mask,
    )
    tl.store(
        out_ri_ptr + base + 1,
        tl.where(is_zero, 0.0, imag_compute / safe_norm),
        mask=mask,
    )


@triton.jit
def _sgn_complex_strided_kernel(
    x_ri_ptr,
    out_ri_ptr,
    n_elements,
    size0,
    size1,
    size2,
    size3,
    size4,
    size5,
    size6,
    size7,
    x_stride0,
    x_stride1,
    x_stride2,
    x_stride3,
    x_stride4,
    x_stride5,
    x_stride6,
    x_stride7,
    out_stride0,
    out_stride1,
    out_stride2,
    out_stride3,
    out_stride4,
    out_stride5,
    out_stride6,
    out_stride7,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    COMPUTE_IN_FP32: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    remaining = offsets
    x_offsets = 0
    out_offsets = 0
    if RANK >= 8:
        index = remaining % size7
        remaining = remaining // size7
        x_offsets += index * x_stride7
        out_offsets += index * out_stride7
    if RANK >= 7:
        index = remaining % size6
        remaining = remaining // size6
        x_offsets += index * x_stride6
        out_offsets += index * out_stride6
    if RANK >= 6:
        index = remaining % size5
        remaining = remaining // size5
        x_offsets += index * x_stride5
        out_offsets += index * out_stride5
    if RANK >= 5:
        index = remaining % size4
        remaining = remaining // size4
        x_offsets += index * x_stride4
        out_offsets += index * out_stride4
    if RANK >= 4:
        index = remaining % size3
        remaining = remaining // size3
        x_offsets += index * x_stride3
        out_offsets += index * out_stride3
    if RANK >= 3:
        index = remaining % size2
        remaining = remaining // size2
        x_offsets += index * x_stride2
        out_offsets += index * out_stride2
    if RANK >= 2:
        index = remaining % size1
        remaining = remaining // size1
        x_offsets += index * x_stride1
        out_offsets += index * out_stride1
    if RANK >= 1:
        x_offsets += remaining * x_stride0
        out_offsets += remaining * out_stride0

    real = tl.load(x_ri_ptr + x_offsets, mask=mask, other=0.0)
    imag = tl.load(x_ri_ptr + x_offsets + 1, mask=mask, other=0.0)
    real_compute = real.to(tl.float32) if COMPUTE_IN_FP32 else real
    imag_compute = imag.to(tl.float32) if COMPUTE_IN_FP32 else imag
    scale = tl.maximum(tl.abs(real_compute), tl.abs(imag_compute))
    is_zero = scale == 0.0
    safe_scale = tl.where(is_zero, 1.0, scale)
    finite_norm = scale * tl.sqrt(
        (real_compute / safe_scale) * (real_compute / safe_scale)
        + (imag_compute / safe_scale) * (imag_compute / safe_scale)
    )
    has_inf = (tl.abs(real_compute) == float("inf")) | (
        tl.abs(imag_compute) == float("inf")
    )
    has_nan = (real_compute != real_compute) | (imag_compute != imag_compute)
    norm = tl.where(
        has_inf,
        float("nan"),
        tl.where(has_nan, float("nan"), finite_norm),
    )
    safe_norm = tl.where(is_zero, 1.0, norm)
    tl.store(
        out_ri_ptr + out_offsets,
        tl.where(is_zero, 0.0, real_compute / safe_norm),
        mask=mask,
    )
    tl.store(
        out_ri_ptr + out_offsets + 1,
        tl.where(is_zero, 0.0, imag_compute / safe_norm),
        mask=mask,
    )


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=_SIGN_CONFIG)
@triton.jit
def _sgn_func(x):
    # vectorized 1D-tile DMA path for numel > _SMALL_NUMEL (same config family
    # as sign); bit-domain body shared with the small masked kernel.
    return _sign_bit(x)


def _sgn_impl(x: torch.Tensor, out: torch.Tensor):
    if x.device != out.device:
        raise RuntimeError("input and out must be on the same device")
    if x.dtype != out.dtype:
        raise RuntimeError(f"out must have dtype {x.dtype}, but got {out.dtype}")
    if out.shape != x.shape:
        out.resize_(x.shape)
    if x.numel() == 0:
        return out

    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    if x.is_complex() and (not x.is_contiguous() or not out.is_contiguous()):
        if x.ndim > 8:
            raise RuntimeError(
                "sgn supports non-contiguous complex tensors with at most 8 dimensions"
            )
        sizes = tuple(x.shape) + (1,) * (8 - x.ndim)
        x_strides = tuple(stride * 2 for stride in x.stride()) + (1,) * (8 - x.ndim)
        out_strides = tuple(stride * 2 for stride in out.stride()) + (1,) * (
            8 - out.ndim
        )
        with torch_device_fn.device(x.device):
            _sgn_complex_strided_kernel[grid](
                torch.view_as_real(x),
                torch.view_as_real(out),
                x.numel(),
                *sizes,
                *x_strides,
                *out_strides,
                BLOCK_SIZE=1024,
                RANK=x.ndim,
                COMPUTE_IN_FP32=x.dtype in (torch.complex32, torch.complex64),
            )
        return out

    if x.is_complex():
        with torch_device_fn.device(x.device):
            _sgn_complex_kernel[grid](
                torch.view_as_real(x).view(-1),
                torch.view_as_real(out).view(-1),
                x.numel(),
                BLOCK_SIZE=1024,
                COMPUTE_IN_FP32=x.dtype in (torch.complex32, torch.complex64),
            )
        return out

    # real path: torch.sgn real == torch.sign (bit-domain body above)
    if x.dtype == torch.bool:
        # torch.sgn(bool) is the identity; comparisons are meaningless on i1.
        if out.data_ptr() != x.data_ptr():
            out.copy_(x)
        return out
    if x.numel() <= _SMALL_NUMEL and x.is_contiguous() and out.is_contiguous():
        _sgn_small_kernel[grid](
            x.view(-1),
            out.view(-1),
            x.numel(),
            BLOCK_SIZE=1024,
        )
        return out
    _sgn_func(x, out0=out)
    return out


def sgn(x: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN SGN")
    return _sgn_impl(x, torch.empty_like(x))


def sgn_out(x: torch.Tensor, *, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN SGN_OUT")
    return _sgn_impl(x, out)
