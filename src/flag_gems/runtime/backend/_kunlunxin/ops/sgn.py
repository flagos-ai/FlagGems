# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _sgn_real_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, IS_BOOL: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    result = x if IS_BOOL else (x > 0).to(x.dtype) - (x < 0).to(x.dtype)
    tl.store(out_ptr + offsets, result, mask=mask)


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


def _sgn_impl(x: torch.Tensor, out: torch.Tensor):
    if x.device != out.device:
        raise RuntimeError("input and out must be on the same device")
    if x.dtype != out.dtype:
        raise RuntimeError(f"out must have dtype {x.dtype}, but got {out.dtype}")
    if out.shape != x.shape:
        out.resize_(x.shape)
    if x.numel() == 0:
        return out

    x_contig = x.contiguous()
    out_contig = out if out.is_contiguous() else torch.empty_like(x_contig)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        if x.is_complex():
            _sgn_complex_kernel[grid](
                torch.view_as_real(x_contig).view(-1),
                torch.view_as_real(out_contig).view(-1),
                x.numel(),
                BLOCK_SIZE=1024,
                COMPUTE_IN_FP32=x.dtype in (torch.complex32, torch.complex64),
            )
        else:
            _sgn_real_kernel[grid](
                x_contig.view(-1),
                out_contig.view(-1),
                x.numel(),
                BLOCK_SIZE=1024,
                IS_BOOL=x.dtype == torch.bool,
            )
    if out_contig is not out:
        out.copy_(out_contig)
    return out


def sgn(x: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN SGN")
    return _sgn_impl(x, torch.empty_like(x))


def sgn_out(x: torch.Tensor, *, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN SGN_OUT")
    return _sgn_impl(x, out)
