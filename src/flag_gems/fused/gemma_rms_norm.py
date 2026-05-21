"""Triton Gemma RMS Normalization kernels.

Implements Gemma-style RMSNorm where the weight uses a +1 offset:
    output = x * rsqrt(mean(x^2) + eps) * (1 + weight)

This differs from standard RMSNorm (which multiplies by weight directly)
because Gemma initializes weights to zero and adds 1 during forward pass.

Two kernels are provided:
    - gemma_rms_norm_kernel: standalone normalization
    - gemma_fused_add_rms_norm_kernel: fused residual addition + normalization
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def gemma_rms_norm_kernel(
    input_ptr,
    w_ptr,
    output_ptr,
    in_stride_r,
    in_stride_c,
    out_stride_r,
    out_stride_c,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid = tl.program_id(0)
    input_ptr += pid * in_stride_r
    output_ptr += pid * out_stride_r

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(input_ptr + cols * in_stride_c, mask=mask, other=0.0).to(cdtype)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(cdtype)

    var = tl.sum(x * x / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)

    y = (x * rrms * (1.0 + w)).to(input_ptr.dtype.element_ty)

    tl.store(output_ptr + cols * out_stride_c, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def gemma_fused_add_rms_norm_kernel(
    input_ptr,
    residual_ptr,
    w_ptr,
    in_stride_r,
    in_stride_c,
    r_stride_r,
    r_stride_c,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid = tl.program_id(0)
    input_ptr += pid * in_stride_r
    residual_ptr += pid * r_stride_r

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(input_ptr + cols * in_stride_c, mask=mask, other=0.0).to(cdtype)
    r = tl.load(residual_ptr + cols * r_stride_c, mask=mask, other=0.0).to(cdtype)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(cdtype)

    x += r
    tl.store(residual_ptr + cols * r_stride_c, x, mask=mask)

    var = tl.sum(x * x / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)

    y = (x * rrms * (1.0 + w)).to(input_ptr.dtype.element_ty)

    tl.store(input_ptr + cols * in_stride_c, y, mask=mask)


def gemma_rms_norm(x, normalized_shape, weight, eps=1e-6):
    logger.debug(
        "GEMS GEMMA_RMS_NORM, [input shape]: %s, [weight shape]: %s",
        x.size(),
        weight.size(),
    )
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    BLOCK_SIZE = triton.next_power_of_2(N)
    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty_like(x)

    with torch_device_fn.device(x.device):
        gemma_rms_norm_kernel[M,](
            x, weight, out, N, 1, N, 1, N, eps, BLOCK_SIZE
        )
    return out


def gemma_fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-6):
    logger.debug(
        "GEMS GEMMA_FUSED_ADD_RMS_NORM, [input shape]: %s, [residual shape]: %s, [weight shape]: %s",
        x.size(),
        residual.size(),
        weight.size(),
    )
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    BLOCK_SIZE = triton.next_power_of_2(N)
    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    with torch_device_fn.device(x.device):
        gemma_fused_add_rms_norm_kernel[M,](
            x, residual, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
        )
    return x, residual
