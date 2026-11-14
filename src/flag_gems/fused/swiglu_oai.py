# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Split SwiGLU-OAI with input-dtype rounding at each staged intermediate."""

import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _staged_oai_kernel(
    input_ptr,
    output_ptr,
    n_inter,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    alpha,
    beta,
    limit,
    BLOCK_I: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = cols < n_inter
    gate = tl.load(
        input_ptr + row * stride_im + cols * stride_in, mask=mask, other=0
    ).to(tl.float32)
    up = tl.load(
        input_ptr + row * stride_im + (n_inter + cols) * stride_in, mask=mask, other=0
    ).to(tl.float32)
    storage_ty = input_ptr.dtype.element_ty
    gate = (
        tl.where(gate != gate, gate, tl.minimum(gate, limit))
        .to(storage_ty)
        .to(tl.float32)
    )
    up = (
        tl.where(up != up, up, tl.minimum(tl.maximum(up, -limit), limit))
        .to(storage_ty)
        .to(tl.float32)
    )
    scaled = (alpha * gate).to(storage_ty).to(tl.float32)
    sigmoid = tl.sigmoid(scaled).to(storage_ty).to(tl.float32)
    gate_sigmoid = (gate * sigmoid).to(storage_ty).to(tl.float32)
    up_beta = (up + beta).to(storage_ty).to(tl.float32)
    result = gate_sigmoid * up_beta
    tl.store(
        output_ptr + row * stride_om + cols * stride_on,
        result.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def swiglu_oai(x, limit=7.0, alpha=1.702, beta=1.0):
    """Forward inference for split ``[gate..., up...]`` FP16/BF16 activations.

    Gate is clamped only above ``limit``; up is clamped to ±limit. Each
    intermediate rounds to the input dtype, matching the staged eager formula.
    2D strided inputs and contiguous higher-rank inputs are supported.
    """
    if x.dtype not in (torch.float16, torch.bfloat16) or x.device.type != "cuda":
        raise NotImplementedError("SwiGLU-OAI supports CUDA FP16/BF16 inputs")
    if x.ndim < 2 or x.shape[-1] % 2:
        raise ValueError("SwiGLU-OAI requires rank >= 2 and an even final dimension")
    if x.ndim > 2 and not x.is_contiguous():
        raise NotImplementedError("Higher-rank SwiGLU-OAI inputs must be contiguous")
    if x.requires_grad:
        raise NotImplementedError(
            "SwiGLU-OAI currently supports forward inference only"
        )
    limit, alpha, beta = float(limit), float(alpha), float(beta)
    if limit < 0 or not all(math.isfinite(p) for p in (limit, alpha, beta)):
        raise ValueError("SwiGLU-OAI parameters must be finite and limit nonnegative")
    width = x.shape[-1] // 2
    output = torch.empty((*x.shape[:-1], width), dtype=x.dtype, device=x.device)
    if output.numel() == 0:
        return output
    rows = math.prod(x.shape[:-1])
    x2 = x if x.ndim == 2 else x.view(rows, 2 * width)
    y2 = output.view(rows, width)
    with torch_device_fn.device(x.device):
        _staged_oai_kernel[(rows, triton.cdiv(width, 256))](
            x2,
            y2,
            width,
            x2.stride(0),
            x2.stride(1),
            y2.stride(0),
            y2.stride(1),
            alpha,
            beta,
            limit,
            BLOCK_I=256,
            enable_fp_fusion=False,
        )
    return output
