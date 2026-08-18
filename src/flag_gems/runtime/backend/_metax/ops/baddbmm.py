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

"""MetaX (MACA) backend specialization for baddbmm.

The original baddbmm backward calls ``bmm()`` from ``flag_gems/ops/bmm.py``
directly. That path lacks the K>=16 padding required by the MACA compiler.

This module re-exports the unchanged forward kernel but overrides the backward
helpers to pad K<16 before calling the triton bmm kernel (without the M/N<32
fallback that the dispatch-level metax bmm uses, since the triton kernel can
handle small M/N fine).
"""

import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn

# Reuse the original forward kernel, launch helper, and bias grad computation
from flag_gems.ops.baddbmm import (
    _baddbmm_launch,
    compute_bias_grad,
)
from flag_gems.ops.mul import mul

# Import the metax bmm kernel (triton kernel) directly for backward use
from .bmm import bmm_kernel

logger = logging.getLogger(__name__)

_MIN_DOT_K = 16


def _bmm_for_backward(A, B):
    """BMM specialized for baddbmm backward: only pads K<16, no M/N fallback.

    The triton bmm kernel handles small M/N correctly. We only need to ensure
    K >= 16 for the tl.dot constraint on MACA backend.
    """
    batch, M, K = A.shape
    _, _, N = B.shape

    # MACA backend K dimension constraint: tl.dot requires K >= 16
    if K < _MIN_DOT_K:
        pad_k = _MIN_DOT_K - K
        logger.debug(
            "GEMS_METAX BADDBMM backward bmm padding K: %s -> %s", K, _MIN_DOT_K
        )
        A = torch.nn.functional.pad(A, (0, pad_k), mode="constant", value=0)
        B = torch.nn.functional.pad(B, (0, 0, 0, pad_k), mode="constant", value=0)
        K = _MIN_DOT_K

    A = A.contiguous()
    B = B.contiguous()
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        bmm_kernel[grid_fn](A, B, out, M, N, K, batch)
    return out


def compute_A_grad(d_output, B, alpha):
    """Compute grad w.r.t. A: alpha * d_output @ B^T"""
    B_T = B.transpose(1, 2)
    if B.dtype == torch.float16:
        Bcopy = B_T.to(torch.float32)
        dcopye = d_output.to(torch.float32)
        mul1 = _bmm_for_backward(dcopye, Bcopy)
        grad_A = mul(mul1, alpha)
        grad_A = grad_A.to(torch.float16)
    else:
        mul1 = _bmm_for_backward(d_output, B_T)
        grad_A = mul(mul1, alpha)
    return grad_A


def compute_B_grad(A, d_output, alpha):
    """Compute grad w.r.t. B: alpha * A^T @ d_output"""
    A_T = A.transpose(1, 2)
    if A.dtype == torch.float16:
        Acopy = A_T.to(torch.float32)
        dcopye = d_output.to(torch.float32)
        mul2 = _bmm_for_backward(Acopy, dcopye)
        grad_B = mul(mul2, alpha)
        grad_B = grad_B.to(torch.float16)
    else:
        mul2 = _bmm_for_backward(A_T, d_output)
        grad_B = mul(mul2, alpha)
    return grad_B


class BaddbmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, A, B, beta, alpha):
        logger.debug("GEMS_METAX BADDBMM FORWARD")

        ctx.save_for_backward(A, B, bias)
        ctx.alpha = alpha
        ctx.beta = beta

        batch, M, K = A.shape
        _, _, N = B.shape
        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)
        _baddbmm_launch(bias, A, B, beta, alpha, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS_METAX BADDBMM BACKWARD")
        A, B, bias = ctx.saved_tensors

        grad_A = None
        grad_B = None
        grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_bias = compute_bias_grad(grad_output, ctx.beta, bias)
        if ctx.needs_input_grad[1]:
            grad_A = compute_A_grad(grad_output, B, ctx.alpha)
        if ctx.needs_input_grad[2]:
            grad_B = compute_B_grad(A, grad_output, ctx.alpha)

        return grad_bias, grad_A, grad_B, None, None


def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    return BaddbmmFunction.apply(
        bias.contiguous(),
        A.contiguous(),
        B.contiguous(),
        beta,
        alpha,
    )
