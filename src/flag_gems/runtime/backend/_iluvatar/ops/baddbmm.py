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

"""
Iluvatar-specific baddbmm implementation that adds padding for small dimensions.

The Iluvatar Triton compiler requires:
1. All dimensions (M, N, K) >= 32
2. K must be a multiple of 128 (EVEN_K=False crashes the compiler)

This implementation pads matrices before calling the baddbmm kernel and slices
the result back to the original size, similar to the mm/addmm fix #5466.
"""

import logging

import torch

from flag_gems.ops.baddbmm import baddbmm_kernel
from flag_gems.ops.mul import mul
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._iluvatar.ops.bmm import bmm as bmm_iluvatar

logger = logging.getLogger(__name__)

# Iluvatar compiler constraints (from mm.py fix #5466)
_MIN_TRITON_DIM = 32
_MAX_BLOCK_K = 128


def _baddbmm_launch_padded(bias, A, B, beta, alpha, out):
    """Launch baddbmm kernel with padding for Iluvatar compiler constraints."""
    batch, M, K = A.shape
    _, _, N = B.shape

    # Check if padding is needed
    need_pad = (
        M < _MIN_TRITON_DIM
        or N < _MIN_TRITON_DIM
        or K < _MIN_TRITON_DIM
        or K % _MAX_BLOCK_K != 0
    )

    if not need_pad:
        # No padding needed, use default kernel directly
        A = A.contiguous()
        B = B.contiguous()
        bbias = torch.broadcast_to(bias, (batch, M, N)).contiguous()
        bias_batch_stride = bbias.stride(0)
        bias_M_stride = bbias.stride(1)
        bias_N_stride = bbias.stride(-1)

        grid = lambda meta: (
            triton.cdiv(meta["M"], meta["TILE_M"]),
            triton.cdiv(meta["N"], meta["TILE_N"]),
            batch,
        )

        with torch_device_fn.device(A.device):
            baddbmm_kernel[grid](
                A,
                B,
                out,
                bbias,
                alpha,
                beta,
                M,
                N,
                K,
                bias_batch_stride=bias_batch_stride,
                bias_M_stride=bias_M_stride,
                bias_N_stride=bias_N_stride,
            )
        return

    # Padding is needed - calculate padding amounts
    pad_M = max(_MIN_TRITON_DIM - M, 0)
    pad_N = max(_MIN_TRITON_DIM - N, 0)
    new_K = K + max(_MIN_TRITON_DIM - K, 0)
    remainder = new_K % _MAX_BLOCK_K
    if remainder:
        new_K += _MAX_BLOCK_K - remainder
    pad_K = new_K - K

    # Pad inputs
    if pad_M or pad_K:
        A_padded = torch.nn.functional.pad(A, (0, pad_K, 0, pad_M))
    else:
        A_padded = A

    if pad_K or pad_N:
        B_padded = torch.nn.functional.pad(B, (0, pad_N, 0, pad_K))
    else:
        B_padded = B

    # Pad bias
    bias_expanded = torch.broadcast_to(bias, (batch, M, N))
    if pad_M or pad_N:
        bias_padded = torch.nn.functional.pad(bias_expanded, (0, pad_N, 0, pad_M))
    else:
        bias_padded = bias_expanded

    # Make everything contiguous
    A_padded = A_padded.contiguous()
    B_padded = B_padded.contiguous()
    bias_padded = bias_padded.contiguous()

    pM, pN, pK = M + pad_M, N + pad_N, new_K

    # Create padded output
    out_padded = torch.empty((batch, pM, pN), dtype=A.dtype, device=A.device)

    # Launch kernel on padded inputs
    import triton

    bias_batch_stride = bias_padded.stride(0)
    bias_M_stride = bias_padded.stride(1)
    bias_N_stride = bias_padded.stride(-1)

    grid = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )

    with torch_device_fn.device(A.device):
        baddbmm_kernel[grid](
            A_padded,
            B_padded,
            out_padded,
            bias_padded,
            alpha,
            beta,
            pM,
            pN,
            pK,
            bias_batch_stride=bias_batch_stride,
            bias_M_stride=bias_M_stride,
            bias_N_stride=bias_N_stride,
        )

    # Slice back to original size
    out.copy_(out_padded[:, :M, :N])


def compute_A_grad(d_output, B, alpha):
    """Compute gradient for A using iluvatar-specialized bmm."""
    B_T = B.transpose(1, 2)
    if B.dtype == torch.float16:
        Bcopy = B_T.to(torch.float32)
        dcopye = d_output.to(torch.float32)
        mul1 = bmm_iluvatar(dcopye, Bcopy)
        grad_A = mul(mul1, alpha)
        grad_A = grad_A.to(torch.float16)
    else:
        mul1 = bmm_iluvatar(d_output, B_T)
        grad_A = mul(mul1, alpha)
    return grad_A


def compute_B_grad(A, d_output, alpha):
    """Compute gradient for B using iluvatar-specialized bmm."""
    A_T = A.transpose(1, 2)
    if A.dtype == torch.float16:
        Acopy = A_T.to(torch.float32)
        dcopye = d_output.to(torch.float32)
        mul2 = bmm_iluvatar(Acopy, dcopye)
        grad_B = mul(mul2, alpha)
        grad_B = grad_B.to(torch.float16)
    else:
        mul2 = bmm_iluvatar(A_T, d_output)
        grad_B = mul(mul2, alpha)
    return grad_B


def compute_bias_grad(d_output, beta, bias):
    """Compute gradient for bias (copied from ops.baddbmm)."""
    grad_bias = mul(d_output, beta)
    if grad_bias.shape != bias.shape:
        # Sum over broadcasted dimensions
        while grad_bias.dim() > bias.dim():
            grad_bias = grad_bias.sum(dim=0)
        for i in range(bias.dim()):
            if bias.shape[i] == 1 and grad_bias.shape[i] > 1:
                grad_bias = grad_bias.sum(dim=i, keepdim=True)
    return grad_bias


class BaddbmmFunction(torch.autograd.Function):
    """Iluvatar-specific BaddbmmFunction with padding and specialized bmm."""

    @staticmethod
    def forward(ctx, bias, A, B, beta, alpha):
        logger.debug("GEMS_ILUVATAR BADDBMM FORWARD")

        ctx.save_for_backward(A, B, bias)
        ctx.alpha = alpha
        ctx.beta = beta

        batch, M, K = A.shape
        _, _, N = B.shape
        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)
        _baddbmm_launch_padded(bias, A, B, beta, alpha, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS_ILUVATAR BADDBMM BACKWARD")
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


def baddbmm(bias, a, b, beta=1.0, alpha=1.0):
    """Batch matrix multiply-add with Iluvatar-specific padding for small dimensions.

    Computes: out = beta * bias + alpha * (a @ b)
    Uses BaddbmmFunction.apply which has specialized backward.
    """
    logger.debug("GEMS_ILUVATAR BADDBMM")
    return BaddbmmFunction.apply(
        bias.contiguous(), a.contiguous(), b.contiguous(), beta, alpha
    )


def baddbmm_out(bias, a, b, *, beta=1.0, alpha=1.0, out):
    """Batch matrix multiply-add with output tensor, using Iluvatar-specific padding."""
    logger.debug("GEMS_ILUVATAR BADDBMM_OUT")

    result = baddbmm(bias, a, b, beta=beta, alpha=alpha)
    out.copy_(result)
    return out
