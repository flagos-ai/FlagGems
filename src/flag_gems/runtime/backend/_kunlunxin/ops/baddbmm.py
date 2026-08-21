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
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .mul import mul

logger = logging.getLogger(__name__)

# Fused batch-GEMM + bias/alpha/beta epilogue.
#
# XPU (kunlunxin) closure 2026-08-17: the previous forward path delegated to the
# bmm() kernel + a separate epilogue kernel (2 launches + 1 intermediate
# tensor). That path (a) rounds the GEMM result to the input dtype BEFORE
# applying alpha/beta/bias, which is numerically wrong for bf16 (double
# rounding, amplified up to ~0.5 relative error when alpha == 100), and (b)
# inherited the bmm kernel's 256x256x256-tile uni_sram compile failures for
# several test shapes. We now use a single fused kernel styled after the
# addmm/mm XPU closure kernels (always-masked loads other=0.0, GROUP_M L2
# swizzle, dtype-dependent reduction tile, no @autotune): bounded heuristics
# avoid the per-shape re-tuning that previously measured ~137ms on 4096^3.
def heur_tile_m(args):
    M = args["M"]
    if M <= 512:
        return 128
    return 256


def heur_tile_n(args):
    N = args["N"]
    if N <= 512:
        return 128
    return 256


def heur_tile_k(args):
    # addmm closure probe: on this backend fp16 prefers BK=256 while
    # bf16/fp32 prefer BK=128 (fp32 BK=256 collapses on 4096^3).
    if args.get("TILE_K_CHOICE", 128) == 256:
        return 256
    return 128


def heur_num_warps(args):
    if args["M"] <= 512 and args["N"] <= 512:
        return 4
    return 8


@libentry()
@triton.heuristics(
    {
        "TILE_M": heur_tile_m,
        "TILE_N": heur_tile_n,
        "TILE_K": heur_tile_k,
        "num_warps": heur_num_warps,
    }
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def baddbmm_kernel(
    A,
    B,
    bias,
    O,
    alpha,
    beta,
    M,
    N,
    K,
    batch_stride_a,
    batch_stride_b,
    batch_stride_o,
    bias_batch_stride,
    bias_M_stride,
    bias_N_stride,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TILE_K_CHOICE,
):
    # batch offsets
    pid_z = ext.program_id(1)
    A += pid_z * batch_stride_a
    B += pid_z * batch_stride_b
    bias += pid_z * bias_batch_stride
    O += pid_z * batch_stride_o

    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, TILE_M)
    grid_n = tl.cdiv(N, TILE_N)
    # re-order program ID for better L2 reuse along the N dimension
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_am = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_bn = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    a_ptrs = A + offs_am[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_bn[None, :]

    accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, TILE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * TILE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * TILE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += TILE_K
        b_ptrs += TILE_K * N

    offs_cm = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_cn = pid_n * TILE_N + tl.arange(0, TILE_N)
    c_ptrs = O + offs_cm[:, None] * N + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    bias_ptrs = bias + offs_cm[:, None] * bias_M_stride + offs_cn[None, :] * bias_N_stride
    bias_value = tl.load(bias_ptrs, mask=c_mask, other=0.0)

    out = accumulator * alpha + bias_value * beta
    # tl.store converts to the output pointer dtype; the .out variant may use
    # an fp32 output with fp16/bf16 inputs and an input-dtype bias.
    tl.store(c_ptrs, out, mask=c_mask)


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def _baddbmm_scalar_kernel(
    A,
    B,
    O,
    bias,
    alpha,
    beta,
    bias_batch_stride: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    a = tl.load(A + pid_b * BLOCK_K + offsets)
    b = tl.load(B + pid_b * BLOCK_K + offsets)
    dot = tl.sum(a.to(tl.float32) * b.to(tl.float32), axis=0)
    bi = tl.load(bias + pid_b * bias_batch_stride).to(tl.float32)
    tl.store(O + pid_b, alpha * dot + beta * bi)


def _direct_baddbmm(A, B, bias, alpha, beta, out=None):
    batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    B = B.contiguous()
    if out is None:
        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    if M == 1 and N == 1 and K == 32:
        bias_view = bias.broadcast_to((batch, M, N))
        with torch_device_fn.device(A.device):
            _baddbmm_scalar_kernel[(batch,)](
                A,
                B,
                out,
                bias_view,
                alpha,
                beta,
                bias_batch_stride=bias_view.stride(0),
                BLOCK_K=32,
            )
        return out

    bias = bias.broadcast_to((batch, M, N))
    tile_k_choice = 256 if A.dtype == torch.float16 else 128
    grid_fn = lambda meta: (
        triton.cdiv(M, meta["TILE_M"]) * triton.cdiv(N, meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        baddbmm_kernel[grid_fn](
            A,
            B,
            bias,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            A.stride(0),
            B.stride(0),
            out.stride(0),
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            GROUP_M=8,
            TILE_K_CHOICE=tile_k_choice,
            num_stages=3,
        )
    return out


def _chunked_bmm(lhs, rhs, scale):
    batch, M, N = lhs.shape[0], lhs.shape[1], rhs.shape[2]
    bias = torch.zeros((batch, M, N), dtype=lhs.dtype, device=lhs.device)
    return _direct_baddbmm(lhs, rhs, bias, scale, 0.0)


class BaddbmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, A, B, beta, alpha):
        logger.debug("GEMS_KUNLUNXIN BADDBMM_FORWARD")

        ctx.save_for_backward(A, B, bias)
        ctx.alpha = alpha
        ctx.beta = beta

        return _direct_baddbmm(A, B, bias, alpha, beta)

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS_KUNLUNXIN BADDBMM_BACKWARD")
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


def compute_bias_grad(d_output, beta, bias):
    grad_bias = mul(d_output, beta)
    if grad_bias.shape != bias.shape:
        # Sum over broadcasted dimensions
        while grad_bias.dim() > bias.dim():
            grad_bias = grad_bias.sum(dim=0)
        for i in range(bias.dim()):
            if bias.shape[i] == 1 and grad_bias.shape[i] > 1:
                grad_bias = grad_bias.sum(dim=i, keepdim=True)
    return grad_bias.view(bias.shape)


def compute_A_grad(d_output, B, alpha):
    output_dtype = B.dtype
    B_T = B.transpose(1, 2)
    if output_dtype in (torch.float16, torch.bfloat16):
        B_T = B_T.to(torch.float32)
        d_output = d_output.to(torch.float32)
    grad_A = _chunked_bmm(d_output, B_T, alpha)
    return grad_A.to(output_dtype)


def compute_B_grad(A, d_output, alpha):
    output_dtype = A.dtype
    A_T = A.transpose(1, 2)
    if output_dtype in (torch.float16, torch.bfloat16):
        A_T = A_T.to(torch.float32)
        d_output = d_output.to(torch.float32)
    grad_B = _chunked_bmm(A_T, d_output, alpha)
    return grad_B.to(output_dtype)


def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    return BaddbmmFunction.apply(bias, A.contiguous(), B.contiguous(), beta, alpha)


def baddbmm_(self, batch1, batch2, *, beta=1.0, alpha=1.0):
    logger.debug("GEMS_KUNLUNXIN BADDBMM_")
    # In-place on the bias: write directly into self via the vendor-tuned out
    # variant, avoiding the generic addmm_+copy_ style path (the previous
    # generic baddbmm_ chained the generic baddbmm + copy_, which crashed in
    # bmm's 256-tile uni_sram compile on this backend).
    return baddbmm_out(self, batch1, batch2, beta=beta, alpha=alpha, out=self)


def baddbmm_out(bias, A, B, beta=1.0, alpha=1.0, *, out):
    output_shape = (A.shape[0], A.shape[1], B.shape[2])
    if tuple(out.shape) != output_shape:
        out.resize_(output_shape)
    return _direct_baddbmm(A, B, bias, alpha, beta, out=out)