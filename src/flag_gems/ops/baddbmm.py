import logging

import torch
import triton
import triton.language as tl

from .. import runtime
from ..runtime import torch_device_fn
from ..utils import libentry, libtuner
from ..utils import triton_lang_extension as tle

if runtime.device.vendor_name == "iluvatar":
    from flag_gems.runtime.backend._iluvatar.ops.bmm import bmm
else:
    from .bmm import bmm

from .mul import mul


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("baddbmm"),
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.heuristics(runtime.get_heuristic_config("baddbmm"))
@triton.jit(do_not_specialize=["alpha", "beta"])
def baddbmm_kernel(
    A,
    B,
    O,
    bias,
    alpha,
    beta,
    M,
    N,
    K,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
    bias_batch_stride: tl.constexpr,
    bias_M_stride: tl.constexpr,
    bias_N_stride: tl.constexpr,
):
    # batch offsets
    pid_b = tle.program_id(2)
    A += pid_b * M * K
    B += pid_b * K * N
    O += pid_b * M * N
    bias += pid_b * bias_batch_stride

    pidx = tle.program_id(0)
    pidy = tle.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        gridx = tle.num_programs(0)
        gridy = tle.num_programs(1)
        pid = pidx + pidy * gridx
        num_CTA_per_group = gridy * GROUP_M
        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    o_ptrs = O + offs_m[:, None] * N + offs_n[None, :]

    num_iters = tl.cdiv(K, TILE_K)
    accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for _ in range(num_iters):
        if DIVISIBLE_K:
            if DIVISIBLE_M:
                mask_a = None
            else:
                mask_a = mask_m[:, None]
            if DIVISIBLE_N:
                mask_b = None
            else:
                mask_b = mask_n[None, :]
        else:
            mask_k = offs_k < K
            if DIVISIBLE_M:
                mask_a = mask_k[None, :]
            else:
                mask_a = mask_m[:, None] & mask_k[None, :]
            if DIVISIBLE_N:
                mask_b = mask_k[:, None]
            else:
                mask_b = mask_k[:, None] & mask_n[None, :]
        a = tl.load(a_ptrs, mask=mask_a)
        b = tl.load(b_ptrs, mask=mask_b)
        accumulator += tl.dot(a, b, allow_tf32=False)
        offs_k += TILE_K
        a_ptrs += TILE_K
        b_ptrs += TILE_K * N

    bias_ptrs = bias + offs_m[:, None] * bias_M_stride + offs_n[None, :] * bias_N_stride

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    else:
        mask_c = True
        if not DIVISIBLE_M:
            mask_c &= offs_m[:, None] < M
        if not DIVISIBLE_N:
            mask_c &= offs_n[None, :] < N

    bi = tl.load(bias_ptrs, mask=mask_c)
    out = accumulator * alpha + bi * beta
    o = out.to(bi.dtype)
    tl.store(o_ptrs, o, mask=mask_c)


class BaddbmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, A, B, beta, alpha):
        logging.debug("GEMS BADDBMM FORWARD")

        ctx.save_for_backward(A, B, bias)
        ctx.alpha = alpha
        ctx.beta = beta

        batch, M, K = A.shape
        _, _, N = B.shape
        A = A.contiguous()
        B = B.contiguous()
        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

        bbias = torch.broadcast_to(bias, (batch, M, N))
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
        return out

    @staticmethod
    def backward(ctx, grad_output):
        logging.debug("GEMS BADDBMM BACKWARD")
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
    B_T = B.transpose(1, 2)
    if B.dtype == torch.float16:
        Bcopy = B_T.to(torch.float32)
        dcopye = d_output.to(torch.float32)
        mul1 = bmm(dcopye, Bcopy)
        grad_A = mul(mul1, alpha)
        grad_A = grad_A.to(torch.float16)
    else:
        mul1 = bmm(d_output, B_T)
        grad_A = mul(mul1, alpha)
    return grad_A


def compute_B_grad(A, d_output, alpha):
    A_T = A.transpose(1, 2)
    if A.dtype == torch.float16:
        Acopy = A_T.to(torch.float32)
        dcopye = d_output.to(torch.float32)
        mul2 = bmm(Acopy, dcopye)
        grad_B = mul(mul2, alpha)
        grad_B = grad_B.to(torch.float16)
    else:
        mul2 = bmm(A_T, d_output)
        grad_B = mul(mul2, alpha)
    return grad_B


def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    return BaddbmmFunction.apply(bias, A.contiguous(), B.contiguous(), beta, alpha)
