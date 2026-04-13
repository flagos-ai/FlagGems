import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

from .mul import mul

logger = logging.getLogger(__name__)


def heur_tile_m(args):
    M = args["M"]
    if M <= 16:
        return 16
    elif M <= 32:
        return 32
    elif M <= 64:
        return 64
    else:
        return 128


def heur_tile_n(args):
    N = args["N"]
    if N <= 16:
        return 16
    elif N <= 32:
        return 32
    elif N <= 64:
        return 64
    else:
        return 128


def heur_tile_k(args):
    return 64


def heur_divisible_m(args):
    return args["M"] % args["TILE_M"] == 0


def heur_divisible_n(args):
    return args["N"] % args["TILE_N"] == 0


def heur_divisible_k(args):
    return args["K"] % args["TILE_K"] == 0


@libentry()
@triton.heuristics(
    {
        "TILE_M": heur_tile_m,
        "TILE_N": heur_tile_n,
        "TILE_K": heur_tile_k,
    }
)
@triton.heuristics(
    {
        "DIVISIBLE_M": heur_divisible_m,
        "DIVISIBLE_N": heur_divisible_n,
        "DIVISIBLE_K": heur_divisible_k,
    }
)
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
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_ob,
    stride_om,
    stride_on,
    bias_batch_stride,
    bias_M_stride,
    bias_N_stride,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    pid_b = tle.program_id(2)
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = (
        A
        + pid_b * stride_ab
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        B
        + pid_b * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, TILE_K)):
        if DIVISIBLE_K:
            if DIVISIBLE_M:
                mask_a = tl.full([TILE_M, TILE_K], value=1, dtype=tl.int1)
            else:
                mask_a = mask_m[:, None]
            if DIVISIBLE_N:
                mask_b = tl.full([TILE_K, TILE_N], value=1, dtype=tl.int1)
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

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=False)

        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk
        offs_k += TILE_K

    o_ptrs = (
        O
        + pid_b * stride_ob
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on
    )

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = tl.full([TILE_M, TILE_N], value=1, dtype=tl.int1)
    elif DIVISIBLE_M and not DIVISIBLE_N:
        mask_c = mask_n[None, :]
    elif not DIVISIBLE_M and DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]

    bias_ptrs = (
        bias
        + pid_b * bias_batch_stride
        + offs_m[:, None] * bias_M_stride
        + offs_n[None, :] * bias_N_stride
    )
    bi = tl.load(bias_ptrs, mask=mask_c, other=0.0)

    out = accumulator * alpha + bi * beta
    o = out.to(bi.dtype)
    tl.store(o_ptrs, o, mask=mask_c)


class BaddbmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, A, B, beta, alpha):
        logger.debug("GEMS BADDBMM FORWARD")

        ctx.save_for_backward(A, B, bias)
        ctx.alpha = alpha
        ctx.beta = beta

        batch, M, K = A.shape
        _, _, N = B.shape
        A = A.contiguous()
        B = B.contiguous()
        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

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
                A.stride(0),
                A.stride(1),
                A.stride(2),
                B.stride(0),
                B.stride(1),
                B.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                bias_batch_stride,
                bias_M_stride,
                bias_N_stride,
            )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS BADDBMM BACKWARD")
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
        mul1 = torch.bmm(dcopye, Bcopy)
        grad_A = mul(mul1, alpha)
        grad_A = grad_A.to(torch.float16)
    else:
        mul1 = torch.bmm(d_output, B_T)
        grad_A = mul(mul1, alpha)
    return grad_A


def compute_B_grad(A, d_output, alpha):
    A_T = A.transpose(1, 2)
    if A.dtype == torch.float16:
        Acopy = A_T.to(torch.float32)
        dcopye = d_output.to(torch.float32)
        mul2 = torch.bmm(Acopy, dcopye)
        grad_B = mul(mul2, alpha)
        grad_B = grad_B.to(torch.float16)
    else:
        mul2 = torch.bmm(A_T, d_output)
        grad_B = mul(mul2, alpha)
    return grad_B


def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    return BaddbmmFunction.apply(
        bias.contiguous(),
        A.contiguous(),
        B.contiguous(),
        beta,
        alpha,
    )
