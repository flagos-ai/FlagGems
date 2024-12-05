import logging

import torch
import triton
import triton.language as tl

from .. import runtime
from ..utils import libentry
from ..utils import triton_lang_extension as tle


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


def heur_num_warps(args):
    if args["N"] <= 1024:
        return 4
    elif args["N"] <= 2048:
        return 8
    else:
        return 16


@libentry()
@triton.autotune(
    configs=runtime.get_triton_config("log_softmax"),
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
        "num_warps": heur_num_warps,
    }
)
@triton.jit
def log_softmax_kernel(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    input_ptrs = input_ptr + offset
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
    row_minus_max = inp - tl.max(inp, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax_output = tl.log(numerator / denominator)
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, softmax_output, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_triton_config("log_softmax"),
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
        "num_warps": heur_num_warps,
    }
)
@triton.jit
def log_softmax_backward_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)

    offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    out_ptrs = out_ptr + offsets
    out = tl.load(out_ptrs, mask=mask).to(tl.float32)
    out_grad_ptrs = out_grad_ptr + offsets
    out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)

    scale = tl.sum(out_grad, 1)
    in_grad = out_grad - tl.exp(out.to(tl.float32)) * scale[:, None]

    in_grad_ptrs = in_grad_ptr + offsets
    tl.store(in_grad_ptrs, in_grad, mask=mask)


class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, dtype):
        logging.debug("GEMS LOG_SOFTMAX")

        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        dim = dim % x.ndim
        M = 1
        N = x.shape[dim]
        for i in range(dim):
            M *= x.shape[i]
        inp = x.contiguous()
        if dtype is None:
            dtype = x.dtype
        out = torch.empty_like(inp, dtype=dtype)
        K = inp.numel() // M // N

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        with torch.cuda.device(inp.device):
            log_softmax_kernel[grid](
                out,
                inp,
                M,
                N,
                K,
            )
        ctx.save_for_backward(out)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS LOG_SOFTMAX VJP")

        dim = ctx.dim
        (out,) = ctx.saved_tensors

        assert dim >= -out.ndim and dim < out.ndim, "Invalid dim"
        dim = dim % out.ndim
        M = 1
        N = out.shape[dim]
        for i in range(dim):
            M *= out.shape[i]

        out_grad = out_grad.contiguous()
        in_grad = torch.empty_like(out)
        K = out.numel() // M // N

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        with torch.cuda.device(in_grad.device):
            log_softmax_backward_kernel[grid](
                out,
                out_grad,
                in_grad,
                M,
                N,
                K,
            )
        return in_grad, None, None


def log_softmax(x, dim=-1, dtype=None):
    return LogSoftmax.apply(x, dim, dtype)
