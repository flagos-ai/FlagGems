import logging

import torch
import triton
import triton.language as tl

from .. import runtime
from ..runtime import torch_device_fn
from ..utils import libentry
from ..utils import triton_lang_extension as tle


@libentry()
@triton.jit
def log_softmax_kernel(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr = 8,
    BLOCK_N: tl.constexpr = 256,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # TODO(chenfeiyu): consider float64 add add a utility function to get accumulator type
    m = tl.full([BLOCK_M, BLOCK_N], value=float("-inf"), dtype=tl.float32)
    z = tl.full([BLOCK_M, BLOCK_N], value=0.0, dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(inp, m)
        all_neg_inf = m_new == float("-inf")
        z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
        m = m_new

    m_reduced = tl.max(m, 1)
    z = tl.sum(z * tl.exp(m - m_reduced[:, None]), 1)
    m = m_reduced

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        o = inp - m[:, None] - tl.log(z[:, None])
        tl.store(output_ptr + offset, o, mask=mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("log_softmax"), key=["M", "N"])
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

    scale = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)
        scale += out_grad
    scale = tl.sum(scale, 1)

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        out_ptrs = out_ptr + offsets
        out = tl.load(out_ptrs, mask=mask).to(tl.float32)
        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)
        in_grad = out_grad - tl.exp(out) * scale[:, None]
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
        with torch_device_fn.device(inp.device):
            log_softmax_kernel[grid](
                out,
                inp,
                M,
                N,
                K,
                num_warps=8,
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
        with torch_device_fn.device(in_grad.device):
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
