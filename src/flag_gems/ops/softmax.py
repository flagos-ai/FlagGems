import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"TILE_K": 32}),
        triton.Config({"TILE_K": 64}),
        triton.Config({"TILE_K": 128}),
        triton.Config({"TILE_K": 256}),
        triton.Config({"TILE_K": 512}),
        triton.Config({"TILE_K": 1024}),
    ],
    key=[
        "K",
    ],
)
@triton.heuristics(
    values={
        "TILE_N": lambda args: max(1, 1024 // args["TILE_K"]),
        "ONE_TILE_PER_CTA": lambda args: args["TILE_N"] >= args["N"],
    },
)
@triton.jit
def softmax_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(0)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets
        mask = (n_offsets[:, None] < N) & (k_offsets < K)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        m = tl.max(inp, 0)
        e = tl.exp(inp - m[None, :])
        z = tl.sum(e, 0)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_K], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_K], value=0.0, dtype=tl.float32)

        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets
        for _ in range(0, N, TILE_N):
            mask = (n_offsets[:, None] < N) & (k_offsets < K)
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, tl.max(inp, 0))
            alpha = m - m_new
            z = z * tl.exp(alpha) + tl.sum(tl.exp(inp - m_new[None, :]), axis=0)
            m = m_new
            n_offsets += TILE_N
            offset += TILE_N * K

        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets
        for _ in range(0, N, TILE_N):
            mask = (n_offsets[:, None] < N) & (k_offsets < K)
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
            o = tl.exp(inp - m[None, :]) / z[None, :]
            output_ptrs = output_ptr + offset
            tl.store(output_ptrs, o, mask=mask)
            n_offsets += TILE_N
            offset += TILE_N * K


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"TILE_N": 32}),
        triton.Config({"TILE_N": 64}),
        triton.Config({"TILE_N": 128}),
        triton.Config({"TILE_N": 256}),
        triton.Config({"TILE_N": 512}),
        triton.Config({"TILE_N": 1024}),
    ],
    key=["N"],
)
@triton.heuristics(
    values={
        "TILE_M": lambda args: max(1, 1024 // args["TILE_N"]),
        "ONE_TILE_PER_CTA": lambda args: args["TILE_N"] >= args["N"],
    },
)
@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = m_offsets[:, None] * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = (m_offsets[:, None] < M) & (n_offsets < N)
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        m = tl.max(inp, 1)
        e = tl.exp(inp - m[:, None])
        z = tl.sum(e, 1)
        out = e / z[:, None]
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_M], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_M], value=0.0, dtype=tl.float32)

        n_offsets = tl.arange(0, TILE_N)
        offset = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, tl.max(inp, 1))
            alpha = m - m_new
            z = z * tl.exp(alpha) + tl.sum(tl.exp(inp - m_new[:, None]), axis=1)
            m = m_new
            n_offsets += TILE_N
            offset += TILE_N

        n_offsets = tl.arange(0, TILE_N)
        offset = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
            o = tl.exp(inp - m[:, None]) / z[:, None]
            output_ptrs = output_ptr + offset
            tl.store(output_ptrs, o, mask=mask)
            n_offsets += TILE_N
            offset += TILE_N


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"TILE_K": 32}),
        triton.Config({"TILE_K": 64}),
        triton.Config({"TILE_K": 128}),
        triton.Config({"TILE_K": 256}),
        triton.Config({"TILE_K": 512}),
        triton.Config({"TILE_K": 1024}),
    ],
    key=[
        "K",
    ],
)
@triton.heuristics(
    values={
        "TILE_N": lambda args: max(1, 1024 // args["TILE_K"]),
        "ONE_TILE_PER_CTA": lambda args: args["TILE_N"] >= args["N"],
    },
)
@triton.jit
def softmax_backward_kernel_non_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offsets_k = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        mask = (offsets_n < N)[:, None] & (offsets_k < K)
        out_tile = tl.load(out_ptr + offsets, mask=mask)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
        scale = tl.sum(out_tile * out_grad_tile, axis=0)
        in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        scale = tl.zeros([TILE_N, TILE_K], dtype=tl.float32)
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            scale += out_tile * out_grad_tile
            offsets_n += TILE_N
            offsets += TILE_N * K
        scale = tl.sum(scale, axis=0)  # (TILE_K)

        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            offsets_n += TILE_N
            offsets += TILE_N * K


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"TILE_N": 32}),
        triton.Config({"TILE_N": 64}),
        triton.Config({"TILE_N": 128}),
        triton.Config({"TILE_N": 256}),
        triton.Config({"TILE_N": 512}),
        triton.Config({"TILE_N": 1024}),
    ],
    key=["N"],
)
@triton.heuristics(
    values={
        "TILE_M": lambda args: max(1, 1024 // args["TILE_N"]),
        "ONE_TILE_PER_CTA": lambda args: args["TILE_N"] >= args["N"],
    },
)
@triton.jit
def softmax_backward_kernel_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        mask = (m_offsets[:, None] < M) & (n_offsets < N)
        out_tile = tl.load(out_ptr + offsets, mask=mask)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
        scale = tl.sum(out_tile * out_grad_tile, 1)
        in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        scale = tl.zeros([TILE_M, TILE_N], dtype=tl.float32)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(out_ptr + offsets, mask=mask)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            scale += out_tile * out_grad_tile
            n_offsets += TILE_N
            offsets += TILE_N
        scale = tl.sum(scale, 1)  # (TILE_M,)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(out_ptr + offsets, mask=mask)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            n_offsets += TILE_N
            offsets += TILE_N


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, dtype):
        logging.debug("GEMS SOFTMAX")

        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        dim = dim % x.ndim
        M = 1
        N = x.shape[dim]
        for i in range(dim):
            M *= x.shape[i]  # pre_dim
        inp = x.contiguous()
        if dtype is None:
            dtype = x.dtype
        out = torch.empty_like(inp, dtype=dtype)
        K = inp.numel() // M // N  # post_dim

        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            softmax_kernel_non_inner[grid](
                out,
                inp,
                M,
                N,
                K,
            )
        else:
            grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
            softmax_kernel_inner[grid](
                out,
                inp,
                M,
                N,
            )
        ctx.save_for_backward(out)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS SOFTMAX VJP")
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

        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            softmax_backward_kernel_non_inner[grid](
                out,
                out_grad,
                in_grad,
                M,
                N,
                K,
            )
        else:
            grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
            softmax_backward_kernel_inner[grid](
                out,
                out_grad,
                in_grad,
                M,
                N,
            )
        return in_grad, None, None


def softmax(x, dim=-1, dtype=None):
    return Softmax.apply(x, dim, dtype)
