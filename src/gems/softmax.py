import torch
import triton
import triton.language as tl
from .__libentry__ import libentry


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1}, num_stages=4),
        triton.Config({"BLOCK_M": 1}, num_stages=5),
        triton.Config({"BLOCK_M": 2}, num_stages=4),
        triton.Config({"BLOCK_M": 2}, num_stages=5),
        triton.Config({"BLOCK_M": 4}, num_stages=4),
        triton.Config({"BLOCK_M": 4}, num_stages=5),
        triton.Config({"BLOCK_M": 8}, num_stages=4),
        triton.Config({"BLOCK_M": 8}, num_stages=5),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    values={
        "BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
        "num_warps": lambda args: 4
        if args["N"] <= 1024
        else (8 if args["N"] <= 2048 else 16),
    },
)
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    input_ptrs = input_ptr + offset
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    row_minus_max = inp - tl.max(inp, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax_output = numerator / denominator
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, softmax_output, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1}, num_stages=4),
        triton.Config({"BLOCK_M": 1}, num_stages=5),
        triton.Config({"BLOCK_M": 2}, num_stages=4),
        triton.Config({"BLOCK_M": 2}, num_stages=5),
        triton.Config({"BLOCK_M": 4}, num_stages=4),
        triton.Config({"BLOCK_M": 4}, num_stages=5),
        triton.Config({"BLOCK_M": 8}, num_stages=4),
        triton.Config({"BLOCK_M": 8}, num_stages=5),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    values={
        "BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
        "num_warps": lambda args: 4
        if args["N"] <= 1024
        else (8 if args["N"] <= 2048 else 16),
    },
)
@triton.jit
def softmax_backward_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    out_ptrs = out_ptr + offsets
    out = tl.load(out_ptrs, mask=mask)
    out_grad_ptrs = out_grad_ptr + offsets
    out_grad = tl.load(out_grad_ptrs, mask=mask)

    scale = tl.sum(out * out_grad, 1)
    in_grad = out * (out_grad - scale[:, None])

    in_grad_ptrs = in_grad_ptr + offsets
    tl.store(in_grad_ptrs, in_grad, mask=mask)


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, dtype):
        if __debug__:
            print("GEMS SOFTMAX")

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
        softmax_kernel[grid](
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
        if __debug__:
            print("GEMS SOFTMAX VJP")
        dim = ctx.dim
        (out,) = ctx.saved_tensors

        assert dim >= -out.ndim and dim < out.ndim, "Invalid dim"
        dim = dim % out.ndim
        M = 1
        N = out.shape[dim]
        for i in range(dim):
            M *= out.shape[i]

        out = out.contiguous()
        out_grad = out_grad.contiguous()
        in_grad = torch.empty_like(out)
        K = out.numel() // M // N

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        softmax_backward_kernel[grid](
            out,
            out_grad,
            in_grad,
            M,
            N,
            K,
        )
        return in_grad, None, None


def softmax(x, dim=-1, dtype=None):
    return Softmax.apply(x, dim, dtype)
