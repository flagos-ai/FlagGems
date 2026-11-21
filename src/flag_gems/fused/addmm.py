import torch
import triton
import triton.language as tl


@triton.jit
def addmm_kernel(
    bias_ptr,
    a_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    beta,
    alpha,
    stride_a_m,
    stride_a_k,
    stride_b_k,
    stride_b_n,
    stride_out_m,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
    b_ptrs = b_ptr + offs_k[:, None] * stride_b_k + offs_n[None, :] * stride_b_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0
        )
        b = tl.load(
            b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0
        )
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_a_k
        b_ptrs += BLOCK_K * stride_b_k

    acc = acc * alpha
    bias = tl.load(bias_ptr + offs_n[None, :], mask=offs_n[None, :] < N, other=0.0)
    acc = acc + bias * beta

    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def addmm(
    input: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    """
    torch.addmm(input, a, b, beta=beta, alpha=alpha)
    out = beta * input + alpha * (a @ b)
    Restriction: input(bias) only support shape (N,)
    """
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "K dimension mismatch between a and b"
    assert input.shape == (N,), f"input bias must be shape ({N},), got {input.shape}"

    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    addmm_kernel[grid](
        input,
        a,
        b,
        out,
        M,
        N,
        K,
        beta,
        alpha,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out
